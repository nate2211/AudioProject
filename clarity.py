# clarity.py
# -------------------------------------------------------------
# Clarity-focused blocks for the audio engine (pair with filters.py & warps.py)
#
# Usage examples:
#   python main.py run --url input.wav --pipeline "deesser|presence|air|normalize" --out out.wav \
#     --extra deesser.freq=6200 deesser.threshold_db=-24 deesser.ratio=6 \
#             presence.freq=3200 presence.gain_db=2.5 presence.q=1.0 \
#             air.freq=10000 air.gain_db=2.0
#
#   python main.py run --url input.wav --pipeline "imagerfocus|transient|claritychain" --out out.wav \
#     --extra imagerfocus.width=0.9 imagerfocus.mono_below=120 \
#             transient.attack=1.3 transient.sustain=0.9 \
#             claritychain.low_cut=30 claritychain.deesser.threshold_db=-22 claritychain.presence.gain_db=1.5
# -------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from filters import AudioFilter, register_filter, build_filter, _DEF_EPS as _EPS

_F32 = np.float32

# ----------------- small helpers -----------------

def _db_to_lin(db: float) -> float:
    return float(10 ** (db / 20.0))

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    return x[:, None] if x.ndim == 1 else x

def _soft_clip(x: np.ndarray, thresh: float = 1.0) -> np.ndarray:
    if thresh <= 0:
        return np.tanh(x)
    k = 1.0 / max(1e-6, thresh)
    y = np.tanh(k * x) / k
    return np.clip(y, -thresh, thresh)

def _rms(x: NDArray[_F32]) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=_F32) + _EPS))

def _butter_sos(kind: str, sr: int, **kwargs) -> np.ndarray:
    # kind in {"low", "high", "band", "highshelf", "lowshelf", "peak"} — peak/shelves via iirpeak/cheby1 fallback
    if kind == "low":
        wc = kwargs["fc"] / (0.5 * sr)
        return signal.butter(int(kwargs.get("order", 4)), wc, btype="low", output="sos")
    if kind == "high":
        wc = kwargs["fc"] / (0.5 * sr)
        return signal.butter(int(kwargs.get("order", 4)), wc, btype="high", output="sos")
    if kind == "band":
        low = kwargs["fl"] / (0.5 * sr)
        high = kwargs["fh"] / (0.5 * sr)
        return signal.butter(int(kwargs.get("order", 4)), [low, high], btype="band", output="sos")
    raise ValueError(f"unsupported kind: {kind}")

def _iir_peaking(sr: int, f0: float, q: float, gain_db: float) -> np.ndarray:
    # biquad peaking filter (Audio EQ Cookbook)
    A  = 10 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * (f0 / sr)
    alpha = np.sin(w0) / (2.0 * max(1e-6, q))
    cosw = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cosw
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.tf2sos(b, a)

def _iir_shelf(sr: int, f0: float, gain_db: float, shelf_type: str = "high", slope: float = 0.707) -> np.ndarray:
    # high/low shelf biquad (Audio EQ Cookbook)
    A  = 10 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * (f0 / sr)
    cosw = np.cos(w0)
    sinw = np.sin(w0)
    alpha = sinw / 2.0 * np.sqrt( (A + 1/A) * (1/slope - 1) + 2 )

    if shelf_type == "high":
        b0 =    A*( (A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha )
        b1 = -2*A*( (A-1) + (A+1)*cosw )
        b2 =    A*( (A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha )
        a0 =        (A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha
        a1 =  2*   ( (A-1) - (A+1)*cosw )
        a2 =        (A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha
    else:
        # low
        b0 =    A*( (A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha )
        b1 =  2*A*( (A-1) - (A+1)*cosw )
        b2 =    A*( (A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha )
        a0 =        (A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha
        a1 = -2*   ( (A-1) + (A+1)*cosw )
        a2 =        (A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.tf2sos(b, a)

def _sosfilt_block(sos: np.ndarray, x: np.ndarray, zi: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if zi is None:
        zi = signal.sosfilt_zi(sos)
        zi = np.tile(zi[:, None, :], (1, x.shape[1], 1)).astype(np.float32)
    y = np.empty_like(x)
    for c in range(x.shape[1]):
        y[:, c], zi[:, c, :] = signal.sosfilt(sos, x[:, c], zi=zi[:, c, :])
    return y.astype(np.float32), zi

# ----------------- filters -----------------

@register_filter(
    "deesser",
    help="Split-band de-esser: Params freq(6000), q(1.0), threshold_db(-20), ratio(6), atk_ms(3), rel_ms(60), makeup_db(0)"
)
class DeEsser(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.f0 = float(params.get("freq", 6000.0))
        self.q = float(params.get("q", 1.0))
        self.th = float(params.get("threshold_db", -20.0))
        self.ratio = float(params.get("ratio", 6.0))
        self.atk = float(params.get("atk_ms", params.get("attack_ms", 3.0)))
        self.rel = float(params.get("rel_ms", params.get("release_ms", 60.0)))
        self.mk = float(params.get("makeup_db", 0.0))

        self._sos_band: Optional[np.ndarray] = None
        self._zi_band: Optional[np.ndarray] = None
        self._env: Optional[np.ndarray] = None

    def _ensure(self, sr: int, C: int):
        if self._sos_band is None or (self._zi_band is not None and self._zi_band.shape[1] != C):
            bw = max(200.0, self.f0 / max(1.0, self.q))  # rough bandwidth
            fl = max(10.0, self.f0 - bw/2)
            fh = min(0.49*sr, self.f0 + bw/2)
            self._sos_band = _butter_sos("band", sr, fl=fl, fh=fh, order=4)
            self._zi_band = None
        if self._env is None or self._env.shape[0] != C:
            self._env = np.zeros((C,), dtype=np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(np.float32))
        C = x.shape[1]
        self._ensure(sr, C)

        band, self._zi_band = _sosfilt_block(self._sos_band, x, self._zi_band)
        level = np.maximum(_EPS, np.max(np.abs(band), axis=0))  # peak per channel
        level_db = 20.0 * np.log10(level)
        over = np.maximum(0.0, level_db - self.th)
        gain_db = -(1.0 - 1.0 / max(1.0, self.ratio)) * over
        # smooth detector
        atk = np.exp(-1.0 / max(1, int(self.atk * 1e-3 * sr)))
        rel = np.exp(-1.0 / max(1, int(self.rel * 1e-3 * sr)))
        g_lin = np.empty((C,), dtype=np.float32)
        for c in range(C):
            env = self._env[c]
            target = _db_to_lin(gain_db[c])
            env = atk * env + (1 - atk) * target if target < env else rel * env + (1 - rel) * target
            self._env[c] = env
            g_lin[c] = env

        # apply in band only (split & recombine)
        y = x.copy()
        y_band = band * g_lin[None, :]
        # subtract original band and add compressed band
        y += (y_band - band)
        if self.mk != 0.0:
            y *= _db_to_lin(self.mk)
        return np.clip(y, -1.0, 1.0)


@register_filter("presence", help="Presence peak EQ. Params: freq(3000), q(1.0), gain_db(2.0)")
class PresenceEQ(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.freq = float(params.get("freq", 3000.0))
        self.q = float(params.get("q", 1.0))
        self.g = float(params.get("gain_db", 2.0))
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None

    def _ensure(self, sr: int, C: int):
        if self._sos is None:
            self._sos = _iir_peaking(sr, self.freq, max(0.1, self.q), self.g)
            self._zi = None
        if self._zi is None or self._zi.shape[1] != C:
            zi = signal.sosfilt_zi(self._sos)
            self._zi = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(np.float32))
        self._ensure(sr, x.shape[1])
        y, self._zi = _sosfilt_block(self._sos, x, self._zi)
        return np.clip(y, -1.0, 1.0)


@register_filter("air", help="High-shelf 'air'. Params: freq(10000), gain_db(2.0), slope(0.8)")
class AirShelf(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.freq = float(params.get("freq", 10000.0))
        self.g = float(params.get("gain_db", 2.0))
        self.slope = float(params.get("slope", 0.8))
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None

    def _ensure(self, sr: int, C: int):
        if self._sos is None:
            self._sos = _iir_shelf(sr, self.freq, self.g, "high", self.slope)
            self._zi = None
        if self._zi is None or self._zi.shape[1] != C:
            zi = signal.sosfilt_zi(self._sos)
            self._zi = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(np.float32))
        self._ensure(sr, x.shape[1])
        y, self._zi = _sosfilt_block(self._sos, x, self._zi)
        return np.clip(y, -1.0, 1.0)


@register_filter("tilt", help="Tilt EQ around pivot. Params: pivot(1000), tilt_db(+/-3)")
class TiltEQ(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.pivot = float(params.get("pivot", 1000.0))
        self.tilt = float(params.get("tilt_db", 0.0))  # + bright / - warm
        self._hi: Optional[np.ndarray] = None
        self._lo: Optional[np.ndarray] = None
        self._zi_hi: Optional[np.ndarray] = None
        self._zi_lo: Optional[np.ndarray] = None

    def _ensure(self, sr: int, C: int):
        if self._hi is None or self._lo is None:
            self._hi = _iir_shelf(sr, self.pivot, +self.tilt/2.0, "high", 0.8)
            self._lo = _iir_shelf(sr, self.pivot, -self.tilt/2.0, "low", 0.8)
            self._zi_hi = self._zi_lo = None
        if self._zi_hi is None or self._zi_hi.shape[1] != C:
            zi = signal.sosfilt_zi(self._hi)
            self._zi_hi = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)
            zi = signal.sosfilt_zi(self._lo)
            self._zi_lo = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(np.float32))
        self._ensure(sr, x.shape[1])
        y, self._zi_hi = _sosfilt_block(self._hi, x, self._zi_hi)
        y, self._zi_lo = _sosfilt_block(self._lo, y, self._zi_lo)
        return np.clip(y, -1.0, 1.0)


@register_filter(
    "transient",
    help="Transient shaper. Params: attack(1.2), sustain(1.0), atk_ms(1.5), rel_ms(80), mix(1.0)"
)
class TransientShaper(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.attack = float(params.get("attack", 1.2))
        self.sustain = float(params.get("sustain", 1.0))
        self.atk_ms = float(params.get("atk_ms", 1.5))
        self.rel_ms = float(params.get("rel_ms", 80.0))
        self.mix = float(params.get("mix", 1.0))
        self._env_f: Optional[np.ndarray] = None
        self._env_s: Optional[np.ndarray] = None

    def _ensure(self, C: int):
        if self._env_f is None or self._env_f.shape[0] != C:
            self._env_f = np.zeros((C,), dtype=np.float32)
            self._env_s = np.zeros((C,), dtype=np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(np.float32))
        C = x.shape[1]
        self._ensure(C)

        atk = np.exp(-1.0 / max(1, int(self.atk_ms * 1e-3 * sr)))
        rel = np.exp(-1.0 / max(1, int(self.rel_ms * 1e-3 * sr)))

        y = np.empty_like(x)
        for n in range(x.shape[0]):
            s = np.abs(x[n, :])
            self._env_f = np.maximum(self._env_f * atk + (1 - atk) * s, s)  # fast
            self._env_s = self._env_s * rel + (1 - rel) * s                # slow
            ratio = (self._env_f + 1e-6) / (self._env_s + 1e-6)
            g = np.clip((ratio ** (self.attack - 1.0)) * (self.sustain ** 0.0), 0.25, 4.0)
            y[n, :] = x[n, :] * g
        out = (1 - self.mix) * x + self.mix * y
        return np.clip(out, -1.0, 1.0)


@register_filter(
    "imagerfocus",
    help="Mid/Side width + mono-below. Params: width(1.0=keep, 0..2), mono_below(0=no mono)"
)
class ImagerFocus(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.width = float(params.get("width", 1.0))
        self.mono_below = float(params.get("mono_below", 0.0))
        self._hp_sos: Optional[np.ndarray] = None
        self._hp_zi: Optional[np.ndarray] = None

    def _ensure(self, sr: int, C: int):
        if C < 2:
            self._hp_sos = None
            self._hp_zi = None
            return
        if self.mono_below and self._hp_sos is None:
            self._hp_sos = _butter_sos("high", sr, fc=self.mono_below, order=2)
            zi = signal.sosfilt_zi(self._hp_sos)
            self._hp_zi = np.tile(zi[:, None, :], (1, 2, 1)).astype(np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(np.float32))
        if x.shape[1] < 2:
            return x  # mono passthrough

        self._ensure(sr, x.shape[1])
        L, R = x[:, 0:1], x[:, 1:2]
        M = 0.5 * (L + R)
        S = 0.5 * (L - R)

        if self.mono_below and self._hp_sos is not None:
            # keep lows mono by highpassing side channel
            S_hp, self._hp_zi = _sosfilt_block(self._hp_sos, S, self._hp_zi)
            S = S_hp

        S *= float(self.width)
        L2 = M + S
        R2 = M - S
        y = np.concatenate([L2, R2], axis=1).astype(np.float32)
        return np.clip(y, -1.0, 1.0)


@register_filter(
    "demud",
    help="Dynamic low-mid cleaner. Params: fl(180) fh(450) threshold_db(-28) ratio(2.0) atk_ms(8) rel_ms(120)"
)
class DeMud(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.fl = float(params.get("fl", 180.0))
        self.fh = float(params.get("fh", 450.0))
        self.th = float(params.get("threshold_db", -28.0))
        self.ratio = float(params.get("ratio", 2.0))
        self.atk = float(params.get("atk_ms", 8.0))
        self.rel = float(params.get("rel_ms", 120.0))
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._env: Optional[np.ndarray] = None

    def _ensure(self, sr: int, C: int):
        if self._sos is None:
            self._sos = _butter_sos("band", sr, fl=self.fl, fh=self.fh, order=4)
            self._zi = None
        if self._zi is None or self._zi.shape[1] != C:
            zi = signal.sosfilt_zi(self._sos)
            self._zi = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)
        if self._env is None or self._env.shape[0] != C:
            self._env = np.zeros((C,), dtype=np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(np.float32))
        C = x.shape[1]
        self._ensure(sr, C)
        band, self._zi = _sosfilt_block(self._sos, x, self._zi)

        # level detector on band energy
        lvl = np.maximum(_EPS, np.mean(np.abs(band), axis=0))
        lvl_db = 20.0 * np.log10(lvl)
        over = np.maximum(0.0, lvl_db - self.th)
        g_db = -(1.0 - 1.0 / max(1.0, self.ratio)) * over
        atk = np.exp(-1.0 / max(1, int(self.atk * 1e-3 * sr)))
        rel = np.exp(-1.0 / max(1, int(self.rel * 1e-3 * sr)))

        g_lin = np.empty((C,), dtype=np.float32)
        for c in range(C):
            env = self._env[c]
            target = _db_to_lin(g_db[c])
            env = atk * env + (1 - atk) * target if target < env else rel * env + (1 - rel) * target
            self._env[c] = env
            g_lin[c] = env

        y = x + (band * g_lin[None, :] - band)  # dynamic dip in that band
        return np.clip(y, -1.0, 1.0)


# ----------------- Curated chain -----------------

@register_filter(
    "claritychain",
    help=(
        "Curated clarity chain: Highpass→DeEsser→Presence→Air→Limiter. "
        "Params: low_cut(30), deesser.{freq,threshold_db,ratio,q}, presence.{freq,q,gain_db}, "
        "air.{freq,gain_db}, ceiling_db(-1)"
    ),
)
class ClarityChain(AudioFilter):
    """
    This is a convenience filter that internally instantiates a short chain
    through your registry so you can tweak per-stage via namespaced params.
    """
    def __init__(self, params: Dict[str, Any]):
        # Extract sub-params with sensible defaults
        self.low_cut = float(params.get("low_cut", 30.0))
        # deesser
        d = {k.split(".",1)[1]: v for k, v in params.items() if k.startswith("deesser.")}
        d.setdefault("freq", 6000.0)
        d.setdefault("threshold_db", -20.0)
        d.setdefault("ratio", 6.0)
        d.setdefault("q", 1.0)
        # presence
        pz = {k.split(".",1)[1]: v for k, v in params.items() if k.startswith("presence.")}
        pz.setdefault("freq", 3000.0)
        pz.setdefault("q", 1.0)
        pz.setdefault("gain_db", 1.5)
        # air
        air = {k.split(".",1)[1]: v for k, v in params.items() if k.startswith("air.")}
        air.setdefault("freq", 10000.0)
        air.setdefault("gain_db", 1.5)
        # limiter
        self.ceiling = float(params.get("ceiling_db", -1.0))

        # lazy-built once SR/channels known
        self._built = False
        self._chain: List[AudioFilter] = []
        self._cache = dict(d=d, pz=pz, air=air)

    def _build(self, sr: int, C: int):
        if self._built:
            return
        # Compose via existing registry
        hp = build_filter("highpass", cutoff=self.low_cut, order=2)
        ds = build_filter("deesser", **self._cache["d"])
        pr = build_filter("presence", **self._cache["pz"])
        ar = build_filter("air", **self._cache["air"])
        lim = build_filter("limiter", ceiling_db=self.ceiling, release_ms=50)
        self._chain = [hp, ds, pr, ar, lim]
        self._built = True

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(np.float32))
        self._build(sr, x.shape[1])
        y = x
        for f in self._chain:
            y = f.process(y, sr)
        return np.clip(y, -1.0, 1.0)

    def flush(self) -> Optional[np.ndarray]:
        if not self._chain:
            return None
        tails = []
        for f in self._chain:
            t = f.flush()
            if t is not None and t.size:
                tails.append(t.astype(np.float32))
        if not tails:
            return None
        T = max(t.shape[0] for t in tails)
        acc = np.zeros((T, tails[0].shape[1]), dtype=np.float32)
        for t in tails:
            if t.shape[0] < T:
                pad = np.zeros((T - t.shape[0], t.shape[1]), dtype=np.float32)
                t = np.concatenate([t, pad], axis=0)
            acc += t
        return np.clip(acc, -1.0, 1.0)