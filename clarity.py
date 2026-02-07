# clarity.py
# -------------------------------------------------------------
# Clarity-focused blocks for the audio engine (pair with filters.py & warps.py)
#
# HARDENED PATCH:
#   - Prevents SciPy butter() Wn ordering issues (Wn[0] must be less than Wn[1])
#   - Prevents NaN/Inf propagation that can crash inside CFFI audio callback
#   - Uses fs=sr for butter() to avoid manual normalization errors
#   - Returns identity SOS (passthrough) on invalid params instead of raising
#   - Makes sosfilt state re-init safe for channel changes
# -------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from filters import AudioFilter, register_filter, build_filter, _DEF_EPS as _EPS

_F32 = np.float32

# =========================
# Small + robust helpers
# =========================

def _db_to_lin(db: float) -> float:
    return float(10 ** (float(db) / 20.0))

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x[:, None] if x.ndim == 1 else x

def _soft_clip(x: np.ndarray, thresh: float = 1.0) -> np.ndarray:
    thresh = float(thresh)
    if thresh <= 0:
        return np.tanh(x)
    k = 1.0 / max(1e-6, thresh)
    y = np.tanh(k * x) / k
    return np.clip(y, -thresh, thresh)

def _rms(x: NDArray[_F32]) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=_F32) + _EPS))

# =======================
# Robust DSP primitives
# =======================

def _identity_sos() -> np.ndarray:
    # 1 SOS section passthrough
    return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

def _finite_f(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return float(v) if np.isfinite(v) else float(default)

def _finite_i(x: Any, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return int(default)
    return int(v)

def _safe_sr(sr: int) -> float:
    s = _finite_f(sr, 48000.0)
    if s < 8000.0:
        s = 8000.0
    return float(s)

def _nyq_safe(sr_f: float) -> float:
    # Keep slightly below Nyquist to avoid edge instability
    return max(100.0, (0.5 * sr_f) * 0.999)

def _safe_cutoff(sr_f: float, fc: Any, *, lo: float, hi: float) -> float:
    v = _finite_f(fc, (lo + hi) * 0.5)
    return float(np.clip(v, lo, hi))

def _safe_band(sr_f: float, fl: Any, fh: Any, *, min_bw_hz: float = 10.0) -> tuple[float, float]:
    lo = _finite_f(fl, 200.0)
    hi = _finite_f(fh, 2000.0)

    nyq = _nyq_safe(sr_f)

    lo = float(np.clip(lo, 10.0, nyq))
    hi = float(np.clip(hi, 20.0, nyq))

    # enforce ordering
    if hi <= lo:
        hi = min(nyq, lo + 200.0)

    # enforce bandwidth
    if (hi - lo) < float(min_bw_hz):
        hi = min(nyq, lo + float(min_bw_hz))

    # last-resort
    if not (hi > lo):
        lo = 200.0
        hi = min(nyq, 2000.0)
        if hi <= lo:
            hi = min(nyq, lo + 50.0)

    return lo, hi

def _valid_sos(sos: Any) -> bool:
    if sos is None:
        return False
    sos = np.asarray(sos)
    if sos.ndim != 2 or sos.shape[1] != 6:
        return False
    return bool(np.all(np.isfinite(sos)))

def _safe_tf2sos(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    try:
        sos = signal.tf2sos(b, a)
        return sos if _valid_sos(sos) else _identity_sos()
    except Exception:
        return _identity_sos()

def _butter_sos(kind: str, sr: int, **kwargs) -> np.ndarray:
    """
    Safe Butterworth SOS builder using fs=sr.
    kind: "low" | "high" | "band"
    """
    try:
        sr_f = _safe_sr(sr)
        nyq = _nyq_safe(sr_f)
        kind = (kind or "").strip().lower()
        order = max(1, _finite_i(kwargs.get("order", 4), 4))

        if kind == "low":
            fc = _safe_cutoff(sr_f, kwargs.get("fc", 8000.0), lo=20.0, hi=nyq)
            sos = signal.butter(order, fc, btype="lowpass", fs=sr_f, output="sos")
            return sos if _valid_sos(sos) else _identity_sos()

        if kind == "high":
            fc = _safe_cutoff(sr_f, kwargs.get("fc", 80.0), lo=10.0, hi=nyq)
            sos = signal.butter(order, fc, btype="highpass", fs=sr_f, output="sos")
            return sos if _valid_sos(sos) else _identity_sos()

        if kind == "band":
            fl, fh = _safe_band(sr_f, kwargs.get("fl", 200.0), kwargs.get("fh", 2000.0), min_bw_hz=10.0)
            sos = signal.butter(order, [fl, fh], btype="bandpass", fs=sr_f, output="sos")
            return sos if _valid_sos(sos) else _identity_sos()

        return _identity_sos()
    except Exception:
        return _identity_sos()

def _iir_peaking(sr: int, f0: float, q: float, gain_db: float) -> np.ndarray:
    """
    Robust peaking EQ biquad (Audio EQ Cookbook).
    Returns identity SOS on invalid params.
    """
    try:
        sr_f = _safe_sr(sr)
        nyq = _nyq_safe(sr_f)

        f0 = float(np.clip(_finite_f(f0, 3000.0), 10.0, nyq))
        q = float(np.clip(_finite_f(q, 1.0), 0.05, 24.0))
        gain_db = float(np.clip(_finite_f(gain_db, 0.0), -24.0, 24.0))

        A = 10 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * (f0 / sr_f)
        alpha = np.sin(w0) / (2.0 * q)
        cosw = np.cos(w0)

        b0 = 1 + alpha * A
        b1 = -2 * cosw
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cosw
        a2 = 1 - alpha / A

        if not np.isfinite(a0) or abs(a0) < 1e-12:
            return _identity_sos()

        b = (np.array([b0, b1, b2], dtype=np.float64) / a0)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return _safe_tf2sos(b, a)
    except Exception:
        return _identity_sos()

def _iir_shelf(sr: int, f0: float, gain_db: float, shelf_type: str = "high", slope: float = 0.707) -> np.ndarray:
    try:
        sr_f = _safe_sr(sr)
        nyq = _nyq_safe(sr_f)

        f0 = float(np.clip(_finite_f(f0, 10000.0), 10.0, nyq))
        gain_db = float(np.clip(_finite_f(gain_db, 0.0), -24.0, 24.0))

        # Keep slope in a safe cookbook range.
        # (S=1.0 is the common default; >1 can make the sqrt term negative for some gains.)
        slope = float(np.clip(_finite_f(slope, 0.707), 0.10, 1.50))

        A = 10 ** (gain_db / 40.0)
        A = float(A) if np.isfinite(A) and A > 0 else 1.0

        w0 = 2.0 * np.pi * (f0 / sr_f)
        cosw = np.cos(w0)
        sinw = np.sin(w0)

        # --- FIX: clamp radicand to avoid invalid sqrt -> NaN ---
        rad = (A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0
        if not np.isfinite(rad):
            rad = 0.0
        rad = max(0.0, float(rad))
        alpha = (sinw / 2.0) * np.sqrt(rad)
        # --------------------------------------------------------

        st = (shelf_type or "high").strip().lower()
        if st not in ("high", "low"):
            st = "high"

        if st == "high":
            b0 = A * ((A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
            b2 = A * ((A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cosw)
            a2 = (A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha
        else:
            b0 = A * ((A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
            b2 = A * ((A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cosw)
            a2 = (A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha

        if not np.isfinite(a0) or abs(a0) < 1e-12:
            return _identity_sos()

        b = (np.array([b0, b1, b2], dtype=np.float64) / a0)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return _safe_tf2sos(b, a)

    except Exception:
        return _identity_sos()

def _sosfilt_block(sos: np.ndarray, x: np.ndarray, zi: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crash-proof SOS filtering:
      - invalid sos -> passthrough
      - state shape mismatch -> re-init
      - NaN/Inf output -> zero it
    """
    x = _ensure_2d(np.asarray(x, dtype=np.float32))

    if not _valid_sos(sos):
        return x, zi

    try:
        sos = np.asarray(sos, dtype=np.float64)

        if zi is None:
            zi0 = signal.sosfilt_zi(sos).astype(np.float32)
            zi = np.tile(zi0[:, None, :], (1, x.shape[1], 1)).astype(np.float32)
        else:
            if zi.ndim != 3 or zi.shape[1] != x.shape[1] or zi.shape[0] != sos.shape[0]:
                zi0 = signal.sosfilt_zi(sos).astype(np.float32)
                zi = np.tile(zi0[:, None, :], (1, x.shape[1], 1)).astype(np.float32)

        y = np.empty_like(x)
        for c in range(x.shape[1]):
            y[:, c], zi[:, c, :] = signal.sosfilt(sos, x[:, c], zi=zi[:, c, :])

        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return y.astype(np.float32), zi
    except Exception:
        return x, zi


# =========================
# Filters
# =========================

@register_filter(
    "deesser",
    help="Split-band de-esser: Params freq(6000), q(1.0), threshold_db(-20), ratio(6), atk_ms(3), rel_ms(60), makeup_db(0)",
)
class DeEsser(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.f0 = _finite_f(params.get("freq", 6000.0), 6000.0)
        self.q = _finite_f(params.get("q", 1.0), 1.0)
        self.th = _finite_f(params.get("threshold_db", -20.0), -20.0)
        self.ratio = _finite_f(params.get("ratio", 6.0), 6.0)
        self.atk = _finite_f(params.get("atk_ms", params.get("attack_ms", 3.0)), 3.0)
        self.rel = _finite_f(params.get("rel_ms", params.get("release_ms", 60.0)), 60.0)
        self.mk = _finite_f(params.get("makeup_db", 0.0), 0.0)

        self._sos_band: Optional[np.ndarray] = None
        self._zi_band: Optional[np.ndarray] = None
        self._env: Optional[np.ndarray] = None
        self._last_sr: Optional[int] = None
        self._last_C: Optional[int] = None

    def _ensure(self, sr: int, C: int):
        sr_i = int(sr)
        if self._env is None or self._env.shape[0] != C:
            self._env = np.ones((C,), dtype=np.float32)  # start at unity gain

        # rebuild SOS if SR changed (or not built yet)
        if self._sos_band is None or self._last_sr != sr_i:
            sr_f = _safe_sr(sr_i)
            nyq = _nyq_safe(sr_f)

            f0 = float(np.clip(self.f0, 50.0, nyq))
            q = float(np.clip(self.q, 0.1, 24.0))

            bw = max(200.0, f0 / max(1.0, q))
            fl = f0 - bw / 2.0
            fh = f0 + bw / 2.0
            fl, fh = _safe_band(sr_f, fl, fh, min_bw_hz=50.0)

            self._sos_band = _butter_sos("band", sr_i, fl=fl, fh=fh, order=4)
            self._zi_band = None
            self._last_sr = sr_i

        # re-init zi if channels changed
        if self._zi_band is not None and self._zi_band.shape[1] != C:
            self._zi_band = None

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(np.asarray(block, dtype=np.float32))
        C = x.shape[1]
        self._ensure(sr, C)

        band, self._zi_band = _sosfilt_block(self._sos_band, x, self._zi_band)

        level = np.maximum(_EPS, np.max(np.abs(band), axis=0))
        level_db = 20.0 * np.log10(level)
        over = np.maximum(0.0, level_db - float(self.th))

        ratio = max(1.0, float(self.ratio))
        gain_db = -(1.0 - 1.0 / ratio) * over

        atk_s = max(1, int(max(0.1, float(self.atk)) * 1e-3 * float(sr)))
        rel_s = max(1, int(max(0.1, float(self.rel)) * 1e-3 * float(sr)))
        atk = float(np.exp(-1.0 / atk_s))
        rel = float(np.exp(-1.0 / rel_s))

        g_lin = np.empty((C,), dtype=np.float32)
        for c in range(C):
            env = float(self._env[c])
            target = _db_to_lin(float(gain_db[c]))
            # smooth towards target (more aggressive when reducing)
            if target < env:
                env = atk * env + (1.0 - atk) * target
            else:
                env = rel * env + (1.0 - rel) * target
            env = float(np.clip(env, 0.0, 1.0))
            self._env[c] = env
            g_lin[c] = env

        y_band = band * g_lin[None, :]
        y = x + (y_band - band)

        if float(self.mk) != 0.0:
            y *= _db_to_lin(float(self.mk))

        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return np.clip(y, -1.0, 1.0).astype(np.float32)


@register_filter("presence", help="Presence peak EQ. Params: freq(3000), q(1.0), gain_db(2.0)")
class PresenceEQ(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.freq = _finite_f(params.get("freq", 3000.0), 3000.0)
        self.q = _finite_f(params.get("q", 1.0), 1.0)
        self.g = _finite_f(params.get("gain_db", 2.0), 2.0)
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._last_sr: Optional[int] = None
        self._last_C: Optional[int] = None

    def _ensure(self, sr: int, C: int):
        sr_i = int(sr)
        if self._sos is None or self._last_sr != sr_i:
            self._sos = _iir_peaking(sr_i, self.freq, max(0.1, self.q), self.g)
            self._zi = None
            self._last_sr = sr_i

        if self._zi is None or self._last_C != C:
            zi = signal.sosfilt_zi(self._sos).astype(np.float32)
            self._zi = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(np.asarray(block, dtype=np.float32))
        self._ensure(sr, x.shape[1])
        y, self._zi = _sosfilt_block(self._sos, x, self._zi)
        return np.clip(y, -1.0, 1.0).astype(np.float32)


@register_filter("air", help="High-shelf 'air'. Params: freq(10000), gain_db(2.0), slope(0.8)")
class AirShelf(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.freq = _finite_f(params.get("freq", 10000.0), 10000.0)
        self.g = _finite_f(params.get("gain_db", 2.0), 2.0)
        self.slope = _finite_f(params.get("slope", 0.8), 0.8)
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._last_sr: Optional[int] = None
        self._last_C: Optional[int] = None

    def _ensure(self, sr: int, C: int):
        sr_i = int(sr)
        if self._sos is None or self._last_sr != sr_i:
            self._sos = _iir_shelf(sr_i, self.freq, self.g, "high", self.slope)
            self._zi = None
            self._last_sr = sr_i

        if self._zi is None or self._last_C != C:
            zi = signal.sosfilt_zi(self._sos).astype(np.float32)
            self._zi = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(np.asarray(block, dtype=np.float32))
        self._ensure(sr, x.shape[1])
        y, self._zi = _sosfilt_block(self._sos, x, self._zi)
        return np.clip(y, -1.0, 1.0).astype(np.float32)


@register_filter("tilt", help="Tilt EQ around pivot. Params: pivot(1000), tilt_db(+/-3)")
class TiltEQ(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.pivot = _finite_f(params.get("pivot", 1000.0), 1000.0)
        self.tilt = _finite_f(params.get("tilt_db", 0.0), 0.0)
        self._hi: Optional[np.ndarray] = None
        self._lo: Optional[np.ndarray] = None
        self._zi_hi: Optional[np.ndarray] = None
        self._zi_lo: Optional[np.ndarray] = None
        self._last_sr: Optional[int] = None
        self._last_C: Optional[int] = None

    def _ensure(self, sr: int, C: int):
        sr_i = int(sr)
        if (self._hi is None) or (self._lo is None) or (self._last_sr != sr_i):
            self._hi = _iir_shelf(sr_i, self.pivot, +self.tilt / 2.0, "high", 0.8)
            self._lo = _iir_shelf(sr_i, self.pivot, -self.tilt / 2.0, "low", 0.8)
            self._zi_hi = None
            self._zi_lo = None
            self._last_sr = sr_i

        if self._zi_hi is None or self._zi_lo is None or self._last_C != C:
            zi = signal.sosfilt_zi(self._hi).astype(np.float32)
            self._zi_hi = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)
            zi = signal.sosfilt_zi(self._lo).astype(np.float32)
            self._zi_lo = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(np.asarray(block, dtype=np.float32))
        self._ensure(sr, x.shape[1])
        y, self._zi_hi = _sosfilt_block(self._hi, x, self._zi_hi)
        y, self._zi_lo = _sosfilt_block(self._lo, y, self._zi_lo)
        return np.clip(y, -1.0, 1.0).astype(np.float32)


@register_filter(
    "transient",
    help="Transient shaper. Params: attack(1.2), sustain(1.0), atk_ms(1.5), rel_ms(80), mix(1.0)",
)
class TransientShaper(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.attack = _finite_f(params.get("attack", 1.2), 1.2)
        self.sustain = _finite_f(params.get("sustain", 1.0), 1.0)
        self.atk_ms = _finite_f(params.get("atk_ms", 1.5), 1.5)
        self.rel_ms = _finite_f(params.get("rel_ms", 80.0), 80.0)
        self.mix = _finite_f(params.get("mix", 1.0), 1.0)
        self._env_f: Optional[np.ndarray] = None
        self._env_s: Optional[np.ndarray] = None

    def _ensure(self, C: int):
        if self._env_f is None or self._env_f.shape[0] != C:
            self._env_f = np.zeros((C,), dtype=np.float32)
        if self._env_s is None or self._env_s.shape[0] != C:
            self._env_s = np.zeros((C,), dtype=np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(np.asarray(block, dtype=np.float32))
        C = x.shape[1]
        self._ensure(C)

        atk_s = max(1, int(max(0.1, float(self.atk_ms)) * 1e-3 * float(sr)))
        rel_s = max(1, int(max(0.1, float(self.rel_ms)) * 1e-3 * float(sr)))
        atk = float(np.exp(-1.0 / atk_s))
        rel = float(np.exp(-1.0 / rel_s))

        y = np.empty_like(x)
        for n in range(x.shape[0]):
            s = np.abs(x[n, :])
            # fast & slow envelopes
            self._env_f = np.maximum(self._env_f * atk + (1.0 - atk) * s, s)
            self._env_s = self._env_s * rel + (1.0 - rel) * s

            ratio = (self._env_f + 1e-6) / (self._env_s + 1e-6)
            g = np.clip((ratio ** (self.attack - 1.0)), 0.25, 4.0)
            y[n, :] = x[n, :] * g

        mix = float(np.clip(self.mix, 0.0, 1.0))
        out = (1.0 - mix) * x + mix * y

        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return np.clip(out, -1.0, 1.0).astype(np.float32)


@register_filter(
    "imagerfocus",
    help="Mid/Side width + mono-below. Params: width(1.0=keep, 0..2), mono_below(0=no mono)",
)
class ImagerFocus(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.width = _finite_f(params.get("width", 1.0), 1.0)
        self.mono_below = _finite_f(params.get("mono_below", 0.0), 0.0)
        self._hp_sos: Optional[np.ndarray] = None
        self._hp_zi: Optional[np.ndarray] = None
        self._last_sr: Optional[int] = None

    def _ensure(self, sr: int, C: int):
        if C < 2:
            self._hp_sos = None
            self._hp_zi = None
            return

        sr_i = int(sr)

        if self.mono_below and (self._hp_sos is None or self._last_sr != sr_i):
            self._hp_sos = _butter_sos("high", sr_i, fc=self.mono_below, order=2)
            self._hp_zi = None
            self._last_sr = sr_i

        if self._hp_sos is not None and (self._hp_zi is None or self._hp_zi.shape[1] != 1):
            # side channel is 1D (shape Nx1)
            zi = signal.sosfilt_zi(self._hp_sos).astype(np.float32)
            self._hp_zi = np.tile(zi[:, None, :], (1, 1, 1)).astype(np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(np.asarray(block, dtype=np.float32))
        if x.shape[1] < 2:
            return x.astype(np.float32)

        self._ensure(sr, x.shape[1])

        L = x[:, 0:1]
        R = x[:, 1:2]
        M = 0.5 * (L + R)
        S = 0.5 * (L - R)

        if self.mono_below and self._hp_sos is not None:
            # highpass the side channel -> lows become mono
            S, self._hp_zi = _sosfilt_block(self._hp_sos, S, self._hp_zi)

        width = float(np.clip(self.width, 0.0, 2.0))
        S *= width

        y = np.concatenate([M + S, M - S], axis=1).astype(np.float32)

        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return np.clip(y, -1.0, 1.0).astype(np.float32)


@register_filter(
    "demud",
    help="Dynamic low-mid cleaner. Params: fl(180) fh(450) threshold_db(-28) ratio(2.0) atk_ms(8) rel_ms(120)",
)
class DeMud(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.fl = _finite_f(params.get("fl", 180.0), 180.0)
        self.fh = _finite_f(params.get("fh", 450.0), 450.0)
        self.th = _finite_f(params.get("threshold_db", -28.0), -28.0)
        self.ratio = _finite_f(params.get("ratio", 2.0), 2.0)
        self.atk = _finite_f(params.get("atk_ms", 8.0), 8.0)
        self.rel = _finite_f(params.get("rel_ms", 120.0), 120.0)

        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._env: Optional[np.ndarray] = None
        self._last_sr: Optional[int] = None
        self._last_C: Optional[int] = None

    def _ensure(self, sr: int, C: int):
        sr_i = int(sr)

        if self._env is None or self._env.shape[0] != C:
            self._env = np.ones((C,), dtype=np.float32)

        if self._sos is None or self._last_sr != sr_i:
            sr_f = _safe_sr(sr_i)
            fl, fh = _safe_band(sr_f, self.fl, self.fh, min_bw_hz=20.0)
            self._sos = _butter_sos("band", sr_i, fl=fl, fh=fh, order=4)
            self._zi = None
            self._last_sr = sr_i

        if self._zi is None or self._last_C != C:
            zi = signal.sosfilt_zi(self._sos).astype(np.float32)
            self._zi = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(np.asarray(block, dtype=np.float32))
        C = x.shape[1]
        self._ensure(sr, C)

        band, self._zi = _sosfilt_block(self._sos, x, self._zi)

        lvl = np.maximum(_EPS, np.mean(np.abs(band), axis=0))
        lvl_db = 20.0 * np.log10(lvl)
        over = np.maximum(0.0, lvl_db - float(self.th))

        ratio = max(1.0, float(self.ratio))
        g_db = -(1.0 - 1.0 / ratio) * over

        atk_s = max(1, int(max(0.1, float(self.atk)) * 1e-3 * float(sr)))
        rel_s = max(1, int(max(0.1, float(self.rel)) * 1e-3 * float(sr)))
        atk = float(np.exp(-1.0 / atk_s))
        rel = float(np.exp(-1.0 / rel_s))

        g_lin = np.empty((C,), dtype=np.float32)
        for c in range(C):
            env = float(self._env[c])
            target = _db_to_lin(float(g_db[c]))
            if target < env:
                env = atk * env + (1.0 - atk) * target
            else:
                env = rel * env + (1.0 - rel) * target
            env = float(np.clip(env, 0.0, 1.0))
            self._env[c] = env
            g_lin[c] = env

        y = x + (band * g_lin[None, :] - band)

        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return np.clip(y, -1.0, 1.0).astype(np.float32)


# =========================
# Curated chain
# =========================

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
    Convenience filter: composes existing filters via registry so you can tweak
    per-stage using namespaced params.
    """
    def __init__(self, params: Dict[str, Any]):
        self.low_cut = _finite_f(params.get("low_cut", 30.0), 30.0)

        d = {k.split(".", 1)[1]: v for k, v in params.items() if k.startswith("deesser.")}
        d.setdefault("freq", 6000.0)
        d.setdefault("threshold_db", -20.0)
        d.setdefault("ratio", 6.0)
        d.setdefault("q", 1.0)

        pz = {k.split(".", 1)[1]: v for k, v in params.items() if k.startswith("presence.")}
        pz.setdefault("freq", 3000.0)
        pz.setdefault("q", 1.0)
        pz.setdefault("gain_db", 1.5)

        air = {k.split(".", 1)[1]: v for k, v in params.items() if k.startswith("air.")}
        air.setdefault("freq", 10000.0)
        air.setdefault("gain_db", 1.5)

        self.ceiling = _finite_f(params.get("ceiling_db", -1.0), -1.0)

        self._built = False
        self._chain: List[AudioFilter] = []
        self._cache = dict(d=d, pz=pz, air=air)

    def _build(self, sr: int, C: int):
        if self._built:
            return
        hp = build_filter("highpass", cutoff=float(self.low_cut), order=2)
        ds = build_filter("deesser", **self._cache["d"])
        pr = build_filter("presence", **self._cache["pz"])
        ar = build_filter("air", **self._cache["air"])
        lim = build_filter("limiter", ceiling_db=float(self.ceiling), release_ms=50)
        self._chain = [hp, ds, pr, ar, lim]
        self._built = True

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(np.asarray(block, dtype=np.float32))
        self._build(sr, x.shape[1])
        y = x
        for f in self._chain:
            # IMPORTANT: keep it crash-proof even if downstream filter has a bug
            try:
                y = f.process(y, sr)
                if not np.all(np.isfinite(y)):
                    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            except Exception:
                # fail-safe: passthrough on error
                y = y
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def flush(self) -> Optional[np.ndarray]:
        if not self._chain:
            return None
        tails = []
        for f in self._chain:
            try:
                t = f.flush()
            except Exception:
                t = None
            if t is not None and getattr(t, "size", 0):
                tt = np.asarray(t, dtype=np.float32)
                if not np.all(np.isfinite(tt)):
                    tt = np.nan_to_num(tt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                tails.append(tt)

        if not tails:
            return None

        T = max(t.shape[0] for t in tails)
        C = tails[0].shape[1] if tails[0].ndim == 2 else 1
        acc = np.zeros((T, C), dtype=np.float32)

        for t in tails:
            t = _ensure_2d(t)
            if t.shape[0] < T:
                pad = np.zeros((T - t.shape[0], t.shape[1]), dtype=np.float32)
                t = np.concatenate([t, pad], axis=0)
            acc += t.astype(np.float32)

        return np.clip(acc, -1.0, 1.0).astype(np.float32)
