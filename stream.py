# stream.py
# --------------------------------------------
# Real-time friendly audio filters for streaming:
# - dcblock          : One-pole DC blocker (per channel)
# - gate             : Downward expander / noise gate with hysteresis
# - autogain         : Slow RMS auto-leveler with slew-limited gain changes
# - lookaheadlimiter : True lookahead peak limiter (fixed-latency FIFO)
# - softclip         : Smooth safety clipper (tanh, ceiling-aware)
# - xfadein          : Clickless ramp-in at stream start/restart
#
# All filters are stateful, allocation-averse, and operate on (N, C) float32.
# --------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from filters import AudioFilter, register_filter, _DEF_EPS as _EPS

_F32 = np.float32


# --------- tiny helpers ---------
def _db_to_lin(db):
    """Vectorized dB -> linear. Accepts float or ndarray, returns float32 ndarray."""
    db_arr = np.asarray(db, dtype=np.float32)
    return np.power(10.0, db_arr / 20.0, dtype=np.float32)

def _lin_to_db_safe(x: NDArray[_F32]) -> NDArray[_F32]:
    """Vectorized linear -> dB with zero guard."""
    x_arr = np.asarray(x, dtype=np.float32)
    return 20.0 * np.log10(np.maximum(x_arr, _EPS))

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    return x[:, None] if x.ndim == 1 else x

def _exp_coef(ms: float, sr: int) -> float:
    # one-pole smoothing coefficient
    ms = max(0.05, float(ms))
    return float(np.exp(-1.0 / (ms * 1e-3 * sr)))


# --------- DC Blocker ---------
@register_filter(
    "dcblock",
    help="One-pole DC blocker per channel. Params: r(0.995..0.9995 default 0.995)"
)
class DCBlock(AudioFilter):
    """
    y[n] = x[n] - x[n-1] + r * y[n-1]
    r close to 1.0 sets the highpass corner very low; good for DC/very low rumble.
    """
    def __init__(self, params: Dict[str, Any]):
        self.r = float(params.get("r", 0.995))
        self._x1: Optional[np.ndarray] = None
        self._y1: Optional[np.ndarray] = None

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(np.float32, copy=False))
        n, C = x.shape
        if self._x1 is None or self._x1.shape[0] != C:
            self._x1 = np.zeros((C,), dtype=_F32)
            self._y1 = np.zeros((C,), dtype=_F32)
        x1 = self._x1; y1 = self._y1
        y = np.empty_like(x, dtype=_F32)

        r = self.r
        for i in range(n):
            xi = x[i]
            yi = xi - x1 + r * y1
            y[i] = yi
            x1 = xi
            y1 = yi

        self._x1 = x1
        self._y1 = y1
        np.clip(y, -1.0, 1.0, out=y)
        return y


# --------- Downward Expander / Gate ---------
@register_filter(
    "gate",
    help=("Downward expander with hysteresis. "
          "Params: threshold_open_db(-42), threshold_close_db(-48), ratio(2.0), "
          "attack_ms(5), release_ms(80), makeup_db(0)")
)
class NoiseGate(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.t_open = float(params.get("threshold_open_db", -42.0))
        self.t_close = float(params.get("threshold_close_db", -48.0))
        self.ratio = max(1.0, float(params.get("ratio", 2.0)))
        self.atk_ms = float(params.get("attack_ms", 5.0))
        self.rel_ms = float(params.get("release_ms", 80.0))
        self.mk_db = float(params.get("makeup_db", 0.0))

        self._env: Optional[np.ndarray] = None
        self._gain: Optional[np.ndarray] = None
        self._atk: Optional[float] = None
        self._rel: Optional[float] = None
        self._mk_lin: float = _db_to_lin(self.mk_db)
        self._open_mask: Optional[np.ndarray] = None  # hysteresis state per channel

    def _ensure(self, C: int, sr: int):
        if self._env is None or self._env.shape[0] != C:
            self._env = np.zeros((C,), dtype=_F32)
            self._gain = np.ones((C,), dtype=_F32)
            self._open_mask = np.zeros((C,), dtype=bool)
        self._atk = _exp_coef(self.atk_ms, sr)
        self._rel = _exp_coef(self.rel_ms, sr)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(_F32, copy=False))
        n, C = x.shape
        self._ensure(C, sr)

        env = self._env
        g = self._gain
        open_m = self._open_mask
        atk = float(self._atk); rel = float(self._rel)
        mk = self._mk_lin

        y = np.empty_like(x, dtype=_F32)
        for i in range(n):
            s = x[i]
            # peak detector with AR smoothing
            a = np.abs(s)
            faster = a > env
            acoef = np.where(faster, atk, rel)
            env = acoef * env + (1.0 - acoef) * a

            # level in dB per channel
            lvl_db = _lin_to_db_safe(env)

            # hysteresis open/close
            open_now = np.where(open_m, lvl_db > self.t_close, lvl_db > self.t_open)
            open_m = open_now

            # compute downward expansion when closed
            # over = (threshold - level); gain_db = -(1 - 1/ratio) * over (but only when below threshold)
            thr = np.where(open_m, self.t_open, self.t_close)
            below = thr - lvl_db
            gr_db = -(1.0 - 1.0 / self.ratio) * np.maximum(0.0, below)
            # smooth gain (own AR to avoid zipper)
            target_g = _db_to_lin(gr_db.astype(float))
            # separate smoothing for opening/closing
            faster_g = target_g < g
            gcoef = np.where(faster_g, atk, rel)
            g = gcoef * g + (1.0 - gcoef) * target_g

            y[i] = s * g * mk

        self._env = env
        self._gain = g
        self._open_mask = open_m
        np.clip(y, -1.0, 1.0, out=y)
        return y


# --------- AutoGain (slow RMS leveler) ---------
@register_filter(
    "autogain",
    help=("Slow RMS auto-leveler with slew-limited gain. "
          "Params: target_rms_db(-20), window_ms(400), max_gain_db(+9), min_gain_db(-9), slew_db_per_s(3)")
)
class AutoGain(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.target_db = float(params.get("target_rms_db", -20.0))
        self.win_ms = float(params.get("window_ms", 400.0))
        self.max_db = float(params.get("max_gain_db", +9.0))
        self.min_db = float(params.get("min_gain_db", -9.0))
        self.slew_db_s = float(params.get("slew_db_per_s", 3.0))

        self._buf: Optional[np.ndarray] = None  # FIFO RMS window per channel
        self._idx: int = 0
        self._filled: int = 0
        self._gain_db: Optional[np.ndarray] = None
        self._sr_last: Optional[int] = None
        self._win_len: Optional[int] = None

    def _ensure(self, C: int, sr: int):
        win_len = max(1, int(self.win_ms * 1e-3 * sr))
        if self._buf is None or self._buf.shape != (win_len, C) or self._sr_last != sr:
            self._buf = np.zeros((win_len, C), dtype=_F32)
            self._idx = 0
            self._filled = 0
            self._gain_db = np.zeros((C,), dtype=_F32)
            self._sr_last = sr
            self._win_len = win_len

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(_F32, copy=False))
        n, C = x.shape
        self._ensure(C, sr)
        buf = self._buf
        idx = self._idx
        filled = self._filled
        gain_db = self._gain_db
        win_len = int(self._win_len or 1)

        # precompute max step per sample
        max_step = self.slew_db_s / max(1, sr)

        y = np.empty_like(x, dtype=_F32)
        for i in range(n):
            # update FIFO with squared frame energy (approx: mean of squared sample per frame)
            buf[idx] = x[i] * x[i]
            idx += 1
            if idx >= win_len:
                idx = 0
            filled = min(filled + 1, win_len)

            # RMS estimate
            rms = float(np.sqrt(np.mean(buf[:filled], axis=0).mean() + _EPS))  # overall RMS across channels
            rms_db = 20.0 * np.log10(max(rms, _EPS))
            # target vs current -> desired gain
            desired_db = float(np.clip(self.target_db - rms_db, self.min_db, self.max_db))

            # slew limit
            delta = desired_db - gain_db  # vector, but all channels same desired; OK to apply per-channel
            step = np.clip(delta, -max_step, +max_step)
            gain_db = gain_db + step

            g_lin = _db_to_lin(gain_db)
            y[i] = x[i] * g_lin

        self._buf = buf
        self._idx = idx
        self._filled = filled
        self._gain_db = gain_db
        np.clip(y, -1.0, 1.0, out=y)
        return y


# --------- Lookahead Limiter (true lookahead, fixed latency) ---------
@register_filter(
    "lookaheadlimiter",
    help=("True lookahead peak limiter with fixed latency. "
          "Params: ceiling_db(-1), lookahead_ms(4), release_ms(50), makeup_db(0)")
)
class LookaheadLimiter(AudioFilter):
    """
    Fixed-latency FIFO per channel. We scan the lookahead window for peaks and
    compute the required gain to keep those peaks under ceiling, then decay via
    a release time constant. Designed to avoid per-block clicks/overs.
    """
    def __init__(self, params: Dict[str, Any]):
        self.ceil_db = float(params.get("ceiling_db", -1.0))
        self.mk_db   = float(params.get("makeup_db", 0.0))
        self.look_ms = float(params.get("lookahead_ms", 4.0))
        self.rel_ms  = float(params.get("release_ms", 50.0))

        self._ceil_lin: float = _db_to_lin(self.ceil_db)
        self._mk_lin: float = _db_to_lin(self.mk_db)

        # runtime
        self._delay: Optional[np.ndarray] = None     # (L, C)
        self._gain: Optional[np.ndarray] = None      # (C,)
        self._write: int = 0
        self._L: int = 0
        self._rel_a: float = 0.0
        self._sr: Optional[int] = None

    def _ensure(self, C: int, sr: int):
        L = max(1, int(self.look_ms * 1e-3 * sr))
        if (self._delay is None) or (self._delay.shape[0] != L) or (self._delay.shape[1] != C) or (self._sr != sr):
            self._delay = np.zeros((L, C), dtype=_F32)
            self._gain = np.ones((C,), dtype=_F32)
            self._write = 0
            self._L = L
            self._sr = sr
            self._rel_a = _exp_coef(self.rel_ms, sr)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(_F32, copy=False))
        n, C = x.shape
        self._ensure(C, sr)

        delay = self._delay
        write = self._write
        L = self._L
        ceil = self._ceil_lin
        rel_a = self._rel_a
        g = self._gain
        mk = self._mk_lin

        y = np.empty_like(x, dtype=_F32)

        # Precompute a per-sample index for reading with lookahead
        # The sample that exits the buffer at time i is at index (write + 1) % L
        for i in range(n):
            # Write current sample to delay line
            delay[write] = x[i]
            # The sample leaving delay line (after lookahead) is at read index:
            read = (write + 1) % L

            # Inspect lookahead window for upcoming peak
            # We can cheaply approximate by scanning the entire buffer magnitude peak
            peak_future = float(np.max(np.abs(delay)))
            need = peak_future / max(ceil, _EPS)  # >= 1 if would exceed

            # Update gain: if need > 1, apply that reduction immediately (hard-knee),
            # otherwise decay toward unity with release constant.
            if need > 1.0:
                target = 1.0 / need
                g = np.minimum(g, target)  # immediate clamp (no attack overs)
            else:
                g = 1.0 - (1.0 - g) * rel_a  # smooth back to 1

            # Read sample that exits delay, apply gain + makeup
            out = delay[read] * g * mk
            # Safety: enforce final ceiling
            np.clip(out, -ceil, ceil, out=out)
            y[i] = out

            # advance write pointer
            write = read

        self._delay = delay
        self._write = write
        self._gain = g
        np.clip(y, -1.0, 1.0, out=y)
        return y


# --------- Soft safety clipper ---------
@register_filter(
    "softclip",
    help="Tanh soft clipper with output ceiling. Params: drive_db(0..+6), ceiling_db(-0.9)"
)
class SoftClip(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.drive = float(params.get("drive_db", 0.0))
        self.ceiling = float(params.get("ceiling_db", -0.9))
        self._gain = _db_to_lin(self.drive)
        self._ceil = _db_to_lin(self.ceiling)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(_F32, copy=False))
        y = np.tanh(x * self._gain, dtype=_F32)
        np.clip(y, -self._ceil, self._ceil, out=y)
        return y


# --------- Ramp-in / Crossfade-in (for stream start/restart) ---------
@register_filter(
    "xfadein",
    help="Simple ramp-in to suppress clicks on stream start. Params: ms(20)"
)
class CrossfadeIn(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.ms = float(params.get("ms", 20.0))
        self._remain: int = 0
        self._sr: Optional[int] = None

    def _ensure(self, sr: int):
        if self._sr != sr:
            self._sr = sr
            self._remain = int(self.ms * 1e-3 * sr)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(_F32, copy=False))
        self._ensure(sr)
        if self._remain <= 0:
            return x
        n = x.shape[0]
        k = min(n, self._remain)
        ramp = (np.linspace(0.0, 1.0, k, dtype=_F32))[:, None]
        x[:k] *= ramp
        self._remain -= k
        return x