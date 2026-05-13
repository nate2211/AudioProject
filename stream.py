# stream.py
# --------------------------------------------
# Native-backed real-time friendly audio filters for streaming:
# - dcblock          : One-pole DC blocker, native when available
# - gate             : Downward expander / noise gate with hysteresis
# - autogain         : Slow RMS auto-leveler with slew-limited gain changes
# - lookaheadlimiter : True lookahead peak limiter
# - softclip         : Native tanh clipper when available
# - xfadein          : Clickless ramp-in at stream start/restart
#
# Same public filter names/signatures as previous stream.py.
# --------------------------------------------

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
from numpy.typing import NDArray

from filters import AudioFilter, register_filter, _DEF_EPS as _EPS


try:
    from native import (
        is_available as _native_is_available,
        NativeDCBlocker,
        NativeSoftClipper,
        NativeLimiter,
    )

    _NATIVE_OK = bool(_native_is_available())
except Exception:
    _NATIVE_OK = False
    NativeDCBlocker = None
    NativeSoftClipper = None
    NativeLimiter = None


_F32 = np.float32


# --------- tiny helpers ---------

def _db_to_lin(db):
    db_arr = np.asarray(db, dtype=np.float32)
    return np.power(10.0, db_arr / 20.0, dtype=np.float32)


def _lin_to_db_safe(x: NDArray[_F32]) -> NDArray[_F32]:
    x_arr = np.asarray(x, dtype=np.float32)
    return 20.0 * np.log10(np.maximum(x_arr, _EPS))


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=_F32)
    if x.ndim == 1:
        return np.ascontiguousarray(x[:, None], dtype=_F32)
    if x.ndim != 2:
        raise ValueError(f"Audio block must be 1D or 2D, got shape={x.shape!r}")
    return np.ascontiguousarray(x, dtype=_F32)


def _exp_coef(ms: float, sr: int) -> float:
    ms = max(0.05, float(ms))
    return float(np.exp(-1.0 / (ms * 1e-3 * sr)))


def _safe_audio(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=_F32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(_F32)
    return np.clip(x, -1.0, 1.0).astype(_F32, copy=False)


# --------- DC Blocker ---------

@register_filter(
    "dcblock",
    help="One-pole DC blocker per channel. Params: r(0.995..0.9995 default 0.995)"
)
class DCBlock(AudioFilter):
    """
    y[n] = x[n] - x[n-1] + r * y[n-1]
    Native-backed through AudioProject.dll when available.
    """
    def __init__(self, params: Dict[str, Any]):
        self.r = float(params.get("r", 0.995))
        self._x1: Optional[np.ndarray] = None
        self._y1: Optional[np.ndarray] = None
        self._native = None

        if _NATIVE_OK and NativeDCBlocker is not None:
            try:
                self._native = NativeDCBlocker(r=self.r, channels=2)
            except Exception:
                self._native = None

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)

        if self._native is not None:
            try:
                self._native.set_r(self.r)
                return self._native.process(x, sr)
            except Exception:
                self._native = None

        n, C = x.shape

        if self._x1 is None or self._x1.shape[0] != C:
            self._x1 = np.zeros((C,), dtype=_F32)
            self._y1 = np.zeros((C,), dtype=_F32)

        x1 = self._x1
        y1 = self._y1
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
        return _safe_audio(y)


# --------- Downward Expander / Gate ---------

@register_filter(
    "gate",
    help=(
        "Downward expander with hysteresis. "
        "Params: threshold_open_db(-42), threshold_close_db(-48), ratio(2.0), "
        "attack_ms(5), release_ms(80), makeup_db(0)"
    )
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
        self._mk_lin: float = float(_db_to_lin(self.mk_db))
        self._open_mask: Optional[np.ndarray] = None

    def _ensure(self, C: int, sr: int):
        if self._env is None or self._env.shape[0] != C:
            self._env = np.zeros((C,), dtype=_F32)
            self._gain = np.ones((C,), dtype=_F32)
            self._open_mask = np.zeros((C,), dtype=bool)

        self._atk = _exp_coef(self.atk_ms, sr)
        self._rel = _exp_coef(self.rel_ms, sr)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        n, C = x.shape
        self._ensure(C, sr)

        env = self._env
        g = self._gain
        open_m = self._open_mask
        atk = float(self._atk)
        rel = float(self._rel)
        mk = self._mk_lin

        y = np.empty_like(x, dtype=_F32)

        for i in range(n):
            s = x[i]
            a = np.abs(s)
            faster = a > env
            acoef = np.where(faster, atk, rel)
            env = acoef * env + (1.0 - acoef) * a

            lvl_db = _lin_to_db_safe(env)
            open_now = np.where(open_m, lvl_db > self.t_close, lvl_db > self.t_open)
            open_m = open_now

            thr = np.where(open_m, self.t_open, self.t_close)
            below = thr - lvl_db
            gr_db = -(1.0 - 1.0 / self.ratio) * np.maximum(0.0, below)
            target_g = _db_to_lin(gr_db.astype(float))

            faster_g = target_g < g
            gcoef = np.where(faster_g, atk, rel)
            g = gcoef * g + (1.0 - gcoef) * target_g

            y[i] = s * g * mk

        self._env = env
        self._gain = g
        self._open_mask = open_m

        return _safe_audio(y)


# --------- AutoGain ---------

@register_filter(
    "autogain",
    help=(
        "Slow RMS auto-leveler with slew-limited gain. "
        "Params: target_rms_db(-20), window_ms(400), max_gain_db(+9), "
        "min_gain_db(-9), slew_db_per_s(3)"
    )
)
class AutoGain(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.target_db = float(params.get("target_rms_db", -20.0))
        self.win_ms = float(params.get("window_ms", 400.0))
        self.max_db = float(params.get("max_gain_db", +9.0))
        self.min_db = float(params.get("min_gain_db", -9.0))
        self.slew_db_s = float(params.get("slew_db_per_s", 3.0))

        self._buf: Optional[np.ndarray] = None
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
        x = _ensure_2d(block)
        n, C = x.shape
        self._ensure(C, sr)

        buf = self._buf
        idx = self._idx
        filled = self._filled
        gain_db = self._gain_db
        win_len = int(self._win_len or 1)
        max_step = self.slew_db_s / max(1, sr)

        y = np.empty_like(x, dtype=_F32)

        for i in range(n):
            buf[idx] = x[i] * x[i]
            idx += 1

            if idx >= win_len:
                idx = 0

            filled = min(filled + 1, win_len)
            rms = float(np.sqrt(np.mean(buf[:filled], axis=0).mean() + _EPS))
            rms_db = 20.0 * np.log10(max(rms, _EPS))
            desired_db = float(np.clip(self.target_db - rms_db, self.min_db, self.max_db))

            delta = desired_db - gain_db
            step = np.clip(delta, -max_step, +max_step)
            gain_db = gain_db + step

            g_lin = _db_to_lin(gain_db)
            y[i] = x[i] * g_lin

        self._buf = buf
        self._idx = idx
        self._filled = filled
        self._gain_db = gain_db

        return _safe_audio(y)


# --------- Lookahead Limiter ---------

@register_filter(
    "lookaheadlimiter",
    help=(
        "True lookahead peak limiter with fixed latency. "
        "Params: ceiling_db(-1), lookahead_ms(4), release_ms(50), makeup_db(0)"
    )
)
class LookaheadLimiter(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.ceil_db = float(params.get("ceiling_db", -1.0))
        self.mk_db = float(params.get("makeup_db", 0.0))
        self.look_ms = float(params.get("lookahead_ms", 4.0))
        self.rel_ms = float(params.get("release_ms", 50.0))

        self._ceil_lin: float = float(_db_to_lin(self.ceil_db))
        self._mk_lin: float = float(_db_to_lin(self.mk_db))

        self._delay: Optional[np.ndarray] = None
        self._gain: Optional[np.ndarray] = None
        self._write: int = 0
        self._L: int = 0
        self._rel_a: float = 0.0
        self._sr: Optional[int] = None

    def _ensure(self, C: int, sr: int):
        L = max(1, int(self.look_ms * 1e-3 * sr))

        if (
            self._delay is None
            or self._delay.shape[0] != L
            or self._delay.shape[1] != C
            or self._sr != sr
        ):
            self._delay = np.zeros((L, C), dtype=_F32)
            self._gain = np.ones((C,), dtype=_F32)
            self._write = 0
            self._L = L
            self._sr = sr
            self._rel_a = _exp_coef(self.rel_ms, sr)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
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

        for i in range(n):
            delay[write] = x[i]
            read = (write + 1) % L

            peak_future = float(np.max(np.abs(delay)))
            need = peak_future / max(ceil, _EPS)

            if need > 1.0:
                target = 1.0 / need
                g = np.minimum(g, target)
            else:
                g = 1.0 - (1.0 - g) * rel_a

            out = delay[read] * g * mk
            np.clip(out, -ceil, ceil, out=out)
            y[i] = out
            write = read

        self._delay = delay
        self._write = write
        self._gain = g

        return _safe_audio(y)


# --------- Soft safety clipper ---------

@register_filter(
    "softclip",
    help="Tanh soft clipper with output ceiling. Params: drive_db(0), ceiling_db(-0.5)"
)
class SoftClip(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.drive_db = float(params.get("drive_db", params.get("drive", 0.0)))
        self.ceiling_db = float(params.get("ceiling_db", -0.5))

        self._native = None
        if _NATIVE_OK and NativeSoftClipper is not None:
            try:
                self._native = NativeSoftClipper(drive=float(_db_to_lin(self.drive_db)))
            except Exception:
                self._native = None

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        drive = float(_db_to_lin(self.drive_db))
        ceiling = float(_db_to_lin(self.ceiling_db))

        if self._native is not None:
            try:
                self._native.set_drive(drive)
                y = self._native.process(x, sr)
                y = y * ceiling
                return _safe_audio(y)
            except Exception:
                self._native = None

        denom = np.tanh(drive)
        if abs(float(denom)) < 1e-8:
            y = x.copy()
        else:
            y = np.tanh(x * drive) / denom

        y *= ceiling
        return _safe_audio(y)


# --------- Fade in ---------

@register_filter(
    "xfadein",
    help="Clickless ramp-in at stream start/restart. Params: ms(5)"
)
class XFadeIn(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.ms = float(params.get("ms", params.get("fade_ms", 5.0)))
        self._done = False
        self._left = 0
        self._total = 0
        self._sr: Optional[int] = None

    def reset(self):
        self._done = False
        self._left = 0
        self._total = 0
        self._sr = None

    def _ensure(self, sr: int):
        if self._sr != sr:
            self._sr = sr
            self._total = max(1, int(self.ms * 1e-3 * sr))
            self._left = self._total
            self._done = False

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        self._ensure(sr)

        if self._done or x.shape[0] == 0:
            return x

        y = x.copy()
        n = y.shape[0]
        fade_n = min(n, self._left)

        if fade_n > 0:
            already = self._total - self._left
            ramp = np.linspace(
                already / max(1, self._total),
                (already + fade_n) / max(1, self._total),
                fade_n,
                dtype=_F32,
            )
            y[:fade_n] *= ramp[:, None]
            self._left -= fade_n

        if self._left <= 0:
            self._done = True

        return _safe_audio(y)