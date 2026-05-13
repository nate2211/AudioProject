# =============================
# filters.py
# Native-backed stream-safe audio filter registry
# =============================
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from scipy import signal


try:
    from native import (
        is_available as _native_is_available,
        NativeSOS as _NativeSOS,
        NativeCompressor as _NativeCompressor,
        NativeLimiter as _NativeLimiter,
        NativeSoftClipper as _NativeSoftClipper,
    )

    _NATIVE_OK = bool(_native_is_available())
except Exception:
    _NATIVE_OK = False
    _NativeSOS = None
    _NativeCompressor = None
    _NativeLimiter = None
    _NativeSoftClipper = None


# =============================================================================
# Param specs for GUI auto-generation
# =============================================================================

@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str = "float"
    default: Any = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    scale: str = "linear"
    unit: str = ""
    choices: Optional[List[str]] = None


@dataclass
class FilterInfo:
    help: str
    factory: Any
    params: List[ParamSpec]


_REGISTRY: Dict[str, FilterInfo] = {}


def register_filter(
    name: str,
    *,
    help: str,
    params: Optional[List[ParamSpec]] = None,
) -> Callable[[Any], Any]:
    key = name.strip().lower()

    def _decorator(factory: Any) -> Any:
        if key in _REGISTRY:
            raise ValueError(f"Duplicate filter name: {name}")
        _REGISTRY[key] = FilterInfo(help=help, factory=factory, params=list(params or []))
        return factory

    return _decorator


def available_filters() -> Dict[str, str]:
    return {k: v.help for k, v in sorted(_REGISTRY.items(), key=lambda kv: kv[0])}


def filter_param_specs(name: str) -> List[ParamSpec]:
    key = (name or "").strip().lower()
    info = _REGISTRY.get(key)
    return list(info.params) if info else []


class AudioFilter:
    def set_params(self, **kwargs: Any) -> None:
        return

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError

    def flush(self) -> Optional[np.ndarray]:
        return None


# =============================================================================
# Utilities
# =============================================================================

_DEF_EPS = 1e-12
_F32 = np.float32


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected audio array shaped (frames, channels), got {x.shape!r}")
    return np.ascontiguousarray(x, dtype=np.float32)


def _db_to_lin(db: float) -> float:
    return float(10 ** (float(db) / 20.0))


def _lin_to_db(x: float) -> float:
    return float(20.0 * math.log10(max(float(x), _DEF_EPS)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def _safe_audio(x: np.ndarray, ceiling: float = 1.0) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.clip(y, -float(ceiling), float(ceiling)).astype(np.float32, copy=False)


def _safe_norm_freq(sr: int, f0: float) -> float:
    f0 = float(max(1e-3, min(float(f0), 0.5 * sr * 0.999)))
    return 2.0 * math.pi * (f0 / sr)


def _biquad_to_sos(b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> np.ndarray:
    if abs(a0) < 1e-12:
        a0 = 1e-12
    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )


def _iir_peaking_sos(sr: int, f0: float, Q: float, gain_db: float) -> np.ndarray:
    Q = float(max(1e-6, Q))
    A = 10.0 ** (float(gain_db) / 40.0)
    w0 = _safe_norm_freq(sr, f0)
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw / (2.0 * Q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A
    return _biquad_to_sos(b0, b1, b2, a0, a1, a2)


def _iir_shelf_sos(
    sr: int,
    f0: float,
    gain_db: float,
    *,
    shelf_type: str = "low",
    slope: float = 1.0,
) -> np.ndarray:
    S = float(max(1e-6, slope))
    A = 10.0 ** (float(gain_db) / 40.0)
    w0 = _safe_norm_freq(sr, f0)
    cw = math.cos(w0)
    sw = math.sin(w0)

    sqrt_arg = (A + 1.0 / A) * (1.0 / S - 1.0) + 2.0
    sqrt_arg = max(0.0, float(sqrt_arg))
    alpha = sw / 2.0 * math.sqrt(sqrt_arg)

    st = shelf_type.lower()

    if st in ("low", "lowshelf", "ls"):
        b0 = A * ((A + 1) - (A - 1) * cw + 2.0 * math.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cw)
        b2 = A * ((A + 1) - (A - 1) * cw - 2.0 * math.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cw + 2.0 * math.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cw)
        a2 = (A + 1) + (A - 1) * cw - 2.0 * math.sqrt(A) * alpha

    elif st in ("high", "highshelf", "hs"):
        b0 = A * ((A + 1) + (A - 1) * cw + 2.0 * math.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - 2.0 * math.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cw + 2.0 * math.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - 2.0 * math.sqrt(A) * alpha

    else:
        raise ValueError(f"Unknown shelf_type '{shelf_type}'")

    return _biquad_to_sos(b0, b1, b2, a0, a1, a2)


def _soft_clip(x: np.ndarray) -> np.ndarray:
    x = _ensure_2d(x)

    if _NATIVE_OK and _NativeSoftClipper is not None:
        try:
            return _NativeSoftClipper(drive=1.0).process(x, 48000)
        except Exception:
            pass

    return np.tanh(x).astype(np.float32, copy=False)


def _ramp(start: float, end: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([end], dtype=np.float32)
    return np.linspace(start, end, n, dtype=np.float32)


# =============================================================================
# Builder
# =============================================================================

def build_filter(name: str, **kwargs: Any) -> AudioFilter:
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown filter '{name}'. Available: {', '.join(available_filters().keys()) or '(none)'}")

    info = _REGISTRY[key]
    factory = info.factory

    if isinstance(factory, type):
        try:
            return factory(**kwargs)
        except TypeError:
            return factory(kwargs)

    try:
        return factory(**kwargs)
    except TypeError:
        return factory(kwargs)


# =============================================================================
# Smooth gain
# =============================================================================

class SmoothGain(AudioFilter):
    def __init__(self, g_lin: float, ramp_ms: float = 5.0):
        self._g = float(g_lin)
        self._target = float(g_lin)
        self._ramp_ms = float(max(0.1, ramp_ms))

    def set_params(self, **kwargs: Any) -> None:
        if "lin" in kwargs and kwargs["lin"] is not None:
            self._target = float(kwargs["lin"])
        elif "db" in kwargs and kwargs["db"] is not None:
            self._target = _db_to_lin(float(kwargs["db"]))

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        n = x.shape[0]

        if n == 0:
            return x

        if self._target == self._g:
            return _safe_audio(x * self._g)

        ramp_n = int(max(1, (self._ramp_ms * 1e-3) * sr))
        ramp_n = min(ramp_n, n)

        g0 = float(self._g)
        g1 = float(self._target)

        y = x.copy()
        y[:ramp_n] *= _ramp(g0, g1, ramp_n)[:, None]

        if ramp_n < n:
            y[ramp_n:] *= g1

        self._g = g1
        return _safe_audio(y)


# =============================================================================
# Native-backed SOS base
# =============================================================================

class _IIRSOS(AudioFilter):
    def __init__(self):
        self.sos: Optional[np.ndarray] = None
        self.zi: Optional[np.ndarray] = None
        self._initd = False
        self._sr: Optional[int] = None
        self._C: Optional[int] = None
        self._native = None

    def _design(self, sr: int) -> np.ndarray:
        raise NotImplementedError

    def _ensure(self, sr: int, C: int, x0: Optional[np.ndarray] = None):
        if self._initd and self._sr == sr and self._C == C:
            return

        sos = self._design(sr).astype(np.float64, copy=False)

        if sos.ndim != 2 or sos.shape[1] != 6:
            raise ValueError("IIR design must return SOS with shape (S,6).")

        self.sos = np.ascontiguousarray(sos.astype(np.float32), dtype=np.float32)
        self._sr = sr
        self._C = C
        self._initd = True
        self.zi = None
        self._native = None

        if _NATIVE_OK and _NativeSOS is not None:
            try:
                self._native = _NativeSOS(self.sos, channels=C, reset_state=True)
                return
            except Exception:
                self._native = None

        zi0 = signal.sosfilt_zi(self.sos.astype(np.float64))
        zi = np.zeros((self.sos.shape[0], C, 2), dtype=np.float32)

        if x0 is not None and x0.size == C:
            for ch in range(C):
                zi[:, ch, :] = (zi0 * float(x0[ch])).astype(np.float32)

        self.zi = zi

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        n, C = x.shape

        if n == 0:
            return x

        self._ensure(sr, C, x0=x[0] if not self._initd else None)

        if self.sos is None:
            return x

        if self._native is not None:
            try:
                return _safe_audio(self._native.process(x, sr))
            except Exception:
                self._native = None
                self.zi = None

        if self.zi is None or self.zi.shape[1] != C or self.zi.shape[0] != self.sos.shape[0]:
            zi0 = signal.sosfilt_zi(self.sos.astype(np.float64))
            self.zi = np.tile(zi0[:, None, :], (1, C, 1)).astype(np.float32)

        y = np.empty_like(x, dtype=np.float32)

        for ch in range(C):
            y[:, ch], self.zi[:, ch, :] = signal.sosfilt(
                self.sos.astype(np.float64),
                x[:, ch],
                zi=self.zi[:, ch, :],
            )

        return _safe_audio(y)


# =============================================================================
# Filters
# =============================================================================

@register_filter(
    "gain",
    help="Smooth gain. Params: db(0) lin(None)",
    params=[
        ParamSpec("db", kind="float", default=0.0, minimum=-60.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("lin", kind="float", default=None, minimum=0.0, maximum=8.0, step=0.01, scale="linear", unit="x"),
        ParamSpec("ramp_ms", kind="float", default=5.0, minimum=0.1, maximum=50.0, step=0.1, scale="linear", unit="ms"),
    ],
)
class Gain(AudioFilter):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        p = dict(params or {})
        p.update(kwargs)

        ramp_ms = float(p.get("ramp_ms", 5.0))
        lin = p.get("lin", None)

        if lin is None:
            lin = _db_to_lin(float(p.get("db", 0.0)))

        self._g = SmoothGain(float(lin), ramp_ms=ramp_ms)

    def set_params(self, **kwargs: Any) -> None:
        self._g.set_params(**kwargs)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        return self._g.process(block, sr)


@register_filter(
    "normalize",
    help="Peak normalize (slowly updates). Params: peak(0.98)",
    params=[
        ParamSpec("peak", kind="float", default=0.98, minimum=0.01, maximum=1.0, step=0.01, scale="linear"),
        ParamSpec("ramp_ms", kind="float", default=10.0, minimum=0.1, maximum=200.0, step=0.5, scale="linear", unit="ms"),
    ],
)
class Normalize(AudioFilter):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        p = dict(params or {})
        p.update(kwargs)
        self.target = float(p.get("peak", 0.98))
        self.ramp_ms = float(p.get("ramp_ms", 10.0))
        self._max = 0.0
        self._g = SmoothGain(1.0, ramp_ms=self.ramp_ms)

    def set_params(self, **kwargs: Any) -> None:
        if "peak" in kwargs and kwargs["peak"] is not None:
            self.target = float(kwargs["peak"])
        if "ramp_ms" in kwargs and kwargs["ramp_ms"] is not None:
            self.ramp_ms = float(kwargs["ramp_ms"])
            self._g._ramp_ms = self.ramp_ms

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        m = float(np.max(np.abs(x))) if x.size else 0.0

        if m > self._max:
            self._max = m

        if self._max < _DEF_EPS:
            return x

        self._g.set_params(lin=float(self.target / self._max))
        return _safe_audio(self._g.process(x, sr))


@register_filter(
    "lowpass",
    help="Butterworth LPF (SOS). Params: cutoff(Hz) order(4)",
    params=[
        ParamSpec("cutoff", kind="float", default=8000.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("order", kind="int", default=4, minimum=1, maximum=12, step=1),
    ],
)
class Lowpass(_IIRSOS):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        super().__init__()
        p = dict(params or {})
        p.update(kwargs)
        self.cutoff = float(p.get("cutoff", 8000.0))
        self.order = int(p.get("order", 4))

    def set_params(self, **kwargs: Any) -> None:
        if "cutoff" in kwargs and kwargs["cutoff"] is not None:
            self.cutoff = float(kwargs["cutoff"])
            self._initd = False
        if "order" in kwargs and kwargs["order"] is not None:
            self.order = int(kwargs["order"])
            self._initd = False

    def _design(self, sr: int) -> np.ndarray:
        nyq = 0.5 * sr
        wn = _clamp(self.cutoff / nyq, 1e-5, 0.9999)
        return signal.butter(int(max(1, min(12, self.order))), wn, btype="low", output="sos")


@register_filter(
    "highpass",
    help="Butterworth HPF (SOS). Params: cutoff(Hz) order(4)",
    params=[
        ParamSpec("cutoff", kind="float", default=80.0, minimum=10.0, maximum=20000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("order", kind="int", default=4, minimum=1, maximum=12, step=1),
    ],
)
class Highpass(_IIRSOS):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        super().__init__()
        p = dict(params or {})
        p.update(kwargs)
        self.cutoff = float(p.get("cutoff", 80.0))
        self.order = int(p.get("order", 4))

    def set_params(self, **kwargs: Any) -> None:
        if "cutoff" in kwargs and kwargs["cutoff"] is not None:
            self.cutoff = float(kwargs["cutoff"])
            self._initd = False
        if "order" in kwargs and kwargs["order"] is not None:
            self.order = int(kwargs["order"])
            self._initd = False

    def _design(self, sr: int) -> np.ndarray:
        nyq = 0.5 * sr
        wn = _clamp(self.cutoff / nyq, 1e-5, 0.9999)
        return signal.butter(int(max(1, min(12, self.order))), wn, btype="high", output="sos")


@register_filter(
    "bandpass",
    help="Butterworth BPF (SOS). Params: low(Hz) high(Hz) order(4)",
    params=[
        ParamSpec("low", kind="float", default=200.0, minimum=10.0, maximum=20000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("high", kind="float", default=2000.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("order", kind="int", default=4, minimum=1, maximum=12, step=1),
    ],
)
class Bandpass(_IIRSOS):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        super().__init__()
        p = dict(params or {})
        p.update(kwargs)
        self.low = float(p.get("low", 200.0))
        self.high = float(p.get("high", 2000.0))
        self.order = int(p.get("order", 4))

    def set_params(self, **kwargs: Any) -> None:
        changed = False
        if "low" in kwargs and kwargs["low"] is not None:
            self.low = float(kwargs["low"])
            changed = True
        if "high" in kwargs and kwargs["high"] is not None:
            self.high = float(kwargs["high"])
            changed = True
        if "order" in kwargs and kwargs["order"] is not None:
            self.order = int(kwargs["order"])
            changed = True
        if changed:
            self._initd = False

    def _design(self, sr: int) -> np.ndarray:
        nyq = 0.5 * sr
        lo = max(10.0, float(self.low))
        hi = max(lo + 10.0, float(self.high))
        lo_n = _clamp(lo / nyq, 1e-5, 0.9998)
        hi_n = _clamp(hi / nyq, lo_n + 1e-5, 0.9999)
        return signal.butter(int(max(1, min(12, self.order))), [lo_n, hi_n], btype="band", output="sos")


@register_filter(
    "compress",
    help="Smooth compressor. Params: threshold_db(-24) ratio(4) attack_ms(5) release_ms(50) makeup_db(0) knee_db(6) stereo_link(true)",
    params=[
        ParamSpec("threshold_db", kind="float", default=-24.0, minimum=-60.0, maximum=0.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("ratio", kind="float", default=4.0, minimum=1.0, maximum=20.0, step=0.1),
        ParamSpec("attack_ms", kind="float", default=5.0, minimum=0.1, maximum=200.0, step=0.1, unit="ms"),
        ParamSpec("release_ms", kind="float", default=50.0, minimum=1.0, maximum=2000.0, step=1.0, unit="ms"),
        ParamSpec("makeup_db", kind="float", default=0.0, minimum=-12.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("knee_db", kind="float", default=6.0, minimum=0.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("stereo_link", kind="bool", default=True),
    ],
)
class Compressor(AudioFilter):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        p = dict(params or {})
        p.update(kwargs)

        self.th = float(p.get("threshold_db", -24.0))
        self.ratio = max(1.0, float(p.get("ratio", 4.0)))
        self.atk_ms = float(p.get("attack_ms", 5.0))
        self.rel_ms = float(p.get("release_ms", 50.0))
        self.mk_db = float(p.get("makeup_db", 0.0))
        self.knee_db = float(p.get("knee_db", 6.0))

        sl = p.get("stereo_link", True)
        self.stereo_link = sl.strip().lower() in ("1", "true", "yes", "on") if isinstance(sl, str) else bool(sl)

        self._env: Optional[np.ndarray] = None
        self._native = None
        self._make_native()

    def _make_native(self):
        self._native = None
        if _NATIVE_OK and _NativeCompressor is not None and self.stereo_link:
            try:
                self._native = _NativeCompressor(
                    threshold_db=self.th,
                    ratio=self.ratio,
                    knee_db=self.knee_db,
                    attack_ms=self.atk_ms,
                    release_ms=self.rel_ms,
                    makeup_db=self.mk_db,
                    mix=1.0,
                )
            except Exception:
                self._native = None

    def set_params(self, **kwargs: Any) -> None:
        rebuild = False

        if "threshold_db" in kwargs and kwargs["threshold_db"] is not None:
            self.th = float(kwargs["threshold_db"])
        if "ratio" in kwargs and kwargs["ratio"] is not None:
            self.ratio = max(1.0, float(kwargs["ratio"]))
        if "attack_ms" in kwargs and kwargs["attack_ms"] is not None:
            self.atk_ms = float(kwargs["attack_ms"])
        if "release_ms" in kwargs and kwargs["release_ms"] is not None:
            self.rel_ms = float(kwargs["release_ms"])
        if "makeup_db" in kwargs and kwargs["makeup_db"] is not None:
            self.mk_db = float(kwargs["makeup_db"])
        if "knee_db" in kwargs and kwargs["knee_db"] is not None:
            self.knee_db = float(kwargs["knee_db"])
        if "stereo_link" in kwargs and kwargs["stereo_link"] is not None:
            self.stereo_link = bool(kwargs["stereo_link"])
            rebuild = True

        if self._native is not None:
            try:
                self._native.set_params(
                    threshold_db=self.th,
                    ratio=self.ratio,
                    knee_db=self.knee_db,
                    attack_ms=self.atk_ms,
                    release_ms=self.rel_ms,
                    makeup_db=self.mk_db,
                    mix=1.0,
                )
            except Exception:
                rebuild = True

        if rebuild:
            self._make_native()

    def _fallback_process(self, x: np.ndarray, sr: int) -> np.ndarray:
        n, C = x.shape
        if self._env is None or self._env.shape[0] != C:
            self._env = np.zeros((C,), dtype=np.float32)

        atk = math.exp(-1.0 / (max(0.1, self.atk_ms) * 1e-3 * max(1, sr)))
        rel = math.exp(-1.0 / (max(0.1, self.rel_ms) * 1e-3 * max(1, sr)))
        mk = _db_to_lin(self.mk_db)
        out = np.empty_like(x)

        for i in range(n):
            row = x[i]
            if self.stereo_link and C > 1:
                peak = float(np.max(np.abs(row)))
                for ch in range(C):
                    coeff = atk if peak > self._env[ch] else rel
                    self._env[ch] = coeff * self._env[ch] + (1.0 - coeff) * peak
                env = float(np.max(self._env))
                over = _lin_to_db(env) - self.th
                gr_db = -max(0.0, over) * (1.0 - 1.0 / self.ratio)
                gain = _db_to_lin(gr_db) * mk
                out[i] = row * gain
            else:
                for ch in range(C):
                    p = abs(float(row[ch]))
                    coeff = atk if p > self._env[ch] else rel
                    self._env[ch] = coeff * self._env[ch] + (1.0 - coeff) * p
                    over = _lin_to_db(float(self._env[ch])) - self.th
                    gr_db = -max(0.0, over) * (1.0 - 1.0 / self.ratio)
                    out[i, ch] = row[ch] * _db_to_lin(gr_db) * mk

        return _safe_audio(out)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)

        if self._native is not None:
            try:
                return _safe_audio(self._native.process(x, sr))
            except Exception:
                self._native = None

        return self._fallback_process(x, sr)


@register_filter(
    "limiter",
    help="Fast peak limiter. Params: ceiling_db(-1) release_ms(50)",
    params=[
        ParamSpec("ceiling_db", kind="float", default=-1.0, minimum=-24.0, maximum=0.0, step=0.1, scale="db", unit="dB"),
        ParamSpec("release_ms", kind="float", default=50.0, minimum=1.0, maximum=2000.0, step=1.0, unit="ms"),
    ],
)
class Limiter(AudioFilter):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        p = dict(params or {})
        p.update(kwargs)

        self.ceil_db = float(p.get("ceiling_db", -1.0))
        self.rel_ms = float(p.get("release_ms", 50.0))
        self._env = 1.0
        self._ceil_lin = _db_to_lin(self.ceil_db)
        self._native = None

        if _NATIVE_OK and _NativeLimiter is not None:
            try:
                self._native = _NativeLimiter(
                    ceiling_db=self.ceil_db,
                    attack_ms=0.1,
                    release_ms=self.rel_ms,
                )
            except Exception:
                self._native = None

    def set_params(self, **kwargs: Any) -> None:
        if "ceiling_db" in kwargs and kwargs["ceiling_db"] is not None:
            self.ceil_db = float(kwargs["ceiling_db"])
            self._ceil_lin = _db_to_lin(self.ceil_db)
        if "release_ms" in kwargs and kwargs["release_ms"] is not None:
            self.rel_ms = float(kwargs["release_ms"])

        if self._native is not None:
            try:
                self._native.set_params(
                    ceiling_db=self.ceil_db,
                    attack_ms=0.1,
                    release_ms=self.rel_ms,
                )
            except Exception:
                self._native = None

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)

        if self._native is not None:
            try:
                return _safe_audio(self._native.process(x, sr), ceiling=1.0)
            except Exception:
                self._native = None

        rel = math.exp(-1.0 / (max(0.1, self.rel_ms) * 1e-3 * max(1, sr)))
        out = np.empty_like(x)
        env = float(self._env)
        ceil_lin = float(self._ceil_lin)

        for i in range(x.shape[0]):
            env *= rel
            peak = float(np.max(np.abs(x[i])))
            needed = peak / max(ceil_lin, _DEF_EPS)
            if needed > 1.0 and needed > env:
                env = needed
            gain = 1.0 / (env if env > 1.0 else 1.0)
            out[i] = x[i] * gain

        self._env = env
        return _safe_audio(out, ceiling=ceil_lin)


@register_filter(
    "mixdown",
    help="Stereo→Mono mixdown. Params: mode(average|sum|mid|left|right), pan_law_db(-3), normalize(false), norm_target_db(-1), norm_max_boost_db(6), ramp_ms(5)",
    params=[
        ParamSpec("mode", kind="enum", default="average", choices=["average", "sum", "mid", "left", "right"]),
        ParamSpec("pan_law_db", kind="float", default=-3.0, minimum=-6.0, maximum=0.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("normalize", kind="bool", default=False),
        ParamSpec("norm_target_db", kind="float", default=-1.0, minimum=-12.0, maximum=-0.1, step=0.1, scale="db", unit="dB"),
        ParamSpec("norm_max_boost_db", kind="float", default=6.0, minimum=0.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("ramp_ms", kind="float", default=5.0, minimum=0.1, maximum=50.0, step=0.1, unit="ms"),
    ],
)
class Mixdown(AudioFilter):
    def __init__(self, params: dict | None = None, **kwargs):
        p = dict(params or {})
        p.update(kwargs)

        self.mode = str(p.get("mode", "average")).lower()
        self.pan_law_db = float(p.get("pan_law_db", -3.0))
        self.normalize = bool(p.get("normalize", False))
        self.norm_target_db = float(p.get("norm_target_db", -1.0))
        self.norm_max_boost_db = float(p.get("norm_max_boost_db", 6.0))
        self.ramp_ms = float(p.get("ramp_ms", 5.0))
        self._pan_gain = SmoothGain(1.0, ramp_ms=self.ramp_ms)
        self._norm_gain = SmoothGain(1.0, ramp_ms=self.ramp_ms)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if k == "mode":
                self.mode = str(v).lower()
            elif hasattr(self, k):
                setattr(self, k, v)

        self._pan_gain._ramp_ms = self.ramp_ms
        self._norm_gain._ramp_ms = self.ramp_ms

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)

        if x.shape[1] == 1:
            return x

        L = x[:, 0]
        R = x[:, 1]

        if self.mode == "left":
            mono = L
        elif self.mode == "right":
            mono = R
        elif self.mode == "mid":
            mono = (L + R) * 0.70710678
        elif self.mode == "sum":
            mono = L + R
        else:
            mono = 0.5 * (L + R)

        if self.mode in ("sum", "average"):
            self._pan_gain.set_params(lin=_db_to_lin(self.pan_law_db))
            mono = self._pan_gain.process(mono[:, None], sr)[:, 0]

        if self.normalize:
            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            if peak > 1e-9:
                target_lin = _db_to_lin(self.norm_target_db)
                desired = target_lin / peak
                if desired > 1.0:
                    desired = min(desired, _db_to_lin(max(0.0, self.norm_max_boost_db)))
                self._norm_gain.set_params(lin=float(desired))
                mono = self._norm_gain.process(mono[:, None], sr)[:, 0]

        return _safe_audio(mono[:, None])


_EQ_TYPES = ["peak", "lowshelf", "highshelf", "lowpass", "highpass"]


def _read_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v) if v is not None else default


@register_filter(
    "parametriceq",
    help="Parametric EQ (4 bands + legacy eq.bandN.*). Params: b1_type,b1_freq,b1_gain_db,b1_q,b1_slope,b1_order ... b4_*",
    params=[
        ParamSpec("b1_type", kind="enum", default="peak", choices=_EQ_TYPES),
        ParamSpec("b1_freq", kind="float", default=120.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("b1_gain_db", kind="float", default=0.0, minimum=-24.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("b1_q", kind="float", default=1.0, minimum=0.1, maximum=24.0, step=0.05),
        ParamSpec("b1_slope", kind="float", default=1.0, minimum=0.1, maximum=4.0, step=0.05),
        ParamSpec("b1_order", kind="int", default=2, minimum=1, maximum=8, step=1),
        ParamSpec("b1_enable", kind="bool", default=True),
        ParamSpec("b2_type", kind="enum", default="peak", choices=_EQ_TYPES),
        ParamSpec("b2_freq", kind="float", default=600.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("b2_gain_db", kind="float", default=0.0, minimum=-24.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("b2_q", kind="float", default=1.0, minimum=0.1, maximum=24.0, step=0.05),
        ParamSpec("b2_slope", kind="float", default=1.0, minimum=0.1, maximum=4.0, step=0.05),
        ParamSpec("b2_order", kind="int", default=2, minimum=1, maximum=8, step=1),
        ParamSpec("b2_enable", kind="bool", default=True),
        ParamSpec("b3_type", kind="enum", default="peak", choices=_EQ_TYPES),
        ParamSpec("b3_freq", kind="float", default=3000.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("b3_gain_db", kind="float", default=0.0, minimum=-24.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("b3_q", kind="float", default=1.0, minimum=0.1, maximum=24.0, step=0.05),
        ParamSpec("b3_slope", kind="float", default=1.0, minimum=0.1, maximum=4.0, step=0.05),
        ParamSpec("b3_order", kind="int", default=2, minimum=1, maximum=8, step=1),
        ParamSpec("b3_enable", kind="bool", default=True),
        ParamSpec("b4_type", kind="enum", default="highshelf", choices=_EQ_TYPES),
        ParamSpec("b4_freq", kind="float", default=10000.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("b4_gain_db", kind="float", default=0.0, minimum=-24.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("b4_q", kind="float", default=1.0, minimum=0.1, maximum=24.0, step=0.05),
        ParamSpec("b4_slope", kind="float", default=1.0, minimum=0.1, maximum=4.0, step=0.05),
        ParamSpec("b4_order", kind="int", default=2, minimum=1, maximum=8, step=1),
        ParamSpec("b4_enable", kind="bool", default=True),
    ],
)
class ParametricEQ(_IIRSOS):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        super().__init__()
        self.params: Dict[str, Any] = dict(params or {})
        self.params.update(kwargs)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)
        self._initd = False

    def _band_sos(self, sr: int, idx: int) -> Optional[np.ndarray]:
        p = self.params
        prefix = f"b{idx}_"

        enabled = _read_bool(p.get(prefix + "enable", True), True)
        if not enabled:
            return None

        typ = str(p.get(prefix + "type", "peak")).lower()
        freq = float(p.get(prefix + "freq", [120, 600, 3000, 10000][idx - 1]))
        gain = float(p.get(prefix + "gain_db", 0.0))
        q = float(p.get(prefix + "q", 1.0))
        slope = float(p.get(prefix + "slope", 1.0))
        order = int(p.get(prefix + "order", 2))

        nyq = max(1.0, 0.5 * sr)

        if typ == "peak":
            return _iir_peaking_sos(sr, freq, q, gain)
        if typ in ("lowshelf", "low"):
            return _iir_shelf_sos(sr, freq, gain, shelf_type="low", slope=slope)
        if typ in ("highshelf", "high"):
            return _iir_shelf_sos(sr, freq, gain, shelf_type="high", slope=slope)
        if typ == "lowpass":
            wn = _clamp(freq / nyq, 1e-5, 0.9999)
            return signal.butter(int(max(1, min(8, order))), wn, btype="low", output="sos")
        if typ == "highpass":
            wn = _clamp(freq / nyq, 1e-5, 0.9999)
            return signal.butter(int(max(1, min(8, order))), wn, btype="high", output="sos")

        return None

    def _design(self, sr: int) -> np.ndarray:
        parts = []

        for i in range(1, 5):
            sos = self._band_sos(sr, i)
            if sos is not None:
                parts.append(sos)

        if not parts:
            return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        return np.vstack(parts)


@register_filter(
    "master",
    help="Master chain: low cut, multiband compression, limiter. Params: low_cut,x1,x2,order,attack_ms,release_ms,makeup_db,ceiling_db",
    params=[
        ParamSpec("low_cut", kind="float", default=30.0, minimum=10.0, maximum=300.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("x1", kind="float", default=200.0, minimum=40.0, maximum=2000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("x2", kind="float", default=4000.0, minimum=500.0, maximum=16000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("order", kind="int", default=2, minimum=1, maximum=8, step=1),
        ParamSpec("attack_ms", kind="float", default=5.0, minimum=0.1, maximum=200.0, step=0.1, unit="ms"),
        ParamSpec("release_ms", kind="float", default=60.0, minimum=1.0, maximum=2000.0, step=1.0, unit="ms"),
        ParamSpec("makeup_db", kind="float", default=1.0, minimum=-12.0, maximum=12.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("ceiling_db", kind="float", default=-1.0, minimum=-12.0, maximum=0.0, step=0.1, scale="db", unit="dB"),
    ],
)
class Master(AudioFilter):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        self.params: Dict[str, Any] = dict(params or {})
        self.params.update(kwargs)

        self._hpf = Highpass({"cutoff": float(self.params.get("low_cut", 30.0)), "order": int(self.params.get("order", 2))})
        self._comp = Compressor(
            {
                "threshold_db": float(self.params.get("threshold_db", -18.0)),
                "ratio": float(self.params.get("ratio", 2.0)),
                "attack_ms": float(self.params.get("attack_ms", 5.0)),
                "release_ms": float(self.params.get("release_ms", 60.0)),
                "makeup_db": float(self.params.get("makeup_db", 1.0)),
                "knee_db": 6.0,
                "stereo_link": True,
            }
        )
        self._lim = Limiter(
            {
                "ceiling_db": float(self.params.get("ceiling_db", -1.0)),
                "release_ms": float(self.params.get("limiter_release_ms", 80.0)),
            }
        )

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

        self._hpf.set_params(
            cutoff=float(self.params.get("low_cut", 30.0)),
            order=int(self.params.get("order", 2)),
        )
        self._comp.set_params(
            threshold_db=float(self.params.get("threshold_db", -18.0)),
            ratio=float(self.params.get("ratio", 2.0)),
            attack_ms=float(self.params.get("attack_ms", 5.0)),
            release_ms=float(self.params.get("release_ms", 60.0)),
            makeup_db=float(self.params.get("makeup_db", 1.0)),
        )
        self._lim.set_params(
            ceiling_db=float(self.params.get("ceiling_db", -1.0)),
            release_ms=float(self.params.get("limiter_release_ms", 80.0)),
        )

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        y = _ensure_2d(block)
        y = self._hpf.process(y, sr)
        y = self._comp.process(y, sr)
        y = self._lim.process(y, sr)
        return _safe_audio(y)