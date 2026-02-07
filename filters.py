# =============================
# filters.py (REWRITE)
# Stream-safe audio filter registry + GUI-friendly param specs + anti-crackle safeguards
# =============================
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal

# =============================================================================
# Param specs for GUI auto-generation
# =============================================================================

@dataclass(frozen=True)
class ParamSpec:
    """
    GUI-friendly parameter metadata.

    kind:
      - "float" | "int" | "bool" | "enum"
    scale:
      - "linear" | "log" | "db"
    """
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
    factory: Any  # class or callable
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
    """
    Base class for streamable audio filters.
      process(block, sr) -> (N,C) float32 [-1,1]
      flush() -> optional tail (T,C) float32
      set_params(**kwargs) -> update live params (no reset pops where possible)
    """
    def set_params(self, **kwargs: Any) -> None:
        # Default: no-op (GUI can rebuild + crossfade if needed)
        return

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError

    def flush(self) -> Optional[np.ndarray]:
        return None


# =============================================================================
# Utilities (anti-NaN, anti-pop, stable biquads)
# =============================================================================

_DEF_EPS = 1e-12

def _db_to_lin(db: float) -> float:
    return float(10 ** (float(db) / 20.0))

def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))

def _safe_norm_freq(sr: int, f0: float) -> float:
    # clamp f0 to (0, Nyquist)
    f0 = float(max(1e-3, min(float(f0), 0.5 * sr * 0.999)))
    return 2.0 * math.pi * (f0 / sr)

def _biquad_to_sos(b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> np.ndarray:
    # normalize a0 to 1
    if a0 == 0.0:
        a0 = 1e-12
    b0n, b1n, b2n = b0 / a0, b1 / a0, b2 / a0
    a1n, a2n = a1 / a0, a2 / a0
    return np.array([[b0n, b1n, b2n, 1.0, a1n, a2n]], dtype=np.float64)

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

def _iir_shelf_sos(sr: int, f0: float, gain_db: float, *, shelf_type: str = "low", slope: float = 1.0) -> np.ndarray:
    # RBJ shelf with guard against sqrt domain errors
    S = float(max(1e-6, slope))
    A = 10.0 ** (float(gain_db) / 40.0)
    w0 = _safe_norm_freq(sr, f0)
    cw = math.cos(w0)
    sw = math.sin(w0)

    # sqrt argument can go slightly negative due to numeric or S>1 usage; clamp to >=0
    sqrt_arg = (A + 1.0 / A) * (1.0 / S - 1.0) + 2.0
    sqrt_arg = max(0.0, float(sqrt_arg))
    alpha = sw / 2.0 * math.sqrt(sqrt_arg)

    st = shelf_type.lower()
    if st in ("low", "lowshelf", "ls"):
        b0 =    A * ((A + 1) - (A - 1) * cw + 2.0 * math.sqrt(A) * alpha)
        b1 =  2*A * ((A - 1) - (A + 1) * cw)
        b2 =    A * ((A + 1) - (A - 1) * cw - 2.0 * math.sqrt(A) * alpha)
        a0 =        (A + 1) + (A - 1) * cw + 2.0 * math.sqrt(A) * alpha
        a1 =   -2 * ((A - 1) + (A + 1) * cw)
        a2 =        (A + 1) + (A - 1) * cw - 2.0 * math.sqrt(A) * alpha
    elif st in ("high", "highshelf", "hs"):
        b0 =    A * ((A + 1) + (A - 1) * cw + 2.0 * math.sqrt(A) * alpha)
        b1 = -2*A * ((A - 1) + (A + 1) * cw)
        b2 =    A * ((A + 1) + (A - 1) * cw - 2.0 * math.sqrt(A) * alpha)
        a0 =        (A + 1) - (A - 1) * cw + 2.0 * math.sqrt(A) * alpha
        a1 =    2 * ((A - 1) - (A + 1) * cw)
        a2 =        (A + 1) - (A - 1) * cw - 2.0 * math.sqrt(A) * alpha
    else:
        raise ValueError(f"Unknown shelf_type '{shelf_type}' (use 'low' or 'high').")

    return _biquad_to_sos(b0, b1, b2, a0, a1, a2)

def _soft_clip(x: np.ndarray) -> np.ndarray:
    # very light safety; keep it predictable
    return np.tanh(x).astype(np.float32, copy=False)

def _ramp(start: float, end: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([end], dtype=np.float32)
    return np.linspace(start, end, n, dtype=np.float32)


# =============================================================================
# Builder (works with classes expecting dict OR kwargs)
# =============================================================================

def build_filter(name: str, **kwargs: Any) -> AudioFilter:
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown filter '{name}'. Available: {', '.join(available_filters().keys()) or '(none)'}")

    info = _REGISTRY[key]
    factory = info.factory

    # 1) If it's a class (subclass AudioFilter), support either __init__(params_dict) or __init__(**kwargs)
    if isinstance(factory, type):
        try:
            return factory(**kwargs)  # type: ignore
        except TypeError:
            return factory(kwargs)    # type: ignore

    # 2) If it's callable factory
    try:
        return factory(**kwargs)
    except TypeError:
        return factory(kwargs)


# =============================================================================
# A) Crackle prevention: smooth gain changes + stable IIR init
# =============================================================================

class SmoothGain(AudioFilter):
    """
    Gain wrapper: ramps gain changes over ~5ms to avoid clicks when GUI updates.
    """
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
        x = block.astype(np.float32, copy=False)
        n = x.shape[0]
        if n == 0:
            return x

        if self._target == self._g:
            y = x * self._g
            return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)

        ramp_n = int(max(1, (self._ramp_ms * 1e-3) * sr))
        ramp_n = min(ramp_n, n)
        g0 = float(self._g)
        g1 = float(self._target)

        gvec = _ramp(g0, g1, ramp_n)
        y = x.copy()
        y[:ramp_n] *= gvec[:, None]
        if ramp_n < n:
            y[ramp_n:] *= g1

        self._g = g1
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)


class _IIRSOS(AudioFilter):
    """
    SOS IIR with:
      - stable wn clamping
      - zi initialization using sosfilt_zi * x0 to avoid first-block pop
      - no NaNs / infs
    """
    def __init__(self):
        self.sos: Optional[np.ndarray] = None
        self.zi: Optional[np.ndarray] = None  # (S, C, 2)
        self._initd = False
        self._sr: Optional[int] = None
        self._C: Optional[int] = None
        self._primed = False

    def _design(self, sr: int) -> np.ndarray:
        raise NotImplementedError

    def _ensure(self, sr: int, C: int, x0: Optional[np.ndarray] = None):
        if self._initd and self._sr == sr and self._C == C:
            return
        sos = self._design(sr).astype(np.float64, copy=False)
        if sos.ndim != 2 or sos.shape[1] != 6:
            raise ValueError("IIR design must return SOS with shape (S,6).")
        self.sos = sos.astype(np.float32, copy=False)
        S = self.sos.shape[0]

        # zi prime: sosfilt_zi gives (S,2) for one channel
        zi0 = signal.sosfilt_zi(self.sos.astype(np.float64))
        zi = np.zeros((S, C, 2), dtype=np.float32)
        if x0 is not None and x0.size == C:
            # zi per channel scaled by first sample value
            for ch in range(C):
                zi[:, ch, :] = (zi0 * float(x0[ch])).astype(np.float32)
            self._primed = True
        else:
            self._primed = False

        self.zi = zi
        self._sr, self._C, self._initd = sr, C, True

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float32, copy=False)
        n, C = x.shape
        if n == 0:
            return x

        # prime on first use with first-sample to avoid pop
        if not self._initd:
            self._ensure(sr, C, x0=x[0])
        else:
            self._ensure(sr, C, x0=None)

        sos = self.sos
        zi = self.zi
        if sos is None or zi is None:
            return x

        y = np.empty_like(x, dtype=np.float32)
        for ch in range(C):
            y[:, ch], zi[:, ch, :] = signal.sosfilt(sos, x[:, ch], zi=zi[:, ch, :])

        self.zi = zi

        # safety
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return y


# =============================================================================
# Filters (all params declared for GUI auto-gen)
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
            db = float(p.get("db", 0.0))
            lin = _db_to_lin(db)

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
    """
    Streaming-friendly normalize:
      - tracks running max
      - ramps gain changes so it doesn't "tick" when max updates
    """
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
            # keep current gain, just change ramp time
            self._g._ramp_ms = self.ramp_ms  # intentional internal tweak

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float32, copy=False)
        m = float(np.max(np.abs(x))) if x.size else 0.0
        if m > self._max:
            self._max = m

        if self._max < _DEF_EPS:
            return x

        desired = float(self.target / self._max)
        self._g.set_params(lin=desired)
        y = self._g.process(x, sr)
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)


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
        order = int(max(1, min(12, self.order)))
        return signal.butter(order, wn, btype="low", output="sos")


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
        order = int(max(1, min(12, self.order)))
        return signal.butter(order, wn, btype="high", output="sos")


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
            self.low = float(kwargs["low"]); changed = True
        if "high" in kwargs and kwargs["high"] is not None:
            self.high = float(kwargs["high"]); changed = True
        if "order" in kwargs and kwargs["order"] is not None:
            self.order = int(kwargs["order"]); changed = True
        if changed:
            self._initd = False

    def _design(self, sr: int) -> np.ndarray:
        nyq = 0.5 * sr
        lo = float(self.low)
        hi = float(self.high)

        # enforce ordering + minimum bandwidth
        lo = max(10.0, lo)
        hi = max(lo + 10.0, hi)

        lo_n = _clamp(lo / nyq, 1e-5, 0.9998)
        hi_n = _clamp(hi / nyq, lo_n + 1e-5, 0.9999)

        order = int(max(1, min(12, self.order)))
        return signal.butter(order, [lo_n, hi_n], btype="band", output="sos")


@register_filter(
    "compress",
    help=("Smooth compressor. Params: threshold_db(-24) ratio(4) attack_ms(5) release_ms(50) "
          "makeup_db(0) knee_db(6) stereo_link(true)"),
    params=[
        ParamSpec("threshold_db", kind="float", default=-24.0, minimum=-60.0, maximum=0.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("ratio", kind="float", default=4.0, minimum=1.0, maximum=20.0, step=0.1, scale="linear"),
        ParamSpec("attack_ms", kind="float", default=5.0, minimum=0.1, maximum=200.0, step=0.1, unit="ms"),
        ParamSpec("release_ms", kind="float", default=50.0, minimum=1.0, maximum=2000.0, step=1.0, unit="ms"),
        ParamSpec("makeup_db", kind="float", default=0.0, minimum=-12.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("knee_db", kind="float", default=6.0, minimum=0.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("stereo_link", kind="bool", default=True),
    ],
)
class Compressor(AudioFilter):
    """
    Stream-safe compressor, tuned to avoid "crackle" on fast UI updates:
      - detector AR smoothing
      - GR AR smoothing
      - stereo link optional
      - all math guarded against NaN/inf
    """
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
        if isinstance(sl, str):
            self.stereo_link = sl.strip().lower() == "true"
        else:
            self.stereo_link = bool(sl)

        self._env: Optional[np.ndarray] = None
        self._gr_db: Optional[np.ndarray] = None
        self._sr: Optional[int] = None

        self._atk_a: float = 0.0
        self._rel_a: float = 0.0
        self._g_atk_a: float = 0.0
        self._g_rel_a: float = 0.0
        self._mk_lin: float = _db_to_lin(self.mk_db)

    @staticmethod
    def _alpha(ms: float, sr: int) -> float:
        sr = int(max(1, sr))
        ms = float(max(0.1, ms))
        return float(np.exp(-1.0 / (ms * 1e-3 * sr)))

    def set_params(self, **kwargs: Any) -> None:
        # live updates (no state wipe)
        if "threshold_db" in kwargs and kwargs["threshold_db"] is not None:
            self.th = float(kwargs["threshold_db"])
        if "ratio" in kwargs and kwargs["ratio"] is not None:
            self.ratio = max(1.0, float(kwargs["ratio"]))
        if "attack_ms" in kwargs and kwargs["attack_ms"] is not None:
            self.atk_ms = float(kwargs["attack_ms"])
            self._sr = None  # force coeff recalc
        if "release_ms" in kwargs and kwargs["release_ms"] is not None:
            self.rel_ms = float(kwargs["release_ms"])
            self._sr = None
        if "makeup_db" in kwargs and kwargs["makeup_db"] is not None:
            self.mk_db = float(kwargs["makeup_db"])
            self._mk_lin = _db_to_lin(self.mk_db)
        if "knee_db" in kwargs and kwargs["knee_db"] is not None:
            self.knee_db = float(kwargs["knee_db"])
        if "stereo_link" in kwargs and kwargs["stereo_link"] is not None:
            self.stereo_link = bool(kwargs["stereo_link"])

    def _ensure(self, C: int, sr: int):
        if self._sr != sr:
            self._sr = sr
            self._atk_a = self._alpha(self.atk_ms, sr)
            self._rel_a = self._alpha(self.rel_ms, sr)
            self._g_atk_a = self._alpha(max(1.0, 0.6 * self.atk_ms), sr)
            self._g_rel_a = self._alpha(self.rel_ms * 1.1, sr)

        if self._env is None or self._env.shape[0] != C:
            self._env = np.zeros((C,), dtype=np.float32)
        if self._gr_db is None or self._gr_db.shape[0] != C:
            self._gr_db = np.zeros((C,), dtype=np.float32)

    def _soft_knee_gr_db(self, level_db: np.ndarray) -> np.ndarray:
        T = float(self.th)
        R = float(self.ratio)
        K = float(max(0.0, self.knee_db))
        over = level_db - T

        if K <= 1e-6:
            gr = np.where(over > 0.0, -(1.0 - 1.0 / R) * over, 0.0)
            return gr.astype(np.float32)

        halfK = 0.5 * K
        knee_start = T - halfK
        knee_end = T + halfK

        below = level_db <= knee_start
        above = level_db >= knee_end
        inside = ~(below | above)

        gr = np.zeros_like(level_db, dtype=np.float32)
        gr[below] = 0.0
        gr[above] = (-(1.0 - 1.0 / R) * over[above]).astype(np.float32)

        knee_pos = (level_db[inside] - knee_start)
        gr[inside] = (-(1.0 - 1.0 / R) * (knee_pos ** 2) / (2.0 * K)).astype(np.float32)
        return gr

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float32, copy=False)
        n, C = x.shape
        if n == 0:
            return x

        self._ensure(C, sr)
        env = self._env
        gr_db = self._gr_db
        if env is None or gr_db is None:
            return x

        atk, rel = self._atk_a, self._rel_a
        gatk, grel = self._g_atk_a, self._g_rel_a
        mk = float(self._mk_lin)

        out = np.empty_like(x, dtype=np.float32)

        for i in range(n):
            s = x[i]
            abs_s = np.abs(s)

            faster = abs_s > env
            a = np.where(faster, atk, rel).astype(np.float32)
            env = a * env + (1.0 - a) * abs_s

            level_db = (20.0 * np.log10(np.maximum(env, _DEF_EPS))).astype(np.float32)
            inst_gr = self._soft_knee_gr_db(level_db)

            if self.stereo_link and C > 1:
                link_gr = float(np.min(inst_gr))
                inst_gr[:] = link_gr

            go_more = inst_gr < gr_db
            ag = np.where(go_more, gatk, grel).astype(np.float32)
            gr_db = ag * gr_db + (1.0 - ag) * inst_gr

            gain_lin = (10.0 ** (gr_db / 20.0)).astype(np.float32) * mk
            out[i] = s * gain_lin

        self._env = env
        self._gr_db = gr_db

        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        np.clip(out, -1.0, 1.0, out=out)
        return out


@register_filter(
    "limiter",
    help="Peak limiter. Params: ceiling_db(-1) release_ms(50)",
    params=[
        ParamSpec("ceiling_db", kind="float", default=-1.0, minimum=-18.0, maximum=0.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("release_ms", kind="float", default=50.0, minimum=1.0, maximum=2000.0, step=1.0, unit="ms"),
    ],
)
class Limiter(AudioFilter):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        p = dict(params or {})
        p.update(kwargs)
        self.ceil_db = float(p.get("ceiling_db", -1.0))
        self.rel_ms = float(p.get("release_ms", 50.0))

        self.env: Optional[np.ndarray] = None
        self._rel_alpha: float = 0.0
        self._ceil_lin: float = _db_to_lin(self.ceil_db)
        self._sr: Optional[int] = None

    def set_params(self, **kwargs: Any) -> None:
        if "ceiling_db" in kwargs and kwargs["ceiling_db"] is not None:
            self.ceil_db = float(kwargs["ceiling_db"])
            self._ceil_lin = _db_to_lin(self.ceil_db)
        if "release_ms" in kwargs and kwargs["release_ms"] is not None:
            self.rel_ms = float(kwargs["release_ms"])
            self._sr = None

    def _ensure(self, C: int, sr: int):
        if self.env is None or self.env.shape[0] != C:
            self.env = np.zeros((C,), dtype=np.float32)

        if self._sr != sr:
            self._sr = sr
            sr = int(max(1, sr))
            # stable release coefficient
            tau = max(1e-4, float(self.rel_ms) * 1e-3)
            self._rel_alpha = float(np.exp(-1.0 / (tau * sr)))

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float32, copy=False)
        n, C = x.shape
        if n == 0:
            return x

        self._ensure(C, sr)
        env = self.env
        if env is None:
            return x

        rel = float(self._rel_alpha)
        ceil_lin = float(self._ceil_lin)

        out = np.empty_like(x, dtype=np.float32)

        for i in range(n):
            env *= rel
            peak = float(np.max(np.abs(x[i])))
            needed = peak / max(ceil_lin, _DEF_EPS)
            if needed > 1.0:
                np.maximum(env, needed, out=env)

            gain = 1.0 / np.maximum(env, 1.0)
            out[i] = x[i] * gain

        np.clip(out, -ceil_lin, ceil_lin, out=out)
        return out


@register_filter(
    "mixdown",
    help=(
        "Stereo→Mono mixdown. "
        "Params: mode(average|sum|mid|left|right), "
        "pan_law_db(-3), normalize(false), norm_target_db(-1), norm_max_boost_db(6), ramp_ms(5)"
    ),
    params=[
        ParamSpec(
            "mode",
            kind="enum",
            default="average",
            choices=["average", "sum", "mid", "left", "right"],
        ),
        ParamSpec(
            "pan_law_db",
            kind="float",
            default=-3.0,
            minimum=-6.0,
            maximum=0.0,
            step=0.5,
            scale="db",
            unit="dB",
        ),
        ParamSpec("normalize", kind="bool", default=False),
        ParamSpec(
            "norm_target_db",
            kind="float",
            default=-1.0,
            minimum=-12.0,
            maximum=-0.1,
            step=0.1,
            scale="db",
            unit="dB",
        ),
        ParamSpec(
            "norm_max_boost_db",
            kind="float",
            default=6.0,
            minimum=0.0,
            maximum=24.0,
            step=0.5,
            scale="db",
            unit="dB",
        ),
        ParamSpec(
            "ramp_ms",
            kind="float",
            default=5.0,
            minimum=0.1,
            maximum=50.0,
            step=0.1,
            unit="ms",
        ),
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

        # Smooth gain stages (click-free)
        self._pan_gain = SmoothGain(1.0, ramp_ms=self.ramp_ms)
        self._norm_gain = SmoothGain(1.0, ramp_ms=self.ramp_ms)

    def set_params(self, **kwargs):
        if "mode" in kwargs and kwargs["mode"] is not None:
            self.mode = str(kwargs["mode"]).lower()
        if "pan_law_db" in kwargs and kwargs["pan_law_db"] is not None:
            self.pan_law_db = float(kwargs["pan_law_db"])

        if "normalize" in kwargs and kwargs["normalize"] is not None:
            self.normalize = bool(kwargs["normalize"])
        if "norm_target_db" in kwargs and kwargs["norm_target_db"] is not None:
            self.norm_target_db = float(kwargs["norm_target_db"])
        if "norm_max_boost_db" in kwargs and kwargs["norm_max_boost_db"] is not None:
            self.norm_max_boost_db = float(kwargs["norm_max_boost_db"])

        if "ramp_ms" in kwargs and kwargs["ramp_ms"] is not None:
            self.ramp_ms = float(kwargs["ramp_ms"])
            self._pan_gain._ramp_ms = self.ramp_ms
            self._norm_gain._ramp_ms = self.ramp_ms

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float32, copy=False)

        # Already mono → passthrough
        if x.shape[1] == 1:
            return x

        L = x[:, 0]
        R = x[:, 1]

        if self.mode == "left":
            mono = L
        elif self.mode == "right":
            mono = R
        elif self.mode == "mid":
            mono = (L + R) * 0.70710678  # 1/sqrt(2)
        elif self.mode == "sum":
            mono = L + R
        else:  # average
            mono = 0.5 * (L + R)

        # Pan-law compensation (sum/average only)
        if self.mode in ("sum", "average"):
            pan_gain = 10.0 ** (self.pan_law_db / 20.0)
            self._pan_gain.set_params(lin=pan_gain)
            mono = self._pan_gain.process(mono[:, None], sr)[:, 0]

        # SAFE normalization:
        # - normalize to a target peak (e.g. -1 dBFS)
        # - limit maximum boost so it can’t explode loudness
        if self.normalize:
            peak = float(np.max(np.abs(mono)))
            if peak > 1e-9:
                target_lin = 10.0 ** (self.norm_target_db / 20.0)
                desired = target_lin / peak

                # clamp boost (only clamp *upwards*, never force attenuation)
                max_boost_lin = 10.0 ** (max(0.0, self.norm_max_boost_db) / 20.0)
                if desired > 1.0:
                    desired = min(desired, max_boost_lin)

                self._norm_gain.set_params(lin=float(desired))
                mono = self._norm_gain.process(mono[:, None], sr)[:, 0]

        mono = np.clip(mono, -1.0, 1.0)
        return mono[:, None].astype(np.float32, copy=False)

# =============================================================================
# ParametricEQ (GUI-friendly fixed bands + backwards-compatible eq.bandN.*)
# =============================================================================

_EQ_TYPES = ["peak", "lowshelf", "highshelf", "lowpass", "highpass"]

def _read_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v) if v is not None else default

@register_filter(
    "parametriceq",
    help=("Parametric EQ (4 bands + legacy eq.bandN.*). "
          "Params: b1_type,b1_freq,b1_gain_db,b1_q,b1_slope,b1_order ... b4_*"),
    params=[
        # Band 1
        ParamSpec("b1_type", kind="enum", default="peak", choices=_EQ_TYPES),
        ParamSpec("b1_freq", kind="float", default=120.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("b1_gain_db", kind="float", default=0.0, minimum=-24.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("b1_q", kind="float", default=1.0, minimum=0.1, maximum=24.0, step=0.05),
        ParamSpec("b1_slope", kind="float", default=1.0, minimum=0.1, maximum=4.0, step=0.05),
        ParamSpec("b1_order", kind="int", default=2, minimum=1, maximum=8, step=1),
        ParamSpec("b1_enable", kind="bool", default=True),
        # Band 2
        ParamSpec("b2_type", kind="enum", default="peak", choices=_EQ_TYPES),
        ParamSpec("b2_freq", kind="float", default=600.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("b2_gain_db", kind="float", default=0.0, minimum=-24.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("b2_q", kind="float", default=1.0, minimum=0.1, maximum=24.0, step=0.05),
        ParamSpec("b2_slope", kind="float", default=1.0, minimum=0.1, maximum=4.0, step=0.05),
        ParamSpec("b2_order", kind="int", default=2, minimum=1, maximum=8, step=1),
        ParamSpec("b2_enable", kind="bool", default=True),
        # Band 3
        ParamSpec("b3_type", kind="enum", default="peak", choices=_EQ_TYPES),
        ParamSpec("b3_freq", kind="float", default=2500.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("b3_gain_db", kind="float", default=0.0, minimum=-24.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("b3_q", kind="float", default=1.0, minimum=0.1, maximum=24.0, step=0.05),
        ParamSpec("b3_slope", kind="float", default=1.0, minimum=0.1, maximum=4.0, step=0.05),
        ParamSpec("b3_order", kind="int", default=2, minimum=1, maximum=8, step=1),
        ParamSpec("b3_enable", kind="bool", default=True),
        # Band 4
        ParamSpec("b4_type", kind="enum", default="highshelf", choices=_EQ_TYPES),
        ParamSpec("b4_freq", kind="float", default=9000.0, minimum=20.0, maximum=24000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("b4_gain_db", kind="float", default=0.0, minimum=-24.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("b4_q", kind="float", default=1.0, minimum=0.1, maximum=24.0, step=0.05),
        ParamSpec("b4_slope", kind="float", default=1.0, minimum=0.1, maximum=4.0, step=0.05),
        ParamSpec("b4_order", kind="int", default=2, minimum=1, maximum=8, step=1),
        ParamSpec("b4_enable", kind="bool", default=True),
    ],
)
class ParametricEQ(AudioFilter):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        self._params: Dict[str, Any] = dict(params or {})
        self._params.update(kwargs)

        self._sr: Optional[int] = None
        self._C: Optional[int] = None
        self._built = False

        self._bands_sos: List[np.ndarray] = []
        self._bands_zi: List[np.ndarray] = []

        # legacy eq.bandN.* support
        self._legacy_defs: List[Dict[str, Any]] = []
        self._parse_legacy_bands()

    def set_params(self, **kwargs: Any) -> None:
        # update without wiping state unless design needs rebuild
        changed = False
        for k, v in kwargs.items():
            if self._params.get(k) != v:
                self._params[k] = v
                changed = True
        if changed:
            self._built = False

    def _parse_legacy_bands(self):
        self._legacy_defs = []
        pat = re.compile(r"eq\.band(\d+)\.(type|freq|gain_db|q|slope|order|enable)$")
        tmp: Dict[int, Dict[str, Any]] = {}
        for k, v in self._params.items():
            m = pat.match(str(k))
            if not m:
                continue
            bi = int(m.group(1))
            pn = m.group(2)
            tmp.setdefault(bi, {})[pn] = v

        for bi in sorted(tmp.keys()):
            d = tmp[bi]
            if "type" not in d or "freq" not in d:
                continue
            d.setdefault("q", 1.0)
            d.setdefault("slope", 1.0)
            d.setdefault("order", 2)
            d.setdefault("enable", True)
            self._legacy_defs.append(d)

    def _fixed_band_defs(self) -> List[Dict[str, Any]]:
        defs: List[Dict[str, Any]] = []
        for i in range(1, 5):
            enable = _read_bool(self._params.get(f"b{i}_enable", True), True)
            if not enable:
                continue
            defs.append({
                "type": str(self._params.get(f"b{i}_type", "peak")).lower(),
                "freq": float(self._params.get(f"b{i}_freq", 1000.0)),
                "gain_db": float(self._params.get(f"b{i}_gain_db", 0.0)),
                "q": float(self._params.get(f"b{i}_q", 1.0)),
                "slope": float(self._params.get(f"b{i}_slope", 1.0)),
                "order": int(self._params.get(f"b{i}_order", 2)),
            })
        return defs

    def _build(self, sr: int, C: int):
        self._bands_sos = []
        self._bands_zi = []
        self._sr, self._C = sr, C

        # prefer legacy if present; else fixed 4-band GUI controls
        self._parse_legacy_bands()
        band_defs = self._legacy_defs if self._legacy_defs else self._fixed_band_defs()

        for bd in band_defs:
            b_type = str(bd.get("type", "peak")).lower()
            b_freq = float(bd.get("freq", 1000.0))
            b_gain = float(bd.get("gain_db", 0.0))
            b_q = float(bd.get("q", 1.0))
            b_slope = float(bd.get("slope", 1.0))
            b_order = int(bd.get("order", 2))

            sos: Optional[np.ndarray] = None

            if b_type == "peak":
                sos = _iir_peaking_sos(sr, b_freq, b_q, b_gain)
            elif b_type == "lowshelf":
                sos = _iir_shelf_sos(sr, b_freq, b_gain, shelf_type="low", slope=b_slope)
            elif b_type == "highshelf":
                sos = _iir_shelf_sos(sr, b_freq, b_gain, shelf_type="high", slope=b_slope)
            elif b_type == "lowpass":
                wn = _clamp(b_freq / (0.5 * sr), 1e-5, 0.9999)
                sos = signal.butter(int(max(1, min(8, b_order))), wn, btype="low", output="sos")
            elif b_type == "highpass":
                wn = _clamp(b_freq / (0.5 * sr), 1e-5, 0.9999)
                sos = signal.butter(int(max(1, min(8, b_order))), wn, btype="high", output="sos")

            if sos is None or sos.ndim != 2 or sos.shape[1] != 6:
                continue

            sos64 = sos.astype(np.float64, copy=False)
            zi_one = signal.sosfilt_zi(sos64)  # (S,2)
            zi = np.tile(zi_one[..., np.newaxis], (1, 1, C)).astype(np.float64)  # (S,2,C)

            self._bands_sos.append(sos64)
            self._bands_zi.append(zi)

        self._built = bool(self._bands_sos)

    def _ensure(self, sr: int, C: int):
        if (not self._built) or (self._sr != sr) or (self._C != C):
            self._build(sr, C)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float64, copy=False)
        n, C = x.shape
        if n == 0:
            return block.astype(np.float32, copy=False)

        self._ensure(sr, C)
        if not self._built:
            return block.astype(np.float32, copy=False)

        y = x
        for i in range(len(self._bands_sos)):
            sos = self._bands_sos[i]
            zi = self._bands_zi[i]
            # zi expected (S,2,C) for axis=0 filtering on (N,C)
            y, self._bands_zi[i] = signal.sosfilt(sos, y, axis=0, zi=zi)

        y = y.astype(np.float32, copy=False)
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)


# =============================================================================
# Master (declares ALL params so your GUI can build them reliably)
# =============================================================================

@register_filter(
    "master",
    help=("Mastering: HPF + 3-band comp + limiter. "
          "Params: low_cut(30) x1(200) x2(4000) order(2) "
          "low_th(-24) low_ratio(2) mid_th(-20) mid_ratio(2.5) high_th(-22) high_ratio(2) "
          "attack_ms(5) release_ms(60) makeup_db(1) ceiling_db(-1)"),
    params=[
        ParamSpec("low_cut", kind="float", default=30.0, minimum=10.0, maximum=400.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("x1", kind="float", default=200.0, minimum=50.0, maximum=2000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("x2", kind="float", default=4000.0, minimum=1000.0, maximum=18000.0, step=1.0, scale="log", unit="Hz"),
        ParamSpec("order", kind="int", default=2, minimum=1, maximum=8, step=1),

        ParamSpec("low_th", kind="float", default=-24.0, minimum=-60.0, maximum=0.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("low_ratio", kind="float", default=2.0, minimum=1.0, maximum=10.0, step=0.1),

        ParamSpec("mid_th", kind="float", default=-20.0, minimum=-60.0, maximum=0.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("mid_ratio", kind="float", default=2.5, minimum=1.0, maximum=10.0, step=0.1),

        ParamSpec("high_th", kind="float", default=-22.0, minimum=-60.0, maximum=0.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("high_ratio", kind="float", default=2.0, minimum=1.0, maximum=10.0, step=0.1),

        ParamSpec("attack_ms", kind="float", default=5.0, minimum=0.1, maximum=200.0, step=0.1, unit="ms"),
        ParamSpec("release_ms", kind="float", default=60.0, minimum=1.0, maximum=2000.0, step=1.0, unit="ms"),
        ParamSpec("makeup_db", kind="float", default=1.0, minimum=-12.0, maximum=24.0, step=0.5, scale="db", unit="dB"),
        ParamSpec("ceiling_db", kind="float", default=-1.0, minimum=-18.0, maximum=0.0, step=0.5, scale="db", unit="dB"),
    ],
)
class Master(AudioFilter):
    def __init__(self, params: Union[Dict[str, Any], None] = None, **kwargs: Any):
        self.params: Dict[str, Any] = dict(params or {})
        self.params.update(kwargs)

        self._initd = False
        self._sr: Optional[int] = None
        self._C: Optional[int] = None

        self._hpf: Optional[Highpass] = None
        self._lp: Optional[Lowpass] = None
        self._bp: Optional[Bandpass] = None
        self._hp: Optional[Highpass] = None

        self._cL: Optional[Compressor] = None
        self._cM: Optional[Compressor] = None
        self._cH: Optional[Compressor] = None
        self._cl: Optional[Limiter] = None

        self._mk: SmoothGain = SmoothGain(1.0, ramp_ms=10.0)
        self._mix_buf: Optional[np.ndarray] = None

    def set_params(self, **kwargs: Any) -> None:
        # update stored params; force rebuild next process to avoid partial rewire pops
        for k, v in kwargs.items():
            self.params[k] = v
        self._initd = False

    def _ensure(self, sr: int, C: int):
        if self._initd and self._sr == sr and self._C == C:
            return

        self._sr, self._C = sr, C

        low_cut = float(self.params.get("low_cut", 30.0))
        x1 = float(self.params.get("x1", 200.0))
        x2 = float(self.params.get("x2", 4000.0))
        order = int(self.params.get("order", 2))

        atk = float(self.params.get("attack_ms", 5.0))
        rel = float(self.params.get("release_ms", 60.0))
        makeup_db = float(self.params.get("makeup_db", 1.0))
        ceiling_db = float(self.params.get("ceiling_db", -1.0))

        low_th = float(self.params.get("low_th", -24.0))
        mid_th = float(self.params.get("mid_th", -20.0))
        high_th = float(self.params.get("high_th", -22.0))
        low_ratio = float(self.params.get("low_ratio", 2.0))
        mid_ratio = float(self.params.get("mid_ratio", 2.5))
        high_ratio = float(self.params.get("high_ratio", 2.0))

        # filters
        self._hpf = Highpass({"cutoff": max(10.0, low_cut), "order": 2})
        self._lp = Lowpass({"cutoff": x1, "order": order})
        self._bp = Bandpass({"low": x1, "high": x2, "order": order})
        self._hp = Highpass({"cutoff": x2, "order": order})

        shared = {"attack_ms": atk, "release_ms": rel, "makeup_db": 0.0, "knee_db": 6.0, "stereo_link": True}
        self._cL = Compressor({"threshold_db": low_th, "ratio": low_ratio, **shared})
        self._cM = Compressor({"threshold_db": mid_th, "ratio": mid_ratio, **shared})
        self._cH = Compressor({"threshold_db": high_th, "ratio": high_ratio, **shared})
        self._cl = Limiter({"ceiling_db": ceiling_db, "release_ms": 80.0})

        self._mk.set_params(db=makeup_db)

        self._mix_buf = None
        self._initd = True

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float32, copy=False)
        if x.shape[0] == 0:
            return x

        n, C = x.shape
        self._ensure(sr, C)

        if self._mix_buf is None or self._mix_buf.shape != (n, C):
            self._mix_buf = np.empty((n, C), dtype=np.float32)

        assert self._hpf and self._lp and self._bp and self._hp and self._cL and self._cM and self._cH and self._cl

        y = self._hpf.process(x, sr)
        low = self._lp.process(y, sr)
        mid = self._bp.process(y, sr)
        high = self._hp.process(y, sr)

        low = self._cL.process(low, sr)
        mid = self._cM.process(mid, sr)
        high = self._cH.process(high, sr)

        np.add(low, mid, out=self._mix_buf)
        self._mix_buf += high

        # smooth makeup
        mk_applied = self._mk.process(self._mix_buf, sr)
        out = self._cl.process(mk_applied, sr)

        # tiny soft-clip to kill edge crackles when users slam params
        out = _soft_clip(out)
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)
