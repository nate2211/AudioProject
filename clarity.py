# clarity.py
# -------------------------------------------------------------
# Native-backed clarity-focused blocks for the audio engine
# -------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from filters import (
    AudioFilter,
    register_filter,
    build_filter,
    _DEF_EPS as _EPS,
    _db_to_lin,
    _safe_audio,
    _ensure_2d,
    Compressor,
    Limiter,
)


try:
    from native import (
        is_available as _native_is_available,
        NativeSOS as _NativeSOS,
        NativeSoftClipper as _NativeSoftClipper,
    )

    _NATIVE_OK = bool(_native_is_available())
except Exception:
    _NATIVE_OK = False
    _NativeSOS = None
    _NativeSoftClipper = None


_F32 = np.float32


def _finite_f(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return float(v) if np.isfinite(v) else float(default)


def _finite_i(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_sr(sr: int) -> float:
    s = _finite_f(sr, 48000.0)
    return float(max(8000.0, s))


def _nyq_safe(sr_f: float) -> float:
    return max(100.0, (0.5 * sr_f) * 0.999)


def _identity_sos() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)


def _valid_sos(sos: Any) -> bool:
    if sos is None:
        return False
    sos = np.asarray(sos)
    return sos.ndim == 2 and sos.shape[1] == 6 and bool(np.all(np.isfinite(sos)))


def _safe_band(sr_f: float, fl: Any, fh: Any, *, min_bw_hz: float = 10.0) -> tuple[float, float]:
    lo = _finite_f(fl, 200.0)
    hi = _finite_f(fh, 2000.0)
    nyq = _nyq_safe(sr_f)

    lo = float(np.clip(lo, 10.0, nyq))
    hi = float(np.clip(hi, 20.0, nyq))

    if hi <= lo:
        hi = min(nyq, lo + 200.0)
    if (hi - lo) < float(min_bw_hz):
        hi = min(nyq, lo + float(min_bw_hz))
    if not (hi > lo):
        lo = 200.0
        hi = min(nyq, 2000.0)

    return lo, hi


def _safe_tf2sos(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    try:
        sos = signal.tf2sos(b, a)
        return sos if _valid_sos(sos) else _identity_sos()
    except Exception:
        return _identity_sos()


def _butter_sos(kind: str, sr: int, **kwargs) -> np.ndarray:
    try:
        sr_f = _safe_sr(sr)
        nyq = _nyq_safe(sr_f)
        kind = (kind or "").strip().lower()
        order = max(1, _finite_i(kwargs.get("order", 4), 4))

        if kind == "low":
            fc = float(np.clip(_finite_f(kwargs.get("fc", 8000.0), 8000.0), 20.0, nyq))
            return signal.butter(order, fc, btype="lowpass", fs=sr_f, output="sos")

        if kind == "high":
            fc = float(np.clip(_finite_f(kwargs.get("fc", 80.0), 80.0), 10.0, nyq))
            return signal.butter(order, fc, btype="highpass", fs=sr_f, output="sos")

        if kind == "band":
            fl, fh = _safe_band(sr_f, kwargs.get("fl", 200.0), kwargs.get("fh", 2000.0), min_bw_hz=10.0)
            return signal.butter(order, [fl, fh], btype="bandpass", fs=sr_f, output="sos")

        return _identity_sos()

    except Exception:
        return _identity_sos()


def _iir_peaking(sr: int, f0: float, q: float, gain_db: float) -> np.ndarray:
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

        b = np.array([b0, b1, b2], dtype=np.float64) / a0
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
        slope = float(np.clip(_finite_f(slope, 0.707), 0.10, 1.50))

        A = 10 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * (f0 / sr_f)
        cosw = np.cos(w0)
        sinw = np.sin(w0)

        rad = (A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0
        rad = max(0.0, float(rad if np.isfinite(rad) else 0.0))
        alpha = (sinw / 2.0) * np.sqrt(rad)

        st = (shelf_type or "high").strip().lower()

        if st == "low":
            b0 = A * ((A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
            b2 = A * ((A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cosw)
            a2 = (A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha
        else:
            b0 = A * ((A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
            b2 = A * ((A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cosw)
            a2 = (A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha

        if not np.isfinite(a0) or abs(a0) < 1e-12:
            return _identity_sos()

        b = np.array([b0, b1, b2], dtype=np.float64) / a0
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return _safe_tf2sos(b, a)

    except Exception:
        return _identity_sos()


class _NativeSOSBlock:
    def __init__(self):
        self.sos: Optional[np.ndarray] = None
        self.zi: Optional[np.ndarray] = None
        self._native = None
        self._C: Optional[int] = None

    def set_sos(self, sos: np.ndarray, C: int):
        sos = np.asarray(sos, dtype=np.float32)

        if not _valid_sos(sos):
            sos = _identity_sos().astype(np.float32)

        if self.sos is not None and self._C == C and self.sos.shape == sos.shape and np.allclose(self.sos, sos):
            return

        self.sos = np.ascontiguousarray(sos, dtype=np.float32)
        self.zi = None
        self._native = None
        self._C = C

        if _NATIVE_OK and _NativeSOS is not None:
            try:
                self._native = _NativeSOS(self.sos, channels=C, reset_state=True)
                return
            except Exception:
                self._native = None

        zi = signal.sosfilt_zi(self.sos.astype(np.float64)).astype(np.float32)
        self.zi = np.tile(zi[:, None, :], (1, C, 1)).astype(np.float32)

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(x)

        if self.sos is None:
            return x

        if self._native is not None:
            try:
                return _safe_audio(self._native.process(x, sr))
            except Exception:
                self._native = None
                self.zi = None

        if self.zi is None or self.zi.shape[1] != x.shape[1] or self.zi.shape[0] != self.sos.shape[0]:
            zi = signal.sosfilt_zi(self.sos.astype(np.float64)).astype(np.float32)
            self.zi = np.tile(zi[:, None, :], (1, x.shape[1], 1)).astype(np.float32)

        y = np.empty_like(x, dtype=np.float32)

        for c in range(x.shape[1]):
            y[:, c], self.zi[:, c, :] = signal.sosfilt(
                self.sos.astype(np.float64),
                x[:, c],
                zi=self.zi[:, c, :],
            )

        return _safe_audio(y)


def _soft_clip(x: np.ndarray, thresh: float = 1.0) -> np.ndarray:
    x = _ensure_2d(x)

    if thresh <= 0:
        thresh = 1.0

    if _NATIVE_OK and _NativeSoftClipper is not None:
        try:
            return _safe_audio(_NativeSoftClipper(drive=1.0 / max(1e-6, thresh)).process(x, 48000), ceiling=thresh)
        except Exception:
            pass

    k = 1.0 / max(1e-6, thresh)
    y = np.tanh(k * x) / k
    return _safe_audio(y, ceiling=thresh)


def _rms(x: NDArray[_F32]) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32) + _EPS))


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

        self._band = _NativeSOSBlock()
        self._env: Optional[np.ndarray] = None
        self._last_sr: Optional[int] = None
        self._last_C: Optional[int] = None

    def _ensure(self, sr: int, C: int):
        if self._env is None or self._env.shape[0] != C:
            self._env = np.ones((C,), dtype=np.float32)

        if self._last_sr != int(sr) or self._last_C != C:
            sr_f = _safe_sr(sr)
            nyq = _nyq_safe(sr_f)
            f0 = float(np.clip(self.f0, 50.0, nyq))
            q = float(np.clip(self.q, 0.1, 24.0))
            bw = max(200.0, f0 / max(1.0, q))
            fl, fh = _safe_band(sr_f, f0 - bw / 2.0, f0 + bw / 2.0, min_bw_hz=50.0)
            self._band.set_sos(_butter_sos("band", sr, fl=fl, fh=fh, order=4), C)
            self._last_sr = int(sr)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        C = x.shape[1]
        self._ensure(sr, C)

        band = self._band.process(x, sr)

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
            if target < env:
                env = atk * env + (1.0 - atk) * target
            else:
                env = rel * env + (1.0 - rel) * target
            env = float(np.clip(env, 0.0, 1.0))
            self._env[c] = env
            g_lin[c] = env

        y = x + ((band * g_lin[None, :]) - band)

        if float(self.mk) != 0.0:
            y *= _db_to_lin(float(self.mk))

        return _safe_audio(y)


@register_filter("presence", help="Presence peak EQ. Params: freq(3000), q(1.0), gain_db(2.0)")
class PresenceEQ(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.freq = _finite_f(params.get("freq", 3000.0), 3000.0)
        self.q = _finite_f(params.get("q", 1.0), 1.0)
        self.g = _finite_f(params.get("gain_db", 2.0), 2.0)
        self._sos = _NativeSOSBlock()
        self._last_sr = None
        self._last_C = None

    def _ensure(self, sr: int, C: int):
        if self._last_sr != int(sr) or self._last_C != C:
            self._sos.set_sos(_iir_peaking(sr, self.freq, max(0.1, self.q), self.g), C)
            self._last_sr = int(sr)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        self._ensure(sr, x.shape[1])
        return self._sos.process(x, sr)


@register_filter("air", help="High-shelf 'air'. Params: freq(10000), gain_db(2.0), slope(0.8)")
class AirShelf(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.freq = _finite_f(params.get("freq", 10000.0), 10000.0)
        self.g = _finite_f(params.get("gain_db", 2.0), 2.0)
        self.slope = _finite_f(params.get("slope", 0.8), 0.8)
        self._sos = _NativeSOSBlock()
        self._last_sr = None
        self._last_C = None

    def _ensure(self, sr: int, C: int):
        if self._last_sr != int(sr) or self._last_C != C:
            self._sos.set_sos(_iir_shelf(sr, self.freq, self.g, "high", self.slope), C)
            self._last_sr = int(sr)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        self._ensure(sr, x.shape[1])
        return self._sos.process(x, sr)


@register_filter("tilt", help="Tilt EQ. Params: pivot(1000), tilt_db(2.0)")
class TiltEQ(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.pivot = _finite_f(params.get("pivot", 1000.0), 1000.0)
        self.tilt_db = _finite_f(params.get("tilt_db", 2.0), 2.0)
        self._lo = _NativeSOSBlock()
        self._hi = _NativeSOSBlock()
        self._last_sr = None
        self._last_C = None

    def _ensure(self, sr: int, C: int):
        if self._last_sr != int(sr) or self._last_C != C:
            self._lo.set_sos(_iir_shelf(sr, self.pivot, -self.tilt_db, "low", 0.8), C)
            self._hi.set_sos(_iir_shelf(sr, self.pivot, self.tilt_db, "high", 0.8), C)
            self._last_sr = int(sr)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        self._ensure(sr, x.shape[1])
        y = self._lo.process(x, sr)
        y = self._hi.process(y, sr)
        return _safe_audio(y)


@register_filter("transient", help="Transient shaper. Params: attack_db(2), sustain_db(0), fast_ms(5), slow_ms(80)")
class TransientShaper(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.attack_db = _finite_f(params.get("attack_db", 2.0), 2.0)
        self.sustain_db = _finite_f(params.get("sustain_db", 0.0), 0.0)
        self.fast_ms = _finite_f(params.get("fast_ms", 5.0), 5.0)
        self.slow_ms = _finite_f(params.get("slow_ms", 80.0), 80.0)
        self._fast: Optional[np.ndarray] = None
        self._slow: Optional[np.ndarray] = None

    def _coef(self, ms: float, sr: int) -> float:
        return float(np.exp(-1.0 / (max(0.1, ms) * 1e-3 * max(1, sr))))

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        C = x.shape[1]

        if self._fast is None or self._fast.shape[0] != C:
            self._fast = np.zeros((C,), dtype=np.float32)
            self._slow = np.zeros((C,), dtype=np.float32)

        af = self._coef(self.fast_ms, sr)
        aslow = self._coef(self.slow_ms, sr)

        atk_gain = _db_to_lin(self.attack_db)
        sus_gain = _db_to_lin(self.sustain_db)

        y = np.empty_like(x)

        for i in range(x.shape[0]):
            a = np.abs(x[i])
            self._fast = af * self._fast + (1.0 - af) * a
            self._slow = aslow * self._slow + (1.0 - aslow) * a
            trans = np.maximum(0.0, self._fast - self._slow)
            sustain = np.maximum(0.0, self._slow - self._fast)
            gain = 1.0 + trans * (atk_gain - 1.0) + sustain * (sus_gain - 1.0)
            y[i] = x[i] * gain

        return _soft_clip(y, 1.0).astype(np.float32)


@register_filter("imagerfocus", help="Stereo focus/imager. Params: width(1.0), mono_bass_hz(120)")
class ImagerFocus(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.width = _finite_f(params.get("width", 1.0), 1.0)
        self.mono_bass_hz = _finite_f(params.get("mono_bass_hz", 120.0), 120.0)
        self._low = _NativeSOSBlock()
        self._last_sr = None
        self._last_C = None

    def _ensure(self, sr: int, C: int):
        if C >= 2 and (self._last_sr != int(sr) or self._last_C != C):
            self._low.set_sos(_butter_sos("low", sr, fc=self.mono_bass_hz, order=2), C)
            self._last_sr = int(sr)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)

        if x.shape[1] < 2:
            return x

        self._ensure(sr, x.shape[1])

        mid = 0.5 * (x[:, 0] + x[:, 1])
        side = 0.5 * (x[:, 0] - x[:, 1]) * float(self.width)

        y = x.copy()
        y[:, 0] = mid + side
        y[:, 1] = mid - side

        low = self._low.process(y, sr)
        low_mid = 0.5 * (low[:, 0] + low[:, 1])
        y[:, 0] = y[:, 0] - low[:, 0] + low_mid
        y[:, 1] = y[:, 1] - low[:, 1] + low_mid

        return _safe_audio(y)


@register_filter("demud", help="Low-mid de-mud cut. Params: freq(300), q(1.0), gain_db(-3)")
class DeMud(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.freq = _finite_f(params.get("freq", 300.0), 300.0)
        self.q = _finite_f(params.get("q", 1.0), 1.0)
        self.gain_db = _finite_f(params.get("gain_db", -3.0), -3.0)
        self._sos = _NativeSOSBlock()
        self._last_sr = None
        self._last_C = None

    def _ensure(self, sr: int, C: int):
        if self._last_sr != int(sr) or self._last_C != C:
            self._sos.set_sos(_iir_peaking(sr, self.freq, self.q, self.gain_db), C)
            self._last_sr = int(sr)
            self._last_C = C

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        self._ensure(sr, x.shape[1])
        return self._sos.process(x, sr)


@register_filter("claritychain", help="Clarity chain: demud -> presence -> air -> deesser -> transient -> limiter")
class ClarityChain(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.demud = DeMud(
            {
                "freq": params.get("mud_freq", 300.0),
                "q": params.get("mud_q", 1.0),
                "gain_db": params.get("mud_gain_db", -2.5),
            }
        )
        self.presence = PresenceEQ(
            {
                "freq": params.get("presence_freq", 3000.0),
                "q": params.get("presence_q", 1.0),
                "gain_db": params.get("presence_gain_db", 2.0),
            }
        )
        self.air = AirShelf(
            {
                "freq": params.get("air_freq", 10000.0),
                "gain_db": params.get("air_gain_db", 1.5),
                "slope": params.get("air_slope", 0.8),
            }
        )
        self.deesser = DeEsser(
            {
                "freq": params.get("deess_freq", 6500.0),
                "q": params.get("deess_q", 1.0),
                "threshold_db": params.get("deess_threshold_db", -20.0),
                "ratio": params.get("deess_ratio", 6.0),
                "atk_ms": params.get("deess_atk_ms", 3.0),
                "rel_ms": params.get("deess_rel_ms", 60.0),
            }
        )
        self.transient = TransientShaper(
            {
                "attack_db": params.get("transient_attack_db", 1.0),
                "sustain_db": params.get("transient_sustain_db", 0.0),
            }
        )
        self.limiter = Limiter(
            {
                "ceiling_db": params.get("ceiling_db", -1.0),
                "release_ms": params.get("limiter_release_ms", 60.0),
            }
        )

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        y = _ensure_2d(block)
        y = self.demud.process(y, sr)
        y = self.presence.process(y, sr)
        y = self.air.process(y, sr)
        y = self.deesser.process(y, sr)
        y = self.transient.process(y, sr)
        y = self.limiter.process(y, sr)
        return _safe_audio(y)