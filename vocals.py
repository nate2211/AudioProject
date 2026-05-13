#!/usr/bin/env python
# voice.py — native-backed vocal filters for rap, singing, adlibs, spoken vocals
# ---------------------------------------------------------------------
# Same public filter names/signatures as before:
#   vocal_eq
#   vocal_compressor
#   vocal_deesser
#   vocal_saturation
#   vocal_limiter
#   vocal_doubler
#   vocal_chain
#   rap_vocal
#   singing_vocal
#   adlib_vocal
#   spoken_vocal
#
# Native substitutions through native.py / AudioProject.dll:
#   - _StatefulSOS -> NativeSOS
#   - VocalCompressor -> NativeCompressor
#   - VocalSaturation tanh stage -> NativeSoftClipper
#   - VocalLimiter -> NativeLimiter
#
# Anything not exposed by the DLL, such as the doubler delay line and
# de-esser gain-envelope loop, remains Python but is kept streaming-safe.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional, List

import math
import numpy as np
from scipy import signal

from filters import AudioFilter, register_filter


try:
    from native import (
        is_available as _native_is_available,
        NativeSOS,
        NativeCompressor,
        NativeLimiter,
        NativeSoftClipper,
    )

    _NATIVE_OK = bool(_native_is_available())
except Exception:
    _NATIVE_OK = False
    NativeSOS = None
    NativeCompressor = None
    NativeLimiter = None
    NativeSoftClipper = None


_F32 = np.float32
_EPS = np.finfo(np.float32).eps


# ============================================================
# Utility helpers
# ============================================================

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=_F32)
    if x.ndim == 1:
        return np.ascontiguousarray(x[:, None], dtype=_F32)
    if x.ndim != 2:
        raise ValueError(f"Audio block must be 1D or 2D, got shape={x.shape!r}")
    return np.ascontiguousarray(x, dtype=_F32)


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _gain_to_db(gain: float) -> float:
    return float(20.0 * math.log10(max(float(gain), 1e-12)))


def _safe_clip(x: np.ndarray, clamp: float = 1.05) -> np.ndarray:
    if x.size == 0:
        return x.astype(_F32, copy=False)

    y = np.asarray(x, dtype=_F32)

    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(_F32)

    if clamp > 0.0:
        peak = float(np.max(np.abs(y)))
        if peak > clamp:
            y = y * (clamp / (peak + _EPS))

    return np.clip(y, -1.0, 1.0).astype(_F32, copy=False)


def _norm_freq(sr: int, hz: float) -> float:
    nyq = max(1.0, float(sr) * 0.5)
    return float(min(0.999, max(1e-5, float(hz) / nyq)))


def _butter_highpass(sr: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    cutoff_hz = max(10.0, float(cutoff_hz))
    return signal.butter(order, _norm_freq(sr, cutoff_hz), btype="highpass", output="sos")


def _butter_lowpass(sr: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    cutoff_hz = max(20.0, float(cutoff_hz))
    return signal.butter(order, _norm_freq(sr, cutoff_hz), btype="lowpass", output="sos")


def _butter_bandpass(sr: int, low_hz: float, high_hz: float, order: int = 2) -> np.ndarray:
    low_hz = max(20.0, float(low_hz))
    high_hz = max(low_hz + 50.0, float(high_hz))
    high_hz = min(high_hz, float(sr) * 0.49)

    return signal.butter(
        order,
        [_norm_freq(sr, low_hz), _norm_freq(sr, high_hz)],
        btype="bandpass",
        output="sos",
    )


def _peaking_eq_sos(sr: int, freq_hz: float, q: float, gain_db: float) -> np.ndarray:
    freq_hz = min(max(20.0, float(freq_hz)), float(sr) * 0.49)
    q = max(0.05, float(q))
    gain_db = float(gain_db)

    if abs(gain_db) < 1e-6:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / float(sr)
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a

    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a

    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )


def _high_shelf_sos(sr: int, freq_hz: float, gain_db: float, slope: float = 0.8) -> np.ndarray:
    freq_hz = min(max(100.0, float(freq_hz)), float(sr) * 0.49)
    gain_db = float(gain_db)

    if abs(gain_db) < 1e-6:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / float(sr)
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    slope = max(0.05, float(slope))

    alpha = sin_w0 / 2.0 * math.sqrt(max(0.0, (a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0))
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha

    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)

    a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
    a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )


def _low_shelf_sos(sr: int, freq_hz: float, gain_db: float, slope: float = 0.8) -> np.ndarray:
    freq_hz = min(max(20.0, float(freq_hz)), float(sr) * 0.49)
    gain_db = float(gain_db)

    if abs(gain_db) < 1e-6:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / float(sr)
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    slope = max(0.05, float(slope))

    alpha = sin_w0 / 2.0 * math.sqrt(max(0.0, (a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0))
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha

    b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha)

    a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)
    a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )


class _StatefulSOS:
    """
    Same signature as old _StatefulSOS, but uses AudioProject.dll NativeSOS
    when available. Falls back to scipy.sosfilt.
    """

    def __init__(self) -> None:
        self.sos: Optional[np.ndarray] = None
        self.zi: Optional[np.ndarray] = None
        self.sr: Optional[int] = None
        self.nch: Optional[int] = None
        self.signature: Optional[str] = None
        self._native = None

    def configure(self, sos: np.ndarray, sr: int, nch: int, signature: str) -> None:
        sos = np.asarray(sos, dtype=np.float64)

        if (
            self.sos is not None
            and self.sr == sr
            and self.nch == nch
            and self.signature == signature
            and self.sos.shape == sos.shape
        ):
            return

        self.sos = sos
        self.sr = sr
        self.nch = nch
        self.signature = signature
        self.zi = None

        if _NATIVE_OK and NativeSOS is not None:
            try:
                if self._native is None:
                    self._native = NativeSOS(sos=sos.astype(_F32), channels=nch, reset_state=True)
                else:
                    self._native.set_sos(sos.astype(_F32), channels=nch, reset_state=True)
                return
            except Exception:
                self._native = None

        zi1 = signal.sosfilt_zi(self.sos).astype(_F32)
        self.zi = np.tile(zi1[:, None, :], (1, nch, 1)).astype(_F32)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = _ensure_2d(x)

        if self.sos is None or x.size == 0:
            return x.astype(_F32, copy=False)

        if self._native is not None:
            try:
                return self._native.process(x, self.sr or 48000).astype(_F32, copy=False)
            except Exception:
                self._native = None
                self.zi = None

        if self.zi is None or self.zi.shape[1] != x.shape[1] or self.zi.shape[0] != self.sos.shape[0]:
            zi1 = signal.sosfilt_zi(self.sos).astype(_F32)
            self.zi = np.tile(zi1[:, None, :], (1, x.shape[1], 1)).astype(_F32)

        y = np.empty_like(x, dtype=_F32)

        for c in range(x.shape[1]):
            y[:, c], self.zi[:, c, :] = signal.sosfilt(
                self.sos,
                x[:, c],
                zi=self.zi[:, c, :],
            )

        return _safe_clip(y, 1.05)

    def reset(self) -> None:
        self.sos = None
        self.zi = None
        self.sr = None
        self.nch = None
        self.signature = None
        if self._native is not None:
            try:
                self._native.reset()
            except Exception:
                pass
        self._native = None


# ============================================================
# 1) Vocal EQ
# ============================================================

@register_filter(
    "vocal_eq",
    help=(
        "Vocal tone EQ. Params: low_cut(80), body_db(0), mud_db(-2), "
        "box_db(-1), presence_db(3), bite_db(1), air_db(1.5), clamp(1.05)"
    ),
)
class VocalEQ(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.low_cut = float(params.get("low_cut", 80.0))
        self.body_db = float(params.get("body_db", 0.0))
        self.mud_db = float(params.get("mud_db", -2.0))
        self.box_db = float(params.get("box_db", -1.0))
        self.presence_db = float(params.get("presence_db", 3.0))
        self.bite_db = float(params.get("bite_db", 1.0))
        self.air_db = float(params.get("air_db", 1.5))
        self.clamp = float(params.get("clamp", 1.05))
        self._sos = _StatefulSOS()

    def _build_sos(self, sr: int) -> np.ndarray:
        parts: List[np.ndarray] = []

        if self.low_cut > 0.0:
            parts.append(_butter_highpass(sr, self.low_cut, order=2))
        if abs(self.body_db) > 1e-6:
            parts.append(_low_shelf_sos(sr, 180.0, self.body_db, slope=0.8))
        if abs(self.mud_db) > 1e-6:
            parts.append(_peaking_eq_sos(sr, 300.0, 0.9, self.mud_db))
        if abs(self.box_db) > 1e-6:
            parts.append(_peaking_eq_sos(sr, 650.0, 1.1, self.box_db))
        if abs(self.presence_db) > 1e-6:
            parts.append(_peaking_eq_sos(sr, 3300.0, 0.85, self.presence_db))
        if abs(self.bite_db) > 1e-6:
            parts.append(_peaking_eq_sos(sr, 5200.0, 1.0, self.bite_db))
        if abs(self.air_db) > 1e-6:
            parts.append(_high_shelf_sos(sr, 10000.0, self.air_db, slope=0.65))

        if not parts:
            return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        return np.vstack(parts)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        signature = (
            f"vocal_eq:{self.low_cut}:{self.body_db}:{self.mud_db}:"
            f"{self.box_db}:{self.presence_db}:{self.bite_db}:{self.air_db}"
        )
        self._sos.configure(self._build_sos(sr), sr, x.shape[1], signature)
        y = self._sos.process(x)
        return _safe_clip(y, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        return None


# ============================================================
# 2) Vocal compressor
# ============================================================

@register_filter(
    "vocal_compressor",
    help=(
        "Vocal compressor. Params: threshold_db(-20), ratio(4), knee_db(6), "
        "attack_ms(4), release_ms(90), makeup_db(3), mix(1), clamp(1.05)"
    ),
)
class VocalCompressor(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.threshold_db = float(params.get("threshold_db", -20.0))
        self.ratio = max(1.0, float(params.get("ratio", 4.0)))
        self.knee_db = max(0.0, float(params.get("knee_db", 6.0)))
        self.attack_ms = max(0.05, float(params.get("attack_ms", 4.0)))
        self.release_ms = max(1.0, float(params.get("release_ms", 90.0)))
        self.makeup_db = float(params.get("makeup_db", 3.0))
        self.mix = float(np.clip(float(params.get("mix", 1.0)), 0.0, 1.0))
        self.clamp = float(params.get("clamp", 1.05))
        self._env = 0.0
        self._sr: Optional[int] = None
        self._native = None

        if _NATIVE_OK and NativeCompressor is not None:
            try:
                self._native = NativeCompressor(
                    threshold_db=self.threshold_db,
                    ratio=self.ratio,
                    knee_db=self.knee_db,
                    attack_ms=self.attack_ms,
                    release_ms=self.release_ms,
                    makeup_db=self.makeup_db,
                    mix=self.mix,
                )
            except Exception:
                self._native = None

    def _gain_reduction_db(self, level_db: float) -> float:
        x = level_db - self.threshold_db
        knee = self.knee_db

        if knee <= 0.0:
            if x <= 0.0:
                return 0.0
            reduction = (1.0 - 1.0 / self.ratio) * x
            return -reduction

        half = knee * 0.5

        if x <= -half:
            return 0.0
        if x >= half:
            reduction = (1.0 - 1.0 / self.ratio) * x
            return -reduction

        reduction = (1.0 - 1.0 / self.ratio) * ((x + half) ** 2) / (2.0 * knee)
        return -reduction

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)

        if self._native is not None:
            try:
                self._native.set_params(
                    threshold_db=self.threshold_db,
                    ratio=self.ratio,
                    knee_db=self.knee_db,
                    attack_ms=self.attack_ms,
                    release_ms=self.release_ms,
                    makeup_db=self.makeup_db,
                    mix=self.mix,
                )
                return _safe_clip(self._native.process(x, sr), self.clamp)
            except Exception:
                self._native = None

        dry = x.copy()

        if self._sr != sr:
            self._sr = sr
            self._env = 0.0

        attack = math.exp(-1.0 / ((self.attack_ms / 1000.0) * sr))
        release = math.exp(-1.0 / ((self.release_ms / 1000.0) * sr))
        makeup = _db_to_gain(self.makeup_db)
        y = np.empty_like(x, dtype=_F32)
        peak = np.max(np.abs(x), axis=1)

        for i in range(x.shape[0]):
            target = float(peak[i])
            coeff = attack if target > self._env else release
            self._env = coeff * self._env + (1.0 - coeff) * target
            level_db = _gain_to_db(self._env)
            gr_db = self._gain_reduction_db(level_db)
            gain = _db_to_gain(gr_db) * makeup
            y[i, :] = x[i, :] * gain

        if self.mix < 1.0:
            y = dry * (1.0 - self.mix) + y * self.mix

        return _safe_clip(y, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        return None


# ============================================================
# 3) De-esser
# ============================================================

@register_filter(
    "vocal_deesser",
    help=(
        "Dynamic sibilance reducer. Params: low_hz(4500), high_hz(10000), "
        "threshold_db(-27), ratio(5), max_reduction_db(9), attack_ms(1), "
        "release_ms(70), amount(1), clamp(1.05)"
    ),
)
class VocalDeEsser(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.low_hz = float(params.get("low_hz", 4500.0))
        self.high_hz = float(params.get("high_hz", 10000.0))
        self.threshold_db = float(params.get("threshold_db", -27.0))
        self.ratio = max(1.0, float(params.get("ratio", 5.0)))
        self.max_reduction_db = max(0.0, float(params.get("max_reduction_db", 9.0)))
        self.attack_ms = max(0.05, float(params.get("attack_ms", 1.0)))
        self.release_ms = max(1.0, float(params.get("release_ms", 70.0)))
        self.amount = float(np.clip(float(params.get("amount", 1.0)), 0.0, 1.0))
        self.clamp = float(params.get("clamp", 1.05))
        self._band = _StatefulSOS()
        self._env = 0.0
        self._sr: Optional[int] = None

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)

        if self._sr != sr:
            self._sr = sr
            self._env = 0.0

        signature = f"deess:{self.low_hz}:{self.high_hz}"
        sos = _butter_bandpass(sr, self.low_hz, self.high_hz, order=2)
        self._band.configure(sos, sr, x.shape[1], signature)
        sib = self._band.process(x)

        attack = math.exp(-1.0 / ((self.attack_ms / 1000.0) * sr))
        release = math.exp(-1.0 / ((self.release_ms / 1000.0) * sr))
        y = np.empty_like(x, dtype=_F32)
        sib_peak = np.max(np.abs(sib), axis=1)

        for i in range(x.shape[0]):
            target = float(sib_peak[i])
            coeff = attack if target > self._env else release
            self._env = coeff * self._env + (1.0 - coeff) * target

            level_db = _gain_to_db(self._env)
            over_db = max(0.0, level_db - self.threshold_db)
            reduction_db = (1.0 - 1.0 / self.ratio) * over_db
            reduction_db = min(self.max_reduction_db, reduction_db) * self.amount
            sib_gain = _db_to_gain(-reduction_db)
            y[i, :] = x[i, :] - sib[i, :] + sib[i, :] * sib_gain

        return _safe_clip(y, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        return None


# ============================================================
# 4) Saturation
# ============================================================

@register_filter(
    "vocal_saturation",
    help=(
        "Soft vocal saturation. Params: drive_db(2), tone_db(0), mix(0.35), "
        "output_db(0), clamp(1.05)"
    ),
)
class VocalSaturation(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.drive_db = float(params.get("drive_db", 2.0))
        self.tone_db = float(params.get("tone_db", 0.0))
        self.mix = float(np.clip(float(params.get("mix", 0.35)), 0.0, 1.0))
        self.output_db = float(params.get("output_db", 0.0))
        self.clamp = float(params.get("clamp", 1.05))
        self._tone = _StatefulSOS()
        self._native_clip = None

        if _NATIVE_OK and NativeSoftClipper is not None:
            try:
                self._native_clip = NativeSoftClipper(drive=_db_to_gain(self.drive_db))
            except Exception:
                self._native_clip = None

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        dry = x.copy()
        drive = _db_to_gain(self.drive_db)

        if self._native_clip is not None:
            try:
                self._native_clip.set_drive(drive)
                wet = self._native_clip.process(x, sr)
            except Exception:
                self._native_clip = None
                norm = math.tanh(drive) if drive > 0.0 else 1.0
                wet = np.tanh(x * drive) / max(norm, _EPS)
        else:
            norm = math.tanh(drive) if drive > 0.0 else 1.0
            wet = np.tanh(x * drive) / max(norm, _EPS)

        if abs(self.tone_db) > 1e-6:
            signature = f"sat_tone:{self.tone_db}"
            sos = _high_shelf_sos(sr, 5500.0, self.tone_db, slope=0.75)
            self._tone.configure(sos, sr, x.shape[1], signature)
            wet = self._tone.process(wet)

        y = dry * (1.0 - self.mix) + wet * self.mix
        y *= _db_to_gain(self.output_db)
        return _safe_clip(y, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        return None


# ============================================================
# 5) Limiter
# ============================================================

@register_filter(
    "vocal_limiter",
    help=(
        "Simple vocal peak limiter. Params: ceiling_db(-0.8), release_ms(60), "
        "input_db(0), clamp(1.0)"
    ),
)
class VocalLimiter(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.ceiling_db = float(params.get("ceiling_db", -0.8))
        self.release_ms = max(1.0, float(params.get("release_ms", 60.0)))
        self.input_db = float(params.get("input_db", 0.0))
        self.clamp = float(params.get("clamp", 1.0))
        self._gain = 1.0
        self._sr: Optional[int] = None
        self._native = None

        if _NATIVE_OK and NativeLimiter is not None:
            try:
                self._native = NativeLimiter(
                    ceiling_db=self.ceiling_db,
                    attack_ms=0.1,
                    release_ms=self.release_ms,
                )
            except Exception:
                self._native = None

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block) * _db_to_gain(self.input_db)

        if self._native is not None:
            try:
                self._native.set_params(
                    ceiling_db=self.ceiling_db,
                    attack_ms=0.1,
                    release_ms=self.release_ms,
                )
                return _safe_clip(self._native.process(x, sr), self.clamp)
            except Exception:
                self._native = None

        if self._sr != sr:
            self._sr = sr
            self._gain = 1.0

        ceiling = _db_to_gain(self.ceiling_db)
        release = math.exp(-1.0 / ((self.release_ms / 1000.0) * sr))
        y = np.empty_like(x, dtype=_F32)

        for i in range(x.shape[0]):
            peak = float(np.max(np.abs(x[i, :])))
            needed = 1.0
            if peak > ceiling:
                needed = ceiling / (peak + _EPS)

            if needed < self._gain:
                self._gain = needed
            else:
                self._gain = release * self._gain + (1.0 - release) * 1.0

            y[i, :] = x[i, :] * self._gain

        return _safe_clip(y, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        return None


# ============================================================
# 6) Vocal doubler
# ============================================================

@register_filter(
    "vocal_doubler",
    help=(
        "Simple vocal doubler. Params: delay_ms(18), spread_ms(7), mix(0.22), "
        "feedback(0), width(1), clamp(1.05)"
    ),
)
class VocalDoubler(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.delay_ms = max(1.0, float(params.get("delay_ms", 18.0)))
        self.spread_ms = max(0.0, float(params.get("spread_ms", 7.0)))
        self.mix = float(np.clip(float(params.get("mix", 0.22)), 0.0, 1.0))
        self.feedback = float(np.clip(float(params.get("feedback", 0.0)), 0.0, 0.85))
        self.width = float(np.clip(float(params.get("width", 1.0)), 0.0, 1.0))
        self.clamp = float(params.get("clamp", 1.05))
        self._sr: Optional[int] = None
        self._nch: Optional[int] = None
        self._buf: Optional[np.ndarray] = None
        self._pos = 0
        self._max_delay = 0

    def _setup(self, sr: int, nch: int) -> None:
        if self._sr == sr and self._nch == nch and self._buf is not None:
            return

        self._sr = sr
        self._nch = nch
        max_ms = self.delay_ms + self.spread_ms + 10.0
        self._max_delay = max(4, int((max_ms / 1000.0) * sr) + 4)
        self._buf = np.zeros((self._max_delay, nch), dtype=_F32)
        self._pos = 0

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        self._setup(sr, x.shape[1])

        if self._buf is None:
            return x

        y = np.empty_like(x, dtype=_F32)
        base = int((self.delay_ms / 1000.0) * sr)
        spread = int((self.spread_ms / 1000.0) * sr)

        for i in range(x.shape[0]):
            dry = x[i, :]
            delayed = np.zeros((x.shape[1],), dtype=_F32)

            for c in range(x.shape[1]):
                if x.shape[1] == 1:
                    d = base
                else:
                    offset = -spread // 2 if c == 0 else spread // 2
                    d = max(1, base + offset)

                read_pos = (self._pos - d) % self._max_delay
                delayed[c] = self._buf[read_pos, c]

            if x.shape[1] >= 2 and self.width > 0.0:
                cross = delayed[::-1]
                delayed = delayed * (1.0 - self.width) + cross * self.width

            y[i, :] = dry * (1.0 - self.mix) + delayed * self.mix
            self._buf[self._pos, :] = dry + delayed * self.feedback
            self._pos = (self._pos + 1) % self._max_delay

        return _safe_clip(y, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        return None


# ============================================================
# 7) Style chain presets
# ============================================================

def _style_defaults(style: str) -> Dict[str, Any]:
    style = str(style).strip().lower()

    if style in ("rap", "rapping", "trap", "hiphop", "hip_hop"):
        return {
            "low_cut": 90.0,
            "body_db": 0.5,
            "mud_db": -2.5,
            "box_db": -1.5,
            "presence_db": 4.0,
            "bite_db": 2.0,
            "air_db": 1.0,
            "comp_threshold_db": -22.0,
            "comp_ratio": 5.5,
            "comp_knee_db": 5.0,
            "comp_attack_ms": 2.0,
            "comp_release_ms": 75.0,
            "comp_makeup_db": 4.0,
            "deess_threshold_db": -28.0,
            "deess_amount": 0.85,
            "sat_drive_db": 3.0,
            "sat_mix": 0.32,
            "sat_tone_db": 0.8,
            "double_mix": 0.0,
            "limiter_ceiling_db": -0.9,
        }

    if style in ("singing", "sung", "lead", "lead_vocal", "pop"):
        return {
            "low_cut": 75.0,
            "body_db": 1.0,
            "mud_db": -2.0,
            "box_db": -1.2,
            "presence_db": 2.6,
            "bite_db": 0.8,
            "air_db": 2.5,
            "comp_threshold_db": -20.0,
            "comp_ratio": 3.2,
            "comp_knee_db": 8.0,
            "comp_attack_ms": 8.0,
            "comp_release_ms": 130.0,
            "comp_makeup_db": 3.0,
            "deess_threshold_db": -29.0,
            "deess_amount": 0.75,
            "sat_drive_db": 1.5,
            "sat_mix": 0.22,
            "sat_tone_db": 0.4,
            "double_mix": 0.12,
            "limiter_ceiling_db": -0.8,
        }

    if style in ("adlib", "adlibs", "background", "bgv", "harmony"):
        return {
            "low_cut": 130.0,
            "body_db": -0.5,
            "mud_db": -3.0,
            "box_db": -1.8,
            "presence_db": 3.2,
            "bite_db": 1.5,
            "air_db": 3.0,
            "comp_threshold_db": -24.0,
            "comp_ratio": 4.2,
            "comp_knee_db": 6.0,
            "comp_attack_ms": 5.0,
            "comp_release_ms": 100.0,
            "comp_makeup_db": 3.5,
            "deess_threshold_db": -30.0,
            "deess_amount": 0.65,
            "sat_drive_db": 3.5,
            "sat_mix": 0.40,
            "sat_tone_db": 1.2,
            "double_mix": 0.28,
            "limiter_ceiling_db": -1.0,
        }

    if style in ("spoken", "speech", "podcast", "voiceover"):
        return {
            "low_cut": 85.0,
            "body_db": 0.5,
            "mud_db": -2.5,
            "box_db": -2.0,
            "presence_db": 2.0,
            "bite_db": 0.5,
            "air_db": 0.5,
            "comp_threshold_db": -21.0,
            "comp_ratio": 3.8,
            "comp_knee_db": 8.0,
            "comp_attack_ms": 4.0,
            "comp_release_ms": 110.0,
            "comp_makeup_db": 3.0,
            "deess_threshold_db": -27.0,
            "deess_amount": 0.9,
            "sat_drive_db": 0.8,
            "sat_mix": 0.12,
            "sat_tone_db": -0.3,
            "double_mix": 0.0,
            "limiter_ceiling_db": -1.0,
        }

    return _style_defaults("singing")


@register_filter(
    "vocal_chain",
    help=(
        "Full vocal processing chain. Params: style('rap'|'singing'|'adlib'|'spoken'), "
        "plus overrides like presence_db, comp_ratio, sat_mix, double_mix, deess_amount."
    ),
)
class VocalChain(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.style = str(params.get("style", "singing")).strip().lower()
        d = _style_defaults(self.style)

        for k, v in params.items():
            d[k] = v

        self.eq = VocalEQ(
            {
                "low_cut": d.get("low_cut", 80.0),
                "body_db": d.get("body_db", 0.0),
                "mud_db": d.get("mud_db", -2.0),
                "box_db": d.get("box_db", -1.0),
                "presence_db": d.get("presence_db", 3.0),
                "bite_db": d.get("bite_db", 1.0),
                "air_db": d.get("air_db", 1.5),
                "clamp": 1.05,
            }
        )

        self.comp = VocalCompressor(
            {
                "threshold_db": d.get("comp_threshold_db", -20.0),
                "ratio": d.get("comp_ratio", 4.0),
                "knee_db": d.get("comp_knee_db", 6.0),
                "attack_ms": d.get("comp_attack_ms", 4.0),
                "release_ms": d.get("comp_release_ms", 90.0),
                "makeup_db": d.get("comp_makeup_db", 3.0),
                "mix": d.get("comp_mix", 1.0),
                "clamp": 1.05,
            }
        )

        self.deesser = VocalDeEsser(
            {
                "low_hz": d.get("deess_low_hz", 4500.0),
                "high_hz": d.get("deess_high_hz", 10000.0),
                "threshold_db": d.get("deess_threshold_db", -27.0),
                "ratio": d.get("deess_ratio", 5.0),
                "max_reduction_db": d.get("deess_max_reduction_db", 9.0),
                "attack_ms": d.get("deess_attack_ms", 1.0),
                "release_ms": d.get("deess_release_ms", 70.0),
                "amount": d.get("deess_amount", 1.0),
                "clamp": 1.05,
            }
        )

        self.sat = VocalSaturation(
            {
                "drive_db": d.get("sat_drive_db", 2.0),
                "tone_db": d.get("sat_tone_db", 0.0),
                "mix": d.get("sat_mix", 0.25),
                "output_db": d.get("sat_output_db", 0.0),
                "clamp": 1.05,
            }
        )

        self.double_mix = float(d.get("double_mix", 0.0))
        self.doubler = VocalDoubler(
            {
                "delay_ms": d.get("double_delay_ms", 18.0),
                "spread_ms": d.get("double_spread_ms", 7.0),
                "mix": self.double_mix,
                "feedback": d.get("double_feedback", 0.0),
                "width": d.get("double_width", 1.0),
                "clamp": 1.05,
            }
        )

        self.limiter = VocalLimiter(
            {
                "ceiling_db": d.get("limiter_ceiling_db", -0.8),
                "release_ms": d.get("limiter_release_ms", 60.0),
                "input_db": d.get("limiter_input_db", 0.0),
                "clamp": 1.0,
            }
        )

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        y = _ensure_2d(block)
        y = self.eq.process(y, sr)
        y = self.comp.process(y, sr)
        y = self.deesser.process(y, sr)
        y = self.sat.process(y, sr)

        if self.double_mix > 0.0:
            y = self.doubler.process(y, sr)

        y = self.limiter.process(y, sr)
        return _safe_clip(y, 1.0)

    def flush(self) -> Optional[np.ndarray]:
        return None


@register_filter(
    "rap_vocal",
    help=(
        "Rap vocal preset chain. Strong compression, forward presence, controlled harshness. "
        "Override params: presence_db, bite_db, comp_ratio, sat_mix, deess_amount, etc."
    ),
)
class RapVocalChain(VocalChain):
    def __init__(self, params: Dict[str, Any]):
        p = dict(params)
        p["style"] = "rap"
        super().__init__(p)


@register_filter(
    "singing_vocal",
    help=(
        "Singing vocal preset chain. Smoother compression, air, warmth, optional doubling. "
        "Override params: air_db, body_db, comp_ratio, double_mix, deess_amount, etc."
    ),
)
class SingingVocalChain(VocalChain):
    def __init__(self, params: Dict[str, Any]):
        p = dict(params)
        p["style"] = "singing"
        super().__init__(p)


@register_filter(
    "adlib_vocal",
    help=(
        "Adlib/background vocal preset chain. Brighter, more saturated, wider by default. "
        "Override params: double_mix, sat_drive_db, air_db, low_cut, etc."
    ),
)
class AdlibVocalChain(VocalChain):
    def __init__(self, params: Dict[str, Any]):
        p = dict(params)
        p["style"] = "adlib"
        super().__init__(p)


@register_filter(
    "spoken_vocal",
    help=(
        "Spoken/podcast vocal preset chain. Clear, controlled, less musical coloration. "
        "Override params: low_cut, presence_db, comp_ratio, deess_amount, etc."
    ),
)
class SpokenVocalChain(VocalChain):
    def __init__(self, params: Dict[str, Any]):
        p = dict(params)
        p["style"] = "spoken"
        super().__init__(p)