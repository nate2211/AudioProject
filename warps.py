#!/usr/bin/env python
# warps.py — native-assisted streaming warps
# ---------------------------------------------------------------------
# AudioProject.dll does not currently expose native FFT/phase-vocoder,
# native WSOLA, or native resample/polyphase functions.
#
# Native substitutions used here:
#   - NativeSoftClipper for final safety clipping
#   - NativeSOS for speed anti-alias filtering
#   - NativeFixedBlockAdapter-compatible output behavior kept by returning
#     whatever each variable-rate warp produces, same as the original design.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
from fractions import Fraction
import math

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from filters import AudioFilter, register_filter


try:
    from native import (
        is_available as _native_is_available,
        NativeSoftClipper as _NativeSoftClipper,
        NativeSOS as _NativeSOS,
    )

    _NATIVE_OK = bool(_native_is_available())
except Exception:
    _NATIVE_OK = False
    _NativeSoftClipper = None
    _NativeSOS = None


_F32 = np.float32
_C64 = np.complex64
_EPS = np.finfo(np.float32).eps
_TWO_PI = 2.0 * math.pi


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=_F32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected audio array shaped (frames, channels), got {x.shape!r}")
    return np.ascontiguousarray(x, dtype=_F32)


def _safe_clip(x: np.ndarray, clamp: float) -> np.ndarray:
    x = np.asarray(x, dtype=_F32)

    if clamp <= 0 or x.size == 0:
        return x.astype(_F32, copy=False)

    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(_F32)

    m = float(np.max(np.abs(x)))
    if m > clamp:
        x = x * (clamp / (m + _EPS))

    if _NATIVE_OK and _NativeSoftClipper is not None:
        try:
            return _NativeSoftClipper(drive=1.0).process(x, 48000)
        except Exception:
            pass

    return np.clip(x, -1.0, 1.0).astype(_F32, copy=False)


def _sqrt_hann(n: int) -> NDArray[_F32]:
    w = signal.windows.hann(n, sym=False).astype(_F32)
    return np.sqrt(w, dtype=_F32)


def _linear_xfade(n: int) -> Tuple[NDArray[_F32], NDArray[_F32]]:
    if n <= 0:
        z = np.zeros((0,), dtype=_F32)
        return z, z
    up = np.linspace(0.0, 1.0, n, dtype=_F32)
    return up, (1.0 - up)


def _eqpow_xfade(n: int) -> Tuple[NDArray[_F32], NDArray[_F32]]:
    if n <= 0:
        z = np.zeros((0,), dtype=_F32)
        return z, z
    t = np.linspace(0.0, 1.0, n, dtype=_F32)
    up = np.sin(0.5 * np.pi * t)
    dn = np.cos(0.5 * np.pi * t)
    return up.astype(_F32), dn.astype(_F32)


def _resize_and_add(target: NDArray[_F32], seg: NDArray[_F32], pos: int) -> NDArray[_F32]:
    pos = max(0, pos)
    end_pos = pos + seg.shape[0]
    if end_pos > target.shape[0]:
        pad = np.zeros((end_pos - target.shape[0], target.shape[1]), dtype=_F32)
        target = np.concatenate([target, pad], axis=0)
    target[pos:end_pos, :] += seg
    return target


def _rms(x: NDArray[_F32]) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=_F32) + _EPS))


def _detect_transient(x: NDArray[_F32], prev_rms: float, alpha: float = 0.9) -> Tuple[float, bool]:
    if x.size == 0:
        return prev_rms, False
    rms = _rms(x)
    smooth = alpha * prev_rms + (1.0 - alpha) * rms
    return smooth, bool(rms > max(1e-4, smooth * 1.99))


class _NativeSOSFilter:
    def __init__(self):
        self.sos: Optional[np.ndarray] = None
        self.zi: Optional[np.ndarray] = None
        self._native = None
        self._C: Optional[int] = None

    def set_sos(self, sos: np.ndarray, C: int):
        sos = np.asarray(sos, dtype=np.float32)
        if sos.ndim != 2 or sos.shape[1] != 6:
            sos = np.array([[1, 0, 0, 1, 0, 0]], dtype=np.float32)

        if self.sos is not None and self._C == C and self.sos.shape == sos.shape and np.allclose(self.sos, sos):
            return

        self.sos = np.ascontiguousarray(sos, dtype=np.float32)
        self._C = C
        self.zi = None
        self._native = None

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
                return self._native.process(x, sr).astype(_F32, copy=False)
            except Exception:
                self._native = None
                self.zi = None

        if self.zi is None or self.zi.shape[1] != x.shape[1] or self.zi.shape[0] != self.sos.shape[0]:
            zi = signal.sosfilt_zi(self.sos.astype(np.float64)).astype(np.float32)
            self.zi = np.tile(zi[:, None, :], (1, x.shape[1], 1)).astype(np.float32)

        y = np.empty_like(x)

        for c in range(x.shape[1]):
            y[:, c], self.zi[:, c, :] = signal.sosfilt(
                self.sos.astype(np.float64),
                x[:, c],
                zi=self.zi[:, c, :],
            )

        return y.astype(_F32, copy=False)


@register_filter(
    "timestretch",
    help=(
        "Phase-vocoder time-stretch (pitch-preserving) with identity phase locking. "
        "Params: rate(1.0) win(2048) hop(win/4) peak_lock(true) mag_smooth(0.15) clamp(1.05) "
        "transients('auto'|'on'|'off')"
    ),
)
class PhaseVocoderStretch(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.rate = float(params.get("rate", 1.0))
        if self.rate <= 0:
            raise ValueError("timestretch.rate must be positive.")

        self.win = int(params.get("win", 2048))
        self.hop = int(params.get("hop", max(1, self.win // 4)))
        self.peak_lock = bool(params.get("peak_lock", True))
        self.mag_smooth = float(params.get("mag_smooth", 0.15))
        self.clamp = float(params.get("clamp", 1.05))
        self.transients = str(params.get("transients", "auto")).lower()
        if self.transients not in ("auto", "on", "off"):
            self.transients = "auto"

        self._syn_hop_base = max(1, int(round(self.hop / self.rate)))
        self._win_analysis = _sqrt_hann(self.win)
        self._win_synthesis = _sqrt_hann(self.win)

        self._sr: Optional[int] = None
        self._nch: Optional[int] = None
        self._in = np.zeros((0, 1), dtype=_F32)
        self._out = np.zeros((0, 1), dtype=_F32)

        n_bins = self.win // 2 + 1
        self._prev_phase: Optional[np.ndarray] = None
        self._prev_spec: Optional[np.ndarray] = None
        self._prev_mag: Optional[np.ndarray] = None
        self._frame_idx = 0
        self._out_off = 0
        self._rms_smooth = 0.0

    def _init_state(self, sr: int, nch: int) -> None:
        if self._sr is not None and self._nch == nch:
            return

        self._sr = sr
        self._nch = nch
        self._in = np.zeros((0, nch), dtype=_F32)
        self._out = np.zeros((0, nch), dtype=_F32)

        n_bins = self.win // 2 + 1
        self._prev_phase = np.zeros((n_bins, nch), dtype=np.float64)
        self._prev_spec = np.zeros((n_bins, nch), dtype=_C64)
        self._prev_mag = np.zeros((n_bins, nch), dtype=np.float64)
        self._frame_idx = 0
        self._out_off = 0
        self._rms_smooth = 0.0

    def _rfft_frame(self, pos: int) -> NDArray[_C64]:
        seg = np.zeros((self.win, self._nch), dtype=_F32)
        end = min(self._in.shape[0], pos + self.win)
        n = end - pos
        if n > 0:
            seg[:n, :] = self._in[pos:end, :]
        seg *= self._win_analysis[:, None]
        return np.fft.rfft(seg, axis=0).astype(_C64)

    def _peak_mask(self, mag: NDArray[_F32]) -> NDArray[np.int32]:
        bins, ch = mag.shape
        peaks = np.zeros_like(mag, dtype=bool)

        left = mag[0:bins - 2, :]
        mid = mag[1:bins - 1, :]
        right = mag[2:bins, :]
        peaks[1:bins - 1, :] = (mid > left) & (mid >= right)

        for c in range(ch):
            if not np.any(peaks[:, c]):
                peaks[int(np.argmax(mag[:, c])), c] = True

        peak_idx = np.zeros_like(mag, dtype=np.int32)

        for c in range(ch):
            pk_bins = np.flatnonzero(peaks[:, c])
            j = 0
            for b in range(mag.shape[0]):
                while j + 1 < len(pk_bins) and abs(pk_bins[j + 1] - b) <= abs(pk_bins[j] - b):
                    j += 1
                peak_idx[b, c] = pk_bins[j]

        return peak_idx

    def _phase_advance(self, bins: int) -> NDArray[_F32]:
        return (_TWO_PI * np.arange(bins, dtype=np.float64) * (self.hop / self.win)).astype(np.float64)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        self._init_state(sr, x.shape[1])
        self._in = np.concatenate([self._in, x], axis=0)

        produced = np.zeros((0, self._nch), dtype=_F32)
        bins = self.win // 2 + 1
        exp_adv = self._phase_advance(bins)[:, None]

        def syn_hop_for_block(b: NDArray[_F32]) -> int:
            if self.transients == "off":
                return self._syn_hop_base
            if self.transients == "on":
                return max(1, self._syn_hop_base // 2)
            self._rms_smooth, is_tr = _detect_transient(b, self._rms_smooth, alpha=0.90)
            return max(1, (self._syn_hop_base // 2) if is_tr else self._syn_hop_base)

        while self._in.shape[0] >= self.win:
            X = self._rfft_frame(0)
            mag = np.abs(X).astype(np.float64)
            pha = np.angle(X).astype(np.float64)

            if self._frame_idx == 0:
                Y = X.copy()
                self._prev_phase = pha.copy()
                self._prev_spec = X.copy()
                self._prev_mag = mag.copy()
            else:
                dphi = pha - np.angle(self._prev_spec).astype(np.float64)
                dphi -= exp_adv
                dphi = (dphi + math.pi) % (2.0 * math.pi) - math.pi
                omega = exp_adv + dphi

                syn_hop = syn_hop_for_block(self._in[:self.win, :])
                phase = self._prev_phase + omega * (float(syn_hop) / float(self.hop))

                if self.peak_lock:
                    peak_of = self._peak_mask(mag.astype(_F32))
                    pk_phase_syn = phase[peak_of, np.arange(phase.shape[1])]
                    pk_phase_cur = pha[peak_of, np.arange(pha.shape[1])]
                    dev = pha - pk_phase_cur
                    dev *= 0.7
                    phase = pk_phase_syn + dev

                if self.mag_smooth > 0.0 and self._prev_mag is not None:
                    mag = (1.0 - self.mag_smooth) * self._prev_mag + self.mag_smooth * mag

                Y = (mag * np.exp(1j * phase)).astype(_C64)

                self._prev_phase = phase
                self._prev_spec = X
                self._prev_mag = mag

                if self.transients == "on":
                    self._prev_phase = pha.copy()

            y = np.fft.irfft(Y, n=self.win, axis=0).astype(_F32)
            y *= self._win_synthesis[:, None]

            if self._frame_idx == 0:
                syn_pos_stream = 0
            else:
                syn_hop = syn_hop_for_block(self._in[:self.win, :])
                syn_pos_stream = int(round(self._frame_idx * syn_hop))

            pos_buf = syn_pos_stream - self._out_off
            self._out = _resize_and_add(self._out, y, pos_buf)

            self._frame_idx += 1
            self._in = self._in[self.hop:, :]

            if self._out.shape[0] > self.win:
                drain = self._out.shape[0] - self.win
                produced = np.concatenate([produced, self._out[:drain, :].copy()], axis=0)
                self._out = self._out[drain:, :]
                self._out_off += drain

        return _safe_clip(produced, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        if self._nch is None:
            return None

        if self._in.shape[0] > 0:
            pad = np.zeros((self.win, self._nch), dtype=_F32)
            self.process(np.concatenate([self._in, pad], axis=0), self._sr or 48000)
            self._in = np.zeros((0, self._nch), dtype=_F32)

        tail = self._out.copy()
        self._init_state(self._sr or 48000, self._nch)
        return _safe_clip(tail, self.clamp)


@register_filter(
    "wsola",
    help=(
        "WSOLA with mean-removed, pre-emphasized NCCF + equal-power crossfades, "
        "energy matching, and pitch-period snapping. "
        "Params: rate(1.0) win(2048) hop(win/4) search(hop/2) xfade(hop/3) clamp(1.02) "
        "pre_emph(0.0..0.97 default 0.85) min_hz(60) max_hz(1000) min_corr(0.15)"
    ),
)
class WSOLAStretch(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.rate = float(params.get("rate", 1.0))
        if self.rate <= 0:
            raise ValueError("wsola.rate must be positive.")

        self.win = int(params.get("win", 2048))
        self.hop = int(params.get("hop", max(1, self.win // 4)))
        self.search = int(params.get("search", max(1, self.hop // 2)))
        self.xfade = int(params.get("xfade", max(1, self.hop // 3)))
        self.clamp = float(params.get("clamp", 1.02))
        self.pre_emph = float(params.get("pre_emph", 0.85))
        self.min_hz = float(params.get("min_hz", 60.0))
        self.max_hz = float(params.get("max_hz", 1000.0))
        self.min_corr = float(params.get("min_corr", 0.15))

        self._syn_hop = max(1, int(round(self.hop / self.rate)))
        self._win = _sqrt_hann(self.win)
        self._xf_up, self._xf_dn = _eqpow_xfade(self.xfade)

        self._sr: Optional[int] = None
        self._nch: Optional[int] = None
        self._in = np.zeros((0, 1), dtype=_F32)
        self._out = np.zeros((0, 1), dtype=_F32)
        self._tmpl: Optional[np.ndarray] = None
        self._syn_pos = 0
        self._ana_pos = 0
        self._out_off = 0

    def _init_state(self, sr: int, nch: int) -> None:
        if self._sr is not None and self._nch == nch:
            return
        self._sr = sr
        self._nch = nch
        self._in = np.zeros((0, nch), dtype=_F32)
        self._out = np.zeros((0, nch), dtype=_F32)
        self._tmpl = None
        self._syn_pos = 0
        self._ana_pos = 0
        self._out_off = 0

    def _preemph(self, x: NDArray[_F32]) -> NDArray[_F32]:
        if not (0.0 < self.pre_emph <= 0.97) or x.shape[0] < 2:
            return x
        y = x.copy()
        y[1:, :] -= self.pre_emph * y[:-1, :]
        return y

    def _nccf(self, a: NDArray[_F32], b: NDArray[_F32]) -> float:
        if a.shape[0] == 0 or b.shape[0] == 0:
            return -1.0

        n = min(a.shape[0], b.shape[0])
        a = self._preemph(a[:n].astype(_F32))
        b = self._preemph(b[:n].astype(_F32))

        a = a - np.mean(a, axis=0, keepdims=True)
        b = b - np.mean(b, axis=0, keepdims=True)

        num = float(np.sum(a * b))
        den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)) + _EPS)
        return num / den

    def _best_offset(self, ref: NDArray[_F32], center: int) -> int:
        best_score = -1e9
        best = center
        lo = max(0, center - self.search)
        hi = min(max(0, self._in.shape[0] - self.win), center + self.search)

        for pos in range(lo, hi + 1):
            cand = self._in[pos:pos + ref.shape[0], :]
            score = self._nccf(ref, cand)

            if score > best_score:
                best_score = score
                best = pos

        if best_score < self.min_corr:
            return center

        return best

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        self._init_state(sr, x.shape[1])
        self._in = np.concatenate([self._in, x], axis=0)

        produced = np.zeros((0, self._nch), dtype=_F32)

        while self._in.shape[0] >= self.win + self.search + self.hop:
            if self._tmpl is None:
                seg = self._in[:self.win, :] * self._win[:, None]
                self._out = _resize_and_add(self._out, seg, self._syn_pos)
                self._tmpl = self._in[self.hop:self.hop + self.xfade, :].copy()
                self._ana_pos += self.hop
                self._syn_pos += self._syn_hop
            else:
                center = self._ana_pos
                best = self._best_offset(self._tmpl, center)

                seg = self._in[best:best + self.win, :].copy()
                seg_win = seg * self._win[:, None]

                pos_buf = self._syn_pos - self._out_off
                self._out = _resize_and_add(self._out, seg_win, pos_buf)

                self._tmpl = seg[self.hop:self.hop + self.xfade, :].copy()
                self._ana_pos = best + self.hop
                self._syn_pos += self._syn_hop

            keep = max(self.win + self.search + self.hop, self.ana_keep if hasattr(self, "ana_keep") else self.win * 2)

            if self._ana_pos > self.hop:
                drop = min(self._ana_pos - self.hop, max(0, self._in.shape[0] - keep))
                if drop > 0:
                    self._in = self._in[drop:, :]
                    self._ana_pos -= drop

            if self._out.shape[0] > self.win:
                drain = self._out.shape[0] - self.win
                produced = np.concatenate([produced, self._out[:drain, :].copy()], axis=0)
                self._out = self._out[drain:, :]
                self._out_off += drain

        return _safe_clip(produced, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        if self._nch is None:
            return None
        tail = self._out.copy()
        self._init_state(self._sr or 48000, self._nch)
        return _safe_clip(tail, self.clamp)


@register_filter(
    "speed",
    help="Speed / resample change. Params: rate(1.0) quality(8) clamp(1.05)",
)
class SpeedResample(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.rate = float(params.get("rate", 1.0))
        if self.rate <= 0:
            raise ValueError("speed.rate must be positive.")

        self.quality = int(params.get("quality", 8))
        self.clamp = float(params.get("clamp", 1.05))
        self._buf = np.zeros((0, 1), dtype=_F32)
        self._sr: Optional[int] = None
        self._nch: Optional[int] = None
        self._aa = _NativeSOSFilter()
        self._aa_ready = False

    def _init_state(self, sr: int, nch: int):
        if self._sr is not None and self._nch == nch:
            return
        self._sr = sr
        self._nch = nch
        self._buf = np.zeros((0, nch), dtype=_F32)
        self._aa_ready = False

    def _ensure_aa(self, sr: int, C: int):
        if self._aa_ready:
            return

        if self.rate > 1.0:
            cutoff = min(0.45 / self.rate, 0.45)
            sos = signal.butter(4, cutoff, btype="low", output="sos").astype(np.float32)
            self._aa.set_sos(sos, C)

        self._aa_ready = True

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block)
        self._init_state(sr, x.shape[1])
        self._ensure_aa(sr, x.shape[1])

        if self.rate > 1.0:
            x = self._aa.process(x, sr)

        self._buf = np.concatenate([self._buf, x], axis=0)

        if self._buf.shape[0] < 16:
            return np.zeros((0, self._nch), dtype=_F32)

        frac = Fraction(1.0 / self.rate).limit_denominator(1000)
        up = max(1, int(frac.numerator))
        down = max(1, int(frac.denominator))

        y = signal.resample_poly(self._buf, up, down, axis=0).astype(_F32)

        self._buf = np.zeros((0, self._nch), dtype=_F32)
        return _safe_clip(y, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        if self._nch is None or self._buf.shape[0] == 0:
            return None

        frac = Fraction(1.0 / self.rate).limit_denominator(1000)
        up = max(1, int(frac.numerator))
        down = max(1, int(frac.denominator))
        y = signal.resample_poly(self._buf, up, down, axis=0).astype(_F32)
        self._buf = np.zeros((0, self._nch), dtype=_F32)
        return _safe_clip(y, self.clamp)