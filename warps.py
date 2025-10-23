#!/usr/bin/env python
# warps.py — ultra‑low‑artifact streaming warps (WSOLA anti‑buzz edition)
# ---------------------------------------------------------------------
# Stream-safe, low-artifact audio warps:
#  - Phase-vocoder time-stretch with identity phase locking + peak tracking
#    (magnitude smoothing + transient seeding to reduce smear)
#  - WSOLA with *normalized, mean-removed, pre-emphasized* correlation,
#    equal‑power crossfades, pitch-period snapping, and energy matching
#  - Speed change with causal anti-alias prefilter and high-quality polyphase
#
# Design goals:
#  • Eliminate WSOLA buzz/clicks: robust overlap search + equal-power xfades.
#  • Reduce phasing/chorusing: identity phase locking around spectral peaks.
#  • Keep transients crisp: transient-aware hop/phase seeding, WSOLA option.
#  • Stable streaming: bounded state, block-by-block processing, small latency.
#  • Preserve clarity/loudness: √Hann OLA, gentle soft clamp, energy-matched xfades.
# ---------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

import math
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from fractions import Fraction

# Assumes a shared AudioFilter base class and registry from a 'filters.py' file.
from filters import AudioFilter, register_filter

# ==============================
# Constants & tiny utilities
# ==============================

_F32 = np.float32
_C64 = np.complex64
_EPS = np.finfo(np.float32).eps
_TWO_PI = 2.0 * math.pi


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    return x[:, None] if x.ndim == 1 else x


def _safe_clip(x: np.ndarray, clamp: float) -> np.ndarray:
    if clamp <= 0 or x.size == 0:
        return x
    m = float(np.max(np.abs(x)))
    if m > clamp:
        x *= (clamp / (m + _EPS))
    return np.clip(x, -1.0, 1.0)


def _sqrt_hann(n: int) -> NDArray[_F32]:
    """COLA-friendly √Hann (analysis*synthesis ≈ Hann) which preserves energy."""
    w = signal.windows.hann(n, sym=False).astype(_F32)
    return np.sqrt(w, dtype=_F32)


def _hann(n: int) -> NDArray[_F32]:
    return signal.windows.hann(n, sym=False).astype(_F32)


def _linear_xfade(n: int) -> Tuple[NDArray[_F32], NDArray[_F32]]:
    if n <= 0:
        z = np.zeros((0,), dtype=_F32)
        return z, z
    up = np.linspace(0.0, 1.0, n, dtype=_F32)
    return up, (1.0 - up)


def _eqpow_xfade(n: int) -> Tuple[NDArray[_F32], NDArray[_F32]]:
    """Equal-power crossfade curves (sin/cos) to avoid gain dips/bumps."""
    if n <= 0:
        z = np.zeros((0,), dtype=_F32)
        return z, z
    t = np.linspace(0.0, 1.0, n, dtype=_F32)
    up = np.sin(0.5 * np.pi * t)
    dn = np.cos(0.5 * np.pi * t)
    return up, dn


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
    """Cheap transient detector: RMS with a fast attack/slow release and threshold."""
    if x.size == 0:
        return prev_rms, False
    rms = _rms(x)
    smooth = alpha * prev_rms + (1.0 - alpha) * rms
    is_transient = (rms > max(1e-4, smooth * 1.99))  # > ~6dB jump
    return smooth, is_transient


# ================================================================
# 1) Phase-Vocoder with Identity Phase Locking (low-artifact)
# ================================================================
@register_filter(
    "timestretch",
    help=(
        "Phase-vocoder time-stretch (pitch-preserving) with identity phase locking. "
        "Params: rate(1.0) win(2048) hop(win/4) peak_lock(true) mag_smooth(0.15) clamp(1.05) "
        "transients('auto'|'on'|'off')"
    ),
)
class PhaseVocoderStretch(AudioFilter):
    """
    Low-artifact phase vocoder based on Laroche & Dolson (1999) identity phase
    locking. Peak bins act as phase anchors; surrounding bins are locked to
    their nearest peak to prevent 'phasiness' and preserve brightness.

    Extras:
      • √Hann analysis/synthesis (energy-preserving OLA)
      • Identity phase locking around spectral peaks (current analysis deviation)
      • Magnitude smoothing across frames (avoids pumping)
      • Transient handling: optional auto phase seeding + hop halving
      • Safety clamp to avoid overs
    """

    def __init__(self, params: Dict[str, Any]):
        self.rate = float(params.get("rate", 1.0))
        if self.rate <= 0:
            raise ValueError("timestretch.rate must be positive.")

        self.win = int(params.get("win", 2048))
        self.hop = int(params.get("hop", max(1, self.win // 4)))
        self.peak_lock = bool(params.get("peak_lock", True))
        self.mag_smooth = float(params.get("mag_smooth", 0.15))  # 0..1, higher = quicker
        self.clamp = float(params.get("clamp", 1.05))
        self.transients = str(params.get("transients", "auto")).lower()  # 'auto'|'on'|'off'
        if self.transients not in ("auto", "on", "off"):
            self.transients = "auto"

        # Derived
        self._syn_hop_base = max(1, int(round(self.hop / self.rate)))
        self._win_analysis = _sqrt_hann(self.win)
        self._win_synthesis = _sqrt_hann(self.win)

        # Streaming state
        self._sr: Optional[int] = None
        self._nch: Optional[int] = None
        self._in = np.zeros((0, 1), dtype=_F32)
        self._out = np.zeros((0, 1), dtype=_F32)

        n_bins = self.win // 2 + 1
        self._prev_phase: Optional[np.ndarray] = None            # (bins, ch) float64
        self._prev_spec: Optional[np.ndarray] = None             # (bins, ch) complex64
        self._prev_mag: Optional[np.ndarray] = None              # (bins, ch) float64
        self._frame_idx = 0
        self._out_off = 0
        self._rms_smooth = 0.0            # for transient detection

    # ---------- helpers ----------
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
        """
        Return nearest-peak index for each bin (per channel).
        Fast heuristic using local-maxima detection; ties favor lower bin.
        """
        bins, ch = mag.shape
        peaks = np.zeros_like(mag, dtype=bool)
        # internal bins only; keep DC/Nyquist off
        left = mag[0:bins - 2, :]
        mid = mag[1:bins - 1, :]
        right = mag[2:bins, :]
        is_peak_mid = (mid > left) & (mid >= right)
        peaks[1:bins - 1, :] = is_peak_mid

        # Fallback: if no peaks in a channel, use argmax
        for c in range(ch):
            if not np.any(peaks[:, c]):
                peaks[int(np.argmax(mag[:, c])), c] = True

        # nearest peak per bin:
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
        # expected advance per analysis hop (per bin)
        return (_TWO_PI * np.arange(bins, dtype=np.float64) * (self.hop / self.win)).astype(np.float64)

    # ---------- processing ----------
    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(_F32))
        self._init_state(sr, x.shape[1])
        self._in = np.concatenate([self._in, x], axis=0)

        produced = np.zeros((0, self._nch), dtype=_F32)
        bins = self.win // 2 + 1
        exp_adv = self._phase_advance(bins)[:, None]  # (bins,1)

        # transient policy (auto shrinks syn hop & can reseed phase at onsets)
        def syn_hop_for_block(b: NDArray[_F32]) -> int:
            if self.transients == "off":
                return self._syn_hop_base
            if self.transients == "on":
                return max(1, self._syn_hop_base // 2)
            # auto
            self._rms_smooth, is_tr = _detect_transient(b, self._rms_smooth, alpha=0.90)
            return max(1, (self._syn_hop_base // 2) if is_tr else self._syn_hop_base)

        while self._in.shape[0] >= self.win:
            # analysis frame
            X = self._rfft_frame(0)  # (bins, ch)
            mag = np.abs(X).astype(np.float64)
            pha = np.angle(X).astype(np.float64)

            if self._frame_idx == 0:
                # seed phases and magnitude memory
                Y = X.copy()
                self._prev_phase = pha.copy()
                self._prev_spec = X.copy()
                self._prev_mag = mag.copy()
            else:
                # true frequency estimation
                dphi = pha - np.angle(self._prev_spec).astype(np.float64)
                dphi -= exp_adv
                dphi = (dphi + math.pi) % (2.0 * math.pi) - math.pi
                omega = exp_adv + dphi  # per hop

                # synthesis phase accumulation
                syn_hop = syn_hop_for_block(self._in[:self.win, :])
                phase = self._prev_phase + omega * (float(syn_hop) / float(self.hop))

                # identity phase locking using current analysis deviation from peaks
                if self.peak_lock:
                    peak_of = self._peak_mask(mag.astype(_F32))  # (bins, ch)
                    pk_phase_syn = phase[peak_of, np.arange(phase.shape[1])]
                    pk_phase_cur = pha[peak_of, np.arange(pha.shape[1])]
                    dev = pha - pk_phase_cur
                    dev *= 0.7  # reduce deviation in tails to suppress phasing
                    phase = pk_phase_syn + dev

                # magnitude smoothing to prevent pumping/blur
                if self.mag_smooth > 0.0 and self._prev_mag is not None:
                    mag = (1.0 - self.mag_smooth) * self._prev_mag + self.mag_smooth * mag

                Y = (mag * np.exp(1j * phase)).astype(_C64)

                # carry forward
                self._prev_phase = phase
                self._prev_spec = X
                self._prev_mag = mag

                if self.transients == "on":
                    self._prev_phase = pha.copy()  # hard reseed on every frame

            # iSTFT with synthesis window
            y = np.fft.irfft(Y, n=self.win, axis=0).astype(_F32)
            y *= self._win_synthesis[:, None]

            # place into output with current synthesis position
            if self._frame_idx == 0:
                syn_pos_stream = 0
            else:
                syn_hop = syn_hop_for_block(self._in[:self.win, :])
                syn_pos_stream = int(round(self._frame_idx * syn_hop))

            pos_buf = syn_pos_stream - self._out_off
            self._out = _resize_and_add(self._out, y, pos_buf)

            # advance analysis
            self._frame_idx += 1
            self._in = self._in[self.hop:, :]

            # drain any fully overlapped region to keep latency bounded
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


# ======================================
# 2) WSOLA (transient-friendly stretch, anti-buzz)
# ======================================
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
    """
    Anti-buzz WSOLA:
      • Correlation uses *mean-removed*, *pre-emphasized* signals (voice-friendly).
      • Equal-power crossfades (sin/cos) to avoid gain dips/bumps.
      • Energy matching in overlap to remove sudden loudness steps.
      • Pitch-period snapping: prefer offsets near local F0 period to avoid zippering.
    """

    def __init__(self, params: Dict[str, Any]):
        self.rate = float(params.get("rate", 1.0))
        if self.rate <= 0:
            raise ValueError("wsola.rate must be positive.")

        self.win = int(params.get("win", 2048))
        self.hop = int(params.get("hop", max(1, self.win // 4)))
        self.search = int(params.get("search", max(1, self.hop // 2)))
        self.xfade = int(params.get("xfade", max(1, self.hop // 3)))
        self.clamp = float(params.get("clamp", 1.02))

        # anti-buzz knobs
        self.pre_emph = float(params.get("pre_emph", 0.85))  # 0..0.97 typical
        self.min_hz = float(params.get("min_hz", 60.0))
        self.max_hz = float(params.get("max_hz", 1000.0))
        self.min_corr = float(params.get("min_corr", 0.15))

        self._syn_hop = max(1, int(round(self.hop / self.rate)))
        self._win = _sqrt_hann(self.win)
        self._xf_up, self._xf_dn = _eqpow_xfade(self.xfade)

        # State
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

    # ---- correlation helpers (anti-buzz) ----
    def _preemph(self, x: NDArray[_F32]) -> NDArray[_F32]:
        if not (0.0 < self.pre_emph <= 0.97) or x.shape[0] < 2:
            return x
        y = x.copy()
        y[1:, :] -= self.pre_emph * y[:-1, :]
        return y

    def _nccf(self, a: NDArray[_F32], b: NDArray[_F32]) -> float:
        """Mean-removed, L2-normalized cross-correlation (multi-channel)."""
        if a.size == 0 or b.size == 0:
            return 0.0
        am = a - np.mean(a, axis=0, keepdims=True)
        bm = b - np.mean(b, axis=0, keepdims=True)
        num = float(np.sum(am * bm))
        den = float(np.sqrt(np.sum(am * am) * np.sum(bm * bm)) + _EPS)
        return num / den

    def _estimate_period(self, x: NDArray[_F32]) -> Optional[int]:
        """Pitch period (samples) from autocorr over summed channels; returns None if unknown."""
        if self._sr is None or x.shape[0] < 8:
            return None
        lo = max(1, int(self._sr / self.max_hz))
        hi = max(lo + 1, int(self._sr / max(self.min_hz, 1.0)))
        xmono = np.mean(x, axis=1)
        xmono = xmono - float(np.mean(xmono))
        # autocorr by direct method in the small window
        best_lag, best_val = None, -1e9
        for lag in range(lo, min(hi, xmono.shape[0] - 1)):
            v = float(np.dot(xmono[:-lag], xmono[lag:]))
            if v > best_val:
                best_val, best_lag = v, lag
        return best_lag

    def _best_overlap(self, target: int) -> Tuple[int, float]:
        """Return (best_pos, corr)
        Search around target using pre-emphasized, mean-removed NCCF, with
        coarse-to-fine strategy and optional pitch-period snapping.
        """
        if self._tmpl is None:
            return max(0, min(target, self._in.shape[0] - self.win)), 1.0

        start = max(0, target - self.search)
        end = min(self._in.shape[0] - self.win, target + self.search)
        if start >= end:
            return max(0, min(target, self._in.shape[0] - self.win)), 0.0

        ol = self.win - self.hop
        # Use the *overlap* part from the template for correlation
        t = self._tmpl[self.hop:self.hop + ol, :].astype(np.float64)
        t = self._preemph(t)

        # Precompute pitch period to favor periodic alignment
        period = self._estimate_period(t.astype(_F32))

        best_pos, best_corr = start, -1.0

        # Coarse step (2 samples)
        for i in range(start, end, 2):
            c = self._in[i:i + ol, :].astype(np.float64)
            c = self._preemph(c)
            r = self._nccf(t, c)
            # Prefer positive correlation to avoid sign flips (buzz)
            if r > best_corr:
                best_corr = r
                best_pos = i

        # Fine refine around coarse best ±4 samples
        refine_start = max(start, best_pos - 4)
        refine_end = min(end, best_pos + 5)
        for i in range(refine_start, refine_end):
            c = self._in[i:i + ol, :].astype(np.float64)
            c = self._preemph(c)
            r = self._nccf(t, c)
            if r > best_corr:
                best_corr = r
                best_pos = i

        # Period snapping (nudge to nearest period boundary if helpful)
        if period is not None and best_corr > self.min_corr:
            cand = int(round(best_pos / period) * period)
            if start <= cand <= end:
                c = self._in[cand:cand + ol, :].astype(np.float64)
                c = self._preemph(c)
                r = self._nccf(t, c)
                if r >= best_corr - 0.02:  # allow tiny tolerance
                    best_pos, best_corr = cand, r

        return best_pos, float(best_corr)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(_F32))
        self._init_state(sr, x.shape[1])
        self._in = np.concatenate([self._in, x], axis=0)

        produced = np.zeros((0, self._nch), dtype=_F32)

        # Seed initial grain
        if self._tmpl is None and self._in.shape[0] >= self.win:
            seg0 = self._in[:self.win, :] * self._win[:, None]
            pos = self._syn_pos - self._out_off
            self._out = _resize_and_add(self._out, seg0, pos)
            self._tmpl = seg0.copy()
            self._syn_pos += self.hop
            self._ana_pos += self.hop

        while (self._in.shape[0] - self._ana_pos) >= self.win:
            pos_in, corr = self._best_overlap(self._ana_pos)
            seg = (self._in[pos_in:pos_in + self.win, :] * self._win[:, None]).astype(_F32)

            # Crossfade safety in the overlap area + ENERGY MATCH + EQ-POWER
            if self._tmpl is not None and self.xfade > 0:
                p = self._syn_pos - self._out_off
                if p < 0:
                    # Not enough rendered audio yet to fade against; skip xfade this round
                    p = 0
                available = max(0, self._out.shape[0] - p)
                ol = min(self.xfade, available, seg.shape[0])
                if ol > 0:
                    out_tail = self._out[p:p + ol, :]
                    seg_head = seg[:ol, :]
                    # energy match (RMS) to avoid level steps
                    rms_out = _rms(out_tail)
                    rms_seg = _rms(seg_head)
                    if rms_seg > 0:
                        seg[:ol, :] *= (rms_out / (rms_seg + _EPS))
                    # equal-power fade
                    self._out[p:p + ol, :] *= self._xf_dn[:ol, None]
                    seg[:ol, :] *= self._xf_up[:ol, None]

            pos = self._syn_pos - self._out_off
            self._out = _resize_and_add(self._out, seg, pos)
            self._tmpl = seg.copy()

            # If correlation is too low (no good match), slightly shorten hop to hide buzz
            syn_hop = self._syn_hop if corr >= self.min_corr else max(1, int(0.9 * self._syn_hop))

            self._syn_pos += syn_hop
            self._ana_pos += self.hop

            # Drain fully synthesized region
            if self._out.shape[0] > self.win:
                drain = self._out.shape[0] - self.win
                produced = np.concatenate([produced, self._out[:drain, :].copy()], axis=0)
                self._out = self._out[drain:, :]
                self._out_off += drain

        # Trim input buffer to keep headroom for search and next win
        need = self.win + self.search + self.hop
        cutoff = max(0, self._ana_pos - need)
        if cutoff > 0:
            self._in = self._in[cutoff:, :]
            self._ana_pos -= cutoff

        return _safe_clip(produced, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        if self._out.shape[0] == 0:
            return None
        tail = self._out.copy()
        if self._nch is not None:
            self._init_state(self._sr or 48000, self._nch)
        return _safe_clip(tail, self.clamp)


# ===========================================
# 3) Speed (band-limited, high quality)
# ===========================================
@register_filter(
    "speed",
    help=(
        "Speed via band-limited polyphase resample (pitch changes). "
        "Params: rate(1.0) quality('kaiser_best'|'kaiser_fast') clamp(1.05) "
        "prefilter(true) carry(640) pre_lp_order(8)"
    ),
)
class SpeedResample(AudioFilter):
    """
    High-quality speed/pitch change using scipy.signal.resample_poly with an
    anti-alias pre-emphasis (optional, causal + stateful) to keep treble clear
    when downsampling, and a small carry buffer for continuity between blocks.
    """

    def __init__(self, params: Dict[str, Any]):
        self.rate = float(params.get("rate", 1.0))
        if self.rate <= 0:
            raise ValueError("speed.rate must be positive.")

        q = str(params.get("quality", "kaiser_best")).lower()
        if q not in ("kaiser_best", "kaiser_fast"):
            raise ValueError("speed.quality must be 'kaiser_best' or 'kaiser_fast'.")
        self.quality = q

        self.clamp = float(params.get("clamp", 1.05))
        self.prefilter = bool(params.get("prefilter", True))
        self._carry_n = int(params.get("carry", 640))
        self._pre_lp_order = int(params.get("pre_lp_order", 8))

        self._carry = np.zeros((0, 1), dtype=_F32)
        # state for causal lowpass across blocks
        self._sr: Optional[int] = None
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None  # (sections, channels, 2)

    def _design_prefilter(self, sr: int, nch: int) -> None:
        if not self.prefilter or self.rate >= 1.0:
            self._sos, self._zi = None, None
            return
        wc = min(0.49, max(0.01, 0.9 * self.rate))  # normalized cutoff
        sos = signal.butter(self._pre_lp_order, wc, btype="lowpass", output="sos")
        zi = signal.sosfilt_zi(sos)
        self._sos = sos
        self._zi = np.tile(zi[:, None, :], (1, nch, 1)).astype(_F32)

    def _prefilt(self, x: NDArray[_F32]) -> NDArray[_F32]:
        if self._sos is None or self._zi is None or x.size == 0:
            return x
        y = np.empty_like(x)
        for c in range(x.shape[1]):
            y[:, c], self._zi[:, c, :] = signal.sosfilt(self._sos, x[:, c], zi=self._zi[:, c, :])
        return y

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d(block.astype(_F32))

        # (Re)design prefilter if sample-rate or channels changed
        if self._sr != sr or self._carry.shape[1] != x.shape[1]:
            self._sr = sr
            self._carry = np.zeros((0, x.shape[1]), dtype=_F32)
            self._design_prefilter(sr, x.shape[1])

        xb = np.concatenate([self._carry, x], axis=0)

        # optional anti-alias prefilter on downsampling (causal, stateful)
        if self.prefilter and self.rate < 1.0 and xb.shape[0] >= 4:
            xb = self._prefilt(xb)

        frac = Fraction(self.rate).limit_denominator(10000)
        up, down = frac.numerator, frac.denominator

        chans = []
        for c in range(xb.shape[1]):
            y = signal.resample_poly(xb[:, c], up, down, window=self.quality)
            chans.append(y.astype(_F32)[:, None])
        yb = np.concatenate(chans, axis=1)

        # carry tail from the *input* side for continuity
        keep = min(self._carry_n, xb.shape[0])
        self._carry = xb[-keep:, :].copy()

        return _safe_clip(yb, self.clamp)

    def flush(self) -> Optional[np.ndarray]:
        if self._carry.size:
            self._carry = np.zeros((0, self._carry.shape[1]), dtype=_F32)
        self._zi = None
        return None
