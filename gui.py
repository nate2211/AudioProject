#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import re
import io
import wave
import shutil
import tempfile
import threading
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import sounddevice as sd
import requests
from pydub import AudioSegment
import time
import warps
import clarity
from helpers import FixedBlockAdapter, DCBlocker, SoftClipper, Limiter
from filters import available_filters, build_filter  # FX rack; SYNC/WSOLA is self-contained

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QThread, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QCursor, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QPushButton, QLabel, QSlider, QScrollArea, QComboBox, QGroupBox,
    QFileDialog, QFormLayout, QDialog, QDialogButtonBox, QCheckBox,
    QProgressBar, QInputDialog, QDoubleSpinBox, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsSimpleTextItem,QSplitter,
)

# Optional: YouTube support
try:
    import yt_dlp
except Exception:
    yt_dlp = None

def _ffmpeg_runs(ffmpeg_path: str) -> bool:
    try:
        p = subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
        return p.returncode == 0
    except Exception:
        return False
# =============================================================================
# FFmpeg PATH + pydub safety pointers
# =============================================================================

def get_ffmpeg_bin_dir() -> str:
    """
    Search order:
      1) PyInstaller onefile extraction dir (_MEIPASS)
      2) ffmpeg(.exe) next to this script OR in ./bin next to it
      3) Environment PATH (shutil.which)
      4) Your dev fallback path
      5) Last fallback: this script directory (even if ffmpeg isn't there)
    Returns: directory containing ffmpeg executable (preferred), or script dir as last resort.
    """

    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    def dir_has_ffmpeg(d: Path) -> bool:
        p = d / exe_name
        return p.exists() and _ffmpeg_runs(str(p))

    # 1) PyInstaller
    if hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)
        if dir_has_ffmpeg(meipass):
            return str(meipass)
        if dir_has_ffmpeg(meipass / "bin"):
            return str(meipass / "bin")

    # 2) Same directory as this file (or ./bin)
    here = Path(__file__).resolve().parent
    if dir_has_ffmpeg(here):
        return str(here)
    if dir_has_ffmpeg(here / "bin"):
        return str(here / "bin")

    # 3) PATH variable (system/global ffmpeg)
    w = shutil.which(exe_name)
    if w and _ffmpeg_runs(w):
        return str(Path(w).resolve().parent)

    # 4) Dev fallback
    dev_path = Path(r"C:\Users\natem\PycharmProjects\audioProject\ffmpeg-8.0-essentials_build\bin")
    if dir_has_ffmpeg(dev_path):
        return str(dev_path)

    # 5) Last fallback (keeps your old behavior)
    return str(here)

ffmpeg_bin_path = get_ffmpeg_bin_dir()

# Add to system path so pydub and subprocess can see it immediately
os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ["PATH"]

# Point pydub specifically to the executables
ffmpeg_exe = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
ffprobe_exe = os.path.join(ffmpeg_bin_path, "ffprobe.exe")

AudioSegment.converter = ffmpeg_exe
AudioSegment.ffprobe = ffprobe_exe


# =============================================================================
# Utilities
# =============================================================================

def _to_stereo_float32(seg: AudioSegment, target_sr: int) -> tuple[np.ndarray, int]:
    if seg.frame_rate != target_sr:
        seg = seg.set_frame_rate(target_sr)
    if seg.channels == 1:
        seg = seg.set_channels(2)

    sr = seg.frame_rate
    sw = seg.sample_width
    arr = np.array(seg.get_array_of_samples())

    arr = arr.reshape((-1, seg.channels)).astype(np.float32)
    denom = float(1 << (8 * sw - 1))
    if denom > 0:
        arr /= denom

    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    return arr, sr


def _simple_bpm_estimate(pcm_stereo: np.ndarray, sr: int) -> float:
    try:
        x = pcm_stereo[:, 0].astype(np.float32, copy=False)
        if x.size < sr:
            return 120.0
        x = x - float(np.mean(x))

        hop = 256
        win = 1024
        n = (len(x) - win) // hop
        if n <= 8:
            return 120.0

        env = np.empty(n, dtype=np.float32)
        for i in range(n):
            s = i * hop
            frame = x[s:s + win]
            env[i] = float(np.mean(frame * frame))
        env = np.maximum(env - np.mean(env), 0.0)

        ac = np.correlate(env, env, mode="full")[len(env) - 1:]
        if ac.size < 10:
            return 120.0

        env_sr = sr / hop
        min_lag = int(env_sr * 60.0 / 180.0)
        max_lag = int(env_sr * 60.0 / 70.0)
        if max_lag <= min_lag + 2 or max_lag >= ac.size:
            return 120.0

        window = ac[min_lag:max_lag]
        lag = int(np.argmax(window) + min_lag)
        bpm = 60.0 * env_sr / max(1, lag)
        bpm = float(np.clip(bpm, 60.0, 200.0))
        if abs(bpm - 120.0) < 1.0:
            bpm = 120.0
        return bpm
    except Exception:
        return 120.0


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


_YT_RE = re.compile(r"(?:youtube\.com|youtu\.be)", re.IGNORECASE)


def _is_youtube_url(url: str) -> bool:
    return bool(_YT_RE.search(url or ""))


def _load_youtube_audio_to_segment(url: str, *, target_sr: int) -> AudioSegment:
    if yt_dlp is None:
        raise RuntimeError("YouTube support requires yt-dlp. Install: pip install -U yt-dlp")

    tmpdir = tempfile.mkdtemp(prefix="yt_audio_")
    outtmpl = os.path.join(tmpdir, "audio.%(ext)s")
    final_path = os.path.join(tmpdir, "audio.mp3")

    cookiefile = (os.environ.get("YTDLP_COOKIES") or "").strip() or None
    browser = (os.environ.get("YTDLP_COOKIES_FROM_BROWSER") or "").strip() or None
    browser_profile = (os.environ.get("YTDLP_BROWSER_PROFILE") or "").strip() or None

    http_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.youtube.com/",
        "Origin": "https://www.youtube.com",
    }

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        "socket_timeout": 20,
        "http_headers": http_headers,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": ffmpeg_bin_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }

    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile
    elif browser:
        ydl_opts["cookiesfrombrowser"] = (browser, browser_profile) if browser_profile else (browser,)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(final_path):
            candidates = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
            candidates = [p for p in candidates if os.path.isfile(p)]
            if not candidates:
                raise RuntimeError("yt-dlp produced no output file.")
            final_path = max(candidates, key=lambda p: os.path.getsize(p))

        seg = AudioSegment.from_file(final_path)
        if seg.frame_rate != target_sr:
            seg = seg.set_frame_rate(target_sr)
        if seg.channels == 1:
            seg = seg.set_channels(2)
        return seg
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# NEW: ffmpeg pitch-shift helper (rubberband preferred, fallback asetrate/atempo)
# =============================================================================

def _ffmpeg_exe() -> str:
    exe = Path(ffmpeg_bin_path) / "ffmpeg.exe"
    if exe.exists():
        return str(exe)
    return "ffmpeg"


def _atempo_chain(rate: float) -> str:
    """
    atempo supports [0.5..2.0]. If we need outside range, chain multiple.
    We want a product of factors ~= rate.
    """
    rate = float(rate)
    if rate <= 0:
        rate = 1.0
    parts: List[float] = []
    x = rate
    while x > 2.0:
        parts.append(2.0)
        x /= 2.0
    while x < 0.5:
        parts.append(0.5)
        x /= 0.5
    parts.append(x)
    return ",".join([f"atempo={p:.8f}" for p in parts])


def ffmpeg_pitch_shift_segment(seg: AudioSegment, *, semitones: float, target_sr: int) -> AudioSegment:
    """
    Pitch shift WITHOUT changing length, using ffmpeg.
    - Preferred: rubberband=pitch=ratio
    - Fallback: asetrate/aresample + atempo inverse (approx)
    """
    ratio = float(2.0 ** (float(semitones) / 12.0))

    tmpdir = tempfile.mkdtemp(prefix="ff_pitch_")
    in_wav = os.path.join(tmpdir, "in.wav")
    out_wav = os.path.join(tmpdir, "out.wav")

    try:
        # Normalize to stereo @ target_sr in a wav container for ffmpeg
        seg2 = seg
        if seg2.channels == 1:
            seg2 = seg2.set_channels(2)
        if seg2.frame_rate != target_sr:
            seg2 = seg2.set_frame_rate(target_sr)
        seg2 = seg2.set_sample_width(2)  # 16-bit for compatibility
        seg2.export(in_wav, format="wav")

        # Try rubberband first
        ff = _ffmpeg_exe()
        filt_rb = f"rubberband=pitch={ratio:.8f}"
        cmd_rb = [
            ff, "-y",
            "-i", in_wav,
            "-vn",
            "-af", filt_rb,
            "-ar", str(int(target_sr)),
            "-ac", "2",
            out_wav
        ]

        def _run(cmd: list[str]) -> tuple[int, str]:
            try:
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return int(p.returncode), (p.stderr or "")
            except Exception as e:
                return 1, repr(e)

        code, err = _run(cmd_rb)
        if code != 0 or (not os.path.exists(out_wav)) or (os.path.getsize(out_wav) < 1024):
            # Fallback: asetrate changes pitch+tempo; compensate tempo with atempo=1/ratio
            # chain atempo to stay within bounds
            inv = 1.0 / max(1e-9, ratio)
            tempo_chain = _atempo_chain(inv)
            # asetrate uses the stream sample rate; we know it's target_sr
            filt_fb = f"asetrate={int(target_sr)}*{ratio:.8f},aresample={int(target_sr)},{tempo_chain}"
            cmd_fb = [
                ff, "-y",
                "-i", in_wav,
                "-vn",
                "-af", filt_fb,
                "-ar", str(int(target_sr)),
                "-ac", "2",
                out_wav
            ]
            code2, err2 = _run(cmd_fb)
            if code2 != 0 or (not os.path.exists(out_wav)) or (os.path.getsize(out_wav) < 1024):
                raise RuntimeError(
                    "FFmpeg pitch shift failed.\n"
                    f"rubberband err:\n{err}\n\nfallback err:\n{err2}"
                )

        return AudioSegment.from_file(out_wav, format="wav")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# Parameter Sanitization Layer (FX rack)
# =============================================================================

def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def _safe_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def sanitize_filter_params(filter_name: str, params: dict, *, sr: int) -> dict:
    """
    Keep the FX rack robust by clamping common parameters to valid ranges.
    Extend this as you add more filters.
    """
    name = (filter_name or "").strip().lower()
    p = dict(params or {})

    nyq = 0.5 * float(sr)
    nyq_safe = max(100.0, nyq * 0.999)

    def getf(key: str, default: float) -> float:
        v = p.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    if name == "gain":
        if "lin" not in p and "db" not in p:
            p["db"] = 0.0
        if "db" in p:
            p["db"] = float(getf("db", 0.0))
        if "lin" in p and p["lin"] is not None:
            p["lin"] = float(getf("lin", 1.0))

    elif name == "normalize":
        p["peak"] = _clamp(getf("peak", 0.98), 0.01, 1.0)

    elif name == "lowpass":
        cutoff = getf("cutoff", min(8000.0, nyq_safe))
        cutoff = _clamp(cutoff, 20.0, nyq_safe)
        p["cutoff"] = cutoff
        p["order"] = max(1, _safe_int(p.get("order", 4), 4))

    elif name == "highpass":
        cutoff = getf("cutoff", 80.0)
        cutoff = _clamp(cutoff, 10.0, nyq_safe)
        p["cutoff"] = cutoff
        p["order"] = max(1, _safe_int(p.get("order", 4), 4))

    elif name == "bandpass":
        low = getf("low", 200.0)
        high = getf("high", 2000.0)

        low = _clamp(low, 10.0, nyq_safe)
        high = _clamp(high, 20.0, nyq_safe)

        if high <= low:
            high = min(nyq_safe, low + 200.0)
        if (high - low) < 10.0:
            high = min(nyq_safe, low + 10.0)

        p["low"] = float(max(10.0, low))
        p["high"] = float(min(nyq_safe, high))
        p["order"] = max(1, _safe_int(p.get("order", 4), 4))

    return p


# =============================================================================
# WSOLA (self-contained) + render cache for SYNC mode
# =============================================================================

def _hann(n: int) -> np.ndarray:
    n = int(max(2, n))
    return np.hanning(n).astype(np.float32)


def _safe_norm(x: np.ndarray) -> float:
    v = float(np.dot(x, x))
    return max(v, 1e-12)


def wsola_stretch_stereo(
        x: np.ndarray,
        out_len: int,
        *,
        win: int = 4096,
        hop_a: int = 1024,
        search: int = 1024,
        min_advance: int | None = None,
        fade_in: int = 64,
) -> np.ndarray:
    """
    Higher-quality WSOLA time-stretch (pitch-preserving) for stereo float32 [-1..1].
    Fixed to prevent early cutoff on long tracks due to integer drift.
    """
    if x is None or not isinstance(x, np.ndarray) or x.size == 0:
        return np.zeros((int(max(1, out_len)), 2), dtype=np.float32)

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    else:
        x = x[:, :2]

    in_len = int(x.shape[0])
    out_len = int(max(1, out_len))

    # Pass-through if lengths match closely (optimization)
    if abs(out_len - in_len) < 256:
        out = np.zeros((out_len, 2), dtype=np.float32)
        take = min(in_len, out_len)
        out[:take] = x[:take]
        return out

    win = int(max(128, win))
    win = int(min(win, in_len))

    hop_a = int(max(32, hop_a))
    hop_a = int(min(hop_a, max(1, win - 1)))

    search = int(max(16, search))
    search = int(min(search, max(0, win - 1)))

    # Calculate stretch ratio
    ratio = float(out_len) / float(in_len)

    # Calculate synthesis hop (input step size)
    # We use a float accumulator to prevent drift over long tracks
    hop_s_float = float(hop_a) / max(1e-6, ratio)

    if min_advance is None:
        min_advance = max(1, int(hop_s_float) // 2)
    else:
        min_advance = int(max(1, min_advance))

    w = _hann(win)
    guide = 0.5 * (x[:, 0] + x[:, 1])

    y = np.zeros((out_len + win + 8, 2), dtype=np.float32)
    norm = np.zeros((out_len + win + 8,), dtype=np.float32)

    # First frame copy
    frame0 = x[0:win]
    y[0:win] += frame0 * w[:, None]
    norm[0:win] += w

    a_pos = 0
    out_pos = hop_a

    # Reference length for correlation
    ref_len_base = max(64, min(win, hop_a * 2))

    # Maximum start index in input to grab a full window
    max_start = max(0, in_len - win)

    # Track input position with float to reduce drift
    exp_pos_float = 0.0

    while out_pos < out_len:
        # Increment float expected position
        exp_pos_float += hop_s_float
        exp = int(exp_pos_float)

        # [FIX] Do not break if exp >= in_len. Clamp instead.
        # This ensures we fill the output buffer even if math drifts slightly.
        if exp > max_start:
            exp = max_start

        # Search window constraints
        lo = max(0, exp - search)
        hi = min(max_start, exp + search)

        # If clamp forced hi < lo, force valid range
        if hi < lo:
            hi = lo

        ref_len = min(ref_len_base, in_len - a_pos)

        # If we are basically at the end of the input, just lock to the end
        if ref_len < 32:
            best = max_start
        else:
            # Standard WSOLA Search
            ref = guide[a_pos:a_pos + ref_len].astype(np.float32, copy=False)
            ref = ref - float(np.mean(ref))
            refn = _safe_norm(ref)

            best = int(min(max(exp, 0), max_start))
            best_score = -1e30

            min_start_idx = min(max_start, a_pos + min_advance)

            # Optimization: If the search window is mostly valid, search it
            if hi >= min_start_idx:
                lo2 = max(lo, min_start_idx)
                hi2 = hi

                # Limit extensive search if window is huge
                if (hi2 - lo2) > 0:
                    candidates = np.arange(lo2, hi2 + 1)
                    # We can vectorize this correlation slightly for speed,
                    # but keeping your loop logic for stability:
                    for cand in range(lo2, hi2 + 1):
                        seg = guide[cand:cand + ref_len]
                        seg = seg - float(np.mean(seg))
                        score = float(np.dot(ref, seg)) / np.sqrt(refn * _safe_norm(seg))
                        if score > best_score:
                            best_score = score
                            best = cand

        # OLA (Overlap-Add)
        fr = x[best:best + win]
        y[out_pos:out_pos + win] += fr * w[:, None]
        norm[out_pos:out_pos + win] += w

        # Update anchor for next iteration
        a_pos = best

        # Update float tracker to match where we actually landed to prevent snaps
        # (Optional: keeps rhythm tighter)
        # exp_pos_float = float(best)

        out_pos += hop_a

    norm = np.maximum(norm, 1e-6)
    out = (y[:out_len] / norm[:out_len, None]).astype(np.float32, copy=False)

    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if fade_in > 0 and out.shape[0] > 1:
        n = min(int(fade_in), out.shape[0])
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        out[:n] *= ramp

    return out
def _apply_edge_fade(buf: np.ndarray, *, fade_in: bool, fade_out: bool, n: int = 256) -> np.ndarray:
    """
    Tiny ramp to prevent clicks/stutters at clip boundaries.
    n=256 @ 48kHz ≈ 5.3ms. Increase to 512 if you still hear edges.
    """
    if buf is None or buf.size == 0:
        return buf
    n = int(max(8, n))
    L = int(buf.shape[0])
    if L <= 1:
        return buf

    n = min(n, L)

    if fade_in:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        buf[:n] *= ramp

    if fade_out:
        ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
        buf[L - n:L] *= ramp

    return buf

class ClipRenderCache:
    """
    Thread-safe cache that allows lock-free reads for the audio callback.
    """
    def __init__(self):
        self._lock = threading.RLock()
        # The authoritative cache (protected by lock)
        self._cache_write: Dict[Tuple[int, int], Tuple[Tuple[int, int, int, int, int], np.ndarray]] = {}
        # The read-only snapshot for the audio thread (swapped atomically)
        self._cache_read: Dict[Tuple[int, int], Tuple[Tuple[int, int, int, int, int], np.ndarray]] = {}

    def invalidate_track(self, track: "AudioTrack"):
        tid = id(track)
        with self._lock:
            kill = [k for k in self._cache_write.keys() if k[0] == tid]
            for k in kill:
                self._cache_write.pop(k, None)
            # Update the read snapshot
            self._cache_read = self._cache_write.copy()

    def invalidate_clip(self, track: "AudioTrack", clip: "TrackClip"):
        k = (id(track), id(clip))
        with self._lock:
            self._cache_write.pop(k, None)
            # Update the read snapshot
            self._cache_read = self._cache_write.copy()

    def _sig_and_src(self, track: "AudioTrack", clip: "TrackClip"):
        dst_len = int(max(1, clip.length_samples))
        src_off = int(max(0, clip.src_offset_samples))
        src_len = int(max(1, clip.src_len_samples))

        if track.nframes <= 1:
            src_off = 0
            src_len = 1
        else:
            src_off = min(src_off, track.nframes - 1)
            src_end = min(track.nframes, src_off + src_len)
            src_len = max(1, src_end - src_off)

        sig = (track.nframes, track.sr, src_off, src_len, dst_len)
        src = track.pcm_data[src_off:src_off + src_len]
        return sig, src, dst_len

    def pre_render_clip(self, track: "AudioTrack", clip: "TrackClip") -> None:
        """Render one clip into the cache (UI Thread Only)."""
        if not getattr(track, "sync_enabled", False):
            return

        sig, src, dst_len = self._sig_and_src(track, clip)
        key = (id(track), id(clip))

        # Check existing without lock first (optimization)
        got = self._cache_write.get(key)
        if got is not None and got[0] == sig:
            return

        # WSOLA render (Heavy calculation - done outside lock)
        rendered = wsola_stretch_stereo(
            src,
            dst_len,
            win=3072,
            hop_a=768,
            search=768,
            fade_in=0,
        )

        with self._lock:
            self._cache_write[key] = (sig, rendered)
            # ATOMIC SWAP: Python dict assignment is atomic.
            # The audio thread will either see the old dict or the new dict,
            # but never a broken half-state.
            self._cache_read = self._cache_write.copy()

    def pre_render_all(self, tracks: List["AudioTrack"]) -> None:
        """Pre-render all clips for all SYNC-enabled tracks."""
        for t in tracks:
            if not getattr(t, "sync_enabled", False):
                continue
            for c in t.clips:
                if not c.enabled:
                    continue
                self.pre_render_clip(t, c)

    def get_render(self, track: "AudioTrack", clip: "TrackClip", *, allow_render: bool = True) -> Optional[np.ndarray]:
        """
        Return rendered clip.
        CRITICAL: If allow_render=False (Audio Callback), this reads from _cache_read
        without acquiring any locks.
        """
        if not getattr(track, "sync_enabled", False):
            return None

        key = (id(track), id(clip))
        sig, src, dst_len = self._sig_and_src(track, clip)

        if not allow_render:
            # --- FAST PATH FOR AUDIO CALLBACK ---
            # No locks. Just dictionary lookup.
            got = self._cache_read.get(key)
            if got is not None and got[0] == sig:
                return got[1]
            return None

        # --- SLOW PATH FOR EXPORT/UI ---
        with self._lock:
            got = self._cache_write.get(key)
            if got is not None and got[0] == sig:
                return got[1]

        # Render now (Main thread/Export only)
        rendered = wsola_stretch_stereo(src, dst_len, win=3072, hop_a=768, search=768, fade_in=0)
        with self._lock:
            self._cache_write[key] = (sig, rendered)
            self._cache_read = self._cache_write.copy()
        return rendered


# =============================================================================
# Sequencer clip model
# =============================================================================

@dataclass
class TrackClip:
    start_sample: int
    length_samples: int
    src_offset_samples: int = 0
    src_len_samples: int = 0          # fixed source span length for SYNC stretching
    enabled: bool = True

    def __post_init__(self):
        if self.src_len_samples <= 0:
            self.src_len_samples = int(max(1, self.length_samples))

    def end_sample(self) -> int:
        return int(self.start_sample + self.length_samples)


# =============================================================================
# Audio Track  (NEW: non-destructive pitch shift via ffmpeg + per-track button)
# =============================================================================

class AudioTrack:
    def __init__(self, path: str, is_url: bool = False, *, target_sr: int = 48000, auto_bpm: bool = True):
        self.path = path
        self.is_url = is_url
        self.name = path.split("/")[-1].split("?")[0] if is_url else Path(path).name

        self.volume = 1.0
        self.muted = False
        self.soloed = False
        self.active = True

        self.sync_enabled = False
        self.bpm = 120.0

        self.active_filters = []
        self.params = [{} for _ in range(8)]
        self.fx_selections = ["None"] * 8
        self.fx_enabled = [True] * 8
        self.filter_lock = threading.Lock()

        # NEW: pitch state
        self.pitch_semitones = 0.0

        seg = self._load_segment(path, is_url=is_url, target_sr=target_sr)
        self._base_segment = seg  # keep original segment for non-destructive pitch
        pcm, sr = _to_stereo_float32(seg, target_sr)

        self.sr = sr
        self.pcm_data = pcm
        self.nframes = int(pcm.shape[0])
        self.preview_data = self.pcm_data[: min(self.sr * 20, self.nframes)]
        self.duration_seconds = float(self.nframes) / float(self.sr)

        if auto_bpm:
            self.bpm = _simple_bpm_estimate(self.pcm_data, self.sr)

        self.clips: List[TrackClip] = [
            TrackClip(
                start_sample=0,
                length_samples=self.nframes,
                src_offset_samples=0,
                src_len_samples=self.nframes,
                enabled=True
            )
        ]

    def _load_segment(self, path: str, *, is_url: bool, target_sr: int) -> AudioSegment:
        if is_url:
            if _is_youtube_url(path):
                return _load_youtube_audio_to_segment(path, target_sr=target_sr)

            # Standard URL Load
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
            }

            r = requests.get(path, headers=headers, timeout=20, stream=True)
            r.raise_for_status()

            # Create a closed temp path reference
            tmp_path = ""

            try:
                # Save to a named temp file
                # The 'with' statement ensures the Python file handle is closed automatically
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    for chunk in r.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    tmp_path = tmp.name

                # Explicitly tell pydub where our bundled ffmpeg is
                seg = AudioSegment.from_file(tmp_path)
                return seg

            except Exception as e:
                raise RuntimeError(f"FFmpeg failed to decode the downloaded stream: {e}")

            finally:
                # Robust Windows Cleanup: Retry deletion if file is locked (WinError 32)
                if tmp_path and os.path.exists(tmp_path):
                    for i in range(5):  # Try 5 times
                        try:
                            os.remove(tmp_path)
                            break  # Success
                        except PermissionError:
                            # File is locked, wait and try again
                            time.sleep(0.1)
                        except Exception:
                            break

        return AudioSegment.from_file(path)

    def apply_pitch_shift(self, semitones: float, *, target_sr: int, keep_length: bool = True) -> None:
        """
        Non-destructive: apply from _base_segment every time.
        Updates pcm_data/nframes/preview and resets clips to full-length (safe default).
        """
        semitones = float(semitones)
        self.pitch_semitones = semitones

        if abs(semitones) < 1e-6:
            seg = self._base_segment
        else:
            seg = ffmpeg_pitch_shift_segment(self._base_segment, semitones=semitones, target_sr=target_sr)

        pcm, sr = _to_stereo_float32(seg, target_sr)

        self.sr = sr
        self.pcm_data = pcm
        self.nframes = int(pcm.shape[0])
        self.preview_data = self.pcm_data[: min(self.sr * 20, self.nframes)]
        self.duration_seconds = float(self.nframes) / float(self.sr)

        # safety: clip model assumes samples; keep arrangement, but ensure src bounds
        # If you want: preserve existing clip timing; we clamp src spans.
        for c in self.clips:
            c.length_samples = int(max(1, c.length_samples))
            c.src_offset_samples = int(max(0, min(c.src_offset_samples, max(0, self.nframes - 1))))
            c.src_len_samples = int(max(1, min(c.src_len_samples, self.nframes)))

        # if there was only the default one, keep it in sync with new length
        if len(self.clips) == 1 and self.clips[0].start_sample == 0:
            self.clips[0].length_samples = int(self.nframes)
            self.clips[0].src_offset_samples = 0
            self.clips[0].src_len_samples = int(self.nframes)


# =============================================================================
# Audio Engine (mixes by sequencer clips)  <<< SYNC cache shared
# =============================================================================

class AudioEngine(QObject):
    finished = pyqtSignal()
    levels = pyqtSignal(float, float)
    position = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.is_playing = False
        self.is_streaming = False
        self._stop_event = threading.Event()

        self._pause_event = threading.Event()
        self.is_paused = False

        self.tracks: list[AudioTrack] = []
        self.live_pipeline = []
        self.live_params = [{} for _ in range(8)]
        self.live_fx_selections = ["None"] * 8
        self.filter_lock = threading.Lock()

        self.master_lock = threading.RLock()
        self.master_bpm = 128.0
        self.beats_per_bar = 4
        self.master_sr = 48000

        self.sample_pos = 0

        self.ui_meter_l = 0.0
        self.ui_meter_r = 0.0
        self.ui_sample_pos = 0

        self._safety_chain = [
            DCBlocker(r=0.995),
            Limiter(-1.0, 1.0, 80.0),
            SoftClipper(drive=1.5),
        ]

        self.render_cache = ClipRenderCache()

    def invalidate_track_renders(self, track: AudioTrack):
        self.render_cache.invalidate_track(track)

    def invalidate_clip_render(self, track: AudioTrack, clip: TrackClip):
        self.render_cache.invalidate_clip(track, clip)

    def pause(self):
        self._pause_event.set()
        self.is_paused = True

    def resume(self):
        self._pause_event.clear()
        self.is_paused = False

    def toggle_pause(self):
        if self._pause_event.is_set():
            self.resume()
        else:
            self.pause()

    def _update_meter(self, data: np.ndarray):
        if data.size == 0:
            return
        self.ui_meter_l = float(np.max(np.abs(data[:, 0])))
        self.ui_meter_r = float(np.max(np.abs(data[:, 1]))) if data.shape[1] > 1 else self.ui_meter_l

    def set_master_bpm(self, bpm: float):
        self.master_bpm = float(max(1.0, bpm))

    def seek_samples(self, new_sample_pos: int):
        with self.master_lock:
            self.sample_pos = int(max(0, new_sample_pos))
            self.ui_sample_pos = int(self.sample_pos)

    def song_length_samples(self) -> int:
        with self.master_lock:
            tracks = tuple(self.tracks)
        end_max = 0
        for t in tracks:
            for c in t.clips:
                if c.enabled:
                    end_max = max(end_max, c.end_sample())
        return int(end_max)

    def render_clip_to_buffer(self, track: AudioTrack, clip: TrackClip) -> np.ndarray:
        """Renders a single clip's current state (Sync + Pitch) to a raw buffer."""
        sr = self.master_sr

        # 1. Get the source audio (already pitched because pitch is applied to track.pcm_data)
        if track.sync_enabled:
            # Get the WSOLA rendered version from cache (or render it now)
            rendered = self.render_cache.get_render(track, clip, allow_render=True)
            return rendered.copy()
        else:
            # Standard slice
            s0 = int(clip.src_offset_samples)
            s1 = int(s0 + clip.src_len_samples)
            return track.pcm_data[s0:s1].copy()

    def _mix_track_clips(self, track: AudioTrack, global_start: int, frames: int) -> np.ndarray:
        out = np.zeros((frames, 2), dtype=np.float32)
        if not track.active or track.nframes <= 1:
            return out

        song_start = int(global_start)
        song_end = int(global_start + frames)
        sync_on = bool(getattr(track, "sync_enabled", False))

        for clip in track.clips:
            if not clip.enabled:
                continue

            c0 = int(clip.start_sample)
            c1 = int(clip.end_sample())
            if c1 <= song_start or c0 >= song_end:
                continue

            ov0 = max(song_start, c0)
            ov1 = min(song_end, c1)
            if ov1 <= ov0:
                continue

            dst0 = ov0 - song_start
            dst1 = ov1 - song_start
            n = dst1 - dst0
            if n <= 0:
                continue

            # offset inside the clip timeline
            clip_off = int(ov0 - c0)

            # edge fade flags relative to the CLIP boundary (not buffer block boundary)
            fade_in = (ov0 == c0)
            fade_out = (ov1 == c1)

            if sync_on:
                # CRITICAL: callback-safe: do NOT render here
                rendered = self.render_cache.get_render(track, clip, allow_render=False)
                if rendered is None or rendered.size == 0:
                    # If not pre-rendered, we output silence for this section (better than glitching).
                    # The fix is to ensure pre_render_all() is called before playback/export.
                    continue

                r0 = clip_off
                r1 = min(r0 + n, rendered.shape[0])
                take = r1 - r0
                if take <= 0:
                    continue

                seg = rendered[r0:r1].copy()
                seg = _apply_edge_fade(seg, fade_in=fade_in, fade_out=fade_out, n=256)
                out[dst0:dst0 + take] += seg
                continue

            # non-sync direct slice path
            src0 = int(clip.src_offset_samples + clip_off)
            src1 = int(src0 + n)

            if src0 >= track.nframes or src1 <= 0:
                continue

            s0 = max(0, src0)
            s1 = min(track.nframes, src1)
            take = s1 - s0
            if take <= 0:
                continue

            adj_left = s0 - src0
            dst0a = dst0 + adj_left
            dst1a = dst0a + take

            if dst0a < 0:
                shift = -dst0a
                dst0a = 0
                s0 += shift
            if dst1a > frames:
                take2 = frames - dst0a
                dst1a = frames
                s1 = s0 + take2

            seg = track.pcm_data[s0:s1].copy()
            seg = _apply_edge_fade(seg, fade_in=fade_in, fade_out=fade_out, n=256)
            out[dst0a:dst1a] += seg

        return out

    def play_all(self):
        if not self.tracks:
            return
        # PRE-RENDER all SYNC clips BEFORE starting audio callback to prevent stutter.
        with self.master_lock:
            tracks = list(self.tracks)
        self.render_cache.pre_render_all(tracks)

        self._stop_event.clear()
        self.resume()
        self.is_playing = True

        scratch = {"mix": None}

        def callback(outdata, frames, time_info, status):
            try:
                if self._stop_event.is_set():
                    raise sd.CallbackStop()

                if self._pause_event.is_set():
                    if outdata.ndim == 1:
                        outdata[:] = 0.0
                    else:
                        outdata[:, :] = 0.0
                    self.ui_meter_l = 0.0
                    self.ui_meter_r = 0.0
                    return

                mix = scratch["mix"]
                if mix is None or mix.shape[0] != frames:
                    mix = np.zeros((frames, 2), dtype=np.float32)
                    scratch["mix"] = mix
                else:
                    mix.fill(0.0)

                with self.master_lock:
                    gs = int(self.sample_pos)
                    tracks = tuple(self.tracks)

                song_end = 0
                for t in tracks:
                    for c in t.clips:
                        if c.enabled:
                            song_end = max(song_end, c.end_sample())

                any_solo = any(bool(getattr(t, "soloed", False)) for t in tracks)

                for t in tracks:
                    if not getattr(t, "active", True):
                        continue
                    if any_solo and not getattr(t, "soloed", False):
                        continue
                    if getattr(t, "muted", False):
                        continue

                    block = self._mix_track_clips(t, gs, frames)

                    try:
                        with t.filter_lock:
                            for flt in t.active_filters:
                                block = flt.process(block, self.master_sr)
                    except Exception:
                        block = 0.0 * block

                    vol = float(getattr(t, "volume", 1.0))
                    if vol != 1.0:
                        block = block * vol

                    mix += block

                if not np.all(np.isfinite(mix)):
                    mix.fill(0.0)
                np.clip(mix, -1.0, 1.0, out=mix)

                self._update_meter(mix)

                if outdata.ndim == 1:
                    outdata[:] = mix[:, 0]
                else:
                    out_ch = outdata.shape[1]
                    outdata[:, :min(out_ch, 2)] = mix[:, :min(out_ch, 2)]
                    if out_ch > 2:
                        outdata[:, 2:] = 0.0

                with self.master_lock:
                    self.sample_pos += int(frames)
                    self.ui_sample_pos = int(self.sample_pos)

                if song_end > 0 and (gs + frames) >= song_end:
                    self._stop_event.set()

            except sd.CallbackStop:
                raise
            except Exception as e:
                try:
                    outdata[:] = 0
                except Exception:
                    pass
                self._stop_event.set()
                print("[AudioEngine] play_all callback error:", repr(e))

        try:
            with sd.OutputStream(
                samplerate=self.master_sr,
                channels=2,
                callback=callback,
                dtype="float32",
                blocksize=4096,
                latency="high",
            ):
                while not self._stop_event.is_set():
                    sd.sleep(50)
        finally:
            self.is_playing = False
            self.is_paused = False
            self._pause_event.clear()
            self.finished.emit()

    def export_mixdown(
        self,
        out_path: str,
        *,
        fmt: str = "wav",
        total_samples: int | None = None,
        blocksize: int = 4096,
        apply_safety: bool = True,
        progress_cb=None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        fmt = (fmt or "wav").strip().lower()
        if fmt not in ("wav", "mp3"):
            raise ValueError("export_mixdown fmt must be 'wav' or 'mp3'")

        with self.master_lock:
            tracks = tuple(self.tracks)
            sr = int(self.master_sr)
        # PRE-RENDER for export too (avoids render spikes / ensures sync clips exist)
        try:
            self.render_cache.pre_render_all(list(tracks))
        except Exception as e:
            print("[AudioEngine] pre_render_all(export) failed:", repr(e))
        if not tracks:
            raise RuntimeError("No tracks to export.")

        if total_samples is None:
            total_samples = self.song_length_samples()
        total_samples = int(max(0, total_samples))
        if total_samples <= 0:
            raise RuntimeError("Nothing to export (song length is 0).")

        any_solo = any(bool(getattr(t, "soloed", False)) for t in tracks)

        out_path = str(out_path)
        if fmt == "wav":
            wav_path = out_path
        else:
            base = os.path.splitext(out_path)[0]
            wav_path = base + ".__tmp__.wav"

        wf = wave.open(wav_path, "wb")
        try:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)

            mix = np.zeros((blocksize, 2), dtype=np.float32)
            pos = 0
            last_percent = -1

            if callable(progress_cb):
                progress_cb(0, 0, total_samples)

            while pos < total_samples:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Export cancelled.")

                frames = min(blocksize, total_samples - pos)
                if mix.shape[0] != frames:
                    mix = np.zeros((frames, 2), dtype=np.float32)
                else:
                    mix.fill(0.0)

                gs = pos

                for t in tracks:
                    if not getattr(t, "active", True):
                        continue
                    if any_solo and not getattr(t, "soloed", False):
                        continue
                    if getattr(t, "muted", False):
                        continue

                    block = self._mix_track_clips(t, gs, frames)

                    try:
                        with t.filter_lock:
                            for flt in t.active_filters:
                                block = flt.process(block, sr)
                    except Exception:
                        block = 0.0 * block

                    vol = float(getattr(t, "volume", 1.0))
                    if vol != 1.0:
                        block = block * vol

                    mix += block

                if apply_safety:
                    try:
                        y = mix
                        for flt in self._safety_chain:
                            y = flt.process(y, sr)
                        mix = y
                    except Exception:
                        pass

                if not np.all(np.isfinite(mix)):
                    mix.fill(0.0)

                np.clip(mix, -1.0, 1.0, out=mix)

                pcm16 = (mix * 32767.0).astype(np.int16)
                wf.writeframes(pcm16.tobytes())

                pos += frames

                if callable(progress_cb):
                    percent = int((pos * 100) // max(1, total_samples))
                    if percent != last_percent:
                        last_percent = percent
                        progress_cb(percent, pos, total_samples)

        finally:
            wf.close()

        if fmt == "mp3":
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Export cancelled.")
                seg = AudioSegment.from_file(wav_path, format="wav")
                seg.export(out_path, format="mp3", bitrate="192k")
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

        if callable(progress_cb):
            progress_cb(100, total_samples, total_samples)

    def stream_live(self, in_idx: int, out_idx: int, *, use_wasapi_loopback: bool = False):
        self._stop_event.clear()
        self.is_streaming = True

        extra = None
        if use_wasapi_loopback:
            try:
                extra = sd.WasapiSettings(loopback=True)
            except Exception:
                extra = None

        scratch = {"y": None}

        def stream_callback(indata, outdata, frames, time_info, status):
            try:
                if self._stop_event.is_set():
                    raise sd.CallbackStop()

                x = np.asarray(indata, dtype=np.float32)
                if x.ndim == 1:
                    x = x[:, None]

                if x.shape[1] == 1:
                    y = scratch["y"]
                    if y is None or y.shape[0] != frames:
                        y = np.zeros((frames, 2), dtype=np.float32)
                        scratch["y"] = y
                    y[:, 0] = x[:, 0]
                    y[:, 1] = x[:, 0]
                    y2 = y
                else:
                    y2 = x[:, :2]

                y = y2

                if self.live_pipeline:
                    with self.filter_lock:
                        lp2 = tuple(self.live_pipeline)
                    for flt in lp2:
                        y = flt.process(y, self.master_sr)

                if y is None or not isinstance(y, np.ndarray):
                    y = np.zeros((frames, 2), dtype=np.float32)

                if y.ndim == 1:
                    y = y[:, None]
                if y.dtype != np.float32:
                    y = y.astype(np.float32, copy=False)

                if y.shape[0] != frames:
                    if y.shape[0] > frames:
                        y = y[:frames, :]
                    else:
                        pad = np.zeros((frames - y.shape[0], y.shape[1]), dtype=np.float32)
                        y = np.vstack([y, pad])

                if y.shape[1] == 1:
                    y = np.repeat(y, 2, axis=1)
                else:
                    y = y[:, :2]

                if (y is None) or (not np.all(np.isfinite(y))):
                    y = scratch["y"]
                    if y is None or y.shape[0] != frames:
                        y = np.zeros((frames, 2), dtype=np.float32)
                        scratch["y"] = y
                    else:
                        y.fill(0.0)

                np.clip(y, -1.0, 1.0, out=y)
                self._update_meter(y)

                if outdata.ndim == 1:
                    outdata[:] = y[:, 0]
                else:
                    out_ch = outdata.shape[1]
                    if out_ch == 1:
                        outdata[:, 0] = y[:, 0]
                    else:
                        outdata[:, :min(out_ch, 2)] = y[:, :min(out_ch, 2)]
                        if out_ch > 2:
                            outdata[:, 2:] = 0.0

            except sd.CallbackStop:
                raise
            except Exception as e:
                try:
                    outdata[:] = 0
                except Exception:
                    pass
                self._stop_event.set()
                print("[AudioEngine] stream_callback error:", repr(e))

        try:
            with sd.Stream(
                device=(in_idx, out_idx),
                samplerate=self.master_sr,
                channels=2,
                callback=stream_callback,
                dtype="float32",
                blocksize=8192,
                latency=0.25,
                extra_settings=extra,
            ):
                while not self._stop_event.is_set():
                    sd.sleep(100)
        finally:
            self.is_streaming = False
            self.finished.emit()


# =============================================================================
# UI Widgets (Waveform + Track list)
# =============================================================================

class WaveformWidget(QWidget):
    def __init__(self, audio_data: np.ndarray):
        super().__init__()
        self.setFixedHeight(50)
        self.samples = None
        if audio_data is not None and len(audio_data) > 0:
            resample_factor = max(1, len(audio_data) // 800)
            self.samples = audio_data[::resample_factor, 0]

    def set_audio(self, audio_data: np.ndarray):
        self.samples = None
        if audio_data is not None and len(audio_data) > 0:
            resample_factor = max(1, len(audio_data) // 800)
            self.samples = audio_data[::resample_factor, 0]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 10))
        if self.samples is None or len(self.samples) == 0:
            return
        painter.setPen(QPen(QColor(0, 255, 0, 120), 1))
        mid_y = self.height() / 2
        step = self.width() / len(self.samples)
        for i in range(len(self.samples) - 1):
            x1 = i * step
            y1 = mid_y + (self.samples[i] * mid_y)
            x2 = (i + 1) * step
            y2 = mid_y + (self.samples[i + 1] * mid_y)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))


# =============================================================================
# NEW: Pitch Dialog (scale-based choices + ffmpeg semitone shift)
# =============================================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES: Dict[str, List[int]] = {
    "Major (Ionian)":            [0, 2, 4, 5, 7, 9, 11],
    "Natural Minor (Aeolian)":   [0, 2, 3, 5, 7, 8, 10],
    "Harmonic Minor":            [0, 2, 3, 5, 7, 8, 11],
    "Melodic Minor":             [0, 2, 3, 5, 7, 9, 11],
    "Major Pentatonic":          [0, 2, 4, 7, 9],
    "Minor Pentatonic":          [0, 3, 5, 7, 10],
    "Blues":                     [0, 3, 5, 6, 7, 10],
    "Chromatic":                 list(range(12)),
}


class PitchDialog(QDialog):
    """
    Picks a pitch shift in semitones, but constrains the choices to a scale
    (so you get a "scale picker" vibe like you requested).

    You still choose the actual shift amount (e.g., +2, -5, +12),
    but the allowed shifts are those whose (shift mod 12) is in the chosen scale.
    """
    def __init__(self, parent=None, *, current_semitones: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle("Pitch Shift (FFmpeg)")
        self.setFixedWidth(420)
        self.setStyleSheet("background:#111; color:#fff;")
        self.selected_semitones = float(current_semitones)

        v = QVBoxLayout(self)

        form = QFormLayout()
        self.cmb_root = QComboBox()
        self.cmb_root.addItems(NOTE_NAMES)
        self.cmb_root.setCurrentText("C")

        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(list(SCALES.keys()))
        self.cmb_scale.setCurrentText("Major (Ionian)")

        self.cmb_shift = QComboBox()

        form.addRow("Root:", self.cmb_root)
        form.addRow("Scale:", self.cmb_scale)
        form.addRow("Shift (semitones):", self.cmb_shift)
        v.addLayout(form)

        tip = QLabel("Tip: Shifts shown are constrained to the chosen scale (mod 12).")
        tip.setStyleSheet("color:#aaa; font-size:11px;")
        v.addWidget(tip)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        v.addWidget(btns)

        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)

        self.cmb_root.currentIndexChanged.connect(self._rebuild_shifts)
        self.cmb_scale.currentIndexChanged.connect(self._rebuild_shifts)

        self._rebuild_shifts()
        self._select_closest(current_semitones)

    def _allowed_shifts(self) -> List[int]:
        root = NOTE_NAMES.index(self.cmb_root.currentText())
        intervals = SCALES.get(self.cmb_scale.currentText(), list(range(12)))
        allowed_mods = {(root + i) % 12 for i in intervals}

        shifts = []
        for s in range(-24, 25):
            if (s % 12) in allowed_mods:
                shifts.append(s)
        # Always include 0
        if 0 not in shifts:
            shifts.append(0)
        shifts = sorted(set(shifts))
        return shifts

    def _rebuild_shifts(self):
        shifts = self._allowed_shifts()
        self.cmb_shift.blockSignals(True)
        self.cmb_shift.clear()
        for s in shifts:
            label = f"{s:+d}" if s != 0 else "0 (no shift)"
            self.cmb_shift.addItem(label, s)
        self.cmb_shift.blockSignals(False)

    def _select_closest(self, semitones: float):
        target = int(round(float(semitones)))
        best_idx = 0
        best_dist = 10**9
        for i in range(self.cmb_shift.count()):
            val = int(self.cmb_shift.itemData(i))
            d = abs(val - target)
            if d < best_dist:
                best_dist = d
                best_idx = i
        self.cmb_shift.setCurrentIndex(best_idx)

    def _accept(self):
        self.selected_semitones = float(self.cmb_shift.currentData())
        self.accept()


class TrackWidget(QFrame):
    selected = pyqtSignal(object)
    reorder = pyqtSignal(object, int)
    removed = pyqtSignal(object)
    track_changed = pyqtSignal(object)

    def __init__(self, track_obj: AudioTrack, engine: AudioEngine):
        super().__init__()
        self.track = track_obj
        self.engine = engine
        self.setObjectName("TrackWidget")

        l = QVBoxLayout(self)
        l.setContentsMargins(5, 5, 5, 5)

        hdr = QHBoxLayout()
        self.name_btn = QPushButton(track_obj.name)
        self.name_btn.setStyleSheet("text-align:left; font-weight:bold; border:none; color:#fff; background:transparent;")
        self.name_btn.clicked.connect(lambda: self.selected.emit(self))
        hdr.addWidget(self.name_btn, 1)

        # per-track SYNC toggle in dock
        self.sync = QPushButton("SYNC")
        self.sync.setCheckable(True)
        self.sync.setChecked(bool(getattr(track_obj, "sync_enabled", False)))
        self.sync.setFixedHeight(20)
        self.sync.setStyleSheet(
            "QPushButton { background:#111; color:#aaa; border:1px solid #333; padding:2px 6px; }"
            "QPushButton:checked { background:#003355; color:#00ffcc; border:1px solid #00ffcc; }"
        )
        self.sync.clicked.connect(self.toggle_sync)
        hdr.addWidget(self.sync)

        # NEW: per-track PITCH button
        self.pitch = QPushButton("PITCH")
        self.pitch.setFixedHeight(20)
        self.pitch.setStyleSheet(
            "QPushButton { background:#111; color:#aaa; border:1px solid #333; padding:2px 6px; }"
            "QPushButton:hover { border:1px solid #00ff00; color:#00ff00; }"
        )
        self.pitch.clicked.connect(self.open_pitch_dialog)
        hdr.addWidget(self.pitch)

        up = QPushButton("▲")
        up.setFixedSize(18, 18)
        up.clicked.connect(lambda: self.reorder.emit(self, -1))
        dn = QPushButton("▼")
        dn.setFixedSize(18, 18)
        dn.clicked.connect(lambda: self.reorder.emit(self, 1))
        rm = QPushButton("✕")
        rm.setFixedSize(18, 18)
        rm.setStyleSheet("color:#f44; background:#222;")
        rm.clicked.connect(lambda: self.removed.emit(self))
        hdr.addWidget(up)
        hdr.addWidget(dn)
        hdr.addWidget(rm)
        l.addLayout(hdr)

        self.wf = WaveformWidget(track_obj.preview_data)
        l.addWidget(self.wf)

        ctrls = QHBoxLayout()
        self.mute = QPushButton("M")
        self.mute.setCheckable(True)
        self.mute.setFixedSize(22, 22)
        self.mute.clicked.connect(self.toggle_mute)

        self.solo = QPushButton("S")
        self.solo.setCheckable(True)
        self.solo.setFixedSize(22, 22)
        self.solo.clicked.connect(self.toggle_solo)

        self.vol = QSlider(Qt.Orientation.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(100)
        self.vol.valueChanged.connect(self.set_vol)

        ctrls.addWidget(self.mute)
        ctrls.addWidget(self.solo)
        ctrls.addWidget(self.vol)
        l.addLayout(ctrls)

        self.set_active(False)
        self._refresh_pitch_button_text()

    def _refresh_pitch_button_text(self):
        st = float(getattr(self.track, "pitch_semitones", 0.0))
        if abs(st) < 1e-6:
            self.pitch.setText("PITCH")
        else:
            self.pitch.setText(f"PITCH {st:+.0f}")

    def open_pitch_dialog(self):
        if self.engine.is_streaming or self.engine.is_playing:
            QMessageBox.information(self, "Pitch", "Stop playback/streaming before pitching a track.")
            return

        dlg = PitchDialog(self, current_semitones=float(getattr(self.track, "pitch_semitones", 0.0)))
        if not dlg.exec():
            return

        new_semi = float(dlg.selected_semitones)

        try:
            # Apply pitch via ffmpeg (non-destructive from base segment)
            self.track.apply_pitch_shift(new_semi, target_sr=int(self.engine.master_sr))

            # Invalidate SYNC renders because the underlying pcm changed
            self.engine.invalidate_track_renders(self.track)

            # Refresh waveform
            self.wf.set_audio(self.track.preview_data)

            self._refresh_pitch_button_text()
            self.track_changed.emit(self)

        except Exception as e:
            QMessageBox.critical(self, "Pitch Error", str(e))

    def toggle_sync(self):
        self.track.sync_enabled = bool(self.sync.isChecked())
        try:
            self.engine.invalidate_track_renders(self.track)
        except Exception:
            pass
        self.track_changed.emit(self)

    def toggle_mute(self):
        self.track.muted = self.mute.isChecked()
        self.mute.setStyleSheet("background:#f00;" if self.track.muted else "")

    def toggle_solo(self):
        self.track.soloed = self.solo.isChecked()
        self.solo.setStyleSheet("background:#0af;" if self.track.soloed else "")

    def set_vol(self, v):
        self.track.volume = v / 100.0

    def set_active(self, state: bool):
        bg = "#000" if not state else "#003366"
        self.setStyleSheet(f"#TrackWidget {{ background-color:{bg}; border:1px solid #333; border-radius:4px; }}")


class FXSlot(QWidget):
    def __init__(self, index: int):
        super().__init__()
        l = QHBoxLayout(self)
        l.setContentsMargins(0, 1, 0, 1)

        self.active = QPushButton("●")
        self.active.setCheckable(True)
        self.active.setChecked(True)
        self.active.setFixedSize(20, 20)
        self.active.setStyleSheet(
            "QPushButton { background:#111; color:#444; border-radius:10px; border:1px solid #333; } "
            "QPushButton:checked { color:#00ff00; background:#000; border:1px solid #00ff00; }"
        )

        self.selector = QComboBox()
        self.selector.addItem("None")
        self.selector.addItems(sorted(available_filters().keys()))
        self.selector.setStyleSheet(
            "QComboBox { background:#000; color:#fff; border:1px solid #333; height:24px; font-size:11px; padding-left:5px; } "
            "QComboBox QAbstractItemView { background-color:#111; color:#fff; selection-background-color:#00ff00; selection-color:#000; }"
        )

        l.addWidget(self.active)
        l.addWidget(self.selector)


# =============================================================================
# Export worker UI
# =============================================================================

class ExportProgressDialog(QDialog):
    canceled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exporting…")
        self.setModal(True)
        self.setFixedWidth(420)
        self.setStyleSheet("background:#111; color:#fff;")

        v = QVBoxLayout(self)

        self.lbl = QLabel("Preparing export…")
        self.lbl.setStyleSheet("color:#ddd;")
        v.addWidget(self.lbl)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        self.bar.setStyleSheet(
            "QProgressBar { background:#000; border:1px solid #333; height:18px; }"
            "QProgressBar::chunk { background-color:#00ff00; }"
        )
        v.addWidget(self.bar)

        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("background:#222; color:#fff; border:1px solid #444; padding:6px 14px;")
        self.btn_cancel.clicked.connect(self._on_cancel)
        row.addWidget(self.btn_cancel)
        v.addLayout(row)

    def _on_cancel(self):
        self.btn_cancel.setEnabled(False)
        self.lbl.setText("Canceling…")
        self.canceled.emit()

    def set_progress(self, percent: int, rendered: int, total: int):
        self.bar.setValue(int(percent))
        self.lbl.setText(f"Exporting… {percent}%  ({rendered:,} / {total:,} samples)")


class ExportWorker(QObject):
    progress = pyqtSignal(int, int, int)
    done = pyqtSignal(bool, str)

    def __init__(self, engine: AudioEngine, path: str, fmt: str, total_samples: int | None):
        super().__init__()
        self.engine = engine
        self.path = path
        self.fmt = fmt
        self.total_samples = total_samples
        self.cancel_event = threading.Event()

    def request_cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            def _cb(percent, rendered, total):
                self.progress.emit(int(percent), int(rendered), int(total))

            self.engine.export_mixdown(
                self.path,
                fmt=self.fmt,
                total_samples=self.total_samples,
                blocksize=4096,
                apply_safety=True,
                progress_cb=_cb,
                cancel_event=self.cancel_event,
            )
            if self.cancel_event.is_set():
                self.done.emit(False, "Export cancelled.")
            else:
                self.done.emit(True, f"Saved:\n{self.path}")
        except Exception as e:
            self.done.emit(False, str(e))


# =============================================================================
# Sequencer View (SYNC logic included)
# =============================================================================

class SequencerView(QGraphicsView):
    clip_selected = pyqtSignal(object, object)  # (AudioTrack, TrackClip)
    clips_changed = pyqtSignal()

    EDGE_PX = 7.0

    def __init__(self, engine: AudioEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setStyleSheet("background:#070707; border:1px solid #222;")

        self.row_h = 34
        self.header_w = 160
        self.px_per_sec = 140.0

        self._selected_track: Optional[AudioTrack] = None

        self.snap_enabled = True
        self.snap_mode = "1/4"

        self._drag_active = False
        self._drag_item: Optional[ClipItem] = None
        self._drag_kind: Optional[str] = None
        self._drag_anchor_scene_x = 0.0
        self._drag_anchor_start = 0
        self._drag_anchor_len = 0
        self._drag_anchor_off = 0
        self._drag_anchor_srclen = 0

        self._clipboard: List[dict] = []
        self._clipboard_min_start = 0

        self._playhead = self.scene.addRect(
            QRectF(0, 0, 2, 5000),
            QPen(QColor(255, 0, 0, 180)),
            QBrush(QColor(255, 0, 0, 60)),
        )
        self._playhead.setZValue(9999)
        self._rebuild_pending = False
        self.rebuild()

    def _samples_per_beat(self) -> int:
        bpm = max(1.0, float(self.engine.master_bpm))
        sr = int(self.engine.master_sr)
        return int(round((60.0 / bpm) * sr))

    def _snap_step_samples(self) -> int:
        if not self.snap_enabled:
            return 1

        spb = max(1, self._samples_per_beat())
        mode = (self.snap_mode or "off").strip().lower()

        if mode in ("off", "none"):
            return 1
        if mode in ("sec", "1s", "1 sec"):
            return int(self.engine.master_sr)

        if mode == "1/8":
            return max(1, spb // 8)
        if mode == "1/4":
            return max(1, spb // 4)
        if mode == "1/2":
            return max(1, spb // 2)
        if mode in ("beat", "1"):
            return spb
        if mode in ("bar", "1 bar"):
            return max(1, spb * int(max(1, self.engine.beats_per_bar)))

        return max(1, spb // 4)

    def quantize_samples(self, s: int, *, disable_snap: bool = False) -> int:
        if disable_snap or (not self.snap_enabled) or (self.snap_mode or "").lower() in ("off", "none"):
            return int(max(0, s))
        step = self._snap_step_samples()
        if step <= 1:
            return int(max(0, s))
        return int(max(0, int(round(s / step) * step)))

    def request_rebuild(self):
        if self._rebuild_pending:
            return
        self._rebuild_pending = True
        QTimer.singleShot(0, self._do_rebuild)

    def _do_rebuild(self):
        self._rebuild_pending = False
        self.rebuild()

    def song_len_samples(self) -> int:
        return self.engine.song_length_samples()

    def samples_to_x(self, samples: int) -> float:
        sr = float(self.engine.master_sr)
        sec = float(samples) / max(1.0, sr)
        return self.header_w + sec * self.px_per_sec

    def x_to_samples(self, x: float) -> int:
        sr = float(self.engine.master_sr)
        sec = float(max(0.0, x - self.header_w)) / max(1.0, self.px_per_sec)
        return int(round(sec * sr))

    def dx_to_samples(self, dx: float) -> int:
        sr = float(self.engine.master_sr)
        return int(round((dx / max(1.0, self.px_per_sec)) * sr))

    def set_selected_track(self, track: Optional[AudioTrack]):
        self._selected_track = track
        self.request_rebuild()

    def set_zoom_px_per_sec(self, pxps: float):
        self.px_per_sec = float(max(20.0, min(600.0, pxps)))
        self.request_rebuild()

    def set_snap(self, enabled: bool, mode: str):
        self.snap_enabled = bool(enabled)
        self.snap_mode = str(mode)
        self.request_rebuild()

    def rebuild(self):
        if self._drag_active:
            return

        sel = None
        for it in self.scene.selectedItems():
            if isinstance(it, ClipItem):
                sel = (it.track, it.clip)
                break

        self._drag_item = None
        self._drag_kind = None

        self.scene.clear()

        with self.engine.master_lock:
            tracks = tuple(self.engine.tracks)
            sr = int(self.engine.master_sr)

        song_len = max(self.song_len_samples(), sr * 10)
        width = self.samples_to_x(song_len) + 600
        height = max(240, len(tracks) * self.row_h + 70)
        self.scene.setSceneRect(0, 0, width, height)

        spb = self._samples_per_beat()
        bpb = int(max(1, self.engine.beats_per_bar))
        bar_samples = max(1, spb * bpb)

        minor = max(1, self._snap_step_samples())
        min_visual = max(minor, spb // 8)
        step = max(1, min_visual)

        n_lines = int(song_len // step) + 64
        for i in range(n_lines):
            s = i * step
            x = self.samples_to_x(s)
            is_bar = (s % bar_samples) == 0
            is_beat = (s % spb) == 0

            if is_bar:
                pen = QPen(QColor(80, 80, 80))
            elif is_beat:
                pen = QPen(QColor(45, 45, 45))
            else:
                pen = QPen(QColor(28, 28, 28))

            self.scene.addLine(x, 0, x, height, pen)

            if is_bar:
                bar_index = int(s // bar_samples)
                lbl = self.scene.addSimpleText(f"Bar {bar_index + 1}")
                lbl.setBrush(QBrush(QColor(140, 255, 140)))
                lbl.setPos(x + 3, 3)
                lbl.setFont(QFont("Consolas", 8))

        for i, t in enumerate(tracks):
            y = 40 + i * self.row_h
            if self._selected_track is t:
                self.scene.addRect(0, y, width, self.row_h, QPen(QColor(0, 0, 0)), QBrush(QColor(0, 60, 90, 120)))
            else:
                self.scene.addRect(0, y, width, self.row_h, QPen(QColor(0, 0, 0)), QBrush(QColor(10, 10, 10, 100)))

            sync_tag = " [SYNC]" if bool(getattr(t, "sync_enabled", False)) else ""
            pitch_tag = ""
            st = float(getattr(t, "pitch_semitones", 0.0))
            if abs(st) >= 1e-6:
                pitch_tag = f" [P{st:+.0f}]"

            name = self.scene.addSimpleText((t.name[:18] + sync_tag + pitch_tag)[:26])
            name.setBrush(QBrush(QColor(220, 220, 220)))
            name.setPos(6, y + 8)
            name.setFont(QFont("Consolas", 9))

            self.scene.addLine(0, y, width, y, QPen(QColor(25, 25, 25)))

            for c in t.clips:
                x0 = self.samples_to_x(c.start_sample)
                x1 = self.samples_to_x(c.end_sample())
                rect = QRectF(x0, y + 4, max(10.0, x1 - x0), self.row_h - 8)

                item = ClipItem(self, t, c, i, rect)
                item.setZValue(10)
                self.scene.addItem(item)

                if sel and sel[0] is t and sel[1] is c:
                    item.setSelected(True)

        self._playhead = self.scene.addRect(
            QRectF(0, 0, 2, height),
            QPen(QColor(255, 0, 0, 180)),
            QBrush(QColor(255, 0, 0, 60)),
        )
        self._playhead.setZValue(9999)
        self.update_playhead(int(getattr(self.engine, "ui_sample_pos", 0)))

    def update_playhead(self, sample_pos: int):
        x = self.samples_to_x(int(sample_pos))
        self._playhead.setRect(QRectF(x, 0, 2, self.scene.sceneRect().height()))

    def _row_at_y(self, y: float, track_count: int) -> int:
        row = int((y - 40) // self.row_h)
        if 0 <= row < track_count:
            return row
        return -1

    def _edge_hit(self, item: "ClipItem", scene_x: float) -> Optional[str]:
        r = item.rect()
        x0 = r.x()
        x1 = r.x() + r.width()
        if abs(scene_x - x0) <= self.EDGE_PX:
            return "left"
        if abs(scene_x - x1) <= self.EDGE_PX:
            return "right"
        return None

    def _set_cursor_for_hover(self, scene_pos):
        it = self.scene.itemAt(scene_pos, self.transform())
        if isinstance(it, ClipItem):
            edge = self._edge_hit(it, float(scene_pos.x()))
            if edge in ("left", "right"):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        pos = self.mapToScene(event.pos())
        x, y = float(pos.x()), float(pos.y())

        it = self.itemAt(event.pos())
        if isinstance(it, ClipItem):
            super().mouseDoubleClickEvent(event)
            return

        with self.engine.master_lock:
            tracks = tuple(self.engine.tracks)

        row = self._row_at_y(y, len(tracks))
        if 0 <= row < len(tracks) and event.button() == Qt.MouseButton.LeftButton:
            track = tracks[row]
            self._selected_track = track

            disable_snap = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            start = self.quantize_samples(self.x_to_samples(x), disable_snap=disable_snap)
            length = int(track.nframes)

            track.clips.append(TrackClip(
                start_sample=int(start),
                length_samples=int(length),
                src_offset_samples=0,
                src_len_samples=int(length),
                enabled=True
            ))
            self.engine.invalidate_track_renders(track)
            self.clips_changed.emit()
            self.rebuild()
            return

        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        sp = self.mapToScene(event.pos())

        if not self._drag_active:
            self._set_cursor_for_hover(sp)
            super().mouseMoveEvent(event)
            return

        it = self._drag_item
        if it is None:
            self._drag_active = False
            return

        alt_disable = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        dx = float(sp.x() - self._drag_anchor_scene_x)
        if fine:
            dx *= 0.1

        ds = self.dx_to_samples(dx)

        clip = it.clip
        track = it.track
        sync_on = bool(getattr(track, "sync_enabled", False))

        if self._drag_kind == "move":
            new_start = self._drag_anchor_start + ds
            new_start = self.quantize_samples(new_start, disable_snap=alt_disable)
            clip.start_sample = int(max(0, new_start))

        elif self._drag_kind == "left":
            old_start = int(self._drag_anchor_start)
            old_end = int(old_start + self._drag_anchor_len)

            new_start = old_start + ds
            new_start = self.quantize_samples(new_start, disable_snap=alt_disable)
            new_start = int(max(0, new_start))
            new_start = min(new_start, old_end - 1)

            new_len = int(old_end - new_start)
            shift = int(new_start - old_start)

            clip.start_sample = int(new_start)
            clip.length_samples = int(max(1, new_len))

            if sync_on:
                clip.src_offset_samples = int(self._drag_anchor_off)
                clip.src_len_samples = int(self._drag_anchor_srclen)
            else:
                clip.src_offset_samples = int(max(0, self._drag_anchor_off + shift))
                clip.src_len_samples = int(max(1, clip.length_samples))

        elif self._drag_kind == "right":
            old_start = int(self._drag_anchor_start)
            end = int(old_start + self._drag_anchor_len + ds)
            end = self.quantize_samples(end, disable_snap=alt_disable)
            new_len = int(end - old_start)
            clip.length_samples = int(max(1, new_len))

            if sync_on:
                clip.src_offset_samples = int(self._drag_anchor_off)
                clip.src_len_samples = int(self._drag_anchor_srclen)
            else:
                clip.src_len_samples = int(max(1, clip.length_samples))

        try:
            self.engine.invalidate_clip_render(track, clip)
        except Exception:
            pass

        row_y = 40 + it.row_index * self.row_h
        x0 = self.samples_to_x(clip.start_sample)
        x1 = self.samples_to_x(clip.end_sample())
        it.setRect(QRectF(x0, row_y + 4, max(10.0, x1 - x0), self.row_h - 8))

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        sp = self.mapToScene(event.pos())
        x, y = float(sp.x()), float(sp.y())

        it = self.scene.itemAt(sp, self.transform())

        if isinstance(it, ClipItem) and event.button() == Qt.MouseButton.LeftButton:
            self.scene.clearSelection()
            it.setSelected(True)

            edge = self._edge_hit(it, x)
            if edge == "left":
                self._drag_kind = "left"
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif edge == "right":
                self._drag_kind = "right"
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self._drag_kind = "move"
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

            self._drag_active = True
            self._drag_item = it
            self._drag_anchor_scene_x = x
            self._drag_anchor_start = int(it.clip.start_sample)
            self._drag_anchor_len = int(it.clip.length_samples)
            self._drag_anchor_off = int(it.clip.src_offset_samples)
            self._drag_anchor_srclen = int(it.clip.src_len_samples)

            self._selected_track = it.track

            track_ref = it.track
            clip_ref = it.clip
            QTimer.singleShot(0, lambda tr=track_ref, cl=clip_ref: self.clip_selected.emit(tr, cl))
            return

        with self.engine.master_lock:
            tracks = tuple(self.engine.tracks)

        row = self._row_at_y(y, len(tracks))
        if 0 <= row < len(tracks):
            self._selected_track = tracks[row]
            self.request_rebuild()
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_active:
            self._drag_active = False
            self._drag_item = None
            self._drag_kind = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.clips_changed.emit()
            self.rebuild()
            return
        super().mouseReleaseEvent(event)

    def _selected_clip_items(self) -> List["ClipItem"]:
        return [it for it in self.scene.selectedItems() if isinstance(it, ClipItem)]

    def _copy_selected_clips(self):
        items = self._selected_clip_items()
        if not items:
            return
        with self.engine.master_lock:
            tracks = tuple(self.engine.tracks)

        buf = []
        min_start = None
        for it in items:
            try:
                ti = tracks.index(it.track)
            except ValueError:
                continue
            c = it.clip
            buf.append({
                "track_index": int(ti),
                "start": int(c.start_sample),
                "len": int(c.length_samples),
                "off": int(c.src_offset_samples),
                "srclen": int(c.src_len_samples),
                "enabled": bool(c.enabled),
            })
            min_start = int(c.start_sample) if min_start is None else min(min_start, int(c.start_sample))

        if not buf:
            return

        self._clipboard = buf
        self._clipboard_min_start = int(min_start if min_start is not None else 0)

    def _paste_clips_at_playhead(self, *, disable_snap: bool):
        if not self._clipboard:
            return

        with self.engine.master_lock:
            tracks = tuple(self.engine.tracks)
            playhead = int(getattr(self.engine, "ui_sample_pos", 0))

        base = int(playhead) - int(self._clipboard_min_start)
        base = self.quantize_samples(base, disable_snap=disable_snap)

        created = False
        for d in self._clipboard:
            ti = int(d.get("track_index", -1))
            if ti < 0 or ti >= len(tracks):
                continue
            t = tracks[ti]

            start = int(d.get("start", 0)) + base
            start = self.quantize_samples(start, disable_snap=disable_snap)
            start = max(0, start)

            new_clip = TrackClip(
                start_sample=int(start),
                length_samples=int(max(1, d.get("len", 1))),
                src_offset_samples=int(max(0, d.get("off", 0))),
                src_len_samples=int(max(1, d.get("srclen", d.get("len", 1)))),
                enabled=bool(d.get("enabled", True)),
            )
            t.clips.append(new_clip)
            self.engine.invalidate_track_renders(t)
            created = True

        if created:
            self.clips_changed.emit()
            self.rebuild()

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_F:
            # We call the main window method
            self.window().consolidate_selected_clip()
            return

        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_C:
            self._copy_selected_clips()
            return

        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_V:
            alt_disable = bool(mods & Qt.KeyboardModifier.AltModifier)
            self._paste_clips_at_playhead(disable_snap=alt_disable)
            return

        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            changed = False
            for it in self._selected_clip_items():
                try:
                    it.track.clips.remove(it.clip)
                    self.engine.invalidate_track_renders(it.track)
                    changed = True
                except Exception:
                    pass
            if changed:
                self.clips_changed.emit()
                self.rebuild()
                return

        if key == Qt.Key.Key_M:
            items = self._selected_clip_items()
            if items:
                it = items[0]
                it.clip.enabled = not bool(it.clip.enabled)
                self.engine.invalidate_clip_render(it.track, it.clip)
                self.clips_changed.emit()
                self.rebuild()
                return

        if key == Qt.Key.Key_D and (mods & Qt.KeyboardModifier.ControlModifier):
            items = self._selected_clip_items()
            if items:
                it = items[0]
                c = it.clip
                t = it.track
                new_start = self.quantize_samples(c.end_sample(), disable_snap=False)
                t.clips.append(TrackClip(
                    start_sample=int(new_start),
                    length_samples=int(c.length_samples),
                    src_offset_samples=int(c.src_offset_samples),
                    src_len_samples=int(c.src_len_samples),
                    enabled=bool(c.enabled),
                ))
                self.engine.invalidate_track_renders(t)
                self.clips_changed.emit()
                self.rebuild()
                return

        super().keyPressEvent(event)


class ClipItem(QGraphicsRectItem):
    def __init__(self, view: SequencerView, track: AudioTrack, clip: TrackClip, row_index: int, rect: QRectF):
        super().__init__(rect)
        self.view = view
        self.track = track
        self.clip = clip
        self.row_index = row_index

        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self.text = QGraphicsSimpleTextItem("", self)
        self.text.setBrush(QBrush(QColor(0, 0, 0)))
        self.text.setFont(QFont("Consolas", 8))

    def refresh_label(self):
        tag = "SYNC" if bool(getattr(self.track, "sync_enabled", False)) else ""
        st = float(getattr(self.track, "pitch_semitones", 0.0))
        ptag = f"P{st:+.0f}" if abs(st) >= 1e-6 else ""
        extra = ""
        if tag and ptag:
            extra = f" [{tag}|{ptag}]"
        elif tag:
            extra = f" [{tag}]"
        elif ptag:
            extra = f" [{ptag}]"
        self.text.setText((self.track.name[:20] + extra)[:30])
        r = self.rect()
        self.text.setPos(r.x() + 4, r.y() + 3)

    def hoverMoveEvent(self, event):
        r = self.rect()
        x = float(event.scenePos().x())
        if abs(x - r.x()) <= self.view.EDGE_PX or abs(x - (r.x() + r.width())) <= self.view.EDGE_PX:
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().hoverMoveEvent(event)

    def paint(self, painter, option, widget=None):
        enabled = bool(self.clip.enabled)
        base = QColor(0, 255, 0, 120) if enabled else QColor(120, 120, 120, 120)
        if bool(getattr(self.track, "sync_enabled", False)):
            base = QColor(0, 200, 255, 120) if enabled else QColor(120, 120, 120, 120)
        if self.isSelected():
            base = QColor(0, 255, 255, 170)

        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setBrush(QBrush(base))
        painter.drawRect(self.rect())

        r = self.rect()
        painter.setPen(QPen(QColor(0, 0, 0), 0))
        painter.setBrush(QBrush(QColor(0, 0, 0, 80)))
        painter.drawRect(QRectF(r.x(), r.y(), 3, r.height()))
        painter.drawRect(QRectF(r.x() + r.width() - 3, r.y(), 3, r.height()))

        self.refresh_label()


# =============================================================================
# Main Window
# =============================================================================

class FLStudioGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Audio Engine Pro (FL-ish Sequencer)")
        self.resize(1450, 980)
        self.setStyleSheet("QMainWindow { background-color:#000; }")

        self.engine = AudioEngine()
        self.engine.finished.connect(self.reset_ui)

        self.track_widgets: list[TrackWidget] = []
        self.active_track_widget: TrackWidget | None = None

        self.setup_ui()

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._poll_engine_ui)
        self._ui_timer.start()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        top_bar = QFrame()
        top_bar.setStyleSheet("background:#000; border-bottom:2px solid #222; padding:5px;")
        top_layout = QHBoxLayout(top_bar)

        self.btn_load = QPushButton("LOAD TRACK")
        self.btn_load.clicked.connect(self.add_track)
        self.btn_load.setStyleSheet("background:#900; color:#fff; font-weight:bold; border:1px solid #f00;")

        self.btn_url = QPushButton("URL TRACK")
        self.btn_url.clicked.connect(self.add_url_track)
        self.btn_url.setStyleSheet("background:#600; color:#fff; border:1px solid #c00;")

        self.btn_play = QPushButton("▶ PLAY")
        self.btn_play.clicked.connect(self.toggle_play_pause)
        self.btn_play.setStyleSheet("background:#003300; color:#00ff00; border:1px solid #00ff00;")

        self.btn_stop = QPushButton("■ STOP")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setStyleSheet("background:#220000; color:#ff6666; border:1px solid #aa3333;")
        self.btn_stop.setEnabled(False)

        self.btn_stream = QPushButton("🌐 STREAM")
        self.btn_stream.clicked.connect(self.toggle_stream)
        self.btn_stream.setStyleSheet("background:#001133; color:#00aaff; border:1px solid #00aaff;")

        self.btn_export = QPushButton("💾 EXPORT")
        self.btn_export.clicked.connect(self.export_mixdown_dialog)
        self.btn_export.setStyleSheet("background:#222; color:#fff; border:1px solid #555;")

        top_layout.addWidget(self.btn_export)
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.btn_url)
        top_layout.addWidget(self.btn_play)
        top_layout.addWidget(self.btn_stop)
        top_layout.addWidget(self.btn_stream)

        top_layout.addSpacing(14)
        self.chk_snap = QCheckBox("SNAP")
        self.chk_snap.setChecked(True)
        self.chk_snap.setStyleSheet("color:#ddd;")
        self.cmb_snap = QComboBox()
        self.cmb_snap.addItems(["Off", "1/8", "1/4", "1/2", "Beat", "Bar", "Sec"])
        self.cmb_snap.setCurrentText("1/4")
        self.cmb_snap.setStyleSheet("background:#111; color:#00ff00; border:1px solid #222; padding:2px;")
        self.chk_snap.stateChanged.connect(self._apply_snap_ui)
        self.cmb_snap.currentTextChanged.connect(self._apply_snap_ui)

        hint = QLabel(
            "Tip: Double-click to place. Drag move. Drag edges trim. Shift=fine. Alt=no snap. "
            "Ctrl+C/V copy/paste clips. SYNC makes edge drags stretch to fit. PITCH uses ffmpeg."
        )
        hint.setStyleSheet("color:#666; font-size:11px;")

        top_layout.addWidget(self.chk_snap)
        top_layout.addWidget(self.cmb_snap)
        top_layout.addWidget(hint, 1)

        top_layout.addSpacing(12)
        top_layout.addWidget(QLabel("MASTER BPM"))
        self.master_bpm = QDoubleSpinBox()
        self.master_bpm.setRange(40.0, 240.0)
        self.master_bpm.setSingleStep(1.0)
        self.master_bpm.setDecimals(1)
        self.master_bpm.setValue(float(self.engine.master_bpm))
        self.master_bpm.valueChanged.connect(lambda v: (self.engine.set_master_bpm(float(v)), self.sequencer.rebuild()))
        self.master_bpm.setStyleSheet("background:#111; color:#00ff00; border:1px solid #222; padding:2px;")
        self.master_bpm.setFixedWidth(110)
        top_layout.addWidget(self.master_bpm)

        top_layout.addSpacing(12)
        top_layout.addWidget(QLabel("ZOOM"))
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(20, 300)
        self.zoom.setValue(140)
        self.zoom.setFixedWidth(160)
        self.zoom.valueChanged.connect(lambda v: self.sequencer.set_zoom_px_per_sec(float(v)))
        self.zoom.setStyleSheet(
            "QSlider::groove:horizontal { height:6px; background:#111; border:1px solid #222; }"
            "QSlider::handle:horizontal { width:10px; background:#00ff00; margin:-6px 0; }"
        )
        top_layout.addWidget(self.zoom)

        main_layout.addWidget(top_bar)

        # ---------- RESIZABLE WORKSPACE (Splitters) ----------
        # Horizontal: Filters | (Sequencer+Tracks) | Parameters
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setChildrenCollapsible(False)
        h_split.setHandleWidth(6)
        h_split.setStyleSheet("QSplitter::handle { background:#111; }")

        # ========== LEFT: Filters ==========
        mixer_box = QGroupBox("Filters")
        mixer_box.setMinimumWidth(180)
        mixer_box.setStyleSheet("background:#000; color:#eee; font-weight:bold; border:1px solid #222;")
        mixer_layout = QVBoxLayout(mixer_box)

        self.fx_slots = []
        for i in range(8):
            slot = FXSlot(i)
            slot.selector.currentIndexChanged.connect(lambda _=None: self.refresh_logic(rebuild_param_ui=True))
            slot.active.clicked.connect(lambda _=None: self.refresh_logic(rebuild_param_ui=True))
            mixer_layout.addWidget(slot)
            self.fx_slots.append(slot)

        mixer_layout.addStretch(1)
        h_split.addWidget(mixer_box)

        # ========== MIDDLE: Sequencer + Tracks Dock (vertical splitter) ==========
        mid_split = QSplitter(Qt.Orientation.Vertical)
        mid_split.setChildrenCollapsible(False)
        mid_split.setHandleWidth(6)
        mid_split.setStyleSheet("QSplitter::handle { background:#111; }")

        # Sequencer container
        seq_box = QGroupBox("Sequencer (FL-ish)")
        seq_box.setMinimumHeight(240)
        seq_box.setStyleSheet("background:#000; color:#aaa; font-weight:bold; border:1px solid #222;")
        seq_layout = QVBoxLayout(seq_box)

        self.sequencer = SequencerView(self.engine)
        self.sequencer.clip_selected.connect(self.on_clip_selected)
        self.sequencer.clips_changed.connect(self._arrangement_changed)
        seq_layout.addWidget(self.sequencer)

        mid_split.addWidget(seq_box)

        # Tracks dock container
        dock_box = QGroupBox("Audio Dock (Tracks)")
        dock_box.setMinimumHeight(140)
        dock_box.setStyleSheet("background:#000; color:#777; font-weight:bold; border:1px solid #222;")
        self.dock_layout = QVBoxLayout(dock_box)

        self.dock_scroll = QScrollArea()
        self.dock_scroll.setWidgetResizable(True)
        self.dock_scroll.setStyleSheet("background:#000; border:none;")
        self.dock_container = QWidget()
        self.dock_inner = QVBoxLayout(self.dock_container)
        self.dock_inner.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.dock_scroll.setWidget(self.dock_container)
        self.dock_layout.addWidget(self.dock_scroll)

        mid_split.addWidget(dock_box)

        # Give sequencer more space than dock by default
        mid_split.setStretchFactor(0, 4)
        mid_split.setStretchFactor(1, 1)

        h_split.addWidget(mid_split)

        # ========== RIGHT: Parameters ==========
        inspector_box = QGroupBox("Parameters")
        inspector_box.setMinimumWidth(260)
        inspector_box.setStyleSheet("background:#000; color:#00ff00; font-weight:bold; border:1px solid #222;")

        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setStyleSheet("background:#000; border:none;")
        self.param_container = QWidget()
        self.param_layout = QVBoxLayout(self.param_container)
        self.param_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.param_scroll.setWidget(self.param_container)

        ins_layout = QVBoxLayout(inspector_box)
        ins_layout.addWidget(self.param_scroll)

        h_split.addWidget(inspector_box)

        # Stretch: Filters small, middle big, parameters medium
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 4)
        h_split.setStretchFactor(2, 1)

        # Optional: initial sizes (pixels). Tune to taste.
        h_split.setSizes([220, 900, 330])

        main_layout.addWidget(h_split, 1)
        # ---------- END RESIZABLE WORKSPACE ----------

        self.meter_l = QProgressBar()
        self.meter_r = QProgressBar()
        for m in (self.meter_l, self.meter_r):
            m.setRange(0, 1000)
            m.setTextVisible(False)
            m.setFixedHeight(8)
            m.setStyleSheet(
                "QProgressBar::chunk { background-color:#00ff00; } "
                "QProgressBar { background:#111; border:1px solid #222; }"
            )
        meter_layout = QHBoxLayout()
        meter_layout.addWidget(QLabel("L"))
        meter_layout.addWidget(self.meter_l)
        meter_layout.addWidget(self.meter_r)
        meter_layout.addWidget(QLabel("R"))
        main_layout.addLayout(meter_layout)

        transport = QFrame()
        transport.setStyleSheet("background:#000; border-top:1px solid #222; padding:6px;")
        tl = QHBoxLayout(transport)

        self.lbl_time = QLabel("00:00.000 / 00:00.000")
        self.lbl_time.setStyleSheet("color:#00ff00; font-family:monospace;")

        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub.setEnabled(False)
        self.scrub.setStyleSheet(
            "QSlider::groove:horizontal { height:6px; background:#111; border:1px solid #222; }"
            "QSlider::handle:horizontal { width:12px; background:#00ff00; margin:-6px 0; }"
        )

        self._scrubbing = False
        self.scrub.sliderPressed.connect(lambda: setattr(self, "_scrubbing", True))
        self.scrub.sliderReleased.connect(self.on_scrub_released)
        self.scrub.sliderMoved.connect(self.on_scrub_moved)

        tl.addWidget(self.lbl_time)
        tl.addWidget(self.scrub, 1)
        main_layout.addWidget(transport)

        self._apply_snap_ui()
        self._update_transport_range()

    def consolidate_selected_clip(self):
        items = self.sequencer._selected_clip_items()
        if not items:
            return

        it = items[0]
        original_track = it.track
        clip = it.clip

        # 1. Render the audio buffer
        rendered_buffer = self.engine.render_clip_to_buffer(original_track, clip)

        # 2. Convert to pydub AudioSegment
        rendered_seg = AudioSegment(
            (rendered_buffer * 32767).astype(np.int16).tobytes(),
            frame_rate=self.engine.master_sr,
            sample_width=2,
            channels=2
        )

        # 3. Use a safe temp file approach for Windows
        temp_path = os.path.join(tempfile.gettempdir(), f"flatten_{id(clip)}.wav")
        try:
            # Export to temp path
            rendered_seg.export(temp_path, format="wav")

            # Initialize new track
            new_track = AudioTrack(temp_path, target_sr=self.engine.master_sr)
            new_track.name = f"Flattened_{original_track.name}"

            # 4. Copy settings and logic
            new_track.volume = original_track.volume
            new_track.fx_selections = list(original_track.fx_selections)
            new_track.fx_enabled = list(original_track.fx_enabled)
            new_track.params = [dict(p) for p in original_track.params]
            new_track.sync_enabled = False
            new_track.pitch_semitones = 0.0

            with self.engine.master_lock:
                self.engine.tracks.append(new_track)

            # Add to UI
            tw = TrackWidget(new_track, self.engine)
            tw.selected.connect(self.on_track_selected)
            tw.reorder.connect(self.move_track)
            tw.removed.connect(self.remove_track)
            tw.track_changed.connect(lambda _=None: self.sequencer.rebuild())

            self.track_widgets.append(tw)
            self.dock_inner.addWidget(tw)

            # Match original timing
            new_track.clips = [TrackClip(
                start_sample=clip.start_sample,
                length_samples=len(rendered_buffer),
                src_offset_samples=0,
                src_len_samples=len(rendered_buffer)
            )]

            self.sequencer.rebuild()

        finally:
            # We wrap this in a try/except because Windows might still hold onto it
            # for a few milliseconds for background indexing
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Cleanup warning: {e}")
    def _apply_snap_ui(self):
        enabled = self.chk_snap.isChecked()
        mode = self.cmb_snap.currentText()
        mode_map = {"Off": "off", "1/8": "1/8", "1/4": "1/4", "1/2": "1/2", "Beat": "beat", "Bar": "bar", "Sec": "sec"}
        self.sequencer.set_snap(enabled, mode_map.get(mode, "1/4"))

    def on_clip_selected(self, track: AudioTrack, clip: TrackClip):
        for tw in self.track_widgets:
            if tw.track is track:
                self.on_track_selected(tw)
                break

    def _arrangement_changed(self):
        self._update_transport_range()

        # Warm SYNC renders after arrangement edits (UI thread).
        try:
            with self.engine.master_lock:
                tracks = list(self.engine.tracks)
            self.engine.render_cache.pre_render_all(tracks)
        except Exception as e:
            print("[GUI] pre_render_all(arrangement) failed:", repr(e))

        self.sequencer.request_rebuild()
    def _poll_engine_ui(self):
        eng = self.engine
        l = float(getattr(eng, "ui_meter_l", 0.0))
        r = float(getattr(eng, "ui_meter_r", 0.0))
        self.update_meters(l, r)

        sp = int(getattr(eng, "ui_sample_pos", 0))
        self.sequencer.update_playhead(sp)

        if self.scrub.isEnabled() and not getattr(self, "_scrubbing", False):
            self.on_engine_position(sp)

    def _song_length_samples(self) -> int:
        return int(self.engine.song_length_samples())

    def _update_transport_range(self):
        n = self._song_length_samples()
        if n <= 0:
            self.scrub.setEnabled(False)
            self.scrub.setRange(0, 1000)
            self.lbl_time.setText("00:00.000 / 00:00.000")
            return

        self.scrub.setEnabled(True)
        self.scrub.setRange(0, n)
        total_sec = n / float(self.engine.master_sr)
        cur_sec = float(getattr(self.engine, "ui_sample_pos", 0)) / float(self.engine.master_sr)
        self.lbl_time.setText(f"{_format_time(cur_sec)} / {_format_time(total_sec)}")

    def on_engine_position(self, sample_pos: int):
        if not self.scrub.isEnabled():
            return

        n = self.scrub.maximum()
        total_sec = n / float(self.engine.master_sr)
        cur_sec = float(sample_pos) / float(self.engine.master_sr)

        self.lbl_time.setText(f"{_format_time(cur_sec)} / {_format_time(total_sec)}")

        if not getattr(self, "_scrubbing", False):
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(min(max(sample_pos, 0), n)))
            self.scrub.blockSignals(False)

    def on_scrub_moved(self, v: int):
        n = max(1, self.scrub.maximum())
        total_sec = n / float(self.engine.master_sr)
        cur_sec = float(v) / float(self.engine.master_sr)
        self.lbl_time.setText(f"{_format_time(cur_sec)} / {_format_time(total_sec)}")
        self.sequencer.update_playhead(int(v))

    def on_scrub_released(self):
        self._scrubbing = False
        v = int(self.scrub.value())
        self.engine.seek_samples(v)
        self.sequencer.update_playhead(int(v))

    def update_meters(self, l, r):
        self.meter_l.setValue(int(l * 1000))
        self.meter_r.setValue(int(r * 1000))

    def add_track(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Audio Track", "", "Audio (*.wav *.mp3 *.flac *.aac *.ogg)")
        if f:
            self._init_track(f, is_url=False)

    def add_url_track(self):
        url, ok = QInputDialog.getText(self, "Load URL", "Paste direct audio URL OR YouTube URL:")
        if ok and url:
            self._init_track(url, is_url=True)

    def _init_track(self, path, is_url):
        try:
            track = AudioTrack(path, is_url=is_url, target_sr=self.engine.master_sr, auto_bpm=True)

            with self.engine.master_lock:
                self.engine.tracks.append(track)

            tw = TrackWidget(track, self.engine)
            tw.selected.connect(self.on_track_selected)
            tw.reorder.connect(self.move_track)
            tw.removed.connect(self.remove_track)
            tw.track_changed.connect(lambda _=None: self.sequencer.rebuild())

            self.track_widgets.append(tw)
            self.dock_inner.addWidget(tw)

            if not self.active_track_widget:
                self.on_track_selected(tw)

            self.refresh_logic(rebuild_param_ui=True)
            self._update_transport_range()
            self.sequencer.rebuild()

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            print(f"Load Error: {e}")

    def remove_track(self, widget: TrackWidget):
        with self.engine.master_lock:
            if widget.track in self.engine.tracks:
                self.engine.tracks.remove(widget.track)
        try:
            self.engine.invalidate_track_renders(widget.track)
        except Exception:
            pass
        self.track_widgets.remove(widget)
        widget.deleteLater()
        if self.active_track_widget == widget:
            self.active_track_widget = None
            self.update_param_ui()
        self._update_transport_range()
        self.sequencer.rebuild()

    def move_track(self, widget: TrackWidget, direction: int):
        idx = self.track_widgets.index(widget)
        new_idx = idx + direction
        if 0 <= new_idx < len(self.track_widgets):
            with self.engine.master_lock:
                self.engine.tracks.insert(new_idx, self.engine.tracks.pop(idx))
            self.track_widgets.insert(new_idx, self.track_widgets.pop(idx))

            for i in reversed(range(self.dock_inner.count())):
                item = self.dock_inner.itemAt(i)
                if item.widget():
                    item.widget().setParent(None)
            for w in self.track_widgets:
                self.dock_inner.addWidget(w)

            self.sequencer.rebuild()

    def on_track_selected(self, widget: TrackWidget):
        if self.engine.is_streaming:
            return

        if self.active_track_widget:
            self.active_track_widget.set_active(False)

        self.active_track_widget = widget
        widget.set_active(True)
        self.sequencer.set_selected_track(widget.track)

        for i, slot in enumerate(self.fx_slots):
            slot.selector.blockSignals(True)
            slot.selector.setCurrentText(widget.track.fx_selections[i])
            slot.active.setChecked(widget.track.fx_enabled[i])
            slot.selector.blockSignals(False)

        self.refresh_logic(rebuild_param_ui=True)

    # --------- export ---------

    def export_mixdown_dialog(self):
        if self.engine.is_streaming:
            QMessageBox.information(self, "Export", "Stop streaming before exporting.")
            return
        if self.engine.is_playing:
            QMessageBox.information(self, "Export", "Stop playback before exporting.")
            return
        if not self.engine.tracks:
            QMessageBox.information(self, "Export", "No tracks loaded.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Mixdown", "", "WAV (*.wav);;MP3 (*.mp3)")
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        fmt = "mp3" if ext == ".mp3" else "wav"

        try:
            total_samples = int(self._song_length_samples())
        except Exception:
            total_samples = None

        for b in (self.btn_export, self.btn_load, self.btn_url, self.btn_play, self.btn_stop, self.btn_stream):
            b.setEnabled(False)

        dlg = ExportProgressDialog(self)
        dlg.set_progress(0, 0, max(1, total_samples or 1))

        thread = QThread(self)
        worker = ExportWorker(self.engine, path, fmt, total_samples)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(dlg.set_progress)

        def _finish(success: bool, msg: str):
            try:
                dlg.close()
            except Exception:
                pass

            for b in (self.btn_export, self.btn_load, self.btn_url, self.btn_play, self.btn_stop, self.btn_stream):
                b.setEnabled(True)

            thread.quit()
            thread.wait(2000)
            worker.deleteLater()
            thread.deleteLater()

            if success:
                QMessageBox.information(self, "Export", msg)
            else:
                QMessageBox.warning(self, "Export", msg)

        worker.done.connect(_finish)
        dlg.canceled.connect(worker.request_cancel)
        dlg.rejected.connect(worker.request_cancel)

        thread.start()
        dlg.exec()

    # --------- filters / params ---------

    def refresh_logic(self, rebuild_param_ui: bool = True):
        pipeline = []
        sr = int(self.engine.master_sr)

        for i, slot in enumerate(self.fx_slots):
            name = slot.selector.currentText()
            active = slot.active.isChecked()

            if self.engine.is_streaming:
                self.engine.live_fx_selections[i] = name
                if name != "None" and active:
                    raw = {k: v for k, v in self.engine.live_params[i].items() if v is not None}
                    p = sanitize_filter_params(name, raw, sr=sr)
                    flt = build_filter(name, **p)
                    if name.strip().lower() in {"timestretch", "wsola", "speed"}:
                        flt = FixedBlockAdapter(flt, channels=2)
                    pipeline.append(flt)
            elif self.active_track_widget:
                t = self.active_track_widget.track
                t.fx_selections[i] = name
                t.fx_enabled[i] = active
                if name != "None" and active:
                    raw = {k: v for k, v in t.params[i].items() if v is not None}
                    p = sanitize_filter_params(name, raw, sr=sr)
                    flt = build_filter(name, **p)
                    if name.strip().lower() in {"timestretch", "wsola", "speed"}:
                        flt = FixedBlockAdapter(flt, channels=2)
                    pipeline.append(flt)

        if self.engine.is_streaming:
            with self.engine.filter_lock:
                self.engine.live_pipeline = pipeline
        elif self.active_track_widget:
            with self.active_track_widget.track.filter_lock:
                self.active_track_widget.track.active_filters = pipeline

        if rebuild_param_ui:
            self.update_param_ui()

    def update_param_ui(self):
        while self.param_layout.count():
            it = self.param_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        source_params = None
        source_fx = None
        if self.engine.is_streaming:
            source_params = self.engine.live_params
            source_fx = self.engine.live_fx_selections
        elif self.active_track_widget:
            source_params = self.active_track_widget.track.params
            source_fx = self.active_track_widget.track.fx_selections

        if not source_fx:
            return

        filt_help = available_filters()

        for i, name in enumerate(source_fx):
            if name == "None":
                continue

            group = QFrame()
            group.setStyleSheet("background:#080808; border:1px solid #222; margin-bottom:8px; border-radius:4px;")
            vbox = QVBoxLayout(group)

            hdr = QLabel(f"SLOT {i + 1}: {name.upper()}")
            hdr.setStyleSheet("color:#00ff00; font-weight:bold; border:none;")
            vbox.addWidget(hdr)

            help_text = filt_help.get(name.strip().lower(), "")
            found = re.findall(r"(\w+)\(([^)]+)\)", help_text)
            if not found:
                found = []

            for p_name, p_def in found:
                num = re.search(r"[-+]?\d*\.\d+|\d+", p_def)
                clean = num.group(0) if num else None

                if p_name not in source_params[i]:
                    if clean is not None:
                        try:
                            source_params[i][p_name] = float(clean)
                        except Exception:
                            source_params[i][p_name] = 0.0

                cur = source_params[i].get(p_name, 0.0)

                pw = QWidget()
                pl = QVBoxLayout(pw)
                pl.setContentsMargins(5, 5, 5, 5)
                lh = QHBoxLayout()

                pt = QLabel(p_name.replace("_", " ").title())
                pt.setStyleSheet("color:#888; border:none;")
                pv = QLabel(str(cur))
                pv.setStyleSheet("color:#00ff00; font-family:monospace; border:none;")

                lh.addWidget(pt)
                lh.addStretch()
                lh.addWidget(pv)

                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 200)
                slider.setValue(self.map_val(p_name, float(cur)))
                slider.valueChanged.connect(self._make_slider_handler(i, p_name, pv))

                pl.addLayout(lh)
                pl.addWidget(slider)
                vbox.addWidget(pw)

            self.param_layout.addWidget(group)

        self._sanitize_all_slots()
        self.refresh_logic(rebuild_param_ui=False)

    def _make_slider_handler(self, slot_idx: int, p_name: str, value_label: QLabel):
        def _handler(raw: int):
            self.handle_change(slot_idx, p_name, raw, value_label)
        return _handler

    def _sanitize_all_slots(self):
        sr = int(self.engine.master_sr)

        if self.engine.is_streaming:
            params_list = self.engine.live_params
            fx_list = self.engine.live_fx_selections
        elif self.active_track_widget:
            params_list = self.active_track_widget.track.params
            fx_list = self.active_track_widget.track.fx_selections
        else:
            return

        for i, fname in enumerate(fx_list):
            if fname == "None":
                continue
            params_list[i] = sanitize_filter_params(fname, params_list[i], sr=sr)

    def map_val(self, p, v):
        pl = (p or "").lower()
        if "db" in pl:
            return int((float(v) + 60) * 2)
        if any(x in pl for x in ["freq", "cutoff", "hz", "f0", "low", "high"]):
            return int((np.log10(max(1.0, float(v))) - 1) * 66)
        return int(float(v) * 20)

    def handle_change(self, s_idx, p, raw, lbl):
        pl = (p or "").lower()

        if "db" in pl:
            v = round((raw * 0.5) - 60, 1)
        elif any(x in pl for x in ["freq", "cutoff", "hz", "f0", "low", "high"]):
            v = round(10 ** (1 + (raw / 66)), 0)
        elif "slope" in pl or "q" in pl or p == "r":
            v = round(raw * 0.05 + 0.01, 2)
        else:
            v = round(raw * 0.1, 2)

        lbl.setText(str(v))

        if self.engine.is_streaming:
            self.engine.live_params[s_idx][p] = v
            fname = self.engine.live_fx_selections[s_idx]
            self.engine.live_params[s_idx] = sanitize_filter_params(fname, self.engine.live_params[s_idx], sr=self.engine.master_sr)
        else:
            if self.active_track_widget:
                self.active_track_widget.track.params[s_idx][p] = v
                fname = self.active_track_widget.track.fx_selections[s_idx]
                self.active_track_widget.track.params[s_idx] = sanitize_filter_params(fname, self.active_track_widget.track.params[s_idx], sr=self.engine.master_sr)

        self.refresh_logic(rebuild_param_ui=False)

    # --------- PLAY/PAUSE/STOP ---------

    def toggle_play_pause(self):
        if self.engine.is_streaming:
            QMessageBox.information(self, "Playback", "Stop streaming before playback.")
            return

        if not self.engine.tracks:
            QMessageBox.information(self, "No tracks", "Load a track first.")
            return

        if not self.engine.is_playing:
            start = int(getattr(self.engine, "ui_sample_pos", 0))
            self.engine.seek_samples(start)

            self.btn_play.setText("⏸ PAUSE")
            self.btn_stop.setEnabled(True)
            self.btn_load.setEnabled(False)
            self.btn_url.setEnabled(False)
            self.btn_stream.setEnabled(False)

            threading.Thread(target=self.engine.play_all, daemon=True).start()
            return

        self.engine.toggle_pause()
        self.btn_play.setText("▶ RESUME" if self.engine.is_paused else "⏸ PAUSE")

    def stop_playback(self):
        if not self.engine.is_playing:
            return
        self.engine._stop_event.set()
        self.engine.resume()

    def toggle_stream(self):
        if self.engine.is_streaming:
            self.engine._stop_event.set()
            self.btn_load.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_url.setEnabled(True)
            self.btn_stream.setText("🌐 STREAM")
            return

        if self.engine.is_playing:
            QMessageBox.information(self, "Stream", "Stop playback before streaming.")
            return

        diag = StreamConfigDialog(self)
        if diag.exec():
            self.btn_load.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.btn_url.setEnabled(False)
            self.btn_stream.setText("■ STOP")

            for slot in self.fx_slots:
                slot.selector.setCurrentText("None")
            self.refresh_logic(rebuild_param_ui=True)

            threading.Thread(
                target=self.engine.stream_live,
                args=(diag.in_dev.currentData(), diag.out_dev.currentData()),
                kwargs={"use_wasapi_loopback": bool(diag.wasapi.isChecked())},
                daemon=True
            ).start()

    def reset_ui(self):
        self.btn_play.setText("▶ PLAY")
        self.btn_stop.setEnabled(False)
        self.btn_stream.setText("🌐 STREAM")
        self.btn_load.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_url.setEnabled(True)
        self.btn_stream.setEnabled(True)

        self._update_transport_range()
        self.sequencer.rebuild()


class StreamConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stream Config")
        self.setFixedWidth(420)
        self.setStyleSheet("background:#111; color:#fff;")

        l = QFormLayout(self)

        self.in_dev = QComboBox()
        self.out_dev = QComboBox()

        devs = sd.query_devices()
        for i, d in enumerate(devs):
            name = d.get("name", f"Device {i}")
            if d.get("max_input_channels", 0) > 0:
                self.in_dev.addItem(name, i)
            if d.get("max_output_channels", 0) > 0:
                self.out_dev.addItem(name, i)

        self.wasapi = QCheckBox("WASAPI Loopback (Windows)")

        l.addRow("Input:", self.in_dev)
        l.addRow("Output:", self.out_dev)
        l.addRow(self.wasapi)

        self.btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        l.addRow(self.btns)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = FLStudioGUI()
    screen = QGuiApplication.primaryScreen()
    avail = screen.availableGeometry()  # excludes taskbar
    w = min(gui.width(), avail.width())
    h = min(gui.height(), avail.height())
    gui.resize(w, h)
    gui.move(avail.left(), avail.top())
    gui.show()
    sys.exit(app.exec())
