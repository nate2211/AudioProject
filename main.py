# main.py — Audio pipeline CLI (YouTube links supported)
# Robust WASAPI loopback with jitter buffer + padding + crossfade + smoothed drift correction.
# Auto-recorder restart (COM-initialized worker) for MF hiccups. YouTube via yt-dlp (ffmpeg on PATH).

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import mimetypes
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import threading

from filters import available_filters as AF_AVAILABLE, build_filter as AF_BUILD, AudioFilter
import warps  # noqa: F401
import clarity
log = logging.getLogger("audiogen")

# ---------------- Small DSP helpers ----------------

def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))

def _upmix_mono_to_stereo(buf: np.ndarray, out_ch: int) -> np.ndarray:
    if out_ch == 1:
        return buf
    return np.tile(buf, (1, out_ch))

def _downmix_to_n(buf: np.ndarray, out_ch: int) -> np.ndarray:
    if buf.shape[1] == out_ch:
        return buf
    if out_ch == 1:
        return buf.mean(axis=1, keepdims=True, dtype=np.float32)
    return buf[:, :out_ch]

def _hard_clip_inplace(x: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> None:
    np.clip(x, lo, hi, out=x)

# ---- Click-safe helpers ----
class DCBlocker:
    """
    Simple one-pole DC blocker per channel:
        y[n] = x[n] - x[n-1] + a * y[n-1]
    where a ~ exp(-2*pi*fc/sr). Keeps state across calls and channels.
    """
    def __init__(self, cutoff_hz: float = 20.0):
        self.cut = float(cutoff_hz)
        self._alpha: Optional[float] = None
        self._y1: Optional[np.ndarray] = None  # (1, C) last y
        self._x1: Optional[np.ndarray] = None  # (1, C) last x
        self._C: Optional[int] = None
        self._sr: Optional[int] = None

    def _ensure(self, C: int, sr: int) -> None:
        if self._C != C or self._sr != sr or self._alpha is None:
            self._C, self._sr = C, sr
            self._alpha = float(np.exp(-2.0 * np.pi * self.cut / float(sr)))
            self._y1 = np.zeros((1, C), dtype=np.float32)
            self._x1 = np.zeros((1, C), dtype=np.float32)

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        # x: (N, C) float32
        N, C = x.shape
        self._ensure(C, sr)
        a = self._alpha  # type: ignore
        y = np.empty_like(x, dtype=np.float32)
        # first sample uses stored state
        y0 = x[0:1, :] - self._x1 + a * self._y1  # type: ignore
        y[0:1, :] = y0
        if N > 1:
            # vectorized running filter:
            # y[i] = (x[i]-x[i-1]) + a*y[i-1]
            d = x[1:, :] - x[:-1, :]
            acc = y0
            for i in range(d.shape[0]):
                acc = d[i:i+1, :] + a * acc  # type: ignore
                y[i+1:i+2, :] = acc
        # update state
        self._y1 = y[-1:, :].copy()
        self._x1 = x[-1:, :].copy()
        return y

def _add_tpdf_dither_inplace(x: np.ndarray, level: float = 1e-5) -> None:
    # ~ -100 dBFS TPDF dither; helps avoid denormals and faint tones with BT stacks/codecs
    n = x.shape[0] * x.shape[1]
    # generate two independent uniform noises and subtract
    d = (np.random.rand(n).astype(np.float32) - np.random.rand(n).astype(np.float32)) * level
    x += d.reshape(x.shape)

# ---------------- Logging ----------------

def setup_logging(verbosity: int = 0) -> None:
    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(verbosity, 2)]
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

# ---------------- File fetcher ----------------

class FileFetcher:
    """Fetch bytes from http(s) / file:// / local path with a tiny, safe cache."""
    def __init__(self, cache_dir: Optional[Path] = None, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "audiogen_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "audiogen/1.0 (+https://local)"})

    def _cache_key(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{h}.bin"

    def fetch(self, src: str) -> Tuple[bytes, Optional[str]]:
        parsed = urlparse(src)
        scheme = (parsed.scheme or "").lower()
        if scheme in ("http", "https"):
            key = self._cache_key(src)
            if key.exists():
                try:
                    raw = key.read_bytes()
                    ctype = mimetypes.guess_type(src)[0]
                    log.info("Cache hit: %s", key.name)
                    return raw, ctype
                except Exception:
                    pass
            log.info("Fetching: %s", src)
            r = self._session.get(src, timeout=self.timeout, stream=True)
            r.raise_for_status()
            raw = r.content
            try:
                key.write_bytes(raw)
            except Exception:
                pass
            return raw, r.headers.get("Content-Type")
        if scheme == "file":
            local_path = unquote(parsed.path)
            if os.name == "nt" and local_path.startswith("/"):
                local_path = local_path[1:]
            p = Path(local_path)
            raw = p.read_bytes()
            return raw, mimetypes.guess_type(p.name)[0]
        if scheme == "":
            p = Path(src)
            raw = p.read_bytes()
            return raw, mimetypes.guess_type(p.name)[0]
        raise ValueError(f"Unsupported URL scheme: {scheme}")

# ---------------- YouTube helpers ----------------

_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}

def is_youtube_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        return host.lower() in _YT_HOSTS
    except Exception:
        return False

def fetch_youtube_wav(url: str, ffmpeg_location: Optional[str] = None) -> Tuple[str, str]:
    """Download a YouTube video's audio as WAV using yt-dlp + ffmpeg. Returns (wav_path, temp_dir)."""
    try:
        import yt_dlp  # type: ignore
    except Exception as e:
        raise RuntimeError("yt-dlp is required for YouTube URLs. Install with `pip install yt-dlp`.") from e

    temp_dir = tempfile.mkdtemp(prefix="audiogen_yt_")
    outtmpl = str(Path(temp_dir) / "%(id)s.%(ext)s")
    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}],
        "prefer_ffmpeg": True,
    }
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid_id = info.get("id")
        if not vid_id:
            raise RuntimeError("Failed to resolve video id from YouTube URL")
        wav_path = os.path.join(temp_dir, f"{vid_id}.wav")
        if not os.path.exists(wav_path):
            cand = next((str(p) for p in Path(temp_dir).glob("*.wav")), None)
            if not cand:
                raise RuntimeError("yt-dlp did not produce a WAV file. Ensure ffmpeg is installed.")
            wav_path = cand
        return wav_path, temp_dir

# ---------------- CLI helpers ----------------

def _coerce(v: str) -> Any:
    if v.isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
    return v

def _parse_kv_pairs(pairs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not pairs:
        return out
    for p in pairs:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = _coerce(v.strip())
    return out

def _parse_pipeline(spec: Optional[str], fallback: Optional[str]) -> List[str]:
    if spec:
        stages = [s.strip().lower() for s in spec.split("|") if s.strip()]
        if not stages:
            raise ValueError("Empty --pipeline. Example: gain|lowpass")
        return stages
    if fallback:
        return [fallback.strip().lower()]
    raise ValueError("Provide --pipeline 'f1|f2|...' or --filter NAME.")

def _split_stage_extras(stages: List[str], raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Unprefixed/all apply to all; name.key; index.key (0-based)."""
    stage_extras = [dict() for _ in stages]
    global_extras: Dict[str, Any] = {}
    by_name: Dict[str, Dict[str, Any]] = {}
    by_idx: Dict[int, Dict[str, Any]] = {}
    for k, v in raw.items():
        if "." not in k:
            global_extras[k] = v
            continue
        prefix, key = k.split(".", 1)
        prefix = prefix.strip().lower()
        key = key.strip()
        if prefix == "all":
            global_extras[key] = v
        elif prefix.isdigit():
            i = int(prefix)
            if 0 <= i < len(stages):
                by_idx.setdefault(i, {})[key] = v
        else:
            by_name.setdefault(prefix, {})[key] = v
    for i, name in enumerate(stages):
        merged: Dict[str, Any] = {}
        merged.update(global_extras)
        if name in by_name:
            merged.update(by_name[name])
        if i in by_idx:
            merged.update(by_idx[i])
        stage_extras[i] = merged
    return stage_extras

# ---------------- Pipeline execution (offline) ----------------

BLOCK = 4096 * 4

def _run_pipeline_stream(reader: sf.SoundFile, stages: List[str], stage_extras: List[Dict[str, Any]], *,
                         sink: Optional[sf.SoundFile] = None) -> Tuple[int, int]:
    sr = reader.samplerate
    filters: List[AudioFilter] = [AF_BUILD(name, **stage_extras[i]) for i, name in enumerate(stages)]
    total = len(reader)
    processed = 0
    while processed < total:
        n = min(BLOCK, total - processed)
        x = reader.read(n, dtype="float32", always_2d=True)
        y = x
        for f in filters:
            y = f.process(y, sr)
        if sink is not None:
            sink.write(y)
        processed += n
    for f in filters:
        tail = f.flush()
        if tail is not None and tail.size:
            if sink is not None:
                sink.write(tail.astype(np.float32))
    return total, processed

# ---------------- Commands ----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audio pipeline (YouTube + URL/local)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List filters").set_defaults(func=cmd_list)

    ldp = sub.add_parser("list-devices", help="List available audio devices")
    ldp.set_defaults(func=cmd_list_devices)

    # ---- run ----
    rp = sub.add_parser("run", help="Run one or more filters as a pipeline")
    rp.add_argument("--url", required=True, help="Input URL (http(s), file path, or YouTube)")
    rp.add_argument("--out", type=Path, required=True, help="Output WAV path")
    rp.add_argument("--pipeline", help="Pipe filters as 'f1|f2|f3'")
    rp.add_argument("--filter", choices=list(AF_AVAILABLE().keys()), help="(Deprecated) single filter")
    rp.add_argument("--extra", nargs="*", help="Extra args like key=val, all.key=val, <name>.key=val, <idx>.key=val")
    rp.add_argument("--start", type=float, default=0.0, help="Start time (sec)")
    rp.add_argument("--duration", type=float, help="Duration (sec)")
    rp.add_argument("--ffmpeg-location", dest="ffmpeg_location", help="Path to ffmpeg/ffprobe for yt-dlp")
    rp.set_defaults(func=cmd_run)

    # ---- bench ----
    bp = sub.add_parser("bench", help="Micro-benchmark a pipeline")
    bp.add_argument("--url", required=True, help="Input URL (http(s), file path, or YouTube)")
    bp.add_argument("--pipeline", help="Pipe filters as 'f1|f2|f3'")
    bp.add_argument("--filter", choices=list(AF_AVAILABLE().keys()))
    bp.add_argument("--extra", nargs="*")
    bp.add_argument("--runs", type=int, default=5)
    bp.add_argument("--ffmpeg-location", dest="ffmpeg_location")
    bp.set_defaults(func=cmd_bench)

    # ---- stream ----
    sp = sub.add_parser("stream", help="Run a pipeline on live system audio")
    sp.add_argument("--in-device", type=str, required=True, help="Name or index of input device (e.g., 'CABLE Output')")
    sp.add_argument("--out-device", type=str, required=True, help="Name or index of output device (e.g., 'Speakers')")
    sp.add_argument("--samplerate", type=int, default=48000, help="Sample rate to run at (must match devices)")
    sp.add_argument("--blocksize", type=int, default=1024, help="Processing block size (latency vs. stability)")
    sp.add_argument("--in-channels", type=int, default=2, help="Channels to capture (1=mono, 2=stereo)")
    sp.add_argument("--out-channels", type=int, default=2, help="Channels to render (1=mono, 2=stereo)")
    sp.add_argument("--latency", type=str, default="low", help="Stream latency ('low','high') or seconds, e.g., 0.02")
    sp.add_argument("--headroom-db", type=float, default=3.0, help="Linear headroom applied before output (dB)")
    sp.add_argument("--hard-limit", action="store_true", help="Apply hard limiter/clipper at [-1,1] after headroom")
    sp.add_argument("--pipeline", help="Pipe filters as 'f1|f2|f3'")
    sp.add_argument("--filter", choices=list(AF_AVAILABLE().keys()), help="(Deprecated) Single filter")
    sp.add_argument("--extra", nargs="*")
    sp.add_argument("--wasapi-loopback", action="store_true",
                    help="Windows: tap the playback mix of the input device (WASAPI loopback). "
                         "Allows in/out to be the same device; recommended with a virtual cable to avoid double-audio.")
    sp.set_defaults(func=cmd_stream)

    return p

def cmd_list(_args: argparse.Namespace) -> int:
    print("Available filters:")
    for name, help_text in AF_AVAILABLE().items():
        print(f"  - {name:10s} : {help_text}")
    return 0

def cmd_run(args: argparse.Namespace) -> int:
    temp_dir: Optional[str] = None
    try:
        stages = _parse_pipeline(args.pipeline, args.filter)
        unknown = [s for s in stages if s not in AF_AVAILABLE().keys()]
        if unknown:
            raise SystemExit(f"Unknown filter(s) in pipeline: {', '.join(unknown)}")
        raw_extras = _parse_kv_pairs(args.extra)
        stage_extras = _split_stage_extras(stages, raw_extras)

        if is_youtube_url(args.url):
            wav_path, temp_dir = fetch_youtube_wav(args.url, ffmpeg_location=args.ffmpeg_location)
            in_obj: sf.SoundFile | io.BytesIO | str = wav_path
            log.info("Downloaded audio to %s", wav_path)
        else:
            raw, _ctype = FileFetcher().fetch(args.url)
            in_obj = io.BytesIO(raw)

        with sf.SoundFile(in_obj, mode="r") as f_in:
            sr = f_in.samplerate
            C = f_in.channels
            total = len(f_in)
            if args.start and args.start > 0:
                f_in.seek(int(args.start * sr))
            remaining = total - f_in.tell()
            if args.duration is not None:
                remaining = min(remaining, int(args.duration * sr))

            args.out.parent.mkdir(parents=True, exist_ok=True)
            with sf.SoundFile(args.out, mode="w", samplerate=sr, channels=C, subtype="PCM_16") as f_out:
                left = remaining
                filters = [AF_BUILD(name, **stage_extras[i]) for i, name in enumerate(stages)]
                dc_block = DCBlocker(20.0)
                while left > 0:
                    n = min(BLOCK, left)
                    x = f_in.read(n, dtype="float32", always_2d=True)
                    y = x
                    for flt in filters:
                        y = flt.process(y, sr)
                    y = dc_block.process(y, sr)
                    _add_tpdf_dither_inplace(y, level=1e-5)
                    f_out.write(y)
                    left -= n
                for flt in filters:
                    tail = flt.flush()
                    if tail is not None and tail.size:
                        y = dc_block.process(tail.astype(np.float32), sr)
                        _add_tpdf_dither_inplace(y, level=1e-5)
                        f_out.write(y)
        log.info("Saved %s", args.out)
        return 0
    except Exception as e:
        log.exception("Failed: %s", e)
        return 1
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def cmd_bench(args: argparse.Namespace) -> int:
    temp_dir: Optional[str] = None
    try:
        stages = _parse_pipeline(args.pipeline, args.filter)
        raw_extras = _parse_kv_pairs(args.extra)
        stage_extras = _split_stage_extras(stages, raw_extras)

        if is_youtube_url(args.url):
            wav_path, temp_dir = fetch_youtube_wav(args.url, ffmpeg_location=args.ffmpeg_location)
            in_obj: sf.SoundFile | io.BytesIO | str = wav_path
        else:
            raw, _ctype = FileFetcher().fetch(args.url)
            in_obj = io.BytesIO(raw)

        with sf.SoundFile(in_obj, mode="r") as f_in:
            x = f_in.read(dtype="float32", always_2d=True)
            sr = f_in.samplerate

        times: List[float] = []
        for _ in range(max(1, args.runs)):
            t0 = time.perf_counter()
            y = x
            filters = [AF_BUILD(name, **stage_extras[i]) for i, name in enumerate(stages)]
            for flt in filters:
                y = flt.process(y, sr)
            # include DC blocker in bench to reflect real-time path
            y = DCBlocker(20.0).process(y, sr)
            for flt in filters:
                tail = flt.flush()
                if tail is not None and tail.size:
                    y = np.concatenate([y, tail.astype(np.float32)], axis=0)
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        print(f"{'|'.join(stages)}: {args.runs} run(s) — avg {avg*1000:.2f} ms, min {min(times)*1000:.2f} ms, max {max(times)*1000:.2f} ms")
        return 0
    except Exception as e:
        log.exception("Bench failed: %s", e)
        return 1
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def cmd_list_devices(_args: argparse.Namespace) -> int:
    import sounddevice as sd

    devices = sd.query_devices()
    apis = sd.query_hostapis()

    print("Available audio devices:\n")
    for i, dev in enumerate(devices):
        api_name = apis[dev['hostapi']]['name']
        name = dev['name']
        in_ch = dev.get('max_input_channels', 0)
        out_ch = dev.get('max_output_channels', 0)
        sr = dev.get('default_samplerate', 0)
        print(f"[{i:02d}] {name}  ({api_name})")
        print(f"     Input channels:  {in_ch}")
        print(f"     Output channels: {out_ch}")
        print(f"     Default SR:      {sr:.0f} Hz\n")
    return 0
# ---------------- Stream (duplex / WASAPI loopback) ----------------

def cmd_stream(args: argparse.Namespace) -> int:
    """
    Live stream modes:
      • Normal duplex (mic/line-in -> out)
      • Windows WASAPI loopback (tap device playback) -> out
    Supports same in/out device when --wasapi-loopback is used.
    """
    import queue
    import warnings
    import ctypes
    from collections import deque

    # Build pipeline
    try:
        stages = _parse_pipeline(args.pipeline, args.filter)
        raw_extras = _parse_kv_pairs(args.extra)
        stage_extras = _split_stage_extras(stages, raw_extras)
        filters: List[AudioFilter] = [AF_BUILD(name, **stage_extras[i]) for i, name in enumerate(stages)]
    except Exception as e:
        log.exception("Pipeline build failed: %s", e)
        return 1

    samplerate = int(args.samplerate)
    in_channels = int(args.in_channels)
    out_channels = int(args.out_channels)
    B = max(1, int(args.blocksize))

    # Latency string or float seconds
    try:
        latency: float | str = float(args.latency)
    except Exception:
        latency = args.latency

    # Resolve device indices (name or index accepted)
    def _resolve_device(d: str | int, kind: str, *, allow_render_for_loopback: bool) -> int:
        try:
            if isinstance(d, int):
                return d
            if isinstance(d, str) and d.isdigit():
                return int(d)
            devs = sd.query_devices()
            for idx, dev in enumerate(devs):
                if dev["name"] != d:
                    continue
                if kind == "in":
                    ok = dev.get("max_input_channels", 0) > 0
                    if allow_render_for_loopback:
                        ok = ok or dev.get("max_output_channels", 0) > 0
                else:
                    ok = dev.get("max_output_channels", 0) > 0
                if ok:
                    return idx
        except Exception:
            pass
        return d

    in_dev = _resolve_device(args.in_device, "in", allow_render_for_loopback=bool(args.wasapi_loopback))
    out_dev = _resolve_device(args.out_device, "out", allow_render_for_loopback=False)

    # Preflight checks (skip input check when loopback; we probe it ourselves)
    if not args.wasapi_loopback:
        try:
            sd.check_input_settings(device=in_dev, samplerate=samplerate, channels=in_channels, dtype="float32")
        except Exception as e:
            log.error("Input device settings invalid: %s", e)
            log.error("Tip: 'python main.py list-devices' to inspect names/channels/samplerates.")
            return 1
    try:
        sd.check_output_settings(device=out_dev, samplerate=samplerate, channels=out_channels, dtype="float32")
    except Exception as e:
        log.error("Output device settings invalid: %s", e)
        log.error("Tip: 'python main.py list-devices' to inspect names/channels/samplerates.")
        return 1

    # Shared DSP buffers
    max_ch = max(in_channels, out_channels)
    scratch_mid = np.zeros((B, max_ch), dtype=np.float32)
    gain_lin = _db_to_lin(-abs(float(args.headroom_db))) if args.headroom_db else 1.0
    hard_limit = bool(args.hard_limit)

    # DC blocker instance (per stream)
    dc_block = DCBlocker(cutoff_hz=20.0)

    def process_block(x: np.ndarray) -> np.ndarray:
        if x.shape[1] == max_ch:
            np.copyto(scratch_mid[:, :max_ch], x)
            cur = scratch_mid[:, :max_ch]
        else:
            cur = _upmix_mono_to_stereo(x, max_ch) if x.shape[1] == 1 else _downmix_to_n(x, max_ch)

        y = cur
        for f in filters:
            y = f.process(y, samplerate)

        # DC-block + tiny TPDF dither before output gain/limiting
        y = dc_block.process(y, samplerate)
        _add_tpdf_dither_inplace(y, level=1e-5)

        if y.shape[1] != out_channels:
            y = _upmix_mono_to_stereo(y, out_channels) if (y.shape[1] == 1 and out_channels > 1) else _downmix_to_n(y, out_channels)

        if gain_lin != 1.0:
            np.multiply(y, gain_lin, out=y)
        if hard_limit:
            _hard_clip_inplace(y, -1.0, 1.0)
        return y

    stop_event = threading.Event()

    # ---------------------- WASAPI loopback path ----------------------
    if args.wasapi_loopback:
        q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

        # Map to WASAPI variant of the device by name if needed
        devs = sd.query_devices()
        apis = sd.query_hostapis()

        def _is_wasapi(idx: int) -> bool:
            try:
                return "WASAPI" in apis[devs[idx]["hostapi"]]["name"]
            except Exception:
                return False

        if not _is_wasapi(in_dev):
            try:
                wanted_name = sd.query_devices(in_dev)["name"]
            except Exception:
                wanted_name = None
            if wanted_name:
                for idx, d in enumerate(devs):
                    try:
                        if d["name"] == wanted_name and _is_wasapi(idx) and (d.get("max_output_channels", 0) > 0 or d.get("max_input_channels", 0) > 0):
                            in_dev = idx
                            break
                    except Exception:
                        pass

        dev_info = sd.query_devices(in_dev)
        try:
            api_name = sd.query_hostapis()[dev_info["hostapi"]]["name"]
        except Exception:
            api_name = "<unknown>"
        if "WASAPI" not in api_name:
            log.error("Selected input device is not WASAPI (got %s). Pick the WASAPI index of the render device.", api_name)
            return 1

        # Lock to device mix rate if present
        try:
            mix_sr = int(dev_info.get("default_samplerate") or samplerate)
        except Exception:
            mix_sr = samplerate
        samplerate = mix_sr

        wasapi_in = sd.WasapiSettings()
        wasapi_in.loopback = True
        wasapi_in.exclusive = False  # shared mode (friendlier to BT)

        wasapi_out = sd.WasapiSettings()
        wasapi_out.exclusive = False

        # ---- Robust probe: try a descending set of candidate channel counts until one opens ----
        reported_out = int(dev_info.get("max_output_channels", 0) or 0)
        reported_in  = int(dev_info.get("max_input_channels", 0) or 0)
        cand_ch = [reported_out, reported_in, 8, 6, 4, 3, 2, 1]
        cand_ch = [c for c in dict.fromkeys([c for c in cand_ch if isinstance(c, int)])]  # unique, >0

        log.info("Starting WASAPI loopback probe on device %s (WASAPI). Mix SR=%d. Channel candidates=%s",
                 str(in_dev), samplerate, cand_ch or "<none>")

        def probe_loopback_open() -> tuple[int, sd.InputStream] | None:
            # Try to open an InputStream for each candidate channel count until success
            for ch in cand_ch:
                try:
                    test_stream = sd.InputStream(device=in_dev,
                                                 samplerate=samplerate,
                                                 blocksize=B,
                                                 dtype="float32",
                                                 channels=ch,
                                                 latency=latency,
                                                 extra_settings=wasapi_in)
                    test_stream.__enter__()  # open
                    log.info("Loopback probe succeeded with channels=%d @ %d Hz", ch, samplerate)
                    return ch, test_stream  # keep it open; we'll reuse it
                except Exception as e:
                    log.debug("Probe failed for channels=%d: %s", ch, e)
            return None

        probe = probe_loopback_open()
        if probe is None:
            log.error("All loopback channel probes failed on this device. Unable to open WASAPI loopback.")
            # Fall back to soundcard path
        else:
            in_ch, in_stream = probe

            try:
                def in_cb(indata, frames, time_info, status):
                    try:
                        if status:
                            log.debug("Loopback status: %s", status)
                        x = indata[:, :in_ch].astype(np.float32, copy=False)
                        y = process_block(x)
                        try:
                            q.put_nowait(y.copy())
                        except queue.Full:
                            try:
                                _ = q.get_nowait()
                            except Exception:
                                pass
                            q.put_nowait(y.copy())
                    except Exception as ex:
                        log.exception("Loopback in_cb error: %s", ex)

                def out_cb(outdata, frames, time_info, status):
                    try:
                        if status:
                            log.debug("Out status: %s", status)
                    except Exception:
                        pass
                    try:
                        y = q.get_nowait()
                        n = min(frames, y.shape[0])
                        outdata[:n, :out_channels] = y[:n, :out_channels]
                        if n < frames:
                            outdata[n:, :out_channels].fill(0.0)
                    except queue.Empty:
                        outdata.fill(0.0)

                # Rebind the callback on the already-opened input stream
                in_stream.callback = in_cb

                with in_stream, \
                     sd.OutputStream(device=out_dev,
                                     samplerate=samplerate,
                                     blocksize=B,
                                     dtype="float32",
                                     channels=out_channels,
                                     latency=latency,
                                     extra_settings=wasapi_out,
                                     callback=out_cb):
                    log.info("Loopback active (input ch=%d, sr=%d). Press Ctrl+C to stop.", in_ch, samplerate)
                    while not stop_event.is_set():
                        stop_event.wait(0.5)

                for f in filters:
                    try:
                        _ = f.flush()
                    except Exception:
                        pass
                return 0

            except KeyboardInterrupt:
                log.info("Stopping stream...")
                try:
                    in_stream.__exit__(None, None, None)
                except Exception:
                    pass
                return 0

            except Exception as e:
                log.warning("WASAPI loopback run failed after probe: %s", e)
                try:
                    in_stream.__exit__(None, None, None)
                except Exception:
                    pass
                # fall through to soundcard fallback

        # ----- Soundcard fallback with COM-initialized worker + jitter buffer & drift correction -----
        try:
            # NumPy 2.x shim for soundcard (it uses np.fromstring on buffers)
            np.fromstring = np.frombuffer  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            import soundcard as sc
            from soundcard.mediafoundation import SoundcardRuntimeWarning  # type: ignore
            warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
        except Exception:
            log.error("Soundcard fallback unavailable. Install with: pip install soundcard")
            return 1

        # Choose the speaker we’re tapping (AirPods if present)
        spk = None
        for s in sc.all_speakers():
            if "AirPods Pro" in s.name:
                spk = s
                break
        if spk is None:
            spk = sc.default_speaker()

        log.info("Fallback loopback via 'soundcard': capturing loopback from '%s' @ %d Hz", spk.name, samplerate)

        class RecorderManager:
            def __init__(self, speaker, sr: int, blocksize: int, max_queue: int = 64):
                self.sc = sc
                self.speaker = speaker
                self.sr = int(sr)
                self.bs = int(blocksize)
                self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue)
                self._stop = threading.Event()
                self._thr = threading.Thread(target=self._worker, name="MF-Loopback", daemon=True)
                self._thr.start()

            def read(self, timeout: float = 0.4) -> np.ndarray:
                try:
                    return self._q.get(timeout=timeout)
                except queue.Empty:
                    return np.zeros((self.bs, 1), dtype=np.float32)

            def close(self):
                self._stop.set()
                try:
                    self._thr.join(timeout=1.0)
                except Exception:
                    pass

            @staticmethod
            def _coinit_mta():
                # COINIT_MULTITHREADED = 0x0
                try:
                    ctypes.windll.ole32.CoInitializeEx(None, 0x0)
                except Exception:
                    pass

            @staticmethod
            def _couninit():
                try:
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass

            def _worker(self):
                self._coinit_mta()
                mic = None
                rec = None

                def safe_close():
                    nonlocal rec
                    if rec is not None:
                        try:
                            rec.__exit__(None, None, None)
                        except Exception:
                            pass
                        rec = None

                def open_rec():
                    nonlocal mic, rec
                    safe_close()
                    mic = self.sc.get_microphone(id=str(self.speaker.name), include_loopback=True)
                    rec = mic.recorder(samplerate=self.sr, blocksize=self.bs)
                    rec.__enter__()
                    # warm-up
                    for _ in range(4):
                        try:
                            _ = rec.record(numframes=self.bs)
                        except Exception:
                            pass

                backoff = 0.05
                while not self._stop.is_set():
                    try:
                        if rec is None:
                            open_rec()
                            backoff = 0.05
                        x = rec.record(numframes=self.bs)  # may throw if device hiccups
                        # sanitize to (B, C) float32 with exact B frames
                        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.size == 0:
                            x = np.zeros((self.bs, 1), dtype=np.float32)
                        elif x.shape[0] < self.bs:
                            pad = np.zeros((self.bs - x.shape[0], x.shape[1]), dtype=np.float32)
                            x = np.vstack([x, pad])
                        x = x.astype(np.float32, copy=False)

                        try:
                            self._q.put_nowait(x)
                        except queue.Full:
                            try:
                                _ = self._q.get_nowait()
                            except Exception:
                                pass
                            self._q.put_nowait(x)

                    except Exception as ex:
                        log.error("Recorder worker error: %s (restarting)", ex)
                        safe_close()
                        time.sleep(backoff)
                        backoff = min(0.5, backoff * 2)

                safe_close()
                self._couninit()

        # ---- Jitter buffer, crossfade, adaptive resampler settings ----
        PREBUFFER_BLOCKS = 10
        TARGET_FILL = PREBUFFER_BLOCKS
        MAX_FILL = PREBUFFER_BLOCKS * 3
        XFADE_MS = 6
        xfade_len = max(1, int(samplerate * XFADE_MS / 1000))

        # Controller gains (smooth + integral centering)
        Kp = 0.00003
        Ki = 0.0000008
        ratio_ema = 1.0
        ratio_alpha = 0.02
        fill_i = 0.0

        class AdaptiveResampler:
            """Keeps output clock aligned to input by nudging resampling ratio with linear interpolation."""
            def __init__(self, channels: int, sr: int):
                self.channels = channels
                self.sr = sr
                self.phase = 0.0
                self.ratio = 1.0
                self.min_ratio = 0.9990
                self.max_ratio = 1.0010
                self.prev_tail: Optional[np.ndarray] = None

            def set_ratio(self, r: float):
                self.ratio = float(np.clip(r, self.min_ratio, self.max_ratio))

            def process(self, x: np.ndarray) -> np.ndarray:
                C = self.channels
                if x.ndim != 2:
                    x = np.atleast_2d(x)
                if x.shape[1] != C:
                    if x.shape[1] == 1 and C > 1:
                        x = np.tile(x, (1, C))
                    elif C == 1:
                        x = x.mean(axis=1, keepdims=True).astype(np.float32)
                    else:
                        x = x[:, :C]

                src = np.vstack([self.prev_tail, x]) if self.prev_tail is not None else x
                N = src.shape[0]
                out_len = max(1, int((N - 1) / self.ratio))
                t_src = self.phase + np.arange(out_len, dtype=np.float32) * self.ratio
                t_src = np.clip(t_src, 0.0, N - 1.0001)

                i0 = np.floor(t_src).astype(np.int32)
                frac = (t_src - i0).astype(np.float32)
                i1 = np.clip(i0 + 1, 0, N - 1)

                y = (src[i0, :] * (1.0 - frac)[:, None] + src[i1, :] * frac[:, None]).astype(np.float32)

                used = t_src[-1] + self.ratio
                self.phase = float(used - (N - 1))
                self.prev_tail = src[-1:, :].copy()
                return y

        def _xfade(prev_block: np.ndarray, cur_block: np.ndarray) -> np.ndarray:
            n = min(xfade_len, prev_block.shape[0], cur_block.shape[0])
            if n <= 1:
                return cur_block
            w = np.linspace(0.0, 1.0, num=n, dtype=np.float32)[:, None]
            out = cur_block.copy()
            out[:n, :] = prev_block[-n:, :] * (1.0 - w) + cur_block[:n, :] * w
            return out

        # Start recorder worker
        rec_mgr = RecorderManager(spk, samplerate, B)

        try:
            with sd.OutputStream(device=out_dev,
                                 samplerate=samplerate,
                                 blocksize=B,
                                 dtype="float32",
                                 channels=out_channels,
                                 latency=latency) as out:

                dq: "deque[np.ndarray]" = deque(maxlen=MAX_FILL)
                last_block: Optional[np.ndarray] = None

                # Prebuffer
                for _ in range(PREBUFFER_BLOCKS):
                    dq.append(rec_mgr.read())

                # Producer thread: fill jitter buffer from worker
                def _producer():
                    try:
                        while not stop_event.is_set():
                            x = rec_mgr.read()
                            if len(dq) == dq.maxlen:
                                _ = dq.popleft()
                            dq.append(x)
                    except Exception as ex:
                        log.exception("Producer loop error: %s", ex)
                        stop_event.set()

                threading.Thread(target=_producer, daemon=True).start()

                # Consumer with adaptive resampler + sample-accurate pacing
                resampler = AdaptiveResampler(channels=max(out_channels, 1), sr=samplerate)
                log.info("Loopback active (soundcard + jitter buffer + drift correction). Press Ctrl+C to stop.")
                next_deadline = time.perf_counter()

                while not stop_event.is_set():
                    if dq:
                        x = dq.popleft()
                    else:
                        hold_len = B // 2
                        if last_block is not None:
                            x = np.repeat(last_block[-1:, :1], hold_len, axis=0)
                        else:
                            x = np.zeros((hold_len, 1), dtype=np.float32)

                    x = x.astype(np.float32, copy=False)
                    y = process_block(x)

                    # PI drift control -> ratio near 1.0
                    fill_err = float(len(dq) - TARGET_FILL)
                    dt_nom = max(1e-6, y.shape[0] / float(samplerate))
                    fill_i += fill_err * dt_nom
                    target_ratio = 1.0 - (Kp * fill_err + Ki * fill_i)
                    ratio_ema = (1.0 - ratio_alpha) * ratio_ema + ratio_alpha * target_ratio
                    resampler.set_ratio(ratio_ema)

                    y = resampler.process(y)

                    if last_block is not None and y.shape == last_block.shape:
                        y = _xfade(last_block, y)

                    if y.shape[1] != out_channels:
                        if y.shape[1] == 1 and out_channels > 1:
                            y = np.tile(y, (1, out_channels))
                        else:
                            y = y[:, :out_channels]

                    out.write(y)
                    last_block = y

                    next_deadline += y.shape[0] / float(samplerate)
                    now = time.perf_counter()
                    if now < next_deadline:
                        time.sleep(next_deadline - now)

                for f in filters:
                    try:
                        _ = f.flush()
                    except Exception:
                        pass
                rec_mgr.close()
                return 0

        except KeyboardInterrupt:
            log.info("Stopping stream...")
            rec_mgr.close()
            return 0
        except Exception as e:
            log.exception("Soundcard loopback failed: %s", e)
            rec_mgr.close()
            return 1

    # ---------------------- Normal duplex (mic/line-in) ----------------------
    log.info("Starting duplex: in=%s (%d ch) -> out=%s (%d ch) @ %d Hz, B=%d, latency=%s",
             str(in_dev), in_channels, str(out_dev), out_channels, samplerate, B, str(latency))

    scratch_in = np.zeros((B, in_channels), dtype=np.float32)

    def duplex_cb(indata, outdata, frames, time_info, status):
        try:
            if status:
                log.debug("Stream status: %s", status)
            x = indata[:, :in_channels]
            if frames != scratch_in.shape[0]:
                x = x[:frames, :]
            y = process_block(x)
            n = min(frames, y.shape[0])
            outdata[:n, :out_channels] = y[:n, :out_channels]
            if n < frames:
                outdata[n:, :out_channels].fill(0.0)
        except Exception as ex:
            log.exception("Duplex callback error: %s", ex)
            outdata.fill(0.0)

    try:
        with sd.Stream(
            samplerate=samplerate,
            blocksize=B,
            latency=latency,
            device=(in_dev, out_dev),
            channels=(in_channels, out_channels),
            dtype="float32",
            callback=duplex_cb,
        ):
            log.info("Duplex active. Press Ctrl+C to stop.")
            while not stop_event.is_set():
                stop_event.wait(0.5)
    except KeyboardInterrupt:
        log.info("Stopping stream...")
    except Exception as e:
        log.exception("Failed to start/maintain stream: %s", e)
        return 1
    finally:
        for f in filters:
            try:
                _ = f.flush()
            except Exception:
                pass
    return 0

# ---------------- Entry ----------------

def build_and_run(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)

if __name__ == "__main__":  # pragma: no cover
    sys.exit(build_and_run())
