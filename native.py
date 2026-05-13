# native.py
# ------------------------------------------------------------
# ctypes wrapper for AudioProject.dll
#
# Put this file next to:
#   AudioProject.dll
#
# Designed for NumPy float32 audio buffers shaped:
#   (frames, channels)
#
# C++ expects interleaved float32:
#   frame0_ch0, frame0_ch1, frame1_ch0, frame1_ch1, ...
#
# Main wrappers:
#   NativeDCBlocker
#   NativeSoftClipper
#   NativeLimiter
#   NativeCompressor
#   NativeSOS
#   NativeBasicChain
#   NativeFixedBlockAdapter
#
# Utility:
#   is_available()
#   info()
#   force_channels()
# ------------------------------------------------------------

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ============================================================
# Constants / ctypes aliases
# ============================================================

_F32 = np.float32

_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_i64 = ctypes.c_longlong


# ============================================================
# Errors
# ============================================================

class NativeAudioError(RuntimeError):
    pass


class NativeAudioUnavailable(NativeAudioError):
    pass


# ============================================================
# NumPy helpers
# ============================================================

def _ensure_2d_float32(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.ndim != 2:
        raise ValueError(f"Audio buffer must be 1D or 2D, got shape={arr.shape!r}")

    return np.ascontiguousarray(arr, dtype=np.float32)


def _empty_audio(frames: int, channels: int) -> np.ndarray:
    frames = max(0, int(frames))
    channels = max(1, int(channels))
    return np.zeros((frames, channels), dtype=np.float32)


def _ptr(arr: np.ndarray) -> Any:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr.ctypes.data_as(_c_float_p)


def _force_channels_py(x: np.ndarray, channels: int) -> np.ndarray:
    x = _ensure_2d_float32(x)
    channels = max(1, int(channels))

    if x.shape[1] == channels:
        return x

    if x.shape[1] == 1 and channels == 2:
        return np.ascontiguousarray(np.repeat(x, 2, axis=1), dtype=np.float32)

    if x.shape[1] > channels:
        return np.ascontiguousarray(x[:, :channels], dtype=np.float32)

    out = np.zeros((x.shape[0], channels), dtype=np.float32)
    out[:, :x.shape[1]] = x
    return out


# ============================================================
# DLL loader
# ============================================================

class _AudioProjectDLL:
    def __init__(self) -> None:
        self.dll_path: Optional[Path] = None
        self.dll: Optional[ctypes.CDLL] = None
        self.load_error: Optional[str] = None
        self._loaded = False

    def load(self) -> ctypes.CDLL:
        if self._loaded and self.dll is not None:
            return self.dll

        here = Path(__file__).resolve().parent

        # Primary behavior: same dir as native.py
        candidates = [
            here / "AudioProject.dll",
            here / "audioproject.dll",
        ]

        # Optional override for debugging
        env_path = os.environ.get("AUDIOPROJECT_DLL", "").strip()
        if env_path:
            candidates.insert(0, Path(env_path).expanduser().resolve())

        last_error = None

        for path in candidates:
            try:
                if not path.exists():
                    continue

                dll = ctypes.CDLL(str(path))
                self.dll_path = path
                self.dll = dll
                self._bind_signatures(dll)
                self._loaded = True
                self.load_error = None
                return dll

            except Exception as exc:
                last_error = exc

        self.load_error = (
            f"Could not load AudioProject.dll from {here}. "
            f"Last error: {last_error}"
        )
        raise NativeAudioUnavailable(self.load_error)

    def _bind_signatures(self, dll: ctypes.CDLL) -> None:
        # --------------------------------------------------------
        # Core
        # --------------------------------------------------------
        dll.ap_version.argtypes = []
        dll.ap_version.restype = ctypes.c_int

        dll.ap_last_error.argtypes = []
        dll.ap_last_error.restype = ctypes.c_char_p

        dll.ap_clear_error.argtypes = []
        dll.ap_clear_error.restype = None

        # --------------------------------------------------------
        # Force channels
        # --------------------------------------------------------
        dll.ap_force_channels.argtypes = [
            _c_float_p,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.ap_force_channels.restype = ctypes.c_int

        # --------------------------------------------------------
        # FixedBlockAdapter
        # --------------------------------------------------------
        dll.ap_fixed_adapter_create.argtypes = [ctypes.c_int]
        dll.ap_fixed_adapter_create.restype = ctypes.c_void_p

        dll.ap_fixed_adapter_destroy.argtypes = [ctypes.c_void_p]
        dll.ap_fixed_adapter_destroy.restype = None

        dll.ap_fixed_adapter_reset.argtypes = [ctypes.c_void_p]
        dll.ap_fixed_adapter_reset.restype = ctypes.c_int

        dll.ap_fixed_adapter_available_frames.argtypes = [ctypes.c_void_p]
        dll.ap_fixed_adapter_available_frames.restype = _c_i64

        dll.ap_fixed_adapter_push.argtypes = [
            ctypes.c_void_p,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
        ]
        dll.ap_fixed_adapter_push.restype = ctypes.c_int

        dll.ap_fixed_adapter_pop.argtypes = [
            ctypes.c_void_p,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
        ]
        dll.ap_fixed_adapter_pop.restype = ctypes.c_int

        dll.ap_fixed_adapter_process_produced.argtypes = [
            ctypes.c_void_p,
            _c_float_p,
            _c_i64,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
        ]
        dll.ap_fixed_adapter_process_produced.restype = ctypes.c_int

        # --------------------------------------------------------
        # DCBlocker
        # --------------------------------------------------------
        dll.ap_dcblocker_create.argtypes = [ctypes.c_float, ctypes.c_int]
        dll.ap_dcblocker_create.restype = ctypes.c_void_p

        dll.ap_dcblocker_destroy.argtypes = [ctypes.c_void_p]
        dll.ap_dcblocker_destroy.restype = None

        dll.ap_dcblocker_reset.argtypes = [ctypes.c_void_p]
        dll.ap_dcblocker_reset.restype = ctypes.c_int

        dll.ap_dcblocker_set_r.argtypes = [ctypes.c_void_p, ctypes.c_float]
        dll.ap_dcblocker_set_r.restype = ctypes.c_int

        dll.ap_dcblocker_process.argtypes = [
            ctypes.c_void_p,
            _c_float_p,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.ap_dcblocker_process.restype = ctypes.c_int

        # --------------------------------------------------------
        # SoftClipper
        # --------------------------------------------------------
        dll.ap_softclip_process.argtypes = [
            _c_float_p,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
            ctypes.c_float,
        ]
        dll.ap_softclip_process.restype = ctypes.c_int

        # --------------------------------------------------------
        # Limiter
        # --------------------------------------------------------
        dll.ap_limiter_create.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        dll.ap_limiter_create.restype = ctypes.c_void_p

        dll.ap_limiter_destroy.argtypes = [ctypes.c_void_p]
        dll.ap_limiter_destroy.restype = None

        dll.ap_limiter_reset.argtypes = [ctypes.c_void_p]
        dll.ap_limiter_reset.restype = ctypes.c_int

        dll.ap_limiter_set_params.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        dll.ap_limiter_set_params.restype = ctypes.c_int

        dll.ap_limiter_process.argtypes = [
            ctypes.c_void_p,
            _c_float_p,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.ap_limiter_process.restype = ctypes.c_int

        # --------------------------------------------------------
        # Compressor
        # --------------------------------------------------------
        dll.ap_compressor_create.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        dll.ap_compressor_create.restype = ctypes.c_void_p

        dll.ap_compressor_destroy.argtypes = [ctypes.c_void_p]
        dll.ap_compressor_destroy.restype = None

        dll.ap_compressor_reset.argtypes = [ctypes.c_void_p]
        dll.ap_compressor_reset.restype = ctypes.c_int

        dll.ap_compressor_set_params.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        dll.ap_compressor_set_params.restype = ctypes.c_int

        dll.ap_compressor_process.argtypes = [
            ctypes.c_void_p,
            _c_float_p,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.ap_compressor_process.restype = ctypes.c_int

        # --------------------------------------------------------
        # SOS
        # --------------------------------------------------------
        dll.ap_sos_create.argtypes = [ctypes.c_int, ctypes.c_int]
        dll.ap_sos_create.restype = ctypes.c_void_p

        dll.ap_sos_destroy.argtypes = [ctypes.c_void_p]
        dll.ap_sos_destroy.restype = None

        dll.ap_sos_reset.argtypes = [ctypes.c_void_p]
        dll.ap_sos_reset.restype = ctypes.c_int

        dll.ap_sos_set.argtypes = [
            ctypes.c_void_p,
            _c_float_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.ap_sos_set.restype = ctypes.c_int

        dll.ap_sos_process.argtypes = [
            ctypes.c_void_p,
            _c_float_p,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
        ]
        dll.ap_sos_process.restype = ctypes.c_int

        # --------------------------------------------------------
        # Basic chain
        # --------------------------------------------------------
        dll.ap_basic_chain_create.argtypes = [ctypes.c_int]
        dll.ap_basic_chain_create.restype = ctypes.c_void_p

        dll.ap_basic_chain_destroy.argtypes = [ctypes.c_void_p]
        dll.ap_basic_chain_destroy.restype = None

        dll.ap_basic_chain_reset.argtypes = [ctypes.c_void_p]
        dll.ap_basic_chain_reset.restype = ctypes.c_int

        dll.ap_basic_chain_set.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.ap_basic_chain_set.restype = ctypes.c_int

        dll.ap_basic_chain_process.argtypes = [
            ctypes.c_void_p,
            _c_float_p,
            _c_float_p,
            _c_i64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.ap_basic_chain_process.restype = ctypes.c_int

    def last_error(self) -> str:
        if self.dll is None:
            return self.load_error or "AudioProject.dll is not loaded."

        try:
            raw = self.dll.ap_last_error()
            if not raw:
                return "Unknown native error."
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:
            return f"Could not read native error: {exc}"

    def check(self, code: int, where: str = "native call") -> None:
        if int(code) != 0:
            raise NativeAudioError(f"{where} failed: {self.last_error()}")


_AP = _AudioProjectDLL()


def load() -> ctypes.CDLL:
    return _AP.load()


def is_available() -> bool:
    try:
        load()
        return True
    except Exception:
        return False


def version() -> Optional[int]:
    try:
        return int(load().ap_version())
    except Exception:
        return None


def dll_path() -> Optional[str]:
    try:
        load()
        return str(_AP.dll_path) if _AP.dll_path else None
    except Exception:
        return None


def info() -> dict[str, Any]:
    try:
        dll = load()
        return {
            "enabled": True,
            "available": True,
            "dll_path": str(_AP.dll_path) if _AP.dll_path else None,
            "load_error": None,
            "version": int(dll.ap_version()),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "dll_path": None,
            "load_error": str(exc),
            "version": None,
        }


# ============================================================
# Stateless utility wrappers
# ============================================================

def force_channels(block: np.ndarray, channels: int) -> np.ndarray:
    """
    Native version of:
      mono -> stereo duplicate
      too many channels -> truncate
      too few channels -> zero-fill
    """
    dll = load()

    x = _ensure_2d_float32(block)
    out_channels = max(1, int(channels))
    out = np.zeros((x.shape[0], out_channels), dtype=np.float32)

    code = dll.ap_force_channels(
        _ptr(x),
        _ptr(out),
        _c_i64(x.shape[0]),
        ctypes.c_int(x.shape[1]),
        ctypes.c_int(out_channels),
    )
    _AP.check(code, "ap_force_channels")
    return out


def softclip(block: np.ndarray, drive: float = 1.0) -> np.ndarray:
    dll = load()

    x = _ensure_2d_float32(block)
    out = np.empty_like(x, dtype=np.float32)

    code = dll.ap_softclip_process(
        _ptr(x),
        _ptr(out),
        _c_i64(x.shape[0]),
        ctypes.c_int(x.shape[1]),
        ctypes.c_float(float(drive)),
    )
    _AP.check(code, "ap_softclip_process")
    return out


# ============================================================
# Base native handle
# ============================================================

class _NativeHandle:
    _destroy_name: str = ""

    def __init__(self) -> None:
        self._dll = load()
        self._handle: Optional[int] = None
        self._closed = False

    @property
    def handle(self) -> int:
        if self._closed or not self._handle:
            raise NativeAudioError(f"{self.__class__.__name__} is closed or was not created.")
        return self._handle

    def close(self) -> None:
        if self._closed:
            return

        handle = self._handle
        self._handle = None
        self._closed = True

        if handle and self._destroy_name:
            try:
                destroy = getattr(self._dll, self._destroy_name)
                destroy(ctypes.c_void_p(handle))
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ============================================================
# Native DCBlocker
# ============================================================

class NativeDCBlocker(_NativeHandle):
    _destroy_name = "ap_dcblocker_destroy"

    def __init__(self, r: float = 0.995, channels: int = 2):
        super().__init__()

        h = self._dll.ap_dcblocker_create(
            ctypes.c_float(float(r)),
            ctypes.c_int(int(channels)),
        )
        if not h:
            raise NativeAudioError(f"ap_dcblocker_create failed: {_AP.last_error()}")

        self._handle = int(h)
        self.channels = int(channels)
        self.r = float(r)

    def reset(self) -> None:
        code = self._dll.ap_dcblocker_reset(ctypes.c_void_p(self.handle))
        _AP.check(code, "ap_dcblocker_reset")

    def set_r(self, r: float) -> None:
        self.r = float(r)
        code = self._dll.ap_dcblocker_set_r(
            ctypes.c_void_p(self.handle),
            ctypes.c_float(self.r),
        )
        _AP.check(code, "ap_dcblocker_set_r")

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d_float32(block)
        self.channels = x.shape[1]
        out = np.empty_like(x, dtype=np.float32)

        code = self._dll.ap_dcblocker_process(
            ctypes.c_void_p(self.handle),
            _ptr(x),
            _ptr(out),
            _c_i64(x.shape[0]),
            ctypes.c_int(x.shape[1]),
            ctypes.c_int(int(sr)),
        )
        _AP.check(code, "ap_dcblocker_process")
        return out

    def flush(self) -> None:
        return None


# ============================================================
# Native SoftClipper
# ============================================================

class NativeSoftClipper:
    def __init__(self, drive: float = 1.0):
        self.drive = float(drive)
        self._dll = load()

    def set_drive(self, drive: float) -> None:
        self.drive = float(drive)

    def reset(self) -> None:
        return None

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        del sr
        x = _ensure_2d_float32(block)
        out = np.empty_like(x, dtype=np.float32)

        code = self._dll.ap_softclip_process(
            _ptr(x),
            _ptr(out),
            _c_i64(x.shape[0]),
            ctypes.c_int(x.shape[1]),
            ctypes.c_float(self.drive),
        )
        _AP.check(code, "ap_softclip_process")
        return out

    def flush(self) -> None:
        return None


# ============================================================
# Native Limiter
# ============================================================

class NativeLimiter(_NativeHandle):
    _destroy_name = "ap_limiter_destroy"

    def __init__(
        self,
        ceiling_db: float = -1.0,
        attack_ms: float = 1.0,
        release_ms: float = 50.0,
    ):
        super().__init__()

        h = self._dll.ap_limiter_create(
            ctypes.c_float(float(ceiling_db)),
            ctypes.c_float(float(attack_ms)),
            ctypes.c_float(float(release_ms)),
        )
        if not h:
            raise NativeAudioError(f"ap_limiter_create failed: {_AP.last_error()}")

        self._handle = int(h)
        self.ceiling_db = float(ceiling_db)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)

    def reset(self) -> None:
        code = self._dll.ap_limiter_reset(ctypes.c_void_p(self.handle))
        _AP.check(code, "ap_limiter_reset")

    def set_params(
        self,
        *,
        ceiling_db: Optional[float] = None,
        attack_ms: Optional[float] = None,
        release_ms: Optional[float] = None,
    ) -> None:
        if ceiling_db is not None:
            self.ceiling_db = float(ceiling_db)
        if attack_ms is not None:
            self.attack_ms = float(attack_ms)
        if release_ms is not None:
            self.release_ms = float(release_ms)

        code = self._dll.ap_limiter_set_params(
            ctypes.c_void_p(self.handle),
            ctypes.c_float(self.ceiling_db),
            ctypes.c_float(self.attack_ms),
            ctypes.c_float(self.release_ms),
        )
        _AP.check(code, "ap_limiter_set_params")

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d_float32(block)
        out = np.empty_like(x, dtype=np.float32)

        code = self._dll.ap_limiter_process(
            ctypes.c_void_p(self.handle),
            _ptr(x),
            _ptr(out),
            _c_i64(x.shape[0]),
            ctypes.c_int(x.shape[1]),
            ctypes.c_int(int(sr)),
        )
        _AP.check(code, "ap_limiter_process")
        return out

    def flush(self) -> None:
        return None


# ============================================================
# Native Compressor
# ============================================================

class NativeCompressor(_NativeHandle):
    _destroy_name = "ap_compressor_destroy"

    def __init__(
        self,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        knee_db: float = 6.0,
        attack_ms: float = 4.0,
        release_ms: float = 90.0,
        makeup_db: float = 3.0,
        mix: float = 1.0,
    ):
        super().__init__()

        h = self._dll.ap_compressor_create(
            ctypes.c_float(float(threshold_db)),
            ctypes.c_float(float(ratio)),
            ctypes.c_float(float(knee_db)),
            ctypes.c_float(float(attack_ms)),
            ctypes.c_float(float(release_ms)),
            ctypes.c_float(float(makeup_db)),
            ctypes.c_float(float(mix)),
        )
        if not h:
            raise NativeAudioError(f"ap_compressor_create failed: {_AP.last_error()}")

        self._handle = int(h)

        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.knee_db = float(knee_db)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.makeup_db = float(makeup_db)
        self.mix = float(mix)

    def reset(self) -> None:
        code = self._dll.ap_compressor_reset(ctypes.c_void_p(self.handle))
        _AP.check(code, "ap_compressor_reset")

    def set_params(
        self,
        *,
        threshold_db: Optional[float] = None,
        ratio: Optional[float] = None,
        knee_db: Optional[float] = None,
        attack_ms: Optional[float] = None,
        release_ms: Optional[float] = None,
        makeup_db: Optional[float] = None,
        mix: Optional[float] = None,
    ) -> None:
        if threshold_db is not None:
            self.threshold_db = float(threshold_db)
        if ratio is not None:
            self.ratio = float(ratio)
        if knee_db is not None:
            self.knee_db = float(knee_db)
        if attack_ms is not None:
            self.attack_ms = float(attack_ms)
        if release_ms is not None:
            self.release_ms = float(release_ms)
        if makeup_db is not None:
            self.makeup_db = float(makeup_db)
        if mix is not None:
            self.mix = float(mix)

        code = self._dll.ap_compressor_set_params(
            ctypes.c_void_p(self.handle),
            ctypes.c_float(self.threshold_db),
            ctypes.c_float(self.ratio),
            ctypes.c_float(self.knee_db),
            ctypes.c_float(self.attack_ms),
            ctypes.c_float(self.release_ms),
            ctypes.c_float(self.makeup_db),
            ctypes.c_float(self.mix),
        )
        _AP.check(code, "ap_compressor_set_params")

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d_float32(block)
        out = np.empty_like(x, dtype=np.float32)

        code = self._dll.ap_compressor_process(
            ctypes.c_void_p(self.handle),
            _ptr(x),
            _ptr(out),
            _c_i64(x.shape[0]),
            ctypes.c_int(x.shape[1]),
            ctypes.c_int(int(sr)),
        )
        _AP.check(code, "ap_compressor_process")
        return out

    def flush(self) -> None:
        return None


# ============================================================
# Native SOS processor
# ============================================================

class NativeSOS(_NativeHandle):
    """
    Runs SOS filters natively.

    Python side should compute SOS coefficients.

    Expected SOS shape:
      (sections, 6)

    Each row:
      [b0, b1, b2, a0, a1, a2]

    Example:
      from scipy import signal
      sos = signal.butter(4, 8000, btype="lowpass", fs=48000, output="sos")
      filt = NativeSOS(sos, channels=2)
      y = filt.process(x, sr=48000)
    """

    _destroy_name = "ap_sos_destroy"

    def __init__(
        self,
        sos: Optional[np.ndarray] = None,
        *,
        channels: int = 2,
        max_sections: int = 32,
        reset_state: bool = True,
    ):
        super().__init__()

        h = self._dll.ap_sos_create(
            ctypes.c_int(int(max_sections)),
            ctypes.c_int(int(channels)),
        )
        if not h:
            raise NativeAudioError(f"ap_sos_create failed: {_AP.last_error()}")

        self._handle = int(h)
        self.channels = int(channels)

        if sos is not None:
            self.set_sos(sos, channels=channels, reset_state=reset_state)

    def reset(self) -> None:
        code = self._dll.ap_sos_reset(ctypes.c_void_p(self.handle))
        _AP.check(code, "ap_sos_reset")

    def set_sos(
        self,
        sos: np.ndarray,
        *,
        channels: Optional[int] = None,
        reset_state: bool = True,
    ) -> None:
        arr = np.asarray(sos, dtype=np.float32)

        if arr.ndim == 1:
            if arr.size != 6:
                raise ValueError("1D SOS must have exactly 6 values.")
            arr = arr[None, :]

        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"SOS must have shape (sections, 6), got {arr.shape!r}")

        arr = np.ascontiguousarray(arr, dtype=np.float32)

        ch = self.channels if channels is None else int(channels)
        ch = max(1, ch)
        self.channels = ch

        code = self._dll.ap_sos_set(
            ctypes.c_void_p(self.handle),
            _ptr(arr),
            ctypes.c_int(arr.shape[0]),
            ctypes.c_int(ch),
            ctypes.c_int(1 if reset_state else 0),
        )
        _AP.check(code, "ap_sos_set")

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        del sr

        x = _ensure_2d_float32(block)

        if x.shape[1] != self.channels:
            self.channels = x.shape[1]

        out = np.empty_like(x, dtype=np.float32)

        code = self._dll.ap_sos_process(
            ctypes.c_void_p(self.handle),
            _ptr(x),
            _ptr(out),
            _c_i64(x.shape[0]),
            ctypes.c_int(x.shape[1]),
        )
        _AP.check(code, "ap_sos_process")
        return out

    def flush(self) -> None:
        return None


# ============================================================
# Native Basic Chain
# DCBlocker -> SoftClipper -> Limiter in one native call
# ============================================================

class NativeBasicChain(_NativeHandle):
    _destroy_name = "ap_basic_chain_destroy"

    def __init__(
        self,
        *,
        channels: int = 2,
        dc_r: float = 0.995,
        soft_drive: float = 1.0,
        limiter_ceiling_db: float = -1.0,
        limiter_attack_ms: float = 1.0,
        limiter_release_ms: float = 50.0,
        use_dc: bool = True,
        use_softclip: bool = True,
        use_limiter: bool = True,
    ):
        super().__init__()

        h = self._dll.ap_basic_chain_create(ctypes.c_int(int(channels)))
        if not h:
            raise NativeAudioError(f"ap_basic_chain_create failed: {_AP.last_error()}")

        self._handle = int(h)
        self.channels = int(channels)

        self.dc_r = float(dc_r)
        self.soft_drive = float(soft_drive)
        self.limiter_ceiling_db = float(limiter_ceiling_db)
        self.limiter_attack_ms = float(limiter_attack_ms)
        self.limiter_release_ms = float(limiter_release_ms)
        self.use_dc = bool(use_dc)
        self.use_softclip = bool(use_softclip)
        self.use_limiter = bool(use_limiter)

        self._sync_params()

    def _sync_params(self) -> None:
        code = self._dll.ap_basic_chain_set(
            ctypes.c_void_p(self.handle),
            ctypes.c_float(self.dc_r),
            ctypes.c_float(self.soft_drive),
            ctypes.c_float(self.limiter_ceiling_db),
            ctypes.c_float(self.limiter_attack_ms),
            ctypes.c_float(self.limiter_release_ms),
            ctypes.c_int(1 if self.use_dc else 0),
            ctypes.c_int(1 if self.use_softclip else 0),
            ctypes.c_int(1 if self.use_limiter else 0),
        )
        _AP.check(code, "ap_basic_chain_set")

    def reset(self) -> None:
        code = self._dll.ap_basic_chain_reset(ctypes.c_void_p(self.handle))
        _AP.check(code, "ap_basic_chain_reset")

    def set_params(
        self,
        *,
        dc_r: Optional[float] = None,
        soft_drive: Optional[float] = None,
        limiter_ceiling_db: Optional[float] = None,
        limiter_attack_ms: Optional[float] = None,
        limiter_release_ms: Optional[float] = None,
        use_dc: Optional[bool] = None,
        use_softclip: Optional[bool] = None,
        use_limiter: Optional[bool] = None,
    ) -> None:
        if dc_r is not None:
            self.dc_r = float(dc_r)
        if soft_drive is not None:
            self.soft_drive = float(soft_drive)
        if limiter_ceiling_db is not None:
            self.limiter_ceiling_db = float(limiter_ceiling_db)
        if limiter_attack_ms is not None:
            self.limiter_attack_ms = float(limiter_attack_ms)
        if limiter_release_ms is not None:
            self.limiter_release_ms = float(limiter_release_ms)
        if use_dc is not None:
            self.use_dc = bool(use_dc)
        if use_softclip is not None:
            self.use_softclip = bool(use_softclip)
        if use_limiter is not None:
            self.use_limiter = bool(use_limiter)

        self._sync_params()

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _ensure_2d_float32(block)
        self.channels = x.shape[1]
        out = np.empty_like(x, dtype=np.float32)

        code = self._dll.ap_basic_chain_process(
            ctypes.c_void_p(self.handle),
            _ptr(x),
            _ptr(out),
            _c_i64(x.shape[0]),
            ctypes.c_int(x.shape[1]),
            ctypes.c_int(int(sr)),
        )
        _AP.check(code, "ap_basic_chain_process")
        return out

    def flush(self) -> None:
        return None


# ============================================================
# Native FixedBlockAdapter
# ============================================================

class NativeFixedBlockAdapter(_NativeHandle):
    """
    Native buffering helper for variable-rate filters.

    Two ways to use:

    1) Wrap a Python/native inner filter:

        adapter = NativeFixedBlockAdapter(inner=my_wsola, channels=2)
        out = adapter.process(block, sr)

    2) Push already-produced audio manually:

        adapter = NativeFixedBlockAdapter(channels=2)
        out = adapter.process_produced(produced, requested_frames=1024)

    The native side buffers produced frames and always returns exactly
    requested_frames, zero-padding when not enough audio is ready.
    """

    _destroy_name = "ap_fixed_adapter_destroy"

    def __init__(self, inner: Any = None, *, channels: int = 2):
        super().__init__()

        h = self._dll.ap_fixed_adapter_create(ctypes.c_int(int(channels)))
        if not h:
            raise NativeAudioError(f"ap_fixed_adapter_create failed: {_AP.last_error()}")

        self._handle = int(h)
        self.inner = inner
        self.channels = int(channels)

    def reset(self) -> None:
        code = self._dll.ap_fixed_adapter_reset(ctypes.c_void_p(self.handle))
        _AP.check(code, "ap_fixed_adapter_reset")

        if self.inner is not None and hasattr(self.inner, "flush"):
            try:
                self.inner.flush()
            except Exception:
                pass

    def available_frames(self) -> int:
        return int(self._dll.ap_fixed_adapter_available_frames(ctypes.c_void_p(self.handle)))

    def push(self, produced: np.ndarray) -> None:
        y = _force_channels_py(produced, self.channels)

        code = self._dll.ap_fixed_adapter_push(
            ctypes.c_void_p(self.handle),
            _ptr(y),
            _c_i64(y.shape[0]),
            ctypes.c_int(y.shape[1]),
        )
        _AP.check(code, "ap_fixed_adapter_push")

    def pop(self, requested_frames: int) -> np.ndarray:
        n = max(0, int(requested_frames))
        out = _empty_audio(n, self.channels)

        code = self._dll.ap_fixed_adapter_pop(
            ctypes.c_void_p(self.handle),
            _ptr(out),
            _c_i64(n),
            ctypes.c_int(self.channels),
        )
        _AP.check(code, "ap_fixed_adapter_pop")
        return out

    def process_produced(
        self,
        produced: np.ndarray,
        *,
        requested_frames: int,
    ) -> np.ndarray:
        n = max(0, int(requested_frames))
        y = _force_channels_py(produced, self.channels)
        out = _empty_audio(n, self.channels)

        code = self._dll.ap_fixed_adapter_process_produced(
            ctypes.c_void_p(self.handle),
            _ptr(y),
            _c_i64(y.shape[0]),
            _ptr(out),
            _c_i64(n),
            ctypes.c_int(self.channels),
        )
        _AP.check(code, "ap_fixed_adapter_process_produced")
        return out

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = _force_channels_py(block, self.channels)
        n = x.shape[0]

        if self.inner is None:
            # No inner filter means passthrough with native fixed buffering.
            return self.process_produced(x, requested_frames=n)

        y = self.inner.process(x, sr)

        if y is None:
            y = _empty_audio(0, self.channels)

        return self.process_produced(y, requested_frames=n)

    def flush(self) -> Optional[np.ndarray]:
        tail_parts = []

        if self.inner is not None and hasattr(self.inner, "flush"):
            try:
                tail = self.inner.flush()
                if tail is not None:
                    tail_parts.append(_force_channels_py(tail, self.channels))
            except Exception:
                pass

        available = self.available_frames()
        if available > 0:
            tail_parts.append(self.pop(available))

        if not tail_parts:
            return None

        return np.concatenate(tail_parts, axis=0).astype(np.float32, copy=False)


# ============================================================
# Compatibility aliases matching your Python class names
# ============================================================

DCBlocker = NativeDCBlocker
SoftClipper = NativeSoftClipper
Limiter = NativeLimiter
Compressor = NativeCompressor
SOSFilter = NativeSOS
BasicChain = NativeBasicChain
FixedBlockAdapter = NativeFixedBlockAdapter


# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":
    print(info())

    sr = 48000
    t = np.linspace(0.0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float32)
    x = 0.2 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    x = np.stack([x, x], axis=1)

    chain = NativeBasicChain(channels=2, soft_drive=1.5)
    y = chain.process(x, sr)

    print("input:", x.shape, x.dtype, float(np.max(np.abs(x))))
    print("output:", y.shape, y.dtype, float(np.max(np.abs(y))))