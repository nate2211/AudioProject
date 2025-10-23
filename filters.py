# =============================
# filters.py
# Scalable audio filter registry + stateful, stream-friendly filters
# mirroring your image pipeline's registry & extras style.
# =============================
from __future__ import annotations

from typing import Callable, Dict, Tuple, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from scipy import signal
import re
# ---------- Base & Registry ----------
class AudioFilter:
    """Base class for streamable audio filters.
    process(block, sr) -> block (N,C) float32 [-1,1]
    flush() -> optional tail (T,C) float32
    """
    def process(self, block: np.ndarray, sr: int) -> np.ndarray:  # (N,C)
        raise NotImplementedError

    def flush(self) -> Optional[np.ndarray]:
        return None

FilterFactory = Callable[[Dict[str, Any]], AudioFilter]
_REGISTRY: Dict[str, Tuple[str, FilterFactory]] = {}


def register_filter(name: str, *, help: str) -> Callable[[FilterFactory], FilterFactory]:
    key = name.strip().lower()
    def _decorator(factory: FilterFactory) -> FilterFactory:
        if key in _REGISTRY:
            raise ValueError(f"Duplicate filter name: {name}")
        _REGISTRY[key] = (help, factory)
        return factory
    return _decorator


def available_filters() -> Dict[str, str]:
    return {k: v[0] for k, v in sorted(_REGISTRY.items())}


def build_filter(name: str, **kwargs: Any) -> AudioFilter:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown filter '{name}'. Available: {', '.join(available_filters().keys()) or '(none)'}")
    _help, factory = _REGISTRY[key]
    return factory(kwargs)


# ---------- Utilities ----------
_DEF_EPS = 1e-12

def _db_to_lin(db: float) -> float:
    return float(10 ** (db / 20.0))


# ---------- Filters ----------
@register_filter("gain", help="Linear/dB gain. Params: db (0), lin (None)")
class Gain(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        g = params.get("lin")
        if g is None:
            db = float(params.get("db", 0.0))
            g = _db_to_lin(db)
        self.g = float(g)
    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        y = block * self.g
        return np.clip(y, -1.0, 1.0)


@register_filter("normalize", help="Peak normalize to target. Params: peak (0.98)")
class Normalize(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.target = float(params.get("peak", 0.98))
        self._max = 0.0
    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        m = float(np.max(np.abs(block)))
        if m > self._max:
            self._max = m
        if self._max < _DEF_EPS:
            return block
        y = block * (self.target / self._max)
        return np.clip(y, -1.0, 1.0)


class _IIRSOS(AudioFilter):
    def __init__(self):
        self.sos: Optional[np.ndarray] = None
        self.zi: Optional[np.ndarray] = None
        self._initd = False
        self._sr: Optional[int] = None
        self._C: Optional[int] = None

    def _design(self, sr: int):
        raise NotImplementedError

    def _ensure(self, sr: int, C: int):
        if self._initd and self._sr == sr and self._C == C:
            return
        self.sos = self._design(sr).astype(np.float32, copy=False)  # (S,6)
        S = self.sos.shape[0]
        # zi shape for sosfilt: (S, C, 2)
        self.zi = np.zeros((S, C, 2), dtype=np.float32)
        self._sr, self._C, self._initd = sr, C, True

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float32, copy=False)
        n, C = x.shape
        self._ensure(sr, C)
        # per-channel filtering with shared zi
        y = np.empty_like(x, dtype=np.float32)
        for ch in range(C):
            y[:, ch], self.zi[:, ch, :] = signal.sosfilt(self.sos, x[:, ch], zi=self.zi[:, ch, :])
        return y


@register_filter("lowpass", help="Butterworth LPF (SOS). Params: cutoff (Hz), order (4)")
class Lowpass(_IIRSOS):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.cutoff = float(params.get("cutoff"))
        self.order = int(params.get("order", 4))
    def _design(self, sr: int):
        wn = self.cutoff / (0.5 * sr)
        return signal.butter(self.order, wn, btype="low", output="sos")


@register_filter("highpass", help="Butterworth HPF (SOS). Params: cutoff (Hz), order (4)")
class Highpass(_IIRSOS):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.cutoff = float(params.get("cutoff"))
        self.order = int(params.get("order", 4))
    def _design(self, sr: int):
        wn = self.cutoff / (0.5 * sr)
        return signal.butter(self.order, wn, btype="high", output="sos")


@register_filter("bandpass", help="Butterworth BPF (SOS). Params: low (Hz), high (Hz), order (4)")
class Bandpass(_IIRSOS):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.low = float(params.get("low"))
        self.high = float(params.get("high"))
        self.order = int(params.get("order", 4))
    def _design(self, sr: int):
        wn = [self.low / (0.5 * sr), self.high / (0.5 * sr)]
        return signal.butter(self.order, wn, btype="band", output="sos")

@register_filter(
    "compress",
    help=("Smooth compressor. Params: "
          "threshold_db (-24), ratio (4), attack_ms (5), release_ms (50), "
          "makeup_db (0), knee_db (6), stereo_link (true|false)"))
class Compressor(AudioFilter):
    """
    Stream-safe compressor:
      • Peak detector with attack/release smoothing (per channel)
      • Soft-knee (knee_db) to avoid zipper/clicks near threshold
      • Gain smoothing (separate attack/release on gain reduction envelope)
      • Optional stereo link so L/R don’t image-wobble
    """
    def __init__(self, params: Dict[str, Any]):
        self.th = float(params.get("threshold_db", -24.0))
        self.ratio = max(1.0, float(params.get("ratio", 4.0))) # Ensure ratio >= 1
        self.atk_ms = float(params.get("attack_ms", 5.0))
        self.rel_ms = float(params.get("release_ms", 50.0))
        self.mk_db = float(params.get("makeup_db", 0.0))
        self.knee_db = float(params.get("knee_db", 6.0))
        # Ensure stereo_link defaults to True if not specified or invalid string
        sl_param = params.get("stereo_link", True)
        if isinstance(sl_param, str):
            self.stereo_link = sl_param.lower() == 'true'
        else:
            self.stereo_link = bool(sl_param)


        # cached / stream state
        self._env: Optional[np.ndarray] = None         # detector env per channel (lin)
        self._gr_db: Optional[np.ndarray] = None       # gain reduction (dB) smoothed per channel
        self._atk_a: Optional[float] = None
        self._rel_a: Optional[float] = None
        self._g_atk_a: Optional[float] = None         # gain smoothing (faster attack)
        self._g_rel_a: Optional[float] = None         # gain smoothing (slower release)
        self._mk_lin: float = _db_to_lin(self.mk_db)
        self._sr: Optional[int] = None # Store sample rate

    @staticmethod
    def _alpha(ms: float, sr: int) -> float:
        # one-pole smoothing coefficient
        if sr <= 0: return 0.0 # Avoid division by zero
        # Clamp ms to avoid issues with very small values
        ms = max(0.1, ms) # Minimum 0.1 ms
        return float(np.exp(-1.0 / (ms * 1e-3 * sr)))


    def _ensure(self, C: int, sr: int):
        # Re-calculate coefficients if SR changed
        if self._sr != sr:
            self._sr = sr
            # detector smoothing (peak follower)
            self._atk_a = self._alpha(self.atk_ms, sr)
            self._rel_a = self._alpha(self.rel_ms, sr)
            # gain smoothing — use slightly *quicker* attack on GR changes, slower release
            self._g_atk_a = self._alpha(max(1.0, 0.6 * self.atk_ms), sr)
            self._g_rel_a = self._alpha(self.rel_ms * 1.1, sr)
            # Reset state if SR changes, as coefficients depend on it
            self._env = None
            self._gr_db = None

        # Initialize state arrays if needed or if channel count changed
        if self._env is None or self._env.shape[0] != C:
            self._env = np.zeros((C,), dtype=np.float32)
        if self._gr_db is None or self._gr_db.shape[0] != C:
             self._gr_db = np.zeros((C,), dtype=np.float32)


    def _soft_knee_gr_db(self, level_db: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous gain reduction in dB using soft-knee.
        level_db: (C,) current detector level in dBFS
        Returns GR in dB (negative or zero).
        """
        T = self.th
        R = self.ratio # Assumed >= 1
        K = max(0.0, self.knee_db)

        # Calculate overshoot relative to threshold
        over = level_db - T

        # Calculate gain reduction based on knee
        if K <= 1e-6:
            # Hard knee: GR is proportional to overshoot above threshold
            gr = np.where(over > 0.0, -(1.0 - 1.0 / R) * over, 0.0)
        else:
            # Soft knee
            halfK = 0.5 * K
            # Calculate boundaries of the knee region
            knee_start = T - halfK
            knee_end = T + halfK

            # Conditions for different regions
            below_knee = level_db <= knee_start
            above_knee = level_db >= knee_end
            in_knee = ~(below_knee | above_knee)

            # Initialize gain reduction array
            gr = np.zeros_like(level_db, dtype=np.float32)

            # Below knee: No gain reduction
            gr[below_knee] = 0.0

            # Above knee: Standard compression formula
            gr[above_knee] = -(1.0 - 1.0 / R) * over[above_knee]

            # Inside knee: Quadratic interpolation
            # Calculate the position within the knee (0 to K)
            knee_pos = level_db[in_knee] - knee_start
            # Apply quadratic formula for smooth transition
            # Formula derived from standard soft knee equations
            gr[in_knee] = -(1.0 - 1.0/R) * (knee_pos**2) / (2.0 * K)


        return gr.astype(np.float32)


    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float32, copy=False)
        n, C = x.shape
        self._ensure(C, sr)

        # Make sure coefficients are valid after _ensure
        atk = self._atk_a; rel = self._rel_a
        gatk = self._g_atk_a; grel = self._g_rel_a
        # Ensure coefficients are floats, handle potential None from _ensure edge cases
        atk = float(atk) if atk is not None else 0.0
        rel = float(rel) if rel is not None else 0.0
        gatk = float(gatk) if gatk is not None else 0.0
        grel = float(grel) if grel is not None else 0.0

        # Ensure state arrays are correctly initialized
        env = self._env if self._env is not None else np.zeros((C,), dtype=np.float32)
        gr_db = self._gr_db if self._gr_db is not None else np.zeros((C,), dtype=np.float32)

        mk = self._mk_lin

        out = np.empty_like(x, dtype=np.float32)

        # process sample-by-sample (vectorized across channels)
        for i in range(n):
            s = x[i]

            # 1) Detector (peak follower with AR smoothing in linear domain)
            abs_s = np.abs(s)
            # env = max(abs_s, env * a + (1-a) * abs_s) — write branchless using different a per element
            faster = abs_s > env
            # Select attack alpha if input > current envelope, else release alpha
            a = np.where(faster, atk, rel)
            # Update envelope using the selected alpha
            env = a * env + (1.0 - a) * abs_s

            # 2) Convert to dB (handle potential log10(0))
            level_db = 20.0 * np.log10(np.maximum(env, _DEF_EPS)).astype(np.float32)

            # 3) Instantaneous GR (dB) with soft-knee
            inst_gr_db = self._soft_knee_gr_db(level_db)

            # 4) Stereo link (use max reduction across channels if enabled)
            if self.stereo_link and C > 1:
                # Find the minimum gain reduction value (most negative dB)
                link_gr = np.min(inst_gr_db)
                # Apply this maximum reduction to all channels
                inst_gr_db[:] = link_gr # Use slicing to modify in place

            # 5) Smooth GR in dB (separate attack/release for gain changes)
            # Determine if gain reduction needs to increase (inst_gr_db < gr_db)
            go_faster_gr = inst_gr_db < gr_db
            # Select gain attack alpha if increasing reduction, else gain release alpha
            ag = np.where(go_faster_gr, gatk, grel)
            # Update smoothed gain reduction using selected alpha
            gr_db = ag * gr_db + (1.0 - ag) * inst_gr_db

            # 6) Apply makeup gain and final gain reduction
            # Convert smoothed GR (dB) to linear gain
            gain_lin = (10.0 ** (gr_db / 20.0)) * mk
            # Apply gain to the input signal
            out[i] = s * gain_lin

        # store state for next block
        self._env = env
        self._gr_db = gr_db

        # final safety clipping
        np.clip(out, -1.0, 1.0, out=out)
        return out

@register_filter("limiter", help="Peak limiter. Params: ceiling_db (-1), release_ms (50)")
class Limiter(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self.ceil = float(params.get("ceiling_db", -1.0))
        self.rel_ms = float(params.get("release_ms", 50.0))
        self.env: Optional[np.ndarray] = None
        self._rel_alpha: Optional[float] = None
        self._ceil_lin: float = _db_to_lin(self.ceil)

    def _ensure(self, C: int, sr: int):
        if self.env is None or self.env.shape[0] != C or self._rel_alpha is None:
            self.env = np.zeros((C,), dtype=np.float32)
            self._rel_alpha = float(np.exp(-1.0 / max(1, int(self.rel_ms * 1e-3 * sr))))

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float32, copy=False)
        n, C = x.shape
        self._ensure(C, sr)
        env = self.env
        rel = self._rel_alpha
        out = np.empty_like(x, np.float32)
        ceil_lin = self._ceil_lin

        for i in range(n):
            # decay
            env *= rel
            # update if above ceiling
            peak = np.max(np.abs(x[i]))
            needed = peak / max(ceil_lin, _DEF_EPS)
            if needed > 1.0:
                np.maximum(env, needed, out=env)
            gain = 1.0 / np.maximum(env, 1.0)
            out[i] = x[i] * gain

        np.clip(out, -ceil_lin, ceil_lin, out=out)
        return out


@register_filter("mixdown", help="Stereo→Mono average")
class Mixdown(AudioFilter):
    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        if block.shape[1] == 1:
            return block
        return np.mean(block, axis=1, keepdims=True).astype(np.float32)

@register_filter(
    "parametriceq",
    help=("Multi-band parametric EQ. Use --extra eq.bandN.* "
          "(types: peak, lowshelf, highshelf, lowpass, highpass) "
          "Params: type, freq, gain_db (for peak/shelf), q (for peak), slope (for shelf, default 1), order (for lp/hp, default 2)")
)
class ParametricEQ(AudioFilter):
    def __init__(self, params: Dict[str, Any]):
        self._params = params
        self._band_defs: List[Dict[str, Any]] = [] # Parsed band definitions
        self._bands_sos: List[np.ndarray] = []    # List of SOS arrays for each band
        self._bands_zi: List[np.ndarray] = []     # List of zi state arrays for each band
        self._sr: Optional[int] = None
        self._C: Optional[int] = None
        self._built = False
        self._parse_bands() # Parse definitions initially

    def _parse_bands(self):
        """Parses band definitions from parameters like eq.bandN.*"""
        self._band_defs = []
        # Find all keys matching the pattern eq.band<number>.<param>
        # Updated regex to include 'order'
        band_pattern = re.compile(r"eq\.band(\d+)\.(type|freq|gain_db|q|slope|order)$")
        bands_data: Dict[int, Dict[str, Any]] = {}

        for key, value in self._params.items():
            match = band_pattern.match(key)
            if match:
                band_index = int(match.group(1))
                param_name = match.group(2)
                if band_index not in bands_data:
                    bands_data[band_index] = {}
                # Store the raw value, type conversion happens during build
                bands_data[band_index][param_name] = value

        # Sort by band index and store valid definitions
        for i in sorted(bands_data.keys()):
            band_info = bands_data[i]
            # Basic validation: ensure type and freq are present
            if 'type' in band_info and 'freq' in band_info:
                 # Check specific requirements based on type
                 b_type_check = str(band_info['type']).lower()
                 if b_type_check in ['peak', 'lowshelf', 'highshelf']:
                     if 'gain_db' not in band_info:
                         print(f"Warning: ParametricEQ band {i} ({b_type_check}) missing 'gain_db', skipping.", flush=True)
                         continue
                 elif b_type_check not in ['lowpass', 'highpass']:
                      # If it's not a known type requiring only freq, check gain_db just in case
                      if 'gain_db' not in band_info:
                          print(f"Warning: ParametricEQ band {i} ({b_type_check}) potentially incomplete (missing gain_db?), skipping.", flush=True)
                          continue

                 # Set defaults for optional params if missing
                 band_info.setdefault('q', 1.0) # Default Q for peak
                 band_info.setdefault('slope', 1.0) # Default slope for shelf
                 band_info.setdefault('order', 2) # Default order for LP/HP
                 self._band_defs.append(band_info)
            else:
                 print(f"Warning: ParametricEQ band {i} missing 'type' or 'freq', skipping. Got: {band_info}", flush=True)

    def _build(self, sr: int, C: int):
        """Builds the SOS filters and initializes zi state based on parsed bands."""
        self._bands_sos = []
        self._bands_zi = []
        self._sr = sr
        self._C = C
        build_successful = True

        for band_def in self._band_defs:
            try:
                b_type = str(band_def['type']).lower()
                b_freq = float(band_def['freq'])
                # Only get gain_db if relevant
                b_gain = float(band_def['gain_db']) if 'gain_db' in band_def else 0.0
                b_q = float(band_def['q'])
                b_slope = float(band_def.get('slope', 1.0))
                b_order = int(band_def.get('order', 2)) # Default order for LP/HP

                sos = None
                if b_type == 'peak':
                    sos = _iir_peaking_sos(sr, b_freq, b_q, b_gain)
                elif b_type == 'lowshelf':
                    sos = _iir_shelf_sos(sr, b_freq, b_gain, shelf_type='low', slope=b_slope)
                elif b_type == 'highshelf':
                    sos = _iir_shelf_sos(sr, b_freq, b_gain, shelf_type='high', slope=b_slope)
                # --- NEW TYPES ---
                elif b_type == 'lowpass':
                    # gain_db and q/slope are ignored for lowpass
                    wn_lp = max(0.0001, min(0.9999, b_freq / (0.5 * sr)))
                    sos = signal.butter(b_order, wn_lp, btype='low', output='sos')
                elif b_type == 'highpass':
                    # gain_db and q/slope are ignored for highpass
                    wn_hp = max(0.0001, min(0.9999, b_freq / (0.5 * sr)))
                    sos = signal.butter(b_order, wn_hp, btype='high', output='sos')
                # --- END NEW TYPES ---
                else:
                    print(f"Warning: Unknown EQ band type '{b_type}', skipping band.", flush=True)
                    continue # Skip this band

                # Validate SOS shape before appending
                if sos is not None and sos.ndim == 2 and sos.shape[1] == 6:
                    self._bands_sos.append(sos.astype(np.float64))
                    # Initialize zi state for this SOS and C channels
                    zi_one = signal.sosfilt_zi(sos) # Shape (n_sections, 2)
                    # Tile to (n_sections, 2, C) - Correct shape for sosfilt axis=0
                    zi_band = np.tile(zi_one[..., np.newaxis], (1, 1, C)).astype(np.float64)
                    self._bands_zi.append(zi_band)
                else:
                    print(f"Warning: Failed to generate valid SOS for band {band_def}, skipping.", flush=True)

            except Exception as e:
                print(f"Error processing EQ band {band_def}: {e}", flush=True)
                build_successful = False # Mark build as potentially incomplete

        self._built = build_successful and bool(self._bands_sos) # Built only if successful and has bands


    def _ensure(self, sr: int, C: int):
        """Rebuilds if SR or C changes, or if not built yet."""
        # Check if zi states match current channel count
        # Ensure _bands_zi is not empty before checking shape
        zi_channels_match = True
        if self._bands_zi:
             # Check the channel dimension (last dimension) of the first zi state array
             zi_channels_match = self._bands_zi[0].shape[2] == C


        if not self._built or self._sr != sr or self._C != C or not zi_channels_match:
            self._parse_bands() # Re-parse in case params changed dynamically (though unlikely here)
            self._build(sr, C)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = block.astype(np.float64) # Use float64 for accuracy
        n, C = x.shape
        self._ensure(sr, C)

        if not self._built or not self._bands_sos:
            return block # Return original if not built or no bands defined

        y = x # Start with input
        try:
            # Apply each band's filter sequentially
            for i in range(len(self._bands_sos)):
                sos = self._bands_sos[i]
                zi = self._bands_zi[i]
                 # Ensure zi state has the correct channel dimension C
                if zi.shape[2] != C:
                     print(f"Warning: ParametricEQ zi state channel mismatch ({zi.shape[2]} != {C}), skipping filter application.", flush=True)
                     # Attempt to rebuild might be too aggressive here, just skip
                     continue # Skip this band for this block

                # Apply filter, update zi state IN PLACE
                y, self._bands_zi[i] = signal.sosfilt(sos, y, axis=0, zi=zi)

            return y.astype(np.float32) # Convert back to float32
        except Exception as e:
            print(f"Error during ParametricEQ processing: {e}", flush=True)
            # Reset state? Consider if this helps or hinders recovery
            # self._reset_state()
            return block # Return original on error

# ---------- Mastering (multi-band glue + HPF + limiter) ----------
@register_filter(
    "master",
    help=(
        "One-stop mastering: HPF + multiband compression + limiter. "
        "Params: low_cut (30), x1 (200), x2 (4000), "
        "low.th (-24) low.ratio (2), mid.th (-20) mid.ratio (2.5), high.th (-22) high.ratio (2), "
        "attack_ms (5), release_ms (60), makeup_db (1), ceiling_db (-1)"
    ),
)
class Master(AudioFilter):
    """
    Gentle, stream-safe master:
      - Highpass @ low_cut Hz to clean subsonics
      - 3 bands via Butterworth: [0..x1], [x1..x2], [x2..Nyquist]
      - Per-band compression with shared attack/release + per-band threshold/ratio
      - Make-up gain then brick limiter at ceiling_db

    Notes:
      * Optimized to minimize allocations and Python-level work per block.
      * Gains & time constants are cached; operations are done in-place where safe.
    """
    def __init__(self, params: Dict[str, Any]):
        # crossover & HPF
        self.low_cut = float(params.get("low_cut", 30.0))
        self.x1 = float(params.get("x1", 200.0))
        self.x2 = float(params.get("x2", 4000.0))
        self._order = int(params.get("order", 2))  # 2 is much faster than 4 and quite transparent

        # shared timing / makeup
        self.atk = float(params.get("attack_ms", 5.0))
        self.rel = float(params.get("release_ms", 60.0))
        self.mk_db = float(params.get("makeup_db", 1.0))
        self.ceiling = float(params.get("ceiling_db", -1.0))
        self._mk_lin = _db_to_lin(self.mk_db)  # cache

        # per-band thresholds/ratios
        self.low_th   = float(params.get("low.th",  params.get("low_threshold_db",  -24.0)))
        self.mid_th   = float(params.get("mid.th",  params.get("mid_threshold_db",  -20.0)))
        self.high_th  = float(params.get("high.th", params.get("high_threshold_db", -22.0)))
        self.low_ratio  = float(params.get("low.ratio",  params.get("low_ratio",  2.0)))
        self.mid_ratio  = float(params.get("mid.ratio",  params.get("mid_ratio",  2.5)))
        self.high_ratio = float(params.get("high.ratio", params.get("high_ratio", 2.0)))

        # Lazy subfilters (init after SR known)
        self._initd = False
        self._sr: Optional[int] = None
        self._C: Optional[int] = None

        self._hpf: Optional[Highpass] = None
        self._lp:  Optional[Lowpass]  = None
        self._bp:  Optional[Bandpass] = None
        self._hp:  Optional[Highpass] = None
        self._cL:  Optional[Compressor] = None
        self._cM:  Optional[Compressor] = None
        self._cH:  Optional[Compressor] = None
        self._cl:  Optional[Limiter] = None

        # scratch buffers reused across calls (shape updated on first process)
        self._mix_buf: Optional[np.ndarray] = None  # mix/sum buffer

    def _ensure(self, sr: int, C: int):
        if self._initd and self._sr == sr and self._C == C:
            return

        self._sr, self._C = sr, C

        # filters (keep orders low for real-time)
        self._hpf = Highpass({"cutoff": self.low_cut, "order": 2})
        self._lp  = Lowpass( {"cutoff": self.x1,     "order": self._order})
        self._bp  = Bandpass({"low": self.x1, "high": self.x2, "order": self._order})
        self._hp  = Highpass({"cutoff": self.x2,     "order": self._order})

        # compressors
        shared = {"attack_ms": self.atk, "release_ms": self.rel, "makeup_db": 0.0}
        self._cL = Compressor({"threshold_db": self.low_th,  "ratio": self.low_ratio,  **shared})
        self._cM = Compressor({"threshold_db": self.mid_th,  "ratio": self.mid_ratio,  **shared})
        self._cH = Compressor({"threshold_db": self.high_th, "ratio": self.high_ratio, **shared})

        # limiter
        self._cl = Limiter({"ceiling_db": self.ceiling, "release_ms": 50.0})

        # warm-up to allocate/prime states inside each filter once
        z = np.zeros((0, C), np.float32)
        for f in (self._hpf, self._lp, self._bp, self._hp, self._cL, self._cM, self._cH, self._cl):
            f.process(z, sr)

        # scratch buffer will be created on first real block
        self._mix_buf = None
        self._initd = True

    @staticmethod
    def _as_float32(x: np.ndarray) -> np.ndarray:
        # Avoid copies unless we must convert dtype
        return x if x.dtype == np.float32 else x.astype(np.float32, copy=False)

    def _ensure_mixbuf(self, n: int, C: int):
        mb = self._mix_buf
        if mb is None or mb.shape[0] != n or mb.shape[1] != C:
            # allocate once; reused every call
            self._mix_buf = np.empty((n, C), dtype=np.float32)

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = self._as_float32(block)
        n, C = x.shape
        self._ensure(sr, C)
        self._ensure_mixbuf(n, C)

        # 1) Subsonic clean (in -> y)
        y = self._hpf.process(x, sr)

        # 2) Split bands (avoid extra copies later by summing in-place)
        low  = self._lp.process(y, sr)
        mid  = self._bp.process(y, sr)
        high = self._hp.process(y, sr)

        # 3) Per-band compression (each returns a view/new array; that’s fine)
        low  = self._cL.process(low,  sr)
        mid  = self._cM.process(mid,  sr)
        high = self._cH.process(high, sr)

        # 4) Sum bands into reusable mix buffer, then apply makeup in-place
        mix = self._mix_buf
        np.add(low, mid, out=mix)
        mix += high
        mix *= self._mk_lin

        # 5) Safety limiter (returns array). If it returns a view of input,
        #    give it mix; otherwise just accept new arr and clip in-place.
        mix = self._cl.process(mix, sr)

        # 6) final clip in-place (avoid another allocation)
        np.clip(mix, -1.0, 1.0, out=mix)
        return mix

    def flush(self) -> Optional[np.ndarray]:
        # IIRs/comp/limiter don’t emit tails here
        return None
