import numpy as np


class FixedBlockAdapter:
    """
    Wraps a variable-rate filter and forces fixed block output length.
    - Buffers input until inner produces output
    - Buffers output and returns exactly N frames each call
    """
    def __init__(self, inner, *, channels=2):
        self.inner = inner
        self.channels = channels
        self._outbuf = np.zeros((0, channels), dtype=np.float32)

    def reset(self):
        self._outbuf = np.zeros((0, self.channels), dtype=np.float32)
        # best-effort: drop inner state by recreating upstream if you can
        if hasattr(self.inner, "flush"):
            try:
                self.inner.flush()
            except Exception:
                pass

    def process(self, block: np.ndarray, sr: int) -> np.ndarray:
        x = np.asarray(block, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != self.channels:
            # force 2ch
            if x.shape[1] == 1 and self.channels == 2:
                x = np.repeat(x, 2, axis=1)
            else:
                x = x[:, :self.channels]

        n = x.shape[0]

        # Feed inner, append whatever it produces
        y = self.inner.process(x, sr)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[1] != self.channels:
            if y.shape[1] == 1 and self.channels == 2:
                y = np.repeat(y, 2, axis=1)
            else:
                y = y[:, :self.channels]

        if y.size:
            self._outbuf = np.concatenate([self._outbuf, y], axis=0)

        # Return exactly n frames
        if self._outbuf.shape[0] >= n:
            out = self._outbuf[:n].copy()
            self._outbuf = self._outbuf[n:]
            return out

        # Not enough produced yet: pad with zeros (avoids shape errors)
        out = np.zeros((n, self.channels), dtype=np.float32)
        if self._outbuf.shape[0] > 0:
            out[: self._outbuf.shape[0]] = self._outbuf
            self._outbuf = np.zeros((0, self.channels), dtype=np.float32)
        return out


class DCBlocker:
    # y[n] = x[n] - x[n-1] + R*y[n-1]
    def __init__(self, r=0.995):
        self.r = float(r)
        self.x1 = None
        self.y1 = None

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        # Ensure 2D: (frames, channels)
        if x.ndim == 1:
            x = x[:, None]

        ch = int(x.shape[1])

        # Lazy-init state to match channel count
        if self.x1 is None or self.y1 is None or self.x1.shape[0] != ch:
            self.x1 = np.zeros((ch,), dtype=np.float32)
            self.y1 = np.zeros((ch,), dtype=np.float32)

        y = np.empty_like(x, dtype=np.float32)
        r = self.r
        x1 = self.x1
        y1 = self.y1

        for i in range(x.shape[0]):
            xi = x[i]                 # (ch,)
            yi = xi - x1 + r * y1     # (ch,)
            y[i, :] = yi              # <- important: assign with channel axis
            x1 = xi
            y1 = yi

        self.x1 = x1
        self.y1 = y1
        return y

class SoftClipper:
    def __init__(self, drive=1.0):
        self.drive = float(drive)

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        # tanh soft clip
        d = self.drive
        y = np.tanh(x * d) / np.tanh(d)
        return y.astype(np.float32, copy=False)

class Limiter:
    def __init__(self, ceiling_db=-1.0, attack_ms=1.0, release_ms=50.0):
        self.ceiling = 10 ** (float(ceiling_db) / 20.0)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.env = 0.0

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        # Ensure 2D (frames, channels)
        if x.ndim == 1:
            x = x[:, None]

        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # per-frame peak across channels
        peak = np.max(np.abs(x), axis=1)

        atk = np.exp(-1.0 / max(1, int(sr * (self.attack_ms / 1000.0))))
        rel = np.exp(-1.0 / max(1, int(sr * (self.release_ms / 1000.0))))

        env = float(self.env)
        out = np.empty_like(x)

        for i in range(x.shape[0]):
            p = float(peak[i])
            if p > env:
                env = atk * env + (1.0 - atk) * p
            else:
                env = rel * env + (1.0 - rel) * p

            g = 1.0 if env <= 1e-9 else min(1.0, self.ceiling / env)

            out[i, :] = x[i, :] * g   # <- channel-agnostic

        self.env = env
        return out