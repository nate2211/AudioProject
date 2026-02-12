# Gemini Audio Engine Pro

An ultra-low-artifact, streaming-capable audio workstation and processing engine. **Gemini Audio Pro** combines a high-performance Python DSP backend with an FL Studio-inspired sequencer for real-time manipulation, time-stretching, and spectral processing.

---

## 🚀 Core Features

### 1. High-Fidelity DSP Suite
* **WSOLA Anti-Buzz Edition:** High-quality time-stretching with pitch preservation using normalized, mean-removed, pre-emphasized correlation and equal-power crossfades.
* **Phase Vocoder:** Low-artifact stretching with identity phase locking and peak tracking to prevent "phasiness" and preserve transients.
* **Clarity Chain:** A curated mastering-grade pipeline including De-Essing, Presence EQ, Air Shelving, and Lookahead Limiting.
* **Dynamic FX Rack:** 8 slots per track supporting Lowpass, Highpass, Bandpass, Tilt EQ, Transient Shapers, and Mid/Side Imagers.

### 2. "FL-ish" Sequencer
* **Multi-Track Arrangement:** Drag-and-drop clip placement with intuitive trimming and moving.
* **Snap-to-Grid:** Quantize movement to 1/8, 1/4, 1/2, Beats, or Bars based on Master BPM.
* **Non-Destructive Pitch:** Per-track pitch shifting via FFmpeg (Rubberband/Atempo) with a scale-aware picker.
* **SYNC Mode:** Real-time WSOLA stretching—drastically resize clips and maintain rhythmic sync automatically.

### 3. Professional Routing & IO
* **WASAPI Loopback:** (Windows) Tap system audio directly to process Spotify, YouTube, or browser audio through the engine.
* **YouTube Integration:** Load audio directly from a URL using `yt-dlp`.
* **High-Quality Export:** Mix down your arrangement to 16-bit WAV or MP3 with automated TPDF dithering and a safety limiting chain.

---

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **GUI:** PyQt6
* **DSP:** NumPy, SciPy (SOS Filtering, FFT)
* **Audio IO:** SoundDevice (PortAudio), SoundCard (MediaFoundation fallback)
* **File Handling:** PyDub, FFmpeg, yt-dlp

---

## 📦 Installation

### Prerequisites
1.  **FFmpeg:** Required for audio decoding, pitch shifting, and YouTube support. Ensure `ffmpeg.exe` and `ffprobe.exe` are in your system PATH or the project root.
2.  **Python Requirements:**
    ```bash
    pip install numpy scipy PyQt6 sounddevice soundcard pydub yt-dlp requests
    ```

### Running the App
```bash
python main.py
