# WinGammaTuner

**WinGammaTuner** is a Windows utility for real-time display gamma and color curve
control using the Win32 GDI gamma ramp API. It provides low-latency, system-wide
display tuning through live 256-step LUT generation, with instant profile switching,
global hotkeys, and persistent configuration.

The tool is designed with a strong focus on **competitive FPS visibility**, while
preserving contrast, highlight stability, UI legibility, and predictable output.
Two independent profiles — **INDOOR** and **OUTDOOR** — can be tuned separately and
toggled instantly to adapt to different lighting conditions.

---

## Features

- Real-time display gamma ramp control via Windows GDI (`SetDeviceGammaRamp`)
- Live generation and application of 256-step gamma lookup tables
- Parametric LUT shaping including:
  - Gamma exponent with optional offset
  - Black floor lift for shadow detail preservation
  - Nonlinear shadow-region lift with configurable cutoff
  - Shadow-only perceptual contrast shaping
  - Optional mid-shadow sigmoid contrast shaping for silhouette separation
  - Optional midtone shaping
  - Highlight compression with near-white UI / HUD preservation
  - Configurable highlight clamp for HUD stability
  - Global vibrance scaling
  - Global RGB channel multipliers
  - Shadow-only color bias for silhouette clarity
  - Optional opponent-channel shadow tuning with luminance preservation
  - Global Shadow Pop Strength modifier
- Optional scene-aware enhancements:
  - Histogram-aware shadow adaptation
  - Edge-aware shadow contrast scaling
  - Motion-aware shadow enhancement
  - HUD-aware highlight exclusion
- Two independent tuning profiles:
  - **INDOOR** — controlled or low-light environments
  - **OUTDOOR** — bright or high-glare environments
- Instant profile switching via global hotkeys
- Lightweight Tkinter GUI with live updates
- Modifier-aware sliders for precision control:
  - Normal drag: full-range adjustment
  - Shift + drag: medium granularity
  - Ctrl + drag: fine-grained adjustment
- Persistent JSON configuration with debounced auto-save
- Cached curve generation with CRC-based gamma ramp validation
- Threaded execution to avoid blocking UI or hotkey handling
- Automatic restoration of the identity gamma ramp when no profile is active or on exit

---

## Requirements

- Windows OS
- Python 3.9+
- Display driver supporting `SetDeviceGammaRamp`

Dependencies:
- numpy
- pynput
- orjson

---

## Usage

Run the application from the project directory:

