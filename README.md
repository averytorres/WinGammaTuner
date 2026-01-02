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

## 🔧 Features

- Real-time display gamma ramp control via Windows GDI (`SetDeviceGammaRamp`)
- Parametric 256-step LUT generation including:
  - Gamma exponent with optional offset
  - Black floor lift for shadow detail preservation
  - Nonlinear shadow-region lift with configurable cutoff
  - Shadow-only perceptual contrast shaping (power-based)
  - Optional mid-shadow sigmoid contrast shaping for silhouette separation
  - Optional midtone shaping (advanced)
  - Highlight compression with near-white (UI / HUD) preservation
  - Configurable HUD / highlight clamp
  - Global vibrance scaling
  - Global RGB channel multipliers
  - Shadow-only color bias for silhouette clarity (FPS-safe)
  - Optional opponent-channel shadow tuning with luminance preservation
  - Global Shadow Pop Strength modifier
- Optional scene-aware enhancements (disabled by default):
  - Histogram-aware shadow adaptation
  - Edge-aware shadow contrast scaling
  - HUD-aware highlight exclusion
- Two independent profiles:
  - **INDOOR** — controlled or low-light environments
  - **OUTDOOR** — bright or high-glare environments
- Instant profile switching via global hotkeys
- Lightweight Tkinter GUI with live updates
- Modifier-aware sliders for precision tuning:
  - Normal drag: full-range adjustment
  - Shift + drag: medium-granularity
  - Ctrl + drag: fine-grained adjustment
- Persistent JSON configuration with debounced auto-save
- Cached curve generation with CRC-based gamma ramp validation
- Threaded execution to avoid blocking UI or hotkey handling
- Automatic restoration of the identity gamma ramp when no profile is active

---

## 🖥 Requirements

- Windows OS
- Python 3.9+
- Display driver supporting `SetDeviceGammaRamp`

```bash
pip install numpy pynput orjson
```
---
## Usage

```bash
python gamma_control.py
```
---
## Global hotkeys

- F6 → Open or show settings GUI
- F8 → Toggle INDOOR profile
- F9 → Toggle OUTDOOR profile

Only one profile may be active at a time. Disabling all profiles restores the
identity gamma ramp.

## Notes

- Gamma ramps affect global display output at the driver level
- Extreme values may cause banding, clipping, or eye strain
- Advanced controls can significantly alter visual output
- Not recommended for Remote Desktop or unsupported GPUs

## Files

- gamma_control.py — main application
- gamma_config.json — auto-generated persistent configuration

## License

MIT
