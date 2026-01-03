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
```
python main.py
```

---

## Global Hotkeys

- **F6** — Open or show settings GUI
- **F8** — Toggle INDOOR profile
- **F9** — Toggle OUTDOOR profile

Only one profile may be active at a time. Disabling all profiles restores the
identity gamma ramp.

---

## Project Structure

- **main.py**  
  Application entry point. Wires together the pipeline, controller, scene analyzer,
  UI, and hotkeys.

- **gamma_pipeline.py**  
  Gamma LUT construction and all color and contrast shaping logic.

- **scene_analyzer.py**  
  Low-resolution desktop sampling and scene analysis for adaptive features.

- **gamma_controller.py**  
  Win32 gamma ramp application, CRC validation, and profile switching.

- **gamma_ui.py**  
  Tkinter-based graphical interface, sliders, collapsible sections, and live updates.

- **gamma_config.json**  
  Auto-generated persistent configuration. This file represents runtime state and
  should not be version-controlled.

---

## Notes and Warnings

- Gamma ramps affect global display output at the driver level
- Extreme values may cause banding, clipping, or eye strain
- Advanced controls can significantly alter visual output
- Not recommended for Remote Desktop sessions or unsupported GPUs

---

## License

MIT

