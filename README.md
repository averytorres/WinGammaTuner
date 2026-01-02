# WinGammaTuner

**WinGammaTuner** is a Windows utility for real-time display gamma and color curve
control using the Win32 GDI gamma ramp API. It provides low-latency, system-wide
display tuning through live 256-step LUT generation, with instant profile switching,
global hotkeys, and persistent configuration.

The tool is designed with a strong focus on **competitive FPS visibility**, while
preserving contrast, highlight stability, **UI legibility**, and predictable output.
Two independent profiles — **INDOOR** and **OUTDOOR** — can be tuned separately and
toggled instantly to adapt to different lighting conditions and use cases.

---

## 🔧 Features

- Real-time display gamma ramp control via Windows GDI (`SetDeviceGammaRamp`)
- Parametric 256-step LUT generation including:
  - Gamma exponent with optional offset
  - Black floor lift for shadow detail preservation
  - Nonlinear shadow-region lift with configurable cutoff
  - **Shadow-only perceptual micro-contrast shaping (logarithmic)**
  - ✅ **Optional mid-shadow sigmoid contrast shaping** for enhanced silhouette separation
  - Optional midtone shaping (advanced)
  - Highlight compression with **near-white (UI / HUD) preservation**
  - ✅ Configurable HUD / highlight clamp to prevent UI distortion
  - Global vibrance scaling
  - Global RGB channel multipliers
  - Shadow-only **luminance bias** for silhouette clarity (FPS-safe)
  - Optional shadow-only chroma suppression for haze reduction
  - ✅ **Global Shadow Pop Strength modifier** for real-time silhouette & clarity boost
- Two independent profiles:
  - **INDOOR** — controlled or low-light environments
  - **OUTDOOR** — bright or high-glare environments
- Instant profile switching via global hotkeys
- Lightweight Tkinter GUI with live updates
- Modifier-aware sliders for precision tuning:
  - Normal drag: full-range adjustment
  - **Shift + drag**: medium-granularity interpolation
  - **Ctrl + drag**: fine-granularity micro-adjustments
- Range-aware slider granularity with quantized value display
- Optional **Advanced Controls** tier:
  - Reveals high-impact parameters such as shadow RGB bias
  - Hidden advanced controls remain active to ensure stable output
- Persistent JSON configuration with debounced auto-save
- Efficient updates using cached curves and CRC-based ramp validation
- ✅ Threaded execution via `ThreadPoolExecutor` to avoid blocking UI or hotkey input
- Automatic restoration of the identity gamma ramp when no profile is active

---

## 🖥 Requirements

- Windows OS
- Python 3.9+
- Display driver supporting `SetDeviceGammaRamp`
- Python dependencies:

```bash
pip install numpy pynput orjson
```
## 🚀 Usage

### ▶️ Run the script

```bash
python gamma_control.py
```
### 🔑 Global hotkeys

F6  → Open or show settings GUI  
F8  → Toggle INDOOR profile  
F9  → Toggle OUTDOOR profile  
---
🛠️ Adjust Parameters in the GUI

- Changes apply instantly  
- Configuration is saved automatically  
- Only one profile can be active at a time  
- Disabling all profiles restores the identity gamma ramp  

---

### 🎚️ Slider precision controls
---
- Normal drag      → full-range adjustment  
- Shift + drag     → controlled interpolation  
- Ctrl  + drag     → fine-grained micro-adjustments  

### ⚠️ Notes & Warning

- Gamma ramps affect the global display output at the driver level  
- Extreme values may cause banding, clipping, or eye strain  
- Advanced controls can significantly alter visual output  
- Not recommended for use over Remote Desktop or unsupported GPUs  

---

### 📁 Files
---
- gamma_control.py      — Main application script  
- gamma_config.json     — Auto-generated persistent configuration file  

### 📄 License

MIT
