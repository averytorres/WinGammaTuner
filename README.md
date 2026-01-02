# WinGammaTuner

**WinGammaTuner** is a Windows utility for real-time display gamma and color curve
control using the Win32 GDI gamma ramp API. It allows precise, low-latency tuning
of display output through live LUT generation, with instant profile switching,
global hotkeys, and persistent configuration.

The tool is designed for users who want fine-grained control over display behavior
in different lighting conditions, with separate **INDOOR** and **OUTDOOR** profiles
that can be tuned independently and toggled instantly.

## 🔧 Features

- Real-time display gamma ramp control via Windows GDI (`SetDeviceGammaRamp`)
- Parametric 256-step LUT generation with:
  - Gamma exponent and offset
  - Shadow and midtone boosting
  - Highlight compression to reduce clipping
  - Global vibrance scaling
  - Independent RGB channel multipliers
- Two independent profiles:
  - **INDOOR** (controlled / low-light environments)
  - **OUTDOOR** (bright or high-glare environments)
- Instant profile switching via global hotkeys
- Lightweight Tkinter GUI with live updates
- Modifier-aware sliders for precision tuning:
  - Normal drag for full-range adjustment
  - **Shift + drag** for controlled coarse interpolation
  - **Ctrl + drag** for fine-grained micro-adjustments
- Persistent JSON configuration with debounced auto-save
- Efficient updates using cached curves and CRC-based ramp validation
- Automatic restoration of the identity gamma ramp when no profile is active

## 🖥 Requirements

- Windows OS
- Python 3.9+
- Display driver supporting `SetDeviceGammaRamp`
- Python dependencies:
  ```bash
  pip install numpy pynput orjson
  
## 🚀 Usage

1. Run the script:
   ```bash
   python gamma_control.py
   
Use global hotkeys:
- `F6` → Open or show settings GUI
- `F8` → Toggle **INDOOR** profile
- `F9` → Toggle **OUTDOOR** profile

Adjust parameters in the GUI:
- Changes apply instantly
- Configuration is saved automatically
- Only one profile can be active at a time
- Disabling all profiles restores the default (identity) gamma ramp
- Sliders support modifier-based precision control:
  - Normal drag: full-range adjustment
  - **Shift + drag**: controlled coarse interpolation
  - **Ctrl + drag**: fine-grained micro-adjustments

## ⚠️ Notes & Warnings

- Gamma ramps affect the global display output at the driver level
- Extreme values may cause banding, clipping, or eye strain
- Not recommended for use over Remote Desktop or unsupported GPUs

## 📁 Files

- `gamma_control.py` — Main application script
- `gamma_config.json` — Auto-generated persistent configuration file

## 📄 License

MIT
