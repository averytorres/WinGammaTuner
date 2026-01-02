# WinGammaTuner

**WinGammaTuner** is a Windows utility for real-time display gamma correction and color tuning. Designed for users who want precise control over indoor and outdoor display profiles, it offers hotkey switching, live GUI adjustment, and persistent configuration.

## 🔧 Features

- Real-time gamma and color ramp control via Windows GDI
- INDOOR and OUTDOOR preset modes
- Hotkey toggle: F6 (GUI), F8 (Indoor), F9 (Outdoor)
- Tkinter-based GUI with fine-tuning sliders
- Persistent JSON config with debounce saving
- Efficient LUT generation with ramp caching and CRC validation

## 🖥 Requirements

- Windows OS
- Python 3.9+
- Compatible display hardware (SetDeviceGammaRamp support)
- Python dependencies:
  ```bash
  pip install numpy pynput orjson
  
## 🚀 Usage

1. Run `gamma_control.py`

2. Use hotkeys:
   - `F6`: Open/close settings GUI  
   - `F8`: Toggle INDOOR mode  
   - `F9`: Toggle OUTDOOR mode

3. Adjust values via GUI as needed — changes apply instantly and are auto-saved

## ⚠️ Notes

- Not recommended for use via Remote Desktop  
- Avoid extreme gamma values — may destabilize display drivers

## 📁 Files

- `gamma_control.py`: Main script  
- `gamma_config.json`: Auto-generated config (saved in script directory)

## 📄 License

MIT
