"""
Gamma Control Utility for Windows
---------------------------------

This script provides real-time, user-adjustable gamma correction and color tuning
for display output, with two configurable modes: INDOOR and OUTDOOR.

Key Features:
- Gamma and color curve manipulation via Windows GDI API
- Preset modes (INDOOR/OUTDOOR) with configurable parameters
- Persistent config saved to JSON (auto-debounced)
- Interactive Tkinter GUI for fine-tuning
- Hotkey support for mode switching and GUI toggle
- Efficient ramp caching and signature checking to prevent redundant updates

System Requirements:
- Windows OS
- Python 3.9+
- Display must support gamma ramp control via SetDeviceGammaRamp
- Dependencies: numpy, tkinter, pynput, orjson

Hotkeys:
- F6  → Open settings GUI
- F8  → Toggle INDOOR mode ON/OFF
- F9  → Toggle OUTDOOR mode ON/OFF

Files:
- gamma_config.json → Automatically generated config file (same directory as script)

Usage:
- Run the script.
- Use hotkeys to toggle modes.
- Open the GUI to adjust parameters live.
- Config changes are saved automatically and applied on the fly.

Caution:
- Gamma changes affect display hardware state. Avoid extreme values.
- Not recommended for remote desktop or unsupported display setups.

"""


import ctypes
import os
import logging
from functools import lru_cache
from threading import Lock, Timer
from enum import Enum
import zlib

import numpy as np
import tkinter as tk
from tkinter import ttk
from pynput import keyboard
import orjson

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "gamma_config.json")

default_config = {
    "GAMMA_INDOOR": 1.30,
    "VIBRANCE_INDOOR": 1.05,
    "SHADOW_BOOST_INDOOR": 1.45,
    "MIDTONE_BOOST_INDOOR": 1.15,
    "HIGHLIGHT_COMPRESS_INDOOR": 0.30,
    "GAMMA_OFFSET_INDOOR": 0.0,
    "RED_MULTIPLIER_INDOOR": 1.0,
    "GREEN_MULTIPLIER_INDOOR": 1.0,
    "BLUE_MULTIPLIER_INDOOR": 1.0,

    "GAMMA_OUTDOOR": 1.12,
    "VIBRANCE_OUTDOOR": 1.04,
    "SHADOW_BOOST_OUTDOOR": 1.20,
    "MIDTONE_BOOST_OUTDOOR": 1.05,
    "HIGHLIGHT_COMPRESS_OUTDOOR": 0.50,
    "GAMMA_OFFSET_OUTDOOR": 0.0,
    "RED_MULTIPLIER_OUTDOOR": 1.0,
    "GREEN_MULTIPLIER_OUTDOOR": 1.0,
    "BLUE_MULTIPLIER_OUTDOOR": 1.0
}

class Mode(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"

root = tk.Tk()
root.withdraw()

# ================= CONFIG MANAGER =================
class ConfigManager:
    def __init__(self, path, defaults):
        self.path = path
        self.defaults = defaults
        self.lock = Lock()
        self.timer = None
        self.data = self.load()

    def load(self):
        try:
            with open(self.path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            data = {}
        for k, v in self.defaults.items():
            data.setdefault(k, v)
        return data

    def save(self):
        with self.lock:
            with open(self.path, "wb") as f:
                f.write(orjson.dumps(self.data))

    def debounce_save(self, delay=0.5):
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(delay, self.save)
        self.timer.daemon = True
        self.timer.start()

config_mgr = ConfigManager(CONFIG_FILE, default_config)
config = config_mgr.data

# ================= GAMMA API =================
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
hdc = user32.GetDC(0)

def to_ramp(arr):
    return (ctypes.c_ushort * 768)(*arr)

def identity_ramp():
    return np.concatenate([
        np.linspace(0, 65535, 256, dtype=np.uint16)
    ] * 3)

# ================= LUT =================
@lru_cache(maxsize=32)
def base_curve(gamma, offset):
    x = np.linspace(0, 1, 256)
    if offset:
        x = np.clip(x + offset, 0, 1)
    if gamma != 1.0:
        x = np.power(x, 1.0 / gamma)
    return x

def fast_array_signature(arr):
    return zlib.crc32(arr.tobytes())

def build_ramp_array(mode):
    m = mode.value.upper()
    p = {k.replace(f"_{m}", ""): config[k] for k in config if k.endswith(m)}

    x = np.linspace(0, 1, 256)
    base = base_curve(p["GAMMA"], p["GAMMA_OFFSET"]).copy()

    base[x < 0.3] *= p["SHADOW_BOOST"]
    base[(x >= 0.3) & (x < 0.6)] *= p["MIDTONE_BOOST"]
    hi = base > 0.8
    base[hi] = 0.8 + (base[hi] - 0.8) * p["HIGHLIGHT_COMPRESS"]
    np.clip(base, 0, 1, out=base)

    scale = 65535 * p["VIBRANCE"]
    r = (base * scale * p["RED_MULTIPLIER"]).astype(np.uint16)
    g = (base * scale * p["GREEN_MULTIPLIER"]).astype(np.uint16)
    b = (base * scale * p["BLUE_MULTIPLIER"]).astype(np.uint16)

    return np.concatenate((r, g, b))

# ================= GAMMA STATE =================
class GammaState:
    def __init__(self):
        self.current_mode = None
        self.last_sig = None
        self.lock = Lock()
        self.timer = None

    def apply(self, arr):
        sig = fast_array_signature(arr)
        if sig == self.last_sig:
            return
        with self.lock:
            gdi32.SetDeviceGammaRamp(hdc, to_ramp(arr))
            self.last_sig = sig

    def rebuild(self):
        if self.current_mode is None:
            self.apply(identity_ramp())
        else:
            self.apply(build_ramp_array(self.current_mode))

    def rebuild_debounced(self, delay=0.05):
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(delay, lambda: root.after(0, self.rebuild))
        self.timer.daemon = True
        self.timer.start()

gamma_state = GammaState()

# ================= GUI =================
bg = "#1e1e1e"
fg = "#ffffff"
accent = "#007acc"
btn = "#2d2d2d"

settings_window = None
scroll_frame = None

def rebuild_gui():
    if scroll_frame is None:
        return

    for w in scroll_frame.winfo_children():
        w.destroy()

    if gamma_state.current_mode is None:
        tk.Label(
            scroll_frame,
            text="Gamma OFF\nPress F8 or F9",
            bg=bg,
            fg="red",
            font=("Segoe UI", 12, "bold")
        ).pack(pady=20)
        return

    m = gamma_state.current_mode.value.upper()

    def slider(label, key, mn, mx):
        var = tk.DoubleVar(value=config[key])

        def apply(*_):
            val = round(var.get(), 4)
            if config[key] != val:
                config[key] = val
                gamma_state.rebuild_debounced()
                config_mgr.debounce_save()

        row = tk.Frame(scroll_frame, bg=bg)
        row.pack(fill=tk.X, padx=6, pady=4)

        tk.Label(row, text=label, bg=bg, fg=fg, width=18, anchor="w").pack(side=tk.LEFT)

        tk.Scale(
            row,
            from_=mn, to=mx, resolution=0.01,
            orient="horizontal", length=240,
            variable=var,
            bg=bg, fg=fg,
            troughcolor="#333",
            activebackground=accent,
            command=apply
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            row,
            text="Reset",
            bg=btn,
            fg=fg,
            width=6,
            command=lambda: (var.set(default_config[key]), apply())
        ).pack(side=tk.LEFT, padx=(6, 0))

    for name, key, mn, mx in [
        ("Gamma", f"GAMMA_{m}", 0.7, 1.8),
        ("Gamma Offset", f"GAMMA_OFFSET_{m}", -0.2, 0.2),
        ("Vibrance", f"VIBRANCE_{m}", 0.9, 1.2),
        ("Shadow Boost", f"SHADOW_BOOST_{m}", 1.0, 2.0),
        ("Midtone Boost", f"MIDTONE_BOOST_{m}", 1.0, 1.5),
        ("Highlight Compress", f"HIGHLIGHT_COMPRESS_{m}", 0.1, 1.0),
        ("Red Multiplier", f"RED_MULTIPLIER_{m}", 0.8, 1.2),
        ("Green Multiplier", f"GREEN_MULTIPLIER_{m}", 0.8, 1.2),
        ("Blue Multiplier", f"BLUE_MULTIPLIER_{m}", 0.8, 1.2),
    ]:
        slider(name, key, mn, mx)

def open_gui():
    global settings_window, scroll_frame

    if settings_window and settings_window.winfo_exists():
        settings_window.deiconify()
        return

    settings_window = tk.Toplevel(root)
    settings_window.title("Gamma Control")
    settings_window.geometry("560x600")
    settings_window.configure(bg=bg)

    canvas = tk.Canvas(settings_window, bg=bg, highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    ttk.Scrollbar(settings_window, orient="vertical", command=canvas.yview).pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=lambda *a: None)

    scroll_frame = tk.Frame(canvas, bg=bg)
    window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    settings_window.protocol("WM_DELETE_WINDOW", settings_window.withdraw)
    rebuild_gui()

# ================= HOTKEYS =================
def update_mode(new_mode):
    if gamma_state.current_mode == new_mode:
        gamma_state.current_mode = None
    else:
        gamma_state.current_mode = new_mode

    gamma_state.rebuild()
    rebuild_gui()

def on_press(key):
    if key == keyboard.Key.f6:
        root.after(0, open_gui)
    elif key == keyboard.Key.f8:
        update_mode(Mode.INDOOR)
    elif key == keyboard.Key.f9:
        update_mode(Mode.OUTDOOR)

keyboard.Listener(on_press=on_press, daemon=True).start()

# ================= START =================
gamma_state.rebuild()
root.mainloop()
