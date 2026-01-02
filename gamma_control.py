"""
    WinGammaTuner
=====================================

A low-latency, real-time gamma and color curve controller for Windows displays,
built on the Win32 GDI gamma ramp API. The utility generates custom 256-step LUTs
and applies them directly to the display pipeline via SetDeviceGammaRamp, allowing
live tuning without restarting the process or reinitializing the display context.

The system supports two independent, fully configurable profiles (INDOOR and
OUTDOOR). Each profile defines its own gamma curve shaping, luminance balancing,
vibrance scaling, and per-channel color multipliers. Profiles can be toggled
instantly, with all changes applied atomically and redundant hardware updates
automatically avoided.

Core Capabilities:
- Direct display gamma ramp control via Win32 GDI
- Parametric LUT generation including:
  - Gamma exponent with optional offset
  - Shadow-region amplification
  - Midtone-region amplification
  - Highlight compression to reduce clipping
  - Global vibrance scaling
  - Independent RGB channel multipliers
- Real-time application with:
  - CRC-based gamma ramp signature detection
  - Debounced rebuilds to prevent excessive GDI calls
  - Cached base curves for efficient recomputation
- Persistent JSON-backed configuration with auto-save debounce
- Lightweight Tkinter GUI for live parameter tuning
- Global hotkeys for profile toggling and GUI visibility

Profiles:
- INDOOR: Intended for controlled or low-light environments
- OUTDOOR: Intended for bright or high-glare environments
Only one profile may be active at a time. Disabling all profiles restores the
identity gamma ramp.

Hotkeys:
- F6  → Open or show settings GUI
- F8  → Toggle INDOOR profile
- F9  → Toggle OUTDOOR profile

GUI Interaction:
- All sliders support modifier-based precision control:
  - Normal drag: direct mapping across the full value range
  - Shift + drag: interpolated movement toward the target (~20% step)
  - Ctrl  + drag: fine-grained interpolated movement (~5% step)
- Modifier keys do not change the slider range; instead they control how quickly
  the current value converges toward the target, enabling both coarse and
  precision tuning without mode switches.

Runtime Behavior:
- LUTs are rebuilt only when parameters or active profile change
- Identical gamma ramps are skipped using a fast CRC signature check
- GUI-driven updates are applied live and persisted automatically
- Identity ramp is restored when no profile is active

System Requirements:
- Windows OS
- Python 3.9+
- Display driver supporting SetDeviceGammaRamp
- Dependencies: numpy, tkinter, pynput, orjson

Notes & Warnings:
- Gamma ramps affect global display output at the driver level
- Extreme values may cause banding, clipping, or eye strain
- Not recommended for remote desktop sessions or unsupported GPUs

"""



import ctypes
import os
import logging
from functools import lru_cache
from threading import Lock, Timer
from enum import Enum
import zlib
import atexit

import numpy as np
import tkinter as tk
from tkinter import ttk
from pynput import keyboard
import orjson

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "gamma_config.json")

default_config = {
    "GAMMA_INDOOR": 1.30,
    "VIBRANCE_INDOOR": 1.05,
    "SHADOW_BOOST_INDOOR": 1.35,
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
    return np.concatenate([np.linspace(0, 65535, 256, dtype=np.uint16)] * 3)

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
        self.apply(identity_ramp() if self.current_mode is None else build_ramp_array(self.current_mode))

    def rebuild_debounced(self, delay=0.03):
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(delay, lambda: root.after(0, self.rebuild))
        self.timer.daemon = True
        self.timer.start()

gamma_state = GammaState()
atexit.register(lambda: gamma_state.apply(identity_ramp()))

# ================= CANVAS SLIDER =================
class CanvasSlider(tk.Canvas):
    def __init__(self, parent, width=240, height=20,
                 min_val=0.0, max_val=1.0,
                 value=0.0, command=None):
        super().__init__(parent, width=width, height=height,
                         bg="#1e1e1e", highlightthickness=0)

        self.min = min_val
        self.max = max_val
        self.command = command
        self.width = width
        self.usable = width - 12

        self.create_rectangle(0, height // 2 - 2,
                              width, height // 2 + 2,
                              fill="#333", outline="")
        self.thumb = self.create_oval(0, 4, 12, 16,
                                      fill="#007acc", outline="")

        self.bind("<ButtonPress-1>", self.start_drag)
        self.bind("<B1-Motion>", self.drag)

        self.set_value(value, notify=False)

    def draw(self):
        t = (self.value - self.min) / (self.max - self.min)
        x = int(t * self.usable)
        self.coords(self.thumb, x, 4, x + 12, 16)

    def set_value(self, value, notify=True):
        self.value = float(np.clip(value, self.min, self.max))
        self.draw()
        if notify and self.command:
            self.command(self.value)

    def start_drag(self, e):
        self.drag(e)

    def drag(self, e):
        t = np.clip(e.x / self.usable, 0.0, 1.0)
        target = self.min + t * (self.max - self.min)

        if e.state & 0x0004:      # Ctrl
            target = self._lerp(self.value, target, 0.05)
        elif e.state & 0x0001:    # Shift
            target = self._lerp(self.value, target, 0.2)

        self.set_value(target)

    @staticmethod
    def _lerp(a, b, f):
        return a + (b - a) * f

# ================= GUI =================
bg, fg, btn = "#1e1e1e", "#ffffff", "#2d2d2d"
settings_window = None
scroll_frame = None

def rebuild_gui():
    if not scroll_frame:
        return
    for w in scroll_frame.winfo_children():
        w.destroy()

    if gamma_state.current_mode is None:
        tk.Label(scroll_frame, text="Gamma OFF\nPress F8 or F9",
                 bg=bg, fg="red",
                 font=("Segoe UI", 12, "bold")).pack(pady=20)
        return

    m = gamma_state.current_mode.value.upper()

    def slider(label, key, mn, mx, fmt):
        row = tk.Frame(scroll_frame, bg=bg)
        row.pack(fill=tk.X, padx=6, pady=6)

        tk.Label(row, text=label, bg=bg, fg=fg,
                 width=24, anchor="w").pack(side=tk.LEFT)

        value_var = tk.StringVar()

        def on_change(v):
            val = round(v, 6)
            config[key] = val
            gamma_state.rebuild_debounced()
            config_mgr.debounce_save()
            value_var.set(fmt.format(val))

        cs = CanvasSlider(row, min_val=mn, max_val=mx,
                          value=config[key], command=on_change)
        cs.pack(side=tk.LEFT, padx=6)

        value_var.set(fmt.format(config[key]))
        tk.Label(row, textvariable=value_var,
                 bg=bg, fg="#aaa",
                 width=8, anchor="e").pack(side=tk.LEFT)

        tk.Button(
            row, text="Reset", bg=btn, fg=fg, width=6,
            command=lambda cs=cs, k=key: cs.set_value(default_config[k])
        ).pack(side=tk.LEFT, padx=(6, 0))

    sliders = [
        ("Gamma", f"GAMMA_{m}", 0.75, 1.35, "{:.3f}"),
        ("Gamma Offset", f"GAMMA_OFFSET_{m}", -0.06, 0.06, "{:.4f}"),
        ("Vibrance", f"VIBRANCE_{m}", 0.9, 1.3, "{:.3f}"),
        ("Shadow Boost", f"SHADOW_BOOST_{m}", 1.0, 1.6, "{:.3f}"),
        ("Midtone Boost", f"MIDTONE_BOOST_{m}", 1.0, 1.4, "{:.3f}"),
        ("Highlight Compress", f"HIGHLIGHT_COMPRESS_{m}", 0.0, 1.0, "{:.4f}"),
        ("Red Multiplier", f"RED_MULTIPLIER_{m}", 0.75, 1.25, "{:.3f}"),
        ("Green Multiplier", f"GREEN_MULTIPLIER_{m}", 0.75, 1.25, "{:.3f}"),
        ("Blue Multiplier", f"BLUE_MULTIPLIER_{m}", 0.75, 1.25, "{:.3f}"),
    ]

    for s in sliders:
        slider(*s)

def open_gui():
    global settings_window, scroll_frame
    if settings_window and settings_window.winfo_exists():
        settings_window.deiconify()
        return

    settings_window = tk.Toplevel(root)
    settings_window.title("Gamma Control")
    settings_window.geometry("640x600")
    settings_window.configure(bg=bg)

    canvas = tk.Canvas(settings_window, bg=bg, highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ttk.Scrollbar(settings_window, orient="vertical",
                  command=canvas.yview).pack(side=tk.RIGHT, fill=tk.Y)

    scroll_frame = tk.Frame(canvas, bg=bg)
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    scroll_frame.bind("<Configure>",
                      lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    settings_window.protocol("WM_DELETE_WINDOW", settings_window.withdraw)
    rebuild_gui()

# ================= HOTKEYS =================
def update_mode(new_mode):
    gamma_state.current_mode = None if gamma_state.current_mode == new_mode else new_mode
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
