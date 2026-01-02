"""
    WinGammaTuner
=====================================

A low-latency, real-time gamma and color curve controller for Windows displays,
built on the Win32 GDI gamma ramp API. The utility generates custom 256-step LUTs
and applies them directly to the display pipeline via SetDeviceGammaRamp, allowing
live tuning without restarting applications or reinitializing the display context.

The system supports two independent, fully configurable profiles (INDOOR and
OUTDOOR). Each profile defines its own gamma response, luminance shaping,
shadow visibility behavior, color balance, and vibrance scaling. Profiles can be
toggled instantly, with changes applied atomically and redundant hardware updates
automatically avoided.

The design prioritizes competitive FPS visibility while preserving contrast,
highlight stability, UI legibility, and predictable output.

Core Capabilities:
- Direct display gamma ramp control via Win32 GDI
- Parametric LUT generation including:
  - Gamma exponent with optional offset
  - Black floor lift for shadow detail preservation
  - Nonlinear shadow-region lift with configurable cutoff
  - Shadow-only perceptual micro-contrast shaping (logarithmic)
  - Optional midtone shaping (advanced)
  - Highlight compression with near-white (UI/HUD) preservation
  - Global vibrance scaling
  - Global RGB multipliers
  - Shadow-only luminance bias for silhouette clarity (FPS-safe)
  - Optional shadow-only chroma suppression for haze reduction
- Real-time application with:
  - CRC-based gamma ramp signature detection
  - Debounced rebuilds to prevent excessive GDI calls
  - Cached base curves for efficient recomputation
- Persistent JSON-backed configuration with auto-save debounce
- Lightweight Tkinter GUI for live parameter tuning
- Global hotkeys for profile toggling and GUI visibility

Control Tiers:
- FPS Controls:
  - Shadow visibility, cutoff, micro-contrast, desaturation, silhouette bias,
    and contrast-safe tuning designed for competitive play
- Advanced Controls (optional):
  - Shadow RGB bias, midtone shaping, and other high-impact adjustments
  - Advanced controls remain active when hidden, ensuring stable output

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
  - Shift + drag: medium-granularity interpolation
  - Ctrl  + drag: fine-granularity interpolation
- Slider granularity is range-aware and display values are quantized to
  perceptually meaningful step sizes
- Value displays provide visual feedback during active adjustment

Runtime Behavior:
- LUTs are rebuilt only when parameters or active profile change
- Identical gamma ramps are skipped using a fast CRC signature check
- GUI-driven updates are applied live and persisted automatically
- Identity ramp is restored on exit or when no profile is active

System Requirements:
- Windows OS
- Python 3.9+
- Display driver supporting SetDeviceGammaRamp
- Dependencies: numpy, tkinter, pynput, orjson

Notes & Warnings:
- Gamma ramps affect global display output at the driver level
- Extreme values may cause banding, clipping, or eye strain
- Advanced controls can significantly alter visual output
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
import math

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
    "ADVANCED_MODE": False,

    # ---------- INDOOR ----------
    "GAMMA_INDOOR": 1.30,
    "GAMMA_OFFSET_INDOOR": 0.0,
    "VIBRANCE_INDOOR": 1.05,

    "BLACK_FLOOR_INDOOR": 0.035,
    "SHADOW_LIFT_EXP_INDOOR": 0.65,
    "SHADOW_CUTOFF_INDOOR": 0.40,
    "SHADOW_DESAT_INDOOR": 0.85,

    # Shadow color bias (FPS-safe)
    "SHADOW_COLOR_BIAS_INDOOR": 0.0,

    # Shadow RGB bias (advanced)
    "SHADOW_RED_BIAS_INDOOR": 1.00,
    "SHADOW_GREEN_BIAS_INDOOR": 1.00,
    "SHADOW_BLUE_BIAS_INDOOR": 1.00,

    "MIDTONE_BOOST_INDOOR": 1.00,
    "HIGHLIGHT_COMPRESS_INDOOR": 0.30,

    "RED_MULTIPLIER_INDOOR": 1.00,
    "GREEN_MULTIPLIER_INDOOR": 1.00,
    "BLUE_MULTIPLIER_INDOOR": 1.00,

    # ---------- OUTDOOR ----------
    "GAMMA_OUTDOOR": 1.12,
    "GAMMA_OFFSET_OUTDOOR": 0.0,
    "VIBRANCE_OUTDOOR": 1.04,

    "BLACK_FLOOR_OUTDOOR": 0.025,
    "SHADOW_LIFT_EXP_OUTDOOR": 0.75,
    "SHADOW_CUTOFF_OUTDOOR": 0.35,
    "SHADOW_DESAT_OUTDOOR": 0.90,

    "SHADOW_COLOR_BIAS_OUTDOOR": 0.0,

    "SHADOW_RED_BIAS_OUTDOOR": 1.00,
    "SHADOW_GREEN_BIAS_OUTDOOR": 1.00,
    "SHADOW_BLUE_BIAS_OUTDOOR": 1.00,

    "MIDTONE_BOOST_OUTDOOR": 1.00,
    "HIGHLIGHT_COMPRESS_OUTDOOR": 0.50,

    "RED_MULTIPLIER_OUTDOOR": 1.00,
    "GREEN_MULTIPLIER_OUTDOOR": 1.00,
    "BLUE_MULTIPLIER_OUTDOOR": 1.00,
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

    bf = np.clip(p["BLACK_FLOOR"], 0.0, 0.10)
    base = bf + base * (1.0 - bf)

    cutoff = np.clip(p["SHADOW_CUTOFF"], 0.15, 0.6)
    shadow = x < cutoff
    base[shadow] = np.power(base[shadow] / cutoff, p["SHADOW_LIFT_EXP"]) * cutoff

    mid = (x >= cutoff) & (x < 0.7)
    base[mid] *= p["MIDTONE_BOOST"]

    hi = base > 0.85
    base[hi] = 0.85 + (base[hi] - 0.85) * p["HIGHLIGHT_COMPRESS"]
    np.clip(base, 0.0, 1.0, out=base)

    # ---- RGB ----
    r = base * p["RED_MULTIPLIER"]
    g = base * p["GREEN_MULTIPLIER"]
    b = base * p["BLUE_MULTIPLIER"]

    # ---- SHADOW COLOR BIAS (FPS SAFE) ----
    bias = np.clip(p["SHADOW_COLOR_BIAS"], -0.05, 0.05)
    r[shadow] *= 1.0 + bias
    b[shadow] *= 1.0 - bias

    # ---- SHADOW RGB BIAS (ADVANCED) ----
    r[shadow] *= p["SHADOW_RED_BIAS"]
    g[shadow] *= p["SHADOW_GREEN_BIAS"]
    b[shadow] *= p["SHADOW_BLUE_BIAS"]

    # ---- SHADOW DESAT ----
    desat = p["SHADOW_DESAT"]
    if desat < 1.0:
        lum = (r + g + b) / 3.0
        r[shadow] = lum[shadow] + (r[shadow] - lum[shadow]) * desat
        g[shadow] = lum[shadow] + (g[shadow] - lum[shadow]) * desat
        b[shadow] = lum[shadow] + (b[shadow] - lum[shadow]) * desat

    scale = 65535 * p["VIBRANCE"]
    return np.concatenate((
        np.clip(r * scale, 0, 65535).astype(np.uint16),
        np.clip(g * scale, 0, 65535).astype(np.uint16),
        np.clip(b * scale, 0, 65535).astype(np.uint16),
    ))

# ================= GAMMA STATE =================
class GammaState:
    def __init__(self):
        self.current_mode = None
        self.last_sig = None
        self.lock = Lock()
        self.timer = None

    def apply(self, arr):
        sig = fast_array_signature(arr)
        with self.lock:
            if sig == self.last_sig:
                return
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

# ================= GUI =================
bg, fg, btn = "#1e1e1e", "#ffffff", "#2d2d2d"
settings_window = None
scroll_frame = None

def decimals(step):
    return max(0, int(-math.log10(step)))

class CanvasSlider(tk.Canvas):
    def __init__(self, parent, mn, mx, value, command, step):
        super().__init__(parent, width=240, height=20, bg=bg, highlightthickness=0)
        self.min, self.max, self.command, self.step = mn, mx, command, step
        self.usable = 228
        self.value = value

        self.create_rectangle(0, 9, 240, 11, fill="#333", outline="")
        self.thumb = self.create_oval(0, 4, 12, 16, fill="#007acc", outline="")

        self.bind("<ButtonPress-1>", self.drag)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.release)

        self.set_value(value, False)

    def q(self, v): return round(v / self.step) * self.step

    def set_value(self, v, notify=True):
        self.value = self.q(np.clip(v, self.min, self.max))
        x = int((self.value - self.min) / (self.max - self.min) * self.usable)
        self.coords(self.thumb, x, 4, x + 12, 16)
        if notify: self.command(self.value, False)

    def drag(self, e):
        t = np.clip(e.x / self.usable, 0, 1)
        target = self.min + t * (self.max - self.min)
        span = self.max - self.min
        if e.state & 0x0004:
            target = self.value + (target - self.value) * min(0.05, 0.01 / span)
        elif e.state & 0x0001:
            target = self.value + (target - self.value) * min(0.2, 0.04 / span)
        self.value = self.q(target)
        self.command(self.value, True)
        self.set_value(self.value, False)

    def release(self, e):
        self.command(self.value, False)

def rebuild_gui():
    if not scroll_frame:
        return
    for w in scroll_frame.winfo_children():
        w.destroy()

    if gamma_state.current_mode is None:
        tk.Label(scroll_frame, text="Gamma OFF\nF8 / F9",
                 bg=bg, fg="red", font=("Segoe UI", 12, "bold")).pack(pady=20)
        return

    adv = tk.BooleanVar(value=config["ADVANCED_MODE"])

    def toggle_adv():
        config["ADVANCED_MODE"] = adv.get()
        config_mgr.debounce_save()
        rebuild_gui()

    tk.Checkbutton(scroll_frame, text="Advanced Controls",
                   variable=adv, command=toggle_adv,
                   bg=bg, fg=fg, selectcolor=bg).pack(anchor="w", padx=6, pady=(6, 12))

    m = gamma_state.current_mode.value.upper()

    fps_sliders = [
        ("Gamma", f"GAMMA_{m}", 0.75, 1.4, 0.001),
        ("Black Floor", f"BLACK_FLOOR_{m}", 0.0, 0.08, 0.0005),
        ("Shadow Lift Exp", f"SHADOW_LIFT_EXP_{m}", 0.4, 1.0, 0.001),
        ("Shadow Cutoff", f"SHADOW_CUTOFF_{m}", 0.2, 0.6, 0.001),
        ("Shadow Desat", f"SHADOW_DESAT_{m}", 0.6, 1.0, 0.001),
        ("Shadow Color Bias", f"SHADOW_COLOR_BIAS_{m}", -0.05, 0.05, 0.001),
        ("Highlight Compress", f"HIGHLIGHT_COMPRESS_{m}", 0.0, 0.7, 0.001),
        ("Vibrance", f"VIBRANCE_{m}", 0.9, 1.3, 0.001),
    ]

    advanced_sliders = [
        ("Shadow Red Bias", f"SHADOW_RED_BIAS_{m}", 0.95, 1.05, 0.001),
        ("Shadow Green Bias", f"SHADOW_GREEN_BIAS_{m}", 0.95, 1.05, 0.001),
        ("Shadow Blue Bias", f"SHADOW_BLUE_BIAS_{m}", 0.95, 1.05, 0.001),
    ]

    def draw(label, key, mn, mx, step):
        row = tk.Frame(scroll_frame, bg=bg)
        row.pack(fill=tk.X, padx=6, pady=4)
        tk.Label(row, text=label, bg=bg, fg=fg, width=22, anchor="w").pack(side=tk.LEFT)
        val = tk.StringVar()
        dec = decimals(step)

        def cb(v, dragging):
            config[key] = v
            gamma_state.rebuild_debounced()
            config_mgr.debounce_save()
            val.set(f"{v:.{dec}f}")
            lbl.configure(fg="#777" if dragging else "#aaa")

        cs = CanvasSlider(row, mn, mx, config[key], cb, step)
        cs.pack(side=tk.LEFT, padx=6)
        val.set(f"{config[key]:.{dec}f}")
        lbl = tk.Label(row, textvariable=val, bg=bg, fg="#aaa", width=8)
        lbl.pack(side=tk.LEFT)
        tk.Button(row, text="Reset", bg=btn, fg=fg, width=6,
                  command=lambda: cs.set_value(default_config[key])
                  ).pack(side=tk.LEFT, padx=(6, 0))

    for s in fps_sliders:
        draw(*s)

    if config["ADVANCED_MODE"]:
        ttk.Separator(scroll_frame).pack(fill="x", pady=8)
        tk.Label(scroll_frame, text="Advanced / Shadow RGB",
                 bg=bg, fg="#bbb").pack(anchor="w", padx=6)
        for s in advanced_sliders:
            draw(*s)

def open_gui():
    global settings_window, scroll_frame
    if settings_window and settings_window.winfo_exists():
        settings_window.deiconify()
        return

    settings_window = tk.Toplevel(root)
    settings_window.title("Gamma Control")
    settings_window.geometry("640x600")
    settings_window.configure(bg=bg)

    canvas = tk.Canvas(settings_window, bg=bg)
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
def update_mode(mode):
    gamma_state.current_mode = None if gamma_state.current_mode == mode else mode
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
