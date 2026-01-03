"""
    WinGammaTuner – Competitive FPS Edition
===========================================

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
  - Shadow-only perceptual contrast shaping (power-based)
  - Optional mid-shadow sigmoid boost for silhouette contrast
  - Optional midtone shaping (advanced)
  - Highlight compression with near-white (UI/HUD) preservation
  - Configurable highlight clamp to preserve HUD/UI brightness
  - Global vibrance scaling
  - Global RGB multipliers
  - Shadow-only color bias for silhouette clarity (FPS-safe)
  - Shadow-only RGB channel bias (advanced)
  - Global Shadow Pop Strength modifier (FPS visibility enhancer)
  - Optional histogram-aware shadow adaptation (scene luminance–responsive)
  - Optional edge-aware shadow contrast preservation
  - Optional opponent-channel shadow tuning with luminance conservation
  - Optional HUD-aware highlight exclusion with configurable threshold
- Real-time application with:
  - CRC-based gamma ramp signature detection
  - Debounced rebuilds to prevent excessive GDI calls
  - Cached base curves for efficient recomputation
  - Lightweight background threading for scene analysis and gamma application
  - Low-frequency scene analysis for adaptive tuning (FPS-safe, foreground-only)
- Persistent JSON-backed configuration with auto-save debounce
- Lightweight Tkinter GUI for live parameter tuning
- Fully refactored, dynamic slider system with easy extensibility
- Global hotkeys for profile toggling and GUI visibility

Control Tiers:
- FPS Controls:
  - Shadow visibility, cutoff, contrast shaping, desaturation, silhouette bias,
    sigmoid shaping, and contrast-safe tuning designed for competitive play
- Advanced Controls (optional):
  - Shadow RGB bias, midtone shaping, and other high-impact adjustments
  - Advanced controls remain active when hidden, ensuring stable output

Profiles:
- INDOOR: Intended for controlled or low-light environments
- OUTDOOR: Intended for bright or high-glare environments
Only one profile may be active at a time. Disabling all profiles restores the
identity gamma ramp.

Global Modifiers:
- Shadow Pop Strength (0.0–1.0):
  - Applies contrast and silhouette enhancements on top of the active profile
  - Tuned for shadow clarity and FPS-friendly visibility

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
- GUI includes toggles for advanced controls and HUD highlight preservation

Runtime Behavior:
- LUTs are rebuilt only when parameters or the active profile change
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tkinter as tk
from tkinter import ttk
from pynput import keyboard
import orjson
import time

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

executor = ThreadPoolExecutor(max_workers=2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "gamma_config.json")

default_config = {
    "ADVANCED_MODE": False,
    "SHADOW_POP_STRENGTH": 0.20,          # Gentle global boost for silhouette clarity without washout
    "PRESERVE_HUD_HIGHLIGHTS": True,      # Required due to bright interiors / snow / UI overlap

    # ================= INDOOR PROFILE (Primary for this map) =================

    "GAMMA_INDOOR": 1.28,                 # Slightly lower than before to avoid global flattening
    "GAMMA_OFFSET_INDOOR": 0.0,

    "VIBRANCE_INDOOR": 1.04,              # Small reduction to prevent color noise in shadows
    "BLACK_FLOOR_INDOOR": 0.030,           # Opens near-black detail without lifting true blacks

    "SHADOW_LIFT_EXP_INDOOR": 0.62,        # Stronger shadow opening, tuned for interior corners
    "SHADOW_CUTOFF_INDOOR": 0.42,          # Targets doorways, stairwells, pits where enemies hide

    "SHADOW_DESAT_INDOOR": 0.82,           # Helps warm player models stand out from cold interiors
    "SHADOW_COLOR_BIAS_INDOOR": 0.0,

    "SHADOW_RED_BIAS_INDOOR": 1.00,
    "SHADOW_GREEN_BIAS_INDOOR": 1.00,
    "SHADOW_BLUE_BIAS_INDOOR": 1.00,

    "MIDTONE_BOOST_INDOOR": 1.02,          # Very light lift to avoid crushing mid-detail
    "HIGHLIGHT_COMPRESS_INDOOR": 0.32,     # Preserves bright lights and HUD readability

    "RED_MULTIPLIER_INDOOR": 1.00,
    "GREEN_MULTIPLIER_INDOOR": 1.00,
    "BLUE_MULTIPLIER_INDOOR": 1.00,

    "SHADOW_SIGMOID_BOOST_INDOOR": 0.35,   # Main silhouette separation tool for this map

    # ================= OUTDOOR PROFILE (Snow / bright transitions) =================

    "GAMMA_OUTDOOR": 1.12,                 # Correct as-is for bright environments
    "GAMMA_OFFSET_OUTDOOR": 0.0,

    "VIBRANCE_OUTDOOR": 1.03,              # Keeps snow from oversaturating UI elements
    "BLACK_FLOOR_OUTDOOR": 0.022,           # Slightly lower to preserve outdoor contrast

    "SHADOW_LIFT_EXP_OUTDOOR": 0.75,        # Less aggressive; outdoor shadows are already readable
    "SHADOW_CUTOFF_OUTDOOR": 0.34,          # Keeps lift out of bright ground planes

    "SHADOW_DESAT_OUTDOOR": 0.90,           # Minimal desat to retain environmental cues
    "SHADOW_COLOR_BIAS_OUTDOOR": 0.0,

    "SHADOW_RED_BIAS_OUTDOOR": 1.00,
    "SHADOW_GREEN_BIAS_OUTDOOR": 1.00,
    "SHADOW_BLUE_BIAS_OUTDOOR": 1.00,

    "MIDTONE_BOOST_OUTDOOR": 1.00,          # Leave neutral to avoid snow glare
    "HIGHLIGHT_COMPRESS_OUTDOOR": 0.48,     # Prevents snow and lights from clipping

    "RED_MULTIPLIER_OUTDOOR": 1.00,
    "GREEN_MULTIPLIER_OUTDOOR": 1.00,
    "BLUE_MULTIPLIER_OUTDOOR": 1.00,

    "SHADOW_SIGMOID_BOOST_OUTDOOR": 0.18,   # Mild separation without overprocessing
}

# === Adaptive Analysis Features (NEW, SAFE DEFAULTS OFF) ===

default_config.update({
    # ================= Histogram-aware shadow shaping =================
    "HISTOGRAM_ADAPTIVE": True,          # Map has frequent bright↔dark interior transitions
    "HISTOGRAM_STRENGTH": 0.30,          # Reduced to avoid fighting tuned static shadows
    "HISTOGRAM_MIN_LUMA": 0.12,          # Matches darker interior baseline
    "HISTOGRAM_MAX_LUMA": 0.55,          # Prevents overreaction to bright rooms

    # ================= Edge-preserving shadow contrast =================
    "EDGE_AWARE_SHADOWS": False,         # Strong geometry already present; avoid clutter boost
    "EDGE_STRENGTH": 0.40,
    "EDGE_MIN": 0.05,
    "EDGE_MAX": 0.35,

    # ================= Color opponent channel tuning =================
    "OPPONENT_TUNING": False,            # Leave opt-in; useful but stylistic
    "OPPONENT_STRENGTH": 0.25,

    # ================= HUD-aware exclusion =================
    "HUD_EXCLUSION": False,              # Highlight clamp already sufficient for this map
    "HUD_EXCLUSION_STRENGTH": 0.60,
    "HUD_EXCLUSION_THRESHOLD": 0.90,
    
    # ================= Motion-aware adaptive shadows =================
    "MOTION_AWARE_SHADOWS": True,        # Visibility failures occur during peeks/movement

    # Profile-aware motion boost
    "MOTION_STRENGTH_INDOOR": 0.55,      # Reduced for stability in tight interiors
    "MOTION_STRENGTH_OUTDOOR": 0.35,     # Subtle assist without snow washout

    "MOTION_SENSITIVITY": 2.3,           # Slightly less trigger-happy
    "MOTION_SMOOTHING": 0.15,            # Already well tuned; leave as-is
    
    # ================= Motion Shadow Emphasis (Cone-like perceptual boost) =================
    "MOTION_SHADOW_EMPHASIS": True,        # Master enable (profile-gated)
    "MOTION_SHADOW_STRENGTH": 0.65,        # Overall intensity (0.0–1.0 safe)
    "MOTION_SHADOW_DARK_LUMA": 0.50,       # Only activate in dark scenes
    "MOTION_SHADOW_MIN_MOTION": 0.015,     # Ignore camera micro-jitter

})


class Mode(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"

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
        self.timer = Timer(delay, lambda: executor.submit(self.save))
        self.timer.daemon = True
        self.timer.start()

config_mgr = ConfigManager(CONFIG_FILE, default_config)
config = config_mgr.data

# === Gamma API
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
hdc = user32.GetDC(0)

def to_ramp(arr):
    return (ctypes.c_ushort * 768)(*arr)

def identity_ramp():
    return np.concatenate([np.linspace(0, 65535, 256, dtype=np.uint16)] * 3)

# Shared LUT x-axis (cached)
X_AXIS = np.linspace(0, 1, 256)

MID_05_MASK = X_AXIS < 0.5
MID_07_MASK = X_AXIS < 0.7
HI_085_MASK = X_AXIS > 0.85
_DITHER_NOISE = (np.random.rand(256) - 0.5) * (1.0 / 65535.0)


# === LOW-RES DESKTOP SAMPLING (FPS-SAFE) ===

def adaptive_enabled():
    return (
        config.get("HISTOGRAM_ADAPTIVE", False) or
        config.get("EDGE_AWARE_SHADOWS", False) or
        config.get("OPPONENT_TUNING", False) or
        config.get("HUD_EXCLUSION", False) or
        config.get("MOTION_AWARE_SHADOWS", False) or
        config.get("MOTION_SHADOW_EMPHASIS", False)
    )

class SceneAnalyzer:
    def __init__(self):
        self.avg_luma = 0.5
        self.shadow_density = 0.0
        self.edge_strength = 0.0
        self.lock = Lock()
        self.last_update = 0.0
        self.motion_strength = 0.0
        self._prev_luma = None

    def update(self):
        # Skip entirely if gamma is OFF
        if gamma_state.current_mode is None:
            self.last_update = time.time()
            return

        # Skip if no adaptive features are enabled
        if not adaptive_enabled():
            self.last_update = time.time()
            return


        now = time.time()
        delta = now - self.last_update

        # Freeze metrics if updates stall badly (FPS drop / hitch)
        if delta > 0.6:
            self.last_update = now
            return

        # 5 Hz cap
        if delta < 0.2:
            return

        self.last_update = now

        hwnd = user32.GetForegroundWindow()
        if not hwnd or user32.IsIconic(hwnd):
            return

        try:
            # Grab tiny desktop sample via GDI
            w, h = 64, 64
            memdc = gdi32.CreateCompatibleDC(hdc)
            bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
            gdi32.SelectObject(memdc, bmp)
            gdi32.StretchBlt(
                memdc, 0, 0, w, h,
                hdc, 0, 0,
                user32.GetSystemMetrics(0),
                user32.GetSystemMetrics(1),
                0x00CC0020
            )

            buf = (ctypes.c_ubyte * (w * h * 4))()
            gdi32.GetBitmapBits(bmp, len(buf), buf)

            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(memdc)

            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            rgb = arr[:, :, :3].astype(np.float32) / 255.0
            luma = rgb.mean(axis=2)
            
            # === Motion detection (FPS-safe) ===
            motion = 0.0

            if config.get("MOTION_AWARE_SHADOWS", False):
                if self._prev_luma is not None:
                    diff = np.abs(luma - self._prev_luma)
                    motion = float(diff.mean())

                self._prev_luma = luma
            else:
                # Hard reset when disabled (prevents stale motion)
                self._prev_luma = None
                self.motion_strength = 0.0


            avg = float(luma.mean())
            shadow = float((luma < 0.25).mean())

            gx = np.abs(np.diff(luma, axis=1))
            gy = np.abs(np.diff(luma, axis=0))
            edge = float(np.mean(gx) + np.mean(gy))
            
            with self.lock:
                alpha = 0.2
                self.avg_luma = self.avg_luma * (1 - alpha) + avg * alpha
                self.shadow_density = self.shadow_density * (1 - alpha) + shadow * alpha
                self.edge_strength = self.edge_strength * (1 - alpha) + edge * alpha

                if config.get("MOTION_AWARE_SHADOWS", False):
                    m_alpha = config.get("MOTION_SMOOTHING", 0.15)
                    self.motion_strength = (
                        self.motion_strength * (1 - m_alpha) +
                        motion * m_alpha
                    )



        except Exception as e:
            logging.debug(f"SceneAnalyzer error: {e}")

scene_analyzer = SceneAnalyzer()

def scene_worker():
    while True:
        # Nothing can change → sleep longer
        if gamma_state.current_mode is None or not adaptive_enabled():
            time.sleep(0.25)
            continue

        # Adaptive + profile active → run normally
        scene_analyzer.update()
        time.sleep(0.05)

executor.submit(scene_worker)

# === Gamma LUT Functions
@lru_cache(maxsize=32)
def base_curve(gamma, offset):
    # Start from shared cached axis (read-only usage)
    x = X_AXIS

    # Any operation below must allocate a new array
    if offset:
        x = np.clip(x + offset, 0.0, 1.0)

    if gamma != 1.0:
        x = np.power(x, 1.0 / gamma)

    return x

def fast_array_signature(arr):
    return zlib.crc32(arr.tobytes())

# === ADAPTIVE HELPERS (SCALAR-ONLY, FPS-SAFE) ===

_adaptive_state = {}   # persistent smoothed adaptive params

def smooth_param(name, target, alpha=0.15):
    prev = _adaptive_state.get(name, target)
    value = prev * (1.0 - alpha) + target * alpha
    _adaptive_state[name] = value
    return value

def get_scene_metrics():
    with scene_analyzer.lock:
        return (
            scene_analyzer.avg_luma,
            scene_analyzer.shadow_density,
            scene_analyzer.edge_strength,
            scene_analyzer.motion_strength,
        )
        
def motion_shadow_emphasis_scale():
    if not (
        config.get("MOTION_AWARE_SHADOWS", False) and
        config.get("MOTION_SHADOW_EMPHASIS", False)
    ):
        return 0.0

    avg, shadow_density, _, motion = get_scene_metrics()

    # Motion threshold gate
    if motion < config.get("MOTION_SHADOW_MIN_MOTION", 0.01):
        return 0.0

    # Normalize motion into [0–1]
    t = np.clip(motion * config.get("MOTION_SENSITIVITY", 2.5), 0.0, 1.0)

    # Shadow presence weighting (prevents snow / fog triggers)
    t *= np.clip(0.5 + shadow_density, 0.5, 1.0)
    
    dark_gate = np.clip(
        config.get("MOTION_SHADOW_DARK_LUMA", 0.5),
        0.35,
        0.60
    )

    if avg > dark_gate:
        return 0.0
        
    strength = np.clip(
        config.get("MOTION_SHADOW_STRENGTH", 0.6),
        0.0,
        0.85
    )

    return t * strength

def apply_histogram_adaptation(p):
    p = dict(p)

    if not config.get("HISTOGRAM_ADAPTIVE", False):
        _adaptive_state.clear()
        return p


    avg, shadow_density, edge, motion = get_scene_metrics()

    lo = config["HISTOGRAM_MIN_LUMA"]
    hi = config["HISTOGRAM_MAX_LUMA"]
    strength = config["HISTOGRAM_STRENGTH"]
    
    profile = p.get("PROFILE")
    if profile == "OUTDOOR":
        strength *= 0.75
    elif profile == "INDOOR":
        strength *= 1.15


    if avg < lo:
        t = np.clip((lo - avg) / lo, 0.0, 1.0) * strength
        t *= (0.5 + shadow_density)

        p["SHADOW_CUTOFF"] -= 0.05 * t
        p["SHADOW_DESAT"] -= 0.20 * t
        p["SHADOW_SIGMOID_BOOST"] += 0.30 * t

    elif avg > hi:
        t = np.clip((avg - hi) / (1.0 - hi), 0.0, 1.0) * strength
        p["SHADOW_SIGMOID_BOOST"] *= (1.0 - 0.5 * t)
        p["SHADOW_DESAT"] += 0.10 * t
        
    p["SHADOW_CUTOFF"] = np.clip(p["SHADOW_CUTOFF"], 0.15, 0.6)
    p["SHADOW_DESAT"] = np.clip(p["SHADOW_DESAT"], 0.5, 1.0)
    p["SHADOW_SIGMOID_BOOST"] = np.clip(p["SHADOW_SIGMOID_BOOST"], 0.0, 1.0)

    # === Temporal smoothing of adaptive parameters ===
    smooth = 0.2

    profile = p.get("PROFILE", "GLOBAL")
    p["SHADOW_CUTOFF"] = smooth_param(
        f"{profile}_SHADOW_CUTOFF", p["SHADOW_CUTOFF"], smooth
    )
    p["SHADOW_DESAT"] = smooth_param(
        f"{profile}_SHADOW_DESAT", p["SHADOW_DESAT"], smooth
    )
    p["SHADOW_SIGMOID_BOOST"] = smooth_param(
        f"{profile}_SHADOW_SIGMOID_BOOST", p["SHADOW_SIGMOID_BOOST"], smooth
    )

    return p

def edge_shadow_scale():
    if not config.get("EDGE_AWARE_SHADOWS", False):
        return 1.0

    _, _, edge, _ = get_scene_metrics()  # ✅ correct
    mn = config["EDGE_MIN"]
    mx = config["EDGE_MAX"]
    strength = config["EDGE_STRENGTH"]

    t = np.clip((edge - mn) / max(mx - mn, 1e-4), 0.0, 1.0)
    return 1.0 + t * strength


def build_ramp_array(mode):
    m = mode.value.upper()
    p = {k.replace(f"_{m}", ""): config[k] for k in config if k.endswith(m)}

    # Tag active profile
    p["PROFILE"] = m

    adaptive = adaptive_enabled()
    motion_emphasis = motion_shadow_emphasis_scale() if adaptive else 0.0

    # === Histogram-aware parameter adaptation (optional) ===
    if adaptive:
        p = apply_histogram_adaptation(p)

    pop = np.clip(config.get("SHADOW_POP_STRENGTH", 0.0), 0.0, 1.0)
    if pop > 0.0:
        p["GAMMA"] *= (1.0 + 0.04 * pop)
        p["SHADOW_SIGMOID_BOOST"] += 0.15 * pop
        p["SHADOW_DESAT"] -= 0.06 * pop
        p["MIDTONE_BOOST"] *= 1.0 + (0.03 * pop)
        p["RED_MULTIPLIER"] *= 1.0 + (0.01 * pop)
        p["BLUE_MULTIPLIER"] *= 1.0 - (0.005 * pop)
        
    p["SHADOW_SIGMOID_BOOST"] = np.clip(p["SHADOW_SIGMOID_BOOST"], 0.0, 1.0)
    p["SHADOW_CUTOFF"] = np.clip(p["SHADOW_CUTOFF"], 0.15, 0.6)
    p["GAMMA"] = np.clip(p["GAMMA"], 0.75, 1.6)


    x = X_AXIS
    base = base_curve(p["GAMMA"], p["GAMMA_OFFSET"]).copy()
    bf = np.clip(p["BLACK_FLOOR"], 0.0, 0.10)
    base = bf + base * (1.0 - bf)

    # === Edge-aware shadow scaling (optional) ===
    edge_scale = edge_shadow_scale() if adaptive else 1.0

    # Base shadow cutoff
    cutoff = np.clip(p["SHADOW_CUTOFF"], 0.15, 0.6)

    # Motion-based temporary expansion (perceptual "cone")
    cutoff += 0.04 * motion_emphasis
    cutoff = np.clip(cutoff, 0.15, 0.6)

    transition = 0.08  # smoke-safe zone
    shadow = x < cutoff
    soft_shadow = (x >= cutoff) & (x < cutoff + transition)
    
    # Reduce edge influence on shadow lift (keep it stronger on sigmoid)
    lift_scale = 1.0 + 0.5 * (edge_scale - 1.0)

    exp = np.clip(
        p["SHADOW_LIFT_EXP"] * lift_scale,
        0.4,
        1.2
    )

    # --- Luminance-preserving shadow lift ---
    lum = base.copy()

    # Protect against divide-by-zero
    chrom = np.divide(base, lum + 1e-5)

    # Hard shadows (true dark regions)
    lum[shadow] = np.power(lum[shadow] / cutoff, exp) * cutoff

    # Soft shadows (smoke-safe transition zone)
    t = (x[soft_shadow] - cutoff) / transition
    lifted = np.power(lum[soft_shadow] / cutoff, exp) * cutoff
    lum[soft_shadow] = lum[soft_shadow] * (1.0 - t) + lifted * t

    # Recombine luminance + chroma
    base = lum * chrom
    
    # --- Minimum shadow contrast floor ---
    min_contrast = 0.015  # subtle, safe
    dx = np.gradient(base)
    dx[shadow] = np.maximum(dx[shadow], min_contrast)
    base = np.cumsum(dx)
    base = np.clip(base / base[-1], 0.0, 1.0)
    
    # --- Deep shadow toe lift (brightens dark-dark only) ---
    toe_end = 0.08      # how far up the lift reaches (0.06–0.10 safe)
    toe_strength = 0.04 # lift amount (0.02–0.06 safe)

    toe = x < toe_end
    t = (toe_end - x[toe]) / toe_end
    base[toe] += toe_strength * (t * t)
    
    if adaptive:
        if motion_emphasis > 0.0:
            base[toe] += (0.025 * motion_emphasis)

    # Sigmoid contrast shaping (mid-shadow)
    sig_strength = np.clip(p.get("SHADOW_SIGMOID_BOOST", 0.0), 0.0, 1.0)

    if adaptive:
        _, shadow_density, _, motion = get_scene_metrics()

        # Shadow density scaling
        sig_strength *= np.clip(0.5 + shadow_density, 0.5, 1.0)

        # Motion-aware boost (only if enabled)
        if config.get("MOTION_AWARE_SHADOWS", False):
            sensitivity = config.get("MOTION_SENSITIVITY", 2.5)
            profile = p.get("PROFILE", "GLOBAL")
            strength = config.get(f"MOTION_STRENGTH_{profile}", 0.6)

            motion_t = np.clip(motion * sensitivity, 0.0, 1.0)
            sig_strength *= (1.0 + motion_t * strength)
            
            # --- Motion-only local contrast (no lift, smoke-safe) ---
            if motion_t > 0.0:
                base += (base - np.mean(base)) * motion_t * 0.04

            if motion_emphasis > 0.0:
                # Stronger silhouette separation
                sig_strength *= (1.0 + 1.2 * motion_emphasis)

                

    sig_strength = np.clip(sig_strength, 0.0, 1.25)

    if sig_strength > 0.0:
        mid = MID_05_MASK & ~shadow & ~soft_shadow
        t = (x[mid] - cutoff) / (0.5 - cutoff)
        sigmoid = 1 / (1 + np.exp(-8 * (t - 0.5)))
        s = sig_strength * min(edge_scale, 1.3)
        base[mid] = base[mid] * (1 - s) + sigmoid * s

    # --- Shadow ceiling compression ---
    ceiling = cutoff + 0.15
    mask = (x > cutoff) & (x < ceiling)
    t = (x[mask] - cutoff) / (ceiling - cutoff)
    base[mask] *= 1.0 - 0.12 * (1.0 - t)

    mid = MID_07_MASK & ~shadow & ~soft_shadow
    base[mid] *= p["MIDTONE_BOOST"]
    
    # --- Subtle midtone micro-contrast (restores depth, no smoke impact) ---
    mc_strength = 0.025  # keep small (0.02–0.04 max)
    mid = (x > 0.35) & (x < 0.75)
    m = np.mean(base[mid])
    base[mid] += (base[mid] - m) * mc_strength
    base[mid] *= 1.015

    # --- Highlight shoulder roll-off (prevents whiteout) ---
    shoulder_start = 0.75
    shoulder_end = 0.95

    mask = (x > shoulder_start) & (x < shoulder_end)
    t = (x[mask] - shoulder_start) / (shoulder_end - shoulder_start)
    
    # Smooth cubic easing (no banding)
    t = t * t * (3 - 2 * t)

    base[mask] = shoulder_start + (base[mask] - shoulder_start) * (1.0 - 0.6 * t)
    hi = HI_085_MASK
    base[hi] = 0.85 + (base[hi] - 0.85) * p["HIGHLIGHT_COMPRESS"]
    
    if adaptive:
        avg, shadow_density, _, _ = get_scene_metrics()
        if avg > 0.6:
            # --- Highlight micro-contrast restoration (snow / white walls safe) ---
            hi_detail = (x > 0.82) & (x < 0.97)

            # Mean-centered expansion preserves brightness while restoring texture
            m = np.mean(base[hi_detail])
            base[hi_detail] += (base[hi_detail] - m) * 0.035  # SAFE RANGE: 0.02–0.05
    
    # --- Minimum highlight contrast floor ---
    dx = np.gradient(base)
    hi = x > 0.85
    dx[hi] = np.maximum(dx[hi], 0.01)
    base = np.cumsum(dx)
    base = np.clip(base / base[-1], 0.0, 1.0)
    
    # --- Perceptual highlight glare reduction (snow / fog safe) ---
    if adaptive:
        avg, shadow_density, _, _ = get_scene_metrics()

        # Trigger only in very bright scenes with low shadow presence
        if avg > 0.65 and shadow_density < 0.35:
            t = np.clip((avg - 0.65) / 0.25, 0.0, 1.0)

            hi = x > 0.82
            base[hi] *= (1.0 - 0.12 * t)

    # === HUD highlight preservation / exclusion ===
    if config.get("PRESERVE_HUD_HIGHLIGHTS", True):
        if config.get("HUD_EXCLUSION", False):
            # adaptive HUD exclusion
            hud_strength = config.get("HUD_EXCLUSION_STRENGTH", 0.6)
            thr = np.clip(config.get("HUD_EXCLUSION_THRESHOLD", 0.9), 0.75, 0.98)
            hi_mask = base > thr
            base[hi_mask] = thr + (base[hi_mask] - thr) * hud_strength
        else:
            # legacy HUD-safe clamp
            base[base > 0.9] = np.clip(base[base > 0.9], 0.9, 1.0)




    np.clip(base, 0.0, 1.0, out=base)

    r = base * p["RED_MULTIPLIER"]
    g = base * p["GREEN_MULTIPLIER"]
    b = base * p["BLUE_MULTIPLIER"]

    bias = np.clip(p["SHADOW_COLOR_BIAS"], -0.05, 0.05)
    r[shadow] *= 1.0 + bias
    b[shadow] *= 1.0 - bias

    r[shadow] *= p["SHADOW_RED_BIAS"]
    g[shadow] *= p["SHADOW_GREEN_BIAS"]
    b[shadow] *= p["SHADOW_BLUE_BIAS"]
    
    # === Opponent-channel shadow tuning (optional) ===
    if config.get("OPPONENT_TUNING", False):
        opp = np.clip(config["OPPONENT_STRENGTH"], 0.0, 1.0)
        
        # --- Saturation risk clamp (prevents channel clipping)
        # Estimate risk from vibrance and shadow color bias
        sat_risk = max(
            p["VIBRANCE"] - 1.0,
            abs(p["SHADOW_COLOR_BIAS"]) * 4.0
        )

        if sat_risk > 0.0:
            # Smoothly reduce opponent strength under high saturation
            opp *= np.clip(1.0 - sat_risk, 0.25, 1.0)


        # Luminance before adjustment
        lum_before = (r + g + b) / 3.0

        # Opponent channels
        rg = r - g
        by = b - (r + g) * 0.5

        rg[shadow] *= (1.0 + opp)
        by[shadow] *= (1.0 + opp)

        r = g + rg
        b = (r + g) * 0.5 + by

        # Luminance after adjustment
        lum_after = (r + g + b) / 3.0

        # Preserve luminance ONLY in shadows
        scale = np.ones_like(lum_before)
        scale[shadow] = lum_before[shadow] / np.maximum(lum_after[shadow], 1e-4)

        r *= scale
        g *= scale
        b *= scale
        
        # === Final shadow-only safety clamp (post opponent tuning)
        # Prevent rare channel spikes without affecting mid/high tones
        r[shadow] = np.clip(r[shadow], 0.0, 1.0)
        g[shadow] = np.clip(g[shadow], 0.0, 1.0)
        b[shadow] = np.clip(b[shadow], 0.0, 1.0)

    desat = p["SHADOW_DESAT"]

    if adaptive:
        if motion_emphasis > 0.0:
            desat *= (1.0 - 0.35 * motion_emphasis)

    if desat < 1.0:
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        r[shadow] = lum[shadow] + (r[shadow] - lum[shadow]) * desat
        g[shadow] = lum[shadow] + (g[shadow] - lum[shadow]) * desat
        b[shadow] = lum[shadow] + (b[shadow] - lum[shadow]) * desat

    # --- Luminance-weighted dithering (kills oil-slick highlights) ---
    dither = _DITHER_NOISE * (0.5 + 0.5 * base)

    r = np.clip(r + dither, 0.0, 1.0)
    g = np.clip(g + dither, 0.0, 1.0)
    b = np.clip(b + dither, 0.0, 1.0)


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
        arr = identity_ramp() if self.current_mode is None else build_ramp_array(self.current_mode)
        executor.submit(self.apply, arr)

    def rebuild_debounced(self, delay=0.03):
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(delay, lambda: root.after(0, self.rebuild))
        self.timer.daemon = True
        self.timer.start()

gamma_state = GammaState()
atexit.register(lambda: gamma_state.apply(identity_ramp()))

# ================= HOTKEY HANDLING =================
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

# ================= ROOT TK =================
root = tk.Tk()
root.withdraw()
# ================= GUI UTILS =================
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
        if e.state & 0x0004:  # Ctrl
            target = self.value + (target - self.value) * min(0.05, 0.01 / span)
        elif e.state & 0x0001:  # Shift
            target = self.value + (target - self.value) * min(0.2, 0.04 / span)
        self.value = self.q(target)
        self.command(self.value, True)
        self.set_value(self.value, False)

    def release(self, e):
        self.command(self.value, False)

# ==== Shared Slider Creator ====
def draw_slider(parent, label, key, mn, mx, step):
    row = tk.Frame(parent, bg=bg)
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
              command=lambda: cs.set_value(default_config[key])).pack(side=tk.LEFT, padx=(6, 0))
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

    # === Top checkboxes
    tk.Checkbutton(scroll_frame, text="Advanced Controls",
                   variable=adv, command=toggle_adv,
                   bg=bg, fg=fg, selectcolor=bg).pack(anchor="w", padx=6, pady=(6, 4))

    # HUD highlight clamp toggle
    hud_var = tk.BooleanVar(value=config.get("PRESERVE_HUD_HIGHLIGHTS", True))
    def toggle_hud():
        config["PRESERVE_HUD_HIGHLIGHTS"] = hud_var.get()
        config_mgr.debounce_save()
        gamma_state.rebuild_debounced()

    tk.Checkbutton(scroll_frame, text="Preserve HUD Highlights",
                   variable=hud_var, command=toggle_hud,
                   bg=bg, fg=fg, selectcolor=bg).pack(anchor="w", padx=6, pady=(0, 10))
                   
    # === Adaptive / Scene-aware Features ===
    ttk.Separator(scroll_frame).pack(fill="x", pady=10)

    tk.Label(
        scroll_frame,
        text="Adaptive / Scene-Aware Enhancements",
        bg=bg,
        fg="#bbb",
        font=("Segoe UI", 10, "bold")
    ).pack(anchor="w", padx=6, pady=(4, 6))

    # Histogram-aware shadows
    hist_var = tk.BooleanVar(value=config["HISTOGRAM_ADAPTIVE"])
    def toggle_hist():
        config["HISTOGRAM_ADAPTIVE"] = hist_var.get()
        config_mgr.debounce_save()
        gamma_state.rebuild_debounced()

    tk.Checkbutton(
        scroll_frame,
        text="Histogram-aware Shadow Shaping",
        variable=hist_var,
        command=toggle_hist,
        bg=bg, fg=fg, selectcolor=bg
    ).pack(anchor="w", padx=6)

    if hist_var.get():
        draw_slider(scroll_frame, "Histogram Strength",
                    "HISTOGRAM_STRENGTH", 0.0, 1.0, 0.01)
        draw_slider(scroll_frame, "Min Scene Luma",
                    "HISTOGRAM_MIN_LUMA", 0.10, 0.50, 0.01)
        draw_slider(scroll_frame, "Max Scene Luma",
                    "HISTOGRAM_MAX_LUMA", 0.40, 0.90, 0.01)

    # Edge-aware shadows
    edge_var = tk.BooleanVar(value=config["EDGE_AWARE_SHADOWS"])
    def toggle_edge():
        config["EDGE_AWARE_SHADOWS"] = edge_var.get()
        config_mgr.debounce_save()
        gamma_state.rebuild_debounced()

    tk.Checkbutton(
        scroll_frame,
        text="Edge-preserving Shadow Contrast",
        variable=edge_var,
        command=toggle_edge,
        bg=bg, fg=fg, selectcolor=bg
    ).pack(anchor="w", padx=6, pady=(6, 0))

    if edge_var.get():
        draw_slider(scroll_frame, "Edge Strength",
                    "EDGE_STRENGTH", 0.0, 1.0, 0.01)
        draw_slider(scroll_frame, "Edge Min",
                    "EDGE_MIN", 0.0, 0.5, 0.01)
        draw_slider(scroll_frame, "Edge Max",
                    "EDGE_MAX", 0.3, 1.0, 0.01)
                    
    # Motion-aware shadows
    motion_var = tk.BooleanVar(value=config["MOTION_AWARE_SHADOWS"]) 

    def toggle_motion():
        config["MOTION_AWARE_SHADOWS"] = motion_var.get()
        config_mgr.debounce_save()
        gamma_state.rebuild_debounced()

    tk.Checkbutton(
        scroll_frame,
        text="Motion-aware Shadow Boost",
        variable=motion_var,
        command=toggle_motion,
        bg=bg, fg=fg, selectcolor=bg
    ).pack(anchor="w", padx=6, pady=(6, 0))
    
    # === Motion Shadow Emphasis (Cone-like perceptual boost) ===
    ttk.Separator(scroll_frame).pack(fill="x", pady=10)

    tk.Label(
        scroll_frame,
        text="Motion Shadow Emphasis",
        bg=bg,
        fg="#bbb",
        font=("Segoe UI", 10, "bold")
    ).pack(anchor="w", padx=6, pady=(4, 6))

    motion_emph_var = tk.BooleanVar(value=config["MOTION_SHADOW_EMPHASIS"])

    def toggle_motion_emph():
        config["MOTION_SHADOW_EMPHASIS"] = motion_emph_var.get()
        config_mgr.debounce_save()
        gamma_state.rebuild_debounced()

    tk.Checkbutton(
        scroll_frame,
        text="Enable Motion-based Shadow Emphasis",
        variable=motion_emph_var,
        command=toggle_motion_emph,
        bg=bg, fg=fg, selectcolor=bg
    ).pack(anchor="w", padx=6)

    if motion_emph_var.get():
        draw_slider(
            scroll_frame,
            "Emphasis Strength",
            "MOTION_SHADOW_STRENGTH",
            0.0,
            0.85,      # HARD SAFE MAX (do not exceed)
            0.01
        )

        draw_slider(
            scroll_frame,
            "Max Scene Luma",
            "MOTION_SHADOW_DARK_LUMA",
            0.35,      # Prevents outdoor abuse
            0.60,      # Above this looks wrong
            0.01
        )

    if motion_var.get():
        m = gamma_state.current_mode.value.upper()

        draw_slider(scroll_frame, f"Motion Strength ({m})",
                    f"MOTION_STRENGTH_{m}", 0.0, 1.0, 0.01)
        draw_slider(scroll_frame, "Motion Sensitivity",
                    "MOTION_SENSITIVITY", 0.5, 5.0, 0.1)


    # Opponent channel tuning
    opp_var = tk.BooleanVar(value=config["OPPONENT_TUNING"])
    def toggle_opp():
        config["OPPONENT_TUNING"] = opp_var.get()
        config_mgr.debounce_save()
        gamma_state.rebuild_debounced()

    tk.Checkbutton(
        scroll_frame,
        text="Opponent-channel Shadow Tuning",
        variable=opp_var,
        command=toggle_opp,
        bg=bg, fg=fg, selectcolor=bg
    ).pack(anchor="w", padx=6, pady=(6, 0))

    if opp_var.get():
        draw_slider(scroll_frame, "Opponent Strength",
                    "OPPONENT_STRENGTH", 0.0, 1.0, 0.01)

    # HUD exclusion
    hud_ex_var = tk.BooleanVar(value=config["HUD_EXCLUSION"])
    def toggle_hud_ex():
        config["HUD_EXCLUSION"] = hud_ex_var.get()
        config_mgr.debounce_save()
        gamma_state.rebuild_debounced()

    tk.Checkbutton(
        scroll_frame,
        text="HUD-aware Highlight Exclusion",
        variable=hud_ex_var,
        command=toggle_hud_ex,
        bg=bg, fg=fg, selectcolor=bg
    ).pack(anchor="w", padx=6, pady=(6, 0))

    if hud_ex_var.get():
        draw_slider(scroll_frame, "HUD Exclusion Strength",
                    "HUD_EXCLUSION_STRENGTH", 0.0, 1.0, 0.01)


    m = gamma_state.current_mode.value.upper()

    # === Shadow Pop Strength Slider
    row = tk.Frame(scroll_frame, bg=bg)
    row.pack(fill=tk.X, padx=6, pady=4)
    tk.Label(row, text="Shadow Pop Strength", bg=bg, fg=fg, width=22, anchor="w").pack(side=tk.LEFT)

    val = tk.StringVar()
    def cb(v, dragging):
        config["SHADOW_POP_STRENGTH"] = v
        gamma_state.rebuild_debounced()
        config_mgr.debounce_save()
        val.set(f"{v:.2f}")
        lbl.configure(fg="#777" if dragging else "#aaa")

    cs = CanvasSlider(row, 0.0, 1.0, config["SHADOW_POP_STRENGTH"], cb, 0.01)
    cs.pack(side=tk.LEFT, padx=6)
    val.set(f"{config['SHADOW_POP_STRENGTH']:.2f}")
    lbl = tk.Label(row, textvariable=val, bg=bg, fg="#aaa", width=8)
    lbl.pack(side=tk.LEFT)
    tk.Button(row, text="Reset", bg=btn, fg=fg, width=6,
              command=lambda: cs.set_value(0.0)).pack(side=tk.LEFT, padx=(6, 0))

    # === Primary Sliders
    sliders = [
        ("Gamma", f"GAMMA_{m}", 0.75, 1.4, 0.001),
        ("Gamma Offset", f"GAMMA_OFFSET_{m}", -0.2, 0.2, 0.001),
        ("Vibrance", f"VIBRANCE_{m}", 0.9, 1.3, 0.001),
        ("Black Floor", f"BLACK_FLOOR_{m}", 0.0, 0.1, 0.0005),
        ("Shadow Lift Exp", f"SHADOW_LIFT_EXP_{m}", 0.4, 1.0, 0.001),
        ("Shadow Cutoff", f"SHADOW_CUTOFF_{m}", 0.15, 0.6, 0.001),
        ("Shadow Desat", f"SHADOW_DESAT_{m}", 0.5, 1.0, 0.001),
        ("Shadow Sigmoid Boost", f"SHADOW_SIGMOID_BOOST_{m}", 0.0, 1.0, 0.01),
        ("Shadow Color Bias", f"SHADOW_COLOR_BIAS_{m}", -0.05, 0.05, 0.001),
        ("Midtone Boost", f"MIDTONE_BOOST_{m}", 0.9, 1.3, 0.001),
        ("Highlight Compress", f"HIGHLIGHT_COMPRESS_{m}", 0.0, 0.7, 0.001),
    ]
    for s in sliders:
        draw_slider(scroll_frame, *s)

    # === Advanced Controls
    if config["ADVANCED_MODE"]:
        ttk.Separator(scroll_frame).pack(fill="x", pady=10)

        tk.Label(
            scroll_frame,
            text="Advanced / RGB Channel Bias",
            bg=bg,
            fg="#bbb",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=6, pady=(4, 6))

        advanced_sliders = [
            ("Shadow Red Bias", f"SHADOW_RED_BIAS_{m}", 0.95, 1.05, 0.001),
            ("Shadow Green Bias", f"SHADOW_GREEN_BIAS_{m}", 0.95, 1.05, 0.001),
            ("Shadow Blue Bias", f"SHADOW_BLUE_BIAS_{m}", 0.95, 1.05, 0.001),
            ("Red Multiplier", f"RED_MULTIPLIER_{m}", 0.95, 1.05, 0.001),
            ("Green Multiplier", f"GREEN_MULTIPLIER_{m}", 0.95, 1.05, 0.001),
            ("Blue Multiplier", f"BLUE_MULTIPLIER_{m}", 0.95, 1.05, 0.001),
        ]
        for s in advanced_sliders:
            draw_slider(scroll_frame, *s)

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
    scrollbar = ttk.Scrollbar(settings_window, orient="vertical", command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    scroll_frame = tk.Frame(canvas, bg=bg)
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    settings_window.protocol("WM_DELETE_WINDOW", settings_window.withdraw)

    rebuild_gui()

# ================= MAIN =================
gamma_state.rebuild()
root.mainloop()
