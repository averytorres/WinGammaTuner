from __future__ import annotations

MONOLITH_DEFAULTS = {
    "SHADOW_POP_STRENGTH": 0.20,          # Gentle global boost for silhouette clarity without washout
    "PRESERVE_HUD_HIGHLIGHTS": True,
    "GLOBAL_BRIGHTNESS_INDOOR": 1.0,
    "GLOBAL_BRIGHTNESS_ENABLED_INDOOR": True,

    "GLOBAL_BRIGHTNESS_OUTDOOR": 1.0,
    "GLOBAL_BRIGHTNESS_ENABLED_OUTDOOR": True,
      # Required due to bright interiors / snow / UI overlap

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

    # ================= Adaptive defaults (monolith parity) =================

    # Histogram-aware shadow shaping
    "HISTOGRAM_ADAPTIVE": True,
    "HISTOGRAM_STRENGTH": 0.30,
    "HISTOGRAM_MIN_LUMA": 0.12,
    "HISTOGRAM_MAX_LUMA": 0.55,

    # Edge-preserving shadow contrast
    "EDGE_AWARE_SHADOWS": False,
    "EDGE_STRENGTH": 0.40,
    "EDGE_MIN": 0.05,
    "EDGE_MAX": 0.35,

    # Color opponent channel tuning
    "OPPONENT_TUNING": False,
    "OPPONENT_STRENGTH": 0.25,

    # HUD-aware exclusion
    "HUD_EXCLUSION": False,
    "HUD_EXCLUSION_STRENGTH": 0.60,
    "HUD_EXCLUSION_THRESHOLD": 0.90,

    # Motion-aware adaptive shadows
    "MOTION_AWARE_SHADOWS": True,
    "MOTION_STRENGTH_INDOOR": 0.55,
    "MOTION_STRENGTH_OUTDOOR": 0.35,
    "MOTION_SENSITIVITY": 2.3,
    "MOTION_SMOOTHING": 0.15,

    # Motion shadow emphasis (cone-like perceptual boost)
    "MOTION_SHADOW_EMPHASIS": True,
    "MOTION_SHADOW_STRENGTH": 0.65,
    "MOTION_SHADOW_DARK_LUMA": 0.50,
    "MOTION_SHADOW_MIN_MOTION": 0.015,

}

"""
Entry point. Run this file.

Hotkeys:
- <f6>: toggle UI
- <f8>: toggle INDOOR
- <f9>: toggle OUTDOOR
"""


import ctypes
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Timer

import orjson
import tkinter as tk
from pynput import keyboard

from gamma_pipeline import GammaPipeline, Mode, DEFAULT_CONFIG
from scene_analyzer import SceneAnalyzer
from gamma_controller import GammaController
from gamma_ui import GammaUI

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")




class ConfigManager:
    """
    Correct persistence contract:

    - Defaults are applied ONLY for keys that do not exist on disk
    - User-modified values are NEVER overwritten on startup
    - Reset buttons explicitly write defaults
    - All writes are persisted
    """

    def __init__(self, path: str, defaults: dict, executor):
        self.path = path
        self.defaults = dict(defaults)
        self.executor = executor

        self._lock = Lock()
        self._timer: Timer | None = None
        self.data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path, "rb") as f:
                data = orjson.loads(f.read())
                if not isinstance(data, dict):
                    data = {}
        except Exception:
            data = {}

        # Apply defaults ONLY for missing keys
        for k, v in self.defaults.items():
            if k not in data:
                data[k] = v

        return data

    def save(self) -> None:
        with self._lock:
            with open(self.path, "wb") as f:
                f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))

    def debounce_save(self, delay: float = 0.4) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()

            def _fire():
                with self._lock:
                    self._timer = None
                self.executor.submit(self.save)

            self._timer = Timer(delay, _fire)
            self._timer.daemon = True
            self._timer.start()
def main() -> None:
    executor = ThreadPoolExecutor(max_workers=2)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "gamma_config.json")

    config_mgr = ConfigManager(config_path, DEFAULT_CONFIG, executor)
    config = config_mgr.data

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hdc = user32.GetDC(0)

    root = tk.Tk()
    root.withdraw()

    controller: GammaController | None = None

    def is_active() -> bool:
        return controller is not None and controller.current_mode is not None

    analyzer = SceneAnalyzer(user32, gdi32, hdc, config, is_active=is_active)
    analyzer.start(executor)

    pipeline = GammaPipeline(config, scene_analyzer=analyzer)

    controller = GammaController(
        root,
        pipeline,
        user32,
        gdi32,
        hdc,
        executor,
    )

    ui = GammaUI(root, controller, config, MONOLITH_DEFAULTS, config_mgr)

    controller.rebuild()

    # ───────────────────────── Hotkeys (robust) ─────────────────────────
    # Use GlobalHotKeys (more reliable for function keys than Key matching in on_press)

    def _toggle_ui():
        root.after(0, ui.toggle)

    def _toggle_indoor():
        controller.toggle_mode(Mode.INDOOR)
        root.after(0, ui.rebuild)
        controller.rebuild(force=True)

    def _toggle_outdoor():
        controller.toggle_mode(Mode.OUTDOOR)
        root.after(0, ui.rebuild)
        controller.rebuild(force=True)

    hotkeys = keyboard.GlobalHotKeys(
        {
            "<f6>": _toggle_ui,
            "<f8>": _toggle_indoor,
            "<f9>": _toggle_outdoor,
        }
    )

    # Also attach a raw listener *just for diagnostics*.
    # If you see these logs, the hook is working and the issue is mapping.
    def _debug_press(k):
        logging.debug("Key event: %r", k)

    debug_listener = keyboard.Listener(on_press=_debug_press)

    # Keep strong refs
    hotkeys.start()
    debug_listener.start()

    try:
        root.mainloop()
    finally:
        try:
            hotkeys.stop()
        except Exception:
            pass
        try:
            debug_listener.stop()
        except Exception:
            pass

        try:
            controller.shutdown()
        except Exception:
            pass

        try:
            config_mgr.save()
        except Exception:
            pass

        executor.shutdown(wait=False)


if __name__ == "__main__":
    main()