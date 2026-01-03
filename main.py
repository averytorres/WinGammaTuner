"""
main.py

Entry point. Run this file.

Hotkeys:
- F6: toggle UI
- F8: toggle INDOOR
- F9: toggle OUTDOOR
"""

from __future__ import annotations

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
    def __init__(self, path: str, defaults: dict, executor: ThreadPoolExecutor):
        self.path = path
        self.defaults = defaults
        self.executor = executor
        self._lock = Lock()
        self._timer: Timer | None = None
        self.data = self.load()

    def load(self) -> dict:
        try:
            with open(self.path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            data = {}

        for k, v in self.defaults.items():
            data.setdefault(k, v)

        return data

    def save(self) -> None:
        with self._lock:
            with open(self.path, "wb") as f:
                f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))

    def debounce_save(self, delay: float = 0.5) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = Timer(delay, lambda: self.executor.submit(self.save))
        self._timer.daemon = True
        self._timer.start()


def main():
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

    controller_placeholder = {"ctrl": None}

    def is_active():
        ctrl = controller_placeholder["ctrl"]
        return (ctrl is not None) and (ctrl.current_mode is not None)

    analyzer = SceneAnalyzer(user32, gdi32, hdc, config, is_active=is_active)
    analyzer.start(executor)

    pipeline = GammaPipeline(config, scene_analyzer=analyzer)

    controller = GammaController(root, pipeline, user32, gdi32, hdc, executor)
    controller_placeholder["ctrl"] = controller

    ui = GammaUI(root, controller, config, DEFAULT_CONFIG, config_mgr)

    controller.rebuild()

    def on_press(key):
        try:
            if key == keyboard.Key.f6:
                root.after(0, ui.toggle)
            elif key == keyboard.Key.f8:
                controller.toggle_mode(Mode.INDOOR)
                root.after(0, ui.rebuild)
            elif key == keyboard.Key.f9:
                controller.toggle_mode(Mode.OUTDOOR)
                root.after(0, ui.rebuild)
        except Exception:
            return

    keyboard.Listener(on_press=on_press, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
