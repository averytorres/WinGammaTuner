"""
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
from typing import Callable

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

        self.data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            data = {}

        for key, value in self.defaults.items():
            data.setdefault(key, value)

        return data

    def save(self) -> None:
        with self._lock:
            payload = orjson.dumps(self.data, option=orjson.OPT_INDENT_2)
            with open(self.path, "wb") as f:
                f.write(payload)

    def debounce_save(self, delay: float = 0.5) -> None:
        if self._timer:
            self._timer.cancel()

        self._timer = Timer(
            delay,
            lambda: self.executor.submit(self.save),
        )
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

    analyzer = SceneAnalyzer(
        user32,
        gdi32,
        hdc,
        config,
        is_active=is_active,
    )
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

    ui = GammaUI(root, controller, config, DEFAULT_CONFIG, config_mgr)
    controller.rebuild()

    def toggle_and_rebuild(mode: Mode) -> None:
        controller.toggle_mode(mode)
        root.after(0, ui.rebuild)

    hotkey_actions: dict[keyboard.Key, Callable[[], None]] = {
        keyboard.Key.f6: lambda: root.after(0, ui.toggle),
        keyboard.Key.f8: lambda: toggle_and_rebuild(Mode.INDOOR),
        keyboard.Key.f9: lambda: toggle_and_rebuild(Mode.OUTDOOR),
    }

    def on_press(key) -> None:
        action = hotkey_actions.get(key)
        if action:
            try:
                action()
            except Exception:
                logging.exception("Hotkey action failed")

    keyboard.Listener(on_press=on_press, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
