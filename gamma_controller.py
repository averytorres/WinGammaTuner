"""
gamma_controller.py

Owns the Win32 gamma ramp application (SetDeviceGammaRamp) + debounced rebuilds.
No UI code. No hotkeys. Calls into GammaPipeline to build ramps.

Public API:
- GammaController(root, pipeline, user32, gdi32, hdc, executor)
    - current_mode (Mode|None)
    - toggle_mode(mode)
    - set_mode(mode|None)
    - rebuild()
    - rebuild_debounced()
    - shutdown()  # restores identity
"""

from __future__ import annotations

import atexit
import zlib
from threading import Lock, Timer
from typing import Optional

import numpy as np


class GammaController:
    def __init__(self, root, pipeline, user32, gdi32, hdc, executor):
        self.root = root
        self.pipeline = pipeline
        self.user32 = user32
        self.gdi32 = gdi32
        self.hdc = hdc
        self.executor = executor

        self.current_mode = None
        self._lock = Lock()
        self._timer: Optional[Timer] = None

        self._last_sig = None
        atexit.register(self.shutdown)

    @staticmethod
    def _to_ramp(arr: np.ndarray):
        import ctypes
        return (ctypes.c_ushort * 768)(*arr.tolist())

    @staticmethod
    def _crc(arr: np.ndarray) -> int:
        return zlib.crc32(arr.tobytes())

    def toggle_mode(self, mode):
        self.current_mode = None if self.current_mode == mode else mode
        self.rebuild()

    def set_mode(self, mode):
        self.current_mode = mode
        self.rebuild()

    def rebuild(self):
        if self.current_mode is None:
            arr = self.pipeline.identity_ramp()
        else:
            arr = self.pipeline.build(self.current_mode)

        self.executor.submit(self._apply, arr)

    def rebuild_debounced(self, delay: float = 0.03):
        if self._timer:
            self._timer.cancel()

        self._timer = Timer(delay, lambda: self.root.after(0, self.rebuild))
        self._timer.daemon = True
        self._timer.start()

    def shutdown(self):
        try:
            arr = self.pipeline.identity_ramp()
            self._apply(arr, force=True)
        except Exception:
            pass

    def _apply(self, arr: np.ndarray, force: bool = False):
        sig = self._crc(arr)
        with self._lock:
            if (not force) and sig == self._last_sig:
                return
            self.gdi32.SetDeviceGammaRamp(self.hdc, self._to_ramp(arr))
            self._last_sig = sig
