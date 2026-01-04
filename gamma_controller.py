"""
Owns the Win32 gamma ramp application (SetDeviceGammaRamp) + debounced rebuilds.
No UI code. No hotkeys. Calls into GammaPipeline to build ramps.
"""

from __future__ import annotations

import atexit
import ctypes
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
        self._last_sig: Optional[int] = None

        atexit.register(self.shutdown)

    # ───────────────────────── Internal helpers ─────────────────────────

    @staticmethod
    def _to_ramp(arr: np.ndarray):
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16, copy=False)
        arr = np.ascontiguousarray(arr)
        return (ctypes.c_ushort * 768)(*arr)

    @staticmethod
    def _signature(arr: np.ndarray) -> int:
        return zlib.crc32(np.ascontiguousarray(arr).tobytes())

    def _build_ramp(self) -> np.ndarray:
        if self.current_mode is None:
            return self.pipeline.identity_ramp()
        return self.pipeline.build(self.current_mode)

    # ───────────────────────── Public API ─────────────────────────

    def toggle_mode(self, mode):
        self.set_mode(None if self.current_mode == mode else mode)

    def set_mode(self, mode):
        self.current_mode = mode
        self.rebuild(force=True)

    def rebuild(self, *, force: bool = False):
        """
        Build and apply gamma ramp asynchronously.
        Redundant applies are skipped via signature comparison.
        """
        arr = self._build_ramp()

        self.executor.submit(self._apply, arr, force=force)

    def rebuild_debounced(self, delay: float = 0.03):
        with self._lock:
            if self._timer:
                self._timer.cancel()

            def _fire():
                with self._lock:
                    self._timer = None
                self.root.after(0, self.rebuild)

            self._timer = Timer(delay, _fire)
            self._timer.daemon = True
            self._timer.start()

    def shutdown(self):
        try:
            arr = self.pipeline.identity_ramp()
            self._apply(arr, force=True)
        except Exception:
            pass

    # ───────────────────────── Win32 apply ─────────────────────────

    def _apply(self, arr: np.ndarray, *, force: bool = False):
        sig = self._signature(arr)

        with self._lock:
            if not force and sig == self._last_sig:
                return
            self._last_sig = sig

        ramp = self._to_ramp(arr)
        self.gdi32.SetDeviceGammaRamp(self.hdc, ramp)
