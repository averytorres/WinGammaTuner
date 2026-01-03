"""
scene_analyzer.py

Low-res desktop sampling + scene metrics for adaptive gamma features.
Separated from the LUT math and from Tk UI.

Public API:
- SceneAnalyzer(user32, gdi32, hdc, config: dict, is_active: callable)
    - start(executor)  # optional background loop
    - stop()
    - update()         # single update (thread-safe)
    - get_metrics() -> (avg_luma, shadow_density, edge_strength, motion_strength)
"""

from __future__ import annotations

import time
from threading import Lock, Event
from typing import Callable, Tuple

import numpy as np


class SceneAnalyzer:
    def __init__(self, user32, gdi32, hdc, config: dict, is_active: Callable[[], bool]):
        self.user32 = user32
        self.gdi32 = gdi32
        self.hdc = hdc
        self.config = config
        self.is_active = is_active

        self.avg_luma = 0.5
        self.shadow_density = 0.0
        self.edge_strength = 0.0
        self.motion_strength = 0.0

        self._prev_luma = None
        self._lock = Lock()
        self._last_update = 0.0

        self._stop_evt = Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def get_metrics(self) -> Tuple[float, float, float, float]:
        with self._lock:
            return (self.avg_luma, self.shadow_density, self.edge_strength, self.motion_strength)

    def start(self, executor) -> None:
        """Start background sampling loop (safe to call once)."""
        executor.submit(self._worker_loop)

    def _worker_loop(self) -> None:
        while not self._stop_evt.is_set():
            if not self.is_active() or not self._adaptive_enabled():
                time.sleep(0.25)
                continue

            self.update()
            time.sleep(0.05)

    def _adaptive_enabled(self) -> bool:
        c = self.config
        return bool(
            c.get("HISTOGRAM_ADAPTIVE", False)
            or c.get("EDGE_AWARE_SHADOWS", False)
            or c.get("OPPONENT_TUNING", False)
            or c.get("HUD_EXCLUSION", False)
            or c.get("MOTION_AWARE_SHADOWS", False)
            or c.get("MOTION_SHADOW_EMPHASIS", False)
        )

    def _sample_desktop_rgb(self, w: int = 64, h: int = 64) -> np.ndarray:
        """
        Capture a low-resolution RGB sample of the current desktop.
        FPS-safe: single StretchBlt + bitmap readback.
        Returns float32 RGB array in [0,1].
        """
        memdc = self.gdi32.CreateCompatibleDC(self.hdc)
        bmp = self.gdi32.CreateCompatibleBitmap(self.hdc, w, h)
        self.gdi32.SelectObject(memdc, bmp)

        self.gdi32.StretchBlt(
            memdc,
            0, 0, w, h,
            self.hdc,
            0, 0,
            self.user32.GetSystemMetrics(0),
            self.user32.GetSystemMetrics(1),
            0x00CC0020,  # SRCCOPY
        )

        import ctypes
        buf = (ctypes.c_ubyte * (w * h * 4))()
        self.gdi32.GetBitmapBits(bmp, len(buf), buf)

        self.gdi32.DeleteObject(bmp)
        self.gdi32.DeleteDC(memdc)

        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        return arr[:, :, :3].astype(np.float32) / 255.0

    def _compute_motion(self, luma: np.ndarray) -> float:
        """
        Motion strength from luma delta (mean abs diff).
        Handles enable/disable and internal state reset.
        """
        if not self.config.get("MOTION_AWARE_SHADOWS", False):
            self._prev_luma = None
            self.motion_strength = 0.0
            return 0.0

        motion = 0.0
        if self._prev_luma is not None:
            diff = np.abs(luma - self._prev_luma)
            motion = float(diff.mean())

        self._prev_luma = luma
        return motion

    def update(self) -> None:
        """
        Update metrics with built-in caps:
          - won't run if window minimized or no foreground window
          - 5Hz cap
          - stall / hitch guard (delta > 0.6s)
        """
        if not self.is_active():
            self._last_update = time.time()
            return

        if not self._adaptive_enabled():
            self._last_update = time.time()
            return

        now = time.time()
        delta = now - self._last_update

        if delta > 0.6:
            self._last_update = now
            return

        if delta < 0.2:
            return

        self._last_update = now

        hwnd = self.user32.GetForegroundWindow()
        if not hwnd or self.user32.IsIconic(hwnd):
            return

        try:
            rgb = self._sample_desktop_rgb()
            luma = rgb.mean(axis=2)

            motion = self._compute_motion(luma)
            avg = float(luma.mean())
            shadow = float((luma < 0.25).mean())

            gx = np.abs(np.diff(luma, axis=1))
            gy = np.abs(np.diff(luma, axis=0))
            edge = float(np.mean(gx) + np.mean(gy))

            with self._lock:
                alpha = 0.2
                self.avg_luma = self.avg_luma * (1 - alpha) + avg * alpha
                self.shadow_density = self.shadow_density * (1 - alpha) + shadow * alpha
                self.edge_strength = self.edge_strength * (1 - alpha) + edge * alpha

                if self.config.get("MOTION_AWARE_SHADOWS", False):
                    m_alpha = float(self.config.get("MOTION_SMOOTHING", 0.15))
                    self.motion_strength = self.motion_strength * (1 - m_alpha) + motion * m_alpha

        except Exception:
            return
