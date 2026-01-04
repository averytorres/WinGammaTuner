"""
Low-res desktop sampling + scene metrics for adaptive gamma features.
Separated from the LUT math and from Tk UI.

Public API:
- SceneAnalyzer(user32, gdi32, hdc, config: dict, is_active: callable)
    - start(executor)
    - stop()
    - update()
    - get_metrics() -> (avg_luma, shadow_density, edge_strength, motion_strength)
"""
from __future__ import annotations

import time
from threading import Lock, Event
from typing import Callable, Tuple

import numpy as np
import ctypes


class SceneAnalyzer:
    __slots__ = (
        "user32", "gdi32", "hdc", "config", "is_active",
        "avg_luma", "shadow_density", "edge_strength", "motion_strength",
        "_prev_luma", "_lock", "_last_update", "_stop_evt",
    )

    def __init__(self, user32, gdi32, hdc, config: dict, is_active: Callable[[], bool]):
        self.user32 = user32
        self.gdi32 = gdi32
        self.hdc = hdc
        self.config = config
        self.is_active = is_active

        # Public metrics (smoothed)
        self.avg_luma = 0.5
        self.shadow_density = 0.0
        self.edge_strength = 0.0
        self.motion_strength = 0.0

        # Internal state
        self._prev_luma: np.ndarray | None = None
        self._lock = Lock()
        self._last_update = 0.0
        self._stop_evt = Event()

    def start(self, executor) -> None:
        executor.submit(self._worker_loop)

    def stop(self) -> None:
        self._stop_evt.set()

    def get_metrics(self) -> Tuple[float, float, float, float]:
        with self._lock:
            return (
                self.avg_luma,
                self.shadow_density,
                self.edge_strength,
                self.motion_strength,
            )

    def _worker_loop(self) -> None:
        sleep = time.sleep
        while not self._stop_evt.is_set():
            if not self.is_active() or not self._adaptive_enabled():
                sleep(0.25)
                continue

            self.update()
            sleep(0.05)

    def _adaptive_enabled(self) -> bool:
        c = self.config
        return any(
            c.get(k, False)
            for k in (
                "HISTOGRAM_ADAPTIVE",
                "EDGE_AWARE_SHADOWS",
                "OPPONENT_TUNING",
                "HUD_EXCLUSION",
                "MOTION_AWARE_SHADOWS",
                "MOTION_SHADOW_EMPHASIS",
            )
        )

    def _sample_desktop_rgb(self, w: int = 64, h: int = 64) -> np.ndarray:
        user32 = self.user32
        gdi32 = self.gdi32

        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)

        memdc = gdi32.CreateCompatibleDC(self.hdc)
        bmp = gdi32.CreateCompatibleBitmap(self.hdc, w, h)
        gdi32.SelectObject(memdc, bmp)

        try:
            gdi32.StretchBlt(
                memdc,
                0, 0, w, h,
                self.hdc,
                0, 0,
                screen_w,
                screen_h,
                0x00CC0020,
            )

            buf = (ctypes.c_ubyte * (w * h * 4))()
            gdi32.GetBitmapBits(bmp, len(buf), buf)

            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            return arr[:, :, :3].astype(np.float32) * (1.0 / 255.0)
        finally:
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(memdc)

    def _compute_motion(self, luma: np.ndarray) -> float:
        if not self.config.get("MOTION_AWARE_SHADOWS", False):
            self._prev_luma = None
            return 0.0

        prev = self._prev_luma
        self._prev_luma = luma

        if prev is None:
            return 0.0

        return float(np.abs(luma - prev).mean())

    def update(self) -> None:
        if not self.is_active() or not self._adaptive_enabled():
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

            avg = float(luma.mean())
            shadow = float((luma < 0.25).mean())

            gx = np.abs(np.diff(luma, axis=1)).mean()
            gy = np.abs(np.diff(luma, axis=0)).mean()
            edge = float(gx + gy)

            motion = self._compute_motion(luma)

            with self._lock:
                alpha = 0.2
                self.avg_luma += (avg - self.avg_luma) * alpha
                self.shadow_density += (shadow - self.shadow_density) * alpha
                self.edge_strength += (edge - self.edge_strength) * alpha

                if self.config.get("MOTION_AWARE_SHADOWS", False):
                    m_alpha = float(self.config.get("MOTION_SMOOTHING", 0.15))
                    self.motion_strength += (
                        (motion - self.motion_strength) * m_alpha
                    )
                else:
                    self.motion_strength = 0.0
        except Exception:
            return
