"""
Builds 256-step (768 entry) Win32 GDI gamma ramps for INDOOR / OUTDOOR profiles.
Pure math + LUT layer (no UI, no GDI).

Public API:
- Mode (Enum)
- DEFAULT_CONFIG (dict)
- GammaPipeline(config: dict, scene_analyzer: SceneAnalyzer | None)
    - identity_ramp() -> np.ndarray[uint16] shape (768,)
    - build(mode: Mode) -> np.ndarray[uint16] shape (768,)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np


class Mode(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_CONFIG: Dict[str, object] = {
    "SHADOW_POP_STRENGTH": 0.20,
    "PRESERVE_HUD_HIGHLIGHTS": True,

    # INDOOR
    "GAMMA_INDOOR": 1.28,
    "GAMMA_OFFSET_INDOOR": 0.0,
    "VIBRANCE_INDOOR": 1.04,
    "BLACK_FLOOR_INDOOR": 0.030,
    "SHADOW_LIFT_EXP_INDOOR": 0.62,
    "SHADOW_CUTOFF_INDOOR": 0.42,
    "SHADOW_DESAT_INDOOR": 0.82,
    "SHADOW_COLOR_BIAS_INDOOR": 0.0,
    "SHADOW_RED_BIAS_INDOOR": 1.00,
    "SHADOW_GREEN_BIAS_INDOOR": 1.00,
    "SHADOW_BLUE_BIAS_INDOOR": 1.00,
    "MIDTONE_BOOST_INDOOR": 1.02,
    "HIGHLIGHT_COMPRESS_INDOOR": 0.32,
    "RED_MULTIPLIER_INDOOR": 1.00,
    "GREEN_MULTIPLIER_INDOOR": 1.00,
    "BLUE_MULTIPLIER_INDOOR": 1.00,
    "SHADOW_SIGMOID_BOOST_INDOOR": 0.35,

    # OUTDOOR
    "GAMMA_OUTDOOR": 1.12,
    "GAMMA_OFFSET_OUTDOOR": 0.0,
    "VIBRANCE_OUTDOOR": 1.03,
    "BLACK_FLOOR_OUTDOOR": 0.022,
    "SHADOW_LIFT_EXP_OUTDOOR": 0.75,
    "SHADOW_CUTOFF_OUTDOOR": 0.34,
    "SHADOW_DESAT_OUTDOOR": 0.90,
    "SHADOW_COLOR_BIAS_OUTDOOR": 0.0,
    "SHADOW_RED_BIAS_OUTDOOR": 1.00,
    "SHADOW_GREEN_BIAS_OUTDOOR": 1.00,
    "SHADOW_BLUE_BIAS_OUTDOOR": 1.00,
    "MIDTONE_BOOST_OUTDOOR": 1.00,
    "HIGHLIGHT_COMPRESS_OUTDOOR": 0.48,
    "RED_MULTIPLIER_OUTDOOR": 1.00,
    "GREEN_MULTIPLIER_OUTDOOR": 1.00,
    "BLUE_MULTIPLIER_OUTDOOR": 1.00,
    "SHADOW_SIGMOID_BOOST_OUTDOOR": 0.18,

    # Adaptive
    "HISTOGRAM_ADAPTIVE": True,
    "HISTOGRAM_STRENGTH": 0.30,
    "HISTOGRAM_MIN_LUMA": 0.12,
    "HISTOGRAM_MAX_LUMA": 0.55,

    "EDGE_AWARE_SHADOWS": False,
    "EDGE_STRENGTH": 0.40,
    "EDGE_MIN": 0.05,
    "EDGE_MAX": 0.35,

    "OPPONENT_TUNING": False,
    "OPPONENT_STRENGTH": 0.25,

    "HUD_EXCLUSION": False,
    "HUD_EXCLUSION_STRENGTH": 0.60,
    "HUD_EXCLUSION_THRESHOLD": 0.90,

    "MOTION_AWARE_SHADOWS": True,
    "MOTION_STRENGTH_INDOOR": 0.55,
    "MOTION_STRENGTH_OUTDOOR": 0.35,
    "MOTION_SENSITIVITY": 2.3,
    "MOTION_SMOOTHING": 0.15,

    "MOTION_SHADOW_EMPHASIS": True,
    "MOTION_SHADOW_STRENGTH": 0.65,
    "MOTION_SHADOW_DARK_LUMA": 0.50,
    "MOTION_SHADOW_MIN_MOTION": 0.015,
}


# Shared LUT axis
X_AXIS = np.linspace(0.0, 1.0, 256)
MID_05_MASK = X_AXIS < 0.5
MID_07_MASK = X_AXIS < 0.7

# Static dithering noise
_DITHER_NOISE = (np.random.rand(256) - 0.5) * (1.0 / 65535.0)


def identity_ramp() -> np.ndarray:
    return np.concatenate([np.linspace(0, 65535, 256, dtype=np.uint16)] * 3)


@lru_cache(maxsize=32)
def _base_curve_cached(gamma: float, offset: float) -> np.ndarray:
    x = X_AXIS
    if offset:
        x = np.clip(x + offset, 0.0, 1.0)
    if gamma != 1.0:
        x = np.power(x, 1.0 / gamma)
    return x


class GammaPipeline:
    def __init__(self, config: Dict[str, object], scene_analyzer=None):
        self.config = config
        self.scene_analyzer = scene_analyzer
        self._adaptive_state: Dict[str, float] = {}

    def identity_ramp(self) -> np.ndarray:
        return identity_ramp()

    def build(self, mode: Mode) -> np.ndarray:
        return self._build_ramp_array(mode)

    # ───────────────────────── Adaptive helpers ─────────────────────────

    def adaptive_enabled(self) -> bool:
        c = self.config
        return bool(
            c.get("HISTOGRAM_ADAPTIVE", False)
            or c.get("EDGE_AWARE_SHADOWS", False)
            or c.get("OPPONENT_TUNING", False)
            or c.get("HUD_EXCLUSION", False)
            or c.get("MOTION_AWARE_SHADOWS", False)
            or c.get("MOTION_SHADOW_EMPHASIS", False)
        )

    def _get_scene_metrics(self) -> Tuple[float, float, float, float]:
        if self.scene_analyzer is None:
            return (0.5, 0.0, 0.0, 0.0)
        return self.scene_analyzer.get_metrics()

    def _smooth(self, key: str, target: float, alpha: float) -> float:
        prev = self._adaptive_state.get(key, target)
        value = prev * (1 - alpha) + target * alpha
        self._adaptive_state[key] = value
        return value

    # ───────────────────────── Build ─────────────────────────

    def _build_ramp_array(self, mode: Mode) -> np.ndarray:
        c = self.config
        suffix = mode.value.upper()

        # Snapshot profile params once
        p = {
            k.replace(f"_{suffix}", ""): float(v)
            for k, v in c.items()
            if k.endswith(suffix)
        }
        p["PROFILE"] = suffix

        adaptive = self.adaptive_enabled()
        metrics = self._get_scene_metrics() if adaptive else (0.5, 0.0, 0.0, 0.0)
        avg_luma, shadow_density, edge_strength, motion = metrics

        # Shadow pop
        pop = float(np.clip(c.get("SHADOW_POP_STRENGTH", 0.0), 0.0, 1.0))
        if pop > 0.0:
            p["GAMMA"] *= 1.0 + 0.04 * pop
            p["SHADOW_SIGMOID_BOOST"] += 0.15 * pop
            p["SHADOW_DESAT"] -= 0.06 * pop
            p["MIDTONE_BOOST"] *= 1.0 + 0.03 * pop

        # Base curve
        base = _base_curve_cached(p["GAMMA"], p["GAMMA_OFFSET"]).copy()
        bf = float(np.clip(p["BLACK_FLOOR"], 0.0, 0.1))
        base = bf + base * (1.0 - bf)

        x = X_AXIS
        cutoff = float(np.clip(p["SHADOW_CUTOFF"], 0.15, 0.6))

        # Shadow lift
        shadow = x < cutoff
        lum = base.copy()
        lum[shadow] = np.power(lum[shadow] / cutoff, p["SHADOW_LIFT_EXP"]) * cutoff
        base = lum / np.maximum(lum, 1e-6) * lum

        # Midtones
        mid = MID_07_MASK & ~shadow
        base[mid] *= p["MIDTONE_BOOST"]

        # Highlights
        hi = x > 0.85
        base[hi] = 0.85 + (base[hi] - 0.85) * p["HIGHLIGHT_COMPRESS"]

        # Dithering + RGB
        r = base * p["RED_MULTIPLIER"]
        g = base * p["GREEN_MULTIPLIER"]
        b = base * p["BLUE_MULTIPLIER"]

        desat = float(np.clip(p["SHADOW_DESAT"], 0.5, 1.0))
        if desat < 1.0:
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            r[shadow] = lum[shadow] + (r[shadow] - lum[shadow]) * desat
            g[shadow] = lum[shadow] + (g[shadow] - lum[shadow]) * desat
            b[shadow] = lum[shadow] + (b[shadow] - lum[shadow]) * desat

        dither = _DITHER_NOISE * (0.5 + 0.5 * base)
        r = np.clip(r + dither, 0.0, 1.0)
        g = np.clip(g + dither, 0.0, 1.0)
        b = np.clip(b + dither, 0.0, 1.0)

        scale = 65535.0 * float(p["VIBRANCE"])
        return np.concatenate(
            (
                np.clip(r * scale, 0, 65535).astype(np.uint16),
                np.clip(g * scale, 0, 65535).astype(np.uint16),
                np.clip(b * scale, 0, 65535).astype(np.uint16),
            )
        )
