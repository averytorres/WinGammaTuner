"""
gamma_pipeline.py

Builds 256-step (768 entry) Win32 GDI gamma ramps for INDOOR / OUTDOOR profiles.
This is the "math + LUT" layer only: no Tk, no hotkeys, no GDI calls.

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
# Defaults (mirrors your original monolith)
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

    # ---------------- Adaptive features (safe defaults) ----------------
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


# Shared LUT axis + masks (matches original)
X_AXIS = np.linspace(0.0, 1.0, 256)
MID_05_MASK = X_AXIS < 0.5
MID_07_MASK = X_AXIS < 0.7

# Perceptual dithering noise (constant per run; matches original intent)
_DITHER_NOISE = (np.random.rand(256) - 0.5) * (1.0 / 65535.0)


def identity_ramp() -> np.ndarray:
    """Identity gamma ramp (768 uint16)."""
    return np.concatenate([np.linspace(0, 65535, 256, dtype=np.uint16)] * 3)


@lru_cache(maxsize=32)
def _base_curve_cached(gamma: float, offset: float) -> np.ndarray:
    """Cached base curve on the shared X axis. Returns float64 array in [0,1]."""
    x = X_AXIS
    if offset:
        x = np.clip(x + offset, 0.0, 1.0)
    if gamma != 1.0:
        x = np.power(x, 1.0 / gamma)
    return x


class GammaPipeline:
    """
    Pure LUT builder. Reads values from `config` (dict-like).

    scene_analyzer (optional) must provide:
        get_metrics() -> (avg_luma, shadow_density, edge_strength, motion_strength)
    """

    def __init__(self, config: Dict[str, object], scene_analyzer=None):
        self.config = config
        self.scene_analyzer = scene_analyzer
        self._adaptive_state: Dict[str, float] = {}

    def identity_ramp(self) -> np.ndarray:
        return identity_ramp()

    def build(self, mode: Mode) -> np.ndarray:
        return self._build_ramp_array(mode)

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

    def _smooth_param(self, name: str, target: float, alpha: float = 0.15) -> float:
        prev = self._adaptive_state.get(name, target)
        value = prev * (1.0 - alpha) + target * alpha
        self._adaptive_state[name] = value
        return value

    def _build_profile_params(self, mode: Mode) -> Dict[str, float]:
        suffix = mode.value.upper()
        p = {k.replace(f"_{suffix}", ""): float(self.config[k]) for k in self.config if k.endswith(suffix)}
        p["PROFILE"] = suffix
        return p

    def _apply_shadow_pop(self, p: Dict[str, float], pop: float) -> Dict[str, float]:
        if pop <= 0.0:
            return p
        p = dict(p)
        p["GAMMA"] *= (1.0 + 0.04 * pop)
        p["SHADOW_SIGMOID_BOOST"] += 0.15 * pop
        p["SHADOW_DESAT"] -= 0.06 * pop
        p["MIDTONE_BOOST"] *= 1.0 + (0.03 * pop)
        p["RED_MULTIPLIER"] *= 1.0 + (0.01 * pop)
        p["BLUE_MULTIPLIER"] *= 1.0 - (0.005 * pop)
        return p

    def _motion_shadow_emphasis_scale(self) -> float:
        c = self.config
        if not (c.get("MOTION_AWARE_SHADOWS", False) and c.get("MOTION_SHADOW_EMPHASIS", False)):
            return 0.0

        avg, shadow_density, _, motion = self._get_scene_metrics()

        if motion < float(c.get("MOTION_SHADOW_MIN_MOTION", 0.01)):
            return 0.0

        t = np.clip(motion * float(c.get("MOTION_SENSITIVITY", 2.5)), 0.0, 1.0)
        t *= np.clip(0.5 + shadow_density, 0.5, 1.0)

        dark_gate = float(np.clip(float(c.get("MOTION_SHADOW_DARK_LUMA", 0.5)), 0.35, 0.60))
        if avg > dark_gate:
            return 0.0

        strength = float(np.clip(float(c.get("MOTION_SHADOW_STRENGTH", 0.6)), 0.0, 0.85))
        return float(t * strength)

    def _apply_histogram_adaptation(self, p: Dict[str, float]) -> Dict[str, float]:
        c = self.config
        p = dict(p)

        if not c.get("HISTOGRAM_ADAPTIVE", False):
            self._adaptive_state.clear()
            return p

        avg, shadow_density, _edge, _motion = self._get_scene_metrics()

        lo = float(c.get("HISTOGRAM_MIN_LUMA", 0.12))
        hi = float(c.get("HISTOGRAM_MAX_LUMA", 0.55))
        strength = float(c.get("HISTOGRAM_STRENGTH", 0.30))

        profile = p.get("PROFILE")
        if profile == "OUTDOOR":
            strength *= 0.75
        elif profile == "INDOOR":
            strength *= 1.15

        if avg < lo:
            t = np.clip((lo - avg) / max(lo, 1e-6), 0.0, 1.0) * strength
            t *= (0.5 + shadow_density)
            p["SHADOW_CUTOFF"] -= 0.05 * t
            p["SHADOW_DESAT"] -= 0.20 * t
            p["SHADOW_SIGMOID_BOOST"] += 0.30 * t
        elif avg > hi:
            t = np.clip((avg - hi) / max(1.0 - hi, 1e-6), 0.0, 1.0) * strength
            p["SHADOW_SIGMOID_BOOST"] *= (1.0 - 0.5 * t)
            p["SHADOW_DESAT"] += 0.10 * t

        p["SHADOW_CUTOFF"] = float(np.clip(p["SHADOW_CUTOFF"], 0.15, 0.6))
        p["SHADOW_DESAT"] = float(np.clip(p["SHADOW_DESAT"], 0.5, 1.0))
        p["SHADOW_SIGMOID_BOOST"] = float(np.clip(p["SHADOW_SIGMOID_BOOST"], 0.0, 1.0))

        smooth = 0.2
        prof = p.get("PROFILE", "GLOBAL")
        p["SHADOW_CUTOFF"] = self._smooth_param(f"{prof}_SHADOW_CUTOFF", p["SHADOW_CUTOFF"], smooth)
        p["SHADOW_DESAT"] = self._smooth_param(f"{prof}_SHADOW_DESAT", p["SHADOW_DESAT"], smooth)
        p["SHADOW_SIGMOID_BOOST"] = self._smooth_param(f"{prof}_SHADOW_SIGMOID_BOOST", p["SHADOW_SIGMOID_BOOST"], smooth)

        return p

    def _edge_shadow_scale(self) -> float:
        c = self.config
        if not c.get("EDGE_AWARE_SHADOWS", False):
            return 1.0
        _avg, _shadow_density, edge, _motion = self._get_scene_metrics()
        mn = float(c.get("EDGE_MIN", 0.05))
        mx = float(c.get("EDGE_MAX", 0.35))
        strength = float(c.get("EDGE_STRENGTH", 0.40))
        t = np.clip((edge - mn) / max(mx - mn, 1e-4), 0.0, 1.0)
        return float(1.0 + t * strength)

    def _build_base_curve(self, p: Dict[str, float]) -> np.ndarray:
        base = _base_curve_cached(p["GAMMA"], p["GAMMA_OFFSET"]).copy()
        bf = float(np.clip(p["BLACK_FLOOR"], 0.0, 0.10))
        return bf + base * (1.0 - bf)

    def _apply_shadow_lift(self, base: np.ndarray, x: np.ndarray, cutoff: float, exp: float, edge_scale: float):
        transition = 0.08
        shadow = x < cutoff
        soft_shadow = (x >= cutoff) & (x < cutoff + transition)

        lift_scale = 1.0 + 0.5 * (edge_scale - 1.0)
        exp = float(np.clip(exp * lift_scale, 0.4, 1.2))

        lum = base.copy()
        chrom = np.divide(base, lum + 1e-5)

        lum[shadow] = np.power(lum[shadow] / cutoff, exp) * cutoff

        t = (x[soft_shadow] - cutoff) / transition
        lifted = np.power(lum[soft_shadow] / cutoff, exp) * cutoff
        lum[soft_shadow] = lum[soft_shadow] * (1.0 - t) + lifted * t

        base2 = lum * chrom
        return base2, shadow, soft_shadow

    def _apply_highlight_shaping(self, base: np.ndarray, x: np.ndarray, p: Dict[str, float], adaptive: bool):
        c = self.config
        shoulder_start = 0.75
        shoulder_end = 0.95
        mask = (x > shoulder_start) & (x < shoulder_end)
        t = (x[mask] - shoulder_start) / (shoulder_end - shoulder_start)
        t = t * t * (3 - 2 * t)
        base[mask] = shoulder_start + (base[mask] - shoulder_start) * (1.0 - 0.6 * t)

        hi = x > 0.85
        base[hi] = 0.85 + (base[hi] - 0.85) * p["HIGHLIGHT_COMPRESS"]

        if adaptive:
            avg, _shadow_density, _edge, _motion = self._get_scene_metrics()
            if avg > 0.6:
                hi_detail = (x > 0.82) & (x < 0.97)
                m = float(np.mean(base[hi_detail]))
                base[hi_detail] += (base[hi_detail] - m) * 0.035

        dx = np.gradient(base)
        hi = x > 0.85
        dx[hi] = np.maximum(dx[hi], 0.01)
        base[:] = np.cumsum(dx)
        base[:] = np.clip(base / base[-1], 0.0, 1.0)

        if adaptive:
            avg, shadow_density, _edge, _motion = self._get_scene_metrics()
            if avg > 0.65 and shadow_density < 0.35:
                t2 = np.clip((avg - 0.65) / 0.25, 0.0, 1.0)
                hi = x > 0.82
                base[hi] *= (1.0 - 0.12 * t2)

        if bool(c.get("PRESERVE_HUD_HIGHLIGHTS", True)):
            if bool(c.get("HUD_EXCLUSION", False)):
                hud_strength = float(c.get("HUD_EXCLUSION_STRENGTH", 0.6))
                thr = float(np.clip(float(c.get("HUD_EXCLUSION_THRESHOLD", 0.9)), 0.75, 0.98))
                hi_mask = base > thr
                base[hi_mask] = thr + (base[hi_mask] - thr) * hud_strength
            else:
                hi_mask = base > 0.9
                base[hi_mask] = np.clip(base[hi_mask], 0.9, 1.0)

        return base

    def _build_ramp_array(self, mode: Mode) -> np.ndarray:
        c = self.config
        p = self._build_profile_params(mode)

        adaptive = self.adaptive_enabled()
        motion_emphasis = self._motion_shadow_emphasis_scale() if adaptive else 0.0

        if adaptive:
            p = self._apply_histogram_adaptation(p)

        pop = float(np.clip(float(c.get("SHADOW_POP_STRENGTH", 0.0)), 0.0, 1.0))
        p = self._apply_shadow_pop(p, pop)

        x = X_AXIS
        base = self._build_base_curve(p)

        edge_scale = self._edge_shadow_scale() if adaptive else 1.0

        cutoff = float(np.clip(p["SHADOW_CUTOFF"], 0.15, 0.6))
        cutoff += 0.04 * motion_emphasis
        cutoff = float(np.clip(cutoff, 0.15, 0.6))

        base, shadow, soft_shadow = self._apply_shadow_lift(base, x, cutoff, p["SHADOW_LIFT_EXP"], edge_scale)

        min_contrast = 0.015
        dx = np.gradient(base)
        dx[shadow] = np.maximum(dx[shadow], min_contrast)
        base[:] = np.cumsum(dx)
        base[:] = np.clip(base / base[-1], 0.0, 1.0)

        toe_end = 0.08
        toe_strength = 0.04
        toe = x < toe_end
        ttoe = (toe_end - x[toe]) / toe_end
        base[toe] += toe_strength * (ttoe * ttoe) * dx[toe]
        if adaptive and motion_emphasis > 0.0:
            base[toe] += (0.025 * motion_emphasis)

        sig_strength = float(np.clip(p.get("SHADOW_SIGMOID_BOOST", 0.0), 0.0, 1.0))

        if adaptive:
            _avg, shadow_density, _edge, motion = self._get_scene_metrics()
            sig_strength *= float(np.clip(0.5 + shadow_density, 0.5, 1.0))

            if bool(c.get("MOTION_AWARE_SHADOWS", False)):
                sensitivity = float(c.get("MOTION_SENSITIVITY", 2.5))
                profile = p.get("PROFILE", "GLOBAL")
                strength = float(c.get(f"MOTION_STRENGTH_{profile}", 0.6))

                motion_t = float(np.clip(motion * sensitivity, 0.0, 1.0))
                sig_strength *= (1.0 + motion_t * strength)

                if motion_t > 0.0:
                    base += (base - float(np.mean(base))) * motion_t * 0.04
                    dx2 = np.gradient(base)
                    dx2 = np.maximum(dx2, 0.012)
                    base = np.cumsum(dx2)
                    base /= base[-1]

                if motion_emphasis > 0.0:
                    sig_strength *= (1.0 + 1.2 * motion_emphasis)

        sig_strength = float(np.clip(sig_strength, 0.0, 1.25))

        if sig_strength > 0.0:
            mid = MID_05_MASK & ~shadow & ~soft_shadow
            denom = max(0.5 - cutoff, 1e-5)
            tt = (x[mid] - cutoff) / denom
            sigmoid = 1.0 / (1.0 + np.exp(-8.0 * (tt - 0.5)))
            s = sig_strength * min(edge_scale, 1.3)
            base[mid] = base[mid] * (1.0 - s) + sigmoid * s

        ceiling = cutoff + 0.15
        mask = (x > cutoff) & (x < ceiling)
        t2 = (x[mask] - cutoff) / max(ceiling - cutoff, 1e-5)
        base[mask] *= 1.0 - 0.12 * (1.0 - t2)

        mid = MID_07_MASK & ~shadow & ~soft_shadow
        base[mid] *= p["MIDTONE_BOOST"]

        mc_strength = 0.025
        mid2 = (x > 0.35) & (x < 0.75)
        mmean = float(np.mean(base[mid2]))
        base[mid2] += (base[mid2] - mmean) * mc_strength
        base[mid2] *= 1.015

        base = self._apply_highlight_shaping(base, x, p, adaptive)

        r = base * p["RED_MULTIPLIER"]
        g = base * p["GREEN_MULTIPLIER"]
        b = base * p["BLUE_MULTIPLIER"]

        bias = float(np.clip(p["SHADOW_COLOR_BIAS"], -0.05, 0.05))
        r[shadow] *= 1.0 + bias
        b[shadow] *= 1.0 - bias

        r[shadow] *= p["SHADOW_RED_BIAS"]
        g[shadow] *= p["SHADOW_GREEN_BIAS"]
        b[shadow] *= p["SHADOW_BLUE_BIAS"]

        if bool(c.get("OPPONENT_TUNING", False)):
            opp = float(np.clip(float(c.get("OPPONENT_STRENGTH", 0.25)), 0.0, 1.0))

            sat_risk = max(p["VIBRANCE"] - 1.0, abs(p["SHADOW_COLOR_BIAS"]) * 4.0)
            if sat_risk > 0.0:
                opp *= float(np.clip(1.0 - sat_risk, 0.25, 1.0))

            lum_before = (r + g + b) / 3.0
            rg = r - g
            by = b - (r + g) * 0.5

            rg[shadow] *= (1.0 + opp)
            by[shadow] *= (1.0 + opp)

            r = g + rg
            b = (r + g) * 0.5 + by

            lum_after = (r + g + b) / 3.0
            scale = np.ones_like(lum_before)
            scale[shadow] = lum_before[shadow] / np.maximum(lum_after[shadow], 1e-4)

            r *= scale
            g *= scale
            b *= scale

            r[shadow] = np.clip(r[shadow], 0.0, 1.0)
            g[shadow] = np.clip(g[shadow], 0.0, 1.0)
            b[shadow] = np.clip(b[shadow], 0.0, 1.0)

        desat = float(p["SHADOW_DESAT"])
        if adaptive and motion_emphasis > 0.0:
            desat *= (1.0 - 0.35 * motion_emphasis)

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
        ramp = np.concatenate(
            (
                np.clip(r * scale, 0, 65535).astype(np.uint16),
                np.clip(g * scale, 0, 65535).astype(np.uint16),
                np.clip(b * scale, 0, 65535).astype(np.uint16),
            )
        )
        return ramp
