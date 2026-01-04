from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

# =========================
# Public API
# =========================

class Mode(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


# NOTE:
# Defaults mirror the monolith (visual_boost.py) so that a fresh refactor run
# produces identical behavior without requiring a pre-existing config file.
DEFAULT_CONFIG: Dict[str, object] = {
    # Global
    "GLOBAL_BRIGHTNESS": 1.0,
    "GLOBAL_BRIGHTNESS_ENABLED": False,
    "GLOBAL_BRIGHTNESS_INDOOR": 1.0,
    "GLOBAL_BRIGHTNESS_ENABLED_INDOOR": False,
    "GLOBAL_BRIGHTNESS_OUTDOOR": 1.0,
    "GLOBAL_BRIGHTNESS_ENABLED_OUTDOOR": False,
    "SHADOW_POP_STRENGTH": 0.20,
    "PRESERVE_HUD_HIGHLIGHTS": True,

    # ================= INDOOR =================
    "GAMMA_INDOOR": 1.28,
    "GAMMA_OFFSET_INDOOR": 0.0,
    "VIBRANCE_INDOOR": 1.04,
    "BLACK_FLOOR_INDOOR": 0.030,
    "SHADOW_LIFT_EXP_INDOOR": 0.62,
    "SHADOW_CUTOFF_INDOOR": 0.42,
    "SHADOW_DESAT_INDOOR": 0.82,
    "SHADOW_COLOR_BIAS_INDOOR": 0.0,
    "SHADOW_RED_BIAS_INDOOR": 1.0,
    "SHADOW_GREEN_BIAS_INDOOR": 1.0,
    "SHADOW_BLUE_BIAS_INDOOR": 1.0,
    "MIDTONE_BOOST_INDOOR": 1.02,
    "HIGHLIGHT_COMPRESS_INDOOR": 0.32,
    "RED_MULTIPLIER_INDOOR": 1.0,
    "GREEN_MULTIPLIER_INDOOR": 1.0,
    "BLUE_MULTIPLIER_INDOOR": 1.0,
    "SHADOW_SIGMOID_BOOST_INDOOR": 0.35,

    # ================= OUTDOOR =================
    "GAMMA_OUTDOOR": 1.12,
    "GAMMA_OFFSET_OUTDOOR": 0.0,
    "VIBRANCE_OUTDOOR": 1.03,
    "BLACK_FLOOR_OUTDOOR": 0.022,
    "SHADOW_LIFT_EXP_OUTDOOR": 0.75,
    "SHADOW_CUTOFF_OUTDOOR": 0.34,
    "SHADOW_DESAT_OUTDOOR": 0.90,
    "SHADOW_COLOR_BIAS_OUTDOOR": 0.0,
    "SHADOW_RED_BIAS_OUTDOOR": 1.0,
    "SHADOW_GREEN_BIAS_OUTDOOR": 1.0,
    "SHADOW_BLUE_BIAS_OUTDOOR": 1.0,
    "MIDTONE_BOOST_OUTDOOR": 1.0,
    "HIGHLIGHT_COMPRESS_OUTDOOR": 0.48,
    "RED_MULTIPLIER_OUTDOOR": 1.0,
    "GREEN_MULTIPLIER_OUTDOOR": 1.0,
    "BLUE_MULTIPLIER_OUTDOOR": 1.0,
    "SHADOW_SIGMOID_BOOST_OUTDOOR": 0.18,

    # ================= Adaptive defaults =================
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

# Shared LUT x-axis + masks
X_AXIS = np.linspace(0.0, 1.0, 256)
MID_05_MASK = X_AXIS < 0.5
MID_07_MASK = X_AXIS < 0.7
HI_085_MASK = X_AXIS > 0.85

# Luminance-weighted dithering noise
_DITHER_NOISE = (np.random.rand(256) - 0.5) * (1.0 / 65535.0)


def _profile(cfg: Dict[str, object], mode: Mode) -> Dict[str, float]:
    suffix = "_" + mode.value.upper()
    p: Dict[str, float] = {}

    # Profile-scoped values
    for k, v in cfg.items():
        if k.endswith(suffix):
            key = k.replace(suffix, "")
            p[key] = v


    # Global values
    for k in (
        "SHADOW_POP_STRENGTH",
        "PRESERVE_HUD_HIGHLIGHTS",
        "HISTOGRAM_ADAPTIVE",
        "HISTOGRAM_STRENGTH",
        "HISTOGRAM_MIN_LUMA",
        "HISTOGRAM_MAX_LUMA",
        "EDGE_AWARE_SHADOWS",
        "EDGE_STRENGTH",
        "EDGE_MIN",
        "EDGE_MAX",
        "HUD_EXCLUSION",
        "HUD_EXCLUSION_STRENGTH",
        "HUD_EXCLUSION_THRESHOLD",
        "MOTION_AWARE_SHADOWS",
        "MOTION_SENSITIVITY",
        "MOTION_SMOOTHING",
        "MOTION_SHADOW_EMPHASIS",
        "MOTION_SHADOW_STRENGTH",
        "MOTION_SHADOW_DARK_LUMA",
        "MOTION_SHADOW_MIN_MOTION",
        "OPPONENT_TUNING",
        "OPPONENT_STRENGTH",
        "MOTION_STRENGTH_INDOOR",
        "MOTION_STRENGTH_OUTDOOR",
    ):
        if k in cfg:
            p[k] = cfg[k]

    p["PROFILE"] = mode.value.upper()
    return p


@lru_cache(maxsize=128)
def _base_curve(gamma: float, offset: float) -> np.ndarray:
    g = float(gamma)
    o = float(offset)
    y = np.power(np.clip(X_AXIS + o, 0.0, 1.0), 1.0 / g)
    return y.astype(np.float64)


class GammaPipeline:
    __slots__ = ("config", "scene_analyzer", "_adaptive_state")

    def __init__(self, config: Dict[str, object], scene_analyzer=None):
        self.config = config
        self.scene_analyzer = scene_analyzer
        self._adaptive_state: Dict[str, float] = {}

    @staticmethod
    def identity_ramp() -> np.ndarray:
        ramp = np.linspace(0, 65535, 256, dtype=np.uint16)
        return np.concatenate((ramp, ramp, ramp))

    def build(self, mode: Optional[Mode]) -> np.ndarray:
        if mode is None:
            return self.identity_ramp()

        profile = _profile(self.config, mode)
        return self._build_ramp_array(profile)

    def _get_scene_metrics(self) -> Tuple[float, float, float, float]:
        if self.scene_analyzer is None:
            return (0.5, 0.0, 0.0, 0.0)
        try:
            return self.scene_analyzer.get_metrics()
        except Exception:
            return (0.5, 0.0, 0.0, 0.0)

    def _adaptive_enabled(self) -> bool:
        cfg = self.config
        return bool(
            cfg.get("HISTOGRAM_ADAPTIVE", False)
            or cfg.get("EDGE_AWARE_SHADOWS", False)
            or cfg.get("OPPONENT_TUNING", False)
            or cfg.get("HUD_EXCLUSION", False)
            or cfg.get("MOTION_AWARE_SHADOWS", False)
            or cfg.get("MOTION_SHADOW_EMPHASIS", False)
        )

    def _smooth_param(self, name: str, target: float, alpha: float = 0.15) -> float:
        prev = self._adaptive_state.get(name, target)
        value = prev * (1.0 - alpha) + target * alpha
        self._adaptive_state[name] = value
        return value

    # ───────────────────────── Monolith-parity transforms ─────────────────────────

    def _motion_shadow_emphasis_scale(self) -> float:
        cfg = self.config
        if not (cfg.get("MOTION_AWARE_SHADOWS", False) and cfg.get("MOTION_SHADOW_EMPHASIS", False)):
            return 0.0

        avg, shadow_density, _, motion = self._get_scene_metrics()

        if motion < float(cfg.get("MOTION_SHADOW_MIN_MOTION", 0.01)):
            return 0.0

        t = np.clip(motion * float(cfg.get("MOTION_SENSITIVITY", 2.5)), 0.0, 1.0)
        t *= np.clip(0.5 + shadow_density, 0.5, 1.0)

        dark_gate = np.clip(float(cfg.get("MOTION_SHADOW_DARK_LUMA", 0.5)), 0.35, 0.60)
        if avg > dark_gate:
            return 0.0

        strength = np.clip(float(cfg.get("MOTION_SHADOW_STRENGTH", 0.6)), 0.0, 0.85)
        return float(t * strength)

    def _apply_histogram_adaptation(self, p: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        p = dict(p)

        if not cfg.get("HISTOGRAM_ADAPTIVE", False):
            self._adaptive_state.clear()
            return p

        avg, shadow_density, _, _ = self._get_scene_metrics()

        lo = float(cfg.get("HISTOGRAM_MIN_LUMA", 0.12))
        hi = float(cfg.get("HISTOGRAM_MAX_LUMA", 0.55))
        strength = float(cfg.get("HISTOGRAM_STRENGTH", 0.30))

        profile = p.get("PROFILE")
        if profile == "OUTDOOR":
            strength *= 0.75
        elif profile == "INDOOR":
            strength *= 1.15

        if avg < lo:
            t = np.clip((lo - avg) / max(lo, 1e-4), 0.0, 1.0) * strength
            t *= (0.5 + shadow_density)

            p["SHADOW_CUTOFF"] = p.get("SHADOW_CUTOFF", 0.35) - 0.05 * t
            p["SHADOW_DESAT"] = p.get("SHADOW_DESAT", 0.9) - 0.20 * t
            p["SHADOW_SIGMOID_BOOST"] = p.get("SHADOW_SIGMOID_BOOST", 0.2) + 0.30 * t

        elif avg > hi:
            t = np.clip((avg - hi) / max(1.0 - hi, 1e-4), 0.0, 1.0) * strength
            p["SHADOW_SIGMOID_BOOST"] = p.get("SHADOW_SIGMOID_BOOST", 0.2) * (1.0 - 0.5 * t)
            p["SHADOW_DESAT"] = p.get("SHADOW_DESAT", 0.9) + 0.10 * t

        p["SHADOW_CUTOFF"] = float(np.clip(p.get("SHADOW_CUTOFF", 0.35), 0.15, 0.6))
        p["SHADOW_DESAT"] = float(np.clip(p.get("SHADOW_DESAT", 0.9), 0.5, 1.0))
        p["SHADOW_SIGMOID_BOOST"] = float(np.clip(p.get("SHADOW_SIGMOID_BOOST", 0.2), 0.0, 1.0))

        smooth = 0.2
        prof = p.get("PROFILE", "GLOBAL")
        p["SHADOW_CUTOFF"] = self._smooth_param(f"{prof}_SHADOW_CUTOFF", p["SHADOW_CUTOFF"], smooth)
        p["SHADOW_DESAT"] = self._smooth_param(f"{prof}_SHADOW_DESAT", p["SHADOW_DESAT"], smooth)
        p["SHADOW_SIGMOID_BOOST"] = self._smooth_param(f"{prof}_SHADOW_SIGMOID_BOOST", p["SHADOW_SIGMOID_BOOST"], smooth)

        return p

    def _edge_shadow_scale(self) -> float:
        cfg = self.config
        if not cfg.get("EDGE_AWARE_SHADOWS", False):
            return 1.0

        _, _, edge, _ = self._get_scene_metrics()
        mn = float(cfg.get("EDGE_MIN", 0.05))
        mx = float(cfg.get("EDGE_MAX", 0.35))
        strength = float(cfg.get("EDGE_STRENGTH", 0.40))

        t = np.clip((edge - mn) / max(mx - mn, 1e-4), 0.0, 1.0)
        return float(1.0 + t * strength)

    @staticmethod
    def _apply_shadow_pop(p: Dict[str, float], pop: float) -> Dict[str, float]:
        if pop <= 0.0:
            return p

        p = dict(p)
        p["GAMMA"] *= (1.0 + 0.04 * pop)
        p["SHADOW_SIGMOID_BOOST"] = p.get("SHADOW_SIGMOID_BOOST", 0.0) + 0.15 * pop
        p["SHADOW_DESAT"] = p.get("SHADOW_DESAT", 1.0) - 0.06 * pop
        p["MIDTONE_BOOST"] *= 1.0 + (0.03 * pop)
        p["RED_MULTIPLIER"] *= 1.0 + (0.01 * pop)
        p["BLUE_MULTIPLIER"] *= 1.0 - (0.005 * pop)
        return p

    @staticmethod
    def _build_base_curve(p: Dict[str, float]) -> np.ndarray:
        base = _base_curve(float(p.get("GAMMA", 1.0)), float(p.get("GAMMA_OFFSET", 0.0))).copy()
        bf = np.clip(float(p.get("BLACK_FLOOR", 0.0)), 0.0, 0.10)
        return bf + base * (1.0 - bf)

    @staticmethod
    def _apply_shadow_lift(base: np.ndarray, x: np.ndarray, cutoff: float, exp: float, edge_scale: float):
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

        base = lum * chrom
        return base, shadow, soft_shadow

    def _apply_highlight_shaping(self, base: np.ndarray, x: np.ndarray, p: Dict[str, float], adaptive: bool) -> np.ndarray:
        shoulder_start = 0.75
        shoulder_end = 0.95

        mask = (x > shoulder_start) & (x < shoulder_end)
        t = (x[mask] - shoulder_start) / (shoulder_end - shoulder_start)
        t = t * t * (3 - 2 * t)
        base[mask] = shoulder_start + (base[mask] - shoulder_start) * (1.0 - 0.6 * t)

        hi = x > 0.85
        base[hi] = 0.85 + (base[hi] - 0.85) * float(p.get("HIGHLIGHT_COMPRESS", 0.5))

        if adaptive:
            avg, _, _, _ = self._get_scene_metrics()
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
            avg, shadow_density, _, _ = self._get_scene_metrics()
            if avg > 0.65 and shadow_density < 0.35:
                tt = np.clip((avg - 0.65) / 0.25, 0.0, 1.0)
                hi = x > 0.82
                base[hi] *= (1.0 - 0.12 * tt)

        cfg = self.config
        if cfg.get("PRESERVE_HUD_HIGHLIGHTS", True):
            if cfg.get("HUD_EXCLUSION", False):
                hud_strength = float(cfg.get("HUD_EXCLUSION_STRENGTH", 0.6))
                thr = float(np.clip(float(cfg.get("HUD_EXCLUSION_THRESHOLD", 0.9)), 0.75, 0.98))
                hi_mask = base > thr
                base[hi_mask] = thr + (base[hi_mask] - thr) * hud_strength
            else:
                hi_mask = base > 0.9
                base[hi_mask] = np.clip(base[hi_mask], 0.9, 1.0)

        return base

    # ───────────────────────── Full LUT build (monolith parity) ─────────────────────────

    def _build_ramp_array(self, p: Dict[str, float]) -> np.ndarray:
        cfg = self.config

        adaptive = self._adaptive_enabled()
        motion_emphasis = self._motion_shadow_emphasis_scale() if adaptive else 0.0

        if adaptive:
            p = self._apply_histogram_adaptation(p)

        p = self._apply_shadow_pop(p, float(np.clip(cfg.get("SHADOW_POP_STRENGTH", 0.0), 0.0, 1.0)))

        x = X_AXIS
        base = self._build_base_curve(p)

        edge_scale = self._edge_shadow_scale() if adaptive else 1.0

        cutoff = float(np.clip(p.get("SHADOW_CUTOFF", 0.35), 0.15, 0.6))
        # Parity with monolith: expand shadow cutoff based on motion emphasis
        if adaptive and motion_emphasis > 0.0:
            cutoff += 0.09 * motion_emphasis
            cutoff = float(np.clip(cutoff, 0.15, 0.6))
        base, shadow, soft_shadow = self._apply_shadow_lift(
            base, x, cutoff, float(p.get("SHADOW_LIFT_EXP", 0.7)), edge_scale
        )

        # Minimum shadow contrast floor
        min_contrast = 0.015
        dx = np.gradient(base)
        dx[shadow] = np.maximum(dx[shadow], min_contrast)
        base[:] = np.cumsum(dx)
        base[:] = np.clip(base / base[-1], 0.0, 1.0)

        # Deep shadow toe lift
        toe_end = 0.08
        toe_strength = 0.04
        toe = x < toe_end
        ttoe = (toe_end - x[toe]) / toe_end
        base[toe] += toe_strength * (ttoe * ttoe) * dx[toe]

        if adaptive and motion_emphasis > 0.0:
            base[toe] += (0.04 * motion_emphasis)
            #base[toe] += (0.025 * motion_emphasis)

        # Sigmoid contrast shaping (mid-shadow)
        sig_strength = float(np.clip(p.get("SHADOW_SIGMOID_BOOST", 0.0), 0.0, 1.0))
        # Edge-aware sigmoid scaling (monolith parity)
        # edge scaling deferred to blend (monolith parity)

        if adaptive:
            _, shadow_density, _, motion = self._get_scene_metrics()

            sig_strength *= float(np.clip(0.5 + shadow_density, 0.5, 1.0))

            if cfg.get("MOTION_AWARE_SHADOWS", False):
                sensitivity = float(cfg.get("MOTION_SENSITIVITY", 2.5))
                profile = p.get("PROFILE", "GLOBAL")
                strength = float(cfg.get(f"MOTION_STRENGTH_{profile}", 0.6))

                motion_t = float(np.clip(motion * sensitivity, 0.0, 1.0))
                sig_strength *= (1.0 + motion_t * strength)

                if motion_t > 0.0:
                    base += (base - float(np.mean(base))) * motion_t * (0.06 + 0.08 * motion_emphasis)
                    #base += (base - float(np.mean(base))) * motion_t * 0.09

                    dx2 = np.gradient(base)
                    dx2 = np.maximum(dx2, 0.012)
                    base = np.cumsum(dx2)
                    base /= base[-1]

                if motion_emphasis > 0.0:
                    sig_strength *= (1.0 + 2.8 * motion_emphasis)
                    #sig_strength *= (1.0 + 2.0 * motion_emphasis)

        sig_strength = float(np.clip(sig_strength, 0.0, 1.25))

        if sig_strength > 0.0:
            mid = MID_05_MASK & ~shadow & ~soft_shadow
            t = np.clip((x[mid] - cutoff) / max(0.5 - cutoff, 1e-4), 0.0, 1.0)
            s = 1.0 / (1.0 + np.exp(-8.0 * (t - 0.5)))
            eff_strength = sig_strength * min(edge_scale, 1.3)
            base[mid] = base[mid] * (1.0 - eff_strength) + s * eff_strength

        # Shadow ceiling compression
        ceiling = cutoff + 0.15
        mask = (x > cutoff) & (x < ceiling)
        tt = (x[mask] - cutoff) / (ceiling - cutoff)
        base[mask] *= 1.0 - 0.12 * (1.0 - tt)

        # Midtone boost
        mid = MID_07_MASK & ~shadow & ~soft_shadow
        base[mid] *= float(p.get("MIDTONE_BOOST", 1.0))

        # Subtle midtone micro-contrast
        mc_strength = 0.025
        mid_mc = (x > 0.35) & (x < 0.75)
        m = float(np.mean(base[mid_mc]))
        base[mid_mc] += (base[mid_mc] - m) * mc_strength
        base[mid_mc] *= 1.015

        base = self._apply_highlight_shaping(base, x, p, adaptive)

        # Channel multipliers
        r = base * float(p.get("RED_MULTIPLIER", 1.0))
        g = base * float(p.get("GREEN_MULTIPLIER", 1.0))
        b = base * float(p.get("BLUE_MULTIPLIER", 1.0))

        # Shadow-only color bias
        bias = float(np.clip(p.get("SHADOW_COLOR_BIAS", 0.0), -0.05, 0.05))
        r[shadow] *= 1.0 + bias
        b[shadow] *= 1.0 - bias

        # Shadow channel biases
        r[shadow] *= float(p.get("SHADOW_RED_BIAS", 1.0))
        g[shadow] *= float(p.get("SHADOW_GREEN_BIAS", 1.0))
        b[shadow] *= float(p.get("SHADOW_BLUE_BIAS", 1.0))

        # Opponent-channel shadow tuning (optional)
        if adaptive and cfg.get("OPPONENT_TUNING", False):
            opp = float(np.clip(cfg.get("OPPONENT_STRENGTH", 0.25), 0.0, 1.0))

            sat_risk = max(
                float(p.get("VIBRANCE", 1.0)) - 1.0,
                abs(float(p.get("SHADOW_COLOR_BIAS", 0.0))) * 4.0
            )
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

        # Shadow desaturation
        desat = float(np.clip(p.get("SHADOW_DESAT", 1.0), 0.5, 1.0))
        if adaptive and motion_emphasis > 0.0:
            desat *= (1.0 - 0.35 * motion_emphasis)

        if desat < 1.0:
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            r[shadow] = lum[shadow] + (r[shadow] - lum[shadow]) * desat
            g[shadow] = lum[shadow] + (g[shadow] - lum[shadow]) * desat
            b[shadow] = lum[shadow] + (b[shadow] - lum[shadow]) * desat

        # Luminance-weighted dithering
        dither = _DITHER_NOISE * (0.5 + 0.5 * base)
        r = np.clip(r + dither, 0.0, 1.0)
        g = np.clip(g + dither, 0.0, 1.0)
        b = np.clip(b + dither, 0.0, 1.0)

        # Exposure (global brightness) – folded into final scale for monolith parity
        if p.get("GLOBAL_BRIGHTNESS_ENABLED", False):
            exposure = float(p.get("GLOBAL_BRIGHTNESS", 1.0))
        else:
            exposure = 1.0


        EXPOSURE_COMP = 1.03  # monolith perceptual parity
        scale = 65535.0 * float(p.get("VIBRANCE", 1.0)) * exposure * EXPOSURE_COMP

        return np.concatenate([
            np.clip(r * scale, 0, 65535).astype(np.uint16),
            np.clip(g * scale, 0, 65535).astype(np.uint16),
            np.clip(b * scale, 0, 65535).astype(np.uint16),
        ])
