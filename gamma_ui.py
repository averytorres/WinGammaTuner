from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional


def _decimals(step: float) -> int:
    if step <= 0:
        return 0
    return max(0, int(-math.log10(step))) if step < 1 else 0


class CanvasSlider(tk.Canvas):
    def __init__(self, parent, mn, mx, value, command, step, *, bg):
        super().__init__(parent, width=240, height=20, bg=bg, highlightthickness=0)
        self.min = float(mn)
        self.max = float(mx)
        self.step = float(step)
        self.command = command
        self.usable = 228
        self.value = float(value)

        self.create_rectangle(0, 9, 240, 11, fill="#333", outline="")
        self.thumb = self.create_oval(0, 4, 12, 16, fill="#007acc", outline="")

        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)

        self.set_value(self.value, notify=False)

    def _quantize(self, v: float) -> float:
        return round(v / self.step) * self.step if self.step > 0 else v

    def set_value(self, v: float, *, notify=True, dragging=False):
        v = max(self.min, min(self.max, float(v)))
        self.value = self._quantize(v)
        span = self.max - self.min
        x = int((self.value - self.min) / span * self.usable) if span else 0
        self.coords(self.thumb, x, 4, x + 12, 16)
        if notify:
            self.command(self.value, dragging)

    def _press(self, e):
        # Ensure widget has focus so modifier state (Ctrl/Shift) is reported reliably
        try:
            self.focus_set()
        except Exception:
            pass
        self._drag(e)

    def _drag(self, e):
        t = max(0.0, min(1.0, e.x / self.usable))
        target = self.min + t * (self.max - self.min)
        span = self.max - self.min

        # Precision modifiers (monolith parity)
        if e.state & 0x0004:  # Ctrl
            target = self.value + (target - self.value) * min(0.05, 0.01 / span)
        elif e.state & 0x0001:  # Shift
            target = self.value + (target - self.value) * min(0.2, 0.04 / span)

        self.set_value(target, notify=True, dragging=True)

    def _release(self, _):
        self.command(self.value, False)


class GammaUI:
    def __init__(self, root, controller, config: Dict[str, object], defaults: Dict[str, object], config_mgr):
        self.root = root
        self.controller = controller
        self.config = config
        self.defaults = defaults
        self.config_mgr = config_mgr

        self.bg = "#1e1e1e"
        self.fg = "#ffffff"
        self.btn = "#2d2d2d"

        self.window: Optional[tk.Toplevel] = None
        self.frames: Dict[str, tk.Frame] = {}

    # ───────── lifecycle ─────────

    def toggle(self):
        if self.window and self.window.winfo_exists():
            if self.window.state() == "withdrawn":
                self.window.deiconify()
            else:
                self.window.withdraw()
        else:
            self.show()

    def show(self):
        if self.window and self.window.winfo_exists():
            self.window.deiconify()
            return

        self.window = tk.Toplevel(self.root)
        self.window.title("Gamma Control")
        self.window.geometry("780x720")
        self.window.configure(bg=self.bg)
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)

        nb = ttk.Notebook(self.window)
        nb.pack(fill=tk.BOTH, expand=True)

        self.frames.clear()
        for name in ("Profile", "Adaptive", "Advanced", "Global"):
            tab = tk.Frame(nb, bg=self.bg)
            nb.add(tab, text=name)
            self.frames[name] = self._make_scrollable(tab)

        self.rebuild()

    # ───────── helpers ─────────

    def _pk(self, key: str) -> str:
        return f"{key}_{self.controller.current_mode.value.upper()}"

    def _get(self, key: str, fallback):
        return self.config.get(key, self.defaults.get(key, fallback))

    def _make_scrollable(self, parent):
        canvas = tk.Canvas(parent, bg=self.bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg=self.bg)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        frame.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        self.root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"), add="+")

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return frame

    # ───────── widgets ─────────

    def _slider(self, parent, label, key, mn, mx, step):
        row = tk.Frame(parent, bg=self.bg)
        row.pack(fill=tk.X, padx=8, pady=5)

        tk.Label(row, text=label, bg=self.bg, fg=self.fg, width=24, anchor="w").pack(side=tk.LEFT)

        dec = _decimals(step)
        init = float(self._get(key, mn))
        val = tk.StringVar(value=f"{init:.{dec}f}")

        def on_change(v, dragging):
            self.config[key] = float(v)
            val.set(f"{v:.{dec}f}")
            value_label.config(fg="#ffffff" if dragging else "#aaa")
            self.controller.rebuild_debounced()
            if not dragging:
                self.config_mgr.debounce_save()

        slider = CanvasSlider(row, mn, mx, init, on_change, step, bg=self.bg)
        slider.pack(side=tk.LEFT, padx=6)

        value_label = tk.Label(row, textvariable=val, bg=self.bg, fg="#aaa", width=8)
        value_label.pack(side=tk.LEFT)

        tk.Button(
            row, text="Reset", bg=self.btn, fg=self.fg, width=6,
            command=lambda: slider.set_value(self.defaults.get(key, init))
        ).pack(side=tk.LEFT)

    def _toggle(self, parent, label, key, *, rebuild_ui=False):
        var = tk.BooleanVar(value=bool(self._get(key, False)))

        def on_toggle():
            self.config[key] = bool(var.get())
            self.controller.rebuild_debounced()
            self.config_mgr.debounce_save()
            if rebuild_ui:
                self.rebuild()

        tk.Checkbutton(parent, text=label, variable=var, command=on_toggle,
                       bg=self.bg, fg=self.fg, selectcolor=self.bg).pack(anchor="w", padx=8, pady=4)
        return var.get()

    # ───────── rebuild ─────────

    def rebuild(self):
        if not self.window or not self.window.winfo_exists():
            return
        self.root.after(0, self._rebuild_impl)

    def _rebuild_impl(self):
        for f in self.frames.values():
            for w in f.winfo_children():
                w.destroy()

        if self.controller.current_mode is None:
            for f in self.frames.values():
                tk.Label(
                    f,
                    text="Gamma OFF\nF8 / F9",
                    fg="red",
                    bg=self.bg
                ).pack(pady=20)
            return

        m = self.controller.current_mode.value.upper()

        # ── Profile ──
        p = self.frames["Profile"]
        self._slider(p, "Gamma", self._pk("GAMMA"), 0.75, 1.4, 0.001)
        self._slider(p, "Gamma Offset", self._pk("GAMMA_OFFSET"), -0.2, 0.2, 0.001)
        self._slider(p, "Vibrance", self._pk("VIBRANCE"), 0.9, 1.3, 0.001)
        self._slider(p, "Black Floor", self._pk("BLACK_FLOOR"), 0.0, 0.1, 0.0005)
        self._slider(p, "Shadow Lift Exp", self._pk("SHADOW_LIFT_EXP"), 0.4, 1.0, 0.001)
        self._slider(p, "Shadow Cutoff", self._pk("SHADOW_CUTOFF"), 0.15, 0.6, 0.001)
        self._slider(p, "Shadow Desat", self._pk("SHADOW_DESAT"), 0.5, 1.0, 0.001)
        self._slider(p, "Shadow Sigmoid Boost", self._pk("SHADOW_SIGMOID_BOOST"), 0.0, 1.0, 0.01)
        self._slider(p, "Shadow Color Bias", self._pk("SHADOW_COLOR_BIAS"), -0.05, 0.05, 0.001)
        self._slider(p, "Midtone Boost", self._pk("MIDTONE_BOOST"), 0.9, 1.3, 0.001)
        self._slider(p, "Highlight Compress", self._pk("HIGHLIGHT_COMPRESS"), 0.0, 0.7, 0.001)

        # ── Adaptive ──
        a = self.frames["Adaptive"]
        if self._toggle(a, "Histogram Adaptive", "HISTOGRAM_ADAPTIVE", rebuild_ui=True):
            self._slider(a, "Histogram Strength", "HISTOGRAM_STRENGTH", 0.0, 1.0, 0.01)
            self._slider(a, "Min Scene Luma", "HISTOGRAM_MIN_LUMA", 0.1, 0.5, 0.01)
            self._slider(a, "Max Scene Luma", "HISTOGRAM_MAX_LUMA", 0.4, 0.9, 0.01)

        if self._toggle(a, "Motion Aware Shadows", "MOTION_AWARE_SHADOWS", rebuild_ui=True):
            self._slider(a, f"Motion Strength ({m})", f"MOTION_STRENGTH_{m}", 0.0, 1.0, 0.01)
            self._slider(a, "Motion Sensitivity", "MOTION_SENSITIVITY", 0.5, 5.0, 0.1)

            if self._toggle(a, "Motion Shadow Emphasis", "MOTION_SHADOW_EMPHASIS", rebuild_ui=True):
                self._slider(a, "Emphasis Strength", "MOTION_SHADOW_STRENGTH", 0.0, 1.0, 0.01)
                self._slider(a, "Dark Luma Gate", "MOTION_SHADOW_DARK_LUMA", 0.35, 0.6, 0.01)
                self._slider(a, "Min Motion Threshold", "MOTION_SHADOW_MIN_MOTION", 0.0, 0.05, 0.001)

        if self._toggle(a, "Edge Aware Shadows", "EDGE_AWARE_SHADOWS", rebuild_ui=True):
            self._slider(a, "Edge Strength", "EDGE_STRENGTH", 0.0, 1.0, 0.01)
            self._slider(a, "Edge Min", "EDGE_MIN", 0.0, 0.5, 0.01)
            self._slider(a, "Edge Max", "EDGE_MAX", 0.3, 1.0, 0.01)

        if self._toggle(a, "HUD Highlight Exclusion", "HUD_EXCLUSION", rebuild_ui=True):
            self._slider(a, "HUD Exclusion Strength", "HUD_EXCLUSION_STRENGTH", 0.0, 1.0, 0.01)
            self._slider(a, "HUD Threshold", "HUD_EXCLUSION_THRESHOLD", 0.75, 0.98, 0.01)

        # ── Advanced ──
        adv = self.frames["Advanced"]
        self._slider(adv, "Shadow Red Bias", self._pk("SHADOW_RED_BIAS"), 0.95, 1.05, 0.001)
        self._slider(adv, "Shadow Green Bias", self._pk("SHADOW_GREEN_BIAS"), 0.95, 1.05, 0.001)
        self._slider(adv, "Shadow Blue Bias", self._pk("SHADOW_BLUE_BIAS"), 0.95, 1.05, 0.001)
        self._toggle(adv, "Opponent Tuning", "OPPONENT_TUNING", rebuild_ui=True)
        self._slider(adv, "Opponent Strength", "OPPONENT_STRENGTH", 0.0, 1.0, 0.01)

        # ── Global ──
        g = self.frames["Global"]
        self._slider(g, "Shadow Pop Strength", "SHADOW_POP_STRENGTH", 0.0, 1.0, 0.01)
        self._toggle(g, "Preserve HUD Highlights", "PRESERVE_HUD_HIGHLIGHTS")
        self._toggle(g, "Global Brightness Enable", self._pk("GLOBAL_BRIGHTNESS_ENABLED"))
        self._slider(g, "Global Brightness", self._pk("GLOBAL_BRIGHTNESS"), 0.5, 1.5, 0.01)

