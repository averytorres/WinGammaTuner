"""
gamma_ui.py

Tkinter UI for live tuning.
Restores the dark themed CanvasSlider, collapsible sections, tabs, and reset behavior.

Public API:
- GammaUI(root, controller, config, defaults, config_mgr)
    - show() / toggle() / hide()
    - rebuild()  # safe to call from controller/main (uses root.after)
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional


def _decimals(step: float) -> int:
    return max(0, int(-math.log10(step))) if step < 1 else 0


class CanvasSlider(tk.Canvas):
    def __init__(self, parent, mn, mx, value, command, step, *, bg, fg_thumb="#007acc"):
        super().__init__(parent, width=240, height=20, bg=bg, highlightthickness=0)
        self.min = mn
        self.max = mx
        self.command = command
        self.step = step
        self.usable = 228
        self.value = value

        self.create_rectangle(0, 9, 240, 11, fill="#333", outline="")
        self.thumb = self.create_oval(0, 4, 12, 16, fill=fg_thumb, outline="")

        self.bind("<ButtonPress-1>", self._drag)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)

        self.set_value(value, notify=False)

    def _q(self, v: float) -> float:
        return round(v / self.step) * self.step

    def set_value(self, v: float, notify: bool = True):
        self.value = float(self._q(max(self.min, min(self.max, v))))
        x = int((self.value - self.min) / (self.max - self.min) * self.usable) if self.max != self.min else 0
        self.coords(self.thumb, x, 4, x + 12, 16)
        if notify:
            self.command(self.value, False)

    def _drag(self, e):
        t = max(0.0, min(1.0, e.x / self.usable))
        target = self.min + t * (self.max - self.min)
        span = self.max - self.min

        if e.state & 0x0004:  # Ctrl
            target = self.value + (target - self.value) * min(0.05, 0.01 / max(span, 1e-9))
        elif e.state & 0x0001:  # Shift
            target = self.value + (target - self.value) * min(0.2, 0.04 / max(span, 1e-9))

        self.value = float(self._q(target))
        self.command(self.value, True)
        self.set_value(self.value, notify=False)

    def _release(self, _e):
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

        self._slider_widgets = {}
        self._value_labels = {}

    def show(self):
        if self.window and self.window.winfo_exists():
            self.window.deiconify()
            self.window.lift()
            return

        self.window = tk.Toplevel(self.root)
        self.window.title("Gamma Control")
        self.window.geometry("640x600")
        self.window.configure(bg=self.bg)
        self.window.protocol("WM_DELETE_WINDOW", self.hide)

        nb = ttk.Notebook(self.window)
        nb.pack(fill=tk.BOTH, expand=True)

        for name in ("Profile", "Adaptive", "Advanced", "Global"):
            tab = tk.Frame(nb, bg=self.bg)
            nb.add(tab, text=name)
            self.frames[name] = self._make_scrollable(tab)

        self.rebuild()

    def hide(self):
        if self.window and self.window.winfo_exists():
            self.window.withdraw()

    def toggle(self):
        if self.window and self.window.winfo_exists() and self.window.state() != "withdrawn":
            self.hide()
        else:
            self.show()

    def rebuild(self):
        self.root.after(0, self._rebuild_impl)

    def _rebuild_impl(self):
        if not self.window or not self.window.winfo_exists():
            return

        for _, frame in self.frames.items():
            for w in frame.winfo_children():
                w.destroy()

        self._slider_widgets.clear()
        self._value_labels.clear()

        if self.controller.current_mode is None:
            prof = self.frames["Profile"]
            tk.Label(
                prof,
                text="Gamma OFF\nF8 / F9",
                bg=self.bg,
                fg="red",
                font=("Segoe UI", 12, "bold"),
            ).pack(pady=20)
            return

        self._build_profile_tab(self.frames["Profile"])
        self._build_adaptive_tab(self.frames["Adaptive"])
        self._build_advanced_tab(self.frames["Advanced"])
        self._build_global_tab(self.frames["Global"])

    def _make_scrollable(self, parent) -> tk.Frame:
        canvas = tk.Canvas(parent, bg=self.bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg=self.bg)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        def _on_configure(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        frame.bind("<Configure>", _on_configure)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return frame

    def _section(self, parent, title: str, start_open: bool = True) -> tk.Frame:
        container = tk.Frame(parent, bg=self.bg)

        header = tk.Frame(container, bg=self.bg)
        header.pack(fill=tk.X, padx=6, pady=(10, 4))

        arrow = tk.StringVar(value="▼" if start_open else "▶")
        arrow_lbl = tk.Label(header, textvariable=arrow, bg=self.bg, fg="#888",
                             font=("Segoe UI", 10, "bold"), width=2)
        arrow_lbl.pack(side=tk.LEFT)

        title_lbl = tk.Label(header, text=title, bg=self.bg, fg="#bbb",
                             font=("Segoe UI", 10, "bold"), cursor="hand2")
        title_lbl.pack(side=tk.LEFT)

        body = tk.Frame(container, bg=self.bg)
        visible = tk.BooleanVar(value=start_open)

        def toggle():
            is_open = not visible.get()
            visible.set(is_open)
            arrow.set("▼" if is_open else "▶")
            if is_open:
                body.pack(fill=tk.X)
            else:
                body.pack_forget()

        def on_enter(_e):
            title_lbl.configure(fg="#ffffff")
            arrow_lbl.configure(fg="#ffffff")

        def on_leave(_e):
            title_lbl.configure(fg="#bbb")
            arrow_lbl.configure(fg="#888")

        for w in (header, arrow_lbl, title_lbl):
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<Button-1>", lambda _e: toggle())

        if start_open:
            body.pack(fill=tk.X)

        container.pack(fill=tk.X)
        return body

    def _profile_key(self, key: str) -> str:
        m = self.controller.current_mode.value.upper()
        return f"{key}_{m}"

    def _draw_slider(self, parent, label: str, key: str, mn: float, mx: float, step: float):
        row = tk.Frame(parent, bg=self.bg)
        row.pack(fill=tk.X, padx=6, pady=4)

        tk.Label(row, text=label, bg=self.bg, fg=self.fg, width=22, anchor="w").pack(side=tk.LEFT)

        val_var = tk.StringVar()
        dec = _decimals(step)

        val_lbl = tk.Label(row, textvariable=val_var, bg=self.bg, fg="#aaa", width=8)
        val_lbl.pack(side=tk.LEFT)

        def cb(v, dragging: bool):
            self.config[key] = float(v)
            self.controller.rebuild_debounced()
            self.config_mgr.debounce_save()
            val_var.set(f"{float(v):.{dec}f}")
            val_lbl.configure(fg="#777" if dragging else "#aaa")

        cs = CanvasSlider(
            row,
            mn, mx,
            float(self.config.get(key, self.defaults.get(key, mn))),
            cb,
            step,
            bg=self.bg,
        )
        cs.pack(side=tk.LEFT, padx=6)

        val_var.set(f"{float(self.config.get(key, mn)):.{dec}f}")

        def do_reset():
            cs.set_value(float(self.defaults[key]), notify=True)

        tk.Button(
            row,
            text="Reset",
            bg=self.btn,
            fg=self.fg,
            width=6,
            command=do_reset,
        ).pack(side=tk.LEFT, padx=(6, 0))

        self._slider_widgets[key] = cs
        self._value_labels[key] = val_var

    def _draw_toggle(self, parent, label: str, key: str, *, indent: int = 6, rebuild: bool = True) -> tk.BooleanVar:
        var = tk.BooleanVar(value=bool(self.config.get(key, False)))

        def on_toggle():
            self.config[key] = bool(var.get())
            self.config_mgr.debounce_save()
            if rebuild:
                self.controller.rebuild_debounced()
                self.rebuild()

        tk.Checkbutton(
            parent,
            text=label,
            variable=var,
            command=on_toggle,
            bg=self.bg,
            fg=self.fg,
            selectcolor=self.bg,
            activebackground=self.bg,
            activeforeground=self.fg,
        ).pack(anchor="w", padx=indent)

        return var

    def _build_profile_tab(self, frame):
        core = self._section(frame, "Core")
        self._draw_slider(core, "Gamma", self._profile_key("GAMMA"), 0.75, 1.4, 0.001)
        self._draw_slider(core, "Gamma Offset", self._profile_key("GAMMA_OFFSET"), -0.2, 0.2, 0.001)
        self._draw_slider(core, "Black Floor", self._profile_key("BLACK_FLOOR"), 0.0, 0.1, 0.0005)
        self._draw_slider(core, "Vibrance", self._profile_key("VIBRANCE"), 0.9, 1.3, 0.001)

        shadows = self._section(frame, "Shadows")
        self._draw_slider(shadows, "Shadow Lift Exp", self._profile_key("SHADOW_LIFT_EXP"), 0.4, 1.0, 0.001)
        self._draw_slider(shadows, "Shadow Cutoff", self._profile_key("SHADOW_CUTOFF"), 0.15, 0.6, 0.001)
        self._draw_slider(shadows, "Shadow Desat", self._profile_key("SHADOW_DESAT"), 0.5, 1.0, 0.001)
        self._draw_slider(shadows, "Shadow Sigmoid Boost", self._profile_key("SHADOW_SIGMOID_BOOST"), 0.0, 1.0, 0.01)
        self._draw_slider(shadows, "Shadow Color Bias", self._profile_key("SHADOW_COLOR_BIAS"), -0.05, 0.05, 0.001)

        mids = self._section(frame, "Midtones")
        self._draw_slider(mids, "Midtone Boost", self._profile_key("MIDTONE_BOOST"), 0.9, 1.3, 0.001)

        hi = self._section(frame, "Highlights")
        self._draw_slider(hi, "Highlight Compress", self._profile_key("HIGHLIGHT_COMPRESS"), 0.0, 0.7, 0.001)

    def _build_adaptive_tab(self, frame):
        hist = self._section(frame, "Histogram-Aware Shadows")
        hist_var = self._draw_toggle(hist, "Enable Histogram Adaptation", "HISTOGRAM_ADAPTIVE")
        if hist_var.get():
            self._draw_slider(hist, "Strength", "HISTOGRAM_STRENGTH", 0.0, 1.0, 0.01)
            self._draw_slider(hist, "Min Scene Luma", "HISTOGRAM_MIN_LUMA", 0.10, 0.50, 0.01)
            self._draw_slider(hist, "Max Scene Luma", "HISTOGRAM_MAX_LUMA", 0.40, 0.90, 0.01)

        motion = self._section(frame, "Motion-Based Visibility")
        motion_var = self._draw_toggle(motion, "Enable Motion-aware Shadows", "MOTION_AWARE_SHADOWS")
        if motion_var.get():
            m = self.controller.current_mode.value.upper()
            self._draw_slider(motion, f"Motion Strength ({m})", f"MOTION_STRENGTH_{m}", 0.0, 1.0, 0.01)
            self._draw_slider(motion, "Motion Sensitivity", "MOTION_SENSITIVITY", 0.5, 5.0, 0.1)

            emph_var = self._draw_toggle(motion, "Enable Motion Shadow Emphasis", "MOTION_SHADOW_EMPHASIS", indent=18)
            if emph_var.get():
                self._draw_slider(motion, "Emphasis Strength", "MOTION_SHADOW_STRENGTH", 0.0, 1.0, 0.01)
                self._draw_slider(motion, "Dark Luma Gate", "MOTION_SHADOW_DARK_LUMA", 0.35, 0.6, 0.01)
                self._draw_slider(motion, "Min Motion Threshold", "MOTION_SHADOW_MIN_MOTION", 0.0, 0.05, 0.001)

        edge = self._section(frame, "Edge-Preserving Contrast")
        edge_var = self._draw_toggle(edge, "Enable Edge-aware Shadows", "EDGE_AWARE_SHADOWS")
        if edge_var.get():
            self._draw_slider(edge, "Edge Strength", "EDGE_STRENGTH", 0.0, 1.0, 0.01)
            self._draw_slider(edge, "Edge Min", "EDGE_MIN", 0.0, 0.5, 0.01)
            self._draw_slider(edge, "Edge Max", "EDGE_MAX", 0.3, 1.0, 0.01)

        hud = self._section(frame, "HUD Awareness")
        hud_var = self._draw_toggle(hud, "Enable HUD Highlight Exclusion", "HUD_EXCLUSION")
        if hud_var.get():
            self._draw_slider(hud, "HUD Exclusion Strength", "HUD_EXCLUSION_STRENGTH", 0.0, 1.0, 0.01)
            self._draw_slider(hud, "HUD Exclusion Threshold", "HUD_EXCLUSION_THRESHOLD", 0.75, 0.98, 0.01)

    def _build_advanced_tab(self, frame):
        m = self.controller.current_mode.value.upper()

        rgb = self._section(frame, "RGB & Channel Bias", start_open=False)
        self._draw_slider(rgb, "Shadow Red Bias", f"SHADOW_RED_BIAS_{m}", 0.95, 1.05, 0.001)
        self._draw_slider(rgb, "Shadow Green Bias", f"SHADOW_GREEN_BIAS_{m}", 0.95, 1.05, 0.001)
        self._draw_slider(rgb, "Shadow Blue Bias", f"SHADOW_BLUE_BIAS_{m}", 0.95, 1.05, 0.001)

        opp = self._section(frame, "Opponent Channel Tuning", start_open=False)
        self._draw_slider(opp, "Opponent Strength", "OPPONENT_STRENGTH", 0.0, 1.0, 0.01)
        self._draw_toggle(opp, "Enable Opponent Tuning", "OPPONENT_TUNING", indent=6, rebuild=True)

    def _build_global_tab(self, frame):
        self._draw_slider(frame, "Shadow Pop Strength", "SHADOW_POP_STRENGTH", 0.0, 1.0, 0.01)

        hud = tk.BooleanVar(value=bool(self.config.get("PRESERVE_HUD_HIGHLIGHTS", True)))

        def on_toggle():
            self.config["PRESERVE_HUD_HIGHLIGHTS"] = bool(hud.get())
            self.controller.rebuild_debounced()
            self.config_mgr.debounce_save()

        tk.Checkbutton(
            frame,
            text="Preserve HUD Highlights",
            variable=hud,
            command=on_toggle,
            bg=self.bg,
            fg=self.fg,
            selectcolor=self.bg,
            activebackground=self.bg,
            activeforeground=self.fg,
        ).pack(anchor="w", padx=6, pady=8)
