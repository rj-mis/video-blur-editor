# -*- coding: utf-8 -*-
"""

Created on Fri Nov 20 14:05:12 2026

VIDEO MASK EDITOR WITH RIGHT-HAND MASK PANEL 
==============================================================
Install in Spyder's Python environment:
    pip install opencv-python numpy Pillow

Tkinter is included with standard Windows Python installations.
Open this file in Spyder, edit VIDEO_PATH below (or leave it empty to browse),
then press F5. The editor opens as a regular desktop window.

WORKFLOW
  1. Seek with the slider. Choose Rectangle or Polygon; draw on the picture.
  2. Click the new mask in the list on the RIGHT.
  3. Edit its start/end time (seconds or MM:SS.ss), coordinates, effect,
     and pixelation grid. Click 'Apply edits'.
  4. Choose Select / move to drag an existing mask or its corner handles.
  5. Save/load masks as JSON; Export MP4 for the processed video.

EXTRA CONTROLS
  SPACE: play/pause (when not typing)      I: mark interval start
  O: mark interval end                    N: clear interval markers
  V: select / move                        B: rectangle drawing
  G: polygon drawing                      ENTER: finish polygon
  ESC: cancel current drawing             DELETE: delete selected mask
  LEFT / RIGHT: step one frame (when not typing)

Important: masks use FIXED coordinates over their time intervals. You can add
several masks to cover a moving subject. Before sharing a video containing
sensitive information, watch the entire rendered output carefully.

The saved JSON is compatible with files from the earlier OpenCV editor.

@author: RaphaelJacquat
"""

from __future__ import annotations

import json
import math
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# ========================== USER SETTINGS ================================
VIDEO_PATH = r""  # e.g. r"C:\Users\You\Videos\example.mp4"
OUTPUT_VIDEO_PATH = r""  # empty -> source_name_pixelated.mp4
DEFAULT_SECONDS_PER_MASK = 4.0
DEFAULT_PIXEL_GRID = 8            # smaller grid = stronger pixelation
GAUSSIAN_BLUR_KERNEL = 75
PRESERVE_AUDIO_IF_FFMPEG_EXISTS = True
WINDOW_TITLE = "Video mask editor - time & coordinates"
# ========================================================================


def format_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:06.3f}"


def parse_time(value):
    """Parse '12.5', '01:12.5', or '01:02:12.5' to seconds."""
    parts = str(value).strip().split(":")
    if not 1 <= len(parts) <= 3 or any(not x.strip() for x in parts):
        raise ValueError("Use seconds (12.5) or MM:SS.ss (01:12.50)")
    try:
        nums = [float(s.strip()) for s in parts]
    except ValueError as exc:
        raise ValueError("Time must contain numbers") from exc
    if any(not math.isfinite(n) or n < 0 for n in nums):
        raise ValueError("Time must be a positive, finite number")
    if len(nums) > 1 and any(n >= 60 for n in nums[1:]):
        raise ValueError("Minutes/seconds components must be below 60")
    result = 0.0
    for n in nums:
        result = result * 60 + n
    return result


def rect_points(x, y, w, h):
    """Rectangle vertices, using left/top and width/height in native px."""
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def is_rectangle(points):
    if len(points) != 4:
        return False
    a = np.asarray(points, dtype=np.int32)
    xs = a[:, 0]
    ys = a[:, 1]
    return (len(set(xs)) == 2 and len(set(ys)) == 2 and
            set(map(tuple, a)) == {
                (int(xs.min()), int(ys.min())),
                (int(xs.max()), int(ys.min())),
                (int(xs.max()), int(ys.max())),
                (int(xs.min()), int(ys.max())),
            })


def validate_polygon(points, width, height):
    if len(points) < 3:
        raise ValueError("A mask needs at least 3 vertices")
    clean = []
    for p in points:
        if len(p) != 2:
            raise ValueError("Each vertex must have two coordinates")
        xx, yy = (int(round(float(v))) for v in p)
        if not (0 <= xx < width and 0 <= yy < height):
            raise ValueError(f"Point ({xx}, {yy}) outside video ({width}x{height})")
        clean.append([xx, yy])
    if abs(cv2.contourArea(np.asarray(clean, dtype=np.float32))) < 16:
        raise ValueError("The selected area is too small")
    return clean


def apply_effects(frame, masks, frame_index):
    """Composite all active masks. Time end is EXCLUSIVE, in frame units."""
    out = frame.copy()
    h, w = out.shape[:2]
    for mask in masks:
        if not (mask["start_frame"] <= frame_index < mask["end_frame"]):
            continue
        pts = np.asarray(mask["points"], dtype=np.int32)
        if len(pts) < 3:
            continue
        x0 = max(0, int(pts[:, 0].min()))
        y0 = max(0, int(pts[:, 1].min()))
        x1 = min(w, int(pts[:, 0].max()) + 1)
        y1 = min(h, int(pts[:, 1].max()) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        if mask.get("mode", "pixelate") == "blur":
            kernel = max(3, int(GAUSSIAN_BLUR_KERNEL) | 1)
            # OpenCV may reject a Gaussian kernel much larger than a tiny ROI;
            # smaller kernels are appropriate for such regions anyway.
            kernel = min(kernel, max(3, (min(rw, rh) * 2 - 1) | 1))
            effect = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        else:
            grid = max(1, int(mask.get("pixel_grid", DEFAULT_PIXEL_GRID)))
            tiny = cv2.resize(roi, (min(grid, rw), min(grid, rh)),
                              interpolation=cv2.INTER_AREA)
            effect = cv2.resize(tiny, (rw, rh), interpolation=cv2.INTER_NEAREST)
        binary = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(binary, [pts - np.array([x0, y0])], 255)
        roi[binary != 0] = effect[binary != 0]
    return out


class MaskVideoEditor:
    def __init__(self, root, source, output):
        self.root = root
        self.source = Path(source).expanduser().resolve()
        self.output = Path(output).expanduser().resolve()
        if not self.source.is_file():
            raise FileNotFoundError(f"Video not found: {self.source}")
        if self.source == self.output:
            raise ValueError("Output must not overwrite the original video")
        self.json_path = self.source.with_name(self.source.stem + "_masks.json")
        self.cap = cv2.VideoCapture(str(self.source))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.source}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 25.0
        if min(self.width, self.height, self.total) < 1:
            raise RuntimeError("Video has no readable dimensions or frames")
        self.duration = self.total / self.fps

        self.masks = []
        self.selected = None
        self.tool = "rectangle"  # select, rectangle, polygon
        self.frame_idx = 0
        self.playing = False
        self.play_start_frame = 0
        self.play_start_wall = time.perf_counter()
        self.mark_in = None
        self.mark_out = None
        self.draft_polygon = []
        self.drag = None
        self.cached_idx = -1
        self.cached_image = None
        self.photo = None
        self.display_bounds = (0, 0, 1, 1)
        self.last_canvas_size = (1, 1)
        self.dirty = True
        self.refreshing_list = False
        self.render_in_progress = False
        self.worker_messages = queue.Queue()
        self.last_progress = -1

        root.title(WINDOW_TITLE)
        root.geometry("1390x860")
        root.minsize(930, 600)
        self.style = ttk.Style(root)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        self.style.configure("TButton", padding=(9, 6))
        self.style.configure("Treeview", rowheight=25)
        self.make_gui()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Key>", self.key_press, add=True)
        root.after(25, self.tick)
        print(f"Opened: {self.source}\n{self.width}x{self.height}, "
              f"{self.fps:.2f} fps, {self.duration:.2f} seconds")

    def make_gui(self):
        self.root.configure(bg="#101a27")
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(main)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        ttk.Button(toolbar, text="Save masks", command=self.save_json).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Load masks", command=self.load_json).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Export MP4", command=self.export).pack(side="left", padx=5)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=9)
        ttk.Button(toolbar, text="Select / move  [V]",
                   command=lambda: self.change_tool("select")).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Rectangle  [B]",
                   command=lambda: self.change_tool("rectangle")).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Polygon  [G]",
                   command=lambda: self.change_tool("polygon")).pack(side="left", padx=3)

        video_panel = ttk.Frame(main)
        video_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        video_panel.rowconfigure(0, weight=1)
        video_panel.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(video_panel, bg="#0e1724", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.canvas_resized)
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<Double-Button-1>", self.double_click)
        self.canvas.bind("<Button-3>", self.cancel_drawing)
        self.canvas.bind("<Motion>", self.mouse_hover)

        transport = ttk.Frame(video_panel, padding=(4, 8, 4, 4))
        transport.grid(row=1, column=0, sticky="ew")
        ttk.Button(transport, text="◀", width=3, command=lambda: self.seek(self.frame_idx-1)).pack(side="left")
        self.play_button = ttk.Button(transport, text="Play", width=7, command=self.toggle_play)
        self.play_button.pack(side="left", padx=5)
        ttk.Button(transport, text="▶", width=3, command=lambda: self.seek(self.frame_idx+1)).pack(side="left")
        self.timeline = ttk.Scale(transport, from_=0, to=max(1, self.total-1),
                                  command=self.on_timeline)
        self.timeline.pack(side="left", fill="x", expand=True, padx=8)
        self.time_label = ttk.Label(transport, text="00:00.00 / " + format_time(self.duration),
                                    width=21, anchor="e")
        self.time_label.pack(side="right")

        interval = ttk.Frame(video_panel)
        interval.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        ttk.Button(interval, text="Mark IN [I]", command=self.set_in).pack(side="left", padx=3)
        ttk.Button(interval, text="Mark OUT [O]", command=self.set_out).pack(side="left", padx=3)
        ttk.Button(interval, text="Reset interval [N]", command=self.clear_interval).pack(side="left", padx=3)
        self.interval_label = ttk.Label(interval, text="New masks: current time + 4.0s")
        self.interval_label.pack(side="left", padx=9)

        self.status = tk.StringVar(value="Draw a mask, then edit its properties in the right panel.")
        ttk.Label(video_panel, textvariable=self.status, anchor="w").grid(
            row=3, column=0, sticky="ew", pady=(8, 2))

        sidebar = ttk.Frame(main, width=365, padding=(8, 0, 0, 0))
        sidebar.grid(row=1, column=1, sticky="ns")
        sidebar.grid_propagate(False)
        sidebar.rowconfigure(2, weight=1)
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="Masks", font=("Segoe UI", 15, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 7))
        ttk.Label(sidebar, text="Select a mask to edit its time, position and effect.",
                  wraplength=342).grid(row=1, column=0, sticky="w", pady=(0, 8))
        list_container = ttk.Frame(sidebar)
        list_container.grid(row=2, column=0, sticky="nsew")
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(list_container, columns=("time", "effect"),
                                 show="tree headings", height=8, selectmode="browse")
        self.tree.heading("#0", text="Mask")
        self.tree.heading("time", text="Start - End")
        self.tree.heading("effect", text="Effect")
        self.tree.column("#0", width=67, minwidth=53, stretch=False)
        self.tree.column("time", width=164, stretch=True)
        self.tree.column("effect", width=80, stretch=False)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_list_selection)

        buttons = ttk.Frame(sidebar)
        buttons.grid(row=3, column=0, sticky="ew", pady=(7, 10))
        ttk.Button(buttons, text="Duplicate", command=self.duplicate).pack(side="left", padx=(0, 5))
        ttk.Button(buttons, text="Delete", command=self.delete_selected).pack(side="left", padx=5)
        ttk.Button(buttons, text="Go to start", command=self.goto_selected).pack(side="left", padx=5)

        editor = ttk.LabelFrame(sidebar, text="Selected mask properties", padding=10)
        editor.grid(row=4, column=0, sticky="ew")
        editor.columnconfigure(1, weight=1)
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="pixelate")
        self.grid_var = tk.StringVar(value=str(DEFAULT_PIXEL_GRID))
        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        self.w_var = tk.StringVar()
        self.h_var = tk.StringVar()

        def entry_row(r, label, var):
            ttk.Label(editor, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=3)
            e = ttk.Entry(editor, textvariable=var, width=20)
            e.grid(row=r, column=1, sticky="ew", pady=3)
            return e

        entry_row(0, "Start", self.start_var)
        entry_row(1, "End", self.end_var)
        ttk.Label(editor, text="Times: seconds or MM:SS.sss", foreground="#64748b").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 5))
        ttk.Label(editor, text="Effect").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(editor, textvariable=self.mode_var, values=("pixelate", "blur"),
                     state="readonly").grid(row=3, column=1, sticky="ew", pady=3)
        entry_row(4, "Pixel grid", self.grid_var)
        ttk.Separator(editor).grid(row=5, column=0, columnspan=2, sticky="ew", pady=8)
        entry_row(6, "X (px)", self.x_var)
        entry_row(7, "Y (px)", self.y_var)
        entry_row(8, "Width (px)", self.w_var)
        entry_row(9, "Height (px)", self.h_var)
        ttk.Label(editor, text="Polygon points (one x,y per line):").grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(7, 3))
        self.points_box = tk.Text(editor, height=4, wrap="none", font=("Consolas", 9))
        self.points_box.grid(row=11, column=0, columnspan=2, sticky="ew")
        ttk.Label(editor, text="For rectangles edit X/Y/Width/Height; for polygons edit points.",
                  wraplength=330, foreground="#64748b").grid(
            row=12, column=0, columnspan=2, sticky="w", pady=(5, 8))
        ttk.Button(editor, text="Apply edits", command=self.apply_edits).grid(
            row=13, column=0, columnspan=2, sticky="ew")

        ttk.Label(sidebar, text="Tip: Select / move lets you drag the entire mask or individual corners.",
                  wraplength=340, foreground="#64748b").grid(row=5, column=0,
                  sticky="w", pady=(10, 0))
        self.editor_enabled(False)

    def editor_enabled(self, enabled):
        # Clear the form when nothing is selected.
        if not enabled:
            for var in (self.start_var, self.end_var, self.x_var, self.y_var,
                        self.w_var, self.h_var):
                var.set("")
            self.points_box.delete("1.0", "end")

    def tell(self, message):
        self.status.set(str(message))
        print(message)

    def change_tool(self, tool):
        self.tool = tool
        self.draft_polygon.clear()
        self.drag = None
        self.canvas.configure(cursor="fleur" if tool == "select" else "crosshair")
        self.tell(f"Tool: {tool}." + (" Click to add points, press Enter to finish." if tool == "polygon" else ""))
        self.dirty = True

    def current_interval(self):
        start = self.mark_in if self.mark_in is not None else self.frame_idx
        if self.mark_in is not None and self.mark_out is not None:
            end = self.mark_out
        else:
            end = min(self.total, start + max(1, round(DEFAULT_SECONDS_PER_MASK * self.fps)))
        return int(start), int(max(start + 1, end))

    def update_interval_label(self):
        if self.mark_in is None:
            value = f"New masks: current time + {DEFAULT_SECONDS_PER_MASK:g}s"
        elif self.mark_out is None:
            value = f"IN {format_time(self.mark_in / self.fps)}; default duration"
        else:
            value = (f"IN {format_time(self.mark_in / self.fps)} - "
                     f"OUT {format_time(self.mark_out / self.fps)}")
        self.interval_label.configure(text=value)

    def set_in(self):
        self.mark_in = self.frame_idx
        self.mark_out = None
        self.update_interval_label()

    def set_out(self):
        if self.mark_in is None:
            self.tell("Set IN first.")
            return
        if self.frame_idx <= self.mark_in:
            self.tell("OUT must be later than IN.")
            return
        self.mark_out = min(self.total, self.frame_idx + 1)
        self.seek(self.mark_in)
        self.update_interval_label()

    def clear_interval(self):
        self.mark_in = self.mark_out = None
        self.update_interval_label()

    def on_timeline(self, value):
        # Tk Scale callback fires on a programmatic set() as well.
        if getattr(self, "updating_slider", False):
            return
        self.seek(int(round(float(value))), update_scale=False)

    def seek(self, frame, update_scale=True):
        self.playing = False
        self.play_button.configure(text="Play")
        self.frame_idx = int(np.clip(frame, 0, self.total-1))
        if update_scale:
            self.updating_slider = True
            self.timeline.set(self.frame_idx)
            self.updating_slider = False
        self.dirty = True

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_start_wall = time.perf_counter()
            self.play_start_frame = self.frame_idx
            self.play_button.configure(text="Pause")
        else:
            self.play_button.configure(text="Play")

    def get_raw_frame(self):
        if self.cached_idx != self.frame_idx:
            if self.cached_idx + 1 != self.frame_idx:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ok, frame = self.cap.read()
            if not ok:
                return self.cached_image
            self.cached_image = frame
            self.cached_idx = self.frame_idx
        return self.cached_image

    def canvas_resized(self, event):
        self.dirty = True

    def video_bounds(self):
        cw = max(2, self.canvas.winfo_width())
        ch = max(2, self.canvas.winfo_height())
        s = min(cw / self.width, ch / self.height)
        w = max(1, int(round(self.width * s)))
        h = max(1, int(round(self.height * s)))
        ox = (cw - w) // 2
        oy = (ch - h) // 2
        return (ox, oy, w, h)

    def native_to_canvas(self, p):
        ox, oy, w, h = self.display_bounds
        return (ox + float(p[0]) * w / self.width,
                oy + float(p[1]) * h / self.height)

    def canvas_to_native(self, x, y, clamp=False):
        ox, oy, w, h = self.display_bounds
        if not clamp and not (ox <= x < ox + w and oy <= y < oy + h):
            return None
        return (int(np.clip(round((x-ox)*self.width/w), 0, self.width-1)),
                int(np.clip(round((y-oy)*self.height/h), 0, self.height-1)))

    def draw_preview(self):
        raw = self.get_raw_frame()
        if raw is None:
            return
        self.display_bounds = self.video_bounds()
        ox, oy, w, h = self.display_bounds
        processed = apply_effects(raw, self.masks, self.frame_idx)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        # Pillow 9/10+ compatibility.
        resampling = getattr(Image, "Resampling", Image)
        pil = Image.fromarray(rgb).resize((w, h), resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, image=self.photo, anchor="nw")
        for i, mask in enumerate(self.masks):
            selected = (i == self.selected)
            active = mask["start_frame"] <= self.frame_idx < mask["end_frame"]
            if not active and not selected:
                continue
            flat = [v for p in mask["points"] for v in self.native_to_canvas(p)]
            color = "#20d5ff" if selected else "#74e6a6"
            self.canvas.create_polygon(flat, fill="", outline=color,
                                       width=3 if selected else 2)
            if selected:
                for p in mask["points"]:
                    x, y = self.native_to_canvas(p)
                    self.canvas.create_rectangle(x-5, y-5, x+5, y+5,
                                                 outline="#072936", fill="#20d5ff")
                if not active:
                    self.canvas.create_text(ox+12, oy+14, anchor="w",
                                            text="Selected mask is outside its active time interval",
                                            fill="#ffffff", font=("Segoe UI", 10, "bold"))
            else:
                lx, ly = self.native_to_canvas(mask["points"][0])
                self.canvas.create_text(lx+4, ly-4, anchor="sw", text=str(i+1),
                                        fill=color, font=("Segoe UI", 11, "bold"))
        if self.drag and self.drag["kind"] == "draw_rect":
            a = self.native_to_canvas(self.drag["start"])
            b = self.native_to_canvas(self.drag["last"])
            self.canvas.create_rectangle(*a, *b, outline="#ffcc65", width=2)
        if self.draft_polygon:
            coords = [self.native_to_canvas(p) for p in self.draft_polygon]
            for x, y in coords:
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="#ffcc65", outline="")
            if len(coords) > 1:
                self.canvas.create_line(*[v for p in coords for v in p],
                                        fill="#ffcc65", width=2)
        self.time_label.configure(text=f"{format_time(self.frame_idx/self.fps)} / "
                                       f"{format_time(self.duration)}")
        self.dirty = False

    def add_mask(self, points, shape):
        try:
            points = validate_polygon(points, self.width, self.height)
        except ValueError as exc:
            self.tell(str(exc))
            return
        start, end = self.current_interval()
        self.masks.append(dict(points=points, shape=shape, start_frame=start,
                               end_frame=end, mode="pixelate",
                               pixel_grid=DEFAULT_PIXEL_GRID))
        self.selected = len(self.masks)-1
        self.refresh_mask_list()
        self.populate_editor()
        self.dirty = True
        self.tell(f"Added mask #{self.selected+1}. Edit its settings on the right.")

    def hit_test(self, p):
        # Selected handles take priority, even outside the mask's time interval.
        if self.selected is not None:
            mask = self.masks[self.selected]
            for vi, pt in enumerate(mask["points"]):
                cx, cy = self.native_to_canvas(pt)
                px, py = self.native_to_canvas(p)
                if math.hypot(cx-px, cy-py) <= 11:
                    return self.selected, "vertex", vi
        candidates = list(range(len(self.masks)-1, -1, -1))
        if self.selected is not None:
            candidates = [self.selected] + [n for n in candidates if n != self.selected]
        for idx in candidates:
            m = self.masks[idx]
            active = m["start_frame"] <= self.frame_idx < m["end_frame"]
            if not active and idx != self.selected:
                continue
            pts = np.asarray(m["points"], dtype=np.float32)
            if cv2.pointPolygonTest(pts, (float(p[0]), float(p[1])), False) >= 0:
                return idx, "move", None
        return None

    def mouse_down(self, event):
        if self.render_in_progress:
            return
        p = self.canvas_to_native(event.x, event.y)
        if p is None:
            return
        self.playing = False
        self.play_button.configure(text="Play")
        if self.tool == "rectangle":
            self.drag = dict(kind="draw_rect", start=p, last=p)
        elif self.tool == "polygon":
            if not self.draft_polygon or p != self.draft_polygon[-1]:
                self.draft_polygon.append(p)
        else:
            hit = self.hit_test(p)
            if hit:
                index, kind, vertex = hit
                self.selected = index
                self.refresh_mask_list()
                self.populate_editor()
                self.drag = dict(kind=kind, index=index, vertex=vertex,
                                 previous=p)
            else:
                self.selected = None
                self.refresh_mask_list()
                self.editor_enabled(False)
        self.dirty = True

    def mouse_drag(self, event):
        if not self.drag:
            return
        p = self.canvas_to_native(event.x, event.y, clamp=True)
        kind = self.drag["kind"]
        if kind == "draw_rect":
            self.drag["last"] = p
        else:
            mask = self.masks[self.drag["index"]]
            if kind == "move":
                prev = self.drag["previous"]
                dx, dy = p[0]-prev[0], p[1]-prev[1]
                xs = [v[0] for v in mask["points"]]
                ys = [v[1] for v in mask["points"]]
                dx = max(-min(xs), min(dx, self.width-1-max(xs)))
                dy = max(-min(ys), min(dy, self.height-1-max(ys)))
                mask["points"] = [[x+dx, y+dy] for x, y in mask["points"]]
                self.drag["previous"] = (prev[0]+dx, prev[1]+dy)
            else:
                index = self.drag["vertex"]
                if mask.get("shape") == "rectangle":
                    opposite = mask["points"][(index+2)%4]
                    x0, x1 = sorted((p[0], opposite[0]))
                    y0, y1 = sorted((p[1], opposite[1]))
                    if x1-x0 >= 2 and y1-y0 >= 2:
                        mask["points"] = rect_points(x0, y0, x1-x0, y1-y0)
                else:
                    trial = deepcopy(mask["points"])
                    trial[index] = [p[0], p[1]]
                    if abs(cv2.contourArea(np.asarray(trial, np.float32))) >= 16:
                        mask["points"] = trial
        self.dirty = True

    def mouse_up(self, event):
        if not self.drag:
            return
        d = self.drag
        self.drag = None
        if d["kind"] == "draw_rect":
            p = self.canvas_to_native(event.x, event.y, clamp=True)
            x0, x1 = sorted((d["start"][0], p[0]))
            y0, y1 = sorted((d["start"][1], p[1]))
            if x1-x0 >= 4 and y1-y0 >= 4:
                self.add_mask(rect_points(x0, y0, x1-x0, y1-y0), "rectangle")
        else:
            self.populate_editor()
        self.dirty = True

    def double_click(self, event):
        if self.tool == "polygon":
            self.finish_polygon()

    def finish_polygon(self):
        pts = self.draft_polygon[:]
        self.draft_polygon.clear()
        # Tk double-click can add the same endpoint twice.
        if len(pts) > 1 and pts[-1] == pts[-2]:
            pts.pop()
        if len(pts) >= 3:
            self.add_mask(pts, "polygon")
        else:
            self.tell("Click at least three polygon corners, then press Enter.")
        self.dirty = True

    def cancel_drawing(self, event=None):
        self.draft_polygon.clear()
        self.drag = None
        self.dirty = True

    def mouse_hover(self, event):
        pass  # reserved for optional cursor/coordinate display

    def refresh_mask_list(self):
        self.refreshing_list = True
        try:
            for item in self.tree.get_children():
                self.tree.delete(item)
            for i, m in enumerate(self.masks):
                interval = (f"{format_time(m['start_frame']/self.fps)} - "
                            f"{format_time(m['end_frame']/self.fps)}")
                self.tree.insert("", "end", iid=str(i), text=f"#{i+1}",
                                 values=(interval, m.get("mode", "pixelate")))
            if self.selected is not None and self.selected < len(self.masks):
                self.tree.selection_set(str(self.selected))
                self.tree.see(str(self.selected))
            else:
                self.selected = None
        finally:
            self.refreshing_list = False

    def on_list_selection(self, event=None):
        if self.refreshing_list:
            return
        current = self.tree.selection()
        self.selected = int(current[0]) if current else None
        self.populate_editor()
        self.dirty = True

    def populate_editor(self):
        if self.selected is None or not 0 <= self.selected < len(self.masks):
            self.editor_enabled(False)
            return
        m = self.masks[self.selected]
        self.start_var.set(format_time(m["start_frame"]/self.fps))
        self.end_var.set(format_time(m["end_frame"]/self.fps))
        self.mode_var.set(m.get("mode", "pixelate"))
        self.grid_var.set(str(m.get("pixel_grid", DEFAULT_PIXEL_GRID)))
        pts = np.asarray(m["points"], dtype=np.int32)
        x0, x1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        y0, y1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        self.x_var.set(str(x0))
        self.y_var.set(str(y0))
        self.w_var.set(str(x1-x0))
        self.h_var.set(str(y1-y0))
        self.points_box.delete("1.0", "end")
        self.points_box.insert("1.0", "\n".join(f"{x}, {y}" for x, y in m["points"]))

    def apply_edits(self):
        if self.selected is None:
            self.tell("Select a mask in the list first.")
            return
        m = self.masks[self.selected]
        try:
            a = parse_time(self.start_var.get())
            b = parse_time(self.end_var.get())
            if not (0 <= a < b <= self.duration + 1e-5):
                raise ValueError(f"Times must satisfy 0 <= start < end <= {self.duration:.2f}s")
            start = int(round(a * self.fps))
            end = int(round(b * self.fps))
            if not 0 <= start < end <= self.total:
                raise ValueError("Interval needs at least one video frame")
            mode = self.mode_var.get()
            if mode not in ("pixelate", "blur"):
                raise ValueError("Choose pixelate or blur")
            grid = int(self.grid_var.get())
            if not 1 <= grid <= 512:
                raise ValueError("Pixel grid must be between 1 and 512")
            if m.get("shape", "rectangle") == "rectangle":
                x = int(self.x_var.get())
                y = int(self.y_var.get())
                w = int(self.w_var.get())
                h = int(self.h_var.get())
                if x < 0 or y < 0 or w < 2 or h < 2 or x+w >= self.width or y+h >= self.height:
                    raise ValueError(f"Rectangle must lie inside {self.width}x{self.height}")
                points = rect_points(x, y, w, h)
            else:
                lines = [v.strip() for v in self.points_box.get("1.0", "end").splitlines()
                         if v.strip()]
                points = []
                for line in lines:
                    parts = line.replace(";", ",").split(",")
                    if len(parts) != 2:
                        raise ValueError("Polygon points: one x,y pair per line")
                    points.append([int(v.strip()) for v in parts])
            points = validate_polygon(points, self.width, self.height)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid mask settings", str(exc), parent=self.root)
            return
        m.update(points=points, start_frame=start, end_frame=end,
                 mode=mode, pixel_grid=grid)
        self.refresh_mask_list()
        self.populate_editor()
        self.dirty = True
        self.tell(f"Updated mask #{self.selected+1}.")

    def goto_selected(self):
        if self.selected is not None:
            self.seek(self.masks[self.selected]["start_frame"])

    def duplicate(self):
        if self.selected is None:
            return
        self.masks.append(deepcopy(self.masks[self.selected]))
        self.selected = len(self.masks)-1
        self.refresh_mask_list()
        self.populate_editor()
        self.dirty = True
        self.tell(f"Duplicated as mask #{self.selected+1}.")

    def delete_selected(self):
        if self.selected is None:
            return
        n = self.selected
        self.masks.pop(n)
        self.selected = min(n, len(self.masks)-1) if self.masks else None
        self.refresh_mask_list()
        self.populate_editor()
        self.dirty = True
        self.tell(f"Deleted mask #{n+1}.")

    def save_json(self):
        payload = dict(format_version=2, source_video=str(self.source),
                       video_width=self.width, video_height=self.height,
                       video_fps=self.fps, video_frame_count=self.total,
                       masks=self.masks)
        self.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.tell(f"Saved {len(self.masks)} mask(s) to {self.json_path}")

    def load_json(self):
        if not self.json_path.exists():
            messagebox.showinfo("No mask file", f"No JSON found:\n{self.json_path}", parent=self.root)
            return
        try:
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
            if payload.get("video_width") != self.width or payload.get("video_height") != self.height:
                raise ValueError("Mask JSON does not match these video dimensions")
            result = []
            for old in payload.get("masks", []):
                points = validate_polygon(old["points"], self.width, self.height)
                start, end = int(old["start_frame"]), int(old["end_frame"])
                if not (0 <= start < end <= self.total):
                    raise ValueError("A saved time interval is outside this video")
                shape = old.get("shape") or ("rectangle" if is_rectangle(points) else "polygon")
                if shape not in ("rectangle", "polygon"):
                    shape = "polygon"
                mode = old.get("mode", "pixelate")
                if mode not in ("pixelate", "blur"):
                    mode = "pixelate"
                result.append(dict(points=points, shape=shape, start_frame=start,
                                   end_frame=end, mode=mode,
                                   pixel_grid=max(1, int(old.get("pixel_grid", DEFAULT_PIXEL_GRID)))))
            self.masks = result
            self.selected = 0 if result else None
            self.refresh_mask_list()
            self.populate_editor()
            self.dirty = True
            self.tell(f"Loaded {len(result)} masks from {self.json_path}")
            if abs(float(payload.get("video_fps", self.fps)) - self.fps) > 0.01:
                messagebox.showwarning("FPS mismatch",
                    "The saved masks were made with a different FPS. Check all time intervals.",
                    parent=self.root)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            messagebox.showerror("Could not load masks", str(exc), parent=self.root)

    def export(self):
        if self.render_in_progress:
            return
        if not self.masks:
            if not messagebox.askyesno("Export without masks",
                   "No masks have been added. Export unchanged video?", parent=self.root):
                return
        try:
            self.save_json()
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            return
        self.playing = False
        self.play_button.configure(text="Play")
        self.render_in_progress = True
        self.last_progress = -1
        masks_snapshot = deepcopy(self.masks)
        self.tell("Rendering video. You can monitor progress below...")
        thread = threading.Thread(target=self.render_worker, args=(masks_snapshot,), daemon=True)
        thread.start()

    def render_worker(self, masks):
        source = None
        writer = None
        temp = None
        try:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix="mask_editor_", suffix=".mp4",
                    dir=str(self.output.parent), delete=False) as handle:
                temp = Path(handle.name)
            source = cv2.VideoCapture(str(self.source))
            if not source.isOpened():
                raise RuntimeError("Cannot re-open the input video")
            writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.fps, (self.width, self.height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create MP4; check OpenCV codec support")
            written = 0
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                writer.write(apply_effects(frame, masks, written))
                written += 1
                if written % max(1, int(round(self.fps))) == 0:
                    self.worker_messages.put(("progress", written))
            writer.release()
            writer = None
            source.release()
            source = None
            if written == 0:
                raise RuntimeError("No frames were decoded")
            ffmpeg = shutil.which("ffmpeg") if PRESERVE_AUDIO_IF_FFMPEG_EXISTS else None
            if ffmpeg:
                cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                       "-i", str(temp), "-i", str(self.source),
                       "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy",
                       "-c:a", "aac", "-b:a", "160k", "-shortest",
                       "-movflags", "+faststart", str(self.output)]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    temp.unlink(missing_ok=True)
                    temp = None
                except (subprocess.CalledProcessError, OSError) as exc:
                    self.worker_messages.put(("warning", "FFmpeg audio merge failed; video saved without audio."))
                    self.output.unlink(missing_ok=True)
                    temp.replace(self.output)
                    temp = None
            else:
                temp.replace(self.output)
                temp = None
            self.worker_messages.put(("finished", f"Video ready: {self.output}"))
        except Exception as exc:
            self.worker_messages.put(("failed", f"Export failed: {exc}"))
        finally:
            if writer:
                writer.release()
            if source:
                source.release()
            if temp:
                temp.unlink(missing_ok=True)

    def key_press(self, event):
        # Avoid capturing keystrokes when the user edits a form field.
        focus = self.root.focus_get()
        if isinstance(focus, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox)):
            return
        key = event.keysym.lower()
        if key == "space":
            self.toggle_play()
        elif key == "i":
            self.set_in()
        elif key == "o":
            self.set_out()
        elif key == "n":
            self.clear_interval()
        elif key == "v":
            self.change_tool("select")
        elif key == "b":
            self.change_tool("rectangle")
        elif key == "g":
            self.change_tool("polygon")
        elif key == "return" and self.tool == "polygon":
            self.finish_polygon()
        elif key == "escape":
            self.cancel_drawing()
        elif key == "delete":
            self.delete_selected()
        elif key == "left":
            self.seek(self.frame_idx-1)
        elif key == "right":
            self.seek(self.frame_idx+1)

    def tick(self):
        if self.playing:
            elapsed = time.perf_counter()-self.play_start_wall
            next_frame = min(self.total-1, self.play_start_frame + int(elapsed*self.fps))
            if next_frame != self.frame_idx:
                self.frame_idx = next_frame
                self.updating_slider = True
                self.timeline.set(next_frame)
                self.updating_slider = False
                self.dirty = True
            if next_frame >= self.total-1:
                self.playing = False
                self.play_button.configure(text="Play")
        for _ in range(50):
            try:
                message_type, payload = self.worker_messages.get_nowait()
            except queue.Empty:
                break
            if message_type == "progress":
                new = int(payload * 100 / self.total)
                if new != self.last_progress:
                    self.last_progress = new
                    self.status.set(f"Rendering: {new}% ({payload}/{self.total} frames)")
            elif message_type == "finished":
                self.render_in_progress = False
                self.tell(payload)
                messagebox.showinfo("Export complete", payload, parent=self.root)
            elif message_type == "failed":
                self.render_in_progress = False
                self.tell(payload)
                messagebox.showerror("Export failed", payload, parent=self.root)
            elif message_type == "warning":
                print(payload)
        if self.dirty:
            self.draw_preview()
        self.root.after(30, self.tick)

    def close(self):
        if self.render_in_progress:
            messagebox.showwarning("Export underway",
                "The MP4 export is still running. Close after it finishes.", parent=self.root)
            return
        self.playing = False
        self.cap.release()
        self.root.destroy()


def pick_video():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Choose a video",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.wmv"),
                   ("All files", "*.*")], parent=root)
    root.destroy()
    return selected


def main():
    path = VIDEO_PATH.strip() or pick_video()
    if not path:
        print("No video selected.")
        return
    source = Path(path).expanduser().resolve()
    output = (OUTPUT_VIDEO_PATH.strip() or
              str(source.with_name(source.stem + "_pixelated.mp4")))
    root = tk.Tk()
    try:
        MaskVideoEditor(root, source, output)
    except Exception:
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
