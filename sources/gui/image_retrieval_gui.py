#!/usr/bin/env python3
"""Tkinter GUI front-end for the C++ image-retrieval engine.

It shells out to the `image-retrieval` binary, which prints JSON when its stdout
is piped, and renders the ranked results as a thumbnail grid.
"""
import glob
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading

# When frozen into a single PyInstaller binary, the Tcl/Tk runtime data is
# embedded under _MEIPASS. Point Tk at it *before* importing tkinter, otherwise
# Tk aborts with "Can't find a usable init.tcl / Tk data directory not found".
if getattr(sys, "frozen", False):
    _base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    for _env, _sub in (("TCL_LIBRARY", "_tcl_data"), ("TK_LIBRARY", "_tk_data")):
        _d = os.path.join(_base, _sub)
        if os.path.isdir(_d):
            os.environ[_env] = _d

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    sys.stderr.write("This GUI requires Pillow (pip install pillow).\n")
    raise

FEATURES = ["combinedw", "color_hist", "correlogram", "sift", "orb", "hu",
            "lbp", "hog", "combined"]
TOP_K_CHOICES = ["3", "5", "11", "21"]
THUMB = 140  # result thumbnail edge (long side) in pixels
# The building shots are landscape, so a thumbnail scales to roughly this wide;
# budgeting the real width (not THUMB) lets a full extra column fit per row.
CELL_W = 128  # per-column budget used to choose how many columns fit
ENGINE_EXE = "image-retrieval.exe" if os.name == "nt" else "image-retrieval"


def find_default_binary():
    """Returns the most likely path to the engine binary.

    In a frozen single-file build the engine is embedded and unpacked into
    _MEIPASS, so we look there first (and restore its executable bit, which the
    unpack can drop on Unix).
    """
    candidates = []
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        candidates += [os.path.join(base, ENGINE_EXE),
                       os.path.join(os.path.dirname(sys.executable), ENGINE_EXE)]
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, ENGINE_EXE),                             # beside GUI
        os.path.join(here, "..", "..", "releases", ENGINE_EXE),     # source tree
    ]
    for c in candidates:
        if os.path.isfile(c):
            if os.name != "nt":
                try:
                    os.chmod(c, 0o755)
                except OSError:
                    pass
            return os.path.abspath(c)
    return ENGINE_EXE  # fall back to PATH


def find_project_root():
    """Walks up looking for the TMBuD dataset; returns its root ("" if none).

    Searches from both the script location and, in a frozen build, the folder
    holding the binary - so shipping the dataset next to the single-file GUI
    still auto-fills the paths.
    """
    starts = [os.path.dirname(os.path.abspath(__file__))]
    if getattr(sys, "frozen", False):
        starts.insert(0, os.path.dirname(os.path.abspath(sys.executable)))
    for start in starts:
        d = start
        for _ in range(5):
            if os.path.isdir(os.path.join(d, "datasets", "TMBuD")):
                return d
            d = os.path.dirname(d)
    return ""


class App:
    def __init__(self, root):
        self.root = root
        root.title("TMBuD Image Retrieval")
        # Size to fit the screen (never taller/wider than it), so the window
        # doesn't overflow on smaller displays.
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry("%dx%d" % (min(1150, sw - 100), min(800, sh - 120)))
        root.minsize(760, 480)
        self.thumb_refs = []  # keep PhotoImage refs alive

        proj = find_project_root()
        self.var_bin = tk.StringVar(value=find_default_binary())
        self.var_db = tk.StringVar(
            value=os.path.join(proj, "db_all.yml") if proj else "")
        self.var_images = tk.StringVar(
            value=os.path.join(proj, "datasets", "TMBuD", "images") if proj else "")
        self.var_csv = tk.StringVar(
            value=os.path.join(proj, "datasets", "TMBuD", "DATASET SPLIT.csv")
            if proj else "")
        self.var_query = tk.StringVar()
        self.var_feature = tk.StringVar(value="combinedw")
        self.var_topk = tk.StringVar(value="21")
        self.var_flann = tk.BooleanVar(value=False)
        self.var_index = tk.StringVar(
            value=os.path.join(proj, "flann_combined.bin") if proj else "")
        self.var_status = tk.StringVar(value="Ready.")
        self.var_qtime = tk.StringVar(value="Query time will show here after a search.")

        self.build_win = None  # the build dialog, when open

        # Pack order matters: the bottom status bar must reserve its space
        # before the results area (which expands) claims the rest.
        self._build_controls()
        self._build_statusbar()
        self._build_results_area()

        # FLANN indexes one metric space; combinedw fuses several, so it can't
        # use a single index. Keep the checkbox in sync with the chosen feature.
        self.var_feature.trace_add("write", lambda *a: self._sync_flann_state())
        self._sync_flann_state()

        # After the window is drawn, offer to build an index if none is present.
        self.root.after(250, self._prompt_build_on_startup)

    # --- widget construction ------------------------------------------------
    def _build_controls(self):
        f = ttk.LabelFrame(self.root, text="Configuration")
        f.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self._path_row(f, 0, "Engine binary", self.var_bin, self._pick_file)
        self._path_row(f, 1, "Feature database", self.var_db, self._pick_file)
        self._path_row(f, 2, "Image directory", self.var_images, self._pick_dir)
        self._path_row(f, 3, "Dataset CSV", self.var_csv, self._pick_file)
        self._path_row(f, 4, "Query image", self.var_query, self._pick_query)

        opt = ttk.Frame(f)
        opt.grid(row=5, column=0, columnspan=3, sticky="w", padx=4, pady=4)
        ttk.Label(opt, text="Feature:").pack(side=tk.LEFT)
        ttk.Combobox(opt, textvariable=self.var_feature, values=FEATURES,
                     width=12, state="readonly").pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(opt, text="Top-k:").pack(side=tk.LEFT)
        ttk.Combobox(opt, textvariable=self.var_topk, values=TOP_K_CHOICES,
                     width=5).pack(side=tk.LEFT, padx=(2, 12))
        self.flann_check = ttk.Checkbutton(
            opt, text="Use FLANN index", variable=self.var_flann,
            command=self._toggle_index)
        self.flann_check.pack(side=tk.LEFT)
        self.index_entry = ttk.Entry(opt, textvariable=self.var_index, width=32)
        self.index_entry.pack(side=tk.LEFT, padx=(6, 0))
        self._toggle_index()

        btns = ttk.Frame(f)
        btns.grid(row=6, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 4))
        ttk.Button(btns, text="Search", command=self.do_search).pack(
            side=tk.LEFT)
        ttk.Button(btns, text="Evaluate MAP", command=self.do_eval).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Build / Rebuild index...",
                   command=self.open_build_dialog).pack(side=tk.LEFT)
        f.columnconfigure(1, weight=1)

    def _path_row(self, parent, row, label, var, picker):
        ttk.Label(parent, text=label + ":").grid(row=row, column=0, sticky="e",
                                                 padx=4, pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="we",
                                                 padx=4, pady=2)
        ttk.Button(parent, text="Browse",
                   command=lambda: picker(var)).grid(row=row, column=2, padx=4)

    def _build_results_area(self):
        outer = ttk.LabelFrame(self.root, text="Results")
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)
        # Prominent, highlighted query-time banner across the top of the results.
        self.qtime_label = tk.Label(
            outer, textvariable=self.var_qtime, anchor="w", padx=12, pady=6,
            font=("TkDefaultFont", 14, "bold"), fg="#0b0b0b",
            background="#FFF1A8")
        self.qtime_label.pack(side=tk.TOP, fill=tk.X)
        self.canvas = tk.Canvas(outer, background="#1e1e1e")
        vsb = ttk.Scrollbar(outer, orient="vertical",
                            command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = tk.Frame(self.canvas, background="#1e1e1e")
        self._inner_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # Keep the results frame exactly as wide as the canvas so the columns
        # can spread across the full width (no dead band on the right), and
        # re-flow the grid when a resize changes how many columns fit.
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._cur_cols = 0
        self._last_render = None
        # Scroll with the mouse wheel while the pointer is over the results.
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _on_canvas_configure(self, e):
        self.canvas.itemconfigure(self._inner_id, width=e.width)
        cols = max(2, (e.width - 12) // CELL_W)
        if self._last_render and cols != self._cur_cols:
            qp, data = self._last_render
            self._render_results(qp, data)

    def _bind_wheel(self, _=None):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)  # Windows / macOS
        self.canvas.bind_all("<Button-4>", self._on_wheel)    # X11 scroll up
        self.canvas.bind_all("<Button-5>", self._on_wheel)    # X11 scroll down

    def _unbind_wheel(self, _=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, e):
        if getattr(e, "num", None) == 4 or getattr(e, "delta", 0) > 0:
            self.canvas.yview_scroll(-1, "units")
        elif getattr(e, "num", None) == 5 or getattr(e, "delta", 0) < 0:
            self.canvas.yview_scroll(1, "units")

    def _build_statusbar(self):
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bar, textvariable=self.var_status, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=2)

    # --- file pickers -------------------------------------------------------
    def _pick_file(self, var):
        p = filedialog.askopenfilename()
        if p:
            var.set(p)

    def _pick_dir(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _pick_query(self, var):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")])
        if p:
            var.set(p)

    def _toggle_index(self):
        self.index_entry.configure(
            state="normal" if self.var_flann.get() else "disabled")

    def _default_index_path(self, feat):
        """A per-feature index filename next to the database.

        Each feature has its own dimensionality, so a shared index file would be
        mismatched; per-feature names keep FLANN from loading the wrong one."""
        db = self.var_db.get()
        base = os.path.dirname(db) if db else (find_project_root() or os.getcwd())
        return os.path.join(base, "flann_%s.bin" % feat)

    def _sync_flann_state(self):
        """FLANN is unavailable for combinedw (multi-metric fusion); grey it out.
        For other features, point the index box at that feature's own file."""
        feat = self.var_feature.get()
        if feat == "combinedw":
            self.var_flann.set(False)
            self.flann_check.configure(state="disabled")
        else:
            self.flann_check.configure(state="normal")
            self.var_index.set(self._default_index_path(feat))
        self._toggle_index()

    # --- engine invocation --------------------------------------------------
    def _run_engine(self, args):
        """Runs the engine with stdout piped and returns parsed JSON (or raises)."""
        env = os.environ.copy()
        if getattr(sys, "frozen", False):
            # Any shared libs bundled beside the engine live in _MEIPASS; make
            # the engine load those first so it runs without a system OpenCV.
            base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
            var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            env[var] = base + os.pathsep + env.get(var, "")
        try:
            proc = subprocess.run([self.var_bin.get()] + args,
                                  capture_output=True, text=True, env=env)
        except FileNotFoundError:
            raise RuntimeError("Engine binary not found: " + self.var_bin.get())
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "engine returned an error")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError("Unexpected engine output:\n" + proc.stdout[:500])

    def do_search(self):
        query = self.var_query.get()
        if not query or not os.path.isfile(query):
            messagebox.showwarning("Search", "Please choose a valid query image.")
            return
        out_png = os.path.join(tempfile.gettempdir(), "retrieval_grid.png")
        feat, k = self.var_feature.get(), self.var_topk.get()
        img_dir = self.var_images.get()
        building = False
        if self.var_flann.get():
            args = ["-retrieveFast", query, out_png, self.var_db.get(),
                    self.var_index.get(), feat, k, img_dir]
            # First FLANN search on a feature builds (and caches) its index.
            building = not os.path.isfile(self.var_index.get())
        else:
            args = ["-retrieve", query, out_png, self.var_db.get(), feat, k,
                    img_dir]
        if building:
            self.var_status.set("Building FLANN index for '%s' (first run)..." % feat)
            self.qtime_label.configure(background="#FFD8A8")
            self.var_qtime.set("Building FLANN index for '%s' - one moment..."
                               % feat)
        else:
            self.var_status.set("Searching...")
        self.root.update_idletasks()
        try:
            data = self._run_engine(args)
        except RuntimeError as e:
            self.var_status.set("Error.")
            self.qtime_label.configure(background="#FFF1A8")
            self.var_qtime.set("Search failed.")
            messagebox.showerror("Search failed", str(e))
            return
        self._render_results(query, data)
        mode, ms = data.get("mode", "?"), data.get("time_ms", 0.0)
        note = "  (index built this run)" if building else ""
        self.qtime_label.configure(background="#FFF1A8")
        self.var_qtime.set("Query time: %.3f ms   |   %s%s" %
                           (ms, mode, note))
        self.var_status.set(
            "%s | %s | %d results in %.3f ms" %
            (mode, data.get("feature", feat), data.get("count", 0), ms))

    def do_eval(self):
        feat = self.var_feature.get()
        self.var_status.set("Evaluating MAP (this scans the whole database)...")
        self.root.update_idletasks()
        try:
            data = self._run_engine(
                ["-evalMAP", self.var_db.get(), self.var_csv.get(), feat])
        except RuntimeError as e:
            self.var_status.set("Error.")
            messagebox.showerror("Evaluate MAP failed", str(e))
            return
        m = data.get("map", {})
        lines = ["MAP@%s = %.4f" % (k, m[k]) for k in ("3", "5", "11", "21")
                 if k in m]
        summary = "feature [%s] over %d queries / %d buildings\n\n%s" % (
            data.get("feature", feat), data.get("n_queries", 0),
            data.get("n_buildings", 0), "\n".join(lines))
        self.var_status.set("MAP [%s]: " % feat + "  ".join(lines))
        messagebox.showinfo("Mean Average Precision", summary)

    # --- build / rebuild index ----------------------------------------------
    def _prompt_build_on_startup(self):
        """On launch, offer to build an index if the configured DB is missing."""
        db = self.var_db.get()
        if db and os.path.isfile(db):
            return  # a usable index already exists
        if messagebox.askyesno(
                "Build feature index",
                "No feature index was found.\n\n"
                "Build one now? You'll pick the image folder next."):
            self.open_build_dialog(startup=True)

    def open_build_dialog(self, startup=False):
        """Opens the dialog that builds/rebuilds the feature database."""
        if self.build_win is not None and self.build_win.winfo_exists():
            self.build_win.lift()
            return
        proj = find_project_root()
        self.b_images = tk.StringVar(
            value=self.var_images.get() or
            (os.path.join(proj, "datasets", "TMBuD", "images") if proj else ""))
        self.b_db = tk.StringVar(
            value=self.var_db.get() or
            (os.path.join(proj, "db_all.yml") if proj else
             os.path.join(os.getcwd(), "db_all.yml")))
        self.b_mode = tk.StringVar(value="all")

        win = tk.Toplevel(self.root)
        win.title("Build / Rebuild feature index")
        win.geometry("640x360")
        win.transient(self.root)
        self.build_win = win

        frm = ttk.Frame(win, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Image folder to index:").grid(
            row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.b_images).grid(
            row=1, column=0, sticky="we", padx=(0, 6))
        ttk.Button(frm, text="Browse",
                   command=lambda: self._pick_dir(self.b_images)).grid(row=1,
                                                                       column=1)
        ttk.Label(frm, text="Output database file:").grid(
            row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.b_db).grid(
            row=3, column=0, sticky="we", padx=(0, 6))
        ttk.Button(frm, text="Browse", command=self._pick_save_db).grid(
            row=3, column=1)

        mode = ttk.Frame(frm)
        mode.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(mode, text="Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode, text="all  (preprocessing + every feature, "
                        "enables combinedw)", value="all",
                        variable=self.b_mode).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Radiobutton(mode, text="basic  (Task-1 features only)",
                        value="basic", variable=self.b_mode).pack(side=tk.LEFT)

        self.b_progress = ttk.Progressbar(frm, mode="determinate")
        self.b_progress.grid(row=5, column=0, columnspan=2, sticky="we",
                             pady=(14, 4))
        self.b_status = tk.StringVar(value="Ready to build.")
        ttk.Label(frm, textvariable=self.b_status).grid(
            row=6, column=0, columnspan=2, sticky="w")

        actions = ttk.Frame(frm)
        actions.grid(row=7, column=0, columnspan=2, sticky="e", pady=(14, 0))
        self.b_build_btn = ttk.Button(actions, text="Build", command=self._start_build)
        self.b_build_btn.pack(side=tk.LEFT)
        ttk.Button(actions, text="Close", command=win.destroy).pack(
            side=tk.LEFT, padx=(6, 0))
        frm.columnconfigure(0, weight=1)

    def _pick_save_db(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".yml",
            filetypes=[("YAML database", "*.yml"), ("All", "*.*")])
        if p:
            self.b_db.set(p)

    def _start_build(self):
        img_dir = self.b_images.get()
        db_out = self.b_db.get()
        if not img_dir or not os.path.isdir(img_dir):
            messagebox.showwarning("Build", "Choose a valid image folder.")
            return
        if not db_out:
            messagebox.showwarning("Build", "Choose an output database file.")
            return
        # The engine indexes *.png directly inside the folder. Catch the common
        # mistake of picking the dataset root instead of its images/ subfolder.
        if not glob.glob(os.path.join(img_dir, "*.png")):
            sub = os.path.join(img_dir, "images")
            hint = ("\n\nTip: this folder has an 'images' subfolder - "
                    "pick that instead.") if glob.glob(
                        os.path.join(sub, "*.png")) else ""
            messagebox.showwarning(
                "Build", "No .png images found directly in:\n%s%s" %
                (img_dir, hint))
            return
        if os.path.isfile(db_out) and not messagebox.askyesno(
                "Overwrite", "%s already exists. Overwrite it?" % db_out):
            return
        self.b_build_btn.configure(state="disabled")
        # Animate until the first "indexed N/M" arrives, so it never looks dead.
        self.b_progress.configure(mode="indeterminate")
        self.b_progress.start(12)
        self.b_status.set("Starting... (extracting features)")
        self._build_queue = queue.Queue()
        # Read every Tk value here on the main thread; the worker thread must not
        # touch Tk objects (Tk is not thread-safe -- doing so hangs the build).
        bin_path = self.var_bin.get()
        args = ["-buildIndex", img_dir, db_out, self.b_mode.get()]
        t = threading.Thread(target=self._build_worker,
                             args=(bin_path, args, db_out), daemon=True)
        t.start()
        self.root.after(100, self._poll_build)

    def _build_worker(self, bin_path, args, db_out):
        """Runs the engine in a thread, streaming progress to a queue.

        Takes plain strings only -- never reads Tk variables from this thread."""
        q = self._build_queue
        env = os.environ.copy()
        if getattr(sys, "frozen", False):
            base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
            var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            env[var] = base + os.pathsep + env.get(var, "")
        try:
            proc = subprocess.Popen([bin_path] + args,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, env=env)
        except FileNotFoundError:
            q.put(("error", "Engine binary not found: " + bin_path))
            return
        # Progress lines look like "  indexed 850/1363" and use \r, so split on
        # both carriage return and newline to catch every update.
        buf = ""
        while True:
            ch = proc.stderr.read(1)
            if not ch:
                break
            if ch in "\r\n":
                m = re.search(r"indexed (\d+)/(\d+)", buf)
                if m:
                    q.put(("progress", int(m.group(1)), int(m.group(2))))
                buf = ""
            else:
                buf += ch
        out = proc.stdout.read()
        proc.wait()
        if proc.returncode != 0:
            q.put(("error", (buf or out or "engine returned an error").strip()))
            return
        entries = None
        try:
            entries = json.loads(out).get("entries")
        except (json.JSONDecodeError, AttributeError):
            pass
        q.put(("done", entries, db_out))

    def _poll_build(self):
        """Drains the build queue on the main thread and updates widgets."""
        try:
            while True:
                msg = self._build_queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    done, total = msg[1], msg[2]
                    if str(self.b_progress["mode"]) == "indeterminate":
                        self.b_progress.stop()
                        self.b_progress.configure(mode="determinate")
                    self.b_progress.configure(maximum=total, value=done)
                    self.b_status.set("Indexing %d / %d images..." % (done, total))
                elif kind == "error":
                    self.b_progress.stop()
                    self.b_build_btn.configure(state="normal")
                    self.b_status.set("Build failed.")
                    messagebox.showerror("Build failed", msg[1])
                    return
                elif kind == "done":
                    entries, db_out = msg[1], msg[2]
                    self.b_progress.stop()
                    # No entries and no output file means nothing was indexed.
                    if entries in (None, 0) and not os.path.isfile(db_out):
                        self.b_progress.configure(mode="determinate", value=0)
                        self.b_build_btn.configure(state="normal")
                        self.b_status.set("Nothing indexed.")
                        messagebox.showerror(
                            "Build failed",
                            "No images were indexed. Make sure the folder "
                            "directly contains the .png files.")
                        return
                    self.b_progress.configure(mode="determinate")
                    self.b_progress.configure(value=self.b_progress["maximum"])
                    n = "%d" % entries if entries is not None else "the"
                    self.b_status.set("Done - indexed %s entries." % n)
                    # Adopt the new database for searching straight away.
                    self.var_db.set(db_out)
                    if self.b_mode.get() != "all":
                        self.var_feature.set("color_hist")
                    self.var_status.set("Index built: " + db_out)
                    self.b_build_btn.configure(state="normal")
                    messagebox.showinfo(
                        "Build complete",
                        "Indexed %s entries into:\n%s\n\nReady to search." %
                        (n, db_out))
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_build)

    # --- rendering ----------------------------------------------------------
    def _load_thumb(self, path, size):
        img = Image.open(path)
        img.thumbnail((size, size))
        photo = ImageTk.PhotoImage(img)
        self.thumb_refs.append(photo)
        return photo

    def _render_results(self, query_path, data):
        for w in self.inner.winfo_children():
            w.destroy()
        self.thumb_refs.clear()
        self._last_render = (query_path, data)

        # Fit as many result columns as the current width allows (min 2), so the
        # grid never spills past the right edge on a narrower window.
        cw = self.canvas.winfo_width()
        cols = max(2, (cw - 12) // CELL_W) if cw > 50 else 6
        self._cur_cols = cols
        # Give every column equal weight so the cells share the full width
        # evenly instead of hugging the left with a blank band on the right.
        for c in range(64):
            self.inner.columnconfigure(
                c, weight=1 if c < cols else 0,
                uniform="rcell" if c < cols else "")

        # Query panel spanning the top row.
        qframe = tk.Frame(self.inner, background="#1e1e1e")
        qframe.grid(row=0, column=0, columnspan=cols, sticky="w", padx=6, pady=6)
        try:
            tk.Label(qframe, image=self._load_thumb(query_path, THUMB),
                     background="#1e1e1e").pack(side=tk.LEFT)
        except Exception:
            pass
        tk.Label(qframe, text="QUERY\n" + os.path.basename(query_path),
                 fg="#ffd000", background="#1e1e1e", justify=tk.LEFT).pack(
            side=tk.LEFT, padx=10)
        for i, r in enumerate(data.get("results", [])):
            cell = tk.Frame(self.inner, background="#1e1e1e", bd=1,
                            relief=tk.FLAT)
            cell.grid(row=1 + i // cols, column=i % cols, padx=5, pady=5)
            path = r.get("path", "")
            if os.path.isfile(path):
                try:
                    tk.Label(cell, image=self._load_thumb(path, THUMB),
                             background="#1e1e1e").pack()
                except Exception:
                    tk.Label(cell, text="(unreadable)", fg="#aaaaaa",
                             background="#1e1e1e").pack()
            else:
                tk.Label(cell, text="(missing)\n" + r.get("file", ""),
                         fg="#aaaaaa", background="#1e1e1e").pack()
            tk.Label(cell,
                     text="#%d  %s\nd=%.3f" % (r.get("rank", i + 1),
                                               r.get("file", ""),
                                               r.get("distance", 0.0)),
                     fg="#7CFC00", background="#1e1e1e").pack()


def _selftest(argv):
    """Decode a sample image the way thumbnails do; verifies a frozen bundle
    ships PIL's codecs + the ImageTk bridge. Usage: --selftest [image.png]."""
    path = argv[argv.index("--selftest") + 1] if len(
        argv) > argv.index("--selftest") + 1 else ""
    if not path:
        proj = find_project_root()
        path = os.path.join(proj, "datasets", "TMBuD", "images", "00001.png") \
            if proj else ""
    root = tk.Tk()
    root.withdraw()
    try:
        img = Image.open(path)
        img.thumbnail((64, 64))
        ImageTk.PhotoImage(img)
        print("selftest OK: decoded %s (mode=%s)" %
              (os.path.basename(path), img.mode))
    except Exception as e:
        print("selftest FAIL: %r" % e)
        sys.exit(1)
    finally:
        root.destroy()


def main():
    if "--selftest" in sys.argv:
        _selftest(sys.argv)
        return
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
