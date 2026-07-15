#!/usr/bin/env python3
"""Validation pipeline for the image-retrieval engine.

Builds the feature databases (if missing), evaluates MAP@{3,5,11,21} for every
feature, times brute-force vs FLANN retrieval, and prints a "normal vs improved"
comparison. Results are written to results/ as CSV, Markdown and PNG charts so
they can be dropped straight into the report. Talks to the C++ engine through its
JSON output (emitted whenever stdout is piped). Uses the standard library plus
matplotlib for the charts.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile

# Basic features (Task 1) are evaluated on the no-preprocessing database; the
# improved set (Task 2) adds shape/edge/texture features and the combined vector
# and is evaluated on the preprocessed database.
BASIC_FEATURES = ["color_hist", "correlogram", "sift", "orb"]
IMPROVED_FEATURES = ["color_hist", "correlogram", "sift", "orb",
                     "hu", "lbp", "hog", "combined", "combinedw"]
KS = ["3", "5", "11", "21"]
K_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]  # one bar colour per k


def _pyplot():
    """Imports matplotlib with a headless backend; raises ImportError if absent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_map(rows, out_path, title):
    """Grouped bar chart of MAP@k per feature; highlights the best feature."""
    plt = _pyplot()
    data = [(feat, m) for _, feat, m in rows if m]
    if not data:
        return None
    feats = [d[0] for d in data]
    best = max(range(len(data)), key=lambda i: data[i][1]["5"])  # best by MAP@5

    width = 0.2
    fig, ax = plt.subplots(figsize=(max(8, len(feats) * 1.15), 4.6))
    for j, k in enumerate(KS):
        xs = [i + (j - 1.5) * width for i in range(len(feats))]
        ax.bar(xs, [d[1][k] for d in data], width, label="MAP@" + k,
               color=K_COLORS[j])
    ax.axvspan(best - 0.5, best + 0.5, color="#FFF1A8", zorder=0)  # best band
    ax.annotate("★ best", xy=(best, data[best][1]["5"]),
                xytext=(0, 12), textcoords="offset points", ha="center",
                fontweight="bold", color="#D19A00")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=25, ha="right")
    ax.get_xticklabels()[best].set_fontweight("bold")
    ax.set_ylabel("MAP")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=9)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_speed(speed, out_path):
    """Two-bar chart of brute-force vs FLANN time; highlights the faster one."""
    if not speed:
        return None
    plt = _pyplot()
    names, vals = ["Brute-force", "FLANN"], [speed["brute"], speed["flann"]]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    bars = ax.bar(names, vals, color=["#C44E52", "#55A868"], width=0.55)
    for b, v in zip(bars, vals):
        ax.annotate("%.3f ms" % v, xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontweight="bold")
    ax.annotate("★ %.1fx faster" % speed["speedup"],
                xy=(1, speed["flann"]), xytext=(0, 24),
                textcoords="offset points", ha="center", fontweight="bold",
                color="#D19A00")
    ax.set_ylabel("Query time (ms)")
    ax.set_title("Brute-force vs FLANN (combined)")
    ax.margins(y=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(engine, args, want_json):
    """Runs the engine (stdout piped) and returns parsed JSON, or None on error."""
    proc = subprocess.run([engine] + args, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write("  ! engine error: " + proc.stderr.strip() + "\n")
        return None
    if not want_json:
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stderr.write("  ! bad output: " + proc.stdout[:200] + "\n")
        return None


def ensure_index(engine, images, db_path, mode, rebuild):
    """Builds a feature database unless it already exists (or rebuild is forced)."""
    if os.path.isfile(db_path) and not rebuild:
        print("  using existing %s" % db_path)
        return True
    print("  building %s (%s features)... this can take a few minutes" %
          (db_path, mode))
    return run(engine, ["-buildIndex", images, db_path, mode], False) is not None


def eval_map(engine, db_path, csv_path, feat):
    """Returns the MAP@k dict for one feature, or None on failure."""
    d = run(engine, ["-evalMAP", db_path, csv_path, feat], True)
    return d["map"] if d else None


def time_retrieve(engine, query, db_path, feat, img_dir, repeats,
                  flann_index=None):
    """Median search time (ms) over `repeats` runs, brute-force or FLANN."""
    out = os.path.join(tempfile.gettempdir(), "bench_grid.png")
    times = []
    for _ in range(repeats):
        if flann_index:
            args = ["-retrieveFast", query, out, db_path, flann_index, feat,
                    "21", img_dir]
        else:
            args = ["-retrieve", query, out, db_path, feat, "21", img_dir]
        d = run(engine, args, True)
        if d:
            times.append(d["time_ms"])
    return statistics.median(times) if times else None


def print_table(title, rows):
    """Prints a MAP table: rows are (label, feature, map_dict)."""
    print("\n" + title)
    print("  %-10s %-12s %8s %8s %8s %8s" %
          ("set", "feature", "MAP@3", "MAP@5", "MAP@11", "MAP@21"))
    print("  " + "-" * 60)
    for label, feat, m in rows:
        if m is None:
            print("  %-10s %-12s   (evaluation failed)" % (label, feat))
        else:
            print("  %-10s %-12s %8.4f %8.4f %8.4f %8.4f" %
                  (label, feat, m["3"], m["5"], m["11"], m["21"]))


def write_outputs(outdir, rows, speed):
    """Writes the scores as CSV and a Markdown table for the report."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "map_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "feature", "MAP@3", "MAP@5", "MAP@11", "MAP@21"])
        for label, feat, m in rows:
            if m:
                w.writerow([label, feat, m["3"], m["5"], m["11"], m["21"]])
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("# Retrieval evaluation\n\n")
        f.write("| Set | Feature | MAP@3 | MAP@5 | MAP@11 | MAP@21 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for label, feat, m in rows:
            if m:
                f.write("| %s | %s | %.4f | %.4f | %.4f | %.4f |\n" %
                        (label, feat, m["3"], m["5"], m["11"], m["21"]))
        if speed:
            f.write("\n## Query speed (combined feature)\n\n")
            f.write("| Method | Median search time (ms) | Speedup |\n")
            f.write("|---|---|---|\n")
            f.write("| Brute-force | %.3f | 1.0x |\n" % speed["brute"])
            f.write("| FLANN kd-tree | %.3f | %.1fx |\n" %
                    (speed["flann"], speed["speedup"]))
    print("\nWrote %s/map_scores.csv and %s/summary.md" % (outdir, outdir))


def main():
    # This script lives in sources/benchmark/, so the project root is two levels up.
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", default=os.path.join(root, "releases",
                                                  "image-retrieval"))
    ap.add_argument("--images", default=os.path.join(root, "datasets", "TMBuD",
                                                     "images"))
    ap.add_argument("--csv", default=os.path.join(root, "datasets", "TMBuD",
                                                  "DATASET SPLIT.csv"))
    ap.add_argument("--db-basic", default=os.path.join(root, "db_basic.yml"))
    ap.add_argument("--db-all", default=os.path.join(root, "db_all.yml"))
    ap.add_argument("--outdir", default=os.path.join(root, "results"))
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the feature databases even if present")
    ap.add_argument("--skip-speed", action="store_true",
                    help="skip the brute-force vs FLANN timing test")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip generating the PNG charts")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    if not os.path.isfile(args.bin):
        sys.exit("engine binary not found: %s (run `make` first)" % args.bin)

    print("== 1. Feature databases ==")
    if not ensure_index(args.bin, args.images, args.db_basic, "basic",
                        args.rebuild):
        sys.exit("failed to build basic index")
    if not ensure_index(args.bin, args.images, args.db_all, "all",
                        args.rebuild):
        sys.exit("failed to build improved index")

    print("\n== 2. MAP evaluation ==")
    basic_rows = [("basic", f, eval_map(args.bin, args.db_basic, args.csv, f))
                  for f in BASIC_FEATURES]
    improved_rows = [("improved", f, eval_map(args.bin, args.db_all, args.csv, f))
                     for f in IMPROVED_FEATURES]
    print_table("Task 1 - basic features (no preprocessing):", basic_rows)
    print_table("Task 2 - improved (preprocessing + extra features + combined):",
                improved_rows)

    # Normal vs improved: best single basic feature vs the weighted fusion.
    print("\n== 3. Normal vs improved ==")
    best_basic = max((r for r in basic_rows if r[2]),
                     key=lambda r: r[2]["21"], default=None)
    improved = next((r for r in improved_rows if r[1] == "combinedw" and r[2]),
                    None)
    if best_basic and improved:
        print("  %-32s %8s %8s %8s %8s" % ("", "MAP@3", "MAP@5", "MAP@11",
                                           "MAP@21"))
        for tag, r in (("normal (best basic: %s)" % best_basic[1], best_basic),
                       ("improved (combinedw fusion)", improved)):
            m = r[2]
            print("  %-32s %8.4f %8.4f %8.4f %8.4f" %
                  (tag, m["3"], m["5"], m["11"], m["21"]))
        gain = improved[2]["21"] - best_basic[2]["21"]
        print("  -> combinedw improves MAP@21 by %+.4f (%+.1f%%)" %
              (gain, 100.0 * gain / best_basic[2]["21"]))

    speed = None
    if not args.skip_speed:
        print("\n== 4. Query speed: brute-force vs FLANN (combined) ==")
        query = os.path.join(args.images, sorted(os.listdir(args.images))[0])
        idx = os.path.join(tempfile.gettempdir(), "bench_flann_combined.bin")
        run(args.bin, ["-buildFlannIndex", args.db_all, "combined", idx], False)
        brute = time_retrieve(args.bin, query, args.db_all, "combined",
                              args.images, args.repeats)
        flann = time_retrieve(args.bin, query, args.db_all, "combined",
                              args.images, args.repeats, flann_index=idx)
        if brute and flann:
            speed = {"brute": brute, "flann": flann, "speedup": brute / flann}
            print("  brute-force : %.3f ms" % brute)
            print("  FLANN       : %.3f ms" % flann)
            print("  speedup     : %.1fx" % speed["speedup"])

    write_outputs(args.outdir, basic_rows + improved_rows, speed)

    if not args.no_plot:
        try:
            os.makedirs(args.outdir, exist_ok=True)
            p1 = plot_map(improved_rows,
                          os.path.join(args.outdir, "map_plot.png"),
                          "MAP per feature (preprocessed database)")
            p2 = plot_speed(speed, os.path.join(args.outdir, "speed_plot.png"))
            print("Wrote " + ", ".join(p for p in (p1, p2) if p))
        except ImportError:
            print("matplotlib not found; skipping charts "
                  "(enter the nix dev shell, or pass --no-plot)")


if __name__ == "__main__":
    main()
