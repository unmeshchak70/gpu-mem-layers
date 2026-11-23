#!/usr/bin/env python3
"""Figures and summary tables from mem_benchmarks CSVs, for the report.

Usage:
    python plot_bench.py results/ghc33.csv [results/ghc47.csv ...] \
        [--stats traces/*/stats.json] [--cache-frac 0.10] [--target 5] [--out results/figs]

Writes to --out (PDF for LaTeX, PNG for previewing):
    fig1_bandwidth_vs_size      contiguous reads / copies vs transfer size
    fig2_gather_vs_rows         fetching m scattered 1 KB rows: bandwidth vs m
    fig3_bulk_vs_zerocopy       bulk_total time / zero-copy time vs m
    fig4_predicted_slowdown     model slowdown vs hit rate (+ LRU hit rates from --stats)
    summary.md                  headline numbers, bulk breakdown, predicted slowdowns

Several CSVs (e.g. runs on different GHC machines) are combined: lines are the median
across runs and a light band shows min..max.

Model (from the proposal): with read hit rate h, B_eff = (h/B_dev + (1-h)/B_fetch)^-1,
so slowdown vs all-in-GPU is S(h) = B_dev/B_eff = h + r(1-h), r = B_dev/B_fetch.
B_dev = dev_gather_ro and B_fetch = zc_gather_ro, both at the largest m measured.
summary.md also gives a dedup-aware estimate: every read is served from device memory
and each *unique* missed row crosses PCIe once, S = 1 + r * u * (1 - h_unique),
u = unique ids per id read.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

# Validated categorical slots 1-4 (light mode, adjacent pairs pass CVD checks). Each memory
# path keeps its color in every figure. Slots 3-4 are under 3:1 contrast on white, so every
# series is also direct-labeled and the numbers are in summary.md.
DEVICE, ZEROCOPY, PINNED, NAIVE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ROW = 1024

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- data

def load_runs(paths):
    """One dict per CSV: bench -> {bytes: (time_ms, gbps)}."""
    runs = []
    for p in paths:
        d = defaultdict(dict)
        with open(p) as f:
            for r in csv.DictReader(f):
                d[r["bench"]][int(r["bytes"])] = (float(r["time_ms"]), float(r["gbps"]))
        runs.append(d)
    return runs


def series(runs, bench, field):
    """(bytes, median, min, max) over runs, on the sizes every run has. field: 'time' | 'gbps'."""
    have = [r[bench] for r in runs if bench in r]
    if not have:
        return None
    xs = sorted(set.intersection(*(set(h) for h in have)))
    i = 0 if field == "time" else 1
    v = np.array([[h[x][i] for x in xs] for h in have])
    return np.array(xs), np.median(v, 0), v.min(0), v.max(0)


def at_largest(runs, bench, field="gbps"):
    s = series(runs, bench, field)
    return None if s is None else float(s[1][-1])


# --------------------------------------------------------------------------- formatting

def fmt_bytes(x, _=None):
    for unit, div in (("G", 2**30), ("M", 2**20), ("K", 2**10)):
        if x >= div:
            v = x / div
            return f"{v:.0f}{unit}" if v == int(v) else f"{v:g}{unit}"
    return f"{x:.0f}"


def fmt_rows(x, _=None):
    return fmt_bytes(x)


def fmt_plain(x, _=None):
    return f"{x:g}"


def log2_axis(axis, lo, hi, step, fmt):
    ticks = [2.0**e for e in range(int(np.log2(lo)), int(np.log2(hi)) + 1, step)]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def log10_axis(axis, lo, hi):
    ticks = [10.0**e for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(fmt_plain))
    axis.set_minor_formatter(NullFormatter())


def end_labels(ax, items, min_gap_px=13):
    """Direct labels at each line's right end, nudged apart vertically so none overlap.

    items: [(x, y, text)]. Text stays in secondary ink; the line beside it carries color.
    """
    fig = ax.figure
    fig.canvas.draw()
    px_to_pt = 72.0 / fig.dpi
    pts = sorted(((ax.transData.transform((x, y))[1], x, y, t) for x, y, t in items),
                 key=lambda p: p[0])
    last = None
    for yd, x, y, t in pts:
        target = yd if last is None else max(yd, last + min_gap_px)
        last = target
        ax.annotate(t, xy=(x, y), xytext=(6, (target - yd) * px_to_pt), textcoords="offset points",
                    fontsize=8, color=INK2, va="center", ha="left", annotation_clip=False)


def plot_series(ax, s, color, x_div=1.0, field_label=None):
    xs, med, lo, hi = s
    x = xs / x_div
    if not np.array_equal(lo, hi):
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, med, color=color, label=field_label)
    return x[-1], med[-1]


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"wrote {out}/{name}.pdf/.png")


# --------------------------------------------------------------------------- figures

def fig_bandwidth_vs_size(runs, out):
    spec = [("dev_seq", DEVICE, "Device read"),
            ("zc_seq", ZEROCOPY, "Zero-copy read"),
            ("h2d_pinned", PINNED, "cudaMemcpy, pinned"),
            ("h2d_pageable", NAIVE, "cudaMemcpy, pageable")]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    labels, xmax, ys = [], 0, []
    for bench, color, name in spec:
        s = series(runs, bench, "gbps")
        if s is None:
            continue
        x, y = plot_series(ax, s, color, field_label=name)
        labels.append((x, y, f"{name}  {y:.1f} GB/s" if y < 100 else f"{name}  {y:.0f} GB/s"))
        xmax, ys = max(xmax, x), ys + list(s[1])
    log2_axis(ax.xaxis, 4096, xmax, 2, fmt_bytes)
    ax.set_ylim(min(ys) * 0.7, max(ys) * 1.6)
    log10_axis(ax.yaxis, *ax.get_ylim())
    ax.set_xlabel("Transfer size (bytes)")
    ax.set_ylabel("Bandwidth (GB/s, log)")
    ax.set_title("Contiguous transfers: device memory vs. host memory over PCIe")
    ax.legend(loc="lower right", ncol=2)
    end_labels(ax, labels)
    save(fig, out, "fig1_bandwidth_vs_size")


def fig_gather_vs_rows(runs, out):
    spec = [("dev_gather_ro", DEVICE, "Device gather (hit)"),
            ("zc_gather_ro", ZEROCOPY, "Zero-copy gather"),
            ("bulk_total", PINNED, "Bulk: CPU pack + 1 copy"),
            ("h2d_per_row", NAIVE, "1 cudaMemcpy per row")]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    labels, xmax, ys = [], 0, []
    for bench, color, name in spec:
        s = series(runs, bench, "gbps")
        if s is None:
            continue
        x, y = plot_series(ax, s, color, x_div=ROW, field_label=name)
        labels.append((x, y, f"{name}  {y:.2f} GB/s" if y < 1 else
                       (f"{name}  {y:.1f} GB/s" if y < 100 else f"{name}  {y:.0f} GB/s")))
        xmax, ys = max(xmax, x), ys + list(s[1])
    log2_axis(ax.xaxis, 1, xmax, 2, fmt_rows)
    ax.set_ylim(min(ys) * 0.7, max(ys) * 1.6)
    log10_axis(ax.yaxis, *ax.get_ylim())
    ax.set_xlabel("Rows fetched per batch, m (1 KB rows, random ids)")
    ax.set_ylabel("Bandwidth (GB/s, log)")
    ax.set_title("Fetching m scattered rows: hit path vs. three miss paths")
    ax.legend(loc="lower right", ncol=2)
    end_labels(ax, labels)
    save(fig, out, "fig2_gather_vs_rows")


def fig_bulk_vs_zerocopy(runs, out):
    zc, bk = series(runs, "zc_gather_ro", "time"), series(runs, "bulk_total", "time")
    if zc is None or bk is None:
        return None
    xs = np.array(sorted(set(zc[0]) & set(bk[0])))
    z = np.array([zc[1][list(zc[0]).index(x)] for x in xs])
    b = np.array([bk[1][list(bk[0]).index(x)] for x in xs])
    ratio, m = b / z, xs / ROW
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.set_xscale("log", base=2)
    ax.axhline(1.0, color=AXIS, linewidth=0.9, zorder=1)
    ax.text(m[0], 1.0, "  parity: bulk = zero-copy", color=MUTED, fontsize=8, va="bottom")
    ax.plot(m, ratio, color=DEVICE)
    i = int(np.argmin(ratio))
    ax.plot([m[i]], [ratio[i]], "o", color=DEVICE, markersize=5, markeredgecolor="white",
            markeredgewidth=1.5)
    ax.annotate(f"closest: {ratio[i]:.2f}× slower at m = {fmt_rows(m[i])}", xy=(m[i], ratio[i]),
                xytext=(0, -16), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    j = len(ratio) - 1
    ax.annotate(f"{ratio[j]:.2f}×", xy=(m[j], ratio[j]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)
    ax.annotate(f"{ratio[0]:.2f}×", xy=(m[0], ratio[0]), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=8, color=INK2)
    log2_axis(ax.xaxis, 1, m[-1], 2, fmt_rows)
    ax.set_ylim(0, max(ratio) * 1.2)
    ax.set_xlabel("Rows fetched per batch, m")
    ax.set_ylabel("Bulk time ÷ zero-copy time")
    ax.set_title("Bulk fetch is slower than zero-copy at every batch size")
    save(fig, out, "fig3_bulk_vs_zerocopy")
    return m, z, b, ratio


def load_stats(paths, cache_frac):
    rows = []
    for p in paths:
        with open(p) as f:
            s = json.load(f)
        m = s["meta"]
        for c in s["caches"]:
            rows.append(dict(trace=os.path.basename(os.path.dirname(os.path.abspath(p))),
                             mode=m["mode"], alpha=m["alpha"], drift=m["drift"],
                             frac=c["frac"], read_hit=c["lru_read_hit"],
                             uniq_hit=c["lru_unique_hit"], u=s["unique_fraction"],
                             selected=abs(c["frac"] - cache_frac) < 1e-9))
    return rows


def fig_predicted_slowdown(r, target, stats, cache_frac, out):
    h = np.linspace(0, 1, 201)
    S = h + r * (1 - h)
    h_need = (r - target) / (r - 1) if r > target else 0.0
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(h, S, color=DEVICE, label="Model: S = h + r(1−h)")
    ax.axhline(target, color=AXIS, linewidth=0.9, zorder=1)
    ax.text(0.01, target, f"  {target:g}× target", color=MUTED, fontsize=8, va="bottom")
    ax.plot([h_need], [target], "o", color=DEVICE, markersize=5, markeredgecolor="white",
            markeredgewidth=1.5, zorder=3)
    ax.annotate(f"needs h ≥ {h_need:.2f}", xy=(h_need, target), xytext=(-8, -14),
                textcoords="offset points", ha="right", fontsize=8, color=INK2)
    sel = [s for s in stats if s["selected"]]
    if sel:
        hs = np.array([s["read_hit"] for s in sel])
        ax.plot(hs, hs + r * (1 - hs), "o", color=ZEROCOPY, markersize=6, markeredgecolor="white",
                markeredgewidth=1.5, zorder=4,
                label=f"LRU, {100 * cache_frac:g}% cache, synthetic traces")
        show_drift = len({s["drift"] for s in sel}) > 1
        for s, x in zip(sel, hs):
            ax.annotate(f"α={s['alpha']:g}" + (f", drift {s['drift']:g}" if show_drift else ""),
                        xy=(x, x + r * (1 - x)), xytext=(6, 6), textcoords="offset points",
                        fontsize=7.5, color=INK2)
        ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, r * 1.05)
    ax.set_xlabel("Read hit rate in the device cache, h")
    ax.set_ylabel("Slowdown vs. all-in-GPU (×)")
    ax.set_title(f"Predicted slowdown, r = B_device / B_fetch = {r:.1f}")
    save(fig, out, "fig4_predicted_slowdown")
    return h_need


# --------------------------------------------------------------------------- summary

def write_summary(runs, csv_paths, out, ratio_data, r, target, h_need, stats):
    L = []
    L.append(f"# Microbenchmark summary\n\nRuns combined (median): {', '.join(csv_paths)}\n")
    L.append("## Peak bandwidth (largest size measured)\n")
    L.append("| Path | Bench | GB/s |\n|---|---|---|")
    for bench, name in [("dev_seq", "Device sequential read"), ("dev_gather_ro", "Device gather, read only"),
                        ("dev_gather", "Device gather + write"), ("zc_seq", "Zero-copy sequential read"),
                        ("zc_gather_ro", "Zero-copy gather, read only"), ("h2d_pinned", "cudaMemcpy H2D, pinned"),
                        ("h2d_pageable", "cudaMemcpy H2D, pageable"), ("bulk_total", "Bulk fetch total"),
                        ("h2d_per_row", "1 KB cudaMemcpy per row")]:
        v = at_largest(runs, bench)
        if v is not None:
            L.append(f"| {name} | `{bench}` | {v:.2f} |")
    s = series(runs, "zc_gather_ro", "gbps")
    if s is not None:
        k = int(np.argmax(s[1] >= 0.9 * s[1].max()))
        L.append(f"\nZero-copy gather reaches 90% of its peak at m = {fmt_rows(s[0][k] / ROW)} rows.")
    s = series(runs, "h2d_per_row", "time")
    if s is not None:
        L.append(f"Per-row cudaMemcpy costs {1000 * s[1][-1] / (s[0][-1] / ROW):.2f} µs per row.")
    s = series(runs, "bulk_total", "gbps")
    if s is not None:
        k = int(np.argmax(s[1]))
        L.append(f"Bulk fetch peaks at {s[1][k]:.2f} GB/s at m = {fmt_rows(s[0][k] / ROW)} rows.")

    if ratio_data is not None:
        m, z, b, ratio = ratio_data
        cpu, xfer = series(runs, "bulk_cpu", "time"), series(runs, "bulk_xfer", "time")
        L.append("\n## Zero-copy vs. bulk fetch (median ms)\n")
        L.append("| m rows | zero-copy | bulk total | bulk CPU pack | bulk copy + scatter | bulk ÷ zc |")
        L.append("|---|---|---|---|---|---|")
        for i in range(len(m)):
            if i % 2 == 0 or i == len(m) - 1:
                c = cpu[1][list(cpu[0] / ROW).index(m[i])] if cpu is not None else float("nan")
                x = xfer[1][list(xfer[0] / ROW).index(m[i])] if xfer is not None else float("nan")
                L.append(f"| {fmt_rows(m[i])} | {z[i]:.4f} | {b[i]:.4f} | {c:.4f} | {x:.4f} | {ratio[i]:.2f} |")

    if r is not None:
        L.append("\n## Performance model\n")
        L.append(f"B_device = {at_largest(runs, 'dev_gather_ro'):.1f} GB/s (`dev_gather_ro`), "
                 f"B_fetch = {at_largest(runs, 'zc_gather_ro'):.2f} GB/s (`zc_gather_ro`), r = {r:.1f}.")
        L.append(f"Slowdown S(h) = h + r(1−h); staying within {target:g}× needs read hit rate h ≥ {h_need:.3f}.")
    if stats:
        L.append("\n## Predicted slowdown on synthetic traces (LRU)\n")
        L.append("Proposal model uses the read hit rate. Dedup-aware: S = 1 + r·u·(1 − h_unique).\n")
        L.append("| trace | mode | α | drift | cache | read hit | unique hit | unique/ids | S (model) | S (dedup-aware) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in sorted(stats, key=lambda s: (s["mode"], s["alpha"], s["drift"], s["frac"])):
            S1 = s["read_hit"] + r * (1 - s["read_hit"])
            S2 = 1 + r * s["u"] * (1 - s["uniq_hit"])
            L.append(f"| {s['trace']} | {s['mode']} | {s['alpha']:g} | {s['drift']:g} | {100 * s['frac']:g}% | "
                     f"{s['read_hit']:.3f} | {s['uniq_hit']:.3f} | {s['u']:.3f} | {S1:.1f}× | {S2:.1f}× |")
    path = os.path.join(out, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {path}")


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="mem_benchmarks output CSV(s)")
    p.add_argument("--stats", nargs="*", default=[], help="workload_gen.py stats.json files")
    p.add_argument("--cache-frac", type=float, default=0.10, help="cache size plotted in fig 4")
    p.add_argument("--target", type=float, default=5.0, help="slowdown target (proposal: 5x)")
    p.add_argument("--out", default="results/figs")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = load_runs(a.csv)
    fig_bandwidth_vs_size(runs, a.out)
    fig_gather_vs_rows(runs, a.out)
    ratio_data = fig_bulk_vs_zerocopy(runs, a.out)
    stats = load_stats(a.stats, a.cache_frac)
    b_dev, b_fetch = at_largest(runs, "dev_gather_ro"), at_largest(runs, "zc_gather_ro")
    r = h_need = None
    if b_dev and b_fetch:
        r = b_dev / b_fetch
        h_need = fig_predicted_slowdown(r, a.target, stats, a.cache_frac, a.out)
    write_summary(runs, a.csv, a.out, ratio_data, r, a.target, h_need, stats)


if __name__ == "__main__":
    main()
