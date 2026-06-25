#!/usr/bin/env python3
"""Plot dense vs sparse benchmark comparison (Output Throughput + TPOT).

Reads .bench_rerun/comparison.csv and produces a 2-row x N-col grid:
  row 1: Output throughput (tok/s)  — higher is better
  row 2: Mean TPOT (ms, decode)     — lower is better
columns = input_len; lines = dense vs sparse over output_len.

Usage:
    python3 test-scripts/plot_compare.py [comparison.csv] [out.png]
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/paddlejob/inference-public/denghaodong/code/sglang/.bench_rerun/comparison.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/root/paddlejob/inference-public/denghaodong/code/sglang/.bench_rerun/comparison.png"

METRICS = [
    ("output_throughput", "Output throughput (tok/s)", True),
    ("mean_tpot_ms",      "Mean TPOT (ms) — decode",   False),
]

# accept both "sparse" and "sparse_new" labels for the sparse line
def norm_mode(m):
    return "sparse" if m.startswith("sparse") else m

MODES = ["dense", "sparse"]
COLORS = {"dense": "#1f77b4", "sparse": "#d62728"}
MARKERS = {"dense": "o", "sparse": "s"}

# input_lens to exclude from the plot (16k lacks dense at 4k/8k); CSV unchanged
EXCLUDE_INPUT_LENS = {16384}


def load(path):
    data = defaultdict(lambda: defaultdict(dict))  # mode -> input_len -> output_len -> {metric:val}
    with open(path) as f:
        for d in csv.DictReader(f):
            mode = norm_mode(d["mode"])
            il = int(d["input_len"])
            ol = int(d["output_len"])
            vals = {}
            for col, _, _ in METRICS:
                try:
                    vals[col] = float(d[col])
                except (KeyError, ValueError):
                    vals[col] = None
            data[mode][il][ol] = vals
    return data


def main():
    data = load(CSV)
    input_lens = sorted({il for m in data.values() for il in m} - EXCLUDE_INPUT_LENS)
    n_rows, n_cols = len(METRICS), len(input_lens)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows),
                             squeeze=False)

    for r, (col, title, higher_better) in enumerate(METRICS):
        for c, il in enumerate(input_lens):
            ax = axes[r][c]
            for mode in MODES:
                cells = data.get(mode, {}).get(il, {})
                if not cells:
                    continue
                ols = sorted(cells)
                xs = [ol for ol in ols if cells[ol][col] is not None]
                ys = [cells[ol][col] for ol in ols if cells[ol][col] is not None]
                if not xs:
                    continue
                ax.plot(xs, ys, marker=MARKERS[mode], color=COLORS[mode],
                        label=mode, linewidth=1.8, markersize=6)
            if r == 0:
                ax.set_title(f"input_len = {il // 1024}k", fontsize=11, fontweight="bold")
            if c == 0:
                arrow = "higher better" if higher_better else "lower better"
                ax.set_ylabel(f"{title}\n({arrow})", fontsize=9)
            ax.set_xlabel("output_len", fontsize=8)
            ax.set_xticks([1024, 2048, 4096, 8192])
            ax.set_xticklabels(["1k", "2k", "4k", "8k"], fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")
            if r == 0 and c == 0:
                ax.legend(fontsize=9, loc="best")

    fig.suptitle("GLM_v2 — dense vs sparse (TP8, fp8)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved: {OUT}")
    # sanity: report how many sparse points were plotted
    sp = sum(1 for il in data.get("sparse", {}) for _ in data["sparse"][il])
    dn = sum(1 for il in data.get("dense", {}) for _ in data["dense"][il])
    print(f"points -> dense:{dn} sparse:{sp}")


if __name__ == "__main__":
    main()
