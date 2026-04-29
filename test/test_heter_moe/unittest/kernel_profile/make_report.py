"""Generate final Task 3 summary: x*(M_global) and T*(M_global) from
measurement-based optimum (NOT prediction). Also writes a clean CSV and
a Markdown table.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import KERN_E, KERN_TOP_K, make_zipf_routing  # noqa: E402

import torch  # noqa: E402

OUT_DIR = "test/test_heter_moe/unittest/kernel_profile/results"

# Read measurements
def read_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        return header, [row for row in r]


hdr, rows = read_csv(f"{OUT_DIR}/optimal_assignment.csv")
hi = {k: i for i, k in enumerate(hdr)}

ax_hdr, ax_rows = read_csv(f"{OUT_DIR}/optimal_assignment.all_x.csv")
ax = {}  # (M, x) -> (lat_ms, tile_key)
for r in ax_rows:
    M = int(r[0]); x = int(r[1]); lat = float(r[2]); tile = r[3]
    ax[(M, x)] = (lat, tile)

X_CANDIDATES = sorted({k[1] for k in ax})

# Recompute T* (per-expert load threshold) at x_star_meas using the same
# Zipf routing as Task 3 used (seed=42).
final_rows = []
for r in rows:
    M = int(r[0]); x_meas = int(r[hi["x_star_meas"]])
    t_meas_star = float(r[hi["t_meas_at_meas_x_ms"]])

    # Recover routing distribution
    _, _, _, _, expert_freq = make_zipf_routing(M, torch.device("cpu"), seed=42)
    counts = expert_freq.cpu().numpy()
    order = np.argsort(-counts)

    if x_meas == 0:
        T_raw = float(counts.max() + 1)  # nothing promoted; threshold above all
    elif x_meas == KERN_E:
        T_raw = 0.0
    else:
        T_raw = (counts[order[x_meas - 1]] + counts[order[x_meas]]) / 2.0
    T_snap = int(round(T_raw / 8) * 8)

    # Latency at x=0 (pure INT4 baseline) for speedup column
    lat_x0 = ax[(M, 0)][0] if (M, 0) in ax else float("nan")
    speedup = lat_x0 / t_meas_star if t_meas_star > 0 else 1.0

    # Tile used at the measurement-based x*
    tile_key = ax.get((M, x_meas), (0, "n/a"))[1]

    final_rows.append({
        "M_global": M,
        "x_star": x_meas,
        "T_star_raw": T_raw,
        "T_star_snapped": T_snap,
        "t_layer_ms": t_meas_star,
        "tile_key": tile_key,
        "t_pure_int4_ms": lat_x0,
        "speedup_vs_pure_int4": speedup,
    })

# CSV output
csv_path = f"{OUT_DIR}/x_star_curve.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "M_global", "x_star", "T_star_raw", "T_star_snapped",
        "t_layer_ms", "tile_key", "t_pure_int4_ms", "speedup_vs_pure_int4",
    ])
    for r in final_rows:
        w.writerow([
            r["M_global"], r["x_star"],
            f"{r['T_star_raw']:.1f}", r["T_star_snapped"],
            f"{r['t_layer_ms']:.4f}", r["tile_key"],
            f"{r['t_pure_int4_ms']:.4f}",
            f"{r['speedup_vs_pure_int4']:.3f}",
        ])
print(f"wrote {csv_path}")

# Markdown table output
md_path = f"{OUT_DIR}/x_star_curve.md"
with open(md_path, "w") as f:
    f.write("# Optimal BF16/INT4 expert assignment vs. global batch size\n\n")
    f.write("Qwen3-30B-A3B (E=128, K=2048, N=768, top_k=8) on A100-SXM4-80GB.\n")
    f.write("Routing: synthetic Zipf (alpha=1.1, seed=42) per "
            "`scripts/heter_moe_collect_routing.py:generate_synthetic_routing`.\n")
    f.write("Promotion policy: top-x experts by routing frequency → BF16; "
            "rest → INT4 (Marlin).\n")
    f.write("BF16 tile pinned per-cell from autotuned "
            "`results/bf16_sparse_configs.json`.\n")
    f.write("Latency = paired sparse-active kernel call "
            "(fused_marlin_moe + outplace_fused_experts), median-of-50 with "
            "L2 flush + CUDA graph.\n\n")
    f.write(
        "| M_global | x* (#BF16) | T* (snap) | t_layer (ms) | "
        "BF16 tile | t_pure_int4 (ms) | speedup |\n"
    )
    f.write(
        "|---:|---:|---:|---:|:---|---:|---:|\n"
    )
    for r in final_rows:
        f.write(
            f"| {r['M_global']} | {r['x_star']} | {r['T_star_snapped']} | "
            f"{r['t_layer_ms']:.4f} | `{r['tile_key']}` | "
            f"{r['t_pure_int4_ms']:.4f} | {r['speedup_vs_pure_int4']:.3f}× |\n"
        )
    f.write("\n## Reading the table\n\n")
    f.write(
        "- **x\\***: optimal number of experts to keep in BF16 at this "
        "M_global. The remaining 128−x* are kept in INT4.\n"
        "- **T\\* (snap)**: threshold (tokens/expert) snapped to the nearest "
        "multiple of 8 — the runtime knob `bf16_promotion_threshold` would "
        "be set to this value to promote the top-x* experts under this "
        "routing distribution at this M_global.\n"
        "- **speedup**: latency reduction vs. pure-INT4 (x=0) at this M_global.\n"
    )
print(f"wrote {md_path}")

# Stdout pretty-print
print()
print("=" * 100)
print(f"{'M_global':>8} {'x*':>4} {'T*':>5} {'t_layer(ms)':>12} "
      f"{'tile':<14} {'t_int4(ms)':>11} {'speedup':>8}")
print("-" * 100)
for r in final_rows:
    print(
        f"{r['M_global']:>8} {r['x_star']:>4} {r['T_star_snapped']:>5} "
        f"{r['t_layer_ms']:>12.4f} {r['tile_key']:<14} "
        f"{r['t_pure_int4_ms']:>11.4f} {r['speedup_vs_pure_int4']:>7.3f}x"
    )
print("=" * 100)
