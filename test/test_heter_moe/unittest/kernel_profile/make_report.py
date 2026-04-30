"""Generate the final Task 3 report: a (M_global × N_active) speedup matrix.

Each row is one M_global. Each of the 16 inner columns is the speedup of
running x = N_active BF16-promoted experts (top-x by routing frequency)
over the pure-INT4 (x=0) baseline at the same M_global. Final 4 columns:
  winner_x, best_speedup, t_int4_pure_ms, t_at_winner_ms.

Inputs:
  results/optimal_assignment.all_x.csv (M_global, x, lat_ms, tile_key)

Outputs:
  results/x_star_curve.csv  (one row per M_global, full matrix)
  results/x_star_curve.md   (Markdown table for sharing)
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import KERN_E, make_zipf_routing  # noqa: E402

import torch  # noqa: E402

OUT_DIR = "test/test_heter_moe/unittest/kernel_profile/results"

# Columns: 16 BF16 promotion levels {4, 8, 12, ..., 64}
N_ACTIVE_COLS = list(range(4, 64 + 1, 4))


def read_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        return header, [row for row in r]


# Load all_x measurements: (M, x) -> (lat_ms, tile_key)
_, ax_rows = read_csv(f"{OUT_DIR}/optimal_assignment.all_x.csv")
ax = {}
for r in ax_rows:
    M = int(r[0]); x = int(r[1]); lat = float(r[2]); tile = r[3]
    ax[(M, x)] = (lat, tile)

M_values = sorted({k[0] for k in ax})

# Per-row:
#   speedup_at_x = t_pure_int4(M) / t_x(M)   (>1 means promotion helps)
#   winner_x = argmax over X cols
#   best_speedup = speedup at winner
report_rows = []
for M in M_values:
    if (M, 0) not in ax:
        # No baseline — can't compute speedup. Skip.
        continue
    t_int4 = ax[(M, 0)][0]

    speedups = {}
    for x in N_ACTIVE_COLS:
        if (M, x) in ax:
            t_x = ax[(M, x)][0]
            speedups[x] = t_int4 / t_x if t_x > 0 else 1.0

    if not speedups:
        continue
    winner_x = max(speedups, key=speedups.get)
    best_speedup = speedups[winner_x]
    t_winner = ax[(M, winner_x)][0]

    # Threshold T* derived from winner_x using same Zipf routing
    _, _, _, _, expert_freq = make_zipf_routing(M, torch.device("cpu"), seed=42)
    counts = expert_freq.cpu().numpy()
    order = np.argsort(-counts)
    if winner_x == 0:
        T_raw = float(counts.max() + 1)
    elif winner_x == KERN_E:
        T_raw = 0.0
    else:
        T_raw = (counts[order[winner_x - 1]] + counts[order[winner_x]]) / 2.0
    T_snap = int(round(T_raw / 8) * 8)

    row = {
        "M_global": M,
        "speedups": speedups,
        "winner_x": winner_x,
        "T_star_snapped": T_snap,
        "best_speedup": best_speedup,
        "t_int4_ms": t_int4,
        "t_winner_ms": t_winner,
        "winner_tile": ax[(M, winner_x)][1] if winner_x > 0 else "n/a",
    }
    report_rows.append(row)

# CSV
csv_path = f"{OUT_DIR}/x_star_curve.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    header = ["M_global"] + [f"x{n}" for n in N_ACTIVE_COLS] + [
        "winner_x", "T_star", "best_speedup", "t_int4_pure_ms", "t_winner_ms",
        "winner_tile",
    ]
    w.writerow(header)

    def _signed_ratio(s):
        # +0.241 = heter 24.1% faster ; -0.277 = heter 27.7% slower
        return f"{s - 1.0:+.3f}"

    for r in report_rows:
        row_out = [r["M_global"]]
        for n in N_ACTIVE_COLS:
            row_out.append(_signed_ratio(r['speedups'].get(n, float('nan'))))
        row_out.extend([
            r["winner_x"], r["T_star_snapped"],
            _signed_ratio(r['best_speedup']),
            f"{r['t_int4_ms']:.4f}", f"{r['t_winner_ms']:.4f}",
            r["winner_tile"],
        ])
        w.writerow(row_out)
print(f"wrote {csv_path}")

# Markdown
md_path = f"{OUT_DIR}/x_star_curve.md"
with open(md_path, "w") as f:
    f.write("# Heter-MoE speedup matrix vs. pure INT4\n\n")
    f.write("Qwen3-30B-A3B (E=128, K=2048, N=768, top_k=8) on A100-SXM4-80GB.\n")
    f.write("Routing: synthetic Zipf (alpha=1.1, seed=42).\n")
    f.write("Promotion policy: top-x experts by routing frequency → BF16; "
            "rest → INT4 (Marlin).\n")
    f.write("BF16 path uses **separately autotuned up + down tiles** "
            "(`bf16_sparse_configs_sep.json`), pinned via override.\n")
    f.write("INT4 path: Marlin with its built-in heuristic.\n")
    f.write("M_global rows are the 11 production-tuned batch sizes "
            "(`E=128,N=768,A100-80GB.json` keys) so both paths run on "
            "their best-tuned tile.\n")
    f.write("Each cell `xN` = `t_pure_int4 / t_at_x_N` (speedup, ≥1 means "
            "promoting N experts to BF16 helps).\n\n")

    # Header
    cells = ["M_global"] + [f"x{n}" for n in N_ACTIVE_COLS]
    cells += ["**x\\***", "**T\\***", "**best**",
              "**t_int4(ms)**", "**t_winner(ms)**"]
    f.write("| " + " | ".join(cells) + " |\n")
    f.write("|" + "---:|" * len(cells) + "\n")

    def _signed(s):
        d = (s - 1.0) * 100
        sign = "+" if d >= 0 else "-"
        return f"{sign}{abs(d):.1f}%"

    for r in report_rows:
        cells = [str(r["M_global"])]
        for n in N_ACTIVE_COLS:
            s = r["speedups"].get(n, float("nan"))
            cells.append(_signed(s))
        cells.extend([
            f"**{r['winner_x']}**",
            f"**{r['T_star_snapped']}**",
            f"**{_signed(r['best_speedup'])}**",
            f"{r['t_int4_ms']:.4f}",
            f"{r['t_winner_ms']:.4f}",
        ])
        f.write("| " + " | ".join(cells) + " |\n")

    f.write("\n## Reading\n\n")
    f.write("- **x* (winner)**: optimal number of experts promoted to BF16 "
            "at this M_global.\n")
    f.write("- **T\\***: per-expert-load threshold (snap-to-8) — set "
            "`bf16_promotion_threshold = T*` in the runtime config to "
            "promote the same set under this routing distribution at this "
            "M_global.\n")
    f.write("- **best speedup**: t_pure_int4 / t_at_winner_x. <1.0 means "
            "the heter-MoE path is slower than pure INT4 at this M_global "
            "(no useful promotion).\n")
    f.write("- Inner cells (`x4`..`x64`): each is the measured speedup at "
            "that specific N_active. Lets you see the shape of the curve "
            "across promotion levels per row.\n")
print(f"wrote {md_path}")

# Stdout pretty-print: collapsed (winner-only) summary
print()
print("=" * 100)
print(f"{'M_global':>8} {'x*':>4} {'T*':>5} {'best speedup':>13} "
      f"{'t_int4(ms)':>12} {'t_winner(ms)':>13} {'winner_tile':<14}")
print("-" * 100)
for r in report_rows:
    print(
        f"{r['M_global']:>8} {r['winner_x']:>4} {r['T_star_snapped']:>5} "
        f"{r['best_speedup']:>12.3f}x "
        f"{r['t_int4_ms']:>12.4f} {r['t_winner_ms']:>13.4f} "
        f"{r['winner_tile']:<14}"
    )
print("=" * 100)
