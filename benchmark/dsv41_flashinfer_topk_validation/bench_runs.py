"""Per-run view of a balanced benchmark invocation (tools/run_bench_balanced.sh, six runs).

    python3 bench_runs.py <result dir>

Prints:
  1. the sgl-kernel branch / base ratio of every run on its own, by suite (geomean of
     per-shape median ratios);
  2. for every variant, each run's geomean relative to that variant's six-run mean over
     all 184 shapes, with the slot the variant ran in;
  3. the fixed per-replay cost: bench_topk.py sizes R = round(3000 / p) from an 8-call
     probe p (us per call) and reports replay / R, and both the probe and the timed replay
     hold one graph launch after an idle gap. Solving each cell as 8 p = 8 t + c and
     replay = R t + c (p recovered as 3000 / R) gives the per-call time t and the fixed
     cost c; the table lists each process's median c and median replay length;
  4. sgl-kernel branch / base of the per-call time t by run, and the geomean over runs.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import re
import statistics
import sys

VARIANTS = [
    "sgl-kernel-base",
    "sgl-kernel",
    "sgl-kernel-mode2",
    "flashinfer",
    "flashinfer-small",
    "flashinfer-small-det",
]
FIRST = [0, 1, 5, 2, 4, 3]  # run_bench_balanced.sh's Williams square: run r is FIRST + r - 1
ORDERS = {run: [VARIANTS[(f + run - 1) % 6] for f in FIRST] for run in range(1, 7)}
SHORT = {
    "sgl-kernel-base": "base",
    "sgl-kernel": "branch",
    "sgl-kernel-mode2": "mode 2",
    "flashinfer": "FI",
    "flashinfer-small": "FI small",
    "flashinfer-small-det": "FI small det",
}
SUITES = ("paged", "paged16", "sparse", "blocks")
PROBE_CALLS = 8
TARGET_US = 3000.0


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def solve(repeats, per_call_us):
    """(per-call time t, fixed per-replay cost c) of one cell, in us."""
    probe = TARGET_US / repeats
    replay = per_call_us * repeats
    t = (replay - PROBE_CALLS * probe) / (repeats - PROBE_CALLS)
    return t, PROBE_CALLS * (probe - t)


def main():
    result_dir = sys.argv[1]
    cells = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(result_dir, "run*.jsonl"))):
        m = re.match(r"run(\d)-(.*)\.jsonl$", os.path.basename(path))
        run, variant = int(m.group(1)), m.group(2)
        with open(path) as f:
            for record in map(json.loads, f):
                if "environment" not in record:
                    cells[(run, variant)][(record["suite"], record["label"])] = (
                        record["repeats"],
                        statistics.median(record["samples_us"]),
                    )

    print("sgl-kernel branch / base, each run on its own (suite geomean of per-shape median ratios):")
    print()
    print("| run (process order) | " + " | ".join(SUITES) + " |")
    print("|---|" + "---:|" * len(SUITES))
    for run, order in ORDERS.items():
        base, branch = cells[(run, "sgl-kernel-base")], cells[(run, "sgl-kernel")]
        ratios = [geomean([branch[k][1] / base[k][1] for k in base if k[0] == suite]) for suite in SUITES]
        print(f"| {run} ({', '.join(SHORT[v] for v in order)}) | " + " | ".join(f"{r:.4f}x" for r in ratios) + " |")

    print()
    print("Each variant's run geomean relative to its six-run mean (all 184 shapes), and its slot:")
    print()
    print("| variant | " + " | ".join(f"run {run}" for run in ORDERS) + " |")
    print("|---|" + "---:|" * len(ORDERS))
    for variant in SHORT:
        keys = list(cells[(1, variant)])
        mean = {k: geomean([cells[(run, variant)][k][1] for run in ORDERS]) for k in keys}
        row = []
        for run, order in ORDERS.items():
            rel = geomean([cells[(run, variant)][k][1] / mean[k] for k in keys])
            row.append(f"{100 * (rel - 1):+.2f}% (slot {order.index(variant) + 1})")
        print(f"| {SHORT[variant]} | " + " | ".join(row) + " |")

    print()
    print(
        "Fixed per-replay cost c (median over the cells with R > 8: the fit is singular at R = 8 and "
        "unreliable below, where 3 ms / R is too coarse a stand-in for the probe) and median replay "
        "length per process:"
    )
    print()
    print("| variant | " + " | ".join(f"run {run}" for run in ORDERS) + " |")
    print("|---|" + "---:|" * len(ORDERS))
    for variant in SHORT:
        row = []
        for run in ORDERS:
            values = list(cells[(run, variant)].values())
            c = statistics.median(solve(r, m)[1] for r, m in values if r > PROBE_CALLS)
            replay = statistics.median(r * m for r, m in values) / 1000.0
            row.append(f"{c:.1f} us, {replay:.2f} ms")
        print(f"| {SHORT[variant]} | " + " | ".join(row) + " |")

    print()
    print("sgl-kernel branch / base of the per-call time t (fixed per-replay cost removed):")
    print()
    print("| run | " + " | ".join(SUITES) + " |")
    print("|---|" + "---:|" * len(SUITES))
    per_run = {}
    for run in ORDERS:
        base, branch = cells[(run, "sgl-kernel-base")], cells[(run, "sgl-kernel")]
        per_run[run] = {k: solve(*branch[k])[0] / solve(*base[k])[0] for k in base}
        ratios = [geomean([v for k, v in per_run[run].items() if k[0] == suite]) for suite in SUITES]
        print(f"| {run} | " + " | ".join(f"{r:.4f}x" for r in ratios) + " |")
    overall = [
        geomean([geomean([per_run[run][k] for run in ORDERS]) for k in per_run[1] if k[0] == suite])
        for suite in SUITES
    ]
    print(f"| geomean of runs 1-{len(ORDERS)} | " + " | ".join(f"{r:.4f}x" for r in overall) + " |")


if __name__ == "__main__":
    main()
