"""Paper main figure: agent-serving collapse envelope, BF16 vs MXFP4.

Two panels (small multiples, one shared x-axis semantic = concurrent sessions):
  (a) continuation-turn miss rate (%) vs N
  (b) continuation TTFT p50 (s, log) vs N
Vertical dashed lines mark the 1.0x oversubscription point per dtype
(pool size / mean live bytes per session).

Usage: python plot_envelope.py [results_dir] [out_prefix]
"""
import glob
import json
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito CVD-safe pair (validated: CVD dE 21.9, normal 31.2)
C_BF16 = "#0072B2"
C_FP4 = "#D55E00"

POOL_TOKENS = {"bf16": 92495, "fp4": 279685}


def stats(path):
    recs = [json.loads(l) for l in open(path)]
    def miss(r):
        return r["prompt_cont"] and r["ttft_cont_ms"] > 0.1 * r["prompt_cont"] + 200
    n = len(recs)
    return (
        sum(1 for r in recs if miss(r)) / n * 100,
        statistics.median([r["ttft_cont_ms"] for r in recs]) / 1000.0,
    )


def collect(results_dir, dtype):
    """Returns sorted [(N, miss%, ttft_p50_s)] for the LRU runs of a dtype."""
    pts = []
    for f in glob.glob(os.path.join(results_dir, f"{dtype}_lru_n*.jsonl")):
        n = int(f.rsplit("_n", 1)[1].split(".")[0])
        m, t = stats(f)
        pts.append((n, m, t))
    return sorted(pts)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "results")
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "envelope")

    # sweep series = plain LRU runs (no _lat suffix, no seed prefix)
    pat = {
        "bf16": [re.compile(r"^(?:v3|d4_bf16)_lru_n(\d+)\.jsonl$"),
                 re.compile(r"^bf16_lru_n(\d+)\.jsonl$")],
        "fp4": [re.compile(r"^(?:d4_)?fp4_lru_n(\d+)\.jsonl$")],
    }

    def dedup(pts):
        d = {}
        for p in pts:
            d[p[0]] = p
        return [d[k] for k in sorted(d)]

    def series(dtype):
        pts = []
        for f in glob.glob(os.path.join(results_dir, "*.jsonl")):
            base = os.path.basename(f)
            if any((m := p.match(base)) for p in pat[dtype]):
                pts.append((int(m.group(1)),) + stats(f))
        return dedup(pts)

    bf16 = series("bf16")
    fp4 = series("fp4")
    print("bf16 points:", bf16)
    print("fp4  points:", fp4)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, (yi, ylab) in zip(axes, [(1, "Continuation-turn miss rate (%)"),
                                     (2, "Continuation TTFT p50 (s)")]):
        ax.plot([p[0] for p in bf16], [p[yi] for p in bf16], "-o", color=C_BF16,
                lw=2, ms=5, label="BF16 KV (92.5k tok pool)")
        ax.plot([p[0] for p in fp4], [p[yi] for p in fp4], "-s", color=C_FP4,
                lw=2, ms=5, label="MXFP4 KV (280k tok pool)")
        ax.set_xlabel("Concurrent agent sessions")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.05, 60)
    # 1.0x oversubscription boundaries (~6.4k mean live tokens per session)
    for ax in axes:
        ax.axvline(92495 / 6400, color=C_BF16, ls="--", lw=1, alpha=0.5)
        ax.axvline(279685 / 6400, color=C_FP4, ls="--", lw=1, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=200)
    fig.savefig(out_prefix + ".pdf")
    print("wrote", out_prefix + ".png/.pdf")


if __name__ == "__main__":
    main()
