"""Analyze benchmark results to determine best configs per head_dim."""
import json
from pathlib import Path
import statistics

results = json.loads((Path(__file__).parent / "results.json").read_text())

# Separate by head_dim
for hd in [64, 128]:
    print(f"\n{'='*60}")
    print(f"HEAD_DIM = {hd}")
    print(f"{'='*60}")

    subset = [r for r in results if r["head_dim"] == hd]
    baseline_key = "w8_s1_m128_n128"

    # Per-config average speedup
    config_speedups = {}
    for r in subset:
        cfg = r["config"]
        baseline = [x for x in subset
                    if x["config"] == baseline_key
                    and x["seq_len"] == r["seq_len"]
                    and x["num_heads"] == r["num_heads"]]
        if baseline:
            speedup = baseline[0]["time_ms"] / r["time_ms"]
            config_speedups.setdefault(cfg, []).append(speedup)

    print("\nConfigs sorted by geomean speedup:")
    for cfg, speedups in sorted(config_speedups.items(), key=lambda x: -statistics.geometric_mean(x[1]))[:10]:
        gmean = statistics.geometric_mean(speedups)
        print(f"  {cfg}: {gmean:.3f}x (min={min(speedups):.3f}x, max={max(speedups):.3f}x)")

    # Also show per-size detail for top 3 configs
    top3 = sorted(config_speedups.items(), key=lambda x: -statistics.geometric_mean(x[1]))[:3]
    print("\nDetail for top 3:")
    for cfg, _ in top3:
        print(f"\n  {cfg}:")
        for r in sorted([x for x in subset if x["config"] == cfg], key=lambda x: (x["seq_len"], x["num_heads"])):
            baseline = [x for x in subset
                        if x["config"] == baseline_key
                        and x["seq_len"] == r["seq_len"]
                        and x["num_heads"] == r["num_heads"]]
            if baseline:
                speedup = baseline[0]["time_ms"] / r["time_ms"]
                print(f"    seq={r['seq_len']:5d} heads={r['num_heads']:2d}: {r['time_ms']:.4f}ms (baseline: {baseline[0]['time_ms']:.4f}ms, speedup: {speedup:.2f}x)")

    # Check: is there a single config that never regresses?
    print("\nConfigs with no regression (speedup >= 0.98 everywhere):")
    for cfg, speedups in sorted(config_speedups.items(), key=lambda x: -statistics.geometric_mean(x[1])):
        if min(speedups) >= 0.98:
            gmean = statistics.geometric_mean(speedups)
            print(f"  {cfg}: {gmean:.3f}x (min={min(speedups):.3f}x)")
