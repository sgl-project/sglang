"""Layer-wise HeterMoE evaluation using routing stats + kernel benchmark data.

Simulates per-layer speedup from heterogeneous precision assignment using:
1. Real/synthetic routing stats (token counts per expert per layer)
2. Kernel benchmark data (latency vs batch size for each precision)

Usage: python3 scripts/heter_moe_benchmark_layers.py
"""

import csv
import glob
import json
import os

import numpy as np

ROUTING_DIR = "/data/heter-moe/routing_stats"
KERNEL_CSV = "/data/heter-moe/profiles/groupgemm/kernel_benchmark.csv"
OUT_DIR = "/data/heter-moe/profiles/layerwise"

NUM_EXPERTS = 128
NUM_LAYERS = 48
COLD_RATIO = 0.8


def load_kernel_benchmarks(csv_path):
    """Load kernel benchmark CSV into lookup: {(kernel, M) -> latency_ms}."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["kernel"], int(row["tokens"]))
            data[key] = float(row["latency_ms"])
    return data


def interpolate_latency(kernel_data, kernel_name, M):
    """Linear interpolation of kernel latency for arbitrary M."""
    points = sorted([(k[1], v) for k, v in kernel_data.items() if k[0] == kernel_name])
    if M <= points[0][0]:
        return points[0][1]
    if M >= points[-1][0]:
        # Extrapolate linearly from last two points
        m1, l1 = points[-2]
        m2, l2 = points[-1]
        return l1 + (l2 - l1) * (M - m1) / (m2 - m1)
    for i in range(len(points) - 1):
        m1, l1 = points[i]
        m2, l2 = points[i + 1]
        if m1 <= M <= m2:
            t = (M - m1) / (m2 - m1)
            return l1 + t * (l2 - l1)
    return points[-1][1]


def estimate_layer_latency_uniform(kernel_data, kernel_name, total_tokens):
    """Estimate MoE layer latency assuming uniform token distribution."""
    tokens_per_expert = total_tokens / NUM_EXPERTS
    return interpolate_latency(kernel_data, kernel_name, max(1, tokens_per_expert))


def estimate_layer_latency_heter(kernel_data, expert_counts, cold_ratio=COLD_RATIO):
    """Estimate HeterMoE layer latency using per-expert routing stats.

    Cold experts (lowest token count, cold_ratio fraction) use a8w8.
    Hot experts (highest token count) use a16w16.
    Total latency = max(cold_kernel_time, hot_kernel_time) [they run sequentially].
    """
    counts = np.array(expert_counts)
    sorted_idx = np.argsort(counts)

    n_cold = int(round(cold_ratio * NUM_EXPERTS))
    cold_experts = sorted_idx[:n_cold]
    hot_experts = sorted_idx[n_cold:]

    cold_total = counts[cold_experts].sum()
    hot_total = counts[hot_experts].sum()

    # Average tokens per expert in each group
    cold_avg_m = cold_total / max(len(cold_experts), 1)
    hot_avg_m = hot_total / max(len(hot_experts), 1)

    # Latency: each group runs its kernel on its subset
    # Approximate by using the average M per expert for that group
    cold_lat = interpolate_latency(kernel_data, "a8w8", max(1, cold_avg_m))
    hot_lat = interpolate_latency(kernel_data, "a16w16", max(1, hot_avg_m))

    return cold_lat + hot_lat, cold_lat, hot_lat


def main():
    kernel_data = load_kernel_benchmarks(KERNEL_CSV)
    os.makedirs(OUT_DIR, exist_ok=True)

    routing_files = sorted(glob.glob(os.path.join(ROUTING_DIR, "batch*_prefill.json")))
    if not routing_files:
        print(
            f"No routing stats found in {ROUTING_DIR}. Run heter_moe_collect_routing.py first."
        )
        return

    results = []
    print(
        f"{'Batch':>6} {'Phase':>8} {'Layer':>6} {'BF16(ms)':>10} {'Heter(ms)':>10} {'Speedup':>8}"
    )
    print("-" * 60)

    for rf in routing_files:
        fname = os.path.basename(rf)
        batch_size = int(fname.split("_")[0].replace("batch", ""))
        phase = fname.split("_")[1].replace(".json", "")

        with open(rf) as f:
            routing = json.load(f)

        layer_speedups = []
        for layer_idx in range(NUM_LAYERS):
            key = f"transformer_block_{layer_idx}"
            if key not in routing:
                continue
            counts = routing[key]
            total_tokens = sum(counts)

            bf16_lat = estimate_layer_latency_uniform(
                kernel_data, "a16w16", total_tokens
            )
            heter_lat, cold_lat, hot_lat = estimate_layer_latency_heter(
                kernel_data, counts
            )

            speedup = bf16_lat / heter_lat if heter_lat > 0 else 1.0
            layer_speedups.append(speedup)
            results.append(
                [
                    batch_size,
                    phase,
                    layer_idx,
                    f"{bf16_lat:.3f}",
                    f"{heter_lat:.3f}",
                    f"{speedup:.3f}",
                ]
            )

        avg_speedup = np.mean(layer_speedups) if layer_speedups else 1.0
        print(
            f"{batch_size:>6} {phase:>8} {'avg':>6} "
            f"{'':>10} {'':>10} {avg_speedup:>7.3f}x"
        )

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "layerwise_eval.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_size", "phase", "layer", "bf16_ms", "heter_ms", "speedup"])
        w.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    # Plot: average speedup vs batch size
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        batch_sizes = sorted(set(int(r[0]) for r in results))
        avg_speedups = []
        for bs in batch_sizes:
            spds = [float(r[5]) for r in results if int(r[0]) == bs]
            avg_speedups.append(np.mean(spds))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(batch_sizes, avg_speedups, "o-", linewidth=2, markersize=6)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Average Speedup (HeterMoE / BF16)")
        ax.set_title(
            f"Layer-wise HeterMoE Speedup (cold={COLD_RATIO:.0%} a8w8, hot={1 - COLD_RATIO:.0%} a16w16)"
        )
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(OUT_DIR, "layerwise_speedup.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved: {plot_path}")

        # Imbalance plot: token distribution for one batch
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        sample_batches = [4, 32, 128, 1024]
        for ax, bs in zip(axes.flat, sample_batches):
            fpath = os.path.join(ROUTING_DIR, f"batch{bs}_prefill.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    rd = json.load(f)
                counts = rd.get("transformer_block_0", [0] * NUM_EXPERTS)
                sorted_counts = sorted(counts, reverse=True)
                ax.bar(range(NUM_EXPERTS), sorted_counts, width=1.0, alpha=0.7)
                ax.set_title(f"Batch={bs}, Layer 0")
                ax.set_xlabel("Expert (sorted by load)")
                ax.set_ylabel("Token count")

        plt.suptitle("Expert Load Imbalance (Synthetic Zipf)", fontsize=14)
        plt.tight_layout()
        plot_path2 = os.path.join(OUT_DIR, "expert_imbalance.png")
        plt.savefig(plot_path2, dpi=150)
        print(f"Plot saved: {plot_path2}")

    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
