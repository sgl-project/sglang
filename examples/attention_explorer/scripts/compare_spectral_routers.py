#!/usr/bin/env python3
"""
Compare Spectral Routers trained on 4B vs 80B models.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery import SpectralRouter


def load_router_stats(router_dir: str):
    """Load router and its metadata."""
    router_path = Path(router_dir)

    # Load metadata
    with open(router_path / "metadata.json") as f:
        metadata = json.load(f)

    # Load router
    router = SpectralRouter.load(str(router_path / "spectral_router.pkl"))

    return router, metadata


def compare_routers(router_4b_dir: str, router_80b_dir: str, output_dir: str):
    """Compare two spectral routers."""

    router_4b, meta_4b = load_router_stats(router_4b_dir)
    router_80b, meta_80b = load_router_stats(router_80b_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Comparison data
    comparison = {
        "4B (Dense)": {
            "n_fingerprints": meta_4b["n_fingerprints"],
            "effective_dimension": meta_4b["effective_dimension"],
            "spectral_gap": meta_4b["spectral_gap"],
            "high_threshold": meta_4b["high_coherence_threshold"],
            "low_threshold": meta_4b["low_coherence_threshold"],
        },
        "80B (MoE)": {
            "n_fingerprints": meta_80b["n_fingerprints"],
            "effective_dimension": meta_80b["effective_dimension"],
            "spectral_gap": meta_80b["spectral_gap"],
            "high_threshold": meta_80b["high_coherence_threshold"],
            "low_threshold": meta_80b["low_coherence_threshold"],
        },
    }

    # Print comparison
    print("\n" + "=" * 70)
    print("SPECTRAL ROUTER COMPARISON: 4B (Dense) vs 80B (MoE)")
    print("=" * 70)

    print("\n### Manifold Structure")
    print(f"{'Metric':<30} {'4B (Dense)':<20} {'80B (MoE)':<20}")
    print("-" * 70)
    print(
        f"{'Fingerprints':<30} {meta_4b['n_fingerprints']:<20,} {meta_80b['n_fingerprints']:<20,}"
    )
    print(
        f"{'Effective Dimension':<30} {meta_4b['effective_dimension']:<20} {meta_80b['effective_dimension']:<20}"
    )
    print(
        f"{'Spectral Gap':<30} {meta_4b['spectral_gap']:<20.6f} {meta_80b['spectral_gap']:<20.6f}"
    )

    print("\n### Calibrated Thresholds")
    print(
        f"{'High Coherence Threshold':<30} {meta_4b['high_coherence_threshold']:<20.3f} {meta_80b['high_coherence_threshold']:<20.3f}"
    )
    print(
        f"{'Low Coherence Threshold':<30} {meta_4b['low_coherence_threshold']:<20.3f} {meta_80b['low_coherence_threshold']:<20.3f}"
    )

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Spectral Router Comparison: 4B (Dense) vs 80B (MoE)",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Effective Dimension comparison
    ax1 = axes[0, 0]
    models = ["4B (Dense)", "80B (MoE)"]
    dims = [meta_4b["effective_dimension"], meta_80b["effective_dimension"]]
    colors = ["#4CAF50", "#2196F3"]
    bars = ax1.bar(models, dims, color=colors, edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Effective Dimension", fontsize=11)
    ax1.set_title("Manifold Complexity", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, dims):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax1.set_ylim(0, max(dims) * 1.2)
    ax1.grid(axis="y", alpha=0.3)

    # 2. Coherence Thresholds
    ax2 = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    high_thresh = [
        meta_4b["high_coherence_threshold"],
        meta_80b["high_coherence_threshold"],
    ]
    low_thresh = [
        meta_4b["low_coherence_threshold"],
        meta_80b["low_coherence_threshold"],
    ]

    bars1 = ax2.bar(
        x - width / 2,
        high_thresh,
        width,
        label="High Threshold",
        color="#FF9800",
        edgecolor="black",
    )
    bars2 = ax2.bar(
        x + width / 2,
        low_thresh,
        width,
        label="Low Threshold",
        color="#9C27B0",
        edgecolor="black",
    )

    ax2.set_ylabel("Coherence Threshold", fontsize=11)
    ax2.set_title("Calibrated Thresholds", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.set_ylim(0.7, 1.0)
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, high_thresh):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars2, low_thresh):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Model Routing Distribution (from validation)
    ax3 = axes[1, 0]
    # Hard-coded from validation runs above
    routing_4b = {"small": 0.10, "medium": 0.28, "large": 0.62}
    routing_80b = {"small": 0.026, "medium": 0.422, "large": 0.552}

    x = np.arange(3)
    width = 0.35
    models_size = ["Small", "Medium", "Large"]
    vals_4b = [routing_4b["small"], routing_4b["medium"], routing_4b["large"]]
    vals_80b = [routing_80b["small"], routing_80b["medium"], routing_80b["large"]]

    bars1 = ax3.bar(
        x - width / 2,
        [v * 100 for v in vals_4b],
        width,
        label="4B (Dense)",
        color="#4CAF50",
        edgecolor="black",
    )
    bars2 = ax3.bar(
        x + width / 2,
        [v * 100 for v in vals_80b],
        width,
        label="80B (MoE)",
        color="#2196F3",
        edgecolor="black",
    )

    ax3.set_ylabel("Routing %", fontsize=11)
    ax3.set_title("Model Size Routing Distribution", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models_size)
    ax3.legend()
    ax3.set_ylim(0, 80)
    ax3.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, vals_4b):
        if val > 0.01:
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for bar, val in zip(bars2, vals_80b):
        if val > 0.01:
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 4. Complexity Distribution
    ax4 = axes[1, 1]
    complexity_4b = {"trivial": 0.178, "moderate": 0.006, "complex": 0.816}
    complexity_80b = {"trivial": 0.174, "moderate": 0.258, "complex": 0.568}

    x = np.arange(3)
    complexity_labels = ["Trivial", "Moderate", "Complex"]
    vals_4b = [
        complexity_4b["trivial"],
        complexity_4b["moderate"],
        complexity_4b["complex"],
    ]
    vals_80b = [
        complexity_80b["trivial"],
        complexity_80b["moderate"],
        complexity_80b["complex"],
    ]

    bars1 = ax4.bar(
        x - width / 2,
        [v * 100 for v in vals_4b],
        width,
        label="4B (Dense)",
        color="#4CAF50",
        edgecolor="black",
    )
    bars2 = ax4.bar(
        x + width / 2,
        [v * 100 for v in vals_80b],
        width,
        label="80B (MoE)",
        color="#2196F3",
        edgecolor="black",
    )

    ax4.set_ylabel("Estimated %", fontsize=11)
    ax4.set_title("Query Complexity Distribution", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(complexity_labels)
    ax4.legend()
    ax4.set_ylim(0, 100)
    ax4.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, vals_4b):
        if val > 0.01:
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for bar, val in zip(bars2, vals_80b):
        if val > 0.01:
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    chart_path = output_path / "spectral_router_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved comparison chart to: {chart_path}")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    dim_diff = meta_80b["effective_dimension"] - meta_4b["effective_dimension"]
    print(
        f"\n1. MANIFOLD COMPLEXITY: 80B has {dim_diff} higher effective dimension ({meta_80b['effective_dimension']} vs {meta_4b['effective_dimension']})"
    )
    print("   -> 80B attention manifold is significantly more complex")
    print("   -> MoE architecture creates richer geometric structure")

    thresh_diff = (
        meta_4b["high_coherence_threshold"] - meta_80b["high_coherence_threshold"]
    )
    print(f"\n2. COHERENCE THRESHOLDS: 4B thresholds are {thresh_diff:.3f} higher")
    print(
        f"   -> 4B: [{meta_4b['low_coherence_threshold']:.3f}, {meta_4b['high_coherence_threshold']:.3f}]"
    )
    print(
        f"   -> 80B: [{meta_80b['low_coherence_threshold']:.3f}, {meta_80b['high_coherence_threshold']:.3f}]"
    )
    print("   -> 4B fingerprints are more tightly clustered (narrower range)")

    print(f"\n3. ROUTING DIFFERENCES:")
    print(
        f"   -> 4B routes {routing_4b['small']*100:.1f}% to small vs 80B's {routing_80b['small']*100:.1f}%"
    )
    print(
        f"   -> 4B routes {routing_4b['large']*100:.1f}% to large vs 80B's {routing_80b['large']*100:.1f}%"
    )
    print("   -> 4B has more extreme routing (small or large)")
    print("   -> 80B has more balanced distribution")

    print(f"\n4. COMPLEXITY PERCEPTION:")
    print(
        f"   -> 4B sees {complexity_4b['complex']*100:.1f}% as complex vs 80B's {complexity_80b['complex']*100:.1f}%"
    )
    print(
        f"   -> 4B sees only {complexity_4b['moderate']*100:.1f}% as moderate vs 80B's {complexity_80b['moderate']*100:.1f}%"
    )
    print("   -> Dense model has binary complexity view (trivial vs complex)")
    print("   -> MoE model recognizes intermediate complexity")

    # Save comparison JSON
    comparison_data = {
        "models": {
            "4B": {
                "architecture": "Dense",
                "params": "4B",
                "effective_dimension": meta_4b["effective_dimension"],
                "high_threshold": meta_4b["high_coherence_threshold"],
                "low_threshold": meta_4b["low_coherence_threshold"],
                "routing": routing_4b,
                "complexity": complexity_4b,
                "coherence_mean": 0.922,
                "coherence_std": 0.032,
                "cot_rate": 0.90,
            },
            "80B": {
                "architecture": "MoE",
                "params": "80B (17B active)",
                "effective_dimension": meta_80b["effective_dimension"],
                "high_threshold": meta_80b["high_coherence_threshold"],
                "low_threshold": meta_80b["low_coherence_threshold"],
                "routing": routing_80b,
                "complexity": complexity_80b,
                "coherence_mean": 0.855,
                "coherence_std": 0.060,
                "cot_rate": 0.936,
            },
        },
        "insights": [
            "80B manifold has 77% higher effective dimension (39 vs 22)",
            "4B fingerprints are more tightly clustered (higher coherence)",
            "4B has binary complexity view; 80B recognizes gradations",
            "MoE architecture creates richer attention geometry",
        ],
    }

    with open(output_path / "spectral_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\nSaved comparison data to: {output_path / 'spectral_comparison.json'}")

    return comparison_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare spectral routers")
    parser.add_argument(
        "--router-4b", default="./router_4b", help="4B router directory"
    )
    parser.add_argument(
        "--router-80b", default="./router_80b", help="80B router directory"
    )
    parser.add_argument(
        "--output", default="./spectral_comparison", help="Output directory"
    )

    args = parser.parse_args()
    compare_routers(args.router_4b, args.router_80b, args.output)
