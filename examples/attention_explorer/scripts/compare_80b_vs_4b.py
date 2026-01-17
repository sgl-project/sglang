#!/usr/bin/env python3
"""Compare attention patterns between Qwen3-80B and Qwen3-4B models."""

import matplotlib.pyplot as plt
import numpy as np

# Model data from explorations
DATA = {
    "Qwen3-80B (MoE)": {
        "db": "./exploration_fingerprints.db",
        "output": "./exploration_outputs",
        "duration_hours": 4.06,
        "total_tokens": 129254,
        "total_prompts": 96,
        "discovery_runs": 8,
        "zones": {
            "structure_ripple": 84674,
            "semantic_bridge": 43565,
            "syntax_floor": 1015,
        },
        "throughput_range": "10-10.8 tok/s",
        "params": "80B total, 3B active",
    },
    "Qwen3-4B (Dense)": {
        "db": "./exploration_4b_fingerprints.db",
        "output": "./exploration_4b_outputs",
        "duration_hours": 1.01,
        "total_tokens": 109346,
        "total_prompts": 75,
        "discovery_runs": 4,
        "zones": {
            "structure_ripple": 92154,
            "semantic_bridge": 16910,
            "syntax_floor": 282,
        },
        "throughput_range": "25-52 tok/s",
        "params": "4B",
    },
}


def calculate_percentages(zones):
    """Calculate zone percentages."""
    total = sum(zones.values())
    return {k: (v / total) * 100 for k, v in zones.items()}


def create_comparison_charts():
    """Create comparison visualizations."""
    fig = plt.figure(figsize=(16, 12))

    # Color scheme
    colors = {
        "structure_ripple": "#ff6b6b",
        "semantic_bridge": "#4ecdc4",
        "syntax_floor": "#45b7d1",
    }

    models = list(DATA.keys())

    # 1. Zone Distribution Bar Chart
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(models))
    width = 0.25

    for i, zone in enumerate(["structure_ripple", "semantic_bridge", "syntax_floor"]):
        percentages = [calculate_percentages(DATA[m]["zones"])[zone] for m in models]
        ax1.bar(
            x + i * width,
            percentages,
            width,
            label=zone.replace("_", " ").title(),
            color=colors[zone],
            edgecolor="black",
            linewidth=0.5,
        )

    ax1.set_ylabel("Percentage (%)", fontsize=12)
    ax1.set_title("Zone Distribution Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.3)

    # Add percentage labels on bars
    for i, zone in enumerate(["structure_ripple", "semantic_bridge", "syntax_floor"]):
        percentages = [calculate_percentages(DATA[m]["zones"])[zone] for m in models]
        for j, pct in enumerate(percentages):
            if pct > 3:  # Only show label if bar is tall enough
                ax1.annotate(
                    f"{pct:.1f}%",
                    xy=(j + i * width, pct + 1),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # 2. Pie Charts Side by Side
    ax2 = fig.add_subplot(2, 2, 2)

    # Create two pie charts side by side using subplots within subplot
    ax2.set_visible(False)

    ax2a = fig.add_axes([0.55, 0.55, 0.18, 0.35])
    ax2b = fig.add_axes([0.75, 0.55, 0.18, 0.35])

    for ax, model in [(ax2a, models[0]), (ax2b, models[1])]:
        pcts = calculate_percentages(DATA[model]["zones"])
        sizes = [
            pcts["structure_ripple"],
            pcts["semantic_bridge"],
            pcts["syntax_floor"],
        ]
        labels = ["Struct. Ripple", "Semantic Bridge", "Syntax Floor"]
        cols = [
            colors["structure_ripple"],
            colors["semantic_bridge"],
            colors["syntax_floor"],
        ]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
            colors=cols,
            startangle=90,
        )
        ax.set_title(model.split()[0], fontsize=11, fontweight="bold")

    # Add legend
    fig.legend(wedges, labels, loc="upper right", bbox_to_anchor=(0.98, 0.52))

    # 3. Throughput and Efficiency
    ax3 = fig.add_subplot(2, 2, 3)

    metrics = ["Tokens Generated", "Prompts Processed", "Discovery Runs"]
    values_80b = [
        DATA[models[0]]["total_tokens"],
        DATA[models[0]]["total_prompts"],
        DATA[models[0]]["discovery_runs"] * 10000,
    ]  # Scale for visibility
    values_4b = [
        DATA[models[1]]["total_tokens"],
        DATA[models[1]]["total_prompts"],
        DATA[models[1]]["discovery_runs"] * 10000,
    ]

    x = np.arange(3)
    width = 0.35

    bars1 = ax3.bar(
        x - width / 2,
        values_80b,
        width,
        label=models[0],
        color="#3498db",
        edgecolor="black",
    )
    bars2 = ax3.bar(
        x + width / 2,
        values_4b,
        width,
        label=models[1],
        color="#e74c3c",
        edgecolor="black",
    )

    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Data Collection Comparison", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Tokens", "Prompts", "Disc. Runs (x10K)"])
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars, vals in [(bars1, values_80b), (bars2, values_4b)]:
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            label = f"{val:,}" if val < 1000 else f"{val/1000:.0f}K"
            ax3.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 4. Key Findings Summary Table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    summary_text = """
    KEY FINDINGS: 80B vs 4B Attention Patterns

    ZONE DISTRIBUTION DIFFERENCE:
    ------------------------------------------
    Zone              | 80B (MoE) | 4B (Dense)
    ------------------------------------------
    structure_ripple  |   65.5%   |   84.3%   (+18.8%)
    semantic_bridge   |   33.7%   |   15.5%   (-18.2%)
    syntax_floor      |    0.8%   |    0.3%   (-0.5%)
    ------------------------------------------

    INSIGHTS:

    1. Dense 4B relies MORE on structure_ripple (84% vs 65%)
       - Long-range structural patterns dominate
       - May indicate simpler attention mechanisms

    2. MoE 80B uses MORE semantic_bridge (34% vs 15%)
       - More balanced mid-range attention
       - Expert routing enables nuanced patterns

    3. Both models minimize syntax_floor (<1%)
       - Short-range local patterns rarely used
       - Modern architectures favor global context

    4. Throughput: 4B is 3-5x faster (25-52 vs 10 tok/s)
       - Same hardware, similar total tokens

    IMPLICATIONS FOR ROUTING:
    - 4B: Better for tasks needing structural coherence
    - 80B: Better for tasks needing semantic reasoning
    """

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(
        "./exploration_outputs/80b_vs_4b_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print("Comparison chart saved to: ./exploration_outputs/80b_vs_4b_comparison.png")


def create_comparison_report():
    """Generate a markdown comparison report."""
    report = """# Qwen3 Model Comparison: 80B (MoE) vs 4B (Dense)

**Date**: 2026-01-10
**Comparison Purpose**: Analyze attention pattern differences between model sizes

## Executive Summary

This comparison reveals significant architectural differences in attention patterns between
the Qwen3-80B Mixture-of-Experts (MoE) model and the dense Qwen3-4B model.

### Key Finding
**The 4B model relies ~20% more heavily on structure_ripple patterns than the 80B model**,
suggesting that smaller dense models compensate for fewer parameters by using more
long-range structural attention, while larger MoE models achieve a more balanced
distribution through expert routing.

## Model Specifications

| Specification | Qwen3-80B (MoE) | Qwen3-4B (Dense) |
|--------------|-----------------|------------------|
| Architecture | Mixture of Experts | Dense |
| Parameters | 80B total, 3B active | 4B |
| Exploration Duration | 4.06 hours | 1.01 hours |
| Prompts Processed | 96 | 75 |
| Tokens Generated | 129,254 | 109,346 |
| Discovery Runs | 8 | 4 |
| Throughput | 10-10.8 tok/s | 25-52 tok/s |

## Zone Distribution Comparison

### Absolute Counts
| Zone | 80B Model | 4B Model |
|------|-----------|----------|
| structure_ripple | 84,674 | 92,154 |
| semantic_bridge | 43,565 | 16,910 |
| syntax_floor | 1,015 | 282 |
| **Total** | **129,254** | **109,346** |

### Percentage Distribution
| Zone | 80B Model | 4B Model | Difference |
|------|-----------|----------|------------|
| structure_ripple | 65.5% | 84.3% | +18.8% |
| semantic_bridge | 33.7% | 15.5% | -18.2% |
| syntax_floor | 0.8% | 0.3% | -0.5% |

## Analysis

### 1. Structure Ripple Dominance in 4B
The 4B dense model shows 84.3% structure_ripple usage compared to 65.5% for the 80B MoE model.

**Interpretation**:
- Smaller models may need to rely more on long-range structural patterns to maintain coherence
- With fewer parameters, the model compensates by building stronger positional/structural dependencies
- This could indicate more "formulaic" or "pattern-matching" behavior in responses

### 2. Semantic Bridge Balance in 80B
The 80B model uses 33.7% semantic_bridge compared to only 15.5% in the 4B model.

**Interpretation**:
- The MoE architecture allows for more nuanced mid-range attention patterns
- Expert routing enables context-specific semantic reasoning
- This suggests the 80B model can better integrate semantic relationships across the context

### 3. Minimal Syntax Floor in Both
Both models show <1% syntax_floor usage (local attention patterns).

**Interpretation**:
- Modern transformer architectures rely on global context even for local computations
- Neither model heavily uses short-range token-to-token attention
- This aligns with the self-attention mechanism's strength in capturing long-range dependencies

## Throughput Analysis

| Metric | 80B Model | 4B Model | Ratio |
|--------|-----------|----------|-------|
| Throughput | 10-10.8 tok/s | 25-52 tok/s | 3-5x faster |
| Tokens/Hour | ~38,000 | ~108,000 | 2.8x more |

The 4B model processes significantly more tokens per unit time, making it suitable for:
- High-volume inference workloads
- Real-time applications
- Resource-constrained deployments

## Routing Implications

### When to Use 80B (MoE)
- Tasks requiring nuanced semantic reasoning
- Complex analysis with multiple concepts
- Creative tasks requiring balanced attention
- Quality-critical applications

### When to Use 4B (Dense)
- Tasks with clear structural patterns
- High-throughput requirements
- Structured output generation (code, lists, formats)
- Cost-sensitive deployments

## Quantization Considerations

Based on zone distributions:

**80B Model**:
- Semantic_bridge zones (34%) may be sensitive to quantization
- Recommend conservative quantization for reasoning tasks
- structure_ripple zones likely more robust

**4B Model**:
- Heavy structure_ripple usage (84%) suggests more tolerance to quantization
- Pattern-based attention is more robust to precision loss
- Aggressive quantization may be viable

## Conclusions

1. **Architecture Matters**: MoE enables more balanced attention distribution
2. **Size Compensates**: Smaller models use more structural patterns
3. **Both Minimize Local Attention**: Global context dominates in modern LLMs
4. **Trade-off**: Speed (4B) vs Semantic Nuance (80B)

## Recommendations

1. **For general chat**: 80B provides better semantic balance
2. **For code/structured output**: 4B's structural dominance is advantageous
3. **For high-throughput**: 4B offers 3-5x better throughput
4. **For critical reasoning**: 80B's semantic_bridge advantage matters

---
*Generated: 2026-01-10T14:38*
"""

    with open("./exploration_outputs/80B_VS_4B_COMPARISON_REPORT.md", "w") as f:
        f.write(report)

    print(
        "Comparison report saved to: ./exploration_outputs/80B_VS_4B_COMPARISON_REPORT.md"
    )


if __name__ == "__main__":
    create_comparison_charts()
    create_comparison_report()

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY: Qwen3-80B (MoE) vs Qwen3-4B (Dense)")
    print("=" * 60)
    print("\nZone Distribution:")
    print("-" * 50)
    print(f"{'Zone':<20} {'80B (MoE)':<15} {'4B (Dense)':<15}")
    print("-" * 50)

    for zone in ["structure_ripple", "semantic_bridge", "syntax_floor"]:
        pct_80b = calculate_percentages(DATA["Qwen3-80B (MoE)"]["zones"])[zone]
        pct_4b = calculate_percentages(DATA["Qwen3-4B (Dense)"]["zones"])[zone]
        diff = pct_4b - pct_80b
        print(f"{zone:<20} {pct_80b:>6.1f}%        {pct_4b:>6.1f}%  ({diff:+.1f}%)")

    print("-" * 50)
    print("\nKey Insight: 4B model relies ~20% more on structure_ripple")
    print("             80B model has ~20% more semantic_bridge usage")
    print("=" * 60)
