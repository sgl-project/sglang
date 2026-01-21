import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the benchmark data
data = pd.read_csv(
    "qk_norm.txt",
    sep=r"\s+",
    header=None,
    names=[
        "idx",
        "head_dim",
        "GQA",
        "num_kv_heads",
        "batch_size",
        "aot",
        "jit",
        "fi",
        "torch",
    ],
)

# Get unique head dimensions
head_dims = sorted(data["head_dim"].unique())
n_dims = len(head_dims)

# Create figure with subplots for each head_dim
n_cols = min(2, n_dims)
n_rows = (n_dims + n_cols - 1) // n_cols
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False
)
axes = axes.flatten()

for i, hd in enumerate(head_dims):
    ax = axes[i]
    subset = data[data["head_dim"] == hd]

    # Group by batch_size and average across GQA/num_kv_heads combinations
    grouped = subset.groupby("batch_size")[["aot", "jit", "fi", "torch"]].mean()

    batch_sizes = grouped.index.values
    x = np.arange(len(batch_sizes))
    width = 0.2

    ax.bar(x - 1.5 * width, grouped["aot"], width, label="SGL AOT", color="orange")
    ax.bar(x - 0.5 * width, grouped["jit"], width, label="SGL JIT", color="blue")
    ax.bar(x + 0.5 * width, grouped["fi"], width, label="FlashInfer", color="green")
    ax.bar(x + 1.5 * width, grouped["torch"], width, label="PyTorch", color="red")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time (us)")
    ax.set_title(f"head_dim = {int(hd)}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(bs)) for bs in batch_sizes], rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("QK-Norm Performance Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("qknorm_benchmark.png", dpi=150, bbox_inches="tight")
print("Saved qknorm_benchmark.png")

# Create speedup comparison plot
fig2, axes2 = plt.subplots(
    n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False
)
axes2 = axes2.flatten()

for i, hd in enumerate(head_dims):
    ax = axes2[i]
    subset = data[data["head_dim"] == hd]
    grouped = subset.groupby("batch_size")[["aot", "jit", "fi", "torch"]].mean()

    batch_sizes = grouped.index.values

    # Calculate speedup: JIT vs AOT (higher = JIT is faster)
    speedup_jit = grouped["aot"] / grouped["jit"]

    ax.plot(
        batch_sizes,
        speedup_jit,
        "b-o",
        label="JIT speedup vs AOT",
        markersize=5,
        linewidth=2,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="AOT baseline")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup (AOT time / JIT time)")
    ax.set_title(f"head_dim = {int(hd)}")
    ax.set_xscale("log", base=2)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add average speedup annotation
    avg_speedup = speedup_jit.mean()
    ax.annotate(
        f"Avg: {avg_speedup:.2f}x",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

# Hide unused subplots
for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

plt.suptitle("JIT Kernel Speedup vs AOT Kernel", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("qknorm_speedup.png", dpi=150, bbox_inches="tight")
print("Saved qknorm_speedup.png")
