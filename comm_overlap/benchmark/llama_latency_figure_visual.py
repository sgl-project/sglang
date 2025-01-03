import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [1, 16, 64, 128]
configs = [(256, 32), (256, 256), (512, 32), (512, 256)]

torch_prefill = [
    [0.02312, 0.04599, 0.46298, 0.29263],  # 256/32
    [0.02305, 0.04196, 0.14980, 0.29213],  # 256/256
    [0.02562, 0.07940, 0.28881, 0.59434],  # 512/32
    [0.02357, 0.07765, 0.28795, 0.57709],  # 512/256
]

torch_decode = [
    [0.00382, 0.00428, 0.00527, 0.00589],  # 256/32
    [0.00381, 0.00427, 0.00529, 0.00595],  # 256/256
    [0.00382, 0.00432, 0.00546, 0.00617],  # 512/32
    [0.00381, 0.00432, 0.00541, 0.00631],  # 512/256
]

te_prefill = [
    [0.02251, 0.04727, 0.14864, 0.28941],  # 256/32
    [0.02318, 0.04203, 0.14796, 0.29031],  # 256/256
    [0.19420, 0.27644, 0.28941, 0.94313],  # 512/32
    [0.02361, 0.07754, 0.28755, 0.57659],  # 512/256
]

te_decode = [
    [0.00381, 0.00427, 0.00528, 0.00589],  # 256/32
    [0.00380, 0.00427, 0.00529, 0.00593],  # 256/256
    [0.00382, 0.00432, 0.00546, 0.00618],  # 512/32
    [0.00381, 0.00433, 0.00542, 0.00629],  # 512/256
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

x = np.arange(len(batch_sizes))
width = 0.1

for i, (input_len, output_len) in enumerate(configs):
    ax1.bar(
        x + i * width * 2,
        torch_prefill[i],
        width,
        label=f"Torch-{input_len}/{output_len}",
        alpha=0.8,
    )
    ax1.bar(
        x + i * width * 2 + width,
        te_prefill[i],
        width,
        label=f"TE-{input_len}/{output_len}",
        alpha=0.8,
    )

for i, (input_len, output_len) in enumerate(configs):
    ax2.bar(
        x + i * width * 2,
        torch_decode[i],
        width,
        label=f"Torch-{input_len}/{output_len}",
        alpha=0.8,
    )
    ax2.bar(
        x + i * width * 2 + width,
        te_decode[i],
        width,
        label=f"TE-{input_len}/{output_len}",
        alpha=0.8,
    )

for ax, title in [(ax1, "Prefill Latency"), (ax2, "Median Decode Latency")]:
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"Torch vs TE {title} Comparison")
    ax.set_xticks(x + width * 3.5)
    ax.set_xticklabels(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()

plt.savefig("latency_comparison_pd.png", bbox_inches="tight")
