#!/usr/bin/env python3
"""
Generate benchmark visualization charts for Z-Image-Turbo multi-resolution analysis.
Outputs PNG charts to zimage_256_256/analysis_report/
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = "/data/home/rhyshen/rhyshen/workspace/sglang/zimage_256_256/zimage_bench"
OUT = "/data/home/rhyshen/rhyshen/workspace/sglang/zimage_256_256/analysis_report"

RESOLUTIONS = ["256_256", "512_512", "1024_1024"]
RES_LABELS = ["256×256", "512×512", "1024×1024"]

CONFIGS = [
    ("baseline_1gpu.json", "1GPU FP32"),
    ("baseline_1gpu_tebf16.json", "1GPU BF16-TE"),
    ("baseline_1gpu_tebf16_cachedit.json", "BF16-TE+CacheDiT"),
    ("baseline_1gpu_tebf16_compile.json", "BF16-TE+Compile"),
    ("baseline_1gpu_tebf16_cachedit_fp8_cutlass.json", "CacheDiT+FP8-CUTLASS"),
    ("baseline_1gpu_tebf16_cachedit_fp8_deepgemm.json", "CacheDiT+FP8-DeepGemm"),
    ("baseline_2gpu.json", "2GPU SP"),
]

def load_data():
    data = {}
    for res in RESOLUTIONS:
        data[res] = {}
        for fname, label in CONFIGS:
            path = os.path.join(BASE, res, fname)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                steps = {s["name"]: s["duration_ms"] for s in d["steps"]}
                denoise_ms = [s["duration_ms"] for s in d["denoise_steps_ms"]]
                ss = np.mean(denoise_ms[3:]) if len(denoise_ms) > 3 else np.mean(denoise_ms)
                peak_mem = d.get("memory_checkpoints", {}).get("mem_analysis", {}).get("peak_reserved_mb",
                           d.get("memory_checkpoints", {}).get("after_forward", {}).get("peak_reserved_mb", 0))
                data[res][label] = {
                    "total": d["total_duration_ms"],
                    "textenc": steps.get("TextEncodingStage", 0),
                    "denoise": steps.get("DenoisingStage", 0),
                    "decode": steps.get("DecodingStage", 0),
                    "ss_step": ss,
                    "peak_mem": peak_mem,
                    "denoise_steps": denoise_ms,
                }
    return data

data = load_data()

# Color scheme
COLORS = {
    "1GPU FP32": "#999999",
    "1GPU BF16-TE": "#4A90D9",
    "BF16-TE+CacheDiT": "#50C878",
    "BF16-TE+Compile": "#FFA500",
    "CacheDiT+FP8-CUTLASS": "#E06666",
    "CacheDiT+FP8-DeepGemm": "#CC0000",
    "2GPU SP": "#9B59B6",
}

# ============================================================
# Chart 1: E2E Latency - Grouped Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Z-Image-Turbo E2E Latency by Resolution & Config (H20 GPU)", fontsize=16, fontweight='bold')

for idx, (res, res_label) in enumerate(zip(RESOLUTIONS, RES_LABELS)):
    ax = axes[idx]
    labels = [l for _, l in CONFIGS if l in data[res]]
    vals = [data[res][l]["total"] for l in labels]
    colors = [COLORS.get(l, "#888") for l in labels]

    bars = ax.barh(range(len(labels)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("E2E Latency (ms)")
    ax.set_title(f"{res_label}", fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}ms', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "multi_res_e2e_latency.png"), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 2: Speedup vs 1GPU Baseline
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

main_configs = [
    "1GPU BF16-TE",
    "BF16-TE+CacheDiT",
    "CacheDiT+FP8-CUTLASS",
    "CacheDiT+FP8-DeepGemm",
    "2GPU SP",
]

x = np.arange(len(RES_LABELS))
width = 0.15
offsets = np.arange(len(main_configs)) - len(main_configs)/2 + 0.5

for i, cfg in enumerate(main_configs):
    speedups = []
    for res in RESOLUTIONS:
        baseline = data[res]["1GPU FP32"]["total"]
        val = data[res].get(cfg, {}).get("total", baseline)
        speedups.append(baseline / val)

    bars = ax.bar(x + offsets[i]*width, speedups, width, label=cfg, color=COLORS.get(cfg, "#888"))
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{s:.2f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Resolution', fontsize=12)
ax.set_ylabel('Speedup vs 1GPU FP32 Baseline', fontsize=12)
ax.set_title('Speedup by Config & Resolution (Higher = Better)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(RES_LABELS)
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='1.0× baseline')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(0, 2.0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "multi_res_speedup.png"), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 3: Denoising Steady-State Step Time
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

for i, cfg in enumerate(main_configs):
    ss_times = [data[res].get(cfg, {}).get("ss_step", 0) for res in RESOLUTIONS]
    ax.plot(RES_LABELS, ss_times, 'o-', label=cfg, color=COLORS.get(cfg, "#888"),
            linewidth=2, markersize=8)
    for j, val in enumerate(ss_times):
        ax.annotate(f'{val:.1f}ms', (RES_LABELS[j], val), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8)

# Add baseline
baseline_ss = [data[res]["1GPU FP32"]["ss_step"] for res in RESOLUTIONS]
ax.plot(RES_LABELS, baseline_ss, 'x--', label="1GPU FP32", color=COLORS["1GPU FP32"],
        linewidth=2, markersize=10)
for j, val in enumerate(baseline_ss):
    ax.annotate(f'{val:.1f}ms', (RES_LABELS[j], val), textcoords="offset points",
               xytext=(0, -15), ha='center', fontsize=8, color='#666')

ax.set_xlabel('Resolution', fontsize=12)
ax.set_ylabel('Steady-State Step Time (ms)', fontsize=12)
ax.set_title('Denoising Steady-State Step Time (Lower = Better)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "multi_res_ss_step.png"), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 4: Peak VRAM Usage
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

vram_configs = ["1GPU FP32", "BF16-TE+CacheDiT", "CacheDiT+FP8-CUTLASS", "CacheDiT+FP8-DeepGemm"]
x = np.arange(len(RES_LABELS))
width = 0.2
offsets = np.arange(len(vram_configs)) - len(vram_configs)/2 + 0.5

for i, cfg in enumerate(vram_configs):
    mems = [data[res].get(cfg, {}).get("peak_mem", 0) / 1024 for res in RESOLUTIONS]
    bars = ax.bar(x + offsets[i]*width, mems, width, label=cfg, color=COLORS.get(cfg, "#888"))
    for bar, m in zip(bars, mems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{m:.1f}GB', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Resolution', fontsize=12)
ax.set_ylabel('Peak VRAM (GB)', fontsize=12)
ax.set_title('Peak VRAM Usage by Config & Resolution', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(RES_LABELS)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "multi_res_vram.png"), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 5: Stage Breakdown - Stacked Bar (Best configs)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Stage Breakdown: TextEncoder + Denoising + Decoding", fontsize=16, fontweight='bold')

stage_configs = ["1GPU FP32", "1GPU BF16-TE", "BF16-TE+CacheDiT",
                 "CacheDiT+FP8-DeepGemm", "2GPU SP"]

for idx, (res, res_label) in enumerate(zip(RESOLUTIONS, RES_LABELS)):
    ax = axes[idx]
    cfgs = [c for c in stage_configs if c in data[res]]
    textenc = [data[res][c]["textenc"] for c in cfgs]
    denoise = [data[res][c]["denoise"] for c in cfgs]
    decode = [data[res][c]["decode"] for c in cfgs]

    y = np.arange(len(cfgs))
    ax.barh(y, textenc, height=0.5, label='TextEncoder', color='#4A90D9')
    ax.barh(y, denoise, height=0.5, left=textenc, label='Denoising', color='#E06666')
    ax.barh(y, decode, height=0.5, left=[t+d for t,d in zip(textenc, denoise)],
            label='Decoding', color='#50C878')

    ax.set_yticks(y)
    ax.set_yticklabels(cfgs, fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"{res_label}", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "multi_res_stage_breakdown.png"), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 6: FP8 Denoising Speedup Trend
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

fp8_configs = ["CacheDiT+FP8-CUTLASS", "CacheDiT+FP8-DeepGemm"]
for cfg in fp8_configs:
    speedups = []
    for res in RESOLUTIONS:
        baseline_denoise = data[res]["BF16-TE+CacheDiT"]["denoise"]
        fp8_denoise = data[res][cfg]["denoise"]
        speedups.append(baseline_denoise / fp8_denoise)
    ax.plot(RES_LABELS, speedups, 'o-', label=cfg, color=COLORS[cfg], linewidth=2.5, markersize=10)
    for j, s in enumerate(speedups):
        ax.annotate(f'{s:.2f}×', (RES_LABELS[j], s), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Resolution', fontsize=12)
ax.set_ylabel('FP8 Denoising Speedup vs BF16+CacheDiT', fontsize=12)
ax.set_title('FP8 Quantization Benefit Increases with Resolution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0.8, 1.6)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "multi_res_fp8_denoising_speedup.png"), dpi=150, bbox_inches='tight')
plt.close()

print("All charts generated:")
for f in sorted(os.listdir(OUT)):
    if f.endswith('.png') and 'multi_res' in f:
        print(f"  {f}")
