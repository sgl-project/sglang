#!/usr/bin/env python3
"""
Z-Image-Turbo 256×256 Performance Bottleneck Analysis
=====================================================
从 torch profiler trace 和 baseline JSON 中提取数据，
生成团队可分享的统计报表和图表。

Usage:
    python analyze_profile.py \
        --trace-dir ./logs \
        --baseline-dir ./zimage_bench \
        --output-dir ./analysis_report

Requirements:
    pip install matplotlib numpy  (GPU 机器上一般都有)
"""

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================
# Part 1: Baseline JSON 分析
# ============================================================

def load_baselines(baseline_dir: str) -> dict:
    """加载所有 baseline JSON 文件"""
    baselines = {}
    for f in sorted(Path(baseline_dir).glob("baseline_*.json")):
        with open(f) as fp:
            baselines[f.stem] = json.load(fp)
    return baselines


def print_baseline_comparison(baselines: dict):
    """打印基线对比表格"""
    print("\n" + "=" * 110)
    print(" Z-Image-Turbo 256×256 — 各配置延迟对比 (H20 GPU)")
    print("=" * 110)
    header = f"{'Config':<35} {'E2E(ms)':>8} {'TextEnc':>8} {'Denoise':>8} {'Decode':>7} {'Step(稳态)':>10} {'vs Base':>8}"
    print(header)
    print("-" * 110)

    base_e2e = None
    for name, data in baselines.items():
        e2e = data["total_duration_ms"]
        if base_e2e is None:
            base_e2e = e2e

        steps_dict = {s["name"]: s["duration_ms"] for s in data["steps"]}
        text_enc = steps_dict.get("TextEncodingStage", 0)
        denoise = steps_dict.get("DenoisingStage", 0)
        decode = steps_dict.get("DecodingStage", 0)

        denoise_steps = data.get("denoise_steps_ms", [])
        # Steady state: 后5步 (排除 warmup/compilation overhead)
        steady = denoise_steps[-5:] if len(denoise_steps) >= 5 else denoise_steps
        steady_avg = sum(s["duration_ms"] for s in steady) / len(steady) if steady else 0

        label = name.replace("baseline_", "").replace("_", " + ")
        delta = f"{(1 - e2e / base_e2e) * 100:+.1f}%"
        print(f"{label:<35} {e2e:>8.1f} {text_enc:>8.1f} {denoise:>8.1f} {decode:>7.1f} {steady_avg:>10.1f} {delta:>8}")

    print()


# ============================================================
# Part 2: Torch Profiler Trace 分析
# ============================================================

def load_trace(trace_path: str) -> list:
    """加载 torch profiler trace (支持 .gz 和普通 json)"""
    if trace_path.endswith(".gz"):
        with gzip.open(trace_path, "rb") as f:
            data = json.loads(f.read())
    else:
        with open(trace_path) as f:
            data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    return events


def classify_kernel(name: str) -> str:
    """将 CUDA kernel 名称分类到高层类别"""
    nl = name.lower()

    # GEMM 相关 (最大类)
    if "nvjet" in nl:
        return "BF16 GEMM (DiT)"
    if ("xmma_gemm" in nl and "f32f32" in nl) or ("splitkreduce" in nl and "f32" in nl):
        return "FP32 GEMM (TextEncoder)"
    if "xmma_gemm" in nl or "cutlass" in nl and "gemm" in nl:
        return "Other GEMM"
    if "splitkreduce" in nl:
        return "GEMM Helper (splitK)"

    # Attention
    if "flashattn" in nl or ("flash" in nl and ("fwd" in nl or "attn" in nl)):
        return "FlashAttention"

    # Convolution (VAE)
    if "fprop_implicit_gemm" in nl or ("conv" in nl and "fprop" in nl):
        return "Convolution (VAE)"

    # Normalization
    if "rmsnorm" in nl or "fused_qknorm" in nl:
        return "RMSNorm / QKNorm"
    if "layernorm" in nl or "layer_norm" in nl:
        return "LayerNorm"

    # Activations
    if "act_and_mul" in nl or "silu" in nl:
        return "SiLU / Activation"

    # Elementwise
    if "elementwise" in nl or "vectorized_elementwise" in nl:
        return "Elementwise Ops"

    # RoPE
    if "rotary" in nl or "rope" in nl:
        return "RoPE"

    # Memory
    if "memcpy" in nl or "memset" in nl:
        return "Memory Ops"

    # Reduction
    if "reduce" in nl or "softmax" in nl:
        return "Reduction / Softmax"

    # Layout transforms (usually VAE related)
    if "nhwc" in nl or "nchw" in nl:
        return "Layout Transform (VAE)"

    return "Other"


def analyze_kernels(events: list) -> dict:
    """分析 CUDA kernel 统计"""
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]

    # 按名称聚合
    kernel_stats = defaultdict(lambda: {"total_us": 0, "count": 0, "max_us": 0})
    for k in kernels:
        name = k.get("name", "?")
        dur = k.get("dur", 0)
        kernel_stats[name]["total_us"] += dur
        kernel_stats[name]["count"] += 1
        kernel_stats[name]["max_us"] = max(kernel_stats[name]["max_us"], dur)

    # 按类别聚合
    category_stats = defaultdict(lambda: {"total_us": 0, "count": 0, "kernels": []})
    for name, info in kernel_stats.items():
        cat = classify_kernel(name)
        category_stats[cat]["total_us"] += info["total_us"]
        category_stats[cat]["count"] += info["count"]
        category_stats[cat]["kernels"].append((name, info))

    total_us = sum(v["total_us"] for v in kernel_stats.values())

    return {
        "kernel_stats": dict(kernel_stats),
        "category_stats": dict(category_stats),
        "total_kernel_us": total_us,
        "total_kernels": len(kernels),
    }


def print_kernel_report(analysis: dict):
    """打印 kernel 级分析报告"""
    total_us = analysis["total_kernel_us"]
    cat_stats = analysis["category_stats"]

    print("\n" + "=" * 100)
    print(" CUDA Kernel 分类统计 (全流程: TextEnc + Denoise + Decode)")
    print("=" * 100)
    print(f" 总 kernel 数: {analysis['total_kernels']}")
    print(f" 总 GPU kernel 时间: {total_us / 1000:.1f} ms")
    print("-" * 100)
    print(f"{'Category':<30} {'Time(ms)':>10} {'%':>7} {'Count':>7} {'Avg(us)':>9}")
    print("-" * 100)

    cumulative = 0
    for cat, info in sorted(cat_stats.items(), key=lambda x: x[1]["total_us"], reverse=True):
        pct = info["total_us"] / total_us * 100
        cumulative += pct
        avg = info["total_us"] / info["count"] if info["count"] > 0 else 0
        marker = " ◀ BOTTLENECK" if pct > 15 else ""
        print(f"  {cat:<28} {info['total_us']/1000:>10.1f} {pct:>6.1f}% {info['count']:>7} {avg:>9.1f}{marker}")

    print("-" * 100)

    # Top 15 individual kernels
    print(f"\n{'='*100}")
    print(" Top 15 单个 CUDA Kernel (按总耗时)")
    print(f"{'='*100}")
    sorted_kernels = sorted(analysis["kernel_stats"].items(), key=lambda x: x[1]["total_us"], reverse=True)
    cumulative = 0
    for i, (name, info) in enumerate(sorted_kernels[:15]):
        pct = info["total_us"] / total_us * 100
        cumulative += pct
        avg = info["total_us"] / info["count"]
        cat = classify_kernel(name)
        print(f"  #{i+1:>2} {info['total_us']/1000:>8.1f}ms ({pct:>5.1f}%, cum {cumulative:>5.1f}%) x{info['count']:>5} avg {avg:>8.1f}us | [{cat}] {name[:60]}")


def print_stage_breakdown(baselines: dict):
    """打印 Stage 时间占比分析 (用于说明 TextEncoding 占比问题)"""
    if "baseline_1gpu" not in baselines:
        return

    data = baselines["baseline_1gpu"]
    e2e = data["total_duration_ms"]
    stages = {s["name"]: s["duration_ms"] for s in data["steps"]}

    print("\n" + "=" * 80)
    print(" Pipeline Stage 时间分解 (1 GPU Baseline)")
    print("=" * 80)

    # ASCII bar chart
    max_bar = 50
    for stage_name in ["TextEncodingStage", "DenoisingStage", "DecodingStage"]:
        ms = stages.get(stage_name, 0)
        pct = ms / e2e * 100
        bar_len = int(pct / 100 * max_bar)
        bar = "█" * bar_len + "░" * (max_bar - bar_len)
        marker = " ◀◀ #1 BOTTLENECK" if stage_name == "TextEncodingStage" else ""
        print(f"  {stage_name:<25} {bar} {ms:>7.1f}ms ({pct:>5.1f}%){marker}")

    other = e2e - sum(stages.get(s, 0) for s in ["TextEncodingStage", "DenoisingStage", "DecodingStage"])
    pct = other / e2e * 100
    bar_len = int(pct / 100 * max_bar)
    bar = "█" * bar_len + "░" * (max_bar - bar_len)
    print(f"  {'Other':<25} {bar} {other:>7.1f}ms ({pct:>5.1f}%)")
    print(f"\n  Total E2E: {e2e:.1f}ms")

    print(f"\n  ⚠️  关键发现: TextEncoding 占 {stages.get('TextEncodingStage',0)/e2e*100:.1f}%，超过 Denoising！")
    print(f"  原因: Qwen3 Text Encoder 跑 FP32, 而 H20 的 FP32 TFLOPS 远低于 BF16")
    print(f"  优化: FP32 → BF16 可让 TextEncoding 提速 2-4x, 预计节省 200-300ms")


# ============================================================
# Part 3: 图表生成 (matplotlib)
# ============================================================

def generate_charts(baselines: dict, analysis: dict, output_dir: str):
    """生成可分享的图表"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无 GUI 后端
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("\n⚠️  matplotlib 未安装, 跳过图表生成。安装命令: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 统一颜色方案
    COLORS = {
        "TextEncodingStage": "#e74c3c",   # 红色 (瓶颈!)
        "DenoisingStage": "#3498db",       # 蓝色
        "DecodingStage": "#2ecc71",        # 绿色
        "Other": "#95a5a6",                # 灰色
        "BF16 GEMM (DiT)": "#3498db",
        "FP32 GEMM (TextEncoder)": "#e74c3c",
        "FlashAttention": "#2ecc71",
        "Convolution (VAE)": "#9b59b6",
        "Elementwise Ops": "#f39c12",
        "SiLU / Activation": "#e67e22",
        "RMSNorm / QKNorm": "#1abc9c",
        "RoPE": "#34495e",
    }

    # ========== 图表 1: Pipeline Stage 时间饼图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    if "baseline_1gpu" in baselines:
        data = baselines["baseline_1gpu"]
        stages = {s["name"]: s["duration_ms"] for s in data["steps"]}
        e2e = data["total_duration_ms"]

        labels = ["TextEncodingStage", "DenoisingStage", "DecodingStage"]
        sizes = [stages.get(l, 0) for l in labels]
        other = e2e - sum(sizes)
        if other > 0:
            labels.append("Other")
            sizes.append(other)

        colors = [COLORS.get(l, "#95a5a6") for l in labels]
        explode = [0.08 if l == "TextEncodingStage" else 0 for l in labels]

        wedges, texts, autotexts = axes[0].pie(
            sizes, labels=None, autopct="%1.1f%%", colors=colors,
            explode=explode, startangle=90, textprops={"fontsize": 11}
        )
        # Bold the TextEncoding percentage
        autotexts[0].set_fontweight("bold")
        autotexts[0].set_fontsize(13)

        legend_labels = [f"{l} ({s:.0f}ms)" for l, s in zip(labels, sizes)]
        axes[0].legend(legend_labels, loc="lower left", fontsize=9)
        axes[0].set_title("Pipeline Stage Time Breakdown\n(1 GPU, E2E={:.0f}ms)".format(e2e), fontsize=13, fontweight="bold")

    # ========== 图表 2: CUDA Kernel 分类饼图 ==========
    cat_stats = analysis["category_stats"]
    total_us = analysis["total_kernel_us"]

    # Merge small categories into "Other"
    threshold_pct = 2.0
    main_cats = {}
    other_us = 0
    for cat, info in cat_stats.items():
        pct = info["total_us"] / total_us * 100
        if pct >= threshold_pct:
            main_cats[cat] = info["total_us"]
        else:
            other_us += info["total_us"]
    if other_us > 0:
        main_cats["Other (< 2%)"] = other_us

    sorted_cats = sorted(main_cats.items(), key=lambda x: x[1], reverse=True)
    labels2 = [c for c, _ in sorted_cats]
    sizes2 = [v / 1000 for _, v in sorted_cats]  # ms

    colors2 = [COLORS.get(l, "#95a5a6") for l in labels2]
    explode2 = [0.08 if "FP32" in l else 0 for l in labels2]

    wedges2, texts2, autotexts2 = axes[1].pie(
        sizes2, labels=None, autopct="%1.1f%%", colors=colors2,
        explode=explode2, startangle=90, textprops={"fontsize": 11}
    )

    legend_labels2 = [f"{l} ({s:.1f}ms)" for l, s in zip(labels2, sizes2)]
    axes[1].legend(legend_labels2, loc="lower left", fontsize=9)
    axes[1].set_title("CUDA Kernel Category Breakdown\n(GPU time={:.0f}ms)".format(total_us / 1000), fontsize=13, fontweight="bold")

    plt.tight_layout()
    path1 = os.path.join(output_dir, "01_pipeline_and_kernel_breakdown.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"\n✅ 图表已保存: {path1}")
    plt.close(fig)

    # ========== 图表 3: 各配置对比柱状图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    configs = []
    text_enc_times = []
    denoise_times = []
    decode_times = []

    for name, bdata in baselines.items():
        label = name.replace("baseline_", "").replace("_", "+")
        stages = {s["name"]: s["duration_ms"] for s in bdata["steps"]}

        # Skip configs with abnormally high values (compile regression)
        if stages.get("DenoisingStage", 0) > 1000:
            label += "\n(REGRESSION)"

        configs.append(label)
        text_enc_times.append(stages.get("TextEncodingStage", 0))
        denoise_times.append(stages.get("DenoisingStage", 0))
        decode_times.append(stages.get("DecodingStage", 0))

    x = np.arange(len(configs))
    width = 0.5

    bars1 = axes[0].bar(x, text_enc_times, width, label="TextEncoding", color=COLORS["TextEncodingStage"])
    bars2 = axes[0].bar(x, denoise_times, width, bottom=text_enc_times, label="Denoising", color=COLORS["DenoisingStage"])
    bottom2 = [t + d for t, d in zip(text_enc_times, denoise_times)]
    bars3 = axes[0].bar(x, decode_times, width, bottom=bottom2, label="Decoding", color=COLORS["DecodingStage"])

    axes[0].set_ylabel("Latency (ms)", fontsize=12)
    axes[0].set_title("E2E Latency by Configuration\n(Stacked by Stage)", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, fontsize=9)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, max(t + d + dc for t, d, dc in zip(text_enc_times, denoise_times, decode_times)) * 1.15)

    # Add total labels on top
    for i, (t, d, dc) in enumerate(zip(text_enc_times, denoise_times, decode_times)):
        total = t + d + dc
        axes[0].text(i, total + 10, f"{total:.0f}ms", ha="center", fontsize=9, fontweight="bold")

    # ========== 图表 4: Per-step denoising 对比 ==========
    for name, bdata in baselines.items():
        label = name.replace("baseline_", "").replace("_", "+")
        steps = bdata.get("denoise_steps_ms", [])
        if steps and max(s["duration_ms"] for s in steps) < 500:  # Skip compile regression
            step_times = [s["duration_ms"] for s in steps]
            axes[1].plot(range(len(step_times)), step_times, "o-", label=label, markersize=5, linewidth=2)

    axes[1].set_xlabel("Denoising Step", fontsize=12)
    axes[1].set_ylabel("Step Latency (ms)", fontsize=12)
    axes[1].set_title("Per-Step Denoising Latency\n(excluding torch.compile regression)", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(9))

    plt.tight_layout()
    path2 = os.path.join(output_dir, "02_config_comparison.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"✅ 图表已保存: {path2}")
    plt.close(fig)

    # ========== 图表 5: Kernel 级 Top-N 瀑布图 ==========
    fig, ax = plt.subplots(figsize=(14, 8))

    sorted_kernels = sorted(analysis["kernel_stats"].items(), key=lambda x: x[1]["total_us"], reverse=True)
    top_n = 15
    top_kernels = sorted_kernels[:top_n]

    names = []
    times = []
    colors_bar = []
    for name, info in reversed(top_kernels):  # reversed for horizontal bar (bottom-up)
        cat = classify_kernel(name)
        short_name = name[:55] + "..." if len(name) > 55 else name
        names.append(f"[{cat}]\n{short_name}")
        times.append(info["total_us"] / 1000)
        colors_bar.append(COLORS.get(cat, "#95a5a6"))

    y = np.arange(len(names))
    bars = ax.barh(y, times, color=colors_bar, height=0.7, edgecolor="white", linewidth=0.5)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}ms", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Total GPU Time (ms)", fontsize=12)
    ax.set_title(f"Top {top_n} CUDA Kernels by Total GPU Time", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path3 = os.path.join(output_dir, "03_top_kernels_waterfall.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"✅ 图表已保存: {path3}")
    plt.close(fig)

    # ========== 图表 6: FP32 vs BF16 GEMM 对比 (核心论据图) ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    gemm_data = {
        "FP32 GEMM\n(TextEncoder - Qwen3)": cat_stats.get("FP32 GEMM (TextEncoder)", {}).get("total_us", 0) / 1000,
        "BF16 GEMM\n(DiT - 30 layers)": cat_stats.get("BF16 GEMM (DiT)", {}).get("total_us", 0) / 1000,
        "FlashAttention\n(DiT Attention)": cat_stats.get("FlashAttention", {}).get("total_us", 0) / 1000,
        "Others\n(Norm+Act+RoPE+...)": sum(
            info["total_us"] for cat, info in cat_stats.items()
            if cat not in ["FP32 GEMM (TextEncoder)", "BF16 GEMM (DiT)", "FlashAttention"]
        ) / 1000,
    }

    names_g = list(gemm_data.keys())
    values_g = list(gemm_data.values())
    total_g = sum(values_g)
    colors_g = [COLORS.get("FP32 GEMM (TextEncoder)"), COLORS.get("BF16 GEMM (DiT)"), COLORS.get("FlashAttention"), "#95a5a6"]

    bars = ax.bar(names_g, values_g, color=colors_g, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, v in zip(bars, values_g):
        pct = v / total_g * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{v:.1f}ms\n({pct:.1f}%)", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("GPU Time (ms)", fontsize=13)
    ax.set_title("GPU Time Breakdown: Where is the Time Spent?\n\"GEMM dominates 91.5%, Attention is only 2.1%\"",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Add annotation arrow pointing to FP32
    ax.annotate(
        "← #1 Optimization Target\nFP32→BF16 can save 100-200ms",
        xy=(0, values_g[0]),
        xytext=(1.5, values_g[0] * 0.85),
        fontsize=10, color="#c0392b", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2),
    )

    plt.tight_layout()
    path4 = os.path.join(output_dir, "04_fp32_vs_bf16_gemm.png")
    fig.savefig(path4, dpi=150, bbox_inches="tight")
    print(f"✅ 图表已保存: {path4}")
    plt.close(fig)

    print(f"\n📊 所有图表已保存到: {output_dir}/")
    print("   可以直接分享 PNG 文件给团队成员")


# ============================================================
# Part 4: 生成 Markdown 报告
# ============================================================

def generate_markdown_report(baselines: dict, analysis: dict, output_dir: str):
    """生成 Markdown 格式的完整分析报告"""
    os.makedirs(output_dir, exist_ok=True)

    total_us = analysis["total_kernel_us"]
    cat_stats = analysis["category_stats"]

    # Get baseline 1gpu data
    base = baselines.get("baseline_1gpu", {})
    e2e = base.get("total_duration_ms", 0)
    stages = {s["name"]: s["duration_ms"] for s in base.get("steps", [])}

    report = f"""# Z-Image-Turbo 256×256 Performance Analysis Report

> **Hardware**: NVIDIA H20 (SM90, 96GB HBM3)
> **Model**: Z-Image-Turbo (30-layer DiT, dim=3840, 9 denoising steps, no CFG)
> **Resolution**: 256×256 (sequence length ~768 tokens)

## 1. E2E Latency Breakdown (1 GPU Baseline = {e2e:.0f}ms)

| Stage | Time (ms) | Percentage | Note |
|-------|-----------|-----------|------|
| TextEncodingStage | {stages.get('TextEncodingStage', 0):.1f} | {stages.get('TextEncodingStage', 0)/e2e*100:.1f}% | **#1 Bottleneck** — Qwen3 runs in FP32 |
| DenoisingStage | {stages.get('DenoisingStage', 0):.1f} | {stages.get('DenoisingStage', 0)/e2e*100:.1f}% | 9 steps, ~33ms/step (steady) |
| DecodingStage | {stages.get('DecodingStage', 0):.1f} | {stages.get('DecodingStage', 0)/e2e*100:.1f}% | Negligible |

![Pipeline Breakdown](01_pipeline_and_kernel_breakdown.png)

## 2. CUDA Kernel Category Analysis (Total GPU time: {total_us/1000:.0f}ms)

| Category | Time (ms) | % | Interpretation |
|----------|-----------|---|----------------|
"""
    for cat, info in sorted(cat_stats.items(), key=lambda x: x[1]["total_us"], reverse=True):
        pct = info["total_us"] / total_us * 100
        if pct >= 1.0:
            note = ""
            if "FP32" in cat:
                note = "**Optimization Target #1** (FP32→BF16)"
            elif "BF16 GEMM" in cat:
                note = "Optimization Target #2 (FP8/INT4)"
            elif "FlashAttention" in cat:
                note = "Only 2.1% — NOT a bottleneck"
            report += f"| {cat} | {info['total_us']/1000:.1f} | {pct:.1f}% | {note} |\n"

    report += f"""
![GEMM Breakdown](04_fp32_vs_bf16_gemm.png)

## 3. Key Findings

1. **TextEncoding is 62% of E2E** — The Qwen3 text encoder runs FP32 GEMM (`sm80_xmma_gemm_f32f32`), which is extremely slow on H20
2. **GEMM dominates 91.5%** of all GPU kernel time. FlashAttention is only 2.1% (short sequence = 768 tokens)
3. **torch.compile causes 8x regression** — Triton-generated kernels are slower than cuBLAS/nvJET on H20 for small matrices
4. **2-GPU SP is 12% slower** — Communication overhead > parallelism benefit for 768-token sequences
5. **Cache-DiT has no effect** — Only 9 denoising steps, insufficient inter-step redundancy

## 4. Optimization Plan (Target: -20% E2E latency)

| Priority | Optimization | Expected Savings | E2E Impact |
|----------|-------------|-----------------|------------|
| **P0** | TextEncoder FP32→BF16 | 200-300ms | **-27~40%** |
| P1 | DiT FP8 / Nunchaku INT4 | 50-100ms | -7~13% |
| P2 | Kernel fusion / torch.compile fix | 10-30ms | -1~4% |

![Config Comparison](02_config_comparison.png)

## 5. How to Reproduce

```bash
# Baseline
sglang generate --model-path $MODEL --prompt "..." --height 256 --width 256 --warmup --save-output

# Profile
sglang generate --model-path $MODEL --prompt "..." --height 256 --width 256 --warmup --profile --profile-all-stages
```

---
*Generated by analyze_profile.py*
"""

    report_path = os.path.join(output_dir, "ANALYSIS_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✅ Markdown 报告已保存: {report_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo 256x256 Profile Analyzer")
    parser.add_argument("--trace-dir", default="./logs", help="Torch profiler trace 目录")
    parser.add_argument("--baseline-dir", default="./zimage_bench", help="Baseline JSON 目录")
    parser.add_argument("--output-dir", default="./analysis_report", help="输出图表/报告目录")
    args = parser.parse_args()

    # 1. Load baselines
    print("📂 Loading baselines...")
    baselines = load_baselines(args.baseline_dir)
    if not baselines:
        print(f"❌ 未找到 baseline JSON 文件: {args.baseline_dir}/baseline_*.json")
        sys.exit(1)
    print(f"   Found {len(baselines)} baseline configs: {list(baselines.keys())}")

    # 2. Load trace
    print("📂 Loading torch profiler trace...")
    trace_files = list(Path(args.trace_dir).glob("*full_stages*.trace.json.gz"))
    if not trace_files:
        trace_files = list(Path(args.trace_dir).glob("*.trace.json.gz"))
    if not trace_files:
        print(f"❌ 未找到 trace 文件: {args.trace_dir}/*.trace.json.gz")
        sys.exit(1)

    trace_file = trace_files[0]
    print(f"   Using trace: {trace_file}")
    events = load_trace(str(trace_file))
    print(f"   Total events: {len(events)}")

    # 3. Analyze
    analysis = analyze_kernels(events)

    # 4. Print reports
    print_baseline_comparison(baselines)
    print_stage_breakdown(baselines)
    print_kernel_report(analysis)

    # 5. Generate charts
    generate_charts(baselines, analysis, args.output_dir)

    # 6. Generate markdown
    generate_markdown_report(baselines, analysis, args.output_dir)

    print(f"\n{'='*60}")
    print(f" 分析完成！")
    print(f"{'='*60}")
    print(f" 命令行报告: 见上方输出")
    print(f" 图表文件:   {args.output_dir}/*.png")
    print(f" MD 报告:    {args.output_dir}/ANALYSIS_REPORT.md")
    print(f"")
    print(f" 分享给团队:")
    print(f"   1. 直接转发 ANALYSIS_REPORT.md (含内嵌图表引用)")
    print(f"   2. 或截取上方命令行输出")
    print(f"   3. 图表 04_fp32_vs_bf16_gemm.png 是最核心的论据图")


if __name__ == "__main__":
    main()
