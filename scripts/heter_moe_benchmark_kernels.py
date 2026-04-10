"""Benchmark MoE kernels (a16w16 vs a8w8) with Qwen3-30B-A3B expert dimensions.

Measures latency and throughput across varying batch sizes to find the
memory-bound → compute-bound knee for each precision config.

Usage: PYTHONPATH=python python3 scripts/heter_moe_benchmark_kernels.py
"""

import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

# Mock ServerArgs before importing sglang MoE code
import sglang.srt.server_args as sa

if sa._global_server_args is None:
    mock = object.__new__(sa.ServerArgs)
    mock.enable_deterministic_inference = False
    mock.disable_moe_autotuning = False
    sa._global_server_args = mock

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts

# Qwen3-30B-A3B expert dimensions
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768  # per expert
NUM_EXPERTS = 128
TOP_K = 8
FUSED_INTERMEDIATE = 2 * INTERMEDIATE_SIZE  # w13 = gate + up fused

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
WARMUP_ITERS = 50
MEASURE_ITERS = 200

OUT_DIR = "/data/heter-moe/profiles/groupgemm"


def make_bf16_weights(E, N, K, device):
    w13 = torch.randn(E, 2 * N, K, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(E, K, N, dtype=torch.bfloat16, device=device)
    return w13, w2


def make_int8_weights(E, N, K, device):
    w13 = torch.randint(-128, 127, (E, 2 * N, K), dtype=torch.int8, device=device)
    w2 = torch.randint(-128, 127, (E, K, N), dtype=torch.int8, device=device)
    w13_scale = torch.rand(E, 2 * N, 1, dtype=torch.float32, device=device) * 0.01
    w2_scale = torch.rand(E, K, 1, dtype=torch.float32, device=device) * 0.01
    return w13, w2, w13_scale, w2_scale


def benchmark_one(fn, warmup=WARMUP_ITERS, iters=MEASURE_ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    # Use median
    return times[len(times) // 2]


def bench_bf16(M, device):
    x = torch.randn(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    w13, w2 = make_bf16_weights(NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE, device)
    topk_w = torch.ones(M, TOP_K, dtype=torch.bfloat16, device=device) / TOP_K
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), device=device)

    def fn():
        return outplace_fused_experts(x, w13, w2, topk_w, topk_ids)

    return benchmark_one(fn)


def bench_int8(M, device):
    x = torch.randn(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    w13, w2, w13_s, w2_s = make_int8_weights(
        NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE, device
    )
    topk_w = torch.ones(M, TOP_K, dtype=torch.bfloat16, device=device) / TOP_K
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), device=device)

    def fn():
        return outplace_fused_experts(
            x,
            w13,
            w2,
            topk_w,
            topk_ids,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=w13_s,
            w2_scale=w2_s,
        )

    return benchmark_one(fn)


def compute_tflops(M, latency_ms):
    # Two GEMMs per MoE forward: w13 (gate+up) and w2 (down)
    # w13: M*TOP_K tokens, each does [1, K] x [K, 2N] → 2*M*TOP_K*K*2N flops
    # w2:  M*TOP_K tokens, each does [1, N] x [N, K] → 2*M*TOP_K*N*K flops
    total_tokens = M * TOP_K
    flops_w13 = 2 * total_tokens * HIDDEN_SIZE * FUSED_INTERMEDIATE
    flops_w2 = 2 * total_tokens * INTERMEDIATE_SIZE * HIDDEN_SIZE
    total_flops = flops_w13 + flops_w2
    return total_flops / (latency_ms * 1e-3) / 1e12


def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"Expert dims: K={HIDDEN_SIZE}, N={INTERMEDIATE_SIZE}, E={NUM_EXPERTS}, top_k={TOP_K}"
    )
    print(f"Warmup={WARMUP_ITERS}, Measure={MEASURE_ITERS} (median)\n")

    results = []
    header = ["tokens", "kernel", "latency_ms", "tflops"]

    for M in BATCH_SIZES:
        print(f"M={M:>5}  ", end="", flush=True)

        lat_bf16 = bench_bf16(M, device)
        tflops_bf16 = compute_tflops(M, lat_bf16)
        results.append([M, "a16w16", f"{lat_bf16:.3f}", f"{tflops_bf16:.2f}"])
        print(
            f"a16w16: {lat_bf16:7.3f}ms ({tflops_bf16:6.2f} TFLOPS)  ",
            end="",
            flush=True,
        )

        lat_int8 = bench_int8(M, device)
        tflops_int8 = compute_tflops(M, lat_int8)
        results.append([M, "a8w8", f"{lat_int8:.3f}", f"{tflops_int8:.2f}"])
        print(f"a8w8: {lat_int8:7.3f}ms ({tflops_int8:6.2f} TFLOPS)")

    # Save CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "kernel_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    # Generate plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bf16_data = [(int(r[0]), float(r[2])) for r in results if r[1] == "a16w16"]
        int8_data = [(int(r[0]), float(r[2])) for r in results if r[1] == "a8w8"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Latency plot
        ax1.plot(
            [d[0] for d in bf16_data],
            [d[1] for d in bf16_data],
            "o-",
            label="a16w16 (BF16)",
        )
        ax1.plot(
            [d[0] for d in int8_data],
            [d[1] for d in int8_data],
            "s-",
            label="a8w8 (INT8)",
        )
        ax1.set_xlabel("Tokens (M)")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("MoE Kernel Latency")
        ax1.set_xscale("log", base=2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Throughput plot
        bf16_tflops = [(int(r[0]), float(r[3])) for r in results if r[1] == "a16w16"]
        int8_tflops = [(int(r[0]), float(r[3])) for r in results if r[1] == "a8w8"]
        ax2.plot(
            [d[0] for d in bf16_tflops],
            [d[1] for d in bf16_tflops],
            "o-",
            label="a16w16 (BF16)",
        )
        ax2.plot(
            [d[0] for d in int8_tflops],
            [d[1] for d in int8_tflops],
            "s-",
            label="a8w8 (INT8)",
        )
        ax2.set_xlabel("Tokens (M)")
        ax2.set_ylabel("Throughput (TFLOPS)")
        ax2.set_title("MoE Kernel Throughput")
        ax2.set_xscale("log", base=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, "kernel_benchmark.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main()
