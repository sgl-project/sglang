import itertools
import math
import os

import torch
import triton
import triton.language as tl
from sgl_kernel import kimi_k2_moe_fused_gate

from sglang.srt.layers.moe.topk import kimi_k2_biased_topk_impl

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def kimi_k2_biased_topk_torch_compile(scores, bias, topk, routed_scaling_factor):
    """Original torch.compile-based implementation"""
    return kimi_k2_biased_topk_impl(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
    )


def kimi_k2_biased_topk_fused_kernel(scores, bias, topk, routed_scaling_factor):
    """Our fused CUDA kernel implementation"""
    return kimi_k2_moe_fused_gate(
        scores,
        bias,
        topk=topk,
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
    )


# CI environment uses simplified parameters
if IS_CI:
    seq_length_range = [5000]  # Only test one sequence length in CI
else:
    seq_length_range = [
        1,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        10000,
        15000,
        20000,
        25000,
        30000,
        35000,
        40000,
    ]

configs = [(sq,) for sq in seq_length_range]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_length"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["torch_compile", "fused_kernel"],
        line_names=["Torch Compile", "Fused Kernel"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="kimi-k2-moe-fused-gate-performance",
        args={},
    )
)
def benchmark(seq_length, provider):
    dtype = torch.float32
    device = torch.device("cuda")
    num_experts, topk = 384, 6  # Kimi K2 configuration
    routed_scaling_factor = 2.872  # Kimi K2's routed scaling factor

    scores = torch.randn((seq_length, num_experts), device=device, dtype=dtype)
    bias = torch.rand(num_experts, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch_compile":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: kimi_k2_biased_topk_torch_compile(
                scores.clone(), bias.clone(), topk, routed_scaling_factor
            ),
            quantiles=quantiles,
        )
    elif provider == "fused_kernel":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: kimi_k2_biased_topk_fused_kernel(
                scores.clone(), bias.clone(), topk, routed_scaling_factor
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    print("=" * 80)
    print("Benchmarking Kimi K2 MoE Fused Gate Performance")
    print("=" * 80)
    print("\nPerformance vs Sequence Length (384 experts, topk=6)")
    benchmark.run(print_data=True, save_path=".")
