import itertools
import math

import torch
import triton
import triton.language as tl
from sgl_kernel import deepseekv3_fused_gate
from sglang.srt.layers.moe.topk import biased_grouped_topk


def biased_grouped_topk_org(scores, bias):
    return biased_grouped_topk(
        scores, scores, bias, topk=8, renormalize=True, num_expert_group=8, topk_group=4
    )

def biased_grouped_topk_org_kernel(scores, bias):
    return deepseekv3_fused_gate(
        scores, bias
    )

seq_length_range = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
configs = [(sq,) for sq in seq_length_range]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_length"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["original", "kernel"],
        line_names=["Original", "SGL Kernel"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="deepseekv3-fused-gate-performance",
        args={},
    )
)
def benchmark(seq_length, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    num_experts = 256

    scores = torch.randn(
        (seq_length, num_experts), device=device, dtype=dtype
    )
    bias = torch.rand(num_experts, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "original":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org(
                scores.clone(), bias.clone()
            ),
            quantiles=quantiles,
        )
    elif provider == "kernel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org_kernel(
                scores.clone(), bias.clone()
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/deepseekv3_fused_gate_sgl/",
        help="Path to save deepseekv3 fused gate benchmark results",
    )
    args = parser.parse_args()

    # Run performance benchmark
    benchmark.run(print_data=True)
