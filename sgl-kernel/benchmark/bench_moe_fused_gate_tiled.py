import itertools
import math

import torch
import triton
import triton.language as tl
from sgl_kernel import moe_fused_gate

from sglang.srt.layers.moe.topk import biased_grouped_topk_impl


def biased_grouped_topk_ref(scores, bias, num_expert_group, topk_group, topk):
    # Match reference used in bench_moe_fused_gate.py
    return biased_grouped_topk_impl(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=2.5,
    )


def moe_fused_gate_kernel(scores, bias, num_expert_group, topk_group, topk):
    return moe_fused_gate(scores, bias, num_expert_group, topk_group, topk)


# Choose a sequence length sweep consistent with existing benchmark style
seq_length_range = [5000, 10000, 15000, 20000]

# Focus on tiled-path configs (VPT > 32)
configs = []
configs += [(sq, 64, 1, 1, 6) for sq in seq_length_range]     # Kimi VL: VPT=64
configs += [(sq, 384, 1, 1, 8) for sq in seq_length_range]    # Kimi K2: VPT=384
configs += [(sq, 1024, 8, 4, 8) for sq in seq_length_range]   # VPT=128
configs += [(sq, 2048, 8, 4, 8) for sq in seq_length_range]   # VPT=256


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_length", "num_experts", "num_expert_group", "topk_group", "topk"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["original", "kernel"],
        line_names=["Original", "SGL Kernel"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="moe-fused-gate-tiled-performance",
        args={},
    )
)
def benchmark(seq_length, num_experts, num_expert_group, topk_group, topk, provider):
    # Follow existing dtype/device choice
    dtype = torch.bfloat16
    device = torch.device("cuda")

    scores = torch.randn((seq_length, num_experts), device=device, dtype=dtype)
    bias = torch.rand(num_experts, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "original":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_ref(
                scores.clone(), bias.clone(), num_expert_group, topk_group, topk
            ),
            quantiles=quantiles,
        )
    elif provider == "kernel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_fused_gate_kernel(
                scores.clone(), bias.clone(), num_expert_group, topk_group, topk
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)


