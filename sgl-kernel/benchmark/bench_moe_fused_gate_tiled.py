import itertools
import math

import torch
import triton
import triton.language as tl
from sgl_kernel import moe_fused_gate

import torch.nn.functional as F


def biased_grouped_topk_ref_impl(scores, bias, num_expert_group, topk_group, topk):
    # Pure PyTorch reference to avoid implicit kernel paths and control compile modes.
    # Logic mirrors biased_grouped_topk_impl without shared experts handling (set to 0 for bench).
    # scores: [N, E], bias: [E]
    n, e = scores.shape
    scores_sig = scores.sigmoid()
    scores_for_choice = scores_sig + bias.unsqueeze(0)

    # group selection via top2 sum
    g = num_expert_group
    per_group = e // g
    view = scores_for_choice.view(n, g, per_group)
    top2 = torch.topk(view, k=2, dim=-1).values
    group_scores = top2.sum(dim=-1)  # [n, g]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices  # [n, topk_group]

    # mask and topk within selected groups
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(n, g, per_group).reshape(n, e)
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    topk_vals, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores_sig.gather(1, topk_ids)

    # renormalize
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-12)
    return topk_weights, topk_ids


# wrap reference with different compile modes
def make_ref_fn(compile_mode: str):
    fn = biased_grouped_topk_ref_impl
    if compile_mode == "eager":
        return fn
    if compile_mode == "compile-static":
        return torch.compile(fn, dynamic=False)
    if compile_mode == "compile-dynamic":
        return torch.compile(fn, dynamic=True)
    raise ValueError(f"Unknown compile_mode: {compile_mode}")


def moe_fused_gate_kernel(scores, bias, num_expert_group, topk_group, topk):
    return moe_fused_gate(scores, bias, num_expert_group, topk_group, topk)


# Choose a sequence length sweep consistent with existing benchmark style
seq_length_range = [2048, 3072, 4096, 10240, 15360, 20480]

# Focus on tiled-path configs (VPT > 32)
configs = []
configs += [(sq, 64, 1, 1, 6) for sq in seq_length_range]     # Kimi VL: VPT=64
configs += [(sq, 384, 1, 1, 8) for sq in seq_length_range]    # Kimi K2: VPT=384


def _bench_template(dtype: torch.dtype, plot_suffix: str):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_length", "num_experts", "num_expert_group", "topk_group", "topk"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["orig-eager", "orig-compile-static", "orig-compile-dynamic", "kernel"],
            line_names=["Original-Eager", "Original-Compile-Static", "Original-Compile-Dynamic", "SGL Kernel"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"moe-fused-gate-tiled-performance-{plot_suffix}",
            args={},
        )
    )
    def benchmark(seq_length, num_experts, num_expert_group, topk_group, topk, provider):
        device = torch.device("cuda")

        scores = torch.randn((seq_length, num_experts), device=device, dtype=dtype)
        bias = torch.rand(num_experts, device=device, dtype=dtype)

        quantiles = [0.5, 0.2, 0.8]

        if provider.startswith("orig"):
            mode = provider.replace("orig-", "")
            ref = make_ref_fn(mode)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: ref(scores.clone(), bias.clone(), num_expert_group, topk_group, topk),
                quantiles=quantiles,
            )
        elif provider == "kernel":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: moe_fused_gate_kernel(
                    scores.clone(), bias.clone(), num_expert_group, topk_group, topk
                ),
                quantiles=quantiles,
            )
        else:
            raise ValueError(provider)

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


benchmark_bf16 = _bench_template(torch.bfloat16, "bf16")
benchmark_fp16 = _bench_template(torch.float16, "fp16")
benchmark_fp32 = _bench_template(torch.float32, "fp32")


if __name__ == "__main__":
    benchmark_bf16.run(print_data=True)
    benchmark_fp16.run(print_data=True)
    benchmark_fp32.run(print_data=True)


