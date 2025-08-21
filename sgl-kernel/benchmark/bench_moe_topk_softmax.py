import itertools

import pytest
import torch
import triton
from sgl_kernel import topk_softmax
from vllm import _custom_ops as vllm_custom_ops


def vllm_topk_softmax(gating_output, topk):
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=torch.float32
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    torch.ops._moe_C.topk_softmax(
        topk_weights, topk_indices, token_expert_indices, gating_output
    )
    return topk_weights, topk_indices


def sglang_topk_softmax(gating_output, topk):
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=torch.float32
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )

    topk_softmax(
        topk_weights=topk_weights,
        topk_ids=topk_indices,
        gating_output=gating_output,
    )

    return topk_weights, topk_indices


def calculate_diff(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )
    weights_vllm, indices_vllm = vllm_topk_softmax(gating_output.clone(), topk)
    weights_sglang, indices_sglang = sglang_topk_softmax(gating_output.clone(), topk)

    weights_diff = torch.abs(weights_vllm - weights_sglang).mean().item()
    indices_match = torch.equal(indices_vllm, indices_sglang)

    if (
        torch.allclose(weights_vllm, weights_sglang, atol=1e-3, rtol=1e-3)
        and indices_match
    ):
        print("✅ VLLM and SGLang topk_softmax implementations match")
    else:
        print(
            f"❌ Implementations differ: Weights diff={weights_diff}, Indices match={indices_match}"
        )


num_tokens_range = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
num_experts_range = [32, 64, 128, 256, 12, 512]
topk_range = [1, 2, 4, 8]

configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang", "vllm"],
        line_names=["SGLang", "VLLM"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Latency (us)",
        plot_name="topk-softmax-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, provider):

    gating_output = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )

    if provider == "vllm" or provider == "vllm1":
        fn = lambda: vllm_topk_softmax(gating_output, topk)
    elif provider == "sglang" or provider == "sglang1":
        fn = lambda: sglang_topk_softmax(gating_output, topk)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    configs = [
        (20, 256, 4),
        (20, 256, 8),
        (20, 12, 4),
        (20, 12, 1),
        (20, 512, 4),
        (20, 512, 1),
    ]
    for num_tokens, num_experts, topk in configs:
        calculate_diff(num_tokens, num_experts, topk)
    benchmark.run(print_data=True)
