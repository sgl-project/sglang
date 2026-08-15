import itertools
import os

import pytest
import torch
import triton
from sgl_kernel import topk_sigmoid

from sglang.utils import is_in_ci

# Optional MUSA import
try:
    from sglang.srt.utils import is_musa

    if is_musa():
        from sglang.srt.hardware_backend.musa.kernels.topk import (
            topk_sigmoid as musa_topk_sigmoid,
        )

        MUSA_AVAILABLE = True
    else:
        musa_topk_sigmoid = None
        MUSA_AVAILABLE = False
except ImportError:
    musa_topk_sigmoid = None
    MUSA_AVAILABLE = False

IS_CI = is_in_ci()


def torch_topk_sigmoid_native(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
):
    scores = gating_output.sigmoid()
    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        _, topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_weights = scores.gather(1, topk_indices)
    else:
        topk_weights, topk_indices = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices


def sglang_topk_sigmoid(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
):
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize=renormalize,
        correction_bias=correction_bias,
    )

    return topk_weights, topk_indices


def musa_topk_sigmoid_fn(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
):
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    musa_topk_sigmoid(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize=renormalize,
        correction_bias=correction_bias,
    )

    return topk_weights, topk_indices


def get_topk_sigmoid_input(num_tokens, num_experts):
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn((num_experts), dtype=torch.float32, device="cuda")
    return gating_output, correction_bias


def calculate_diff(num_tokens, num_experts, topk):
    gating_output, correction_bias = get_topk_sigmoid_input(num_tokens, num_experts)

    weights_torch, indices_torch = torch_topk_sigmoid_native(
        gating_output.clone(),
        topk,
        True,
        correction_bias.clone(),
    )
    weights_sglang, indices_sglang = sglang_topk_sigmoid(
        gating_output.clone(),
        topk,
        True,
        correction_bias.clone(),
    )

    weights_diff = torch.abs(weights_torch - weights_sglang).mean().item()
    indices_match = torch.equal(indices_torch, indices_sglang)

    if (
        torch.allclose(weights_torch, weights_sglang, atol=1e-3, rtol=1e-3)
        and indices_match
    ):
        print("✅ Torch and SGLang topk_sigmoid implementations match")
    else:
        print(
            f"❌ Implementations differ: Weights diff={weights_diff}, Indices match={indices_match}"
        )

    if MUSA_AVAILABLE:
        weights_musa, indices_musa = musa_topk_sigmoid_fn(
            gating_output.clone(),
            topk,
            True,
            correction_bias.clone(),
        )
        weights_diff_musa = torch.abs(weights_sglang - weights_musa).mean().item()
        indices_match_musa = torch.equal(indices_sglang, indices_musa)

        if (
            torch.allclose(weights_sglang, weights_musa, atol=1e-3, rtol=1e-3)
            and indices_match_musa
        ):
            print("✅ SGLang and MUSA topk_sigmoid implementations match")
        else:
            print(
                f"❌ MUSA vs SGLang differ: Weights diff={weights_diff_musa}, Indices match={indices_match_musa}"
            )
    else:
        print("⚠️ MUSA not available, skipping MUSA comparison")


# CI environment uses simplified parameters
if IS_CI:
    num_tokens_range = [128]  # Single value for CI
    num_experts_range = [32]  # Single value for CI
    topk_range = [2]  # Single value for CI
else:
    num_tokens_range = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_experts_range = [32, 64, 128, 256, 12, 512]
    topk_range = [1, 2, 4, 8]

configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


# Filter providers based on availability
line_vals = ["sglang", "torch"]
line_names = ["SGLang", "Torch"]
styles = [("blue", "-"), ("green", "-")]

if MUSA_AVAILABLE:
    line_vals.append("musa")
    line_names.append("MUSA")
    styles.append(("red", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="Latency (us)",
        plot_name="topk-sigmoid-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, provider):
    gating_output, correction_bias = get_topk_sigmoid_input(num_tokens, num_experts)

    if provider == "torch" or provider == "torch1":

        def fn():
            return torch_topk_sigmoid_native(
                gating_output,
                topk,
                True,
                correction_bias,
            )

    elif provider == "sglang" or provider == "sglang1":

        def fn():
            return sglang_topk_sigmoid(gating_output, topk, True, correction_bias)

    elif provider == "musa" or provider == "musa1":
        if not MUSA_AVAILABLE:
            return (0, 0, 0)

        def fn():
            return musa_topk_sigmoid_fn(gating_output, topk, True, correction_bias)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Simplify configs for CI environment
    if IS_CI:
        test_configs = [(20, 32, 2)]  # Single config for CI
    else:
        test_configs = [
            (20, 256, 4),
            (20, 256, 8),
            (20, 12, 4),
            (20, 12, 1),
            (20, 512, 4),
            (20, 512, 1),
        ]

    for num_tokens, num_experts, topk in test_configs:
        calculate_diff(num_tokens, num_experts, topk)
    benchmark.run(print_data=True)
