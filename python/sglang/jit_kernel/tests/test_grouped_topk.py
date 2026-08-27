import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.grouped_topk import grouped_topk as jit_grouped_topk
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.srt.layers.moe.topk import biased_grouped_topk_impl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


CORRECTNESS_CASES = get_ci_test_range(
    full_range=list(
        itertools.product(
            [1, 17, 128],
            [16, 32, 64, 128, 192, 256, 384, 512],
            [1, 2, 3, 4, 5, 6, 7, 8],
        )
    ),
    ci_range=[
        (1, 16, 3),  # smallest non-power-of-two topk
        (17, 128, 6),  # Nemotron-3-Nano shape that exposed the bug
        (128, 192, 8),  # Hunyuan-3 shape, power-of-two topk sanity case
        (33, 512, 7),  # largest expert-count tier with non-power-of-two topk
    ],
)


def _make_inputs(num_tokens: int, num_experts: int, seed: int):
    torch.manual_seed(seed)
    hidden_states = torch.empty((num_tokens, 1), dtype=torch.float32, device="cuda")
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1
    return hidden_states, gating_output, correction_bias


def _scatter_by_expert(
    weights: torch.Tensor, ids: torch.Tensor, num_experts: int
) -> torch.Tensor:
    dense = torch.zeros(
        (weights.shape[0], num_experts), dtype=torch.float32, device=weights.device
    )
    dense.scatter_(1, ids.long(), weights)
    return dense


@pytest.mark.parametrize("num_tokens,num_experts,topk", CORRECTNESS_CASES)
def test_grouped_topk_renormalize_matches_reference(
    num_tokens: int, num_experts: int, topk: int
) -> None:
    hidden_states, gating_output, correction_bias = _make_inputs(
        num_tokens, num_experts, seed=1000 + num_experts * 10 + topk
    )
    scaling_factor = 2.826 if (num_experts, topk) == (192, 8) else 1.0

    topk_weights, topk_ids = jit_grouped_topk(
        gating_output,
        correction_bias,
        1,
        1,
        topk,
        True,
        scaling_factor,
    )
    ref_weights, ref_ids = biased_grouped_topk_impl(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        True,
        1,
        1,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _scatter_by_expert(topk_weights, topk_ids, num_experts),
        _scatter_by_expert(ref_weights, ref_ids, num_experts),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        topk_weights.sum(dim=-1),
        torch.full((num_tokens,), scaling_factor, dtype=torch.float32, device="cuda"),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("topk", [3, 5, 6, 7])
def test_grouped_topk_non_power_of_two_renormalize(topk: int) -> None:
    hidden_states, gating_output, correction_bias = _make_inputs(
        num_tokens=64, num_experts=128, seed=2000 + topk
    )

    topk_weights, topk_ids = jit_grouped_topk(
        gating_output,
        correction_bias,
        1,
        1,
        topk,
        True,
        1.0,
    )
    ref_weights, ref_ids = biased_grouped_topk_impl(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        True,
        1,
        1,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _scatter_by_expert(topk_weights, topk_ids, 128),
        _scatter_by_expert(ref_weights, ref_ids, 128),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        topk_weights.sum(dim=-1),
        torch.ones((64,), dtype=torch.float32, device="cuda"),
        rtol=1e-5,
        atol=1e-6,
    )


def test_grouped_topk_negative_choice_scores_match_reference() -> None:
    hidden_states, gating_output, correction_bias = _make_inputs(
        num_tokens=64, num_experts=128, seed=23758
    )
    correction_bias.fill_(-2.0)

    topk_weights, topk_ids = jit_grouped_topk(
        gating_output,
        correction_bias,
        1,
        1,
        6,
        True,
        1.0,
    )
    ref_weights, ref_ids = biased_grouped_topk_impl(
        hidden_states,
        gating_output,
        correction_bias,
        6,
        True,
        1,
        1,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _scatter_by_expert(topk_weights, topk_ids, 128),
        _scatter_by_expert(ref_weights, ref_ids, 128),
        rtol=1e-5,
        atol=1e-6,
    )


def test_grouped_topk_without_renormalize_matches_reference() -> None:
    hidden_states, gating_output, correction_bias = _make_inputs(
        num_tokens=64, num_experts=128, seed=3006
    )

    topk_weights, topk_ids = jit_grouped_topk(
        gating_output,
        correction_bias,
        1,
        1,
        6,
        False,
        1.0,
    )
    ref_weights, ref_ids = biased_grouped_topk_impl(
        hidden_states,
        gating_output,
        correction_bias,
        6,
        False,
        1,
        1,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _scatter_by_expert(topk_weights, topk_ids, 128),
        _scatter_by_expert(ref_weights, ref_ids, 128),
        rtol=1e-5,
        atol=1e-6,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
