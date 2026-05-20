from __future__ import annotations

import sys
from typing import Tuple

import pytest
import torch

from sglang.jit_kernel.moe_fused_gate import (
    moe_fused_gate,
    moe_fused_gate_jit,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-unit-1-gpu-large")


def _torch_reference(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager torch reference matching the CUDA JIT semantics."""
    if scoring_func == "sigmoid":
        activated = scores.sigmoid()
    else:
        activated = torch.nn.functional.softplus(scores).sqrt()

    biased = activated + bias.unsqueeze(0)
    num_experts = scores.size(1)
    num_routed = topk - num_fused_shared_experts

    # Top-k_routed by biased score; tie-break by lower expert id (matches the kernel).
    bs = biased.size(0)
    work = biased.clone()
    routed_idx = torch.empty(bs, num_routed, dtype=torch.int32, device=scores.device)
    routed_wgt = torch.empty(bs, num_routed, dtype=torch.float32, device=scores.device)
    for k in range(num_routed):
        vals, _ = work.max(dim=1, keepdim=True)
        is_max = work == vals
        # Smallest expert id wins on ties.
        lane = torch.where(
            is_max,
            torch.arange(num_experts, device=scores.device).unsqueeze(0),
            torch.tensor(num_experts + 1, device=scores.device),
        )
        winner = lane.min(dim=1).values.to(torch.int32)
        routed_idx[:, k] = winner
        routed_wgt[:, k] = activated.gather(1, winner.long().unsqueeze(1)).squeeze(1)
        work.scatter_(1, winner.long().unsqueeze(1), float("-inf"))

    routed_sum = routed_wgt.sum(dim=1, keepdim=True)
    weights = torch.empty(bs, topk, dtype=torch.float32, device=scores.device)
    indices = torch.empty(bs, topk, dtype=torch.int32, device=scores.device)
    weights[:, :num_routed] = routed_wgt
    indices[:, :num_routed] = routed_idx
    if num_fused_shared_experts > 0:
        shared_w = routed_sum / routed_scaling_factor
        weights[:, num_routed:] = shared_w
        for j in range(num_fused_shared_experts):
            indices[:, num_routed + j] = num_experts + j

    if renormalize:
        norm = torch.where(routed_sum > 0.0, routed_sum, torch.ones_like(routed_sum))
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor
    return weights, indices


# Power-of-two num_experts that the CUDA reference supports; 128-512 is the
# sweet-spot range for big MoEs (DeepSeek V3/V4 = 256, Kimi K2 ??? 384/512).
_NUM_EXPERTS_FULL = [128, 256, 384, 512]
_NUM_EXPERTS_CI = [128, 256, 512]
_M_FULL = [1, 7, 64, 256, 1024]
_M_CI = [1, 64, 1024]


@pytest.mark.parametrize("M", get_ci_test_range(_M_FULL, _M_CI))
@pytest.mark.parametrize(
    "num_experts", get_ci_test_range(_NUM_EXPERTS_FULL, _NUM_EXPERTS_CI)
)
@pytest.mark.parametrize("topk", [4, 6, 8])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("num_shared", [0, 1])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("apply_scale", [True, False])
def test_router_triton_matches_reference(
    M: int,
    num_experts: int,
    topk: int,
    scoring_func: str,
    num_shared: int,
    renormalize: bool,
    apply_scale: bool,
) -> None:
    if topk <= num_shared:
        pytest.skip("topk must be > num_fused_shared_experts")

    torch.manual_seed(0)
    device = "cuda:0"
    scores = torch.randn(M, num_experts, dtype=torch.float32, device=device) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.5
    scale = 2.5

    triton_w, triton_i = moe_fused_gate(
        scores,
        bias,
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_shared,
        renormalize=renormalize,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=apply_scale,
    )

    ref_w, ref_i = _torch_reference(
        scores,
        bias,
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_shared,
        renormalize=renormalize,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=apply_scale,
    )

    assert torch.equal(
        triton_i, ref_i
    ), f"Indices mismatch: first 5 differing slots {(triton_i != ref_i).nonzero()[:5].tolist()}"
    torch.testing.assert_close(triton_w, ref_w, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "M,num_experts,topk,num_shared,scoring_func",
    [
        # DeepSeek V4-ish: 256 experts, sqrtsoftplus, no group
        (8192, 256, 6, 0, "sqrtsoftplus"),
        (8192, 384, 6, 0, "sqrtsoftplus"),
        # Kimi K2 family ish: 384 experts, sigmoid, no group
        (8192, 384, 8, 0, "sigmoid"),
        # Generic large MoE w/ a fused shared expert
        (8192, 512, 8, 1, "sigmoid"),
    ],
)
def test_router_triton_matches_cuda_jit(
    M: int, num_experts: int, topk: int, num_shared: int, scoring_func: str
) -> None:
    """Sanity check that router_triton agrees with the CUDA JIT reference."""
    torch.manual_seed(123)
    device = "cuda:0"
    scores = torch.randn(M, num_experts, dtype=torch.float32, device=device) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.5
    scale = 2.5

    triton_w, triton_i = moe_fused_gate(
        scores,
        bias,
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_shared,
        renormalize=True,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=True,
    )
    cuda_w, cuda_i = moe_fused_gate_jit(
        scores,
        bias,
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_shared,
        renormalize=True,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=True,
    )
    assert torch.equal(triton_i, cuda_i)
    torch.testing.assert_close(triton_w, cuda_w, rtol=1e-4, atol=1e-5)


def test_router_triton_shapes_and_dtypes() -> None:
    """Sanity-check output shapes and dtypes for a representative DeepSeek-V4 config."""
    torch.manual_seed(0)
    device = "cuda:0"
    M, N, K = 64, 256, 8
    scores = torch.randn(M, N, dtype=torch.float32, device=device)
    bias = torch.randn(N, dtype=torch.float32, device=device)

    w, i = moe_fused_gate(scores, bias, topk=K, scoring_func="sqrtsoftplus")
    assert w.shape == (M, K)
    assert i.shape == (M, K)
    assert w.dtype == torch.float32
    assert i.dtype == torch.int32

    # Selected expert ids are in [0, N).
    assert (i >= 0).all() and (i < N).all()
    # Renormalized weights sum to 1 (no shared experts).
    row_sums = w.sum(dim=1)
    torch.testing.assert_close(
        row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-5
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
