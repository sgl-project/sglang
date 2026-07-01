"""Correctness tests for the Triton :func:`moe_fused_gate` router.

The Triton kernel is a drop-in reimplementation of the CUDA fused gate for the
ungrouped case (``num_expert_group == 1``). We validate it three ways:

* against an explicit, definition-based torch reference (documents the math),
* against the CUDA JIT kernel it mirrors (:func:`moe_fused_gate_jit`), and
* against the production ``biased_grouped_topk_impl`` for the sigmoid / no-shared
  path that the kernel actually replaces in ``topk.py``.

Comparisons are order-independent: weights are scattered back to a dense
``[M, num_experts + num_shared]`` layout so the per-row column order (and any
tie-break choice) does not matter.
"""

from __future__ import annotations

import sys
from typing import Tuple

import pytest
import torch

from sglang.jit_kernel.moe_fused_gate import moe_fused_gate, moe_fused_gate_jit
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.srt.layers.moe.topk import biased_grouped_topk_impl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"


def _scatter_by_expert(
    weights: torch.Tensor, indices: torch.Tensor, num_columns: int
) -> torch.Tensor:
    """Scatter (weight, id) pairs into a dense ``[M, num_columns]`` tensor.

    Makes the comparison independent of the per-row slot order, so the test does
    not depend on how ties between equal scores are broken.
    """
    dense = torch.zeros(
        (weights.shape[0], num_columns), dtype=torch.float32, device=weights.device
    )
    dense.scatter_(1, indices.long(), weights.float())
    return dense


def _reference_gate(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Definition-based eager reference matching the CUDA fused-gate semantics."""
    if scoring_func == "sigmoid":
        activated = scores.sigmoid()
    else:
        activated = torch.nn.functional.softplus(scores).sqrt()

    biased = activated + bias.unsqueeze(0)
    num_experts = scores.size(1)
    num_routed = topk - num_fused_shared_experts

    # Top-k_routed by biased score; lowest expert id wins on ties (matches kernel).
    bs = biased.size(0)
    work = biased.clone()
    arange = torch.arange(num_experts, device=scores.device).unsqueeze(0)
    routed_idx = torch.empty(bs, num_routed, dtype=torch.int32, device=scores.device)
    routed_wgt = torch.empty(bs, num_routed, dtype=torch.float32, device=scores.device)
    for k in range(num_routed):
        vals, _ = work.max(dim=1, keepdim=True)
        lane = torch.where(work == vals, arange, num_experts + 1)
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
        weights[:, num_routed:] = routed_sum / routed_scaling_factor
        for j in range(num_fused_shared_experts):
            indices[:, num_routed + j] = num_experts + j

    if renormalize:
        norm = torch.where(routed_sum > 0.0, routed_sum, torch.ones_like(routed_sum))
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor
    return weights, indices


def _make_inputs(M: int, num_experts: int, seed: int):
    torch.manual_seed(seed)
    scores = torch.randn(M, num_experts, dtype=torch.float32, device=DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=DEVICE) * 0.5
    return scores, bias


_NUM_EXPERTS = get_ci_test_range([128, 256, 384, 512], [128, 384, 512])
_M = get_ci_test_range([1, 7, 64, 256, 1024], [1, 64, 1024])


@pytest.mark.parametrize("M", _M)
@pytest.mark.parametrize("num_experts", _NUM_EXPERTS)
@pytest.mark.parametrize("topk", [4, 6, 8])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("num_shared", [0, 1])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("apply_scale", [True, False])
def test_moe_fused_gate_matches_reference(
    M: int,
    num_experts: int,
    topk: int,
    scoring_func: str,
    num_shared: int,
    renormalize: bool,
    apply_scale: bool,
) -> None:
    scores, bias = _make_inputs(M, num_experts, seed=num_experts * 100 + topk)
    scale = 2.5

    kwargs = dict(
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_shared,
        renormalize=renormalize,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=apply_scale,
    )
    triton_w, triton_i = moe_fused_gate(scores, bias, **kwargs)
    ref_w, ref_i = _reference_gate(scores, bias, **kwargs)
    torch.cuda.synchronize()

    num_columns = num_experts + num_shared
    torch.testing.assert_close(
        _scatter_by_expert(triton_w, triton_i, num_columns),
        _scatter_by_expert(ref_w, ref_i, num_columns),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "M,num_experts,topk,num_shared,scoring_func",
    [
        # DeepSeek-V4-ish: sqrtsoftplus, ungrouped
        (8192, 256, 6, 0, "sqrtsoftplus"),
        (8192, 384, 6, 0, "sqrtsoftplus"),
        # Kimi-K2 family: 384 experts, sigmoid, ungrouped
        (8192, 384, 8, 0, "sigmoid"),
        # Generic large MoE with a fused shared expert
        (8192, 512, 8, 1, "sigmoid"),
    ],
)
def test_moe_fused_gate_matches_cuda_jit(
    M: int, num_experts: int, topk: int, num_shared: int, scoring_func: str
) -> None:
    """Triton output must match the CUDA JIT kernel it reimplements."""
    scores, bias = _make_inputs(M, num_experts, seed=123)

    kwargs = dict(
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_shared,
        renormalize=True,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=True,
    )
    triton_w, triton_i = moe_fused_gate(scores, bias, **kwargs)
    cuda_w, cuda_i = moe_fused_gate_jit(scores, bias, **kwargs)
    torch.cuda.synchronize()

    num_columns = num_experts + num_shared
    torch.testing.assert_close(
        _scatter_by_expert(triton_w, triton_i, num_columns),
        _scatter_by_expert(cuda_w, cuda_i, num_columns),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("num_experts,topk", [(256, 6), (384, 8), (512, 8)])
@pytest.mark.parametrize("apply_scale", [True, False])
def test_moe_fused_gate_matches_production_impl(
    num_experts: int, topk: int, apply_scale: bool
) -> None:
    """Match production ``biased_grouped_topk_impl`` on the path it replaces.

    The kernel supersedes the ungrouped sigmoid CUDA path in ``topk.py``; that
    reference hardcodes sigmoid and no fused shared expert, and the production
    path always renormalizes (the impl only applies the scaling factor when
    ``renormalize`` is set), so we compare on the renormalized path.
    """
    M = 128
    scores, bias = _make_inputs(M, num_experts, seed=7)
    scale = 2.5

    triton_w, triton_i = moe_fused_gate(
        scores,
        bias,
        topk=topk,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=apply_scale,
    )
    hidden_states = torch.empty((M, 1), dtype=torch.float32, device=DEVICE)
    ref_w, ref_i = biased_grouped_topk_impl(
        hidden_states,
        scores,
        bias,
        topk,
        True,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=apply_scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _scatter_by_expert(triton_w, triton_i, num_experts),
        _scatter_by_expert(ref_w, ref_i, num_experts),
        rtol=1e-4,
        atol=1e-5,
    )


def test_moe_fused_gate_shapes_and_dtypes() -> None:
    """Output shapes/dtypes and renormalized weights for a DeepSeek-V4 config."""
    M, N, K = 64, 256, 8
    scores, bias = _make_inputs(M, N, seed=0)

    w, i = moe_fused_gate(scores, bias, topk=K, scoring_func="sqrtsoftplus")
    assert w.shape == (M, K)
    assert i.shape == (M, K)
    assert w.dtype == torch.float32
    assert i.dtype == torch.int32

    # Selected expert ids are valid (no shared experts here).
    assert (i >= 0).all() and (i < N).all()
    # Renormalized weights sum to 1 per row.
    torch.testing.assert_close(
        w.sum(dim=1), torch.ones(M, device=DEVICE), rtol=1e-4, atol=1e-5
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
