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

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate, moe_fused_gate_jit
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


# Keep CI coverage representative without exploding into a large cartesian grid.
_REFERENCE_CASES = get_ci_test_range(
    [
        (1, 128, 4, "sigmoid", 0, True, True),
        (7, 256, 6, "sqrtsoftplus", 1, False, False),
        (64, 384, 8, "sigmoid", 0, False, True),
        (256, 384, 6, "sqrtsoftplus", 0, True, False),
        (1024, 128, 4, "sigmoid", 1, False, False),
        (1024, 512, 8, "sqrtsoftplus", 1, True, True),
        (7, 512, 4, "sigmoid", 1, True, False),
        (256, 256, 8, "sqrtsoftplus", 0, False, True),
    ],
    [
        (1, 128, 4, "sigmoid", 0, True, True),
        (7, 256, 6, "sqrtsoftplus", 1, False, False),
        (64, 384, 8, "sigmoid", 0, False, True),
        (256, 384, 6, "sqrtsoftplus", 0, True, False),
        (1024, 128, 4, "sigmoid", 1, False, False),
        (1024, 512, 8, "sqrtsoftplus", 1, True, True),
    ],
)


@pytest.mark.parametrize(
    "M,num_experts,topk,scoring_func,num_shared,renormalize,apply_scale",
    _REFERENCE_CASES,
)
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
    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

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


def _reference_softmax(
    gating: torch.Tensor, topk: int, renormalize: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain-softmax topk reference (the AOT ``topk_softmax`` semantics)."""
    num_experts = gating.size(1)
    probs = torch.softmax(gating.float(), dim=-1)
    work = gating.float().clone()
    arange = torch.arange(num_experts, device=gating.device).unsqueeze(0)
    M = gating.size(0)
    idx = torch.empty(M, topk, dtype=torch.int32, device=gating.device)
    wgt = torch.empty(M, topk, dtype=torch.float32, device=gating.device)
    for k in range(topk):
        vals, _ = work.max(dim=1, keepdim=True)
        lane = torch.where(work == vals, arange, num_experts + 1)
        winner = lane.min(dim=1).values.to(torch.int32)
        idx[:, k] = winner
        wgt[:, k] = probs.gather(1, winner.long().unsqueeze(1)).squeeze(1)
        work.scatter_(1, winner.long().unsqueeze(1), float("-inf"))
    if renormalize:
        wgt = wgt / wgt.sum(dim=1, keepdim=True)
    return wgt, idx


_SOFTMAX_AOT_CASES = get_ci_test_range(
    [
        (1, 128, 4, True, torch.float32),
        (1, 256, 8, False, torch.float32),
        (200, 128, 4, False, torch.bfloat16),
        (200, 256, 8, True, torch.bfloat16),
        (1024, 512, 6, False, torch.float32),
        (1024, 512, 6, True, torch.bfloat16),
    ],
    [
        (1, 128, 4, True, torch.float32),
        (200, 256, 8, False, torch.bfloat16),
        (1024, 512, 6, False, torch.float32),
    ],
)


@pytest.mark.parametrize(
    "M,num_experts,topk,renormalize,dtype",
    _SOFTMAX_AOT_CASES,
)
def test_moe_fused_gate_softmax_matches_aot(
    M: int, num_experts: int, topk: int, renormalize: bool, dtype: torch.dtype
) -> None:
    """Triton softmax path matches the AOT ``topk_softmax`` it replaces in fused_topk."""
    sgl_kernel = pytest.importorskip("sgl_kernel")
    torch.manual_seed(num_experts * 13 + topk)
    gating = torch.randn(M, num_experts, dtype=dtype, device=DEVICE) * 2.0
    zero_bias = torch.zeros(num_experts, dtype=torch.float32, device=DEVICE)

    tri_w, tri_i = moe_fused_gate(
        gating, zero_bias, topk=topk, scoring_func="softmax", renormalize=renormalize
    )
    ref_w, ref_i = _reference_softmax(gating, topk, renormalize)

    aot_w = torch.empty(M, topk, dtype=torch.float32, device=DEVICE)
    aot_i = torch.empty(M, topk, dtype=torch.int32, device=DEVICE)
    sgl_kernel.topk_softmax(aot_w, aot_i, gating, renormalize)
    torch.cuda.synchronize()

    dense_tri = _scatter_by_expert(tri_w, tri_i, num_experts)
    torch.testing.assert_close(
        dense_tri, _scatter_by_expert(ref_w, ref_i, num_experts), rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        dense_tri, _scatter_by_expert(aot_w, aot_i, num_experts), rtol=1e-3, atol=1e-3
    )


_SIGMOID_AOT_CASES = get_ci_test_range(
    [
        (1, 128, 4, True, True, torch.float32),
        (1, 256, 8, False, True, torch.float32),
        (200, 128, 4, False, False, torch.bfloat16),
        (200, 256, 8, True, False, torch.bfloat16),
        (1024, 128, 4, True, False, torch.float32),
        (1024, 256, 8, False, True, torch.bfloat16),
    ],
    [
        (1, 128, 4, True, True, torch.float32),
        (200, 128, 4, False, False, torch.bfloat16),
        (200, 256, 8, True, False, torch.bfloat16),
        (1024, 256, 8, False, True, torch.float32),
    ],
)


@pytest.mark.parametrize(
    "M,num_experts,topk,renormalize,with_bias,dtype",
    _SIGMOID_AOT_CASES,
)
def test_moe_fused_gate_sigmoid_matches_aot(
    M: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    with_bias: bool,
    dtype: torch.dtype,
) -> None:
    """Triton sigmoid path matches the AOT ``topk_sigmoid`` it replaces in fused_topk."""
    sgl_kernel = pytest.importorskip("sgl_kernel")
    torch.manual_seed(num_experts * 17 + topk)
    gating = torch.randn(M, num_experts, dtype=dtype, device=DEVICE) * 2.0
    bias = (
        torch.randn(num_experts, dtype=torch.float32, device=DEVICE) * 0.5
        if with_bias
        else torch.zeros(num_experts, dtype=torch.float32, device=DEVICE)
    )

    tri_w, tri_i = moe_fused_gate(
        gating, bias, topk=topk, scoring_func="sigmoid", renormalize=renormalize
    )
    aot_w = torch.empty(M, topk, dtype=torch.float32, device=DEVICE)
    aot_i = torch.empty(M, topk, dtype=torch.int32, device=DEVICE)
    sgl_kernel.topk_sigmoid(
        aot_w, aot_i, gating, renormalize, bias if with_bias else None
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _scatter_by_expert(tri_w, tri_i, num_experts),
        _scatter_by_expert(aot_w, aot_i, num_experts),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "num_experts,num_expert_group,topk_group,topk",
    [
        (256, 8, 4, 8),  # DeepSeek-V3
        (128, 8, 4, 6),
        (256, 4, 2, 8),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_fused_gate_grouped_matches_production_impl(
    num_experts: int,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    dtype: torch.dtype,
) -> None:
    """Grouped Triton routing must match the definitional biased_grouped_topk_impl.

    The kernel adds DeepSeek-V3 grouped routing (per-group top-2-sum group scores,
    keep topk_group groups, then top-k within). biased_grouped_topk_impl is the
    eager reference the production grouped path is defined against.
    """
    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

    M = 256
    torch.manual_seed(num_experts * 7 + num_expert_group * 13 + topk)
    gating = torch.randn(M, num_experts, dtype=dtype, device=DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=DEVICE) * 0.5
    hidden = torch.randn(M, 16, dtype=dtype, device=DEVICE)

    tri_w, tri_i = moe_fused_gate(
        gating,
        bias,
        topk=topk,
        scoring_func="sigmoid",
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )
    ref_w, ref_i = biased_grouped_topk_impl(
        hidden,
        gating,
        bias,
        topk,
        True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _scatter_by_expert(tri_w, tri_i, num_experts),
        _scatter_by_expert(ref_w, ref_i, num_experts),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "num_experts,num_expert_group,topk_group,topk,num_fused_shared_experts",
    [
        (256, 8, 4, 8, 0),  # DeepSeek-V3
        (256, 8, 4, 9, 1),  # DeepSeek-V3 + one fused shared expert
    ],
)
def test_grouped_dispatch_flag_matches_default(
    num_experts: int,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int,
) -> None:
    """The opt-in SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK dispatch must match the
    default grouped path (flashinfer/AOT) that biased_grouped_topk_gpu selects when
    the flag is off. This covers the wiring, not just the raw kernel — validated
    bit-exact on DeepSeek-V3.2 e2e; here we assert parity against the default path.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu

    M = 256
    torch.manual_seed(num_experts * 3 + num_expert_group * 5 + topk)
    # fp32 gating: both the default (flashinfer upcasts to fp32) and the Triton
    # dispatch (also upcasts) operate on the same fp32 scores, so no bf16
    # borderline-expert divergence is expected.
    gating = torch.randn(M, num_experts, dtype=torch.float32, device=DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=DEVICE) * 0.5
    hidden = torch.randn(M, 16, dtype=torch.float32, device=DEVICE)

    kwargs = dict(
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=False,
    )
    with envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.override(False):
        def_w, def_i = biased_grouped_topk_gpu(
            hidden, gating, bias, topk, True, **kwargs
        )
    with envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.override(True):
        jit_w, jit_i = biased_grouped_topk_gpu(
            hidden, gating, bias, topk, True, **kwargs
        )
    torch.cuda.synchronize()

    # Compare routed experts only (shared-expert slot ids are placeholders the
    # downstream fusion overwrites; the routed selection + weights are what matter).
    topk_routed = topk - num_fused_shared_experts
    torch.testing.assert_close(
        _scatter_by_expert(def_w[:, :topk_routed], def_i[:, :topk_routed], num_experts),
        _scatter_by_expert(jit_w[:, :topk_routed], jit_i[:, :topk_routed], num_experts),
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
