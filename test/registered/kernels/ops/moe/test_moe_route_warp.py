"""The warp router must select like the Triton router (ids, order, padding) for
sigmoid / softmax, the DeepSeek grouped-routing reference, an fp64 sqrtsoftplus
reference (DeepSeek-V4 / V4.1), and srt must dispatch to it."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.moe import moe_route_warp as route_warp_module
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.moe.moe_route_warp import covered, moe_route_warp
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"

# name -> (experts, topk, scoring, groups, topk_groups, logits dtype)
CONFIGS = {
    "deepseek_v3": (256, 8, "sigmoid", 8, 4, torch.float32),
    "kimi_k2": (384, 8, "sigmoid", 1, 1, torch.float32),
    "glm4_5": (160, 8, "sigmoid", 1, 1, torch.float32),
    "glm5": (256, 8, "sigmoid", 1, 1, torch.float32),
    "qwen3_moe": (128, 8, "softmax", 1, 1, torch.bfloat16),
    "qwen3_5_moe": (512, 10, "softmax", 1, 1, torch.bfloat16),
    "gpt_oss": (128, 4, "softmax", 1, 1, torch.bfloat16),
}


def _inputs(config: str, num_tokens: int, seed: int = 0):
    num_experts, _, scoring, _, _, dtype = CONFIGS[config]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    scores = (torch.randn(num_tokens, num_experts, device=DEVICE, generator=g) * 2).to(
        dtype
    )
    bias = None
    if scoring == "sigmoid":
        bias = torch.randn(num_experts, device=DEVICE, generator=g) * 0.1
    return scores, bias


def _kwargs(config: str, **overrides):
    _, topk, scoring, groups, topk_groups, _ = CONFIGS[config]
    kwargs = dict(
        topk=topk,
        scoring_func=scoring,
        renormalize=True,
        routed_scaling_factor=2.5,
        num_expert_group=groups,
        topk_group=topk_groups,
    )
    kwargs.update(overrides)
    return kwargs


def _triton(scores, bias, **kwargs):
    with patch.object(route_warp_module, "covered", lambda *args: False):
        return moe_fused_gate(scores, bias, **kwargs)


def _grouped_reference(scores, bias, topk, num_groups, topk_groups):
    """DeepSeek noaux_tc in fp64; a group scores the sum of its two largest values."""
    act = scores.double().sigmoid()
    biased = act + bias.double()
    num_tokens, num_experts = scores.shape
    group_scores = biased.view(num_tokens, num_groups, -1).topk(2, -1).values.sum(-1)
    keep = torch.zeros_like(group_scores)
    keep.scatter_(1, group_scores.topk(topk_groups, -1).indices, 1)
    keep = keep.repeat_interleave(num_experts // num_groups, dim=1)
    ids = biased.masked_fill(keep == 0, float("-inf")).topk(topk, -1).indices
    weights = act.gather(1, ids)
    return weights / weights.sum(-1, keepdim=True), ids


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("num_tokens", [1, 3, 64, 600, 3000])
def test_matches_triton_router(config: str, num_tokens: int) -> None:
    scores, bias = _inputs(config, num_tokens, seed=num_tokens)
    kwargs = _kwargs(config)
    assert covered(
        scores.dtype,
        scores.shape[1],
        kwargs["topk"],
        kwargs["scoring_func"],
        0,
        kwargs["num_expert_group"],
        kwargs["topk_group"],
    )
    w, i = moe_route_warp(scores, bias, **kwargs)
    tw, ti = _triton(scores, bias, **kwargs)
    # Same ids in the same order; weights within fp32 rounding.
    assert torch.equal(i, ti)
    torch.testing.assert_close(w, tw, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("num_tokens", [1, 64, 3000])
def test_grouped_matches_reference(num_tokens: int) -> None:
    config = "deepseek_v3"
    scores, bias = _inputs(config, num_tokens, seed=7)
    kwargs = _kwargs(config, routed_scaling_factor=1.0)
    w, i = moe_route_warp(scores, bias, **kwargs)
    rw, ri = _grouped_reference(scores, bias, kwargs["topk"], 8, 4)
    assert torch.equal(i.long(), ri)
    torch.testing.assert_close(w.double(), rw, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("config", ["kimi_k2", "glm4_5", "qwen3_moe", "qwen3_5_moe"])
def test_exact_ties_and_nan(config: str) -> None:
    scores, bias = _inputs(config, 64, seed=3)
    if bias is not None:
        bias.zero_()
    tied = (scores.float() * 2).round().div(2).to(scores.dtype)
    nan = scores.clone()
    nan[::3, ::7] = float("nan")
    kwargs = _kwargs(config)
    assert torch.equal(
        moe_route_warp(tied, bias, **kwargs)[1], _triton(tied, bias, **kwargs)[1]
    )
    assert torch.equal(
        moe_route_warp(nan, bias, **kwargs)[1], _triton(nan, bias, **kwargs)[1]
    )


def test_grouped_exact_ties_lowest_id() -> None:
    # All-equal scores: groups 0..3 kept, experts 0..7 chosen.
    scores = torch.zeros(4, 256, device=DEVICE)
    bias = torch.zeros(256, device=DEVICE)
    _, i = moe_route_warp(scores, bias, **_kwargs("deepseek_v3"))
    assert torch.equal(
        i, torch.arange(8, device=DEVICE, dtype=torch.int32).expand(4, 8)
    )


@pytest.mark.parametrize("config", ["deepseek_v3", "kimi_k2", "qwen3_moe"])
@pytest.mark.parametrize(
    "overrides",
    [
        dict(apply_routed_scaling_factor_on_output=True),
        dict(num_fused_shared_experts=1),
        dict(num_fused_shared_experts=1, apply_routed_scaling_factor_on_output=True),
        dict(renormalize=False),
        dict(renormalize_epsilon=1e-20),
    ],
)
def test_options_match_triton(config: str, overrides) -> None:
    scores, bias = _inputs(config, 100, seed=11)
    kwargs = _kwargs(config, **overrides)
    kwargs["topk"] += overrides.get("num_fused_shared_experts", 0)
    w, i = moe_route_warp(scores, bias, **kwargs)
    tw, ti = _triton(scores, bias, **kwargs)
    assert torch.equal(i, ti)
    torch.testing.assert_close(w, tw, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("config", ["kimi_k2", "qwen3_moe"])
def test_padding(config: str) -> None:
    scores, bias = _inputs(config, 48, seed=5)
    kwargs = _kwargs(config)
    n = torch.tensor(29, dtype=torch.int32, device=DEVICE)
    w, i = moe_route_warp(scores, bias, num_token_non_padded=n, **kwargs)
    tw, ti = _triton(scores, bias, num_token_non_padded=n, **kwargs)
    assert torch.equal(i, ti)
    assert (w[29:] == 0).all() and (i[29:] == -1).all()


@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float16])
def test_low_precision_bias(bias_dtype: torch.dtype) -> None:
    scores, bias = _inputs("kimi_k2", 64, seed=9)
    bias = bias.to(bias_dtype)
    kwargs = _kwargs("kimi_k2")
    w, i = moe_route_warp(scores, bias, **kwargs)
    w32, i32 = moe_route_warp(scores, bias.float(), **kwargs)
    assert torch.equal(i, i32) and torch.equal(w, w32)


def test_dispatch() -> None:
    scores, bias = _inputs("kimi_k2", 32, seed=1)
    kwargs = _kwargs("kimi_k2")
    w, i = moe_fused_gate(scores, bias, **kwargs)
    rw, ri = moe_route_warp(scores, bias, **kwargs)
    assert torch.equal(i, ri) and torch.equal(w, rw)


def _sqrtsoftplus_reference(scores, bias, topk):
    """fp64 sqrt(softplus) routing; lowest id wins ties."""
    x = scores.double()
    act = (x.clamp(min=0) + torch.log1p(torch.exp(-x.abs()))).sqrt()
    biased = act + bias.double()
    biased = biased.masked_fill(torch.isnan(biased), -1e30)
    ids = torch.sort(biased, dim=-1, descending=True, stable=True).indices[:, :topk]
    weights = act.gather(1, ids)
    top = biased.topk(topk + 1, -1).values
    near_tie = (top[:, topk - 1] - top[:, topk]).abs() < 1e-6 * top[:, topk - 1].abs()
    return weights / weights.sum(-1, keepdim=True), ids, near_tie


@pytest.mark.parametrize("scale,shift", [(1, 0), (8, 0), (30, 0), (1, -12)])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
def test_sqrtsoftplus_matches_fp64(scale: float, shift: float, bias_dtype) -> None:
    # DeepSeek-V4 / V4.1 routing: 384 experts, top-6, ungrouped, fp32 logits.
    g = torch.Generator(device=DEVICE).manual_seed(int(scale * 10 - shift))
    scores = torch.randn(4000, 384, device=DEVICE, generator=g) * scale + shift
    bias = (torch.randn(384, device=DEVICE, generator=g) * 0.1).to(bias_dtype)
    kwargs = dict(topk=6, scoring_func="sqrtsoftplus", renormalize=True)
    assert covered(scores.dtype, 384, 6, "sqrtsoftplus")
    w, i = moe_route_warp(scores, bias, **kwargs)
    rw, ri, near_tie = _sqrtsoftplus_reference(scores, bias, 6)
    same = (i.long().sort(-1).values == ri.sort(-1).values).all(-1)
    assert bool((same | near_tie).all())
    err = (w.double().gather(1, i.long().argsort(-1)) - rw.gather(1, ri.argsort(-1)))[
        same
    ].abs()
    assert err.max().item() < 1e-6


def test_sqrtsoftplus_conventions() -> None:
    g = torch.Generator(device=DEVICE).manual_seed(2)
    scores = torch.randn(64, 384, device=DEVICE, generator=g) * 2
    bias = torch.zeros(384, device=DEVICE)
    kwargs = dict(topk=6, scoring_func="sqrtsoftplus", renormalize=True)
    # exact ties: lowest id wins, like the Triton router
    tied = (scores * 2).round() / 2
    assert torch.equal(
        moe_route_warp(tied, bias, **kwargs)[1], _triton(tied, bias, **kwargs)[1]
    )
    # NaN logits rank last, like the DeepSeek-V4 Triton router
    nan = scores.clone()
    nan[::3, ::7] = float("nan")
    assert torch.equal(
        moe_route_warp(nan, bias, **kwargs)[1], _triton(nan, bias, **kwargs)[1]
    )
    # padding and the packed FlashInfer ids
    n = torch.tensor(40, dtype=torch.int32, device=DEVICE)
    packed = torch.empty(64, 6, dtype=torch.int32, device=DEVICE)
    w, i = moe_route_warp(
        scores, bias, num_token_non_padded=n, packed_out=packed, **kwargs
    )
    assert (w[40:] == 0).all() and (i[40:] == -1).all()
    bits = w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.equal(packed, (i << 16) | bits)


def test_sqrtsoftplus_dispatch() -> None:
    scores = torch.randn(32, 384, device=DEVICE)
    bias = torch.randn(384, device=DEVICE) * 0.1
    kwargs = dict(topk=6, scoring_func="sqrtsoftplus", renormalize=True)
    w, i = moe_fused_gate(scores, bias, sqrtsoftplus_log1p=True, **kwargs)
    rw, ri = moe_route_warp(scores, bias, **kwargs)
    assert torch.equal(i, ri) and torch.equal(w, rw)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_token_bias_matches_fp64(id_dtype) -> None:
    # DeepSeek-V4.1 VL: image-token rows rank with the vision correction bias.
    g = torch.Generator(device=DEVICE).manual_seed(3)
    scores = torch.randn(3000, 384, device=DEVICE, generator=g)
    bias = torch.randn(384, device=DEVICE, generator=g) * 0.1
    bias_alt = torch.randn(384, device=DEVICE, generator=g) * 0.1
    input_ids = torch.randint(0, 4, (3000,), device=DEVICE, dtype=id_dtype, generator=g)
    kwargs = dict(topk=6, scoring_func="sqrtsoftplus", renormalize=True)
    token_kwargs = dict(input_ids=input_ids, bias_alt=bias_alt, bias_alt_token_id=2)
    assert covered(scores.dtype, 384, 6, "sqrtsoftplus")
    w, i = moe_route_warp(scores, bias, **kwargs, **token_kwargs)
    row_bias = torch.where((input_ids == 2)[:, None], bias_alt, bias)
    rw, ri, near_tie = _sqrtsoftplus_reference(scores, row_bias, 6)
    same = (i.long().sort(-1).values == ri.sort(-1).values).all(-1)
    assert bool((same | near_tie).all())
    err = (w.double().gather(1, i.long().argsort(-1)) - rw.gather(1, ri.argsort(-1)))[
        same
    ].abs()
    assert err.max().item() < 1e-6
    fw, fi = moe_fused_gate(
        scores, bias, sqrtsoftplus_log1p=True, **kwargs, **token_kwargs
    )
    assert torch.equal(fi, i) and torch.equal(fw, w)


def test_not_covered() -> None:
    fp32 = torch.float32
    assert not covered(fp32, 96 + 16, 8, "sigmoid")  # not a multiple of 32
    assert not covered(fp32, 64, 6, "sigmoid")  # small expert counts stay on Triton
    assert not covered(fp32, 896, 16, "sqrtsoftplus")  # Kimi-K3 stays on radix
    assert not covered(fp32, 896, 8, "sigmoid", num_expert_group=3, topk_group=2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
