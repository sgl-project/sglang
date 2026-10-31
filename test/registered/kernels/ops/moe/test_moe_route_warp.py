"""The warp router must select the same experts as an fp64 reference (up to near
ties) with weights within fp32 rounding, keep its own contracts (lowest id wins
exact ties, padding, packed ids, fused shared slots), and srt must dispatch to it."""

from __future__ import annotations

import sys

import pytest
import torch

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
    "deepseek_v4": (384, 6, "sqrtsoftplus", 1, 1, torch.float32),
}


def _inputs(config: str, num_tokens: int, seed: int = 0):
    num_experts, _, scoring, _, _, dtype = CONFIGS[config]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    scores = torch.randn(num_tokens, num_experts, device=DEVICE, generator=g) * 2
    bias = None
    if scoring != "softmax":
        bias = torch.randn(num_experts, device=DEVICE, generator=g) * 0.1
    return scores.to(dtype), bias


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


def _reference(
    scores,
    bias,
    *,
    topk,
    scoring_func,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
    num_expert_group=1,
    topk_group=1,
    num_fused_shared_experts=0,
    renormalize_epsilon=0.0,
):
    """fp64 routing: (weights, ids, near_tie rows, ranking scores); ties go to the lowest id."""
    x = scores.double()
    if scoring_func == "softmax":
        act = x.softmax(-1)
        ranking = x
    else:
        if scoring_func == "sigmoid":
            act = x.sigmoid()
        else:
            act = (x.clamp(min=0) + torch.log1p(torch.exp(-x.abs()))).sqrt()
        ranking = act + bias.double()
    num_tokens, num_experts = x.shape
    if num_expert_group > 1:
        groups = ranking.view(num_tokens, num_expert_group, -1)
        keep = torch.zeros(num_tokens, num_expert_group, device=x.device)
        keep.scatter_(
            1, groups.topk(2, -1).values.sum(-1).topk(topk_group, -1).indices, 1
        )
        keep = keep.repeat_interleave(num_experts // num_expert_group, dim=1)
        ranking = ranking.masked_fill(keep == 0, float("-inf"))
    k = topk - num_fused_shared_experts
    order = torch.sort(ranking, dim=-1, descending=True, stable=True)
    ids = order.indices[:, :k]
    gap = (order.values[:, k - 1] - order.values[:, k]).abs()
    near_tie = gap <= 1e-6 * order.values[:, k - 1].abs().clamp(min=1)
    weights = act.gather(1, ids)
    if renormalize:
        total = weights.sum(-1, keepdim=True)
        weights = weights / (total + renormalize_epsilon)
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor
    if num_fused_shared_experts:
        shared = (
            1.0
            if apply_routed_scaling_factor_on_output
            else 1.0 / routed_scaling_factor
        )
        weights = torch.cat(
            [weights, weights.new_full((num_tokens, num_fused_shared_experts), shared)],
            1,
        )
        extra = torch.arange(num_fused_shared_experts, device=x.device) + num_experts
        ids = torch.cat([ids, extra.expand(num_tokens, -1)], 1)
    return weights, ids, near_tie, ranking


def _assert_matches(w, i, ref, k_routed):
    rw, ri, near_tie, ranking = ref
    i = i.long()
    same = (i[:, :k_routed].sort(-1).values == ri[:, :k_routed].sort(-1).values).all(-1)
    assert bool((same | near_tie).all())
    assert torch.equal(i[:, k_routed:], ri[:, k_routed:])
    # Descending ranking order.
    ranked = ranking.gather(1, i[:, :k_routed])
    assert bool((ranked[:, :-1] >= ranked[:, 1:] - 1e-6).all())
    got = w.double().gather(1, i.argsort(-1))
    want = rw.gather(1, ri.argsort(-1))
    torch.testing.assert_close(got[same], want[same], rtol=1e-5, atol=1e-6)


def _check(scores, bias, **kwargs):
    w, i = moe_route_warp(scores, bias, **kwargs)
    k_routed = kwargs["topk"] - kwargs.get("num_fused_shared_experts", 0)
    _assert_matches(w, i, _reference(scores, bias, **kwargs), k_routed)


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("num_tokens", [1, 3, 64, 600, 5000])
def test_matches_reference(config: str, num_tokens: int) -> None:
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
    _check(scores, bias, **kwargs)


@pytest.mark.parametrize(
    "config", ["deepseek_v3", "kimi_k2", "qwen3_moe", "deepseek_v4"]
)
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
def test_options_match_reference(config: str, overrides) -> None:
    scores, bias = _inputs(config, 100, seed=11)
    kwargs = _kwargs(config, **overrides)
    kwargs["topk"] += overrides.get("num_fused_shared_experts", 0)
    _check(scores, bias, **kwargs)


@pytest.mark.parametrize("scale,shift", [(1, 0), (8, 0), (30, 0), (1, -12)])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
def test_sqrtsoftplus_logit_ranges(scale: float, shift: float, bias_dtype) -> None:
    # DeepSeek-V4: activation accuracy across the logit range.
    g = torch.Generator(device=DEVICE).manual_seed(int(scale * 10 - shift))
    scores = torch.randn(4000, 384, device=DEVICE, generator=g) * scale + shift
    bias = (torch.randn(384, device=DEVICE, generator=g) * 0.1).to(bias_dtype)
    _check(scores, bias, topk=6, scoring_func="sqrtsoftplus", renormalize=True)


@pytest.mark.parametrize(
    "config", ["kimi_k2", "glm4_5", "qwen3_moe", "qwen3_5_moe", "deepseek_v4"]
)
def test_exact_ties_lowest_id(config: str) -> None:
    scores, bias = _inputs(config, 64, seed=3)
    if bias is not None:
        bias.zero_()
    tied = (scores.float() * 2).round().div(2).to(scores.dtype)
    kwargs = _kwargs(config)
    _, i = moe_route_warp(tied, bias, **kwargs)
    _, ri, _, _ = _reference(tied, bias, **kwargs)
    assert torch.equal(i.long(), ri)


def test_grouped_exact_ties_lowest_id() -> None:
    # All-equal scores: groups 0..3 kept, experts 0..7 chosen.
    scores = torch.zeros(4, 256, device=DEVICE)
    bias = torch.zeros(256, device=DEVICE)
    _, i = moe_route_warp(scores, bias, **_kwargs("deepseek_v3"))
    assert torch.equal(
        i, torch.arange(8, device=DEVICE, dtype=torch.int32).expand(4, 8)
    )


@pytest.mark.parametrize("config", ["kimi_k2", "qwen3_moe", "deepseek_v4"])
def test_padding_and_packed(config: str) -> None:
    scores, bias = _inputs(config, 48, seed=5)
    kwargs = _kwargs(config)
    n = torch.tensor(29, dtype=torch.int32, device=DEVICE)
    packed = torch.empty(48, kwargs["topk"], dtype=torch.int32, device=DEVICE)
    w, i = moe_route_warp(
        scores, bias, num_token_non_padded=n, packed_out=packed, **kwargs
    )
    _assert_matches(
        w[:29], i[:29], _reference(scores[:29], bias, **kwargs), kwargs["topk"]
    )
    assert (w[29:] == 0).all() and (i[29:] == -1).all()
    bits = w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.equal(packed, (i << 16) | bits)


@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float16])
def test_low_precision_bias(bias_dtype: torch.dtype) -> None:
    scores, bias = _inputs("kimi_k2", 64, seed=9)
    bias = bias.to(bias_dtype)
    kwargs = _kwargs("kimi_k2")
    w, i = moe_route_warp(scores, bias, **kwargs)
    w32, i32 = moe_route_warp(scores, bias.float(), **kwargs)
    assert torch.equal(i, i32) and torch.equal(w, w32)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_token_bias(id_dtype) -> None:
    # DeepSeek-V4.1 VL: image-token rows rank with the vision correction bias.
    g = torch.Generator(device=DEVICE).manual_seed(3)
    scores = torch.randn(3000, 384, device=DEVICE, generator=g)
    bias = torch.randn(384, device=DEVICE, generator=g) * 0.1
    bias_alt = torch.randn(384, device=DEVICE, generator=g) * 0.1
    input_ids = torch.randint(0, 4, (3000,), device=DEVICE, dtype=id_dtype, generator=g)
    kwargs = dict(topk=6, scoring_func="sqrtsoftplus", renormalize=True)
    token_kwargs = dict(input_ids=input_ids, bias_alt=bias_alt, bias_alt_token_id=2)
    w, i = moe_route_warp(scores, bias, **kwargs, **token_kwargs)
    row_bias = torch.where((input_ids == 2)[:, None], bias_alt, bias)
    _assert_matches(w, i, _reference(scores, row_bias, **kwargs), 6)
    fw, fi = moe_fused_gate(
        scores, bias, sqrtsoftplus_log1p=True, **kwargs, **token_kwargs
    )
    assert torch.equal(fi, i) and torch.equal(fw, w)


@pytest.mark.parametrize(
    "config", ["deepseek_v3", "kimi_k2", "qwen3_moe", "deepseek_v4"]
)
def test_dispatch(config: str) -> None:
    scores, bias = _inputs(config, 32, seed=1)
    kwargs = _kwargs(config)
    w, i = moe_fused_gate(scores, bias, **kwargs)
    rw, ri = moe_route_warp(scores, bias, **kwargs)
    assert torch.equal(i, ri) and torch.equal(w, rw)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
