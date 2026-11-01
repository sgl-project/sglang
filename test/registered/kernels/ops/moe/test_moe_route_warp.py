"""The warp router must select the same experts as an fp64 reference (up to near
ties) with weights within fp32 rounding, keep its own contracts (lowest id wins
exact ties, padding, packed ids, fused shared slots), and srt must dispatch to it."""

from __future__ import annotations

import functools
import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.jit.utils import parallel_compile
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
    "deepseek_v4": (384, 6, "sqrtsoftplus", 1, 1, torch.float32),
}


@pytest.fixture(scope="module", autouse=True)
def _prebuild() -> None:
    # (experts, routed topk, groups, topk_groups, scoring, logits, bias dtype, has_bias, token_bias)
    f32, bf16 = torch.float32, torch.bfloat16
    variants = set()
    for experts, topk, scoring, groups, topk_groups, dtype in CONFIGS.values():
        has_bias = scoring != "softmax"
        variants.add(
            (experts, topk, groups, topk_groups, scoring, dtype, f32, has_bias, False)
        )
        variants.add(
            (experts, topk, groups, topk_groups, scoring, dtype, f32, True, False)
        )
    variants |= {
        (384, 6, 1, 1, "sqrtsoftplus", f32, bf16, True, False),
        (384, 8, 1, 1, "sigmoid", f32, bf16, True, False),
        (384, 8, 1, 1, "sigmoid", f32, torch.float16, True, False),
        (384, 6, 1, 1, "sqrtsoftplus", f32, f32, True, True),
        (128, 8, 1, 1, "softmax", bf16, bf16, True, False),
        (512, 10, 1, 1, "softmax", bf16, bf16, True, False),
        (128, 4, 1, 1, "softmax", bf16, bf16, True, False),
        (128, 8, 1, 1, "softmax", bf16, f32, True, True),
    }
    parallel_compile(
        functools.partial(
            route_warp_module._jit_route_warp_module,
            *v[:8],
            muse_spark=False,
            token_bias=v[8],
        )
        for v in sorted(variants, key=str)
    )


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
        ranking = x if bias is None else x + bias.double()
        act = ranking.softmax(-1)
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
        if renormalize_epsilon > 0:
            weights = weights / (total + renormalize_epsilon)
        else:
            # Zero-sum guard of the routers: an all-zero row stays zero.
            weights = weights / torch.where(total > 0, total, torch.ones_like(total))
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


@pytest.mark.parametrize("shift", [-50, -88, -104, -120, -170])
@pytest.mark.parametrize("zero_bias", [True, False])
def test_sqrtsoftplus_negative_tail(shift: float, zero_bias: bool) -> None:
    # exp(x) is subnormal or zero here, sqrt(softplus(x)) ~ exp(x / 2) is still normal.
    g = torch.Generator(device=DEVICE).manual_seed(-shift)
    scores = torch.randn(1000, 384, device=DEVICE, generator=g) + shift
    bias = torch.randn(384, device=DEVICE, generator=g) * 0.1
    if zero_bias:
        bias.zero_()
    kwargs = dict(topk=6, scoring_func="sqrtsoftplus", renormalize=True)
    w, i = moe_route_warp(scores, bias, **kwargs)
    rw, ri, _, _ = _reference(scores, bias, **kwargs)
    assert torch.equal(i.long(), ri)
    torch.testing.assert_close(w.double(), rw, rtol=1e-5, atol=1e-6)


def test_sqrtsoftplus_uniform_tail() -> None:
    scores = torch.full((4, 384), -88.0, device=DEVICE)
    bias = torch.zeros(384, device=DEVICE)
    w, i = moe_route_warp(scores, bias, topk=6, scoring_func="sqrtsoftplus")
    assert torch.equal(
        i, torch.arange(6, device=DEVICE, dtype=torch.int32).expand(4, 6)
    )
    torch.testing.assert_close(w, torch.full_like(w, 1 / 6))


@pytest.mark.parametrize("renormalize_epsilon", [0.0, 1e-20])
def test_sqrtsoftplus_underflow(renormalize_epsilon: float) -> None:
    # exp(x / 2) underflows: the activation is 0, and an all-zero row stays zero.
    scores = torch.full((6, 384), -1000.0, device=DEVICE)
    scores[1, 7] = -50.0  # one live expert among underflowed ones
    scores[2, [3, 90, 200]] = torch.tensor([-60.0, -20.0, 1.0], device=DEVICE)
    scores[3] = torch.linspace(-300, -100, 384, device=DEVICE)  # partly underflowed
    scores[4, ::2] = float("-inf")
    scores[4, 1::2] = torch.randn(192, device=DEVICE)
    bias = torch.zeros(384, device=DEVICE)
    kwargs = dict(
        topk=6,
        scoring_func="sqrtsoftplus",
        renormalize=True,
        renormalize_epsilon=renormalize_epsilon,
    )
    w, i = moe_route_warp(scores, bias, **kwargs)
    rw, ri, _, _ = _reference(scores, bias, **kwargs)
    assert torch.equal(i.long(), ri)
    torch.testing.assert_close(w.double(), rw, rtol=1e-5, atol=1e-6)
    assert (w[0] == 0).all() and (w[5] == 0).all()
    torch.testing.assert_close(w[1], torch.tensor([1.0, 0, 0, 0, 0, 0], device=DEVICE))


def _softmax_bias(scores: torch.Tensor, dtype=torch.float32):
    g = torch.Generator(device=DEVICE).manual_seed(scores.shape[0] + 17)
    return (torch.randn(scores.shape[1], device=DEVICE, generator=g) * 2).to(dtype)


@pytest.mark.parametrize("config", ["qwen3_moe", "qwen3_5_moe", "gpt_oss"])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "overrides",
    [
        dict(),
        dict(renormalize=False),
        dict(apply_routed_scaling_factor_on_output=True),
        dict(num_fused_shared_experts=1),
        dict(num_fused_shared_experts=1, renormalize=False),
        dict(renormalize_epsilon=1e-20),
    ],
)
def test_softmax_bias(config: str, bias_dtype, overrides) -> None:
    scores, _ = _inputs(config, 600, seed=21)
    bias = _softmax_bias(scores, bias_dtype)
    kwargs = _kwargs(config, **overrides)
    kwargs["topk"] += overrides.get("num_fused_shared_experts", 0)
    w, i = moe_route_warp(scores, bias, **kwargs)
    k_routed = kwargs["topk"] - overrides.get("num_fused_shared_experts", 0)
    _assert_matches(w, i, _reference(scores, bias, **kwargs), k_routed)
    # The bias must change the selection, or this test checks nothing.
    _, i0 = moe_route_warp(scores, None, **kwargs)
    assert not torch.equal(i, i0)
    # Same selection and weights as the Triton router.
    with patch.object(route_warp_module, "covered", lambda *args: False):
        tw, ti = moe_fused_gate(scores, bias, **kwargs)
    assert torch.equal(ti, i)
    torch.testing.assert_close(tw, w, rtol=1e-5, atol=1e-7)


def test_softmax_bias_dispatch() -> None:
    scores, _ = _inputs("qwen3_moe", 64, seed=4)
    bias = _softmax_bias(scores)
    kwargs = _kwargs("qwen3_moe")
    w, i = moe_fused_gate(scores, bias, **kwargs)
    rw, ri = moe_route_warp(scores, bias, **kwargs)
    assert torch.equal(i, ri) and torch.equal(w, rw)


@pytest.mark.parametrize("config", ["kimi_k2", "qwen3_moe"])
def test_strided_bias_falls_back(config: str) -> None:
    # A non-contiguous bias view is not loaded by the kernel: Triton serves it.
    scores, _ = _inputs(config, 64, seed=12)
    num_experts = scores.shape[1]
    g = torch.Generator(device=DEVICE).manual_seed(12)
    full = torch.randn(2 * num_experts, device=DEVICE, generator=g)
    bias = full[::2]
    assert not route_warp_module.bias_layout_ok(bias)
    assert not route_warp_module.bias_layout_ok(full[1 : num_experts + 1])
    kwargs = _kwargs(config)
    w, i = moe_fused_gate(scores, bias, **kwargs)
    _assert_matches(w, i, _reference(scores, bias, **kwargs), kwargs["topk"])


def test_softmax_bias_grouped_falls_back() -> None:
    # Grouped softmax stays on the Triton router, bias included.
    scores, _ = _inputs("qwen3_moe", 64, seed=6)
    bias = _softmax_bias(scores)
    kwargs = _kwargs("qwen3_moe", num_expert_group=8, topk_group=4)
    assert not covered(scores.dtype, 128, 8, "softmax", 0, 8, 4)
    w, i = moe_fused_gate(scores, bias, **kwargs)
    _assert_matches(w, i, _reference(scores, bias, **kwargs), 8)


def test_softmax_token_bias() -> None:
    scores, _ = _inputs("qwen3_moe", 600, seed=8)
    bias = _softmax_bias(scores)
    bias_alt = bias.flip(0).contiguous()
    g = torch.Generator(device=DEVICE).manual_seed(8)
    input_ids = torch.randint(
        0, 4, (600,), device=DEVICE, dtype=torch.int32, generator=g
    )
    kwargs = _kwargs("qwen3_moe")
    token_kwargs = dict(input_ids=input_ids, bias_alt=bias_alt, bias_alt_token_id=2)
    w, i = moe_route_warp(scores, bias, **kwargs, **token_kwargs)
    row_bias = torch.where((input_ids == 2)[:, None], bias_alt, bias)
    _assert_matches(w, i, _reference(scores, row_bias, **kwargs), 8)


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
