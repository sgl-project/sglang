"""Correctness coverage for the MegaMoE TopK -> pre-dispatch PDL chain."""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch
from sglang.srt.layers.moe.topk import (
    biased_topk_jit_kernel_impl,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

FP8_MAX = 448.0
EPS = 1.0e-10


def _reference_quant(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = x.float().unflatten(-1, (-1, group_size))
    absmax = grouped.abs().amax(dim=-1).clamp_min(EPS)
    raw_scale = (absmax / FP8_MAX).contiguous()
    bits = raw_scale.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    inv_scale = ((254 - exponent) << 23).view(torch.float32)
    quant = (
        (grouped * inv_scale.unsqueeze(-1))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .flatten(-2)
    )
    return quant, exponent.to(torch.uint8)


def _allocate_outputs(
    padded_max: int, hidden: int, topk: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_groups = hidden // group_size
    return (
        torch.zeros((padded_max, hidden), dtype=torch.float8_e4m3fn, device="cuda"),
        torch.zeros((padded_max, num_groups // 4), dtype=torch.int32, device="cuda"),
        torch.full((padded_max, topk), -777, dtype=torch.int64, device="cuda"),
        torch.full((padded_max, topk), float("nan"), device="cuda"),
    )


def _reset_outputs(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights = outputs
    buf_x.zero_()
    buf_x_sf.zero_()
    buf_topk_idx.fill_(-777)
    buf_topk_weights.fill_(float("nan"))


def _assert_outputs(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    group_size: int,
) -> None:
    buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights = outputs
    num_tokens, hidden = x.shape
    num_groups = hidden // group_size
    quant_ref, exponent_ref = _reference_quant(x, group_size)

    assert torch.equal(buf_x[:num_tokens].view(torch.int8), quant_ref.view(torch.int8))
    exponent = buf_x_sf.view(torch.uint8).view(buf_x_sf.size(0), num_groups)
    assert torch.equal(exponent[:num_tokens], exponent_ref)
    assert torch.equal(buf_topk_idx[:num_tokens], topk_ids.to(torch.int64))
    assert torch.equal(buf_topk_weights[:num_tokens], topk_weights)
    assert torch.all(buf_topk_idx[num_tokens:] == -1)
    assert torch.all(buf_topk_weights[num_tokens:] == 0)


@pytest.mark.parametrize(
    "num_tokens,padded_max,hidden,group_size,topk",
    [
        (0, 16, 2048, 32, 6),
        (1, 1, 2048, 32, 6),
        (7, 64, 4096, 64, 4),
        (32, 32, 7168, 128, 8),
    ],
)
@torch.inference_mode()
def test_mega_moe_pre_dispatch_chain(
    num_tokens: int,
    padded_max: int,
    hidden: int,
    group_size: int,
    topk: int,
) -> None:
    torch.manual_seed(num_tokens * 1009 + hidden + group_size)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    outputs = _allocate_outputs(padded_max, hidden, topk, group_size)

    # Finish JIT compilation before launching the primary so this test exercises
    # the actual PDL hand-off instead of letting host-side compilation serialize it.
    warmup_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")
    warmup_weights = torch.zeros((num_tokens, topk), device="cuda")
    mega_moe_pre_dispatch(
        x,
        warmup_ids,
        warmup_weights,
        *outputs,
        quant_group_size=group_size,
    )
    torch.cuda.synchronize()
    _reset_outputs(outputs)

    if num_tokens:
        num_experts = 256
        scores = torch.randn((num_tokens, num_experts), device="cuda")
        bias = torch.randn((num_experts,), device="cuda")
        topk_weights, topk_ids = biased_topk_jit_kernel_impl(
            hidden_states=x,
            gating_output=scores,
            correction_bias=bias,
            topk=topk,
            renormalize=True,
            scoring_func="sqrtsoftplus",
        )
    else:
        topk_ids = torch.empty((0, topk), dtype=torch.int32, device="cuda")
        topk_weights = torch.empty((0, topk), device="cuda")

    mega_moe_pre_dispatch(
        x,
        topk_ids,
        topk_weights,
        *outputs,
        quant_group_size=group_size,
    )
    torch.cuda.synchronize()

    _assert_outputs(x, topk_ids, topk_weights, outputs, group_size)


@torch.inference_mode()
def test_mega_moe_pre_dispatch_cuda_graph() -> None:
    num_tokens, padded_max, hidden, group_size, topk = 4, 16, 2048, 32, 6
    torch.manual_seed(17)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((num_tokens, 256), device="cuda")
    bias = torch.randn((256,), device="cuda")
    outputs = _allocate_outputs(padded_max, hidden, topk, group_size)

    # Compile both kernels before capture; replay exercises the PDL primary and
    # late-wait dependent as adjacent graph nodes.
    topk_weights, topk_ids = biased_topk_jit_kernel_impl(
        hidden_states=x,
        gating_output=scores,
        correction_bias=bias,
        topk=topk,
        renormalize=True,
        scoring_func="sqrtsoftplus",
    )
    mega_moe_pre_dispatch(
        x,
        topk_ids,
        topk_weights,
        *outputs,
        quant_group_size=group_size,
    )
    torch.cuda.synchronize()
    _reset_outputs(outputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_weights, topk_ids = biased_topk_jit_kernel_impl(
            hidden_states=x,
            gating_output=scores,
            correction_bias=bias,
            topk=topk,
            renormalize=True,
            scoring_func="sqrtsoftplus",
        )
        mega_moe_pre_dispatch(
            x,
            topk_ids,
            topk_weights,
            *outputs,
            quant_group_size=group_size,
        )
    graph.replay()
    torch.cuda.synchronize()

    _assert_outputs(x, topk_ids, topk_weights, outputs, group_size)
