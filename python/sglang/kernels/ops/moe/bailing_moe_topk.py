"""Specialized biased grouped top-k kernel for Bailing MoE V3 routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_bailing_moe_topk_module() -> Module:
    return load_jit(
        "bailing_moe_topk",
        cuda_files=["moe/bailing_moe_topk.cuh"],
        cuda_wrappers=[
            ("bailing_moe_biased_grouped_topk", "bailing_moe_biased_grouped_topk")
        ],
    )


@register_custom_op(mutates_args=["topk_weights", "topk_ids"])
def _jit_bailing_moe_biased_grouped_topk_op(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> None:
    module = _jit_bailing_moe_topk_module()
    module.bailing_moe_biased_grouped_topk(
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


def bailing_moe_biased_grouped_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused sigmoid + bias + grouped top-k for Bailing MoE V3.

    Supported shape: 512 experts, 8 groups, topk_group=4, routed top-k <= 8.
    Supports fp32 router logits and bias; outputs fp32 weights and int32
    expert ids.
    """
    assert gating_output.dtype == torch.float32
    assert correction_bias.dtype == torch.float32

    num_tokens = gating_output.shape[0]

    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )

    if num_tokens == 0:
        return topk_weights, topk_ids

    _jit_bailing_moe_biased_grouped_topk_op(
        gating_output.contiguous(),
        correction_bias.contiguous(),
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    return topk_weights, topk_ids
