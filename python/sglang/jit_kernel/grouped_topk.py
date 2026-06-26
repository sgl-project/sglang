"""Fused grouped top-k kernel for MoE routing (single-group, sigmoid scoring)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_grouped_topk_module() -> Module:
    return load_jit(
        "grouped_topk",
        cuda_files=["moe/grouped_topk.cuh"],
        cuda_wrappers=[("grouped_topk", "grouped_topk")],
    )


@register_custom_op(mutates_args=["topk_values", "topk_indices"])
def _jit_grouped_topk_op(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    scaling_factor: float,
) -> None:
    module = _jit_grouped_topk_module()
    module.grouped_topk(
        scores,
        bias,
        topk_values,
        topk_indices,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        scaling_factor,
    )


def grouped_topk(
    scores: torch.Tensor,
    bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused sigmoid + bias + top-k + renormalize for MoE routing.

    Replaces the naive PyTorch path that uses 3x torch.topk + scatter + masked_fill.
    Currently supports num_expert_group=1, topk_group=1, num_experts<=512, topk<=8.
    """
    num_tokens = scores.shape[0]

    topk_values = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=scores.device
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=scores.device
    )

    if num_tokens == 0:
        return topk_values, topk_indices

    _jit_grouped_topk_op(
        scores.contiguous(),
        bias.contiguous(),
        topk_values,
        topk_indices,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        scaling_factor,
    )
    return topk_values, topk_indices
