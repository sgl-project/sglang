from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_fused_gate_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_fused_gate",
        *args,
        cuda_files=["moe/moe_fused_gate.cuh"],
        cuda_wrappers=[("moe_fused_gate", f"moe_fused_gate<{args}>")],
    )


@register_custom_op(
    op_name="moe_fused_gate_out",
    mutates_args=["topk_weights", "topk_ids"],
)
def moe_fused_gate_out(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> None:
    """
    Fused MoE gate: sigmoid → add bias → group exclusion → topk → rescale.

    Writes results into pre-allocated output tensors (destination-passing style).

    Args:
        input:    [num_rows, num_experts], fp32/fp16/bf16
        bias:     [num_experts], same dtype as input
        topk_weights: [num_rows, topk], float32, pre-allocated output
        topk_ids:     [num_rows, topk], int32,   pre-allocated output
        num_expert_group:  number of expert groups
        topk_group:        number of groups to select
        topk:              total number of experts to select (incl. fused shared)
        num_fused_shared_experts: shared experts appended at the end of topk slots
        routed_scaling_factor:    scale factor applied to weights
        apply_routed_scaling_factor_on_output: if True, multiply final weights by scale
    """
    module = _jit_moe_fused_gate_module(input.dtype)
    module.moe_fused_gate(
        input,
        bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


def moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused MoE gate with the same call signature as ``sgl_kernel.moe_fused_gate``.

    Returns:
        topk_weights: [num_rows, topk] float32
        topk_ids:     [num_rows, topk] int32
    """
    num_rows = input.shape[0]
    topk_weights = torch.empty(
        (num_rows, topk), dtype=torch.float32, device=input.device
    )
    topk_ids = torch.empty((num_rows, topk), dtype=torch.int32, device=input.device)
    moe_fused_gate_out(
        input,
        bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    return topk_weights, topk_ids
