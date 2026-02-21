from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_kimi_k2_moe_fused_gate_module() -> Module:
    return load_jit(
        "kimi_k2_moe_fused_gate",
        cuda_files=["moe/kimi_k2_moe_fused_gate.cuh"],
        cuda_wrappers=[("kimi_k2_moe_fused_gate", "kimi_k2_moe_fused_gate")],
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="kimi_k2_moe_fused_gate_out",
    mutates_args=["topk_weights", "topk_ids"],
)
def kimi_k2_moe_fused_gate_out(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> None:
    """
    Kimi K2 fused MoE gate: sigmoid → add bias → top-k → renormalize.

    Writes results into pre-allocated output tensors (destination-passing style).

    Args:
        input:      [num_rows, 384], float32
        bias:       [384], float32
        topk_weights: [num_rows, topk], float32, pre-allocated output
        topk_ids:     [num_rows, topk], int32,   pre-allocated output
        topk:         number of experts to select
        renormalize:  whether to renormalize weights to sum to 1
        routed_scaling_factor: scale factor applied after renormalization
        apply_routed_scaling_factor_on_output: if True, multiply weights by scale
    """
    module = _jit_kimi_k2_moe_fused_gate_module()
    module.kimi_k2_moe_fused_gate(
        input,
        bias,
        topk_weights,
        topk_ids,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


def kimi_k2_moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Kimi K2 fused MoE gate with the same call signature as
    ``sgl_kernel.kimi_k2_moe_fused_gate``.

    Hard-coded for 384 experts (Kimi K2 architecture).

    Args:
        input:      [num_rows, 384], float32
        bias:       [384], float32
        topk:       number of experts to select (Kimi K2 uses topk=6)
        renormalize: whether to renormalize weights to sum to 1
        routed_scaling_factor: scale factor applied after renormalization
        apply_routed_scaling_factor_on_output: if True, multiply weights by scale

    Returns:
        topk_weights: [num_rows, topk] float32
        topk_ids:     [num_rows, topk] int32
    """
    num_rows = input.shape[0]
    topk_weights = torch.empty(
        (num_rows, topk), dtype=torch.float32, device=input.device
    )
    topk_ids = torch.empty((num_rows, topk), dtype=torch.int32, device=input.device)
    kimi_k2_moe_fused_gate_out(
        input,
        bias,
        topk_weights,
        topk_ids,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    return topk_weights, topk_ids
