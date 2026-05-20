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
    )


@register_custom_op(
    op_name="jit_kimi_k2_moe_fused_gate",
    mutates_args=["output", "indices"],
)
def _kimi_k2_moe_fused_gate_op(
    input: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    indices: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> None:
    module = _jit_kimi_k2_moe_fused_gate_module()
    module.kimi_k2_moe_fused_gate(
        input, bias, output, indices,
        topk, renormalize, routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


def kimi_k2_moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> list[torch.Tensor]:
    num_rows = input.size(0)
    output = torch.empty(num_rows, topk, dtype=torch.float32, device=input.device)
    indices = torch.empty(num_rows, topk, dtype=torch.int32, device=input.device)

    _kimi_k2_moe_fused_gate_op(
        input, bias, output, indices,
        topk, renormalize, routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )

    return [output, indices]
