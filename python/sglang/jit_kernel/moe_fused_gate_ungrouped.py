from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_fused_gate_ungrouped_module(num_experts: int) -> Module:
    args = make_cpp_args(num_experts)
    return load_jit(
        "moe_fused_gate_ungrouped",
        *args,
        cuda_files=["moe/moe_fused_gate_ungrouped.cu"],
        cuda_wrappers=[
            (
                "moe_fused_gate_ungrouped",
                f"MoeFusedGateUngroupedKernel<{args}>::run",
            ),
        ],
    )


def _moe_fused_gate_ungrouped_fake(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
    output: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    pass


@register_custom_op(
    op_name="moe_fused_gate_ungrouped",
    mutates_args=["output", "indices"],
    fake_impl=_moe_fused_gate_ungrouped_fake,
)
def moe_fused_gate_ungrouped(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
    output: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Fused sigmoid + bias + topk gate for ungrouped MoE (num_expert_group=1).

    The kernel writes into output/indices using output.size(1) as the row stride
    (supports pre-reserved shared expert slots).
    """
    assert topk <= 8, f"topk must be <= 8 (kernel shared memory constraint), got {topk}"

    num_experts = input.size(1)
    assert (
        num_experts % 128 == 0
    ), f"num_experts must be divisible by 128 (WARP_SIZE * VEC_SIZE), got {num_experts}"

    module = _jit_moe_fused_gate_ungrouped_module(num_experts)
    module.moe_fused_gate_ungrouped(
        input,
        bias,
        output,
        indices,
        topk,
        renormalize,
        routed_scaling_factor if routed_scaling_factor is not None else 1.0,
        apply_routed_scaling_factor_on_output,
    )
