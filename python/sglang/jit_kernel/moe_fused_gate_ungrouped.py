from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

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


def moe_fused_gate_ungrouped(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
    output: torch.Tensor = None,
    indices: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused sigmoid + bias + topk gate for ungrouped MoE (num_expert_group=1).

    If output/indices are provided, the kernel writes into them using
    output.size(1) as the row stride (supports pre-reserved shared expert slots).
    Otherwise, tensors of shape (num_rows, topk) are allocated.
    """
    num_rows = input.size(0)
    num_experts = input.size(1)

    if output is None:
        output = torch.empty((num_rows, topk), dtype=torch.float32, device=input.device)
    if indices is None:
        indices = torch.empty((num_rows, topk), dtype=torch.int32, device=input.device)

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
    return output, indices
