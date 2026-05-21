from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fused_share_gate_sigmoid_mul_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_share_gate_sigmoid_mul",
        *args,
        cuda_files=["moe/fused_share_gate_sigmoid_mul.cuh"],
        cuda_wrappers=[
            (
                "fused_share_gate_sigmoid_mul",
                f"FusedShareGateSigmoidMulKernel<{args}>::run",
            )
        ],
    )


def fused_share_gate_sigmoid_mul(
    hidden_state: torch.Tensor,
    share_gate_weight: torch.Tensor,
    share_expert_output: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert hidden_state.size(1) % 8 == 0
    module = _jit_fused_share_gate_sigmoid_mul_module(share_expert_output.dtype)
    if output is None:
        output = torch.empty_like(hidden_state)
    module.fused_share_gate_sigmoid_mul(
        output, hidden_state, share_gate_weight, share_expert_output
    )
    return output
