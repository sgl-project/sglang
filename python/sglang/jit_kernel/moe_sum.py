from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_sum_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_sum",
        *args,
        cuda_files=["moe/moe_sum.cuh"],
        cuda_wrappers=[("moe_sum", f"moe_sum<{args}>")],
    )


@register_custom_op(
    op_name="moe_sum_out",
    mutates_args=["output"],
)
def moe_sum_out(
    input: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Sum over the topk expert dimension (destination-passing style).

    Args:
        input:  [num_tokens, topk, hidden_size], fp32/fp16/bf16
        output: [num_tokens, hidden_size], same dtype, pre-allocated output
    """
    module = _jit_moe_sum_module(input.dtype)
    module.moe_sum(input, output)


def moe_sum(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    """
    Sum over the topk expert dimension.

    Matches the call signature of ``sgl_kernel.moe_sum``.

    Args:
        input_tensor:  [num_tokens, topk, hidden_size], fp32/fp16/bf16
        output_tensor: [num_tokens, hidden_size], written in-place
    """
    moe_sum_out(input_tensor, output_tensor)
