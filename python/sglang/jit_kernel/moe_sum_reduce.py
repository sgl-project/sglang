from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_sum_reduce_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_sum_reduce",
        *args,
        cuda_files=["moe/moe_sum_reduce.cuh"],
        cuda_wrappers=[("moe_sum_reduce", f"moe_sum_reduce<{args}>")],
    )


@register_custom_op(
    op_name="moe_sum_reduce_out",
    mutates_args=["output"],
)
def moe_sum_reduce_out(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float,
) -> None:
    """
    Fused weighted sum-reduce over the top-k expert dimension (destination-passing style).

    Args:
        input:  [token_num, topk_num, hidden_dim], fp32/fp16/bf16, contiguous
        output: [token_num, hidden_dim], same dtype, pre-allocated output
        routed_scaling_factor: scalar multiplied into each output element
    """
    module = _jit_moe_sum_reduce_module(input.dtype)
    module.moe_sum_reduce(input, output, routed_scaling_factor)


def moe_sum_reduce(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    routed_scaling_factor: float = 0.0,
) -> None:
    """
    Fused weighted sum-reduce over the top-k expert dimension.

    Matches the call signature of ``sgl_kernel.moe_sum_reduce``.

    Args:
        input_tensor:  [token_num, topk_num, hidden_dim], fp32/fp16/bf16
        output_tensor: [token_num, hidden_dim], written in-place
        routed_scaling_factor: scalar multiplied into each output element
    """
    moe_sum_reduce_out(input_tensor, output_tensor, routed_scaling_factor)
