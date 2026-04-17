from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_sum_reduce_module(topk: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, topk)
    return load_jit(
        "moe_sum_reduce",
        *args,
        cuda_files=["moe/moe_sum_reduce.cuh"],
        cuda_wrappers=[("moe_sum_reduce", f"moe_sum_reduce<{args}>")],
    )


@register_custom_op(
    op_name="jit_moe_sum_reduce",
    mutates_args=["output"],
)
def moe_sum_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> None:
    topk = input.size(1)

    if topk in (2, 3, 4, 8, 9):
        module = _jit_moe_sum_reduce_module(topk, input.dtype)
        module.moe_sum_reduce(input, output, routed_scaling_factor)
    else:
        torch.sum(input, dim=1, out=output)
        output *= routed_scaling_factor
