from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_sum_module(topk: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, topk)
    return load_jit(
        "moe_sum",
        *args,
        cuda_files=["moe/moe_sum.cuh"],
        cuda_wrappers=[("moe_sum", f"moe_sum<{args}>")],
    )


@register_custom_op(
    op_name="jit_moe_sum",
    mutates_args=["output"],
)
def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
) -> None:
    topk = input.size(1)

    if topk in (2, 3, 4):
        module = _jit_moe_sum_module(topk, input.dtype)
        module.moe_sum(input, output)
    else:
        # Fallback for unsupported topk values
        torch.sum(input, dim=1, out=output)
