from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
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


@register_custom_op(op_name="moe_sum_out", mutates_args=["output"])
def moe_sum_out(input: torch.Tensor, output: torch.Tensor) -> None:
    """Sum top-k expert outputs into a preallocated output tensor."""
    _jit_moe_sum_module(input.dtype).moe_sum(input, output)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    """Sum ``input`` over its top-k dimension into ``output`` in place."""
    if input.shape[1] in (2, 3, 4):
        moe_sum_out(input, output)
    else:
        torch.sum(input, dim=1, out=output)
