"""Reduce routed rows while skipping inactive permutation entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_hip_runtime,
    is_musa_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.jit.utils.common import is_sail_runtime
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_shuffle_weighted_sum_module(
    dtype: torch.dtype, weight_dtype: torch.dtype, index_dtype: torch.dtype, topk: int
) -> Module:
    args = make_cpp_args(dtype, weight_dtype, index_dtype, topk, 128)
    return load_jit(
        "shuffle_weighted_sum",
        *args,
        cuda_files=["moe/shuffle_weighted_sum.cuh"],
        cuda_wrappers=[("reduce", f"shuffle_weighted_sum_cuda<{args}>")],
    )


@triton.jit(do_not_specialize=["rows", "weight_stride0", "weight_stride1"])
def _shuffle_weighted_sum_kernel(
    inputs,
    output,
    permutation,
    weights,
    width: tl.constexpr,
    rows,
    topk: tl.constexpr,
    input_stride: tl.constexpr,
    output_stride: tl.constexpr,
    weight_stride0,
    weight_stride1,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(topk):
        row = tl.load(permutation + token * topk + slot).to(tl.int64)
        active = (row >= 0) & (row < rows)
        values = tl.load(
            inputs + row * input_stride + columns,
            mask=active & (columns < width),
            other=0,
        ).to(tl.float32)
        factor = tl.load(
            weights + token * weight_stride0 + slot * weight_stride1,
            mask=active,
            other=0,
        ).to(tl.float32)
        total += values * factor
    tl.store(
        output + token.to(tl.int64) * output_stride + columns,
        total,
        mask=columns < width,
    )


def _shuffle_weighted_sum_triton(
    inputs: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Sum mapped expert outputs; inactive maps read neither data nor weights."""
    assert inputs.ndim == output.ndim == weights.ndim == 2
    assert inputs.shape[1] == output.shape[1]
    assert inputs.stride(1) == output.stride(1) == 1
    assert weights.shape[0] == output.shape[0]
    assert permutation.is_contiguous() and permutation.numel() == weights.numel()
    if output.numel() == 0:
        return
    _shuffle_weighted_sum_kernel[(output.shape[0], triton.cdiv(output.shape[1], 256))](
        inputs,
        output,
        permutation,
        weights,
        output.shape[1],
        inputs.shape[0],
        weights.shape[1],
        inputs.stride(0),
        output.stride(0),
        weights.stride(0),
        weights.stride(1),
        BLOCK=256,
    )


@register_custom_op(op_name="shuffle_weighted_sum", mutates_args=["output"])
def shuffle_weighted_sum(
    inputs: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Sum mapped expert outputs; inactive maps read neither data nor weights."""
    if (
        inputs.is_cuda
        and inputs.dtype in (torch.float16, torch.bfloat16)
        and output.dtype == inputs.dtype
        and weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and permutation.dtype in (torch.int32, torch.int64)
        and inputs.ndim == output.ndim == weights.ndim == 2
        and output.numel() != 0
        and not is_hip_runtime()
        and not is_musa_runtime()
        and not is_sail_runtime()
    ):
        _jit_shuffle_weighted_sum_module(
            inputs.dtype, weights.dtype, permutation.dtype, weights.shape[1]
        ).reduce(inputs, output, permutation, weights)
    else:
        _shuffle_weighted_sum_triton(inputs, output, permutation, weights)
