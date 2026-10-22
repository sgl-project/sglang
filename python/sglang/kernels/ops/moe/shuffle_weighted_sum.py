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
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    index_dtype: torch.dtype,
    topk: int,
    cast_weights_to_input_dtype: bool = False,
) -> Module | None:
    if (
        dtype not in (torch.float16, torch.bfloat16)
        or output_dtype != dtype
        or weight_dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or index_dtype not in (torch.int32, torch.int64)
        or is_hip_runtime()
        or is_musa_runtime()
        or is_sail_runtime()
    ):
        return None
    args = make_cpp_args(
        dtype, weight_dtype, index_dtype, topk, 128, cast_weights_to_input_dtype
    )
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
    CAST_WEIGHTS_TO_INPUT_DTYPE: tl.constexpr = False,
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
        )
        if CAST_WEIGHTS_TO_INPUT_DTYPE:
            factor = factor.to(inputs.dtype.element_ty)
        factor = factor.to(tl.float32)
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
    *,
    cast_weights_to_input_dtype: bool = False,
) -> None:
    """Sum mapped rows, optionally rounding scores to the input dtype first."""
    assert inputs.ndim == output.ndim == weights.ndim == 2
    assert inputs.shape[1] == output.shape[1]
    assert inputs.stride(1) == output.stride(1) == 1
    assert weights.shape[0] == output.shape[0]
    assert permutation.is_contiguous() and permutation.numel() == weights.numel()
    if output.numel() == 0:
        return
    if cast_weights_to_input_dtype and (
        inputs.dtype not in (torch.float16, torch.bfloat16)
        or weights.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not inputs.is_cuda
        or is_hip_runtime()
        or is_musa_runtime()
        or is_sail_runtime()
    ):
        # Preserve Torch conversion semantics outside the fused CUDA types.
        weights = weights.to(inputs.dtype)
        cast_weights_to_input_dtype = False
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
        CAST_WEIGHTS_TO_INPUT_DTYPE=cast_weights_to_input_dtype,
    )


@register_custom_op(op_name="shuffle_weighted_sum", mutates_args=["output"])
def shuffle_weighted_sum(
    inputs: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
    weights: torch.Tensor,
    *,
    cast_weights_to_input_dtype: bool = False,
) -> None:
    """Sum mapped rows, optionally rounding scores to the input dtype first."""
    if (
        inputs.is_cuda
        and inputs.dtype in (torch.float16, torch.bfloat16)
        and inputs.ndim == output.ndim == weights.ndim == 2
        and output.numel() != 0
    ):
        module = _jit_shuffle_weighted_sum_module(
            inputs.dtype,
            output.dtype,
            weights.dtype,
            permutation.dtype,
            weights.shape[1],
            cast_weights_to_input_dtype,
        )
        if module is not None:
            module.reduce(inputs, output, permutation, weights)
            return
    _shuffle_weighted_sum_triton(
        inputs,
        output,
        permutation,
        weights,
        cast_weights_to_input_dtype=cast_weights_to_input_dtype,
    )
