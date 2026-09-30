# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _strided_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    row_stride,
    groups: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    cast_before_weight: tl.constexpr,
    zero_centered: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0)
    row, group = index // groups, index % groups
    columns = tl.arange(0, block)
    mask = columns < width
    x = tl.load(x_ptr + row * row_stride + group * width + columns, mask, 0).to(
        tl.float32
    )
    variance = tl.sum(x * x, 0) / width
    normalized = x * tl.rsqrt(variance + epsilon)
    weight = tl.load(weight_ptr + columns, mask, 0).to(tl.float32)
    if zero_centered:
        weight = weight + 1.0
    if cast_before_weight:
        normalized = normalized.to(output_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output_ptr + index * width + columns, normalized * weight, mask)


def _fake_strided_norm(x, weight, epsilon, cast_before_weight, zero_centered):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@register_custom_op(fake_impl=_fake_strided_norm)
def transformers_strided_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    cast_before_weight: bool,
    zero_centered: bool,
) -> torch.Tensor:
    output = _fake_strided_norm(x, weight, epsilon, cast_before_weight, zero_centered)
    rows, groups, width = x.shape
    if rows * groups:
        _strided_norm_kernel[(rows * groups,)](
            x,
            weight,
            output,
            x.stride(0),
            groups,
            width,
            epsilon,
            cast_before_weight,
            zero_centered,
            triton.next_power_of_2(width),
            num_warps=4 if width <= 2048 else 8,
            enable_fp_fusion=False,
        )
    return output


def head_sliced_rows(x):
    """A [..., groups, width] view whose groups are contiguous but whose leading
    dims are not (a head slice of a fused projection), or None."""
    if x.ndim < 3 or x.is_contiguous() or x.stride(-1) != 1:
        return None
    if x.stride(-2) != x.shape[-1]:
        return None
    try:
        return x.view(-1, x.shape[-2], x.shape[-1])
    except RuntimeError:
        return None


def strided_rms_norm(norm, x):
    rows = head_sliced_rows(x)
    if (
        rows is None
        or not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or norm.weight.dtype != x.dtype
        or x.shape[-1] > 16384
    ):
        return None
    return transformers_strided_rms_norm(
        rows,
        norm.weight.contiguous(),
        norm.variance_epsilon,
        norm.cast_x_before_out_mul,
        norm._hf_zero_centered,
    ).reshape(x.shape)
