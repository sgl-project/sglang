# SPDX-License-Identifier: Apache-2.0
"""``x * (1 + scale[idx]) + shift[idx]`` fused with MXFP8 quantization, scales in
the cuBLASLt ``SWIZZLE_32_4_4`` layout of ``ops.quantization.mxfp8_swizzled_triton``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import round_bf16_to_fp32
from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import (
    _alloc,
    _mx_e8m0_from_amax,
    _mx_scale_offsets,
    can_use_mxfp8_swizzled,
)


@triton.jit
def _indexed_scale_shift_mxfp8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    n_groups,
    n_col_blocks,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    STORE_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)
    xrow = x_ptr + row.to(tl.int64) * stride_x_row
    x = tl.load(xrow + columns, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    # the rounding points of _indexed_scale_shift_bf16_kernel
    one_plus_scale = round_bf16_to_fp32(1.0 + scale)
    scaled = round_bf16_to_fp32(x * one_plus_scale)
    out = round_bf16_to_fp32(scaled + shift)
    if STORE_BF16:
        tl.store(xrow + columns, out, mask=mask)
    v3 = tl.reshape(out, [BLOCK_N // 32, 32])
    amax = tl.max(tl.abs(v3), axis=1)
    sbyte, inv = _mx_e8m0_from_amax(amax)
    q = tl.reshape(v3 * inv[:, None], [BLOCK_N])
    tl.store(
        q_ptr + row.to(tl.int64) * hidden_size + columns,
        q.to(tl.float8e4nv),
        mask=mask,
    )
    g = tl.arange(0, BLOCK_N // 32)
    tl.store(
        s_ptr + _mx_scale_offsets(row, g, n_col_blocks),
        sbyte.to(tl.uint8),
        mask=g < n_groups,
    )


def indexed_scale_shift_mxfp8_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    keep_bf16: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """``x * (1 + scale[idx]) + shift[idx]`` -> ``(x | None, fp8, scales)``; with
    ``keep_bf16`` the bf16 result is also written into ``x`` and returned."""
    if not can_use_mxfp8_swizzled(x):
        raise ValueError(
            "expected a row-major bf16 CUDA [rows, hidden] tensor, hidden % 32 == 0"
        )
    rows, hidden_size = x.shape
    q, s = _alloc(rows, hidden_size, x.device)
    if rows == 0:
        return (x if keep_bf16 else None), q, s
    n_groups = hidden_size // 32
    n_col_blocks = triton.cdiv(n_groups, 4)
    block_n = triton.next_power_of_2(hidden_size)
    with torch.get_device_module().device(x.device):
        _indexed_scale_shift_mxfp8_kernel[(rows,)](
            x,
            q,
            s,
            shift,
            scale,
            indices,
            hidden_size,
            n_groups,
            n_col_blocks,
            x.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            STORE_BF16=keep_bf16,
            BLOCK_N=block_n,
            num_warps=8,
        )
    return (x if keep_bf16 else None), q, s
