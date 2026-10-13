# SPDX-License-Identifier: Apache-2.0
"""Fuse 128-wide RMSNorm and complex RoPE with native rounding boundaries."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.norm.rmsnorm_preserve_reduction import (
    can_use_rmsnorm_preserve_reduction,
)
from sglang.kernels.ops.diffusion.rope.complex_rope_triton import (
    _fuse_real_sin,
    can_use_fused_complex_rope,
)
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _qknorm_complex_rope_rows(
    x_ptr,
    weight_ptr,
    rope_ptr,
    row,
    ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    EPS: tl.constexpr,
    FUSE_REAL_SIN: tl.constexpr,
):
    # Four rows / four warps gives each lane four consecutive components.
    # Match aten's vectorized 128-wide FP32 mean: combine four components
    # left-to-right, then reduce 32 lanes with decreasing shuffle offsets.
    # Increasing rows per warp changes this order and is not bit-exact.
    column = tl.arange(0, 128)
    mask = row[:, None] < ROWS
    value = tl.load(x_ptr + row[:, None] * 128 + column[None, :], mask, 0).to(
        tl.float32
    )
    square = tl.reshape(value * value, (4, 32, 2, 2))
    even, odd = tl.split(square)
    a, c = tl.split(even)
    b, d = tl.split(odd)
    variance = tl.sum(((a + b) + c) + d, 1) * (1.0 / 128)
    inv = tl.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + column).to(tl.float32)
    value = (value * inv[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)
    value = (value * weight[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
    real, imag = tl.split(tl.reshape(value, (4, 64, 2)))
    token = row // HEADS % SEQ
    rotation = tl.load(rope_ptr + token[:, None] * 128 + column[None, :], mask, 0)
    cos, sin = tl.split(tl.reshape(rotation, (4, 64, 2)))
    out_real = tl.fma(real, cos, -imag * sin)
    if FUSE_REAL_SIN:
        out_imag = tl.fma(real, sin, imag * cos)
    else:
        out_imag = tl.fma(imag, cos, real * sin)
    return tl.reshape(tl.join(out_real, out_imag), (4, 128))


@triton.jit
def _qknorm_complex_rope_onepass_kernel(
    x_ptr,
    weight_ptr,
    rope_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    EPS: tl.constexpr,
    FUSE_REAL_SIN: tl.constexpr,
):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    out = _qknorm_complex_rope_rows(
        x_ptr, weight_ptr, rope_ptr, row, ROWS, SEQ, HEADS, EPS, FUSE_REAL_SIN
    )
    tl.store(
        out_ptr + row[:, None] * 128 + tl.arange(0, 128)[None, :],
        out,
        row[:, None] < ROWS,
    )


def can_use_qknorm_complex_rope(x, weight, rope):
    return (
        can_use_rmsnorm_preserve_reduction(x, weight)
        and can_use_fused_complex_rope(x, rope)
        and x.shape[-1] == 128
    )


def _fake_qknorm_complex_rope(x, weight, rope, eps):
    return torch.empty_like(x)


@register_custom_op(
    op_name="qknorm_complex_rope",
    mutates_args=[],
    fake_impl=_fake_qknorm_complex_rope,
)
def qknorm_complex_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    rope: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert can_use_qknorm_complex_rope(x, weight, rope)
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _qknorm_complex_rope_onepass_kernel[(triton.cdiv(x.numel() // 128, 4),)](
            x,
            weight,
            torch.view_as_real(rope),
            out,
            x.numel() // 128,
            x.shape[1],
            x.shape[2],
            eps,
            _fuse_real_sin(x.device),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
