# SPDX-License-Identifier: Apache-2.0
"""Fuse 128-wide RMSNorm and complex RoPE with native rounding boundaries."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.rope.complex_rope_triton import _fuse_real_sin
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _bshd_row_offsets(row, HEADS: tl.constexpr, TOKEN_STRIDE: tl.constexpr):
    # row enumerates (token, head) pairs; a packed QKV view keeps heads
    # contiguous but strides tokens by the whole projection width
    return (row // HEADS).to(tl.int64) * TOKEN_STRIDE + (row % HEADS) * 128


@triton.jit
def _qknorm_complex_rope_rows(
    x_ptr,
    weight_ptr,
    rope_ptr,
    row,
    ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    X_TOKEN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    FUSE_REAL_SIN: tl.constexpr,
):
    # Four rows / four warps gives each lane four consecutive components.
    # Match aten's vectorized 128-wide FP32 mean: combine four components
    # left-to-right, then reduce 32 lanes with decreasing shuffle offsets.
    # Increasing rows per warp changes this order and is not bit-exact.
    column = tl.arange(0, 128)
    mask = row[:, None] < ROWS
    offset = _bshd_row_offsets(row, HEADS, X_TOKEN_STRIDE)
    value = tl.load(x_ptr + offset[:, None] + column[None, :], mask, 0).to(tl.float32)
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
    X_TOKEN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    FUSE_REAL_SIN: tl.constexpr,
):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    out = _qknorm_complex_rope_rows(
        x_ptr,
        weight_ptr,
        rope_ptr,
        row,
        ROWS,
        SEQ,
        HEADS,
        X_TOKEN_STRIDE,
        EPS,
        FUSE_REAL_SIN,
    )
    tl.store(
        out_ptr + row[:, None] * 128 + tl.arange(0, 128)[None, :],
        out,
        row[:, None] < ROWS,
    )


def is_bshd_head128(x: torch.Tensor) -> bool:
    """``[B, S, H, 128]`` with contiguous heads; the token stride may exceed
    ``H * 128`` so views into a packed ``[B, S, 3 * H * 128]`` projection qualify."""
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim == 4
        and x.shape[-1] == 128
        and x.numel() > 0
        and x.stride(3) == 1
        and x.stride(2) == 128
        and x.stride(1) >= x.shape[2] * 128
        and x.stride(0) == x.shape[1] * x.stride(1)
    )


def can_use_qknorm_complex_rope(x, weight, rope):
    return (
        is_bshd_head128(x)
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (128,)
        and weight.is_contiguous()
        and rope.dtype == torch.complex64
        and rope.device == x.device
        and rope.shape == (x.shape[1], 64)
        and rope.is_contiguous()
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
            x.stride(1),
            eps,
            _fuse_real_sin(x.device),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
