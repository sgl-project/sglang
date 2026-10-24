# SPDX-License-Identifier: Apache-2.0
"""MXFP8 attention for gfx1250 via aiter's ``fmha_fwd_mxfp8_asm``.

The ASM kernel takes e4m3 q/k/v in bhsd memory order with e8m0 block scales
(32 elements along head_dim for q/k, along seq for v) laid out in the
kernel's tiled order; the layouts mirror aiter's
``op_tests/test_fmha_fwd_mxfp8_asm.py``. The kernel also requires the kv
length to be a multiple of 128, so k/v are zero-padded and the softmax
denominator is corrected exactly from the returned LSE.
"""

import aiter
import torch
import triton
import triton.language as tl
from aiter import dtypes

_FP8 = dtypes.fp8
_FP8_MAX = float(torch.finfo(_FP8).max)
_BLOCK = 32
_SUB_Q = 256
_SUB_K = 128
_HEAD_DIM = 128  # the only head_dim shipped in hsa/gfx1250/fmha_fwd_mxfp8


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@triton.jit
def _e8m0_exponent(amax, FP8_MAX: tl.constexpr):
    e = tl.where(amax > 0, tl.ceil(tl.log2(amax / FP8_MAX)), 0.0)
    return tl.minimum(tl.maximum(e, -127.0), 127.0)


@triton.jit
def _qk_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    S,
    S_pad,
    H,
    stride_xb,
    stride_xs,
    stride_xh,
    n_tiles,
    FP8_MAX: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, D)
    x = tl.load(
        x_ptr + b * stride_xb + h * stride_xh + rs[:, None] * stride_xs + rd[None, :],
        mask=(rs < S)[:, None],
        other=0.0,
    ).to(tl.float32)
    xb = tl.reshape(x, (BLOCK_S, D // 32, 32))
    e = _e8m0_exponent(tl.max(tl.abs(xb), axis=2), FP8_MAX)
    y = xb * tl.exp2(-e)[:, :, None]
    y = tl.reshape(tl.minimum(tl.maximum(y, -FP8_MAX), FP8_MAX), (BLOCK_S, D))
    tl.store(
        y_ptr + (pid_bh * S_pad + rs[:, None]) * D + rd[None, :],
        y.to(y_ptr.dtype.element_ty),
        mask=(rs < S_pad)[:, None],
    )
    # Within each 64-row tile the kernel reads row r of the first 16 at slot 2r
    # and row 16 + r at slot 2r + 1, per 32-row half.
    r = rs % 64
    slot = (r // 32) * 32 + 2 * (r % 16) + ((r % 32) // 16)
    row = (pid_bh * n_tiles + rs // 64) * 64 + slot
    rc = tl.arange(0, D // 32)
    tl.store(
        s_ptr + row[:, None] * (D // 32) + rc[None, :],
        (e + 127.0).to(tl.uint8),
        mask=(rs < S_pad)[:, None],
    )


@triton.jit
def _v_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    S,
    S_pad,
    H,
    stride_xb,
    stride_xs,
    stride_xh,
    n_tiles,
    FP8_MAX: tl.constexpr,
    D: tl.constexpr,
    NR: tl.constexpr,
    SUBR: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_r = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rs = pid_r * NR * 32 + tl.arange(0, NR * 32)
    rd = tl.arange(0, D)
    x = tl.load(
        x_ptr + b * stride_xb + h * stride_xh + rs[:, None] * stride_xs + rd[None, :],
        mask=(rs < S)[:, None],
        other=0.0,
    ).to(tl.float32)
    xb = tl.reshape(x, (NR, 32, D))
    e = _e8m0_exponent(tl.max(tl.abs(xb), axis=1), FP8_MAX)
    y = xb * tl.exp2(-e)[:, None, :]
    y = tl.reshape(tl.minimum(tl.maximum(y, -FP8_MAX), FP8_MAX), (NR * 32, D))
    tl.store(
        y_ptr + (pid_bh * S_pad + rs[:, None]) * D + rd[None, :],
        y.to(y_ptr.dtype.element_ty),
        mask=(rs < S_pad)[:, None],
    )
    # One scale row per 32 seq rows, grouped into SUBR-row tiles that are
    # split into 64-wide head_dim units: [tile, unit, row_in_tile, 64].
    row = pid_r * NR + tl.arange(0, NR)
    off = (
        ((pid_bh * n_tiles + row[:, None] // SUBR) * (D // 64) + rd[None, :] // 64)
        * (SUBR * 64)
        + (row[:, None] % SUBR) * 64
        + rd[None, :] % 64
    )
    tl.store(s_ptr + off, (e + 127.0).to(tl.uint8), mask=(row < S_pad // 32)[:, None])


def _quant_qk(
    x: torch.Tensor, sub: int, s_pad: int, extra_tiles: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, H, D = x.shape
    s_aligned = _align(s_pad, sub)
    y = torch.empty(B, H, s_pad, D, dtype=_FP8, device=x.device)
    # Rows past s_pad and the trailing guard tiles (the K-scale over-read) keep
    # scale 1.0 (0x7f).
    s = torch.full(
        (B * H * s_aligned * (D // 32) + extra_tiles * sub * (D // 32),),
        127,
        dtype=torch.uint8,
        device=x.device,
    )
    block_s = 64
    _qk_quant_kernel[(B * H, triton.cdiv(s_pad, block_s))](
        x,
        y,
        s,
        S,
        s_pad,
        H,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        s_aligned // 64,
        FP8_MAX=_FP8_MAX,
        D=D,
        BLOCK_S=block_s,
    )
    return y, s.view(torch.float8_e8m0fnu)


def _quant_v(x: torch.Tensor, s_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, H, D = x.shape
    sub_rows = _SUB_K // _BLOCK
    n_rows = _align(s_pad, _SUB_K) // _BLOCK
    y = torch.empty(B, H, s_pad, D, dtype=_FP8, device=x.device)
    s = torch.full((B * H * n_rows * D,), 127, dtype=torch.uint8, device=x.device)
    nr = 2
    _v_quant_kernel[(B * H, triton.cdiv(s_pad, nr * _BLOCK))](
        x,
        y,
        s,
        S,
        s_pad,
        H,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        n_rows // sub_rows,
        FP8_MAX=_FP8_MAX,
        D=D,
        NR=nr,
        SUBR=sub_rows,
    )
    return y, s.view(torch.float8_e8m0fnu)


def can_use_mxfp8_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool
) -> bool:
    return (
        not causal
        and query.dtype == torch.bfloat16
        and query.shape[-1] == key.shape[-1] == value.shape[-1] == _HEAD_DIM
        and query.shape[2] == key.shape[2] == value.shape[2]
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    )


# Dynamo cannot functionalize fmha_fwd_mxfp8_asm; Inductor fails with
# "auto_functionalized_v2 was not removed".
@torch.compiler.disable
def mxfp8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Non-causal MHA on bf16 (batch, seq, heads, 128) inputs; bf16 output."""
    seqlen_k = key.shape[1]
    seqlen_k_pad = _align(seqlen_k, _SUB_K)
    q_fp8, q_scale = _quant_qk(query, _SUB_Q, query.shape[1])
    k_fp8, k_scale = _quant_qk(key, _SUB_K, seqlen_k_pad, extra_tiles=2)
    v_fp8, v_scale = _quant_v(value, seqlen_k_pad)
    out, lse = aiter.fmha_fwd_mxfp8_asm(
        q_fp8.transpose(1, 2),
        k_fp8.transpose(1, 2),
        v_fp8.transpose(1, 2),
        q_scale,
        k_scale,
        v_scale,
        softmax_scale=softmax_scale,
        return_lse=True,
    )
    num_pad = seqlen_k_pad - seqlen_k
    if num_pad:
        # Each zero key adds exp(0) = 1 to the softmax denominator and nothing
        # to the numerator, so rescale by Z_padded / (Z_padded - num_pad).
        correction = 1.0 / (1.0 - num_pad * torch.exp(-lse))
        out.mul_(correction.transpose(1, 2).unsqueeze(-1))
    return out
