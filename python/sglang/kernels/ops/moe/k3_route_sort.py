# SPDX-License-Identifier: Apache-2.0
"""Decode MoE sort + stage-1 mxfp8 activation quant + moe_buf zeroing in one
launch, for M <= block_size (every used expert owns exactly one block).

Output layouts match moe_sorting_small (and aiter):
  sorted_ids[i]        = (topk_slot << 24) | token   (padding: (topk << 24) | M)
  sorted_weights       = pair weight                 (padding: 0)
  sorted_expert_ids[b] = expert of block b
  num_valid_ids        = [num_blocks * block_size, M]
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _mx_e8m0(x2):
    # per-32 group e8m0 scale, rounded up as aiter does: (exponent byte, scale)
    amax = tl.maximum(tl.max(tl.abs(x2), axis=1), 1e-10)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    exp = tl.where((bits & 0x7FFFFF) != 0, exp + 1, exp)
    return exp, (exp << 23).to(tl.float32, bitcast=True)


@triton.jit
def _mx_scale_row(dest, SCALEN_PAD: tl.constexpr):
    # aiter mx_scale_shuffle_idx, row part
    return (dest // 32) * (SCALEN_PAD * 32) + (dest % 16) * 4 + (dest % 32) // 16


@triton.jit
def _mx_scale_col(y):
    # aiter mx_scale_shuffle_idx, column (group) part
    return (y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2


@triton.jit
def _sort_quant_program(
    pid,
    topk_ids_ptr,  # [M, TOPK] i32
    topk_weights_ptr,  # [M, TOPK] fp32
    sorted_ids_ptr,
    sorted_weights_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    moe_buf_ptr,  # [M, N_COLS] zeroed here
    qx_ptr,  # [M, N_COLS] routed activations, row stride stride_qx
    stride_qx,
    qout_ptr,  # [M, N_COLS] fp8 out
    qscale_ptr,  # swizzled e8m0, one per (sorted_row, group)
    M,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P_POW2: tl.constexpr,
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,
    SCALEN_PAD: tl.constexpr,
    MOE_BUF_ZERO: tl.constexpr,
):
    # A used expert's block index is the number of used experts below it.
    # CTA (t, c) quantizes token t's column chunk c and scatters its scales to
    # the token's TOPK sorted rows; the c == 0 CTA also writes the token's
    # sorted entries and the block heads/padding of experts it picks first.
    CHUNKS: tl.constexpr = N_COLS // QCHUNK
    t = pid // CHUNKS
    c0 = (pid % CHUNKS) * QCHUNK
    P = M * TOPK
    offs_p = tl.arange(0, P_POW2)
    mask_p = offs_p < P
    SENT: tl.constexpr = 0x7FFFFFFF
    e = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=SENT)
    # distinct used experts, ascending (tl.histogram miscompiles on gfx950)
    srt = tl.sort(e)
    prv = tl.gather(srt, tl.maximum(offs_p - 1, 0), axis=0)
    first = ((offs_p == 0) | (srt != prv)) & (srt != SENT)

    offs_k = tl.arange(0, TOPK)
    ek = tl.load(topk_ids_ptr + t * TOPK + offs_k)
    match = e[None, :] == ek[:, None]
    rank_k = tl.sum(tl.where(match & (offs_p[None, :] < t * TOPK), 1, 0), axis=1)
    bb_k = tl.sum(tl.where(first[None, :] & (srt[None, :] < ek[:, None]), 1, 0), axis=1)
    dest_k = bb_k * BLOCK_SIZE + rank_k

    if c0 == 0:
        wk = tl.load(topk_weights_ptr + t * TOPK + offs_k)
        tl.store(sorted_ids_ptr + dest_k, (offs_k << 24) | t)
        tl.store(sorted_weights_ptr + dest_k, wk)
        lead = rank_k == 0
        cnt_k = tl.sum(tl.where(match, 1, 0), axis=1)
        tl.store(sorted_expert_ids_ptr + bb_k, ek, mask=lead)
        offs_b = tl.arange(0, BLOCK_SIZE)
        pm = lead[:, None] & (offs_b[None, :] >= cnt_k[:, None])
        pad_at = bb_k[:, None] * BLOCK_SIZE + offs_b[None, :]
        tl.store(
            sorted_ids_ptr + pad_at,
            tl.full((TOPK, BLOCK_SIZE), (TOPK << 24), tl.int32) + M,
            mask=pm,
        )
        tl.store(
            sorted_weights_ptr + pad_at,
            tl.zeros((TOPK, BLOCK_SIZE), tl.float32),
            mask=pm,
        )
        if t == 0:
            num_valid = tl.sum(first.to(tl.int32), axis=0) * BLOCK_SIZE
            tl.store(
                num_valid_ids_ptr + tl.arange(0, 2),
                tl.where(tl.arange(0, 2) == 0, num_valid, M),
            )

    offs_q = tl.arange(0, QCHUNK)
    x = tl.load(qx_ptr + t * stride_qx + c0 + offs_q).to(tl.float32)
    x2 = tl.reshape(x, (QCHUNK // 32, 32))
    exp, scale = _mx_e8m0(x2)
    q = tl.clamp(x2 / scale[:, None], -448.0, 448.0)
    tl.store(
        qout_ptr + t * N_COLS + c0 + offs_q,
        tl.reshape(q, (QCHUNK,)).to(qout_ptr.dtype.element_ty),
    )
    if MOE_BUF_ZERO:
        tl.store(
            moe_buf_ptr + t * N_COLS + c0 + offs_q,
            tl.zeros((QCHUNK,), moe_buf_ptr.dtype.element_ty),
        )
    offs_g = tl.arange(0, QCHUNK // 32)
    y = c0 // 32 + offs_g
    base_sw = _mx_scale_row(dest_k, SCALEN_PAD)
    sw = base_sw[:, None] + _mx_scale_col(y)[None, :]
    tl.store(
        qscale_ptr + sw,
        tl.broadcast_to(exp[None, :], (TOPK, QCHUNK // 32)).to(tl.uint8),
    )


@triton.jit(do_not_specialize=["M", "stride_qx"])
def _k3_sort_quant_kernel(
    topk_ids_ptr,  # [M, TOPK] i32
    topk_weights_ptr,  # [M, TOPK] fp32
    sorted_ids_ptr,
    sorted_weights_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    moe_buf_ptr,  # [M, N_COLS] zeroed here
    qx_ptr,  # [M, N_COLS] routed activations, row stride stride_qx
    stride_qx,
    qout_ptr,  # [M, N_COLS] fp8 out
    qscale_ptr,  # swizzled e8m0, one per (sorted_row, group)
    M,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P_POW2: tl.constexpr,
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,
    SCALEN_PAD: tl.constexpr,
    MOE_BUF_ZERO: tl.constexpr,
):
    _sort_quant_program(
        tl.program_id(0),
        topk_ids_ptr,
        topk_weights_ptr,
        sorted_ids_ptr,
        sorted_weights_ptr,
        sorted_expert_ids_ptr,
        num_valid_ids_ptr,
        moe_buf_ptr,
        qx_ptr,
        stride_qx,
        qout_ptr,
        qscale_ptr,
        M,
        TOPK,
        BLOCK_SIZE,
        P_POW2,
        N_COLS,
        QCHUNK,
        SCALEN_PAD,
        MOE_BUF_ZERO,
    )


SORT_QCHUNK = 512


def sort_quant_supported(m: int, topk: int, block_size: int, n_cols: int) -> bool:
    return (
        0 < m <= block_size
        and topk & (topk - 1) == 0
        and block_size & (block_size - 1) == 0
        and n_cols % SORT_QCHUNK == 0
    )


def sort_quant(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    block_size,
    mx_quant_input,  # [M, N] bf16, row-strided ok
    qout,  # [M, N] fp8 out
    qscale,  # u8 [ceil(max_padded/32)*32, scalen_pad] out
):
    """Sort (one block per used expert, needs M <= block_size) + stage-1 mxfp8
    quant + moe_buf zero in one launch; layouts as moe_sorting_small."""
    m, topk = topk_ids.shape
    n = mx_quant_input.shape[1]
    _k3_sort_quant_kernel[(m * (n // SORT_QCHUNK),)](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        mx_quant_input,
        mx_quant_input.stride(0),
        qout,
        qscale,
        m,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        P_POW2=triton.next_power_of_2(m * topk),
        N_COLS=n,
        QCHUNK=SORT_QCHUNK,
        SCALEN_PAD=qscale.shape[1],
        MOE_BUF_ZERO=moe_buf.numel() > 0,
        num_warps=8,
    )
