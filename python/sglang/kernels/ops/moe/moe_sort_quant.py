# SPDX-License-Identifier: Apache-2.0
"""One-launch MoE sorting, plus the MXFP4 stage1 activation quant, for decode-sized
batches on the aiter MoE runner (gfx950).

Replaces aiter's two-phase moe_sorting_opus and its fused_dynamic_mxfp4_quant_moe_sort
with outputs equal bit for bit:
  sorted_ids[i]        = (topk_slot << 24) | token   (padding: (topk << 24) | M)
  sorted_weights[i]    = pair weight                 (padding: 0)
  sorted_expert_ids[b] = expert of block b
  num_valid_ids        = [num_blocks * block_size, M]
  a1                   = e2m1 codes of x / scale per token row
  a1_scale             = e8m0 ceil_pow2(amax / 6) per (sorted row, 32-column group) at
                         aiter's mx_scale_shuffle_idx address
Token CTAs place their pairs (and quantize their rows), expert CTAs write block expert ids
and block-tail padding, zero CTAs clear the accumulation buffer. Every CTA recomputes the
expert histogram it needs from topk_ids, so no CTA waits on another.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Measured on gfx950 (E=128, top-4, in a CUDA graph): one launch beats aiter's sort up to
# MAX_TOKENS and aiter's sort + quant up to MAX_QUANT_TOKENS; past those the histogram
# every CTA recomputes (it grows with M * topk) costs more than the launches it saves.
MAX_TOKENS = 1024
MAX_QUANT_TOKENS = 768
_EXPERTS_PER_CTA = 16
_BUF_BLOCK = 4096


@triton.jit
def _fp4_pair(lo, hi, scale):
    # e2m1(lo / scale) | e2m1(hi / scale) << 4 in byte 0, the conversion aiter's quant uses
    return tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        [lo, hi, scale],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _quant_rows(
    x_ptr,
    a1_ptr,
    offs_t,
    mt,
    stride_x,
    TB: tl.constexpr,
    D: tl.constexpr,
    GP: tl.constexpr,
):
    """e2m1 codes of rows offs_t into a1; returns the e8m0 exponent per [TB, GP] group."""
    offs_g = tl.arange(0, GP)
    mg = offs_g < D // 32
    mask = mt[:, None, None] & mg[None, :, None]
    x = tl.load(
        x_ptr
        + offs_t[:, None, None] * stride_x
        + (offs_g * 32)[None, :, None]
        + tl.arange(0, 32)[None, None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=2), 1e-10)
    bits = (amax * 0.16666667).to(tl.int32, bitcast=True)
    ex = (bits >> 23) & 0xFF
    ex = tl.where((ex < 0xFF) & ((bits & 0x7FFFFF) != 0), ex + 1, ex)
    scale = (ex << 23).to(tl.float32, bitcast=True)
    lo, hi = tl.split(tl.reshape(x, (TB, GP, 16, 2)))
    q = _fp4_pair(lo, hi, tl.broadcast_to(scale[:, :, None], (TB, GP, 16))) & 0xFF
    tl.store(
        a1_ptr
        + offs_t[:, None, None] * (D // 2)
        + (offs_g * 16)[None, :, None]
        + tl.arange(0, 16)[None, None, :],
        q.to(tl.uint8),
        mask=mask,
    )
    return ex


@triton.jit(do_not_specialize=["M", "num_tok_cta", "moe_buf_numel"])
def _moe_sort_quant_kernel(
    topk_ids_ptr,  # [M, TOPK] i32
    topk_w_ptr,  # [M, TOPK] f32
    sorted_ids_ptr,
    sorted_w_ptr,
    sorted_eids_ptr,
    num_valid_ptr,
    x_ptr,  # [M, D] bf16 (QUANT)
    a1_ptr,  # [M, D // 2] u8 (QUANT)
    a1s_ptr,  # e8m0 bytes, swizzled (QUANT)
    moe_buf_ptr,
    moe_buf_numel,
    M,
    num_tok_cta,
    stride_x,
    TOPK: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    TB: tl.constexpr,  # tokens per token CTA
    E_PER: tl.constexpr,  # experts per expert CTA
    HCHUNK: tl.constexpr,
    QUANT: tl.constexpr,
    D: tl.constexpr,
    GP: tl.constexpr,  # D // 32 rounded up to a power of two
    SCALEN_PAD: tl.constexpr,
    BUF_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_E_CTA: tl.constexpr = E // E_PER
    if pid >= num_tok_cta + NUM_E_CTA:
        offs = (pid - num_tok_cta - NUM_E_CTA) * BUF_BLOCK + tl.arange(0, BUF_BLOCK)
        tl.store(
            moe_buf_ptr + offs,
            tl.zeros((BUF_BLOCK,), moe_buf_ptr.dtype.element_ty),
            mask=offs < moe_buf_numel,
        )
        return

    is_tok = pid < num_tok_cta
    offs_t = pid * TB + tl.arange(0, TB)
    mt = (offs_t < M) & is_tok
    if QUANT:
        # the codes need no sort, so their loads overlap the histogram below
        ex = _quant_rows(x_ptr, a1_ptr, offs_t, mt, stride_x, TB, D, GP)

    P = M * TOPK
    p0 = tl.where(is_tok, pid * TB * TOPK, 0)
    offs_e = tl.arange(0, E)
    hist = tl.zeros((E,), tl.int32)
    before = tl.zeros((E,), tl.int32)  # pairs of each expert ahead of this CTA's
    for c in range(0, P, HCHUNK):
        offs = c + tl.arange(0, HCHUNK)
        m = offs < P
        e = tl.load(topk_ids_ptr + offs, mask=m, other=0)
        h = tl.histogram(e, E, mask=m)
        hist += h
        if c + HCHUNK <= p0:
            before += h
        elif c < p0:
            before += tl.histogram(e, E, mask=m & (offs < p0))
    blocks = (hist + BM - 1) // BM
    bb = tl.cumsum(blocks, 0) - blocks  # blocks of smaller experts

    if is_tok:
        offs_q = p0 + tl.arange(0, TB * TOPK)
        mq = offs_q < P
        eq = tl.load(topk_ids_ptr + offs_q, mask=mq, other=E)
        wq = tl.load(topk_w_ptr + offs_q, mask=mq, other=0.0)
        # aiter's order within an expert is the pair order: rank among earlier pairs
        same = (eq[:, None] == eq[None, :]) & (offs_q[None, :] < offs_q[:, None])
        start = bb * BM + before
        dest = tl.sum(
            tl.where(eq[:, None] == offs_e[None, :], start[None, :], 0), axis=1
        ) + tl.sum(same.to(tl.int32), axis=1)
        tl.store(
            sorted_ids_ptr + dest, ((offs_q % TOPK) << 24) | (offs_q // TOPK), mask=mq
        )
        tl.store(sorted_w_ptr + dest, wq, mask=mq)
        if QUANT:
            # each pair's sorted row gets its token's scales; aiter writes 0 for a zero weight
            row = tl.reshape(dest, (TB, TOPK))[:, :, None]
            y = tl.arange(0, GP)[None, None, :]
            addr = (
                (row // 32) * (SCALEN_PAD * 32)
                + (row % 16) * 4
                + (row % 32) // 16
                + (y // 8) * 256
                + (y % 4) * 64
                + ((y % 8) // 4) * 2
            )
            zero_w = tl.reshape(wq, (TB, TOPK))[:, :, None] == 0.0
            tl.store(
                a1s_ptr + addr,
                tl.where(zero_w, 0, ex[:, None, :]).to(tl.uint8),
                mask=mt[:, None, None] & (y < D // 32),
            )
    else:
        offs_x = (pid - num_tok_cta) * E_PER + tl.arange(0, E_PER)
        sel = offs_x[:, None] == offs_e[None, :]
        cnt_x = tl.sum(tl.where(sel, hist[None, :], 0), axis=1)
        blocks_x = tl.sum(tl.where(sel, blocks[None, :], 0), axis=1)
        bb_x = tl.sum(tl.where(sel, bb[None, :], 0), axis=1)
        for j in range(0, tl.max(blocks_x, axis=0)):
            tl.store(sorted_eids_ptr + bb_x + j, offs_x, mask=j < blocks_x)
        offs_b = tl.arange(0, BM)
        pos = (bb_x * BM + cnt_x)[:, None] + offs_b[None, :]
        pad = (cnt_x[:, None] + offs_b[None, :]) < (blocks_x * BM)[:, None]
        pad_id = (TOPK << 24) | M
        tl.store(
            sorted_ids_ptr + pos, tl.zeros((E_PER, BM), tl.int32) + pad_id, mask=pad
        )
        tl.store(sorted_w_ptr + pos, tl.zeros((E_PER, BM), tl.float32), mask=pad)
        if pid == num_tok_cta:
            num_valid = tl.sum(blocks, axis=0) * BM
            two = tl.arange(0, 2)
            tl.store(num_valid_ptr + two, tl.where(two == 0, num_valid, M))


def moe_sort_quant_supported(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, num_experts: int
) -> bool:
    m, topk = topk_ids.shape
    return (
        0 < m <= MAX_TOKENS
        # measured against aiter's sort on MiniMax-M3.1's routing (128 experts, top-4) only
        and num_experts == 128
        and topk == 4
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    )


def moe_sort_quant(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int,
    moe_buf: torch.Tensor,
    x: torch.Tensor | None = None,
):
    """(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids[, a1, a1_scale]) in
    aiter's layouts; a1 / a1_scale (fp4x2 / e8m0) only when ``x`` [M, D] is given. A
    non-empty ``moe_buf`` is zeroed in the same launch."""
    dev = topk_ids.device
    m, topk = topk_ids.shape
    max_padded = m * topk + num_experts * block_size - topk
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
    sorted_weights = torch.empty(max_padded, dtype=torch.float32, device=dev)
    sorted_expert_ids = torch.empty(
        triton.cdiv(max_padded, block_size), dtype=torch.int32, device=dev
    )
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=dev)
    quant = x is not None
    if quant:
        d = x.shape[1]
        scalen_pad = triton.cdiv(d // 32, 8) * 8
        a1 = torch.empty(m, d // 2, dtype=torch.uint8, device=dev)
        a1_scale = torch.empty(
            triton.cdiv(max_padded, 32) * 32, scalen_pad, dtype=torch.uint8, device=dev
        )
    else:
        d, scalen_pad = 32, 8
        x = a1 = a1_scale = sorted_ids  # unused pointers
    # Measured on gfx950: 8 tokens per CTA sort-only, 2 with the quant (row-load bound).
    tb = 2 if quant else 8
    num_tok_cta = triton.cdiv(m, tb)
    num_buf = triton.cdiv(moe_buf.numel(), _BUF_BLOCK)
    grid = (num_tok_cta + num_experts // _EXPERTS_PER_CTA + num_buf,)
    _moe_sort_quant_kernel[grid](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        x,
        a1,
        a1_scale,
        moe_buf,
        moe_buf.numel(),
        m,
        num_tok_cta,
        x.stride(0),
        TOPK=topk,
        E=num_experts,
        BM=block_size,
        TB=tb,
        E_PER=_EXPERTS_PER_CTA,
        HCHUNK=512 if m * topk <= 2048 else 1024,
        QUANT=quant,
        D=d,
        GP=triton.next_power_of_2(d // 32),
        SCALEN_PAD=scalen_pad,
        BUF_BLOCK=_BUF_BLOCK,
        num_warps=8,
    )
    sorted_ret = (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids)
    if not quant:
        return sorted_ret
    return (
        *sorted_ret,
        a1.view(torch.float4_e2m1fn_x2),
        a1_scale.view(torch.float8_e8m0fnu),
    )
