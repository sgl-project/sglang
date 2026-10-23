# SPDX-License-Identifier: Apache-2.0
"""MoE sorting plus the MXFP4 stage1 activation quant for prefill-sized batches on the
aiter MoE runner (gfx950), in four launches in place of aiter's multi-phase sort, per-token
quant and scale sort. Outputs equal aiter's bit for bit; the scale rows of padding slots
are 0, as aiter's coalesced scale sort writes them.

  count:  the expert histogram of each chunk of _CHUNK_TOKENS tokens
  prefix: per expert, its pairs in earlier chunks and in all chunks
  place:  each pair's sorted row (expert base + earlier chunks + rank in its chunk), each
          token's e2m1 codes and per-token e8m0 scales; expert CTAs write block expert
          ids, block-tail padding and num_valid_ids; zero CTAs clear moe_buf
  scales: the scales of every 32 sorted rows at aiter's mx_scale_shuffle_idx addresses
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.moe_sort_quant import _fp4_pair

# Measured on gfx950 (E=128, top-4, hidden 6144, 2k-32k tokens): 32-token chunks, one token
# per place CTA and 64-group row loads (vs one masked full-row load) were fastest.
_CHUNK_TOKENS = 32
_EXPERTS_PER_PREFIX_CTA = 4
_GROUPS_PER_LOAD = 64
_BUF_BLOCK = 4096


@triton.jit(do_not_specialize=["P"])
def _count_kernel(topk_ids_ptr, hist_ptr, P, CP: tl.constexpr, E: tl.constexpr):
    c = tl.program_id(0)
    offs = c * CP + tl.arange(0, CP)
    m = offs < P
    e = tl.load(topk_ids_ptr + offs, mask=m, other=0)
    tl.store(hist_ptr + c * E + tl.arange(0, E), tl.histogram(e, E, mask=m))


@triton.jit(do_not_specialize=["C"])
def _prefix_kernel(
    hist_ptr,  # [C, E] i32
    prefix_ptr,  # [C, E] i32: pairs of the expert in earlier chunks
    totals_ptr,  # [E] i32
    C,
    E: tl.constexpr,
    E_PER: tl.constexpr,
    C_POW2: tl.constexpr,
):
    offs_e = tl.program_id(0) * E_PER + tl.arange(0, E_PER)
    offs_c = tl.arange(0, C_POW2)
    mc = (offs_c < C)[:, None]
    idx = offs_c[:, None] * E + offs_e[None, :]
    tile = tl.load(hist_ptr + idx, mask=mc, other=0)
    tl.store(prefix_ptr + idx, tl.cumsum(tile, axis=0) - tile, mask=mc)
    tl.store(totals_ptr + offs_e, tl.sum(tile, axis=0))


@triton.jit
def _load_groups(x_ptr, offs_t, mt, stride_x, G0: tl.constexpr, GS: tl.constexpr):
    offs_g = G0 + tl.arange(0, GS)
    return tl.load(
        x_ptr
        + offs_t[:, None, None] * stride_x
        + (offs_g * 32)[None, :, None]
        + tl.arange(0, 32)[None, None, :],
        mask=mt[:, None, None],
        other=0.0,
    )


@triton.jit
def _quant_groups(
    x, a1_ptr, ts_ptr, offs_t, mt, G0: tl.constexpr, GS: tl.constexpr, D: tl.constexpr
):
    """e2m1 codes of 32-column groups [G0, G0 + GS) into a1, their e8m0 exponents into ts."""
    offs_g = G0 + tl.arange(0, GS)
    x = x.to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=2), 1e-10)
    bits = (amax * 0.16666667).to(tl.int32, bitcast=True)
    ex = (bits >> 23) & 0xFF
    ex = tl.where((ex < 0xFF) & ((bits & 0x7FFFFF) != 0), ex + 1, ex)
    scale = (ex << 23).to(tl.float32, bitcast=True)
    lo, hi = tl.split(tl.reshape(x, (x.shape[0], GS, 16, 2)))
    q = _fp4_pair(lo, hi, tl.broadcast_to(scale[:, :, None], (x.shape[0], GS, 16)))
    tl.store(
        a1_ptr
        + offs_t[:, None, None] * (D // 2)
        + (offs_g * 16)[None, :, None]
        + tl.arange(0, 16)[None, None, :],
        (q & 0xFF).to(tl.uint8),
        mask=mt[:, None, None],
    )
    tl.store(
        ts_ptr + offs_t[:, None] * (D // 32) + offs_g[None, :],
        ex.to(tl.uint8),
        mask=mt[:, None],
    )


@triton.jit
def _write_expert_blocks(
    e,
    cnt,
    blocks,
    base,
    sorted_ids_ptr,
    sorted_w_ptr,
    sorted_eids_ptr,
    num_valid_ptr,
    M,
    TOPK: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    sel = tl.arange(0, E) == e
    cnt_e = tl.sum(tl.where(sel, cnt, 0), axis=0)
    blocks_e = tl.sum(tl.where(sel, blocks, 0), axis=0)
    base_e = tl.sum(tl.where(sel, base, 0), axis=0)
    offs_j = tl.arange(0, MAX_BLOCKS)
    tl.store(
        sorted_eids_ptr + base_e // BM + offs_j,
        tl.zeros((MAX_BLOCKS,), tl.int32) + e,
        mask=offs_j < blocks_e,
    )
    offs_b = tl.arange(0, BM)
    pad = offs_b < blocks_e * BM - cnt_e
    pos = base_e + cnt_e + offs_b
    pad_id = (TOPK << 24) | M
    tl.store(sorted_ids_ptr + pos, tl.zeros((BM,), tl.int32) + pad_id, mask=pad)
    tl.store(sorted_w_ptr + pos, tl.zeros((BM,), tl.float32), mask=pad)
    if e == 0:
        two = tl.arange(0, 2)
        valid = tl.sum(blocks, axis=0) * BM
        tl.store(num_valid_ptr + two, tl.where(two == 0, valid, M))


@triton.jit(do_not_specialize=["M", "moe_buf_numel"])
def _place_quant_kernel(
    topk_ids_ptr,  # [M, TOPK] i32
    topk_w_ptr,  # [M, TOPK] f32
    prefix_ptr,
    totals_ptr,
    sorted_ids_ptr,
    sorted_w_ptr,
    sorted_eids_ptr,
    num_valid_ptr,
    x_ptr,  # [M, D] bf16
    a1_ptr,  # [M, D // 2] u8
    ts_ptr,  # [M, D // 32] e8m0 bytes per token
    moe_buf_ptr,
    moe_buf_numel,
    M,
    stride_x,
    TOPK: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    CHUNK: tl.constexpr,
    D: tl.constexpr,
    GS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BUF_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M + E:
        offs = (pid - M - E) * BUF_BLOCK + tl.arange(0, BUF_BLOCK)
        tl.store(
            moe_buf_ptr + offs,
            tl.zeros((BUF_BLOCK,), moe_buf_ptr.dtype.element_ty),
            mask=offs < moe_buf_numel,
        )
        return
    cnt = tl.load(totals_ptr + tl.arange(0, E))
    blocks = (cnt + BM - 1) // BM
    base = (tl.cumsum(blocks, 0) - blocks) * BM
    if pid >= M:
        _write_expert_blocks(
            pid - M,
            cnt,
            blocks,
            base,
            sorted_ids_ptr,
            sorted_w_ptr,
            sorted_eids_ptr,
            num_valid_ptr,
            M,
            TOPK,
            E,
            BM,
            MAX_BLOCKS,
        )
        return

    # one token per CTA; the row loads go first since the stores below may alias x
    offs_t = pid + tl.arange(0, 1)
    mt = offs_t < M
    # hidden 6144 is three loads of GS groups (moe_sort_quant_large_supported)
    tl.static_assert(D == 3 * 32 * GS)
    x0 = _load_groups(x_ptr, offs_t, mt, stride_x, 0, GS)
    x1 = _load_groups(x_ptr, offs_t, mt, stride_x, GS, GS)
    x2 = _load_groups(x_ptr, offs_t, mt, stride_x, 2 * GS, GS)

    c = pid // CHUNK
    P = M * TOPK
    offs_q = c * (CHUNK * TOPK) + tl.arange(0, CHUNK * TOPK)
    eq = tl.load(topk_ids_ptr + offs_q, mask=offs_q < P, other=E)
    offs_p = pid * TOPK + tl.arange(0, TOPK)
    ep = tl.load(topk_ids_ptr + offs_p)
    wp = tl.load(topk_w_ptr + offs_p)
    # aiter orders the rows of an expert by pair index
    rank = tl.sum(
        ((eq[None, :] == ep[:, None]) & (offs_q[None, :] < offs_p[:, None])).to(
            tl.int32
        ),
        axis=1,
    )
    base_p = tl.sum(
        tl.where(ep[:, None] == tl.arange(0, E)[None, :], base[None, :], 0), axis=1
    )
    dest = base_p + tl.load(prefix_ptr + c * E + ep) + rank
    tl.store(sorted_ids_ptr + dest, (tl.arange(0, TOPK) << 24) | pid)
    tl.store(sorted_w_ptr + dest, wp)

    _quant_groups(x0, a1_ptr, ts_ptr, offs_t, mt, 0, GS, D)
    _quant_groups(x1, a1_ptr, ts_ptr, offs_t, mt, GS, GS, D)
    _quant_groups(x2, a1_ptr, ts_ptr, offs_t, mt, 2 * GS, GS, D)


@triton.jit(do_not_specialize=["M"])
def _scale_tile_kernel(
    sorted_ids_ptr,
    num_valid_ptr,
    ts_ptr,
    a1s_ptr,
    M,
    SG: tl.constexpr,  # scale groups per row
    SG_POW2: tl.constexpr,
):
    t = tl.program_id(0)
    if t * 32 >= tl.load(num_valid_ptr):
        return
    tok = tl.load(sorted_ids_ptr + t * 32 + tl.arange(0, 32)) & 0xFFFFFF
    y = tl.arange(0, SG_POW2)
    s = tl.load(
        ts_ptr + tok[:, None] * SG + y[None, :],
        mask=(tok < M)[:, None] & (y < SG)[None, :],
        other=0,
    )
    # [row_hi, row_lo, y_blk, y_hi, y_lo] -> [y_blk, y_lo, row_lo, y_hi, row_hi] is the
    # mx_scale_shuffle_idx order of one 32-row tile
    s = tl.permute(tl.reshape(s, (2, 16, SG_POW2 // 8, 2, 4)), (2, 4, 1, 3, 0))
    o = tl.arange(0, SG_POW2 * 32)
    tl.store(
        a1s_ptr + t * (SG * 32) + o, tl.reshape(s, (SG_POW2 * 32,)), mask=o < SG * 32
    )


def moe_sort_quant_large_supported(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    x: torch.Tensor,
) -> bool:
    m, topk = topk_ids.shape
    return (
        0 < m < (1 << 24)
        # measured against aiter on MiniMax-M3.1's routing (128 experts, top-4,
        # hidden 6144) only
        and num_experts == 128
        and topk == 4
        and x.shape[1] == 6144
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_contiguous()
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    )


def moe_sort_quant_large(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int,
    moe_buf: torch.Tensor,
    x: torch.Tensor,
):
    """(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, a1, a1_scale) in
    aiter's layouts, a1 / a1_scale being the fp4x2 / e8m0 stage1 quant of x [M, D]. A
    non-empty ``moe_buf`` is zeroed."""
    dev = topk_ids.device
    m, topk = topk_ids.shape
    d = x.shape[1]
    groups = d // 32
    max_padded = m * topk + num_experts * block_size - topk
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
    sorted_weights = torch.empty(max_padded, dtype=torch.float32, device=dev)
    sorted_expert_ids = torch.empty(
        triton.cdiv(max_padded, block_size), dtype=torch.int32, device=dev
    )
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=dev)
    a1 = torch.empty(m, d // 2, dtype=torch.uint8, device=dev)
    a1_scale = torch.empty(
        triton.cdiv(max_padded, 32) * 32, groups, dtype=torch.uint8, device=dev
    )
    num_chunks = triton.cdiv(m, _CHUNK_TOKENS)
    hist = torch.empty(num_chunks, num_experts, dtype=torch.int32, device=dev)
    prefix = torch.empty_like(hist)
    totals = torch.empty(num_experts, dtype=torch.int32, device=dev)
    token_scales = torch.empty(m, groups, dtype=torch.uint8, device=dev)

    _count_kernel[(num_chunks,)](
        topk_ids, hist, m * topk, CP=_CHUNK_TOKENS * topk, E=num_experts, num_warps=4
    )
    _prefix_kernel[(num_experts // _EXPERTS_PER_PREFIX_CTA,)](
        hist,
        prefix,
        totals,
        num_chunks,
        E=num_experts,
        E_PER=_EXPERTS_PER_PREFIX_CTA,
        C_POW2=triton.next_power_of_2(num_chunks),
        num_warps=8,
    )
    num_buf = triton.cdiv(moe_buf.numel(), _BUF_BLOCK)
    _place_quant_kernel[(m + num_experts + num_buf,)](
        topk_ids,
        topk_weights,
        prefix,
        totals,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        x,
        a1,
        token_scales,
        moe_buf,
        moe_buf.numel(),
        m,
        x.stride(0),
        TOPK=topk,
        E=num_experts,
        BM=block_size,
        CHUNK=_CHUNK_TOKENS,
        D=d,
        GS=_GROUPS_PER_LOAD,
        MAX_BLOCKS=triton.next_power_of_2(triton.cdiv(m * topk, block_size)),
        BUF_BLOCK=_BUF_BLOCK,
        num_warps=4,
    )
    _scale_tile_kernel[(triton.cdiv(max_padded, 32),)](
        sorted_ids,
        num_valid_ids,
        token_scales,
        a1_scale,
        m,
        SG=groups,
        SG_POW2=triton.next_power_of_2(groups),
        num_warps=4,
    )
    return (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        a1.view(torch.float4_e2m1fn_x2),
        a1_scale.view(torch.float8_e8m0fnu),
    )
