# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.op import exp, safe_exp
from sglang.kernels.ops.attention.fla.utils import check_shared_mem, is_nvidia_hopper
from sglang.srt.utils import use_gfx1151_kernels

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


# @triton.autotune(
#     configs=[
#         triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for BK in BKV_LIST
#         for BV in BKV_LIST
#         for num_warps in NUM_WARPS
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT"],
# )
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, tl.trans(b_h))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o_grouped_three_head(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # head groups vary before chunks across the grid, so the programs running at once read
    # the same token rows ([T, H, D] puts each head's row of a chunk 4 KB apart)
    i_v, i_hg, i_tg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_t = (
        tl.load(chunk_indices + i_tg * 2).to(tl.int32),
        tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32),
    )
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int32),
        tl.load(cu_seqlens + i_n + 1).to(tl.int32),
    )
    T = eos - bos

    # This kernel is dispatched only for Hg=16, H=48, and K=V=128.
    q += (bos * 16 + i_hg) * 128
    k += (bos * 16 + i_hg) * 128

    # Compute the raw QK product once for the three value heads that share q/k.
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(128, BK)):
        p_q = tl.make_block_ptr(
            q, (T, 128), (16 * 128, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (128, T), (1, 16 * 128), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_q, b_k)

    o_i = tl.arange(0, BT)
    b_A = tl.where(o_i[:, None] >= o_i[None, :], b_A, 0.0)

    # Keep one head-specific output accumulator live at a time.
    for j in tl.static_range(3):
        i_h = i_hg * 3 + j
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(128, BK)):
            p_q = tl.make_block_ptr(
                q,
                (T, 128),
                (16 * 128, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_h = tl.make_block_ptr(
                h + (i_tg * 48 + i_h).to(tl.int64) * 128 * 128,
                (128, 128),
                (128, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_q, tl.trans(b_h))

        p_g = tl.make_block_ptr(
            g + bos * 48 + i_h,
            (T,),
            (48,),
            (i_t * BT,),
            (BT,),
            (0,),
        )
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o *= exp(b_g)[:, None]
        b_A_gated = b_A * safe_exp(b_g[:, None] - b_g[None, :])

        p_v = tl.make_block_ptr(
            v + (bos * 48 + i_h) * 128,
            (T, 128),
            (48 * 128, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * 48 + i_h) * 128,
            (T, 128),
            (48 * 128, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A_gated.to(b_v.dtype), b_v) * scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _use_grouped_three_head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int,
    total_chunks: int,
) -> bool:
    return (
        use_gfx1151_kernels()
        and chunk_size == 64
        and cu_seqlens is not None
        and g is not None
        and q.dtype == k.dtype == v.dtype == h.dtype == torch.bfloat16
        and q.shape[0] == 1
        and q.shape[2:] == (16, 128)
        and k.shape == q.shape
        and v.shape == (1, q.shape[1], 48, 128)
        and g.shape == (1, q.shape[1], 48)
        and h.shape == (1, total_chunks, 48, 128, 128)
    )


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.zeros_like(v)

    use_grouped = _use_grouped_three_head(q, k, v, h, g, cu_seqlens, chunk_size, NT)
    if use_grouped:
        chunk_fwd_kernel_o_grouped_three_head[(2, 16, NT)](
            q,
            k,
            v,
            h,
            g,
            o,
            cu_seqlens,
            chunk_indices,
            scale,
            T=T,
            BT=BT,
            BK=64,
            BV=64,
            num_warps=4,
            num_stages=1,
        )
    else:

        def grid(meta):
            return (triton.cdiv(V, meta["BV"]), NT, B * H)

        chunk_fwd_kernel_o[grid](
            q,
            k,
            v,
            h,
            g,
            o,
            cu_seqlens,
            chunk_indices,
            scale,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=128,
            BV=64,
            USE_G=g is not None,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
            num_stages=2,
        )
    return o
