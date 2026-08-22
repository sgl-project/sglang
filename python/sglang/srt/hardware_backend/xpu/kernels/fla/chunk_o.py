# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# XPU variant: uses TMA tensor descriptors (make_tensor_descriptor) instead of
# make_block_ptr for all 2-D loads/stores, and adds triton autotune.

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.op import exp, make_tensor_descriptor, safe_exp
from sglang.kernels.ops.attention.fla.utils import autotune_cache_kwargs


@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [64]
        for num_warps in [8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "USE_G", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
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
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # Adjust base pointers to the current sequence / head (same offsets as original).
    q_base = q + (bos * Hg + i_h // (H // Hg)) * K
    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    v_base = v + (bos * H + i_h) * V
    o_base = o + (bos * H + i_h) * V
    h_base = h + (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    # Tensor descriptors for 2-D loads in the K-loop.
    # q / k share the same layout (interleaved across Hg heads).
    q_desc = make_tensor_descriptor(
        base=q_base,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BT, BK),
    )
    # k is loaded transposed ([BK, BT]) so we describe the natural [T, K] layout
    # and transpose after loading to obtain [BK, BT].
    k_desc = make_tensor_descriptor(
        base=k_base,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BT, BK),
    )
    # h has shape [V, K] after the head offset (row-major).
    h_desc = make_tensor_descriptor(
        base=h_base,
        shape=(V, K),
        strides=(K, 1),
        block_shape=(BV, BK),
    )

    for i_k in range(tl.cdiv(K, BK)):
        # [BT, BK]
        b_q = q_desc.load([i_t * BT, i_k * BK])
        # [BT, BK] -> transpose to [BK, BT]
        b_k = tl.trans(k_desc.load([i_t * BT, i_k * BK]))
        # [BV, BK]
        b_h = h_desc.load([i_v * BV, i_k * BK])

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, tl.trans(b_h))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        # g is 1-D (interleaved), keep make_block_ptr.
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    # v and o share the same [T, V] interleaved layout.
    v_desc = make_tensor_descriptor(
        base=v_base,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BT, BV),
    )
    o_desc = make_tensor_descriptor(
        base=o_base,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BT, BV),
    )
    b_v = v_desc.load([i_t * BT, i_v * BV])

    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    o_desc.store([i_t * BT, i_v * BV], b_o.to(o.dtype.element_ty))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.zeros_like(v)

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
        USE_G=g is not None,
        IS_VARLEN=cu_seqlens is not None,
    )
    return o
