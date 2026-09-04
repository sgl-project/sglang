# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# XPU variant: uses TMA tensor descriptors (make_tensor_descriptor) instead of
# make_block_ptr for all 2-D loads/stores, and adds triton autotune.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.op import make_tensor_descriptor
from sglang.kernels.ops.attention.fla.utils import autotune_cache_kwargs


@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64]
        for BV in [64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # 1-D accesses (beta, g) still use make_block_ptr — tensor descriptors require 2-D.
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    # 2-D accesses use tensor descriptors.  Descriptors are created once and the
    # tile offset is passed to .load() / .store() at runtime.
    a_desc = make_tensor_descriptor(
        base=A + (bos * H + i_h) * BT,
        shape=(T, BT),
        strides=(H * BT, 1),
        block_shape=(BT, BT),
    )
    b_A = a_desc.load([i_t * BT, 0])

    v_desc = make_tensor_descriptor(
        base=v + (bos * H + i_h) * V,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BT, BV),
    )
    u_desc = make_tensor_descriptor(
        base=u + (bos * H + i_h) * V,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BT, BV),
    )
    for i_v in range(tl.cdiv(V, BV)):
        b_v = v_desc.load([i_t * BT, i_v * BV])
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        u_desc.store([i_t * BT, i_v * BV], b_u.to(u.dtype.element_ty))

    k_desc = make_tensor_descriptor(
        base=k + (bos * Hg + i_h // (H // Hg)) * K,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BT, BK),
    )
    w_desc = make_tensor_descriptor(
        base=w + (bos * H + i_h) * K,
        shape=(T, K),
        strides=(H * K, 1),
        block_shape=(BT, BK),
    )
    for i_k in range(tl.cdiv(K, BK)):
        b_k = k_desc.load([i_t * BT, i_k * BK])
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        w_desc.store([i_t * BT, i_k * BK], b_w.to(w.dtype.element_ty))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_indices: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
    )
    return w, u


fwd_recompute_w_u = recompute_w_u_fwd
