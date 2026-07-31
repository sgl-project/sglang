# DVR-specific GDN kernels. The recurrent producer is adapted from FLA:
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import math
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.fla.op import exp, safe_exp
from sglang.kernels.ops.attention.fla.utils import (
    autotune_cache_kwargs,
    is_nvidia_hopper,
)

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]
CHUNK_SIZE = 64
GDN_CHUNK_H_BV = int(os.getenv("SGLANG_GDN_CHUNK_H_BV", "32"))
GDN_CHUNK_H_NUM_WARPS = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_WARPS", "4"))
GDN_CHUNK_H_NUM_STAGES = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_STAGES", "2"))


@triton.autotune(
    # Single hardcoded config. The kernel writes ht (final state) back into
    # initial_state in-place; with multiple configs, triton's autotune benchmark
    # phase invokes the kernel many times for timing and corrupts the cache pool,
    # producing silently wrong output on the first user request. Restoring via
    # `restore_value=["initial_state"]` works for unit tests but OOMs on
    # production-scale models (e.g. Kimi-Linear-48B at default mem_fraction)
    # because cloning the cache pool for each benchmark exceeds available memory.
    # NT_BUCKET is kept in the autotune key for forward-compatibility (allows
    # future per-bucket configs once the kernel is refactored to write final
    # state to a separate output buffer). The env knobs keep this single-config
    # property while allowing model/hardware-local validation of the selected
    # tile without corrupting the state pool through multi-config autotune.
    configs=[
        triton.Config(
            {"BV": GDN_CHUNK_H_BV},
            num_warps=GDN_CHUNK_H_NUM_WARPS,
            num_stages=GDN_CHUNK_H_NUM_STAGES,
        )
    ],
    key=["H", "K", "V", "BT", "USE_GK", "NT_BUCKET"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _dvr_chunk_gated_delta_rule_fwd_kernel_h(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    boundary_state,
    boundary_state_indices,
    boundary_state_steps,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    WRITE_BOUNDARY_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_BUCKET: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    h0 = initial_state + index * stride_h
    ht = initial_state + index * stride_h
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        if WRITE_BOUNDARY_STATE:
            boundary_index = tl.load(boundary_state_indices + i_n).to(tl.int64)
            boundary_step = tl.load(boundary_state_steps + i_n).to(tl.int32)
            boundary_ptr = (
                boundary_state + (boundary_index * H + i_h) * V * K + i_v * BV * K
            )
            boundary_mask = i_t == boundary_step
            tl.store(
                boundary_ptr
                + tl.arange(0, BV)[:, None] * K
                + tl.arange(0, 64)[None, :],
                b_h1,
                mask=boundary_mask
                & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                & (tl.arange(0, 64)[None, :] < K),
            )
            if K > 64:
                tl.store(
                    boundary_ptr
                    + tl.arange(0, BV)[:, None] * K
                    + 64
                    + tl.arange(0, 64)[None, :],
                    b_h2,
                    mask=boundary_mask
                    & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                    & (64 + tl.arange(0, 64)[None, :] < K),
                )
            if K > 128:
                tl.store(
                    boundary_ptr
                    + tl.arange(0, BV)[:, None] * K
                    + 128
                    + tl.arange(0, 64)[None, :],
                    b_h3,
                    mask=boundary_mask
                    & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                    & (128 + tl.arange(0, 64)[None, :] < K),
                )
            if K > 192:
                tl.store(
                    boundary_ptr
                    + tl.arange(0, BV)[:, None] * K
                    + 192
                    + tl.arange(0, 64)[None, :],
                    b_h4,
                    mask=boundary_mask
                    & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                    & (192 + tl.arange(0, 64)[None, :] < K),
                )

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    # epilogue
    if WRITE_BOUNDARY_STATE:
        boundary_index = tl.load(boundary_state_indices + i_n).to(tl.int64)
        boundary_step = tl.load(boundary_state_steps + i_n).to(tl.int32)
        boundary_ptr = (
            boundary_state + (boundary_index * H + i_h) * V * K + i_v * BV * K
        )
        boundary_mask = boundary_step == NT
        tl.store(
            boundary_ptr + tl.arange(0, BV)[:, None] * K + tl.arange(0, 64)[None, :],
            b_h1,
            mask=boundary_mask
            & (i_v * BV + tl.arange(0, BV)[:, None] < V)
            & (tl.arange(0, 64)[None, :] < K),
        )
        if K > 64:
            tl.store(
                boundary_ptr
                + tl.arange(0, BV)[:, None] * K
                + 64
                + tl.arange(0, 64)[None, :],
                b_h2,
                mask=boundary_mask
                & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                & (64 + tl.arange(0, 64)[None, :] < K),
            )
        if K > 128:
            tl.store(
                boundary_ptr
                + tl.arange(0, BV)[:, None] * K
                + 128
                + tl.arange(0, 64)[None, :],
                b_h3,
                mask=boundary_mask
                & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                & (128 + tl.arange(0, 64)[None, :] < K),
            )
        if K > 192:
            tl.store(
                boundary_ptr
                + tl.arange(0, BV)[:, None] * K
                + 192
                + tl.arange(0, 64)[None, :],
                b_h4,
                mask=boundary_mask
                & (i_v * BV + tl.arange(0, BV)[:, None] < V)
                & (192 + tl.arange(0, 64)[None, :] < K),
            )

    if INPLACE_UPDATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def _dvr_chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    inplace_update: bool = True,
    boundary_state: Optional[torch.Tensor] = None,
    boundary_state_indices: Optional[torch.Tensor] = None,
    boundary_state_steps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    write_boundary_state = boundary_state is not None
    h = k.new_empty(B, NT, H, V, K)

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    _dvr_chunk_gated_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        boundary_state=boundary_state,
        boundary_state_indices=boundary_state_indices,
        boundary_state_steps=boundary_state_steps,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=inplace_update,
        WRITE_BOUNDARY_STATE=write_boundary_state,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT_BUCKET=(0 if NT <= 32 else (1 if NT <= 128 else 2)),
    )
    return h, v_new


@torch.compiler.disable
def dvr_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    scale: Optional[float] = None,
    boundary_state: Optional[torch.Tensor] = None,
    boundary_state_indices: Optional[torch.Tensor] = None,
    boundary_state_steps: Optional[torch.Tensor] = None,
):
    """Run the FLA GDN chunk path without mutating the committed checkpoint.

    DVR verify reads an exact recurrent checkpoint, emits a possible 64-token
    boundary into request-owned storage, and leaves the source checkpoint
    untouched. The ordinary FLA path remains in-place and keeps its upstream API.
    """

    boundary_outputs = (
        boundary_state,
        boundary_state_indices,
        boundary_state_steps,
    )
    if any(value is not None for value in boundary_outputs) and not all(
        value is not None for value in boundary_outputs
    ):
        raise ValueError(
            "boundary_state, boundary_state_indices, and boundary_state_steps "
            "must be provided together."
        )
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype == torch.float32:
        raise ValueError("DVR GDN q, k, and v must share a non-fp32 dtype.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError("Variable-length DVR GDN expects a flattened batch.")
        if initial_state_indices.shape[0] != len(cu_seqlens) - 1:
            raise ValueError("initial_state_indices must contain one row per sequence.")

    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        if cu_seqlens is not None
        else None
    )
    g = chunk_local_cumsum(
        g,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new = _dvr_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        inplace_update=False,
        boundary_state=boundary_state,
        boundary_state_indices=boundary_state_indices,
        boundary_state_steps=boundary_state_steps,
    )
    output = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=k.shape[-1] ** -0.5 if scale is None else scale,
        cu_seqlens=cu_seqlens,
    )
    return output.to(q.dtype), None, h


@triton.jit
def _dvr_scatter_state_kernel(
    src,
    dst,
    source_rows,
    destination_rows,
    source_steps,
    elements_per_row: tl.constexpr,
    src_layer_stride,
    src_request_stride,
    src_step_stride,
    dst_layer_stride,
    dst_request_stride,
    src_request_count,
    src_step_count,
    dst_request_count,
    BLOCK_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    block = tl.program_id(2).to(tl.int64)
    step = tl.load(source_steps + request).to(tl.int64)
    if step < 0:
        return

    source = tl.load(source_rows + request).to(tl.int64)
    destination = tl.load(destination_rows + request).to(tl.int64)
    if not (
        (source >= 0)
        & (source < src_request_count)
        & (step < src_step_count)
        & (destination >= 0)
        & (destination < dst_request_count)
    ):
        return

    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements_per_row
    src_offset = (
        layer * src_layer_stride
        + source * src_request_stride
        + step * src_step_stride
        + offsets
    )
    dst_offset = layer * dst_layer_stride + destination * dst_request_stride + offsets
    tl.store(dst + dst_offset, tl.load(src + src_offset, mask=mask), mask=mask)


def dvr_scatter_state(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    source_rows: torch.Tensor,
    destination_rows: torch.Tensor,
    source_steps: torch.Tensor,
) -> None:
    """Publish selected request-owned recurrent states into cache slots."""

    request_count = source_steps.shape[0]
    if request_count == 0:
        return
    if (
        destination.ndim < 2
        or source.ndim < 3
        or destination.shape[0] != source.shape[0]
        or destination.shape[2:] != source.shape[3:]
    ):
        raise ValueError("DVR recurrent source and destination shapes do not match.")
    if not destination.is_contiguous() or not source.is_contiguous():
        raise ValueError("DVR recurrent-state scatter requires contiguous tensors.")

    source_rows = source_rows.to(device=source.device, dtype=torch.int32).contiguous()
    destination_rows = destination_rows.to(
        device=source.device, dtype=torch.int32
    ).contiguous()
    source_steps = source_steps.to(device=source.device, dtype=torch.int32).contiguous()
    elements_per_row = destination[0, 0].numel()
    block_size = 1024
    _dvr_scatter_state_kernel[
        (
            request_count,
            destination.shape[0],
            triton.cdiv(elements_per_row, block_size),
        )
    ](
        source,
        destination,
        source_rows,
        destination_rows,
        source_steps,
        elements_per_row,
        source.stride(0),
        source.stride(1),
        source.stride(2),
        destination.stride(0),
        destination.stride(1),
        source.shape[1],
        source.shape[2],
        destination.shape[1],
        BLOCK_SIZE=block_size,
    )


@triton.jit
def _dvr_scatter_conv_window_kernel(
    src,
    dst,
    source_rows,
    destination_rows,
    source_steps,
    elements_per_row: tl.constexpr,
    window_size: tl.constexpr,
    src_layer_stride,
    src_request_stride,
    src_step_stride,
    src_dim_stride,
    src_window_stride,
    dst_layer_stride,
    dst_request_stride,
    src_request_count,
    src_step_count,
    dst_request_count,
    BLOCK_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    block = tl.program_id(2).to(tl.int64)
    step = tl.load(source_steps + request).to(tl.int64)
    if step < 0:
        return

    source = tl.load(source_rows + request).to(tl.int64)
    destination = tl.load(destination_rows + request).to(tl.int64)
    if not (
        (source >= 0)
        & (source < src_request_count)
        & (step < src_step_count)
        & (destination >= 0)
        & (destination < dst_request_count)
    ):
        return

    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements_per_row
    dim = offsets // window_size
    window = offsets % window_size
    src_offset = (
        layer * src_layer_stride
        + source * src_request_stride
        + step * src_step_stride
        + dim * src_dim_stride
        + window * src_window_stride
    )
    dst_offset = layer * dst_layer_stride + destination * dst_request_stride + offsets
    tl.store(dst + dst_offset, tl.load(src + src_offset, mask=mask), mask=mask)


def dvr_scatter_conv_window(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    source_rows: torch.Tensor,
    destination_rows: torch.Tensor,
    source_steps: torch.Tensor,
) -> None:
    """Publish selected overlapping conv windows into contiguous cache rows."""

    request_count = source_steps.shape[0]
    if request_count == 0:
        return
    if (
        destination.ndim != 4
        or source.ndim != 5
        or destination.shape[0] != source.shape[0]
        or destination.shape[2:] != source.shape[3:]
    ):
        raise ValueError("DVR conv-window source and destination shapes do not match.")
    if not destination.is_contiguous():
        raise ValueError("DVR conv-state destination must be contiguous.")

    source_rows = source_rows.to(device=source.device, dtype=torch.int32).contiguous()
    destination_rows = destination_rows.to(
        device=source.device, dtype=torch.int32
    ).contiguous()
    source_steps = source_steps.to(device=source.device, dtype=torch.int32).contiguous()
    window_size = destination.shape[-1]
    elements_per_row = destination.shape[-2] * window_size
    block_size = 1024
    _dvr_scatter_conv_window_kernel[
        (
            request_count,
            destination.shape[0],
            triton.cdiv(elements_per_row, block_size),
        )
    ](
        source,
        destination,
        source_rows,
        destination_rows,
        source_steps,
        elements_per_row,
        window_size,
        source.stride(0),
        source.stride(1),
        source.stride(2),
        source.stride(3),
        source.stride(4),
        destination.stride(0),
        destination.stride(1),
        source.shape[1],
        source.shape[2],
        destination.shape[1],
        BLOCK_SIZE=block_size,
    )


@triton.jit
def _dvr_compact_state_window_kernel(
    cache,
    indices,
    crosses_boundary,
    accepted_tail_lens,
    layer_stride,
    slot_stride,
    token_stride,
    E: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_req = tl.program_id(0)
    i_layer = tl.program_id(1).to(tl.int64)
    i_block = tl.program_id(2).to(tl.int64)
    if not tl.load(crosses_boundary + i_req):
        return

    slot = tl.load(indices + i_req).to(tl.int64)
    count = tl.load(accepted_tail_lens + i_req).to(tl.int64) * E
    offsets = i_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < count
    token = offsets // E
    element = offsets % E
    base = i_layer * layer_stride + slot * slot_stride
    values = tl.load(
        cache + base + (CHUNK_SIZE + token) * token_stride + element,
        mask=mask,
    )
    tl.store(cache + base + token * token_stride + element, values, mask=mask)


@triton.jit
def _dvr_scatter_prefill_transitions_kernel(
    cache,
    values,
    request_rows,
    prefix_lens,
    extend_lens,
    extend_start_loc,
    cache_slot_stride,
    cache_token_stride,
    value_token_stride,
    E: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_req = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token = offsets // E
    element = offsets % E

    prefix_len = tl.load(prefix_lens + i_req).to(tl.int64)
    extend_len = tl.load(extend_lens + i_req).to(tl.int64)
    seq_len = prefix_len + extend_len
    boundary = seq_len // CHUNK_SIZE * CHUNK_SIZE
    write_start = tl.maximum(prefix_len, boundary)
    token_count = seq_len - write_start
    source_start = (
        tl.load(extend_start_loc + i_req).to(tl.int64) + write_start - prefix_len
    )
    destination_start = write_start - boundary
    mask = (token < token_count) & (element < E)

    value = tl.load(
        values + (source_start + token) * value_token_stride + element,
        mask=mask,
    )
    slot = tl.load(request_rows + i_req).to(tl.int64)
    tl.store(
        cache
        + slot * cache_slot_stride
        + (destination_start + token) * cache_token_stride
        + element,
        value,
        mask=mask,
    )


@triton.jit
def _dvr_pack_verify_window_kernel(
    cache0,
    candidate0,
    output0,
    cache1,
    candidate1,
    output1,
    request_rows,
    accepted_tail_lens,
    valid_mask,
    cache0_slot_stride,
    cache0_token_stride,
    candidate0_batch_stride,
    candidate0_token_stride,
    output0_batch_stride,
    output0_token_stride,
    cache1_slot_stride,
    cache1_token_stride,
    candidate1_batch_stride,
    candidate1_token_stride,
    output1_batch_stride,
    output1_token_stride,
    DRAFT_TOKENS: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    HAS_SECOND: tl.constexpr,
    READ_FIRST_CACHE: tl.constexpr,
    WRITE_FIRST_CACHE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_req = tl.program_id(0).to(tl.int64)
    i_token = tl.program_id(1).to(tl.int64)
    offsets = tl.program_id(2).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    slot = tl.load(request_rows + i_req).to(tl.int64)
    tail = tl.load(accepted_tail_lens + i_req).to(tl.int64)
    valid = tl.load(valid_mask + i_req)
    candidate_token = i_token - tail
    is_candidate = valid & (candidate_token >= 0) & (candidate_token < DRAFT_TOKENS)

    cache0_ptr = (
        cache0 + slot * cache0_slot_stride + i_token * cache0_token_stride + offsets
    )
    candidate0_ptr = (
        candidate0
        + i_req * candidate0_batch_stride
        + candidate_token * candidate0_token_stride
        + offsets
    )
    if READ_FIRST_CACHE:
        cache0_value = tl.load(cache0_ptr, mask=offsets < E0, other=0)
    else:
        cache0_value = 0.0
    candidate0_value = tl.load(
        candidate0_ptr, mask=(offsets < E0) & is_candidate, other=0
    )
    value0 = tl.where(is_candidate, candidate0_value, cache0_value)
    tl.store(
        output0
        + i_req * output0_batch_stride
        + i_token * output0_token_stride
        + offsets,
        value0,
        mask=offsets < E0,
    )
    if WRITE_FIRST_CACHE:
        tl.store(
            cache0_ptr,
            candidate0_value,
            mask=(offsets < E0) & is_candidate,
        )

    if HAS_SECOND:
        cache1_ptr = (
            cache1 + slot * cache1_slot_stride + i_token * cache1_token_stride + offsets
        )
        candidate1_ptr = (
            candidate1
            + i_req * candidate1_batch_stride
            + candidate_token * candidate1_token_stride
            + offsets
        )
        cache1_value = tl.load(cache1_ptr, mask=offsets < E1, other=0)
        candidate1_value = tl.load(
            candidate1_ptr, mask=(offsets < E1) & is_candidate, other=0
        )
        value1 = tl.where(is_candidate, candidate1_value, cache1_value)
        tl.store(
            output1
            + i_req * output1_batch_stride
            + i_token * output1_token_stride
            + offsets,
            value1,
            mask=offsets < E1,
        )
        tl.store(
            cache1_ptr,
            candidate1_value,
            mask=(offsets < E1) & is_candidate,
        )


@triton.jit
def _dvr_gather_verify_output_kernel(
    source,
    output,
    accepted_tail_lens,
    source_batch_stride,
    source_token_stride,
    output_batch_stride,
    output_token_stride,
    E: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_req = tl.program_id(0).to(tl.int64)
    i_token = tl.program_id(1).to(tl.int64)
    offsets = tl.program_id(2).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    source_token = tl.load(accepted_tail_lens + i_req).to(tl.int64) + i_token
    values = tl.load(
        source
        + i_req * source_batch_stride
        + source_token * source_token_stride
        + offsets,
        mask=offsets < E,
    )
    tl.store(
        output + i_req * output_batch_stride + i_token * output_token_stride + offsets,
        values,
        mask=offsets < E,
    )


@triton.jit
def _dvr_gdn_rebuild_draft_state_kernel(
    k,
    v,
    g,
    beta,
    state_src,
    state_dst,
    request_rows,
    boundary_slots,
    destination_indices,
    token_count,
    N: tl.constexpr,
    S: tl.constexpr,
    CS: tl.constexpr,
    CD: tl.constexpr,
    WINDOW: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_ln, i_hv = i_nh // HV, i_nh % HV
    i_l, i_n = i_ln // N, i_ln % N
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_input_idx = tl.load(request_rows + i_n).to(tl.int64)
    boundary_idx = tl.load(boundary_slots + i_n).to(tl.int64)
    destination_idx = tl.load(destination_indices + i_n).to(tl.int64)
    state_offset = (i_l * CS + boundary_idx) * HV * V * K + i_hv * V * K
    p_h0 = state_src + state_offset + o_v[:, None] * K + o_k[None, :]
    recurrent_state = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    steps = tl.load(token_count + i_n).to(tl.int64)
    p_k = k + (((i_l * S + state_input_idx) * WINDOW * H + i_h) * K + o_k)
    p_v = v + (((i_l * S + state_input_idx) * WINDOW * HV + i_hv) * V + o_v)
    p_g = g + ((i_l * S + state_input_idx) * WINDOW * HV + i_hv)
    p_beta = beta + ((i_l * S + state_input_idx) * WINDOW * HV + i_hv)

    for step in range(0, MAX_STEPS):
        if step < steps:
            key = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            value = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            key /= tl.sqrt(tl.sum(key * key) + 1e-6)
            recurrent_state *= exp(tl.load(p_g).to(tl.float32))
            value -= tl.sum(recurrent_state * key[None, :], 1)
            value *= tl.load(p_beta).to(tl.float32)
            recurrent_state += value[:, None] * key[None, :]

        p_k += H * K
        p_v += HV * V
        p_g += HV
        p_beta += HV

    state_offset = (i_l * CD + destination_idx) * HV * V * K + i_hv * V * K
    p_ht = state_dst + state_offset + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, recurrent_state.to(p_ht.dtype.element_ty), mask=mask_h)


def _pack_verify_window_pair(
    cache0: torch.Tensor,
    candidate0: torch.Tensor,
    *,
    request_rows: torch.Tensor,
    accepted_tail_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    cache1: Optional[torch.Tensor] = None,
    candidate1: Optional[torch.Tensor] = None,
    read_cache0: bool = True,
    persist_cache0: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Materialize one or two verify windows while persisting candidate rows."""

    has_second = cache1 is not None
    assert has_second == (candidate1 is not None)

    batch_size, draft_tokens = candidate0.shape[:2]
    output0 = cache0.new_empty((batch_size, *cache0.shape[1:]))
    output1 = cache1.new_empty((batch_size, *cache1.shape[1:])) if has_second else None
    e0 = math.prod(cache0.shape[2:])
    e1 = math.prod(cache1.shape[2:]) if has_second else 1
    if math.prod(candidate0.shape[2:]) != e0:
        raise ValueError("DVR candidate and cache element shapes do not match.")
    if has_second and math.prod(candidate1.shape[2:]) != e1:
        raise ValueError("DVR candidate and cache element shapes do not match.")

    block_size = 256
    grid = (
        batch_size,
        cache0.shape[1],
        triton.cdiv(max(e0, e1), block_size),
    )
    second_cache = cache1 if has_second else cache0
    second_candidate = candidate1 if has_second else candidate0
    second_output = output1 if has_second else output0
    _dvr_pack_verify_window_kernel[grid](
        cache0,
        candidate0,
        output0,
        second_cache,
        second_candidate,
        second_output,
        request_rows,
        accepted_tail_lens,
        valid_mask,
        cache0.stride(0),
        cache0.stride(1),
        candidate0.stride(0),
        candidate0.stride(1),
        output0.stride(0),
        output0.stride(1),
        second_cache.stride(0),
        second_cache.stride(1),
        second_candidate.stride(0),
        second_candidate.stride(1),
        second_output.stride(0),
        second_output.stride(1),
        DRAFT_TOKENS=draft_tokens,
        E0=e0,
        E1=e1,
        HAS_SECOND=has_second,
        READ_FIRST_CACHE=read_cache0,
        WRITE_FIRST_CACHE=persist_cache0,
        BLOCK_SIZE=block_size,
    )
    return output0, output1


def _gather_verify_output(
    source: torch.Tensor,
    *,
    accepted_tail_lens: torch.Tensor,
    draft_tokens: int,
) -> torch.Tensor:
    """Gather logical candidate rows directly into the target output layout."""

    batch_size = accepted_tail_lens.shape[0]
    source = source.view(batch_size, -1, *source.shape[-2:])
    output = source.new_empty((batch_size, draft_tokens, *source.shape[2:]))
    elements = math.prod(source.shape[2:])
    block_size = 256
    _dvr_gather_verify_output_kernel[
        (batch_size, draft_tokens, triton.cdiv(elements, block_size))
    ](
        source,
        output,
        accepted_tail_lens,
        source.stride(0),
        source.stride(1),
        output.stride(0),
        output.stride(1),
        E=elements,
        BLOCK_SIZE=block_size,
    )
    return output.reshape(1, batch_size * draft_tokens, *source.shape[2:])


def _scatter_prefill_transitions(
    cache: torch.Tensor,
    values: torch.Tensor,
    *,
    request_rows: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    chunk_size: int,
) -> None:
    """Cache the accepted tail after each request's latest chunk boundary."""

    elements = math.prod(cache.shape[2:])
    values = values.reshape(values.shape[0], -1)
    if values.shape[1] != elements:
        raise ValueError("DVR transition value and cache element shapes do not match.")

    block_size = 256
    _dvr_scatter_prefill_transitions_kernel[
        (
            request_rows.numel(),
            triton.cdiv(chunk_size * elements, block_size),
        )
    ](
        cache,
        values,
        request_rows,
        prefix_lens,
        extend_lens,
        extend_start_loc,
        cache.stride(0),
        cache.stride(1),
        values.stride(0),
        E=elements,
        CHUNK_SIZE=chunk_size,
        BLOCK_SIZE=block_size,
    )


def _compact_gdn_transition_windows(
    transition_windows: tuple[torch.Tensor, ...],
    *,
    indices: torch.Tensor,
    crosses_chunk_boundary: torch.Tensor,
    accepted_tail_lens: torch.Tensor,
    chunk_size: int,
) -> None:
    """Move the accepted post-boundary tail to the start of each GDN window."""

    tail_capacity = transition_windows[0].shape[2] - chunk_size
    if tail_capacity <= 0 or indices.numel() == 0:
        return

    crosses_chunk_boundary = crosses_chunk_boundary.to(torch.bool)
    accepted_tail_lens = accepted_tail_lens.to(
        device=crosses_chunk_boundary.device, dtype=torch.long
    ).contiguous()
    torch._assert_async(
        (
            ~crosses_chunk_boundary
            | ((accepted_tail_lens >= 0) & (accepted_tail_lens <= tail_capacity))
        ).all(),
        "DVR compact received an invalid linear-state tail length.",
    )
    indices = indices.to(
        device=crosses_chunk_boundary.device, dtype=torch.long
    ).contiguous()
    crosses_chunk_boundary = crosses_chunk_boundary.contiguous()

    for cache in transition_windows:
        elements_per_token = cache[0, 0, 0].numel()
        _dvr_compact_state_window_kernel[
            (
                indices.numel(),
                cache.shape[0],
                triton.cdiv(tail_capacity * elements_per_token, 256),
            )
        ](
            cache,
            indices,
            crosses_chunk_boundary,
            accepted_tail_lens,
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            E=elements_per_token,
            CHUNK_SIZE=chunk_size,
            BLOCK_SIZE=256,
        )


def _rebuild_gdn_self_draft_state(
    transition_windows: tuple[torch.Tensor, ...],
    *,
    boundary_state: torch.Tensor,
    draft_state: torch.Tensor,
    request_rows: torch.Tensor,
    boundary_slots: torch.Tensor,
    token_count: torch.Tensor,
) -> None:
    """Rebuild request-owned self-draft state from an exact boundary and tail."""

    k, v, g, beta = transition_windows
    num_layers, num_slots, num_tokens, num_key_heads, key_dim = k.shape
    _, _, _, num_value_heads, value_dim = v.shape
    num_reqs = request_rows.numel()
    if num_reqs == 0:
        return
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 8)
    _dvr_gdn_rebuild_draft_state_kernel[
        (
            triton.cdiv(value_dim, block_v),
            num_layers * num_reqs * num_value_heads,
        )
    ](
        k=k,
        v=v,
        g=g,
        beta=beta,
        state_src=boundary_state,
        state_dst=draft_state[:, :, 0],
        request_rows=request_rows,
        boundary_slots=boundary_slots,
        destination_indices=request_rows,
        token_count=token_count.contiguous(),
        N=num_reqs,
        S=num_slots,
        CS=boundary_state.shape[1],
        CD=draft_state.shape[1],
        WINDOW=num_tokens,
        MAX_STEPS=CHUNK_SIZE,
        H=num_key_heads,
        HV=num_value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        num_warps=1,
        num_stages=3,
    )
