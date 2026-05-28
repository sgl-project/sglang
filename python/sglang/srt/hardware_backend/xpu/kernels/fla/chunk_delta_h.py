from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.fla.op import exp, make_tensor_descriptor, safe_exp
from sglang.srt.layers.attention.fla.utils import (
    autotune_cache_kwargs,
)

CHUNK_SIZE = 64


# This kernel handles K blocks in a for loop to minimize register spills.
# Time is the OUTER loop; K blocks are processed in two inner phases per step:
#   Phase 1: store h to output, accumulate v_correction = sum_k(w_k @ h_k^T)
#   Phase 2: update h = gate * h + k^T @ v_gated, save to scratch (initial_state)
@triton.autotune(
    configs=[triton.Config({"BV": 64}, num_warps=8, num_stages=2)],
    key=["H", "K", "V", "BT", "USE_GK", "USE_INITIAL_STATE", "NT_BUCKET"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_k_loop(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
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
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_BUCKET: tl.constexpr,  # this arg is kept to align with the triton kernel for CUDA
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

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

    w_desc = make_tensor_descriptor(
        base=w,
        shape=(T, K),
        strides=(stride_w, 1),
        block_shape=(BT, 64),
    )
    v_desc = make_tensor_descriptor(
        base=v,
        shape=(T, V),
        strides=(stride_v, 1),
        block_shape=(BT, BV),
    )
    k_desc = make_tensor_descriptor(
        base=k,
        shape=(T, K),
        strides=(stride_k, 1),
        block_shape=(BT, 64),
    )
    if SAVE_NEW_VALUE:
        v_new_desc = make_tensor_descriptor(
            base=v_new,
            shape=(T, V),
            strides=(stride_v, 1),
            block_shape=(BT, BV),
        )

    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    h0 = initial_state + index * stride_h
    ht = initial_state + index * stride_h
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    # main recurrence — time is the outer loop
    for i_t in range(NT):
        ########################################################################
        # Phase 1: store h to output, compute v_new = u - sum_k(w_k @ h_k^T)
        ########################################################################
        b_v_corr = tl.zeros([BT, BV], dtype=tl.float32)
        for k_blk in range(0, K, 64):
            # Load h: from initial_state (i_t==0) or scratch (i_t>0)
            if i_t == 0:
                if USE_INITIAL_STATE:
                    p_hs = tl.make_block_ptr(
                        h0, (V, K), (K, 1), (i_v * BV, k_blk), (BV, 64), (1, 0)
                    )
                    b_h = tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)
                else:
                    b_h = tl.zeros([BV, 64], dtype=tl.float32)
            else:
                p_hs = tl.make_block_ptr(
                    ht, (V, K), (K, 1), (i_v * BV, k_blk), (BV, 64), (1, 0)
                )
                b_h = tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)

            # Store pre-update h to output
            p_ho = tl.make_block_ptr(
                h + i_t * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, k_blk),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_ho, b_h.to(p_ho.dtype.element_ty), boundary_check=(0, 1))

            # Accumulate correction: w_k @ h_k^T
            b_w = w_desc.load([i_t * BT, k_blk])
            b_v_corr += tl.dot(b_w, tl.trans(b_h).to(b_w.dtype))

        # v_new = u - correction
        b_v = v_desc.load([i_t * BT, i_v * BV]) - b_v_corr

        if SAVE_NEW_VALUE:
            v_new_desc.store([i_t * BT, i_v * BV], b_v.to(v_new.dtype.element_ty))

        # Apply gate to v
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * tl.expand_dims(safe_exp(b_g_last - b_g), 1)
            b_g_last = exp(b_g_last)

        b_v = b_v.to(k.dtype.element_ty)

        ########################################################################
        # Phase 2: reload h, apply gate, update h += k^T @ v, save to scratch
        ########################################################################
        for k_blk in range(0, K, 64):
            # Reload h (same source as Phase 1)
            if i_t == 0:
                if USE_INITIAL_STATE:
                    p_hs = tl.make_block_ptr(
                        h0, (V, K), (K, 1), (i_v * BV, k_blk), (BV, 64), (1, 0)
                    )
                    b_h = tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)
                else:
                    b_h = tl.zeros([BV, 64], dtype=tl.float32)
            else:
                p_hs = tl.make_block_ptr(
                    ht, (V, K), (K, 1), (i_v * BV, k_blk), (BV, 64), (1, 0)
                )
                b_h = tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)

            # Gate decay on h
            if USE_G:
                b_h = b_h * b_g_last

            if USE_GK:
                o_k1 = tl.arange(0, 64) + k_blk
                b_gk_last1 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                    mask=(o_k1 < K),
                    other=0.0,
                )
                b_h *= tl.expand_dims(exp(b_gk_last1), 0)

            # Delta update: h += k^T @ v
            b_k = tl.trans(k_desc.load([i_t * BT, k_blk]))
            b_h += tl.trans(tl.dot(b_k, b_v))

            # Save updated h to scratch (initial_state) for next time step
            if INPLACE_UPDATE:
                p_hs = tl.make_block_ptr(
                    ht, (V, K), (K, 1), (i_v * BV, k_blk), (BV, 64), (1, 0)
                )
                tl.store(p_hs, b_h.to(p_hs.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    h = k.new_empty(B, NT, H, V, K)

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    kernel = chunk_gated_delta_rule_fwd_kernel_h_blockdim64_k_loop

    kernel[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
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
        INPLACE_UPDATE=True,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT_BUCKET=(0 if NT <= 32 else (1 if NT <= 128 else 2)),
    )
    return h, v_new
