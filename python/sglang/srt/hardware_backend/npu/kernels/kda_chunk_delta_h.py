# Adapted from the Kimi K3 NPU implementation on the 0728_dspark branch.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.op import exp, exp2, safe_exp


CHUNK_SIZE = 64


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_npu(
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
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    """Build chunk states in KxV layout without Triton transposes.

    Ascend's Triton compiler can compile the GPU VxK implementation, but its
    transpose-heavy recurrence does not complete on A3.  This is the compact
    K<=128 form of the original 0728 NPU kernel.
    """
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

    # The NPU implementation keeps each state tile as [K, V].
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K

    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    state = initial_state + index * stride_h + i_h * K * V

    if USE_INITIAL_STATE:
        p_initial_state1 = tl.make_block_ptr(
            state,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (64, BV),
            (1, 0),
        )
        b_h1 += tl.load(p_initial_state1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_initial_state2 = tl.make_block_ptr(
                state,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            b_h2 += tl.load(p_initial_state2, boundary_check=(0, 1)).to(
                tl.float32
            )

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (64, BV),
            (1, 0),
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w,
                (T, K),
                (stride_w, 1),
                (i_t * BT, 64),
                (BT, 64),
                (1, 0),
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v,
            (T, V),
            (stride_v, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(
                v_new,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(
                p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1)
            )

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h,
                (T,),
                (H,),
                (i_t * BT,),
                (BT,),
                (0,),
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last

        if USE_GK:
            o_k = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k,
                mask=o_k < K,
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[:, None]
            else:
                b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=o_k2 < K,
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[:, None]
                else:
                    b_h2 *= exp(b_gk_last2)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)

    p_state1 = tl.make_block_ptr(
        state, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
    )
    tl.store(
        p_state1, b_h1.to(p_state1.dtype.element_ty), boundary_check=(0, 1)
    )
    if K > 64:
        p_state2 = tl.make_block_ptr(
            state, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(
            p_state2, b_h2.to(p_state2.dtype.element_ty), boundary_check=(0, 1)
        )


def chunk_gated_delta_rule_fwd_h_npu(
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
    use_exp2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert not (
        use_exp2 and g is not None
    ), "use_exp2 covers only the per-channel gk path"
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    if K > 128:
        raise ValueError("The Kimi K3 NPU chunk-state kernel supports K <= 128")

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, CHUNK_SIZE), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE),
        )

    h = k.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u) if save_new_value else None

    # The unified cache remains [..., H, V, K] for the NPU verify/decode
    # kernels. Select only this request's slots, convert them to the original
    # KxV prefill layout, then scatter the updated states back after launch.
    if initial_state is not None:
        if initial_state_indices is None:
            raise ValueError("initial_state_indices are required with initial_state")
        source_indices = initial_state_indices[:N].to(torch.long)
        kernel_state = (
            initial_state.index_select(0, source_indices)
            .transpose(-1, -2)
            .contiguous()
        )
        kernel_indices = torch.arange(
            N, dtype=torch.long, device=initial_state.device
        )
    else:
        source_indices = None
        # The kernel always materializes its final tile. Keep that write in a
        # private scratch buffer when the caller does not own a state cache.
        kernel_state = torch.empty(
            max(N, 1), H, K, V, dtype=torch.float32, device=k.device
        )
        kernel_indices = torch.arange(
            max(N, 1), dtype=torch.long, device=k.device
        )

    grid = (triton.cdiv(V, 32), N * H)
    chunk_gated_delta_rule_fwd_kernel_h_npu[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=kernel_state,
        initial_state_indices=kernel_indices,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=CHUNK_SIZE,
        BV=32,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        USE_EXP2=use_exp2,
        num_warps=4,
        num_stages=2,
    )

    if initial_state is not None:
        updated_state = kernel_state.transpose(-1, -2).contiguous()
        initial_state.index_copy_(0, source_indices, updated_state)
    return h, v_new
