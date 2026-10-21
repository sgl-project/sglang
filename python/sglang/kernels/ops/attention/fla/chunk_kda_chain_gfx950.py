from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.op import exp2
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported

CHUNK_SIZE = 64


@triton.jit(do_not_specialize=["T"])
def chunk_kda_chain_fwd_kernel(
    q,
    k,
    v,
    w,
    gk,
    A,
    o,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    track_state,
    track_chunk_idx,
    stride_track_state,
    scale,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    TRACK_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)

    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    stride_qk = H * 128
    stride_v = H * 128
    stride_A = H * BT
    q += (bos * H + i_h) * 128
    k += (bos * H + i_h) * 128
    v += (bos * H + i_h) * 128
    w += (bos * H + i_h) * 128
    gk += (bos * H + i_h) * 128
    A += (bos * H + i_h) * BT
    o += (bos * H + i_h) * 128

    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    valid_state = index >= 0
    state = initial_state + index * stride_init_state + i_h * 128 * 128
    if valid_state:
        p_h1 = tl.make_block_ptr(
            state, (128, 128), (128, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        p_h2 = tl.make_block_ptr(
            state, (128, 128), (128, 1), (i_v * BV, 64), (BV, 64), (1, 0)
        )
        b_h1 += tl.load(p_h1, boundary_check=(0, 1)).to(tl.float32)
        b_h2 += tl.load(p_h2, boundary_check=(0, 1)).to(tl.float32)

    if TRACK_STATE:
        i_track = tl.load(track_chunk_idx + i_n).to(tl.int32)
        track_base = track_state + (i_n * stride_track_state + i_h * 128 * 128).to(
            tl.int64
        )
    else:
        i_track = -1
        track_base = track_state

    causal_mask = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    for i_t in range(NT):
        if TRACK_STATE and i_t == i_track:
            p_t1 = tl.make_block_ptr(
                track_base,
                (128, 128),
                (128, 1),
                (i_v * BV, 0),
                (BV, 64),
                (1, 0),
            )
            p_t2 = tl.make_block_ptr(
                track_base,
                (128, 128),
                (128, 1),
                (i_v * BV, 64),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_t1, b_h1, boundary_check=(0, 1))
            tl.store(p_t2, b_h2, boundary_check=(0, 1))

        b_h1_out = b_h1.to(k.dtype.element_ty)
        b_h2_out = b_h2.to(k.dtype.element_ty)
        p_q1 = tl.make_block_ptr(
            q, (T, 128), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_q2 = tl.make_block_ptr(
            q, (T, 128), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        p_g1 = tl.make_block_ptr(
            gk, (T, 128), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_g2 = tl.make_block_ptr(
            gk, (T, 128), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))
        b_g1 = tl.load(p_g1, boundary_check=(0, 1))
        b_g2 = tl.load(p_g2, boundary_check=(0, 1))
        b_qg1 = (b_q1 * scale * exp2(b_g1)).to(b_q1.dtype)
        b_qg2 = (b_q2 * scale * exp2(b_g2)).to(b_q2.dtype)
        b_o = tl.dot(b_qg1, tl.trans(b_h1_out))
        b_o += tl.dot(b_qg2, tl.trans(b_h2_out))

        p_w1 = tl.make_block_ptr(
            w, (T, 128), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_w2 = tl.make_block_ptr(
            w, (T, 128), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_v = tl.dot(b_w1, tl.trans(b_h1_out))
        b_v += tl.dot(b_w2, tl.trans(b_h2_out))
        p_v = tl.make_block_ptr(
            v, (T, 128), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
        b_v_out = b_v.to(v.dtype.element_ty)

        p_A = tl.make_block_ptr(
            A, (T, BT), (stride_A, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_A = tl.where(causal_mask, b_A, 0.0).to(b_v_out.dtype)
        b_o += tl.dot(b_A, b_v_out, allow_tf32=False)
        p_o = tl.make_block_ptr(
            o, (T, 128), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        offsets = tl.arange(0, 64)
        g_last1 = tl.load(gk + last_idx * stride_qk + offsets)
        g_last2 = tl.load(gk + last_idx * stride_qk + 64 + offsets)
        b_h1 *= exp2(g_last1)[None, :]
        b_h2 *= exp2(g_last2)[None, :]

        p_k1 = tl.make_block_ptr(
            k, (128, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1)
        )
        p_k2 = tl.make_block_ptr(
            k, (128, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1)
        )
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_v_update = b_v.to(k.dtype.element_ty)
        b_h1 += tl.trans(tl.dot(b_k1, b_v_update))
        b_h2 += tl.trans(tl.dot(b_k2, b_v_update))

    if valid_state:
        p_h1 = tl.make_block_ptr(
            state, (128, 128), (128, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        p_h2 = tl.make_block_ptr(
            state, (128, 128), (128, 1), (i_v * BV, 64), (BV, 64), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))


def can_use_kda_chain_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    output_intermediate_states: bool,
) -> bool:
    if cu_seqlens is None:
        return False
    num_sequences = len(cu_seqlens) - 1
    measured_cell = q.shape[1] == 16384 and (
        (q.shape[-2] == 16 and num_sequences == 3)
        or (q.shape[-2] == 8 and num_sequences in (2, 3, 4))
    )
    return (
        envs.SGLANG_OPT_KDA_CHAIN_PREFILL.get()
        and is_gfx95_supported()
        and not output_intermediate_states
        and measured_cell
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and gk.dtype == torch.float32
        and A.dtype == torch.bfloat16
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and w.is_contiguous()
        and gk.is_contiguous()
        and A.is_contiguous()
        and initial_state is not None
        and q.shape[0] == 1
        and q.shape[-2] in (8, 16)
        and q.shape[-1] == 128
        and k.shape == q.shape
        and v.shape == q.shape
        and w.shape == q.shape
        and gk.shape == q.shape
        and A.shape[:-1] == q.shape[:-1]
        and A.shape[-1] == CHUNK_SIZE
    )


def chunk_kda_chain_fwd(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    track_state: Optional[torch.Tensor] = None,
    track_chunk_idx: Optional[torch.Tensor] = None,
    bv: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    assert (track_state is None) == (track_chunk_idx is None)
    if track_state is not None:
        assert track_state.dtype == torch.float32
    T, H = q.shape[1:3]
    N = len(cu_seqlens) - 1
    if bv is None:
        if H == 8 and N in (2, 3):
            bv, num_warps, num_stages = 16, 4, 3
        else:
            bv, num_warps, num_stages = 64, 4, 2
    assert num_warps is not None and num_stages is not None
    output = v
    grid = (triton.cdiv(128, bv), N * H)
    chunk_kda_chain_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        w=w,
        gk=gk,
        A=A,
        o=output,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        stride_init_state=initial_state.stride(0),
        cu_seqlens=cu_seqlens,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
        stride_track_state=(track_state.stride(0) if track_state is not None else 0),
        scale=scale,
        T=T,
        H=H,
        BT=CHUNK_SIZE,
        BV=bv,
        TRACK_STATE=track_state is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
