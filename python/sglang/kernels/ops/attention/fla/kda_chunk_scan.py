# KDA chunked-prefill state scan that writes the output in the same pass.
#
# Replaces chunk_gated_delta_rule_fwd_h + chunk_gla_fwd_o_gk for KDA: each program
# keeps a [BV values, K keys] fp32 slice of the V-major state in registers across
# the sequence's chunks, computes v_new = u - w S^T and the chunk output
# o = (q * exp2(g)) S^T + tril(Aqk) v_new, then S = S * exp2(g_last) + v_new^T kg.
# No per-chunk state checkpoint or v_new round trip. The gated key comes
# chunk-major ([chunks, H, K, BT], tokens contiguous, tail tokens zero) so the
# update's reduction dimension loads straight into the MFMA operand.

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import prepare_chunk_offsets
from sglang.kernels.ops.attention.fla.op import exp2

CHUNK_SIZE = 64


@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_scan_o(
    q,
    kgc,
    w,
    u,
    gk,
    A,
    o,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    track_state,
    track_chunk_idx,
    stride_track_state,
    scale,
    T,
    NUM_HEADS: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRACK_STATE: tl.constexpr,
):
    NV: tl.constexpr = V // BV
    pid = tl.program_id(0)
    # Workgroups go to XCDs round-robin; give each XCD whole heads so a head's
    # V tiles share one L2 for the per-chunk q / w / kg / Aqk reads.
    i_nh = (pid // NUM_XCDS) // NV * NUM_XCDS + pid % NUM_XCDS
    i_v = (pid // NUM_XCDS) % NV
    if i_nh >= NUM_HEADS:
        return
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos = i_n * T
        boc = i_n * tl.cdiv(T, BT)
    NT = tl.cdiv(T, BT)

    q += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    gk += (bos * H + i_h).to(tl.int64) * K
    u += (bos * H + i_h).to(tl.int64) * V
    o += (bos * H + i_h).to(tl.int64) * V
    A += (bos * H + i_h).to(tl.int64) * BT
    kgc += (boc * H + i_h).to(tl.int64) * K * BT

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    p_vk = o_v[:, None] * K + o_k[None, :]

    # Padded rows carry the -1 sentinel: run the chunks, skip the state I/O.
    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    valid_state = index >= 0
    h0 = initial_state + index * stride_init_state + i_h * V * K
    b_s = tl.zeros([BV, K], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if valid_state:
            b_s += tl.load(h0 + p_vk)

    i_track = -1
    if TRACK_STATE:
        i_track = tl.load(track_chunk_idx + i_n).to(tl.int32)
        track_state += (i_n * stride_track_state + i_h * V * K).to(tl.int64)

    # [BT(s), BT(t)]: Aqk^T restricted to s <= t.
    m_st = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]

    for i_t in range(NT):
        if TRACK_STATE:
            if i_t == i_track:
                tl.store(track_state + p_vk, b_s)
        b_sb = b_s.to(q.dtype.element_ty)

        # v_new^T = u^T - S w^T: [BV, BT]
        p_wT = tl.make_block_ptr(w, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1))
        p_uT = tl.make_block_ptr(
            u, (V, T), (1, H * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1)
        )
        b_vT = tl.load(p_uT, boundary_check=(0, 1)).to(tl.float32) - tl.dot(
            b_sb, tl.load(p_wT, boundary_check=(0, 1))
        )
        b_vTb = b_vT.to(q.dtype.element_ty)

        # o^T = S (q * scale * exp2(g))^T + v_new^T tril(Aqk)^T, rounded as
        # chunk_gla_fwd_kernel_o rounds qg and the bf16 state checkpoint.
        p_qT = tl.make_block_ptr(q, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1))
        p_gT = tl.make_block_ptr(gk, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1))
        b_qT = (tl.load(p_qT, boundary_check=(0, 1)) * scale).to(q.dtype.element_ty)
        b_qgT = (b_qT * exp2(tl.load(p_gT, boundary_check=(0, 1)))).to(
            q.dtype.element_ty
        )
        p_AT = tl.make_block_ptr(
            A, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1)
        )
        b_AT = tl.where(m_st, tl.load(p_AT, boundary_check=(0, 1)), 0).to(
            q.dtype.element_ty
        )
        b_oT = tl.dot(b_sb, b_qgT) + tl.dot(b_vTb, b_AT)
        p_oT = tl.make_block_ptr(
            o, (V, T), (1, H * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1)
        )
        tl.store(p_oT, b_oT.to(o.dtype.element_ty), boundary_check=(0, 1))

        # S = S * exp2(g_last) + v_new^T kg
        last_idx = min((i_t + 1) * BT, T) - 1
        b_s *= exp2(tl.load(gk + last_idx * H * K + o_k))[None, :]
        p_kg = tl.make_block_ptr(
            kgc + i_t * H * K * BT, (BT, K), (1, BT), (0, 0), (BT, K), (0, 1)
        )
        b_s += tl.dot(b_vTb, tl.load(p_kg))

    if valid_state:
        tl.store(h0 + p_vk, b_s)


def chunk_kda_fwd_scan_o(
    q: torch.Tensor,
    kgc: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    track_state: torch.Tensor | None = None,
    track_chunk_idx: torch.Tensor | None = None,
    block_v: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    num_xcds: int = 8,
) -> torch.Tensor:
    """Writes o [B, T, H, V] and the final state into initial_state in place.

    q, w: [B, T, H, K]; u, o: [B, T, H, V]; gk: [B, T, H, K] fp32 chunk-local log2
    cumsum; A: Aqk [B, T, H, BT]; kgc: [chunks, H, K, BT] chunk-major gated key.
    """
    B, T, H, K = q.shape
    V = u.shape[-1]
    assert (K, V) == (128, 128) and V % block_v == 0
    assert initial_state is not None and initial_state.dtype == torch.float32
    assert (track_state is None) == (track_chunk_idx is None)
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)
    else:
        N = B
    num_heads = N * H
    nv = V // block_v
    grid = (num_xcds * triton.cdiv(num_heads, num_xcds) * nv,)
    chunk_kda_fwd_kernel_scan_o[grid](
        q=q,
        kgc=kgc,
        w=w,
        u=u,
        gk=gk,
        A=A,
        o=o,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        stride_init_state=initial_state.stride(0),
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
        stride_track_state=(track_state.stride(0) if track_state is not None else 0),
        scale=scale,
        T=T,
        NUM_HEADS=num_heads,
        H=H,
        K=K,
        V=V,
        BT=CHUNK_SIZE,
        BV=block_v,
        NUM_XCDS=num_xcds,
        USE_INITIAL_STATE=True,
        IS_VARLEN=cu_seqlens is not None,
        TRACK_STATE=track_state is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o
