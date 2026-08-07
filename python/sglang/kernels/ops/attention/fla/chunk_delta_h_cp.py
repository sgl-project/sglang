# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/chunk_delta_h.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Context-parallel (CP) state pre-process for the chunked delta-rule recurrence
# (KDA / GDN family), extended from FLA's training-oriented KCP design with the
# serving semantics sglang needs:
#
#   * the merge chain is seeded from the mamba state-pool slot (chunked prefill
#     carries state across scheduler steps), instead of asserting
#     `initial_state is None` like FLA;
#   * the merge chain is replayed over ALL ranks (not just the prefix), so every
#     CP rank obtains the identical global final state and writes it back to its
#     pool-slot replica with zero extra communication;
#   * per-rank entry states are written to a fresh scratch tensor rather than
#     the pool, so the main kernel's INPLACE_UPDATE epilogue cannot clobber the
#     global final state the merge already wrote.
#
# Algorithm (see kda_cp_analysis.md at the repo root for the full writeup):
# the inter-chunk recurrence S' = (Diag(gamma) - kg^T w) S + kg^T u is affine in
# S, so each rank's whole shard collapses to one affine map (M, S_ext).
#   Phase 1  local pre-scan with S_0 = 0        -> hm = [S_ext | M]  fp32
#   Phase 2  one all-gather of hm               -> [W, N, HV, K, K+V]
#   Phase 3  local merge: replay the tiny fp32 affine chain
#            prefix over ranks j < r  -> this rank's entry state h0
#            full chain over W ranks  -> global final state, back to the pool
#
# Sharding contract: every sequence in the batch is split contiguously across
# all CP ranks (rank r holds the r-th slice of each sequence). Empty shards are
# naturally safe: they produce (S_ext, M) = (0, I), the identity affine map.
#
# Correctness invariant inherited from FLA (fla/ops/cp/README.md): the pre-scan
# MUST receive exactly the tensors the main kernel receives — for KDA that is
# the pre-gated kg, the WY tensors w/u, and the log2-domain cumulative gate g
# (USE_GK + use_exp2); for GDN the original k and the natural-log scalar gate.

from typing import List, Optional, Tuple

import msgspec
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.op import exp, exp2, safe_exp
from sglang.kernels.ops.attention.fla.utils import autotune_cache_kwargs

CHUNK_SIZE = 64


class LinearAttnCPContext(msgspec.Struct, frozen=True):
    """Context-parallel topology for linear-attention state pre-processing.

    ``group`` is the torch.distributed ProcessGroup used for the hm all-gather.
    It may be None in single-process tests that drive the pure pieces
    (``cp_local_transition`` / ``cp_merge_states``) directly.

    A rank's local batch MUST NOT contain zero-length sequences: the base
    chunk pipeline (prepare_chunk_indices attributes chunks to sequences by
    counting chunk-index resets) silently mis-drives the intra kernels on
    empty sequences, leaving kg/w/u uninitialized. When a sequence has no
    tokens on this rank, the caller drops it from the local batch and records
    the surviving sequences' global positions in ``local_seq_ids``
    (``build_cp_shard_layout`` produces exactly this). Dropped sequences
    contribute the identity affine map to the all-gather and still receive
    their final-state writeback in the merge.

    ``num_global_seqs`` / ``local_seq_ids`` may be None when every global
    sequence has tokens on this rank (no compaction).
    """

    world_size: int
    rank: int
    group: Optional[object] = None
    num_global_seqs: Optional[int] = None
    local_seq_ids: Optional[object] = None  # int32 [N_local] device tensor

    @property
    def is_active(self) -> bool:
        return self.world_size > 1


def build_cp_shard_layout(
    cu_seqlens: List[int],
    world_size: int,
    rank: int,
) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    """Per-sequence contiguous split of a packed varlen batch.

    Returns (local cu_seqlens values, global [start, end) token range per
    kept sequence, global ids of the kept sequences). Sequences with no
    tokens on this rank are dropped (see LinearAttnCPContext): callers use
    the ranges to relayout tokens, the local cu_seqlens for every
    shard-local kernel, and the ids as ``local_seq_ids``.
    """
    local_cu = [0]
    shard_ranges = []
    local_seq_ids = []
    for n in range(len(cu_seqlens) - 1):
        seq_start, seq_end = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        seq_len = seq_end - seq_start
        lo = seq_start + (seq_len * rank) // world_size
        hi = seq_start + (seq_len * (rank + 1)) // world_size
        if hi > lo:
            shard_ranges.append((lo, hi))
            local_seq_ids.append(n)
            local_cu.append(local_cu[-1] + (hi - lo))
    return local_cu, shard_ranges, local_seq_ids


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
    }
)
@triton.autotune(
    configs=[
        # >= 4 warps: the stage-2 column block keeps [BK1, BLOCK_SIZE] fp32
        # live; fewer warps concentrate it into spilling register pressure.
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=["H", "HV", "K", "V", "BT", "USE_G", "USE_GK", "USE_EXP2"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def cp_pre_scan_fwd_kernel(
    k,
    w,
    u,
    g,
    gk,
    hm,
    seq_map,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    """Phase 1: per (sequence, head), scan the local shard with S_0 = 0 and emit
    the shard's affine map hm[n, h] = [S_ext (K x V) | M (K x K)] in fp32.

    Grid: (cdiv(V, BLOCK_SIZE) + cdiv(K, BLOCK_SIZE), N, HV). Column blocks
    below V compute S_ext (same recurrence as the main kernel); the rest build
    M by left-multiplying per-chunk transition matrices from the identity.
    """
    i_col, i_n, i_h = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2),
    )
    # Data offsets use the local sequence index; the hm row uses the global
    # one (locally-dropped sequences keep their pre-initialized identity row).
    i_n_global = tl.load(seq_map + i_n).to(tl.int64)
    hm += (i_n_global * HV + i_h) * K * (K + V)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = (eos - bos).to(tl.int32)
    NT = tl.cdiv(T, BT)

    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    stride_k = H * K
    stride_w = HV * K

    is_h_part = i_col * BLOCK_SIZE < V
    if is_h_part:
        # ====== Stage 1: S_ext (K x V), same recurrence as the main kernel ======
        u += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_vb = i_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m_vb = o_vb < V
        o_k1 = tl.arange(0, 64)
        m_k1 = o_k1 < K
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        o_k3 = 128 + o_k1
        m_k3 = o_k3 < K
        o_k4 = 192 + o_k1
        m_k4 = o_k4 < K

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
                b_v_decay += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                b_v_decay += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                b_v_decay += tl.dot(b_w, b_h4.to(b_w.dtype))

            p_u = u + o_t[:, None] * stride_v + o_vb[None, :]
            b_v = tl.load(p_u, mask=m_t[:, None] & m_vb[None, :], other=0.0) - b_v_decay

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                b_g = tl.load(g + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
                b_v = b_v * tl.where(m_t, safe_exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            if USE_GK:
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=m_k1, other=0.0).to(
                    tl.float32
                )
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
                if K > 64:
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=m_k2, other=0.0).to(
                        tl.float32
                    )
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
                if K > 128:
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=m_k3, other=0.0).to(
                        tl.float32
                    )
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
                if K > 192:
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=m_k4, other=0.0).to(
                        tl.float32
                    )
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]
            b_v = b_v.to(k.dtype.element_ty)

            p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            b_h1 += tl.dot(b_k, b_v)
            if K > 64:
                p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                b_h2 += tl.dot(b_k, b_v)
            if K > 128:
                p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                b_h3 += tl.dot(b_k, b_v)
            if K > 192:
                p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                b_h4 += tl.dot(b_k, b_v)

        stride_hm_kv = K + V
        p_h1 = hm + o_k1[:, None] * stride_hm_kv + o_vb[None, :]
        tl.store(
            p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :]
        )
        if K > 64:
            p_h2 = hm + o_k2[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(
                p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :]
            )
        if K > 128:
            p_h3 = hm + o_k3[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(
                p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :]
            )
        if K > 192:
            p_h4 = hm + o_k4[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(
                p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :]
            )
    else:
        # ====== Stage 2: M (K x K), product of per-chunk transitions ======
        # M_c = Diag(gamma_c) - kg_c^T w_c applied by left-multiplication. The
        # product is regrouped as Diag(gamma) M - kg^T (w M): materializing the
        # [K, K] kg^T w intermediate (as FLA does) spills to gigabytes of
        # CUDA local memory at launch, which OOMs on busy GPUs.
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # M_0 = I: the empty shard is the identity affine map.
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            p_k = k + o_t[:, None] * stride_k + row[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
            p_w = w + o_t[:, None] * stride_w + row[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                b_g = tl.load(g + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
                b_k = b_k * tl.where(m_t, safe_exp(b_g_last - b_g), 0)[:, None]
                b_gamma = exp(b_g_last) + tl.zeros([BK1], dtype=tl.float32)
            elif USE_GK:
                b_gk_last = tl.load(
                    gk + last_idx * HV * K + row, mask=(row < K), other=0.0
                ).to(tl.float32)
                if USE_EXP2:
                    b_gamma = exp2(b_gk_last)
                else:
                    b_gamma = exp(b_gk_last)
            else:
                b_gamma = tl.zeros([BK1], dtype=tl.float32) + 1.0

            # The M chain must stay true fp32 (FLA PR #740): tf32's 10-bit
            # mantissa compounds over the per-chunk products.
            b_wm = tl.dot(b_w.to(tl.float32), b_m, input_precision="ieee")
            b_m = b_gamma[:, None] * b_m - tl.dot(
                tl.trans(b_k).to(tl.float32), b_wm, input_precision="ieee"
            )

        stride_hm_kv = K + V
        p_m = hm + V + row[:, None] * stride_hm_kv + col[None, :]
        tl.store(
            p_m,
            b_m.to(p_m.dtype.element_ty),
            mask=(row < K)[:, None] & (col < K)[None, :],
        )


# No autotune: the kernel writes the state pool in place (same hazard as the
# main chunk_gated_delta_rule_fwd_h kernel — multi-config benchmarking would
# read back its own writes and corrupt the pool).
@triton.jit(do_not_specialize=["world_size", "rank"])
def cp_merge_fwd_kernel(
    ag_hm,
    h0,
    initial_state,
    initial_state_indices,
    stride_init_state,
    world_size,
    rank,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_POOL: tl.constexpr,
):
    """Phase 3: replay the gathered affine chain (fp32).

    Grid: (cdiv(V, BV), N, HV). The pool state (v-first [V, K]) is tiled along
    V; the transposed recurrence S'^T = S^T M^T + S_ext^T acts on each V-row
    tile independently. At j == rank the running prefix is stored as this
    rank's entry state h0; after the full chain the (rank-independent) global
    final state is written back to the pool slot.
    """
    i_v, i_n, i_h = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2),
    )
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    valid_state = False
    p_pool = initial_state
    if HAS_POOL:
        index = tl.load(initial_state_indices + i_n).to(tl.int64)
        # Padded rows carry the -1 sentinel: seed zeros, skip the writeback.
        valid_state = index >= 0
        p_pool = (
            initial_state
            + index * stride_init_state
            + i_h * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )

    if HAS_POOL and valid_state:
        b_h = tl.load(p_pool, mask=m_v[:, None] & m_k[None, :], other=0.0).to(
            tl.float32
        )
    else:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)

    stride_hm = K + V
    for j in range(world_size):
        if j == rank:
            p_h0 = (
                h0
                + (i_n * HV + i_h) * V * K
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(
                p_h0, b_h.to(p_h0.dtype.element_ty), mask=m_v[:, None] & m_k[None, :]
            )
        # grid axis 1 is N, so num_programs(1) recovers the batch dim of ag_hm
        # [W, N, HV, K, K+V] without specializing the kernel on N.
        base = ((j * tl.num_programs(1) + i_n) * HV + i_h) * K * (K + V)
        # S_ext is stored k-first ([K, V]); M is [K, K] at column offset V.
        p_he = ag_hm + base + o_k[:, None] * stride_hm + o_v[None, :]
        b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0)
        p_m = ag_hm + base + V + o_k[:, None] * stride_hm + o_k[None, :]
        b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0)
        # v-first update: S'^T = S^T M^T + S_ext^T. True fp32 (no tf32): the
        # replayed chain is the whole point of the fp32 hm buffer.
        b_h = tl.dot(b_h, tl.trans(b_m), input_precision="ieee") + tl.trans(b_he)

    if HAS_POOL and valid_state:
        tl.store(
            p_pool,
            b_h.to(initial_state.dtype.element_ty),
            mask=m_v[:, None] & m_k[None, :],
        )


def cp_local_transition(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_exp2: bool = False,
    num_global_seqs: Optional[int] = None,
    local_seq_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Phase 1 wrapper: local shard -> hm [N_global, HV, K, K+V] fp32 (pure).

    Rows of globally-known but locally-absent sequences (compaction, see
    LinearAttnCPContext) stay at the pre-initialized identity affine (0, I).
    """
    assert not (
        use_exp2 and g is not None
    ), "use_exp2 covers only the per-channel gk path; scalar g stays natural-exp"
    assert cu_seqlens is not None, "CP pre-scan requires varlen (cu_seqlens)"
    B, T, H, K, V = *k.shape, u.shape[-1]
    HV = u.shape[-2]
    assert B == 1, "varlen mode requires packed batch (B == 1)"
    n_local = len(cu_seqlens) - 1
    n_global = num_global_seqs if num_global_seqs is not None else n_local
    if local_seq_ids is None:
        assert n_global == n_local
        local_seq_ids = torch.arange(
            n_local, device=cu_seqlens.device, dtype=torch.int32
        )

    hm = k.new_zeros(n_global, HV, K, K + V, dtype=torch.float32)
    diag = torch.arange(K, device=hm.device)
    hm[:, :, diag, V + diag] = 1.0
    if n_local > 0:
        block_size = 32 if K <= 64 else 64
        grid = (
            triton.cdiv(V, block_size) + triton.cdiv(K, block_size),
            n_local,
            HV,
        )
        cp_pre_scan_fwd_kernel[grid](
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            hm=hm,
            seq_map=local_seq_ids,
            cu_seqlens=cu_seqlens,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=CHUNK_SIZE,
            BLOCK_SIZE=block_size,
            BK1=triton.next_power_of_2(K),
            USE_EXP2=use_exp2,
        )
    return hm


def cp_merge_states(
    ag_hm: torch.Tensor,
    rank: int,
    world_size: int,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Phase 3 wrapper: gathered hm -> per-rank entry states h0 [N, HV, V, K].

    Side effect: when a state pool is given, every rank writes the identical
    global final state back into its pool-slot replica (in place).
    """
    W, N, HV, K, KV = ag_hm.shape
    V = KV - K
    assert W == world_size
    assert K <= 128, "merge kernel holds a full KxK tile; K > 128 needs tiling"

    h0 = ag_hm.new_empty(N, HV, V, K)
    bv = min(64, triton.next_power_of_2(V))
    grid = (triton.cdiv(V, bv), N, HV)
    cp_merge_fwd_kernel[grid](
        ag_hm=ag_hm,
        h0=h0,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        stride_init_state=(
            initial_state.stride(0) if initial_state is not None else 0
        ),
        world_size=world_size,
        rank=rank,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=bv,
        HAS_POOL=initial_state is not None,
        # 8 warps: the [BK, BK] transition tile alone is 64 KiB of fp32;
        # spreading it over 256 threads keeps it out of local-memory spill.
        num_warps=8,
    )
    return h0


def chunk_gated_delta_rule_fwd_h_cp_pre_process(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cp_context: Optional[LinearAttnCPContext] = None,
    use_exp2: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """CP orchestrator: pre-scan -> all-gather -> merge.

    ``initial_state_indices`` are the pool slots of ALL global sequences
    ([N_global]) — the merge seeds from and writes back to every one of them
    on every rank, including sequences with no tokens on this rank.

    Returns (h0_scratch, scratch_indices) to be fed to the main
    chunk_gated_delta_rule_fwd_h call in place of the pool state, so its
    INPLACE_UPDATE epilogue lands on scratch instead of the pool. The
    scratch indices map each LOCAL sequence to its global h0 row. Passes the
    inputs through unchanged when CP is inactive, so callers may invoke it
    unconditionally.
    """
    if cp_context is None or not cp_context.is_active:
        return initial_state, initial_state_indices
    assert cp_context.group is not None, "active CP requires a process group"

    hm = cp_local_transition(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        cu_seqlens=cu_seqlens,
        use_exp2=use_exp2,
        num_global_seqs=cp_context.num_global_seqs,
        local_seq_ids=cp_context.local_seq_ids,
    )
    ag_hm = hm.new_empty(cp_context.world_size, *hm.shape)
    dist.all_gather_into_tensor(ag_hm, hm, group=cp_context.group)

    h0 = cp_merge_states(
        ag_hm=ag_hm,
        rank=cp_context.rank,
        world_size=cp_context.world_size,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
    )
    index_dtype = (
        initial_state_indices.dtype
        if initial_state_indices is not None
        else torch.int32
    )
    if cp_context.local_seq_ids is not None:
        scratch_indices = cp_context.local_seq_ids.to(index_dtype)
    else:
        scratch_indices = torch.arange(
            h0.shape[0], device=h0.device, dtype=index_dtype
        )
    return h0, scratch_indices
