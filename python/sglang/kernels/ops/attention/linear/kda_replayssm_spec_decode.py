# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""ReplaySSM speculative-decode state commit for KDA (Kimi Delta Attention).

KDA keeps its own recurrent verify kernel for the per-step OUTPUT (unchanged),
so — unlike the GDN spec kernel (gdn_replayssm_spec_decode.py) — this module does
NOT reconstruct the verify output. It only replaces the per-draft-token full-SSM
snapshot cache (``intermediate_ssm``, ``max_running×(γ+1)×[HV,V,K]``, the memory
hog that collapses dspark concurrency) with a small per-request input window +
an exact-fold on commit.

Scheme (fold-every-commit, no circular ring / periodic flush):
  * during verify, the KDA backend stores the draft window's raw inputs
    (raw v, raw pre-norm k, per-K log-decay gate ``gk`` (fp32), beta (fp32)) into
    the per-slot ring at positions ``0..spec_len``;
  * on commit, :func:`commit_kda_replayssm_spec` replays the *accepted* prefix
    ``0..accept_len`` from the persistent checkpoint ``h0`` into ``h0`` in place —
    ``h0`` is always the current committed state, so the next verify reads it
    directly as its initial state (no lag, no chunked reconstruction).

The exact-fold is a BITWISE CLONE of ``fused_recurrent_gated_delta_rule_fwd_kernel``'s
``IS_KDA`` branch (fused_recurrent.py): same full-K fp32 rows with num_warps=1, so
each V column's K reduction matches; BV only partitions independent V columns
across programs and need not match the recurrent kernel's. Same division-form L2
norm (eps inside sqrt), same per-K gate decay ``h *= exp(gk)``, same
decay -> delta -> rank-1 op order. Given identical inputs the folded checkpoint
is bit-identical to the recurrent baseline's committed state (the delta-rule
recurrence is contractive, so no length-dependent error). Do NOT reorder into
tl.dot / reciprocal-multiply; keep num_warps=1 so the reduction trees match.

Linear chain only (dspark γ chain, topk<=1).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Measured on GB300 (K=V=128). On FP32 state any BV gives the same bits. On BF16
# state the compiled code differs with BV and a few elements move by 1-2 BF16
# ulps, so re-tuning it changes the BF16 bits (see _sparse_commit_block_v).
KDA_FOLD_BLOCK_V = 8


@triton.jit
def kda_replayssm_exact_fold_kernel(
    h0,  # [num_slots, HV, V, K] fp32 checkpoint (folded in place)
    rawv_cache,  # [num_slots, HV, L, V]  raw v
    rawk_cache,  # [num_slots, H,  L, K]  raw pre-norm k
    gk_cache,  # [num_slots, HV, L, K]  fp32 per-K log-decay gate
    beta_cache,  # [num_slots, HV, L]     fp32 beta
    ssm_state_indices,  # [B] int  physical slot per request
    accept_lens,  # [B] int  committed prefix length per request
    mamba_track_indices,  # [B] int  extra_buffer track slot (or NULL) per request
    mamba_steps_to_track,  # [B] int  crossing step (or -1) per request
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_gk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    stride_rawk_layer: tl.constexpr,
    stride_gk_layer: tl.constexpr,
    stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_accept: tl.constexpr,
    stride_track: tl.constexpr,
    stride_steps: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    ring_indices,
    stride_ring: tl.constexpr,
    SEPARATE_RING_INDEX: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    # program_id(2) packs (layer, v-head): layer-major so a single launch folds
    # all KDA layers. num_layers=1 launches (per-layer entry) keep i_layer == 0.
    i_hvl = tl.program_id(2)
    # int64: layer stride * i_layer overflows int32 at K3 scale (69 layers x
    # ~34M-element per-layer stride > 2^31).
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    # Shift the layer-indexed bases once; every pointer below is layer-relative.
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    gk_cache = gk_cache + i_layer * stride_gk_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    if SEPARATE_RING_INDEX:
        ring_idx = tl.load(ring_indices + i_n * stride_ring).to(tl.int64)
        # Padding rows are -1. A negative slot aliases request 0's ring.
        if ring_idx < 0:
            return
    else:
        ring_idx = state_idx
    n_commit = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_commit <= 0:
        return

    # extra_buffer: snapshot the interval-crossing state into the track ping-pong
    # slot on the step it crosses (mask-gated: -1 track step / NULL slot = skip).
    if HAS_TRACK:
        track_idx = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
        track_step = tl.load(mamba_steps_to_track + i_n * stride_steps).to(tl.int32)
    else:
        track_idx = NULL_BLOCK_ID
        track_step = -1

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # [BK, BV] tile: K rows / V cols, matching the recurrent baseline's memory
    # offset (v * K + k, K contiguous).
    p_h0 = (
        h0
        + state_idx * stride_state_slot
        + i_hv * V * K
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    p_k = rawk_cache + ring_idx * stride_rawk_slot + i_h * MAX_CACHE_LEN * K + o_k
    p_v = rawv_cache + ring_idx * stride_rawv_slot + i_hv * MAX_CACHE_LEN * V + o_v
    p_g = gk_cache + ring_idx * stride_gk_slot + i_hv * MAX_CACHE_LEN * K + o_k
    p_b = beta_cache + ring_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN
    nxt_k = tl.load(p_k, mask=mask_k, other=0.0)
    nxt_v = tl.load(p_v, mask=mask_v, other=0.0)
    nxt_g = tl.load(p_g, mask=mask_k, other=0.0)
    nxt_b = tl.load(p_b)
    for t in range(0, n_commit):
        b_k = nxt_k.to(tl.float32)
        b_v = nxt_v.to(tl.float32)
        b_gk = nxt_g.to(tl.float32)
        b_beta = nxt_b.to(tl.float32)
        has_next = t + 1 < n_commit
        nxt_k = tl.load(p_k + (t + 1) * K, mask=mask_k & has_next, other=0.0)
        nxt_v = tl.load(p_v + (t + 1) * V, mask=mask_v & has_next, other=0.0)
        nxt_g = tl.load(p_g + (t + 1) * K, mask=mask_k & has_next, other=0.0)
        nxt_b = tl.load(p_b + (t + 1), mask=has_next, other=0.0)

        # --- verbatim recurrent update, IS_KDA branch (see module docstring) ---
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_gk[:, None])  # per-K gate decay, broadcast over V
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        # Interval-crossing snapshot -> track slot (state AFTER step `track_step`).
        if HAS_TRACK:
            if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                tl.store(
                    h0
                    + track_idx * stride_state_slot
                    + i_hv * V * K
                    + o_v[None, :] * K
                    + o_k[:, None],
                    b_h.to(h0.dtype.element_ty),
                    mask=mask_h,
                )

    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


# K31HT_KDA_FOLD=tp4: the stock fold body per 8-column V tile, KDA_FOLD_TP_VT consecutive tiles per one-warp program, with the
# next tile's state load issued before the current tile's token chain (pipelines the state load under the chain). Each tile runs
# the same op order as kda_replayssm_exact_fold_kernel, so the folded checkpoint (and the track snapshot) is bit-identical to it.
KDA_FOLD_TP_VT = 4
LAST_FOLD = {"form": None}


@triton.jit
def kda_replayssm_tp_fold_kernel(
    h0,
    rawv_cache,
    rawk_cache,
    gk_cache,
    beta_cache,
    ssm_state_indices,
    accept_lens,
    mamba_track_indices,
    mamba_steps_to_track,
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_gk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    stride_rawk_layer: tl.constexpr,
    stride_gk_layer: tl.constexpr,
    stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_accept: tl.constexpr,
    stride_track: tl.constexpr,
    stride_steps: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    VT: tl.constexpr,
):
    i_vg = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hvl = tl.program_id(2)
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    gk_cache = gk_cache + i_layer * stride_gk_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    n_fold = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_fold <= 0:
        return
    if HAS_TRACK:
        trk_idx = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
        trk_step = tl.load(mamba_steps_to_track + i_n * stride_steps).to(tl.int32)
    else:
        trk_idx = NULL_BLOCK_ID
        trk_step = -1

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    o_v = (i_vg * VT) * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_tile = h0 + state_idx * stride_state_slot + i_hv * V * K + o_v[None, :] * K + o_k[:, None]
    b_h = tl.load(p_tile, mask=mask_h, other=0.0).to(tl.float32)
    for vt in tl.static_range(VT):
        o_v = (i_vg * VT + vt) * BV + tl.arange(0, BV)
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]
        if vt + 1 < VT:
            # host guarantees V % (BV * VT) == 0, so every tile is full and the current mask covers the next tile
            pre_h = tl.load(p_tile + BV * K, mask=mask_h, other=0.0).to(tl.float32)
        p_k = rawk_cache + state_idx * stride_rawk_slot + i_h * MAX_CACHE_LEN * K + o_k
        p_v = rawv_cache + state_idx * stride_rawv_slot + i_hv * MAX_CACHE_LEN * V + o_v
        p_g = gk_cache + state_idx * stride_gk_slot + i_hv * MAX_CACHE_LEN * K + o_k
        p_b = beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN
        nxt_k = tl.load(p_k, mask=mask_k, other=0.0)
        nxt_v = tl.load(p_v, mask=mask_v, other=0.0)
        nxt_g = tl.load(p_g, mask=mask_k, other=0.0)
        nxt_b = tl.load(p_b)
        for t in range(0, n_fold):
            b_k = nxt_k.to(tl.float32)
            b_v = nxt_v.to(tl.float32)
            b_gk = nxt_g.to(tl.float32)
            b_beta = nxt_b.to(tl.float32)
            has_next = t + 1 < n_fold
            nxt_k = tl.load(p_k + (t + 1) * K, mask=mask_k & has_next, other=0.0)
            nxt_v = tl.load(p_v + (t + 1) * V, mask=mask_v & has_next, other=0.0)
            nxt_g = tl.load(p_g + (t + 1) * K, mask=mask_k & has_next, other=0.0)
            nxt_b = tl.load(p_b + (t + 1), mask=has_next, other=0.0)

            # --- verbatim recurrent update, IS_KDA branch (same as kda_replayssm_exact_fold_kernel) ---
            if USE_QK_L2NORM_IN_KERNEL:
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
            b_h *= tl.exp(b_gk[:, None])
            b_v -= tl.sum(b_h * b_k[:, None], 0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            if HAS_TRACK:
                if (t == trk_step) and (trk_idx > NULL_BLOCK_ID):
                    tl.store(
                        h0
                        + trk_idx * stride_state_slot
                        + i_hv * V * K
                        + o_v[None, :] * K
                        + o_k[:, None],
                        b_h.to(h0.dtype.element_ty),
                        mask=mask_h,
                    )
        tl.store(p_tile, b_h.to(p_tile.dtype.element_ty), mask=mask_h)
        if vt + 1 < VT:
            b_h = pre_h
            p_tile = p_tile + BV * K


# K31HT_KDA_FOLD=tp4rm: kda_replayssm_tp_fold_kernel with a request-major grid (V-groups, layer * head, request): the same
# per-program body and op order (bit-identical result); one request's programs dispatch together, which removes most of TP4's
# ragged-accept penalty. Renamed so a profile shows which form ran.
@triton.jit
def kda_replayssm_tp_fold_rm_kernel(
    h0,
    rawv_cache,
    rawk_cache,
    gk_cache,
    beta_cache,
    ssm_state_indices,
    accept_lens,
    mamba_track_indices,
    mamba_steps_to_track,
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_gk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    stride_rawk_layer: tl.constexpr,
    stride_gk_layer: tl.constexpr,
    stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_accept: tl.constexpr,
    stride_track: tl.constexpr,
    stride_steps: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    VT: tl.constexpr,
):
    i_vg = tl.program_id(0)
    i_n = tl.program_id(2)
    i_hvl = tl.program_id(1)
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    gk_cache = gk_cache + i_layer * stride_gk_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    n_fold = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_fold <= 0:
        return
    if HAS_TRACK:
        trk_idx = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
        trk_step = tl.load(mamba_steps_to_track + i_n * stride_steps).to(tl.int32)
    else:
        trk_idx = NULL_BLOCK_ID
        trk_step = -1

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    o_v = (i_vg * VT) * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_tile = h0 + state_idx * stride_state_slot + i_hv * V * K + o_v[None, :] * K + o_k[:, None]
    b_h = tl.load(p_tile, mask=mask_h, other=0.0).to(tl.float32)
    for vt in tl.static_range(VT):
        o_v = (i_vg * VT + vt) * BV + tl.arange(0, BV)
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]
        if vt + 1 < VT:
            # host guarantees V % (BV * VT) == 0, so every tile is full and the current mask covers the next tile
            pre_h = tl.load(p_tile + BV * K, mask=mask_h, other=0.0).to(tl.float32)
        p_k = rawk_cache + state_idx * stride_rawk_slot + i_h * MAX_CACHE_LEN * K + o_k
        p_v = rawv_cache + state_idx * stride_rawv_slot + i_hv * MAX_CACHE_LEN * V + o_v
        p_g = gk_cache + state_idx * stride_gk_slot + i_hv * MAX_CACHE_LEN * K + o_k
        p_b = beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN
        nxt_k = tl.load(p_k, mask=mask_k, other=0.0)
        nxt_v = tl.load(p_v, mask=mask_v, other=0.0)
        nxt_g = tl.load(p_g, mask=mask_k, other=0.0)
        nxt_b = tl.load(p_b)
        for t in range(0, n_fold):
            b_k = nxt_k.to(tl.float32)
            b_v = nxt_v.to(tl.float32)
            b_gk = nxt_g.to(tl.float32)
            b_beta = nxt_b.to(tl.float32)
            has_next = t + 1 < n_fold
            nxt_k = tl.load(p_k + (t + 1) * K, mask=mask_k & has_next, other=0.0)
            nxt_v = tl.load(p_v + (t + 1) * V, mask=mask_v & has_next, other=0.0)
            nxt_g = tl.load(p_g + (t + 1) * K, mask=mask_k & has_next, other=0.0)
            nxt_b = tl.load(p_b + (t + 1), mask=has_next, other=0.0)

            # --- verbatim recurrent update, IS_KDA branch (same as kda_replayssm_exact_fold_kernel) ---
            if USE_QK_L2NORM_IN_KERNEL:
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
            b_h *= tl.exp(b_gk[:, None])
            b_v -= tl.sum(b_h * b_k[:, None], 0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            if HAS_TRACK:
                if (t == trk_step) and (trk_idx > NULL_BLOCK_ID):
                    tl.store(
                        h0
                        + trk_idx * stride_state_slot
                        + i_hv * V * K
                        + o_v[None, :] * K
                        + o_k[:, None],
                        b_h.to(h0.dtype.element_ty),
                        mask=mask_h,
                    )
        tl.store(p_tile, b_h.to(p_tile.dtype.element_ty), mask=mask_h)
        if vt + 1 < VT:
            b_h = pre_h
            p_tile = p_tile + BV * K


def commit_kda_replayssm_spec(
    checkpoint_state: torch.Tensor,  # [num_slots, HV, V, K] fp32, folded in place
    rawv_cache: torch.Tensor,  # [num_slots, HV, L, V]
    rawk_cache: torch.Tensor,  # [num_slots, H,  L, K]
    gk_cache: torch.Tensor,  # [num_slots, HV, L, K] fp32
    beta_cache: torch.Tensor,  # [num_slots, HV, L]    fp32
    ssm_state_indices: torch.Tensor,  # [B] int
    accept_lens: torch.Tensor,  # [B] int  (incl. the bonus token)
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: torch.Tensor | None = None,  # [B] extra_buffer track slot
    mamba_steps_to_track: torch.Tensor | None = None,  # [B] crossing step (or -1)
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = 0,
    ring_indices: torch.Tensor | None = None,
) -> None:
    """Replay each request's accepted window into its fp32 checkpoint in place.

    Full-K rows with num_warps=1 keep each V column's K reduction identical to the
    recurrent kernel, so the folded checkpoint is bit-identical to the recurrent
    baseline's committed state for any BV. With extra_buffer (mamba_track_indices
    given) the same replay snapshots the interval-crossing state into the track
    slot in one pass, so no separate track scatter / force-flush is needed.
    """
    num_slots, HV, V, K = checkpoint_state.shape
    B = ssm_state_indices.shape[0]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), KDA_FOLD_BLOCK_V)
    grid = (triton.cdiv(V, BV), B, HV)
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    if has_track:
        track_idx_t = mamba_track_indices
        steps_t = mamba_steps_to_track
        stride_track = track_idx_t.stride(0)
        stride_steps = steps_t.stride(0)
    else:
        track_idx_t = ssm_state_indices  # unused (HAS_TRACK False); pass a valid ptr
        steps_t = accept_lens
        stride_track = 0
        stride_steps = 0
    kda_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        gk_cache,
        beta_cache,
        ssm_state_indices,
        accept_lens,
        track_idx_t,
        steps_t,
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        gk_cache.stride(0),
        beta_cache.stride(0),
        0,  # stride_*_layer unused: single-layer entry, i_layer == 0
        0,
        0,
        0,
        0,
        ssm_state_indices.stride(0),
        accept_lens.stride(0),
        stride_track,
        stride_steps,
        H=num_k_heads,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_CACHE_LEN=max_cache_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        NULL_BLOCK_ID=null_block_id,
        HAS_TRACK=has_track,
        ring_indices=ssm_state_indices if ring_indices is None else ring_indices,
        stride_ring=0 if ring_indices is None else ring_indices.stride(0),
        SEPARATE_RING_INDEX=ring_indices is not None,
        num_warps=1,
        num_stages=3,
    )


def commit_kda_replayssm_spec_all_layers(
    checkpoint_state: torch.Tensor,  # [num_layers, num_slots, HV, V, K] fp32, in place
    rawv_cache: torch.Tensor,  # [num_layers, num_slots, HV, L, V]
    rawk_cache: torch.Tensor,  # [num_layers, num_slots, H,  L, K]
    gk_cache: torch.Tensor,  # [num_layers, num_slots, HV, L, K] fp32
    beta_cache: torch.Tensor,  # [num_layers, num_slots, HV, L]    fp32
    ssm_state_indices: torch.Tensor,  # [B] int   (shared across layers)
    accept_lens: torch.Tensor,  # [B] int
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = 0,
    ring_indices: torch.Tensor | None = None,
    block_v: int | None = None,
    fold: str | None = None,
) -> None:
    """Fold every layer's accepted window in a single launch.

    Replaces the per-layer Python loop over commit_kda_replayssm_spec (one launch
    per KDA layer -> ~69 tiny eager launches at bs=1, dispatch-bound). The layer
    is packed into the head grid axis (program_id(2) = layer * HV + head), so the
    result is bit-identical to the loop -- each (layer, head, v-tile) block runs
    the same per-slot recurrent replay. ssm_state_indices / accept_lens / track
    are per-request and shared across layers.
    """
    # fold (None -> env K31HT_KDA_FOLD): unset / '0' = this stock kernel; 'tp4' = kda_replayssm_tp_fold_kernel
    # ('tp4rm' = kda_replayssm_tp_fold_rm_kernel, request-major grid) (bit-identical result); anything else raises.
    fold_knob = os.environ.get("K31HT_KDA_FOLD", "0") if fold is None else fold
    if fold_knob not in ("", "0", "tp4", "tp4rm"):
        raise ValueError(f"K31HT_KDA_FOLD: unset | 0 | tp4 | tp4rm; got {fold_knob!r}")
    num_layers, num_slots, HV, V, K = checkpoint_state.shape
    B = ssm_state_indices.shape[0]
    BK = triton.next_power_of_2(K)
    # BV only partitions independent V columns: bit-identical for any value on
    # FP32 state, not on BF16 state (see KDA_FOLD_BLOCK_V).
    BV = min(triton.next_power_of_2(V), block_v or KDA_FOLD_BLOCK_V)
    grid = (triton.cdiv(V, BV), B, HV * num_layers)
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    if has_track:
        track_idx_t = mamba_track_indices
        steps_t = mamba_steps_to_track
        stride_track = track_idx_t.stride(0)
        stride_steps = steps_t.stride(0)
    else:
        track_idx_t = ssm_state_indices  # unused (HAS_TRACK False); valid ptr
        steps_t = accept_lens
        stride_track = 0
        stride_steps = 0
    if fold_knob == "tp4rm" and V % (BV * KDA_FOLD_TP_VT) == 0:
        kda_replayssm_tp_fold_rm_kernel[(V // (BV * KDA_FOLD_TP_VT), HV * num_layers, B)](
            checkpoint_state,
            rawv_cache,
            rawk_cache,
            gk_cache,
            beta_cache,
            ssm_state_indices,
            accept_lens,
            track_idx_t,
            steps_t,
            checkpoint_state.stride(1),
            rawv_cache.stride(1),
            rawk_cache.stride(1),
            gk_cache.stride(1),
            beta_cache.stride(1),
            checkpoint_state.stride(0),
            rawv_cache.stride(0),
            rawk_cache.stride(0),
            gk_cache.stride(0),
            beta_cache.stride(0),
            ssm_state_indices.stride(0),
            accept_lens.stride(0),
            stride_track,
            stride_steps,
            H=num_k_heads,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            MAX_CACHE_LEN=max_cache_len,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            NULL_BLOCK_ID=null_block_id,
            HAS_TRACK=has_track,
            VT=KDA_FOLD_TP_VT,
            num_warps=1,
            num_stages=3,
        )
        LAST_FOLD["form"] = "tp4rm"
        return
    if fold_knob == "tp4" and V % (BV * KDA_FOLD_TP_VT) == 0:
        kda_replayssm_tp_fold_kernel[(V // (BV * KDA_FOLD_TP_VT), B, HV * num_layers)](
            checkpoint_state,
            rawv_cache,
            rawk_cache,
            gk_cache,
            beta_cache,
            ssm_state_indices,
            accept_lens,
            track_idx_t,
            steps_t,
            checkpoint_state.stride(1),
            rawv_cache.stride(1),
            rawk_cache.stride(1),
            gk_cache.stride(1),
            beta_cache.stride(1),
            checkpoint_state.stride(0),
            rawv_cache.stride(0),
            rawk_cache.stride(0),
            gk_cache.stride(0),
            beta_cache.stride(0),
            ssm_state_indices.stride(0),
            accept_lens.stride(0),
            stride_track,
            stride_steps,
            H=num_k_heads,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            MAX_CACHE_LEN=max_cache_len,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            NULL_BLOCK_ID=null_block_id,
            HAS_TRACK=has_track,
            VT=KDA_FOLD_TP_VT,
            num_warps=1,
            num_stages=3,
        )
        LAST_FOLD["form"] = "tp4"
        return
    kda_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        gk_cache,
        beta_cache,
        ssm_state_indices,
        accept_lens,
        track_idx_t,
        steps_t,
        checkpoint_state.stride(1),
        rawv_cache.stride(1),
        rawk_cache.stride(1),
        gk_cache.stride(1),
        beta_cache.stride(1),
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        gk_cache.stride(0),
        beta_cache.stride(0),
        ssm_state_indices.stride(0),
        accept_lens.stride(0),
        stride_track,
        stride_steps,
        H=num_k_heads,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_CACHE_LEN=max_cache_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        NULL_BLOCK_ID=null_block_id,
        HAS_TRACK=has_track,
        ring_indices=ssm_state_indices if ring_indices is None else ring_indices,
        stride_ring=0 if ring_indices is None else ring_indices.stride(0),
        SEPARATE_RING_INDEX=ring_indices is not None,
        num_warps=1,
        num_stages=3,
    )
    if fold_knob not in ("", "0"):  # the engagement record is written only when the knob is set
        LAST_FOLD["form"] = "stock"


def commit_kda_replayssm_after_verify(
    *,
    spec_state,  # MambaPool.SpeculativeState (all layers)
    state_batch_indices: torch.Tensor,  # [B] per-req mamba slot
    accept_lens: torch.Tensor,  # [B] int, incl. the bonus token
    last_correct_step_indices: torch.Tensor,  # [B] conv rollback target step
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    src_indices_raw: torch.Tensor | None = None,
    null_block_id: int = -1,
    ring_indices: torch.Tensor | None = None,
    block_v: int | None = None,
    fold_pending: torch.Tensor | None = None,
) -> None:
    """Fold each layer's accepted window into `temporal` and roll back conv.

    Single commit entry point shared by the generic spec_utils commit and the
    dspark/dflash direct `update_mamba_state_after_mtp_verify` path. The SSM state
    lives in the per-slot ring (written during verify); the fold replays the
    accepted prefix into the fp32 checkpoint, so `temporal` stays current. Conv
    still needs its usual accept-rollback.

    ``fold_pending`` (fold mode, SGLANG_OPT_KDA_SPEC_FOLD_VERIFY): defer the
    fold to the next verify kernel instead (see
    commit_kda_replayssm_fold_after_verify).
    """
    if fold_pending is not None:
        assert src_indices_raw is None, "KDA fold mode does not support PP rows"
        commit_kda_replayssm_fold_after_verify(
            spec_state=spec_state,
            pending=fold_pending,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=null_block_id,
            block_v=block_v,
        )
        return
    L = spec_state.replayssm_rawv.shape[-2]
    num_k_heads = spec_state.replayssm_rawk.shape[2]
    commit_kda_replayssm_spec_all_layers(
        checkpoint_state=spec_state.temporal,
        rawv_cache=spec_state.replayssm_rawv,
        rawk_cache=spec_state.replayssm_rawk,
        gk_cache=spec_state.replayssm_g,
        beta_cache=spec_state.replayssm_beta,
        ssm_state_indices=state_batch_indices,
        accept_lens=accept_lens,
        max_cache_len=L,
        num_k_heads=num_k_heads,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        null_block_id=null_block_id,
        ring_indices=ring_indices,
        block_v=block_v,
    )
    _conv_rollback(
        spec_state,
        state_batch_indices,
        last_correct_step_indices,
        mamba_track_indices,
        mamba_steps_to_track,
        src_indices_raw,
    )


def _conv_rollback(
    spec_state,
    state_batch_indices,
    last_correct_step_indices,
    mamba_track_indices,
    mamba_steps_to_track,
    src_indices_raw,
):
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    # Conv rollback + track-slot conv snapshot, per conv group (fold already did
    # the ssm side via HAS_TRACK). Loop mirrors the recurrent commit's zip; track
    # scatter is mask-gated (step -1 => skip).
    for conv_states, interm_conv in zip(
        spec_state.conv, spec_state.intermediate_conv_window
    ):
        fused_conv_window_scatter_with_mask(
            conv_states,
            interm_conv,
            state_batch_indices,
            last_correct_step_indices,
            src_indices_raw,
        )
        if mamba_track_indices is not None and mamba_steps_to_track is not None:
            fused_conv_window_scatter_with_mask(
                conv_states,
                interm_conv,
                mamba_track_indices,
                mamba_steps_to_track,
                src_indices_raw,
            )


# --- Fold mode (commit-then-verify; SGLANG_OPT_KDA_SPEC_FOLD_VERIFY) -----------------------------------------------
# pending[slot] = a | half << 8: the slot's last verify window sits in ring half `half` (records half * L/2 ...) and its
# first `a` records were accepted but are not yet in `temporal`. Invariant between forwards:
#   temporal[slot] + replay(ring[slot][half * L/2 : half * L/2 + a]) == the stock committed state.
# The fold verify kernel (kda_decode_mtp.py FOLD) replays that window and writes its own window to half ^ 1. A verify
# that cannot fold (any non-nv_cutedsl kernel) writes from ring row 0, so the flush that must precede it leaves half = 1.
FOLD_HALF_SHIFT = 8


def _fold_half(pending_rows: torch.Tensor) -> torch.Tensor:
    return (pending_rows >> FOLD_HALF_SHIFT) & 1


def commit_kda_replayssm_fold_after_verify(
    *,
    spec_state,  # MambaPool.SpeculativeState (all layers)
    pending: torch.Tensor,  # [slots] int32 (see above)
    state_batch_indices: torch.Tensor,  # [B] per-req mamba slot
    accept_lens: torch.Tensor,  # [B] int, incl. the bonus token
    last_correct_step_indices: torch.Tensor,  # [B] conv rollback target step
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
    block_v: int | None = None,
) -> None:
    """Fold-mode post-verify: defer the SSM commit to the next verify.

    The verify just wrote its window to ring half ``half ^ 1`` of each row's
    pending half. Rows whose window crosses an extra_buffer track boundary are
    committed now (sparse commit with the track snapshot, so the radix tree's
    track slot is written in this step, as stock does) and keep 0 pending
    records; every other row records ``accept_len`` pending. The conv rollback
    is unchanged.
    """
    slots = state_batch_indices.to(torch.int64)
    acc = accept_lens.to(torch.int32)
    valid = slots > null_block_id
    safe = torch.where(valid, slots, torch.zeros_like(slots))
    old = pending[safe]
    half_new = _fold_half(old) ^ 1
    if mamba_track_indices is not None and mamba_steps_to_track is not None:
        crossing = (mamba_steps_to_track >= 0) & valid
        _sparse_commit(
            spec_state,
            crossing,
            state_batch_indices,
            acc,
            half_new * (spec_state.replayssm_rawv.shape[-2] // 2),
            mamba_track_indices,
            mamba_steps_to_track,
            null_block_id,
            block_v=block_v,
        )
        acc = torch.where(crossing, torch.zeros_like(acc), acc)
    new = (acc | (half_new << FOLD_HALF_SHIFT)).to(torch.int32)
    # Null rows rewrite their (dummy slot 0) entry unchanged.
    pending.index_put_((safe,), torch.where(valid, new, old))
    _conv_rollback(
        spec_state,
        state_batch_indices,
        last_correct_step_indices,
        mamba_track_indices,
        mamba_steps_to_track,
        None,
    )


def kda_fold_flush(
    *,
    spec_state,
    pending: torch.Tensor,  # [slots] int32
    state_batch_indices: torch.Tensor,  # [B] mamba slots (null rows <= null_block_id)
    null_block_id: int = -1,
    block_v: int | None = None,
) -> None:
    """Commit every pending window of these slots into `temporal`.

    Runs before any forward that reads `temporal` without the fold verify kernel
    (extend, plain decode, a verify that cannot take the nv_cutedsl path).
    Leaves pending = 0 records in half 1, so a following non-fold verify (which
    writes from ring row 0) is recorded as half 0 by the post-verify.
    """
    slots = state_batch_indices.to(torch.int64)
    if slots.numel() == 0:
        return
    valid = slots > null_block_id
    safe = torch.where(valid, slots, torch.zeros_like(slots))
    old = pending[safe]
    lens = torch.where(valid, old & 0xFF, torch.zeros_like(old))
    _sparse_commit(
        spec_state,
        lens > 0,
        state_batch_indices,
        lens,
        _fold_half(old) * (spec_state.replayssm_rawv.shape[-2] // 2),
        None,
        None,
        null_block_id,
        # Most rows of a flushed batch hold a window: one program per row.
        programs=int(slots.numel()),
        block_v=block_v,
    )
    flushed = torch.full_like(old, 1 << FOLD_HALF_SHIFT)
    pending.index_put_((safe,), torch.where(valid, flushed, old))


# Programs per (V tile, layer x head) in a sparse commit; rows past it loop.
_SPARSE_COMMIT_PROGRAMS = 8


def _sparse_commit_block_v(
    state_dtype: torch.dtype, v_dim: int, block_v: int | None = None
) -> int:
    """V tile of the sparse commit. FP32 state gives the same bits for any tile,
    so it keeps the one-warp 16. On BF16 state the stock commit's bits depend on
    the tile, so the sparse commit takes the tile the stock commit takes for the
    same ``block_v`` and stays bit-identical with
    commit_kda_replayssm_spec_all_layers."""
    if state_dtype == torch.bfloat16:
        return min(triton.next_power_of_2(v_dim), block_v or KDA_FOLD_BLOCK_V)
    return 16


def _sparse_commit(
    spec_state,
    mask,
    state_batch_indices,
    lens,
    ring_base,
    track_indices,
    steps,
    null_block_id,
    programs=_SPARSE_COMMIT_PROGRAMS,
    block_v=None,
):
    """Commit only the rows in `mask` without a host sync: masked rows are
    compacted to the front (stable argsort) and their count stays on device."""
    from sglang.kernels.ops.attention.linear.kda_replayssm_commit_v2 import (
        commit_kda_replayssm_v2_all_layers,
    )

    order = torch.argsort((~mask).to(torch.int8), stable=True)
    count = mask.sum(dtype=torch.int32).reshape(1)
    has_track = track_indices is not None and steps is not None
    commit_kda_replayssm_v2_all_layers(
        spec_state.temporal,
        spec_state.replayssm_rawv,
        spec_state.replayssm_rawk,
        spec_state.replayssm_g,
        spec_state.replayssm_beta,
        state_batch_indices[order].to(torch.int32).contiguous(),
        lens[order].to(torch.int32).contiguous(),
        max_cache_len=spec_state.replayssm_rawv.shape[-2],
        num_k_heads=spec_state.replayssm_rawk.shape[2],
        mamba_track_indices=(
            track_indices[order].to(torch.int32).contiguous() if has_track else None
        ),
        mamba_steps_to_track=(
            steps[order].to(torch.int32).contiguous() if has_track else None
        ),
        null_block_id=null_block_id,
        block_v=_sparse_commit_block_v(
            spec_state.temporal.dtype, spec_state.temporal.shape[-2], block_v
        ),
        req_programs=programs,
        row_count=count,
        ring_base=ring_base[order].to(torch.int32).contiguous(),
    )
