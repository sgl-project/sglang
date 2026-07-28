# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM fold-every-commit state commit for GDN (Gated Delta Network).

Counterpart of the KDA fold-every-commit spec-verify (Kimi-K3): the verify
output comes from the unchanged recurrent verify kernel reading the
always-current fp32 checkpoint, so — unlike the circular-ring GDN spec kernel
(gdn_replayssm_spec_decode.py) — this module does NOT reconstruct the verify
output and keeps no cross-step cursors. It only replaces the per-draft-token
full-SSM snapshot cache (``intermediate_ssm``, the dominant speculative
scratch) with a small per-request input window + an exact fold on commit.

Scheme:
  * during verify, the fused ring-write inside
    ``fused_sigmoid_gating_delta_rule_update`` (CACHE_RING) stores the draft
    window's raw inputs (raw v, raw pre-norm k, fp32 scalar gate ``g``, fp32
    ``beta``) into the per-slot window at positions ``0..spec_len``;
  * on commit, :func:`commit_gdn_replayssm_fold_all_layers` replays the
    *accepted* prefix ``0..accept_len`` from the checkpoint (``temporal``)
    into it in place — ``temporal`` is always the current committed state, so
    the next verify reads it directly (no lag, no periodic flush).

The exact fold is a BITWISE CLONE of
``fused_sigmoid_gating_delta_rule_update_kernel``'s GDN (IS_KDA=False) branch:
same [BK, BV] fp32 tile (K rows, V cols), same division-form L2 norm (eps
inside sqrt), same scalar gate decay ``h *= exp(g)``, same
decay→delta→rank-1 op order. Given identical inputs the folded checkpoint is
bit-identical to the recurrent baseline's committed state. Do NOT reorder into
tl.dot / reciprocal-multiply; keep num_warps=1 so the reduction trees match.

With mamba extra_buffer (radix prefix caching) the same replay snapshots the
interval-crossing state into the track slot in one pass, so no device-side
force-flush or separate SSM track scatter is needed — which is what makes this
protocol compatible with extra_buffer and hence the overlap scheduler.

Linear chain only (NEXTN / MTP, topk <= 1).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def gdn_replayssm_exact_fold_kernel(
    h0,  # [num_slots, HV, K, V] fp32 checkpoint (folded in place)
    rawv_cache,  # [num_slots, HV, RL, V]  raw v
    rawk_cache,  # [num_slots, H,  RL, K]  raw pre-norm k
    g_cache,  # [num_slots, HV, RL]     fp32 scalar log-decay gate
    beta_cache,  # [num_slots, HV, RL]     fp32 beta
    ssm_state_indices,  # [B] int  physical slot per request
    accept_lens,  # [B] int  committed prefix length per request (incl. bonus)
    mamba_track_indices,  # [B] int  extra_buffer track slot (or NULL) per request
    mamba_steps_to_track,  # [B] int  crossing step (or -1) per request
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    stride_rawk_layer: tl.constexpr,
    stride_g_layer: tl.constexpr,
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
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    # program_id(2) packs (layer, v-head): layer-major so a single launch folds
    # all GDN layers. num_layers=1 launches (per-layer entry) keep i_layer == 0.
    i_hvl = tl.program_id(2)
    # int64: layer stride * i_layer can overflow an int32 index product.
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    g_cache = g_cache + i_layer * stride_g_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    n_commit = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_commit <= 0:
        return

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
    # offset (v * K + k, K contiguous). Launch-config note (B200, bs 1..128):
    # the kernel is bound by the mandatory checkpoint r+w (~94 MB/request at
    # 45 layers x [16,128,128] fp32) at ~5.0-5.6 TB/s; num_stages is a no-op
    # (the t-loop loads are tiny), num_warps > 1 gains ~6% at bs=1 but changes
    # the tl.sum reduction tree and breaks the bitwise-clone guarantee, and an
    # evict_first hint on this tile wins ~7% only with a warm L2 (back-to-back
    # replays) while losing ~3% in the representative cold-L2 regime -- so the
    # recurrent kernel's (num_warps=1, num_stages=3, plain policy) is kept.
    p_h0 = (
        h0
        + state_idx * stride_state_slot
        + i_hv * K * V
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(0, n_commit):
        phys = t.to(tl.int64)
        b_k = tl.load(
            rawk_cache
            + state_idx * stride_rawk_slot
            + (i_h * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            rawv_cache
            + state_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys) * V
            + o_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        b_g = tl.load(
            g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys
        ).to(tl.float32)
        b_beta = tl.load(
            beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + phys
        ).to(tl.float32)

        # --- verbatim recurrent update, GDN branch (see module docstring) ---
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        # Interval-crossing snapshot -> track slot (state AFTER step `track_step`).
        if HAS_TRACK:
            if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                tl.store(
                    h0
                    + track_idx * stride_state_slot
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None],
                    b_h.to(h0.dtype.element_ty),
                    mask=mask_h,
                )

    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def commit_gdn_replayssm_fold_all_layers(
    checkpoint_state: torch.Tensor,  # [num_layers, num_slots, HV, K, V], in place
    rawv_cache: torch.Tensor,  # [num_layers, num_slots, HV, RL, V]
    rawk_cache: torch.Tensor,  # [num_layers, num_slots, H,  RL, K]
    g_cache: torch.Tensor,  # [num_layers, num_slots, HV, RL] fp32
    beta_cache: torch.Tensor,  # [num_layers, num_slots, HV, RL] fp32
    ssm_state_indices: torch.Tensor,  # [B] int   (shared across layers)
    accept_lens: torch.Tensor,  # [B] int, incl. the bonus token
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = -1,
) -> None:
    """Fold every layer's accepted window in a single launch.

    The layer is packed into the head grid axis (program_id(2) = layer * HV +
    head), so the result is bit-identical to a per-layer loop — each (layer,
    head, v-tile) block runs the same per-slot recurrent replay. Tiling clones
    the recurrent verify kernel (full-K rows, BV = min(np2(V), 32) cols,
    num_warps=1) so the folded checkpoint is bit-identical to the recurrent
    baseline's committed state.
    """
    num_layers, num_slots, HV = checkpoint_state.shape[:3]
    K = rawk_cache.shape[-1]
    V = rawv_cache.shape[-1]
    B = ssm_state_indices.shape[0]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (triton.cdiv(V, BV), B, HV * num_layers)
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    if has_track:
        track_idx_t = mamba_track_indices
        steps_t = mamba_steps_to_track
        stride_track = track_idx_t.stride(0)
        stride_steps = steps_t.stride(0)
    else:
        # Unused when HAS_TRACK is False; pass a valid pointer.
        track_idx_t = ssm_state_indices
        steps_t = accept_lens
        stride_track = 0
        stride_steps = 0
    gdn_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        g_cache,
        beta_cache,
        ssm_state_indices,
        accept_lens,
        track_idx_t,
        steps_t,
        checkpoint_state.stride(1),
        rawv_cache.stride(1),
        rawk_cache.stride(1),
        g_cache.stride(1),
        beta_cache.stride(1),
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        g_cache.stride(0),
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
        num_warps=1,
        num_stages=3,
    )


def commit_gdn_replayssm_fold_after_verify(
    *,
    spec_state,  # MambaPool.SpeculativeState (all layers)
    state_batch_indices: torch.Tensor,  # [B] per-req mamba slot
    accept_lens: torch.Tensor,  # [B] int, incl. the bonus token
    last_correct_step_indices: torch.Tensor,  # [B] conv rollback target step
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
) -> None:
    """Fold each layer's accepted window into ``temporal`` and roll back conv.

    The SSM state lives in the per-slot window (written by the fused ring-write
    during verify); the fold replays the accepted prefix into the checkpoint,
    so ``temporal`` stays current. Conv still needs its usual accept-rollback,
    plus the track-slot conv snapshot under extra_buffer (the fold already did
    the SSM side via HAS_TRACK; the track scatter is mask-gated, step -1 =>
    skip).
    """
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    max_cache_len = spec_state.replayssm_rawv.shape[-2]
    num_k_heads = spec_state.replayssm_rawk.shape[2]
    commit_gdn_replayssm_fold_all_layers(
        checkpoint_state=spec_state.temporal,
        rawv_cache=spec_state.replayssm_rawv,
        rawk_cache=spec_state.replayssm_rawk,
        g_cache=spec_state.replayssm_g,
        beta_cache=spec_state.replayssm_beta,
        ssm_state_indices=state_batch_indices,
        accept_lens=accept_lens,
        max_cache_len=max_cache_len,
        num_k_heads=num_k_heads,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        null_block_id=null_block_id,
    )
    for conv_states, interm_conv in zip(
        spec_state.conv, spec_state.intermediate_conv_window
    ):
        fused_conv_window_scatter_with_mask(
            conv_states, interm_conv, state_batch_indices, last_correct_step_indices
        )
        if mamba_track_indices is not None and mamba_steps_to_track is not None:
            fused_conv_window_scatter_with_mask(
                conv_states, interm_conv, mamba_track_indices, mamba_steps_to_track
            )
