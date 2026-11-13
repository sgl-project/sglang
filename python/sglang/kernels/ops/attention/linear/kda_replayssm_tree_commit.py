# SPDX-License-Identifier: Apache-2.0
"""UBTree tree commit for KDA ReplaySSM raw factors.

Raw factors are stored by request-local tree node, and the commit folds only
the sampled root-to-node path, so rejected branches never materialize or
replay full recurrent states.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def kda_replayssm_tree_exact_fold_kernel(
    h0,
    rawv_cache,
    rawk_cache,
    gk_cache,
    beta_cache,
    ssm_state_indices,
    factor_cache_indices,
    accept_index,
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
    stride_accept_row: tl.constexpr,
    stride_track: tl.constexpr,
    stride_steps: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    LOGICAL_TREE_WIDTH: tl.constexpr,
    CACHE_TREE_WIDTH: tl.constexpr,
    MAX_ACCEPT: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    """Fold one sampled root-to-node path into the persistent KDA state."""
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hvl = tl.program_id(2)
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)

    h0 += i_layer * stride_state_layer
    rawv_cache += i_layer * stride_rawv_layer
    rawk_cache += i_layer * stride_rawk_layer
    gk_cache += i_layer * stride_gk_layer
    beta_cache += i_layer * stride_beta_layer

    slot = tl.load(ssm_state_indices + i_n).to(tl.int64)
    factor_slot = tl.load(factor_cache_indices + i_n).to(tl.int64)
    n_commit = tl.load(accept_lens + i_n).to(tl.int32)
    if slot <= NULL_BLOCK_ID or factor_slot < 0 or n_commit <= 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_h = h0 + slot * stride_state_slot + i_hv * V * K + o_v[None, :] * K + o_k[:, None]
    state = tl.load(p_h, mask=mask_h, other=0.0).to(tl.float32)

    if HAS_TRACK:
        track_slot = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
        track_node = tl.load(mamba_steps_to_track + i_n * stride_steps).to(tl.int64)
    else:
        track_slot = NULL_BLOCK_ID
        track_node = -1

    request_node_base = i_n * LOGICAL_TREE_WIDTH
    for step in range(MAX_ACCEPT):
        active = step < n_commit
        global_node = tl.load(
            accept_index + i_n * stride_accept_row + step,
            mask=active,
            other=request_node_base,
        ).to(tl.int64)
        node = global_node - request_node_base
        active = active & (node >= 0) & (node < LOGICAL_TREE_WIDTH)
        safe_node = tl.maximum(node, 0)

        b_k = tl.load(
            rawk_cache
            + factor_slot * stride_rawk_slot
            + (i_h * CACHE_TREE_WIDTH + safe_node) * K
            + o_k,
            mask=active & mask_k,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            rawv_cache
            + factor_slot * stride_rawv_slot
            + (i_hv * CACHE_TREE_WIDTH + safe_node) * V
            + o_v,
            mask=active & mask_v,
            other=0.0,
        ).to(tl.float32)
        b_g = tl.load(
            gk_cache
            + factor_slot * stride_gk_slot
            + (i_hv * CACHE_TREE_WIDTH + safe_node) * K
            + o_k,
            mask=active & mask_k,
            other=0.0,
        ).to(tl.float32)
        b_beta = tl.load(
            beta_cache
            + factor_slot * stride_beta_slot
            + i_hv * CACHE_TREE_WIDTH
            + safe_node,
            mask=active,
            other=0.0,
        ).to(tl.float32)

        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        next_state = state * tl.exp(b_g[:, None])
        b_v -= tl.sum(next_state * b_k[:, None], 0)
        b_v *= b_beta
        next_state += b_k[:, None] * b_v[None, :]
        state = tl.where(active, next_state, state)

        if HAS_TRACK:
            if active:
                if node == track_node:
                    if track_slot > NULL_BLOCK_ID:
                        tl.store(
                            h0
                            + track_slot * stride_state_slot
                            + i_hv * V * K
                            + o_v[None, :] * K
                            + o_k[:, None],
                            state.to(h0.dtype.element_ty),
                            mask=mask_h,
                        )

    tl.store(p_h, state.to(p_h.dtype.element_ty), mask=mask_h)


def commit_kda_replayssm_tree_all_layers(
    *,
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    gk_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    factor_cache_indices: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    tree_width: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
) -> None:
    """Commit sampled UBTree paths for every KDA layer in one launch."""
    num_layers, _, num_value_heads, value_dim, key_dim = checkpoint_state.shape
    num_key_heads = rawk_cache.shape[2]
    cache_tree_width = rawv_cache.shape[-2]
    if tree_width > cache_tree_width:
        raise ValueError(
            f"logical tree width {tree_width} exceeds cache width {cache_tree_width}"
        )
    block_key = triton.next_power_of_2(key_dim)
    batch_size = accept_lens.shape[0]
    if batch_size != 1:
        raise ValueError(
            f"factorized UBTree commit supports only BS=1, got {batch_size}"
        )
    # The commit keeps a full [BK, BV] FP32 state tile live across the
    # accept-path loop. BV=8 avoids register/occupancy pressure for V=128.
    block_value = 8 if value_dim == 128 else min(triton.next_power_of_2(value_dim), 32)
    if block_value not in (8, 16, 32):
        raise ValueError(f"unsupported tree commit block_value={block_value}")
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    track_indices = mamba_track_indices if has_track else ssm_state_indices
    track_steps = mamba_steps_to_track if has_track else accept_lens
    grid = (
        triton.cdiv(value_dim, block_value),
        batch_size,
        num_layers * num_value_heads,
    )
    kda_replayssm_tree_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        gk_cache,
        beta_cache,
        ssm_state_indices,
        factor_cache_indices,
        accept_index,
        accept_lens,
        track_indices,
        track_steps,
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
        accept_index.stride(0),
        track_indices.stride(0) if has_track else 0,
        track_steps.stride(0) if has_track else 0,
        H=num_key_heads,
        HV=num_value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        LOGICAL_TREE_WIDTH=tree_width,
        CACHE_TREE_WIDTH=cache_tree_width,
        MAX_ACCEPT=accept_index.shape[1],
        NULL_BLOCK_ID=null_block_id,
        HAS_TRACK=has_track,
        num_warps=1,
        num_stages=3,
    )


def commit_kda_tree_after_verify(
    *,
    spec_state,
    state_batch_indices: torch.Tensor,
    factor_cache_indices: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    tree_width: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
) -> None:
    """Commit only the sampled UBTree path from compact per-node KDA inputs."""
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_tree_raw_commit_with_mask,
    )

    commit_kda_replayssm_tree_all_layers(
        checkpoint_state=spec_state.temporal,
        rawv_cache=spec_state.tree_rawv,
        rawk_cache=spec_state.tree_rawk,
        gk_cache=spec_state.tree_g,
        beta_cache=spec_state.tree_beta,
        ssm_state_indices=state_batch_indices,
        factor_cache_indices=factor_cache_indices,
        accept_index=accept_index,
        accept_lens=accept_lens,
        tree_width=tree_width,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        null_block_id=null_block_id,
    )
    for conv_states, interm_conv in zip(
        spec_state.conv, spec_state.intermediate_conv_window
    ):
        fused_conv_tree_raw_commit_with_mask(
            conv_states,
            interm_conv,
            state_batch_indices,
            factor_cache_indices,
            accept_index,
            accept_lens,
            tree_width=tree_width,
            track_indices=mamba_track_indices,
            track_nodes=mamba_steps_to_track,
        )
