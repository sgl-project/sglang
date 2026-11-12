# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""KDA ReplaySSM commit for sparse row sets (fold-mode flushes and track crossings).

The stock exact-fold kernel (kda_replayssm_spec_decode.py) with its tile, op
order and one-warp programs unchanged (so, at the same BV, bit-identical), plus
a request grid-stride: program (v, p, layer*head) folds rows p, p+P, p+2P, ...
of a row list whose live length can come from a device tensor. A sparse commit
compacts its rows to the front and launches a small P, so idle programs exit
after one load instead of B x layers x heads x V/BV early exits (the stock grid
is dominated by them on small row sets). The first version tiled [BV, K] with
4-8 warps and was slower than the stock kernel at every N; dropped.

Rows with commit_len <= 0 or a null slot are skipped; HAS_TRACK snapshots the
state after step ``mamba_steps_to_track`` into the track slot; ``ring_base``
gives each row's first ring record (KDA fold mode keeps windows in ping-pong
ring halves).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def kda_replayssm_fold_v2_kernel(
    h0,  # [layers, slots, HV, V, K] fp32 / bf16 (folded in place, in fp32)
    rawv_cache,  # [layers, slots, HV, L, V]
    rawk_cache,  # [layers, slots, H, L, K]
    gk_cache,  # [layers, slots, HV, L, K] fp32 log-decay
    beta_cache,  # [layers, slots, HV, L] fp32
    ssm_state_indices,  # [B] int
    commit_lens,  # [B] int
    mamba_track_indices,  # [B] int (or dummy)
    mamba_steps_to_track,  # [B] int (or dummy)
    row_count,  # [1] int32: rows [0, count) are live (HAS_COUNT), else B
    ring_base,  # [B] int32: first ring record of each row's window (HAS_BASE)
    B,
    REQ_STRIDE,
    stride_state_slot,
    stride_rawv_slot,
    stride_rawk_slot,
    stride_gk_slot,
    stride_beta_slot,
    stride_state_layer,
    stride_rawv_layer,
    stride_rawk_layer,
    stride_gk_layer,
    stride_beta_layer,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    HAS_BASE: tl.constexpr,
):
    # Same tile and op order as the stock exact-fold kernel ([K, BV] tile, one
    # warp per program -> the same K reduction tree), plus a request
    # grid-stride over (optionally compacted) rows.
    i_v = tl.program_id(0)
    i_p = tl.program_id(1)
    i_hvl = tl.program_id(2)
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    gk_cache = gk_cache + i_layer * stride_gk_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    # [K, BV] tile of the [V, K] (K-contiguous) state, like the stock kernel.
    tile_off = i_hv * V * K + o_v[None, :] * K + o_k[:, None]

    if HAS_COUNT:
        n_rows = tl.load(row_count).to(tl.int32)
    else:
        n_rows = B
    for i_n in range(i_p, n_rows, REQ_STRIDE):
        state_idx = tl.load(ssm_state_indices + i_n).to(tl.int64)
        n_commit = tl.load(commit_lens + i_n).to(tl.int32)
        if (state_idx > NULL_BLOCK_ID) & (n_commit > 0):
            if HAS_TRACK:
                track_idx = tl.load(mamba_track_indices + i_n).to(tl.int64)
                track_step = tl.load(mamba_steps_to_track + i_n).to(tl.int32)
            else:
                track_idx = NULL_BLOCK_ID
                track_step = -1
            p_h = h0 + state_idx * stride_state_slot + tile_off
            b_h = tl.load(p_h).to(tl.float32)
            if HAS_BASE:
                r0 = tl.load(ring_base + i_n).to(tl.int64)
            else:
                r0 = 0
            p_k = rawk_cache + state_idx * stride_rawk_slot + (i_h * MAX_CACHE_LEN + r0) * K + o_k
            p_v = rawv_cache + state_idx * stride_rawv_slot + (i_hv * MAX_CACHE_LEN + r0) * V + o_v
            p_g = gk_cache + state_idx * stride_gk_slot + (i_hv * MAX_CACHE_LEN + r0) * K + o_k
            p_b = beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + r0
            nxt_k = tl.load(p_k)
            nxt_v = tl.load(p_v)
            nxt_g = tl.load(p_g)
            nxt_b = tl.load(p_b)
            for t in range(0, n_commit):
                b_k = nxt_k.to(tl.float32)
                b_v = nxt_v.to(tl.float32)
                b_gk = nxt_g.to(tl.float32)
                b_beta = nxt_b.to(tl.float32)
                has_next = t + 1 < n_commit
                nxt_k = tl.load(p_k + (t + 1) * K, mask=has_next, other=0.0)
                nxt_v = tl.load(p_v + (t + 1) * V, mask=has_next, other=0.0)
                nxt_g = tl.load(p_g + (t + 1) * K, mask=has_next, other=0.0)
                nxt_b = tl.load(p_b + (t + 1), mask=has_next, other=0.0)
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
                b_h *= tl.exp(b_gk[:, None])
                b_v -= tl.sum(b_h * b_k[:, None], 0)
                b_v *= b_beta
                b_h += b_k[:, None] * b_v[None, :]
                if HAS_TRACK:
                    if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                        tl.store(
                            h0 + track_idx * stride_state_slot + tile_off,
                            b_h.to(h0.dtype.element_ty),
                        )
            # A BF16 state is rounded once per commit, as in the stock kernel.
            tl.store(p_h, b_h.to(p_h.dtype.element_ty))


def commit_kda_replayssm_v2_all_layers(
    checkpoint_state: torch.Tensor,  # [layers, slots, HV, V, K] fp32 / bf16, in place
    rawv_cache: torch.Tensor,  # [layers, slots, HV, L, V]
    rawk_cache: torch.Tensor,  # [layers, slots, H, L, K]
    gk_cache: torch.Tensor,  # [layers, slots, HV, L, K] fp32
    beta_cache: torch.Tensor,  # [layers, slots, HV, L] fp32
    ssm_state_indices: torch.Tensor,  # [B] int
    commit_lens: torch.Tensor,  # [B] int (<= 0: skip the row)
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
    req_programs: int | None = None,
    block_v: int = 16,
    num_warps: int = 1,
    row_count: torch.Tensor | None = None,
    ring_base: torch.Tensor | None = None,
) -> None:
    """Fold each request's first ``commit_lens`` ring records into its state.

    ``req_programs`` (P) sets the request grid-stride: None = one program per
    request (full commit). A sparse commit (e.g. only the requests whose window
    crosses a track boundary) compacts its rows to the front, passes their
    device-side count as ``row_count`` and a small P, so idle programs exit
    after one load instead of scanning B rows.
    """
    num_layers, num_slots, HV, V, K = checkpoint_state.shape
    assert checkpoint_state.dtype in (torch.float32, torch.bfloat16)
    assert checkpoint_state.stride()[2:] == (V * K, K, 1)
    assert rawv_cache.stride()[2:] == (max_cache_len * V, V, 1)
    assert rawk_cache.stride()[2:] == (max_cache_len * K, K, 1)
    assert gk_cache.stride()[2:] == (max_cache_len * K, K, 1)
    assert beta_cache.stride()[2:] == (max_cache_len, 1)
    assert K == triton.next_power_of_2(K) and V % block_v == 0
    B = ssm_state_indices.shape[0]
    if B == 0:
        return
    P = B if req_programs is None else max(1, min(int(req_programs), B))
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    track_t = mamba_track_indices if has_track else ssm_state_indices
    steps_t = mamba_steps_to_track if has_track else commit_lens
    assert ssm_state_indices.is_contiguous() and commit_lens.is_contiguous()
    if has_track:
        assert track_t.is_contiguous() and steps_t.is_contiguous()
    grid = (V // block_v, P, num_layers * HV)
    kda_replayssm_fold_v2_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        gk_cache,
        beta_cache,
        ssm_state_indices,
        commit_lens,
        track_t,
        steps_t,
        row_count if row_count is not None else commit_lens,
        ring_base if ring_base is not None else commit_lens,
        B,
        P,
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
        H=num_k_heads,
        HV=HV,
        K=K,
        V=V,
        BV=block_v,
        MAX_CACHE_LEN=max_cache_len,
        NULL_BLOCK_ID=null_block_id,
        HAS_TRACK=has_track,
        HAS_COUNT=row_count is not None,
        HAS_BASE=ring_base is not None,
        num_warps=num_warps,
        num_stages=1,
    )
