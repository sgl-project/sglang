# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM speculative verify for Mamba2 SSD mixers (linear draft chain).

The target verify keeps the per-draft SSM states out of memory: it records each
draft token's post-conv x, B, softplus(dt + dt_bias) and exp(A * dt) into a ring
indexed by verify batch row, and the commit replays the accepted prefix into
the SSM state. Both kernels repeat the tied-head-dim arithmetic of
``_selective_scan_update_kernel`` (triton_ops/mamba_ssm.py), so the committed
state matches the one the per-draft snapshot path would scatter.

Coverage: CUDA, ``dt``/``A``/``dt_bias`` tied across head_dim (one decay per
head), any head_dim / d_state; rows are ``[rows, T, ...]`` contiguous rings.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import softplus


@triton.jit
def _mamba2_replayssm_record_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    dt_bias_ptr,
    a_ptr,
    x_ring_ptr,
    b_ring_ptr,
    dt_ring_ptr,
    decay_ring_ptr,
    stride_x_tok,
    stride_b_tok,
    stride_dt_tok,
    num_steps,
    RING_T: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    slot = (tok // num_steps) * RING_T + tok % num_steps

    offs_v = tl.arange(0, BV)
    x = tl.load(x_ptr + tok * stride_x_tok + h * V + offs_v, mask=offs_v < V)
    tl.store(x_ring_ptr + (slot * HV + h) * V + offs_v, x, mask=offs_v < V)

    dt = tl.load(dt_ptr + tok * stride_dt_tok + h).to(tl.float32)
    dt += tl.load(dt_bias_ptr + h).to(tl.float32)
    dt = softplus(dt)
    a = tl.load(a_ptr + h).to(tl.float32)
    tl.store(dt_ring_ptr + slot * HV + h, dt)
    tl.store(decay_ring_ptr + slot * HV + h, tl.exp(a * dt))

    if h % HEADS_PER_GROUP == 0:
        g = h // HEADS_PER_GROUP
        offs_k = tl.arange(0, BK)
        b = tl.load(b_ptr + tok * stride_b_tok + g * K + offs_k, mask=offs_k < K)
        tl.store(b_ring_ptr + (slot * G + g) * K + offs_k, b, mask=offs_k < K)


def record_mamba2_replayssm_spec(
    *,
    x: torch.Tensor,
    B: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    x_ring: torch.Tensor,
    b_ring: torch.Tensor,
    dt_ring: torch.Tensor,
    decay_ring: torch.Tensor,
    num_steps: int,
) -> None:
    """Record one layer's verify window of ``num_steps`` tokens per request.

    ``x`` [rows * num_steps, HV * V], ``B`` [rows * num_steps, G * K] and raw
    ``dt`` [rows * num_steps, HV] may be row-strided views; token ``i`` lands in
    ring row ``i // num_steps`` step ``i % num_steps``. Rings: ``x_ring``
    [R, T, HV, V], ``b_ring`` [R, T, G, K], ``dt_ring`` / ``decay_ring``
    [R, T, HV] fp32, with ``num_steps <= T`` and enough rows for every request.
    """
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return
    num_rows, ring_t, HV, V = x_ring.shape
    G, K = b_ring.shape[-2:]
    assert x.stride(-1) == 1 and B.stride(-1) == 1
    assert 0 < num_steps <= ring_t and num_tokens % num_steps == 0
    assert num_tokens // num_steps <= num_rows
    for ring in (x_ring, b_ring, dt_ring, decay_ring):
        assert ring.is_contiguous()
    _mamba2_replayssm_record_kernel[(num_tokens, HV)](
        x,
        B,
        dt,
        dt_bias,
        A,
        x_ring,
        b_ring,
        dt_ring,
        decay_ring,
        x.stride(0),
        B.stride(0),
        dt.stride(0),
        num_steps,
        RING_T=ring_t,
        HV=HV,
        V=V,
        G=G,
        K=K,
        HEADS_PER_GROUP=HV // G,
        BV=triton.next_power_of_2(V),
        BK=triton.next_power_of_2(K),
        num_warps=1,
    )


@triton.jit
def _mamba2_replayssm_fold_kernel(
    state_ptr,
    x_ring_ptr,
    b_ring_ptr,
    dt_ring_ptr,
    decay_ring_ptr,
    state_indices_ptr,
    last_step_ptr,
    src_rows_ptr,
    track_indices_ptr,
    track_steps_ptr,
    stride_state_layer,
    stride_state_slot,
    stride_state_head,
    stride_state_dim,
    stride_ring_x_layer,
    stride_ring_b_layer,
    stride_ring_dt_layer,
    T: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    HAS_SRC_ROWS: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    req = tl.program_id(1)
    pid_lh = tl.program_id(2)
    layer = (pid_lh // HV).to(tl.int64)
    h = pid_lh % HV
    g = h // HEADS_PER_GROUP

    slot = tl.load(state_indices_ptr + req).to(tl.int64)
    n_commit = tl.load(last_step_ptr + req).to(tl.int32) + 1
    if (slot < 0) | (n_commit <= 0):
        return
    if HAS_SRC_ROWS:
        row = tl.load(src_rows_ptr + req).to(tl.int64)
    else:
        row = req.to(tl.int64)
    if HAS_TRACK:
        track_slot = tl.load(track_indices_ptr + req).to(tl.int64)
        track_step = tl.load(track_steps_ptr + req).to(tl.int32)
    else:
        track_slot = -1
        track_step = -1

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < V
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]
    head_offs = (
        layer * stride_state_layer
        + h * stride_state_head
        + offs_m[:, None] * stride_state_dim
        + offs_k[None, :]
    )
    state_ptrs = state_ptr + slot * stride_state_slot + head_offs
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    tok0 = row * T
    x_base = x_ring_ptr + layer * stride_ring_x_layer + h * V + offs_m
    b_base = b_ring_ptr + layer * stride_ring_b_layer + g * K + offs_k
    dt_base = dt_ring_ptr + layer * stride_ring_dt_layer + h
    decay_base = decay_ring_ptr + layer * stride_ring_dt_layer + h
    for t in range(0, n_commit):
        tok = tok0 + t
        x = tl.load(x_base + tok * HV * V, mask=mask_m, other=0.0).to(tl.float32)
        B = tl.load(b_base + tok * G * K, mask=mask_k, other=0.0).to(tl.float32)
        dt = tl.load(dt_base + tok * HV)
        dA = tl.load(decay_base + tok * HV)
        dB = B * dt
        state = state * dA + dB * x[:, None]
        if HAS_TRACK:
            if (t == track_step) and (track_slot >= 0):
                tl.store(
                    state_ptr + track_slot * stride_state_slot + head_offs,
                    state.to(state_ptr.dtype.element_ty),
                    mask=mask,
                )

    tl.store(state_ptrs, state.to(state_ptr.dtype.element_ty), mask=mask)


def fold_mamba2_replayssm_spec(
    *,
    temporal: torch.Tensor,
    x_ring: torch.Tensor,
    b_ring: torch.Tensor,
    dt_ring: torch.Tensor,
    decay_ring: torch.Tensor,
    state_indices: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    src_rows: torch.Tensor | None = None,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
) -> None:
    """Replay each request's accepted steps ``0..last_step`` into ``temporal``.

    All layers in one launch. ``temporal`` [L, slots, HV, V, K]; rings
    [L, R, T, ...] as recorded by :func:`record_mamba2_replayssm_spec`. Request
    ``i`` reads ring row ``src_rows[i]`` (default ``i``); a negative slot or step
    skips it. With track indices, the state after ``mamba_steps_to_track[i]``
    is also written to that slot.
    """
    num_reqs = state_indices.shape[0]
    if num_reqs == 0 or temporal.numel() == 0:
        return
    num_layers, _, HV, V, K = temporal.shape
    T = x_ring.shape[2]
    G = b_ring.shape[-2]
    assert temporal.stride(-1) == 1
    has_track = mamba_track_indices is not None
    if has_track:
        assert mamba_steps_to_track is not None
    BM = min(16, triton.next_power_of_2(V))
    grid = (triton.cdiv(V, BM), num_reqs, HV * num_layers)
    _mamba2_replayssm_fold_kernel[grid](
        temporal,
        x_ring,
        b_ring,
        dt_ring,
        decay_ring,
        state_indices,
        last_correct_step_indices,
        src_rows if src_rows is not None else state_indices,
        mamba_track_indices if has_track else state_indices,
        mamba_steps_to_track if has_track else state_indices,
        temporal.stride(0),
        temporal.stride(1),
        temporal.stride(2),
        temporal.stride(3),
        x_ring.stride(0),
        b_ring.stride(0),
        dt_ring.stride(0),
        T=T,
        HV=HV,
        V=V,
        G=G,
        K=K,
        HEADS_PER_GROUP=HV // G,
        BM=BM,
        BK=triton.next_power_of_2(K),
        HAS_SRC_ROWS=src_rows is not None,
        HAS_TRACK=has_track,
        num_warps=2,
    )


def commit_mamba2_replayssm_after_verify(
    *,
    spec_state,
    state_batch_indices: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    src_indices_raw: torch.Tensor | None = None,
) -> None:
    """Fold the accepted SSM steps and roll back conv from the verify windows."""
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    track_steps = mamba_steps_to_track if mamba_track_indices is not None else None
    fold_mamba2_replayssm_spec(
        temporal=spec_state.temporal,
        x_ring=spec_state.replayssm_rawv,
        b_ring=spec_state.replayssm_rawk,
        dt_ring=spec_state.replayssm_g,
        decay_ring=spec_state.replayssm_decay,
        state_indices=state_batch_indices,
        last_correct_step_indices=last_correct_step_indices,
        src_rows=src_indices_raw,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=track_steps,
    )
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
        if mamba_track_indices is not None:
            fused_conv_window_scatter_with_mask(
                conv_states,
                interm_conv,
                mamba_track_indices,
                track_steps,
                src_indices_raw,
            )
