# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM speculative verify for Mamba2 SSD mixers (linear draft chain).

FlashInfer's checkpointing SSU computes the verify outputs from the committed
state and appends each draft token's post-conv x, B and processed dt to a ring
on the request's mamba slot; the commit replays the accepted prefix into the
state with ``replayssm_materialize``. Every commit folds, so the ring is empty
before each verify and ``ring_start`` / ``prev_num_accepted_tokens`` stay zero.

Coverage: CUDA with FlashInfer Mamba, BF16 activations, ``dt``/``A``/
``dt_bias``/``D`` tied across head_dim, at most 8 draft tokens (the ring holds
two windows and FlashInfer caps the replay window at 16).
"""

from __future__ import annotations

import torch

MAX_DRAFT_TOKENS = 8
_pointer_tables: dict[tuple, tuple] = {}


def mamba2_replayssm_ring_len(num_draft_tokens: int) -> int:
    if not 0 < num_draft_tokens <= MAX_DRAFT_TOKENS:
        raise ValueError(
            "Mamba2 ReplaySSM supports 1 to "
            f"{MAX_DRAFT_TOKENS} draft tokens, got {num_draft_tokens}."
        )
    return 2 * num_draft_tokens


def verify_mamba2_replayssm_spec(
    *,
    state: torch.Tensor,
    x_ring: torch.Tensor,
    b_ring: torch.Tensor,
    dt_ring: torch.Tensor,
    cursor: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    state_batch_indices: torch.Tensor,
) -> None:
    """One layer's verify: ``x`` [reqs, T, HV, V], rings of at least ``2 * T`` rows.

    ``cursor`` is a zero int32 [slots] tensor: the ring start and the count of
    pending accepted rows, both always zero under fold-every-commit.
    """
    from flashinfer.mamba import checkpointing_ssu

    assert x.dtype == torch.bfloat16, x.dtype
    assert 2 * x.shape[1] <= x_ring.shape[2], (x.shape, x_ring.shape)
    checkpointing_ssu(
        state,
        x_ring,
        b_ring,
        dt_ring,
        cursor,
        cursor,
        x,
        dt,
        A,
        B,
        C,
        out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
        pad_slot_id=-1,
    )


def _layer_table(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    step = t.stride(0) * t.element_size()
    ptrs = [t.data_ptr() + i * step for i in range(t.shape[0])]
    return (
        torch.tensor(ptrs, dtype=torch.int64, device=t.device),
        torch.full((t.shape[0],), t.stride(1), dtype=torch.int64, device=t.device),
    )


def _pointer_tables_for(
    temporal: torch.Tensor,
    x_ring: torch.Tensor,
    b_ring: torch.Tensor,
    dt_ring: torch.Tensor,
    A_by_layer: list[torch.Tensor],
) -> tuple:
    key = tuple(
        (t.data_ptr(), t.dtype, tuple(t.shape), t.stride())
        for t in (temporal, x_ring, b_ring, dt_ring)
    ) + tuple(a.data_ptr() for a in A_by_layer)
    tables = _pointer_tables.get(key)
    if tables is None:
        a_ptrs = torch.tensor(
            [a.data_ptr() for a in A_by_layer],
            dtype=torch.int64,
            device=temporal.device,
        )
        no_scale = torch.zeros_like(a_ptrs)
        tables = (
            *_layer_table(temporal),
            *_layer_table(x_ring),
            *_layer_table(b_ring),
            *_layer_table(dt_ring),
            a_ptrs,
            no_scale,
            no_scale,
        )
        _pointer_tables[key] = tables
    return tables


def fold_mamba2_replayssm_spec(
    *,
    temporal: torch.Tensor,
    x_ring: torch.Tensor,
    b_ring: torch.Tensor,
    dt_ring: torch.Tensor,
    A_by_layer: list[torch.Tensor],
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    replay_len: torch.Tensor,
) -> None:
    """Write ``src`` advanced by its first ``replay_len`` ring rows into ``dst``.

    All layers in one launch. ``temporal`` [L, slots, HV, V, K]; rings
    [L, slots, HV, R, V], [L, slots, G, R, K] and [L, slots, HV, R];
    ``A_by_layer`` holds each layer's fp32 [HV] ``A``. A slot of -1 skips the
    request; ``src`` and ``dst`` may be the same slot.
    """
    from flashinfer.mamba import replayssm_materialize

    num_reqs = src_slots.shape[0]
    if num_reqs == 0:
        return
    num_layers, _, num_heads, dim, dstate = temporal.shape
    ring_len = x_ring.shape[3]
    assert len(A_by_layer) == num_layers
    assert all(a.dtype == torch.float32 and a.is_contiguous() for a in A_by_layer)
    device = temporal.device
    replayssm_materialize(
        *_pointer_tables_for(temporal, x_ring, b_ring, dt_ring, A_by_layer),
        src_slots.to(torch.int32).expand(num_layers, num_reqs).contiguous(),
        dst_slots.to(torch.int32).expand(num_layers, num_reqs).contiguous(),
        torch.zeros(num_reqs, dtype=torch.int32, device=device),
        replay_len.to(torch.int32).clamp(0, ring_len // 2),
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        state_dtype=temporal.dtype,
        input_dtype=x_ring.dtype,
        matrixA_dtype=torch.float32,
        dim=dim,
        dstate=dstate,
        num_heads=num_heads,
        heads_per_group=num_heads // b_ring.shape[2],
        max_window=ring_len // 2,
        ring_buffer_len=ring_len,
        dependency_inputs=[x_ring, b_ring, dt_ring],
        dependency_outputs=[temporal],
    )


def commit_mamba2_replayssm_after_verify(
    *,
    spec_state,
    A_by_layer: list[torch.Tensor],
    state_batch_indices: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
) -> None:
    """Fold the accepted SSM steps and roll back conv from the verify windows.

    A track slot receives the state after ``mamba_steps_to_track`` when that
    step was accepted; it is replayed from the pre-commit working state.
    """
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    working = torch.where(
        last_correct_step_indices >= 0, state_batch_indices.to(torch.int32), -1
    )
    rings = dict(
        temporal=spec_state.temporal,
        x_ring=spec_state.replayssm_rawv,
        b_ring=spec_state.replayssm_rawk,
        dt_ring=spec_state.replayssm_g,
        A_by_layer=A_by_layer,
    )
    if mamba_track_indices is not None:
        tracked = (mamba_steps_to_track >= 0) & (
            mamba_steps_to_track <= last_correct_step_indices
        )
        fold_mamba2_replayssm_spec(
            **rings,
            src_slots=working,
            dst_slots=torch.where(tracked, mamba_track_indices.to(torch.int32), -1),
            replay_len=mamba_steps_to_track + 1,
        )
    fold_mamba2_replayssm_spec(
        **rings,
        src_slots=working,
        dst_slots=working,
        replay_len=last_correct_step_indices + 1,
    )
    for conv_states, interm_conv in zip(
        spec_state.conv, spec_state.intermediate_conv_window
    ):
        fused_conv_window_scatter_with_mask(
            conv_states,
            interm_conv,
            state_batch_indices,
            last_correct_step_indices,
            None,
        )
        if mamba_track_indices is not None:
            fused_conv_window_scatter_with_mask(
                conv_states,
                interm_conv,
                mamba_track_indices,
                mamba_steps_to_track,
                None,
            )
