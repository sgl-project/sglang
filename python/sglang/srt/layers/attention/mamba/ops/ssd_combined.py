# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_combined.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py

# ruff: noqa: E501

import torch
from einops import rearrange

from .ssd_bmm import _bmm_chunk_fwd
from .ssd_chunk_scan import _chunk_scan_fwd
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from .ssd_state_passing import _state_passing_fwd


def is_int_pow_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    return_intermediate_states=False,
    seq_idx=None,
    cu_seqlens=None,
    cu_chunk_seqlens=None,
    last_chunk_indices=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    assert out.shape == x.shape
    assert cu_seqlens is not None, "cu_seqlens must be provided for varlen input"
    assert cu_chunk_seqlens is not None, "cu_chunk_seqlens must be provided"
    assert last_chunk_indices is not None, "last_chunk_indices must be provided"
    assert seq_idx is not None, "seq_idx must be provided"
    assert seq_idx.shape == (cu_chunk_seqlens.shape[0] - 1,)
    assert last_chunk_indices.shape == (cu_seqlens.shape[0] - 1,)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if initial_states is not None:
        assert initial_states.shape == (len(cu_seqlens) - 1, nheads, headdim, dstate)

    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(0) != 1:
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(0) != 1:
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()

    # 1. Compute chunked cumsum of A * dt.
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        cu_chunk_seqlens,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )

    # 2. Compute the state for each intra-chunk.
    states = _chunk_state_fwd(
        B,
        x,
        dt,
        dA_cumsum,
        cu_chunk_seqlens,
        states_in_fp32=True,
    )

    # 3. Compute the inter-chunk SSM recurrence. Returns start-of-chunk states
    # (init_state or end-of-previous-chunk) in `states`, and per-sequence
    # final states for the SSM cache.
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum,
        last_chunk_indices,
        initial_states=(
            rearrange(initial_states, "... p n -> ... (p n)")
            if initial_states is not None
            else None
        ),
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
    )
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    final_states = rearrange(final_states, "... (p n) -> ... p n", n=dstate)

    # 4. Compute batched matrix multiply for C_j^T B_i terms.
    CB = _bmm_chunk_fwd(
        C,
        B,
        chunk_size,
        cu_chunk_seqlens,
        output_dtype=torch.float32,
    )

    # 5. Scan and compute diagonal blocks. chunk_scan reads
    # `states[pid_c]` directly as prev_state (start-of-chunk state) — no
    # init_state / seq_idx branching needed inside the kernel.
    _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        cu_chunk_seqlens,
        out,
        D=D,
        z=z,
    )

    if return_intermediate_states:
        return states, final_states
    return final_states


def mamba_chunk_scan_combined(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    cu_seqlens,
    cu_chunk_seqlens,
    last_chunk_indices,
    seq_idx,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_intermediate_states=False,
    state_dtype=None,
):
    """
    Argument:
        x: (seqlen, nheads, headdim)
        dt: (seqlen, nheads)
        A: (nheads,)
        B: (seqlen, ngroups, dstate)
        C: (seqlen, ngroups, dstate)
        chunk_size: int
        cu_seqlens: (num_sequences + 1,)
        cu_chunk_seqlens: (nchunks + 1,)
        last_chunk_indices: (num_sequences,)
        seq_idx: (nchunks,)
        out: (seqlen, nheads, headdim) preallocated output tensor
        D: (nheads, headdim) or (nheads,)
        z: (seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (num_sequences, nheads, headdim, dstate)
        dt_softplus: Whether to apply softplus to dt
        state_dtype: The data type of the SSM state
    Return:
        final_states: (num_sequences, nheads, headdim, dstate), or
        (start-of-chunk states, final_states) when return_intermediate_states=True.
    """
    return _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        out,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        return_intermediate_states=return_intermediate_states,
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        state_dtype=state_dtype,
    )
