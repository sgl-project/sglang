# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_combined.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py

# ruff: noqa: E501

import torch
import torch.nn.functional as F
import triton
from einops import rearrange
from packaging import version

from .ssd_bmm import _bmm_chunk_fwd
from .ssd_chunk_scan import _chunk_scan_fwd
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd, chunk_state_varlen
from .ssd_state_passing import _state_passing_fwd

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def is_int_pow_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def _track_states_at(
    B,
    x,
    dt,
    dA_cumsum,
    states,
    cu_seqlens,
    initial_states,
    track_seq_idx,
    track_end_locs,
):
    starts = cu_seqlens[track_seq_idx]
    ends = track_end_locs.to(device=starts.device, dtype=starts.dtype)
    bounds = torch.stack([starts, ends], dim=1).flatten().contiguous()
    if initial_states is not None:
        initial_states = initial_states[track_seq_idx].repeat_interleave(2, dim=0)[:-1]
    return chunk_state_varlen(
        B.squeeze(0),
        x.squeeze(0),
        dt.squeeze(0),
        dA_cumsum.squeeze(0),
        bounds,
        states.squeeze(0),
        initial_states=initial_states,
    )[::2].unsqueeze(0)


def _mamba_chunk_scan_cpu_reference(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D,
    z,
    dt_bias,
    initial_states,
    seq_idx,
    cu_seqlens,
    dt_softplus,
    dt_limit,
    out,
    return_final_states,
    return_varlen_states,
    return_intermediate_states,
    state_dtype,
    return_track_states,
    track_seq_idx,
    track_end_locs,
):
    batch, seqlen, nheads, _ = x.shape
    ngroups = B.shape[2]
    groups_per_head = nheads // ngroups
    b_heads = B.float().repeat_interleave(groups_per_head, dim=2)
    c_heads = C.repeat_interleave(groups_per_head, dim=2)
    output = torch.empty_like(x) if out is None else out
    result_dtype = state_dtype or C.dtype
    final_by_batch = []
    varlen_states = []
    intermediate_by_batch = [[] for _ in range(batch)]

    for batch_idx in range(batch):
        if cu_seqlens is not None:
            assert batch == 1, "CPU varlen Mamba scan requires batch=1"
            offsets = cu_seqlens.tolist()
            spans = [
                (int(offsets[i]), int(offsets[i + 1])) for i in range(len(offsets) - 1)
            ]
        elif seq_idx is not None:
            ids = seq_idx[batch_idx].tolist()
            starts = [0] + [i for i in range(1, seqlen) if ids[i] != ids[i - 1]]
            spans = list(zip(starts, starts[1:] + [seqlen]))
        else:
            spans = [(0, seqlen)]

        last_state = torch.zeros(
            nheads, x.shape[-1], B.shape[-1], device=x.device, dtype=torch.float32
        )
        for sequence_idx, (start, end) in enumerate(spans):
            if initial_states is None:
                state = torch.zeros_like(last_state)
            else:
                init_idx = sequence_idx if cu_seqlens is not None else batch_idx
                state = initial_states[init_idx].float().clone()
            for chunk_start in range(start, end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end)
                for token in range(chunk_start, chunk_end):
                    delta = dt[batch_idx, token].float()
                    if dt_bias is not None:
                        delta = delta + dt_bias.float()
                    if dt_softplus:
                        delta = F.softplus(delta)
                    delta = delta.clamp(*dt_limit)
                    x_token = x[batch_idx, token].float()
                    b_token = b_heads[batch_idx, token].float()
                    c_token = c_heads[batch_idx, token]
                    state = (
                        state
                        * torch.exp(delta[:, None, None] * A.float()[:, None, None])
                        + delta[:, None, None]
                        * b_token[:, None, :]
                        * x_token[:, :, None]
                    )
                    value = (state.to(C.dtype) * c_token[:, None, :]).sum(dim=-1)
                    if D is not None:
                        d_value = D.float()[:, None] if D.dim() == 1 else D.float()
                        value = value + (x_token * d_value).to(value.dtype)
                    if z is not None:
                        value = value * F.silu(z[batch_idx, token])
                    output[batch_idx, token].copy_(value.to(x.dtype))
                intermediate_by_batch[batch_idx].append(state.to(result_dtype))
            last_state = state
            varlen_states.append(state.to(result_dtype))
        final_by_batch.append(last_state.to(result_dtype))

    final_states = torch.stack(final_by_batch)
    varlen_result = torch.stack(varlen_states) if cu_seqlens is not None else None
    intermediate_states = None
    if return_intermediate_states or return_track_states:
        if batch == 1:
            intermediate_states = torch.stack(intermediate_by_batch[0]).unsqueeze(0)
        else:
            chunk_count = max(map(len, intermediate_by_batch))
            padded = []
            for batch_states in intermediate_by_batch:
                if len(batch_states) < chunk_count:
                    batch_states += [batch_states[-1]] * (
                        chunk_count - len(batch_states)
                    )
                padded.append(torch.stack(batch_states))
            intermediate_states = torch.stack(padded)

    track_states = None
    if return_track_states:
        assert cu_seqlens is not None, "return_track_states requires cu_seqlens"
        if (
            track_seq_idx is None
            or track_end_locs is None
            or track_end_locs.numel() == 0
        ):
            track_states = torch.empty(
                (0, nheads, x.shape[-1], B.shape[-1]),
                device=x.device,
                dtype=result_dtype,
            )
        else:
            offsets = cu_seqlens.tolist()
            tracked = []
            for seq, end in zip(track_seq_idx.tolist(), track_end_locs.tolist()):
                seq = int(seq)
                start = int(offsets[seq])
                end = int(end)
                if initial_states is None:
                    state = torch.zeros_like(last_state)
                else:
                    state = initial_states[seq].float().clone()
                for token in range(start, end):
                    delta = dt[0, token].float()
                    if dt_bias is not None:
                        delta = delta + dt_bias.float()
                    if dt_softplus:
                        delta = F.softplus(delta)
                    delta = delta.clamp(*dt_limit)
                    state = (
                        state
                        * torch.exp(delta[:, None, None] * A.float()[:, None, None])
                        + delta[:, None, None]
                        * b_heads[0, token, :, None, :].float()
                        * x[0, token].float()[:, :, None]
                    )
                tracked.append(state.to(result_dtype))
            track_states = torch.stack(tracked)

    if return_track_states:
        return intermediate_states, varlen_result, track_states
    if return_intermediate_states:
        if return_varlen_states:
            return (
                (intermediate_states, final_states, varlen_result)
                if return_final_states
                else (
                    intermediate_states,
                    varlen_result,
                )
            )
        return (
            (intermediate_states, final_states)
            if return_final_states
            else intermediate_states
        )
    if return_varlen_states:
        return (final_states, varlen_result) if return_final_states else varlen_result
    return final_states if return_final_states else None


def _mamba_chunk_scan_cpu_varlen(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D,
    z,
    dt_bias,
    initial_states,
    cu_seqlens,
    dt_softplus,
    dt_limit,
    out,
    return_final_states,
    return_intermediate_states,
    state_dtype,
    return_track_states,
    track_seq_idx,
    track_end_locs,
):
    import sgl_kernel  # noqa: F401

    batch, _, nheads, headdim = x.shape
    dstate = B.shape[-1]
    offsets = [int(value) for value in cu_seqlens.tolist()]
    output = torch.empty_like(x) if out is None else out
    final_by_sequence = []
    chunk_states = []
    result_dtype = state_dtype or C.dtype
    cpu_scan = torch.ops.sgl_kernel.mamba_chunk_scan_combined_cpu

    def run_segment(seq_idx, start, end, segment_out=None):
        init = (
            initial_states[seq_idx : seq_idx + 1]
            if initial_states is not None
            else None
        )
        return cpu_scan(
            x[:, start:end],
            dt[:, start:end],
            A,
            B[:, start:end],
            C[:, start:end],
            chunk_size,
            D,
            z[:, start:end] if z is not None else None,
            dt_bias,
            init,
            dt_softplus,
            dt_limit[0],
            dt_limit[1],
            state_dtype,
            segment_out,
        )[1]

    for seq_idx, (start, end) in enumerate(zip(offsets, offsets[1:])):
        segment_out = output[:, start:end]
        final_state = run_segment(seq_idx, start, end, segment_out)
        final_by_sequence.append(final_state)
        if return_intermediate_states or return_track_states:
            for chunk_end in range(start + chunk_size, end + chunk_size, chunk_size):
                chunk_end = min(chunk_end, end)
                if chunk_end > start:
                    chunk_states.append(run_segment(seq_idx, start, chunk_end))

    if final_by_sequence:
        varlen_states = torch.cat(final_by_sequence, dim=0)
        final_states = final_by_sequence[-1]
    else:
        final_states = torch.zeros(
            batch, nheads, headdim, dstate, dtype=result_dtype, device=x.device
        )
        varlen_states = final_states.new_empty((0, nheads, headdim, dstate))

    if chunk_states:
        intermediate_states = torch.stack(chunk_states).unsqueeze(0)
    else:
        intermediate_states = torch.empty(
            (1, 0, nheads, headdim, dstate), dtype=result_dtype, device=x.device
        )

    if return_track_states:
        if (
            track_seq_idx is None
            or track_end_locs is None
            or track_end_locs.numel() == 0
        ):
            track_states = torch.empty(
                (0, nheads, headdim, dstate), dtype=result_dtype, device=x.device
            )
        else:
            tracked = []
            for seq_idx, track_end in zip(
                track_seq_idx.tolist(), track_end_locs.tolist()
            ):
                start = offsets[int(seq_idx)]
                tracked.append(run_segment(int(seq_idx), start, int(track_end))[0])
            track_states = torch.stack(tracked)
        return intermediate_states, varlen_states, track_states

    if return_intermediate_states:
        if return_final_states:
            return intermediate_states, final_states, varlen_states
        return intermediate_states, varlen_states
    if return_final_states:
        return final_states, varlen_states
    return varlen_states


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
    out=None,
    track_seq_idx=None,
    track_end_locs=None,
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if (
        x.stride(-1) != 1 and x.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        if cu_seqlens is None:
            assert initial_states.shape == (batch, nheads, headdim, dstate)
        else:
            assert initial_states.shape == (
                len(cu_seqlens) - 1,
                nheads,
                headdim,
                dstate,
            )

    # This function executes 5 sub-functions for computing mamba
    # - a good resource is the blog https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   which has a minimal implementation to understand the below operations
    # - as explained by the blog, mamba is a special case of causal attention
    # - the idea is to chunk the attention matrix and compute each
    #   submatrix separately using different optimizations.
    # - see the blog and paper for a visualization of the submatrices
    #   which we refer to in the comments below

    # 1. Compute chunked cumsum of A * dt
    # - here dt may go through a softplus activation
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit
    )

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # - for handling chunked prefill, this requires i) initial_states
    #   ii) seq_idx iii) is_cont_batched and (iv) chunk_offsets to be all specified.
    # - When a new seq_idx is detected, we will stop passing the prev_state
    #   and switch accordingly to the init_state corresponding to the new seq_idx.
    # - We will also make sure that the dA_cumsum is taken only from the start of the
    #   sequence (hence we need the full dA_cumsum tensor and not just the values at chunk boundaries)
    # - this will ensure that states will be updated with the rightmost flushed seq_idx
    #   of the previous chunk. This implies that the first chunk of states is either 0
    #   or equal to init_states of the first example.
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum,
        initial_states=(
            rearrange(initial_states, "... p n -> ... (p n)")
            if initial_states is not None
            else None
        ),
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
        is_cont_batched=cu_seqlens is not None,
        chunk_offsets=chunk_offsets,
    )
    states, final_states = (
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]
    )

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)

    # 5. Scan and compute the diagonal blocks, taking into
    #    account past causal states.
    # - if initial states are provided, then states information will be
    #   augmented with initial_states.
    # - to do this properly, we need to account for example changes in
    #   the continuous batch, therefore we introduce pseudo chunks, which is
    #   a chunk that is split up each time an example changes.
    # - in each (pseudo) chunk, we detect if the previous (pseudo) chunk had
    #   a seq_idx change, in which case we take states information from
    #   init_states.
    out_x = _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        D=D,
        z=z,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        initial_states=initial_states,
        out=out,
    )
    if cu_seqlens is None:
        return out_x, dt, dA_cumsum, states, final_states
    else:
        assert batch == 1, (
            "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        )
        varlen_states = chunk_state_varlen(
            B.squeeze(0),
            x.squeeze(0),
            dt.squeeze(0),
            dA_cumsum.squeeze(0),
            cu_seqlens,
            states.squeeze(0),
            initial_states=initial_states,
        )
        track_states = None
        if (
            track_seq_idx is not None
            and track_end_locs is not None
            and track_end_locs.numel() > 0
        ):
            track_states = _track_states_at(
                B,
                x,
                dt,
                dA_cumsum,
                states,
                cu_seqlens,
                initial_states,
                track_seq_idx,
                track_end_locs,
            )
        return out_x, dt, dA_cumsum, states, final_states, varlen_states, track_states


def mamba_chunk_scan_combined(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    out=None,
    return_final_states=False,
    return_varlen_states=False,
    return_intermediate_states=False,
    state_dtype=None,
    return_track_states=False,
    track_seq_idx=None,
    track_end_locs=None,
):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
        out: Preallocated output tensor
        state_dtype: The data type of the ssm state
    """

    if x.device.type == "cpu":
        use_native = x.dtype in (torch.bfloat16, torch.float16)
        if return_varlen_states:
            assert cu_seqlens is not None, (
                "cu_seqlens must be provided if return_varlen_states is True"
            )
        if use_native and return_varlen_states:
            assert x.shape[0] == 1, "CPU varlen Mamba scan requires batch=1"
            return _mamba_chunk_scan_cpu_varlen(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                D,
                z,
                dt_bias,
                initial_states,
                cu_seqlens,
                dt_softplus,
                dt_limit,
                out,
                return_final_states,
                return_intermediate_states,
                state_dtype,
                return_track_states,
                track_seq_idx,
                track_end_locs,
            )
        if (
            not use_native
            or seq_idx is not None
            or chunk_indices is not None
            or chunk_offsets is not None
            or return_varlen_states
            or return_intermediate_states
            or return_track_states
        ):
            if return_track_states:
                assert return_varlen_states, (
                    "return_track_states requires return_varlen_states (cu_seqlens mode)"
                )
            return _mamba_chunk_scan_cpu_reference(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                D,
                z,
                dt_bias,
                initial_states,
                seq_idx,
                cu_seqlens if return_varlen_states else None,
                dt_softplus,
                dt_limit,
                out,
                return_final_states,
                return_varlen_states,
                return_intermediate_states,
                state_dtype,
                return_track_states,
                track_seq_idx,
                track_end_locs,
            )
        import sgl_kernel  # noqa: F401

        _, final_states = torch.ops.sgl_kernel.mamba_chunk_scan_combined_cpu(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D,
            z,
            dt_bias,
            initial_states,
            dt_softplus,
            dt_limit[0],
            dt_limit[1],
            state_dtype,
            out,
        )
        return final_states if return_final_states else None

    if not return_varlen_states:
        cu_seqlens = None
    else:
        assert cu_seqlens is not None, (
            "cu_seqlens must be provided if return_varlen_states is True"
        )
    out_x, dt_out, dA_cumsum, states, final_states, *rest = (
        _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            out=out,
            state_dtype=state_dtype,
            track_seq_idx=track_seq_idx,
            track_end_locs=track_end_locs,
        )
    )
    if return_track_states:
        assert return_varlen_states, (
            "return_track_states requires return_varlen_states (cu_seqlens mode)"
        )
        # `states` rides along: a tracked position on the grid is read from it
        # directly, which stays bit-exact against a recompute-free run.
        return states, rest[0], rest[1]
    if return_intermediate_states:
        if return_varlen_states:
            varlen_states = rest[0]
            if return_final_states:
                return states, final_states, varlen_states
            else:
                return states, varlen_states
        else:
            if return_final_states:
                return states, final_states
            else:
                return states

    if not return_varlen_states:
        if not return_final_states:
            return
        else:
            return final_states
    else:
        varlen_states = rest[0]
        return (
            (varlen_states)
            if not return_final_states
            else (final_states, varlen_states)
        )
