from typing import Optional

import torch


# mamba
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
):
    torch.ops.sgl_kernel.causal_conv1d_fwd(
        x,
        weight,
        bias_,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor],
    conv_state_indices: Optional[torch.Tensor],
    pad_slot_id: int,
):
    torch.ops.sgl_kernel.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias_,
        silu_activation,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )


def causal_conv1d_fn_cpu(
    mixed_qkv_transposed,
    conv_weights,
    bias,
    activation,
    conv_states,
    has_initial_state,
    cache_indices,
    query_start_loc,
    seq_lens_cpu,
):
    return torch.ops.sgl_kernel.causal_conv1d_fwd_cpu(
        mixed_qkv_transposed,
        conv_weights,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation == "silu",
        -1,
        True,
    )


def causal_conv1d_update_cpu(
    mixed_qkv, conv_states, conv_weights, bias, activation, conv_state_indices
):
    return torch.ops.sgl_kernel.causal_conv1d_update_cpu(
        mixed_qkv.contiguous(),
        conv_states,
        conv_weights,
        bias,
        activation == "silu",
        None,
        conv_state_indices,
        -1,
        True,
    )


def chunk_gated_delta_rule_cpu(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    head_first,
    use_qk_l2norm_in_kernel,
    initial_state_indices,
):
    core_attn_out = torch.ops.sgl_kernel.chunk_gated_delta_rule_cpu(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        True,  # output_final_state
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel,
        initial_state_indices,
    )
    last_recurrent_state = initial_state
    h = None  # Todo: add return h support
    return core_attn_out, last_recurrent_state, h


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
    total_tokens = x.shape[1]
    output = torch.empty_like(x) if out is None else out
    final_by_sequence = []
    result_dtype = state_dtype or C.dtype
    cpu_scan = torch.ops.sgl_kernel.mamba_chunk_scan_combined_cpu

    def initial_state_for(seq_idx):
        if initial_states is not None:
            return initial_states[seq_idx].to(result_dtype)
        return torch.zeros(nheads, headdim, dstate, dtype=result_dtype, device=x.device)

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

    if final_by_sequence:
        varlen_states = torch.cat(final_by_sequence, dim=0)
        final_states = final_by_sequence[-1]
    else:
        final_states = torch.zeros(
            batch, nheads, headdim, dstate, dtype=result_dtype, device=x.device
        )
        varlen_states = final_states.new_empty((0, nheads, headdim, dstate))

    if return_intermediate_states or return_track_states:
        num_global_chunks = (total_tokens + chunk_size - 1) // chunk_size
        chunk_states = []
        seq_idx = 0
        for chunk_idx in range(num_global_chunks):
            boundary = chunk_idx * chunk_size
            while seq_idx + 1 < len(offsets) - 1 and boundary >= offsets[seq_idx + 1]:
                seq_idx += 1
            start = offsets[seq_idx]
            if boundary == start:
                chunk_states.append(initial_state_for(seq_idx))
            else:
                chunk_states.append(run_segment(seq_idx, start, boundary)[0])
        intermediate_states = (
            torch.stack(chunk_states).unsqueeze(0)
            if chunk_states
            else torch.empty(
                (1, 0, nheads, headdim, dstate), dtype=result_dtype, device=x.device
            )
        )
    else:
        intermediate_states = None

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
    if x.device.type != "cpu":
        raise ValueError("sgl_kernel.mamba.mamba_chunk_scan_combined only supports CPU")

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


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    pad_slot_id=-1,
    out=None,
    disable_state_update=False,
    intermediate_states_buffer=None,
    cache_steps=None,
    retrieve_parent_token=None,
    intermediate_state_indices=None,
    enable_stochastic_rounding=False,
    cache_philox_rounds=0,
):
    if cache_philox_rounds < 0:
        raise ValueError("cache_philox_rounds must be non-negative.")
    if enable_stochastic_rounding and state.dtype != torch.float16:
        raise ValueError(
            "Stochastic rounding for the Mamba SSM cache requires state dtype "
            f"torch.float16, got {state.dtype}."
        )

    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if dt.dim() == 3:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None:
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if z.dim() == 3:
            z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out is None:
        out = torch.empty_like(x)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if out.dim() == 3:
        out = out.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch, T, _, _ = x.shape

    assert x.shape == (batch, T, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, T, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    assert out.shape == x.shape

    if x.device.type != "cpu":
        raise ValueError("sgl_kernel.mamba.selective_state_update only supports CPU")
    if (
        intermediate_states_buffer is not None
        or retrieve_parent_token is not None
        or intermediate_state_indices is not None
        or enable_stochastic_rounding
        or cache_steps is not None
    ):
        raise NotImplementedError(
            "The native CPU selective state update does not support speculative-state caching"
        )

    import sgl_kernel  # noqa: F401

    torch.ops.sgl_kernel.selective_state_update_cpu(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        disable_state_update,
        out,
    )
