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
        mixed_qkv,
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
    core_attn_out, last_recurrent_state = (
        torch.ops.sgl_kernel.chunk_gated_delta_rule_cpu(
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
    )
    h = None  # Todo: add return h support
    return core_attn_out, last_recurrent_state, h


def chunk_kda_cpu(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    initial_state_indices=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    output_intermediate_states=False,
    track_state=None,
    track_chunk_idx=None,
    beta_is_raw=False,
    **kwargs,
):
    output, _final_state, h = torch.ops.sgl_kernel.chunk_kda_cpu(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        initial_state_indices,
        A_log,
        dt_bias,
        lower_bound,
        beta_is_raw,
        use_qk_l2norm_in_kernel,
        output_intermediate_states,
        track_state,
        track_chunk_idx,
        scale=scale,
    )
    if output_intermediate_states:
        return output, h
    return output
