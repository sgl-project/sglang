import torch


# mamba
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
        )
    )
    h = None  # Todo: add return h support
    return core_attn_out, last_recurrent_state, h
