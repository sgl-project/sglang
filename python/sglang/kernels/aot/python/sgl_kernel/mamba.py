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


def inkling2_short_convolution_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    safe_idx: torch.Tensor,
    cache_mask: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    dynamic_weights: torch.Tensor | None = None,
    decode: bool = False,
    use_residual: bool = True,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.inkling2_short_convolution_cpu(
        x,
        weight,
        cache,
        safe_idx,
        cache_mask,
        cu,
        si,
        dynamic_weights,
        decode,
        use_residual,
        track_mask,
        track_indices,
        window,
    )


def inkling2_routed_sconv_epilogue_cpu(
    down: torch.Tensor,
    src2dst: torch.Tensor | None,
    top_k: int,
    weight: torch.Tensor,
    cache: torch.Tensor,
    safe_idx: torch.Tensor,
    cache_mask: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    attention: torch.Tensor,
    partial: torch.Tensor | None = None,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
    window: torch.Tensor | None = None,
    round_bf16: bool = False,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.inkling2_routed_sconv_epilogue_cpu(
        down,
        src2dst,
        top_k,
        weight,
        cache,
        safe_idx,
        cache_mask,
        cu,
        si,
        attention,
        partial,
        track_mask,
        track_indices,
        window,
        round_bf16,
    )


def inkling2_update_sconv_cache_cpu(
    x: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.inkling2_update_sconv_cache_cpu(
        x,
        sconv_cache,
        cache_indices,
        has_initial_state,
        query_start_loc,
    )
