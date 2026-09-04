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


def _causal_conv1d_update_cpu_verify(
    mixed_qkv,
    conv_states,
    conv_weights,
    bias,
    activation,
    conv_state_indices,
    intermediate_conv_window,
    intermediate_state_indices,
    retrieve_parent_token,
):
    """Target-verify conv over a linear draft chain.

    With DFlash's chain metadata (``retrieve_next_token[t] == t + 1``, no
    siblings) the Triton kernel's parent walk degenerates to a rolling window,
    so the block is just the decode kernel replayed token by token. The C++
    kernel takes a VNNI-prepacked weight whose layout torch cannot convolve
    with directly, so it has to do the arithmetic.
    """
    seq_len = mixed_qkv.size(-1)
    width = conv_weights.size(-1)
    cache_idx = conv_state_indices.to(torch.int64)
    window_idx = intermediate_state_indices.to(torch.int64)

    # The decode kernel appends the raw input to the state, so the state after
    # token t is just the width-1 window of [initial_state, block] ending at t.
    # Batching them avoids one index_put_ per token, which costs ~12us
    # regardless of how few bytes it moves.
    extended = torch.cat([conv_states[cache_idx], mixed_qkv], dim=-1)
    intermediate_conv_window[window_idx, :seq_len] = extended.unfold(-1, width - 1, 1)[
        :, :, 1:
    ].permute(0, 2, 1, 3)

    # Roll a scratch copy so the real pool only sees the final state.
    scratch = conv_states[cache_idx].contiguous()
    step_state_indices = torch.arange(
        cache_idx.numel(), dtype=torch.int32, device=mixed_qkv.device
    )

    result = torch.empty_like(mixed_qkv)
    for t in range(seq_len):
        result[:, :, t] = torch.ops.sgl_kernel.causal_conv1d_update_cpu(
            mixed_qkv[:, :, t].contiguous(),
            scratch,
            conv_weights,
            bias,
            activation in ("silu", "swish"),
            None,
            step_state_indices,
            -1,
            True,
        )

    conv_states[cache_idx] = scratch

    if retrieve_parent_token is not None:
        # The Triton kernel fuses this; downstream GDN verify reads it.
        retrieve_parent_token[:, 0] = 0
        retrieve_parent_token[:, 1:seq_len] = torch.arange(
            seq_len - 1,
            device=retrieve_parent_token.device,
            dtype=retrieve_parent_token.dtype,
        )

    return result


def causal_conv1d_update_cpu(
    mixed_qkv,
    conv_states,
    conv_weights,
    bias,
    activation,
    conv_state_indices,
    intermediate_conv_window=None,
    intermediate_state_indices=None,
    retrieve_next_token=None,
    retrieve_next_sibling=None,
    retrieve_parent_token=None,
):
    if intermediate_conv_window is not None:
        if retrieve_next_sibling is not None and bool(
            (retrieve_next_sibling != -1).any()
        ):
            raise NotImplementedError(
                "causal_conv1d_update_cpu target-verify supports a linear draft "
                "chain only (topk <= 1); EAGLE tree verify has siblings."
            )
        return _causal_conv1d_update_cpu_verify(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices,
            intermediate_conv_window,
            intermediate_state_indices,
            retrieve_parent_token,
        )
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


def _assert_linear_chain(retrieve_parent_token, steps):
    if retrieve_parent_token is None:
        return
    expected = torch.arange(
        -1,
        steps - 1,
        device=retrieve_parent_token.device,
        dtype=retrieve_parent_token.dtype,
    )
    expected[0] = 0
    if not torch.equal(
        retrieve_parent_token, expected.expand_as(retrieve_parent_token)
    ):
        raise NotImplementedError(
            "fused_sigmoid_gating_delta_rule_update_cpu target-verify supports a "
            "linear draft chain only (topk <= 1); got a branching EAGLE tree."
        )


def fused_sigmoid_gating_delta_rule_update_cpu(
    *,
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    use_qk_l2norm_in_kernel,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    is_kda=False,
    disable_state_update=False,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    retrieve_parent_token=None,
):
    if intermediate_states_buffer is None:
        return torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            initial_state_source,
            initial_state_indices,
            cu_seqlens,
            use_qk_l2norm_in_kernel,
            softplus_beta,
            softplus_threshold,
        )

    # Target verify. Tokens arrive packed as n * steps + t, which is exactly the
    # layout the spec op expects, so the whole draft block goes out in a single
    # dispatch: the kernel walks t sequentially per (sequence, v_head) and writes
    # the state after each token straight into intermediate_states_buffer. The
    # committed ssm_states are only read unless disable_state_update is False.
    assert not is_kda, "KDA target_verify is not supported on CPU"
    assert a.dim() == 2, f"expected per-token gating [tokens, heads], got {a.shape}"

    batch = cu_seqlens.numel() - 1
    steps = q.shape[1] // batch
    _assert_linear_chain(retrieve_parent_token, steps)

    return torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_spec_cpu(
        A_log,
        dt_bias,
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        initial_state_source,
        initial_state_indices,
        intermediate_states_buffer,
        intermediate_state_indices.contiguous(),
        steps,
        use_qk_l2norm_in_kernel,
        disable_state_update,
        softplus_beta,
        softplus_threshold,
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
