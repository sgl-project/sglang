"""Causal conv1d with fused split q/k/v output for prefill."""

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID


@triton.jit()
def _causal_conv1d_fwd_split_kernel(
    x_ptr, w_ptr, bias_ptr,
    initial_states_ptr, cache_indices_ptr, has_initial_states_ptr, query_start_loc_ptr,
    q_ptr, k_ptr, v_ptr,
    key_dim: tl.constexpr, value_dim: tl.constexpr, dim: tl.constexpr,
    seqlen: tl.int32, num_cache_lines: tl.constexpr,
    stride_x_dim: tl.constexpr, stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr, stride_istate_dim: tl.constexpr, stride_istate_token: tl.constexpr,
    stride_q_token: tl.constexpr, stride_q_dim: tl.constexpr,
    stride_k_token: tl.constexpr, stride_k_dim: tl.constexpr,
    stride_v_token: tl.constexpr, stride_v_dim: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr, KERNEL_WIDTH: tl.constexpr, SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr, HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr, USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused causal conv1d + split q/k/v output for prefill."""
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
        
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return
            
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )

    w_base = w_ptr + (idx_feats * stride_w_dim)

    if chunk_offset == 0:
        load_init_state = False
        if HAS_INITIAL_STATES:
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                col0 = tl.load(prior_tokens, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                col1 = tl.load(prior_tokens, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                col2 = tl.load(prior_tokens, mask_w, 0.0)
                col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
                VAL = state_len - seqlen
                x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = (
                    conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = (
                    conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            col0 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            col1 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col1 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token

    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w, other=0.0)
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w, other=0.0)

    mask_x_1d = idx_feats < dim

    for idx_token in range(segment_len):
        acc = acc_preload
        matrix_w = w_col0
        matrix_x = col0
        
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        global_token_idx = sequence_start_index + token_offset + idx_token
        mask_feat = (idx_token < segment_len) & (idx_feats < dim)

        is_query = idx_feats < key_dim
        q_ptrs = q_ptr + global_token_idx * stride_q_token + idx_feats * stride_q_dim
        tl.store(q_ptrs, acc, mask=mask_feat & is_query)

        is_key = (idx_feats >= key_dim) & (idx_feats < 2 * key_dim)
        k_ptrs = k_ptr + global_token_idx * stride_k_token + (idx_feats - key_dim) * stride_k_dim
        tl.store(k_ptrs, acc, mask=mask_feat & is_key)

        is_value = (idx_feats >= 2 * key_dim) & (idx_feats < 2 * key_dim + value_dim)
        v_ptrs = v_ptr + global_token_idx * stride_v_token + (idx_feats - 2 * key_dim) * stride_v_dim
        tl.store(v_ptrs, acc, mask=mask_feat & is_value)


def causal_conv1d_fn_split_qkv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    k_dim: int,
    v_dim: int,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: str = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal conv1d with fused split output for prefill. Returns q, k, v as contiguous."""
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    q_out = torch.empty(cu_seqlen, k_dim, device=x.device, dtype=x.dtype)
    k_out = torch.empty(cu_seqlen, k_dim, device=x.device, dtype=x.dtype)
    v_out = torch.empty(cu_seqlen, v_dim, device=x.device, dtype=x.dtype)

    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    
    num_cache_lines = 0
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    if conv_states is not None:
        num_cache_lines = conv_states.size(0)
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)

    def grid(META):
        max_seq_len = max(seq_lens_cpu)
        return (
            len(seq_lens_cpu),
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_split_kernel[grid](
        x, weight, bias, conv_states, cache_indices, has_initial_state, query_start_loc,
        q_out, k_out, v_out,
        k_dim, v_dim, dim, cu_seqlen, num_cache_lines,
        stride_x_dim, stride_x_token, stride_w_dim, stride_w_width,
        stride_istate_seq, stride_istate_dim, stride_istate_token,
        q_out.stride(0), q_out.stride(1),
        k_out.stride(0), k_out.stride(1),
        v_out.stride(0), v_out.stride(1),
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=8,
        BLOCK_N=256,
        num_stages=2,
    )

    return q_out, k_out, v_out

