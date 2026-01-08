"""Optimized causal_conv1d_update: directly output split q/k/v."""

from typing import Optional, Tuple, Union
import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit()
def _causal_conv1d_update_split_qkv_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_q_seq: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_k_seq: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_v_seq: tl.constexpr,
    stride_v_dim: tl.constexpr,
    stride_v_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq

    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: Update conv state (same as original)
    idx_tokens = tl.arange(0, NP2_STATELEN)

    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
    )

    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)
    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )
    mask_store = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask_store)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4: PRE-LOAD WEIGHTS
    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token and write to split buffers
    for idx_token in tl.static_range(seqlen):
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

        # Update sliding window
        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        # Apply activation
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        mask_feat = (idx_token < seqlen) & (idx_feats < dim)

        # Query: idx_feats in [0, key_dim)
        is_query = idx_feats < key_dim
        q_feat_idx = idx_feats  # 0-based index within query
        q_ptrs = (
            q_ptr
            + idx_seq * stride_q_seq
            + idx_token * stride_q_token
            + q_feat_idx * stride_q_dim
        )
        tl.store(q_ptrs, acc, mask=mask_feat & is_query)

        # Key: idx_feats in [key_dim, 2*key_dim)
        is_key = (idx_feats >= key_dim) & (idx_feats < 2 * key_dim)
        k_feat_idx = idx_feats - key_dim
        k_ptrs = (
            k_ptr
            + idx_seq * stride_k_seq
            + idx_token * stride_k_token
            + k_feat_idx * stride_k_dim
        )
        tl.store(k_ptrs, acc, mask=mask_feat & is_key)

        # Value: idx_feats in [2*key_dim, 2*key_dim+value_dim)
        is_value = (idx_feats >= 2 * key_dim) & (idx_feats < 2 * key_dim + value_dim)
        v_feat_idx = idx_feats - 2 * key_dim
        v_ptrs = (
            v_ptr
            + idx_seq * stride_v_seq
            + idx_token * stride_v_token
            + v_feat_idx * stride_v_dim
        )
        tl.store(v_ptrs, acc, mask=mask_feat & is_value)


def causal_conv1d_update_split_qkv(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    key_dim: int,
    value_dim: int,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = "silu",
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized causal_conv1d_update that directly outputs split q, k, v."""
    # Validate and prepare
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    assert dim == 2 * key_dim + value_dim, f"dim {dim} != 2*{key_dim} + {value_dim}"

    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()

    # 创建输出 buffer（已经是分离的！）
    query = torch.empty(
        (batch, key_dim, seqlen),
        dtype=x.dtype,
        device=x.device,
    )
    key = torch.empty(
        (batch, key_dim, seqlen),
        dtype=x.dtype,
        device=x.device,
    )
    value = torch.empty(
        (batch, value_dim, seqlen),
        dtype=x.dtype,
        device=x.device,
    )

    # Triton kernel launch
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    BLOCK_N = 256
    grid = (batch, triton.cdiv(dim, BLOCK_N))

    _causal_conv1d_update_split_qkv_kernel[grid](
        x_ptr=x,
        w_ptr=weight,
        bias_ptr=bias,
        conv_state_ptr=conv_state,
        conv_state_indices_ptr=conv_state_indices,
        q_ptr=query,
        k_ptr=key,
        v_ptr=value,
        key_dim=key_dim,
        value_dim=value_dim,
        batch=batch,
        dim=dim,
        seqlen=seqlen,
        state_len=state_len,
        num_cache_lines=num_cache_lines,
        stride_x_seq=x.stride(0),
        stride_x_dim=x.stride(1),
        stride_x_token=x.stride(2),
        stride_w_dim=weight.stride(0),
        stride_w_width=weight.stride(1),
        stride_conv_state_seq=conv_state.stride(0),
        stride_conv_state_dim=conv_state.stride(1),
        stride_conv_state_tok=conv_state.stride(2),
        stride_state_indices=stride_state_indices,
        stride_q_seq=query.stride(0),
        stride_q_dim=query.stride(1),
        stride_q_token=query.stride(2),
        stride_k_seq=key.stride(0),
        stride_k_dim=key.stride(1),
        stride_k_token=key.stride(2),
        stride_v_seq=value.stride(0),
        stride_v_dim=value.stride(1),
        stride_v_token=value.stride(2),
        pad_slot_id=pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=BLOCK_N,
    )

    if unsqueeze:
        query = query.squeeze(-1)
        key = key.squeeze(-1)
        value = value.squeeze(-1)

    return query, key, value
