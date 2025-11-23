from typing import Optional

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit
def _causal_conv_speculative_step_kernel(
    X_ptr,  # Pointer to current input x [B, T, D]
    Cache_ptr,  # Pointer to cache [N, D, W]
    Kernels_ptr,  # Pointer to generated kernels [B, T, D, W]
    Out_ptr,  # Pointer to output tensor [B, T, D]
    Cache_idx_ptr,  # Pointer to cache indices [B]
    Intermediate_conv_window_ptr,  # Pointer to intermediate conv window [N, W-1+T, D]
    B,
    D,
    X_stride_b,
    X_stride_t,
    X_stride_d,
    Cache_stride_b,
    Cache_stride_d,
    Cache_stride_w,
    Kernels_stride_b,
    Kernels_stride_t,
    Kernels_stride_d,
    Kernels_stride_w,
    Out_stride_b,
    Out_stride_t,
    Out_stride_d,
    Cache_idx_stride_b,
    Intermediate_conv_window_stride_b,
    Intermediate_conv_window_stride_t,
    Intermediate_conv_window_stride_d,
    pad_slot_id: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_time = tl.program_id(1)
    pid_d_block = tl.program_id(2)
    cache_idx = tl.load(Cache_idx_ptr + pid_batch * Cache_idx_stride_b)
    if cache_idx == pad_slot_id:
        return
    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D
    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    offs_w = tl.arange(0, W)
    k_ptrs = Kernels_ptr + (
        pid_batch * Kernels_stride_b
        + pid_time * Kernels_stride_t
        + offs_d[:, None] * Kernels_stride_d
        + offs_w[None, :] * Kernels_stride_w
    )
    k_vals = tl.load(k_ptrs, mask=d_mask[:, None], other=0.0)
    x_abs_indices = pid_time + offs_w - W + 1
    x_ptrs = X_ptr + (
        pid_batch * X_stride_b
        + x_abs_indices[None, :] * X_stride_t
        + offs_d[:, None] * X_stride_d
    )
    x_final_load_mask = d_mask[:, None] & (x_abs_indices >= 0)[None, :]
    x_input_vals = tl.load(x_ptrs, mask=x_final_load_mask, other=0.0)
    cache_ptrs = Cache_ptr + (
        cache_idx * Cache_stride_b
        + (x_abs_indices + W)[None, :] * Cache_stride_w
        + offs_d[:, None] * Cache_stride_d
    )
    cache_final_load_mask = d_mask[:, None] & (x_abs_indices < 0)[None, :]
    vals_from_cache = tl.load(cache_ptrs, mask=cache_final_load_mask, other=0.0)
    x_vals = x_input_vals + vals_from_cache
    product = k_vals * x_vals
    accumulator += tl.sum(product, axis=1)
    out_ptrs = Out_ptr + (
        pid_batch * Out_stride_b + pid_time * Out_stride_t + offs_d * Out_stride_d
    )
    tl.store(out_ptrs, accumulator, mask=d_mask)

    intermediate_conv_window_ptrs = Intermediate_conv_window_ptr + (
        cache_idx * Intermediate_conv_window_stride_b
        + (W - 2 + pid_time) * Intermediate_conv_window_stride_t
        + offs_d * Intermediate_conv_window_stride_d
    )
    offs_w = tl.arange(0, W)
    last_col_mask = offs_w == W - 1
    x_vals_last_col = tl.sum(x_vals * last_col_mask[None, :], axis=1)
    tl.store(intermediate_conv_window_ptrs, x_vals_last_col, mask=d_mask)


def causal_conv_step_triton_speculative(
    x: torch.Tensor,  # Input tensor [B, T, D]
    cache: torch.Tensor,  # Cache tensor [N, D, W]
    kernels: torch.Tensor,  # Kernels tensor [B, T, D, W]
    cache_indices: torch.Tensor,  # Cache indices tensor [B]
    intermediate_conv_window: torch.Tensor,  # Intermediate conv window tensor [N, W-2+T, D], updated in-place
) -> torch.Tensor:  # Returns output tensor [B, D] (before activation)
    B, T, D = x.shape
    W = cache.shape[2]
    assert 1 < W <= 4, f"Kernel W={W}, this optimized version assumes 1 < W <= 4"
    out = torch.empty_like(x)

    grid = lambda meta: (B, T, triton.cdiv(D, meta["BLOCK_SIZE_D"]))
    BLOCK_SIZE_D = 64

    _causal_conv_speculative_step_kernel[grid](
        x,
        cache,
        kernels,
        out,
        cache_indices,
        intermediate_conv_window,
        B,
        D,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        kernels.stride(0),
        kernels.stride(1),
        kernels.stride(2),
        kernels.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        cache_indices.stride(0),
        intermediate_conv_window.stride(0),
        intermediate_conv_window.stride(1),
        intermediate_conv_window.stride(2),
        pad_slot_id=PAD_SLOT_ID,
        W=W,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return out


# modified from causal_conv1d_triton.py
@triton.jit()
def _causal_dynamic_conv1d_update_kernel(
    x_ptr,
    w_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    intermediate_conv_window_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    retrieve_parent_token_ptr,
    o_ptr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_seq: tl.constexpr,
    stride_w_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_retrieve_next_token_seq: tl.constexpr,
    stride_retrieve_next_token_token: tl.constexpr,
    stride_retrieve_next_sibling_seq: tl.constexpr,
    stride_retrieve_next_sibling_token: tl.constexpr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    conv_state_batch_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices
    ).to(tl.int64)
    if conv_state_batch_coord == pad_slot_id:
        return

    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    conv_state_token_offset = (
        state_len - KERNEL_WIDTH + 1
    )  # our conv state is right aligned, not left aligned like causal_conv1d_triton.py
    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)

    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    idx_tokens = tl.arange(0, NP2_SEQLEN)
    mask_retrieve = idx_tokens < seqlen
    retrieve_next_token_base = (
        retrieve_next_token_ptr
        + (idx_seq * stride_retrieve_next_token_seq)
        + idx_tokens * stride_retrieve_next_token_token
    )
    retrieve_next_tokens = tl.load(retrieve_next_token_base, mask_retrieve)
    retrieve_next_sibling_base = (
        retrieve_next_sibling_ptr
        + (idx_seq * stride_retrieve_next_sibling_seq)
        + idx_tokens * stride_retrieve_next_sibling_token
    )
    retrieve_next_siblings = tl.load(retrieve_next_sibling_base, mask_retrieve)
    parent_idx_tokens = tl.zeros((NP2_SEQLEN,), dtype=tl.int32)

    w_base = w_ptr + (idx_seq * stride_w_seq + idx_feats * stride_w_dim)

    x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)
    mask_x_1d = idx_feats < dim

    for idx_token in tl.static_range(seqlen):
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
        w_base += stride_w_token

        acc = acc_preload

        retrieve_next_token_idx = tl.sum(
            tl.where(idx_tokens == idx_token, retrieve_next_tokens, 0)
        )
        if retrieve_next_token_idx != pad_slot_id:
            parent_idx_tokens = tl.where(
                idx_tokens == retrieve_next_token_idx,
                idx_token,
                parent_idx_tokens,
            )

        retrieve_sibling_token_idx = tl.sum(
            tl.where(idx_tokens == idx_token, retrieve_next_siblings, 0)
        )
        if retrieve_sibling_token_idx != pad_slot_id:
            parent_idx_token = tl.sum(
                tl.where(idx_tokens == idx_token, parent_idx_tokens, 0)
            )
            parent_idx_tokens = tl.where(
                idx_tokens == retrieve_sibling_token_idx,
                parent_idx_token,
                parent_idx_tokens,
            )

        _idx_token = idx_token
        x_ptrs_1d = x_base_1d + _idx_token * stride_x_token
        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 0:
                    matrix_w = w_col1
                else:
                    matrix_w = w_col0
            elif KERNEL_WIDTH == 3:
                if j == 0:
                    matrix_w = w_col2
                elif j == 1:
                    matrix_w = w_col1
                else:
                    matrix_w = w_col0
            elif KERNEL_WIDTH == 4:
                if j == 0:
                    matrix_w = w_col3
                elif j == 1:
                    matrix_w = w_col2
                elif j == 2:
                    matrix_w = w_col1
                else:
                    matrix_w = w_col0

            base_ptr = (
                intermediate_conv_window_ptr
                + conv_state_batch_coord * stride_inter_seq
                + idx_token * stride_inter_step
                + idx_feats * stride_inter_dim
            )

            if KERNEL_WIDTH - j - 2 >= 0:
                tl.store(
                    base_ptr + (state_len - j - 1) * stride_inter_win,
                    matrix_x,
                    mask=mask_w,
                )

            acc += matrix_x * matrix_w

            if _idx_token > 0:
                _idx_token = tl.sum(
                    tl.where(idx_tokens == _idx_token, parent_idx_tokens, 0)
                )
                x_ptrs_1d = x_base_1d + _idx_token * stride_x_token
                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            else:
                if KERNEL_WIDTH == 2:
                    if _idx_token == 0:
                        matrix_x = col0
                elif KERNEL_WIDTH == 3:
                    if _idx_token == 0:
                        matrix_x = col1
                    else:
                        matrix_x = col0
                elif KERNEL_WIDTH == 4:
                    if _idx_token == 0:
                        matrix_x = col2
                    elif _idx_token == -1:
                        matrix_x = col1
                    else:
                        matrix_x = col0
                _idx_token = _idx_token - 1

        acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)
        tl.store(
            retrieve_parent_token_ptr
            + idx_seq * stride_retrieve_parent_token_seq
            + idx_tokens * stride_retrieve_parent_token_token,
            parent_idx_tokens,
            mask=mask_retrieve,
        )


def causal_dynamic_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_indices: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    batch, dim, seqlen = x.shape
    width = weight.shape[3]
    _, _, state_len = conv_state.size()

    out = torch.empty_like(x)
    stride_w_seq, stride_w_token, stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_seqlen = triton.next_power_of_2(seqlen)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
        intermediate_conv_window.stride(0),
        intermediate_conv_window.stride(1),
        intermediate_conv_window.stride(2),
        intermediate_conv_window.stride(3),
    )

    stride_retrieve_next_token_seq, stride_retrieve_next_token_token = (
        retrieve_next_token.stride(0),
        retrieve_next_token.stride(1),
    )

    stride_retrieve_next_sibling_seq, stride_retrieve_next_sibling_token = (
        retrieve_next_sibling.stride(0),
        retrieve_next_sibling.stride(1),
    )

    stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token = (
        retrieve_parent_token.stride(0),
        retrieve_parent_token.stride(1),
    )

    _causal_dynamic_conv1d_update_kernel[grid](
        x,
        weight,
        conv_state,
        conv_state_indices,
        intermediate_conv_window,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_seq,
        stride_w_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_retrieve_next_token_seq,
        stride_retrieve_next_token_token,
        stride_retrieve_next_sibling_seq,
        stride_retrieve_next_sibling_token,
        stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        KERNEL_WIDTH=width,
        NP2_SEQLEN=np2_seqlen,
        BLOCK_N=256,
    )
    return out
