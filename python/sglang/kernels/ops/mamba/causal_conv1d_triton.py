# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py

from typing import List, Optional, Union

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

PAD_SLOT_ID = -1


# Tokens per program of _causal_conv1d_fwd_kernel; a block_table enumerates
# (sequence, chunk) pairs in these units.
CAUSAL_CONV1D_FWD_BLOCK_M = 8


@triton.jit(do_not_specialize=["token_block_start"])
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    block_table_ptr,  # (num_blocks, 2) int32 (seq, chunk) rows; USE_BLOCK_TABLE only
    # Matrix dimensions
    dim: tl.constexpr,
    total_tokens: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    UPDATE_CONV_STATES: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BLOCK_TABLE: tl.constexpr = False,
    OUTPUT_SPLIT_SIZE: tl.constexpr = 0,
    token_block_start: tl.int32 = 0,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    if USE_BLOCK_TABLE:
        # CUDA-graph safe launch: the grid is a fixed upper bound on the
        # (sequence, BLOCK_M-chunk) pairs of a token bucket and each program
        # reads its pair from the table; padded rows carry pad_slot_id.
        idx_block = tl.program_id(0)
        idx_seq = tl.load(block_table_ptr + idx_block * 2)
        chunk_offset = tl.load(block_table_ptr + idx_block * 2 + 1)
    else:
        # single-sequence id
        idx_seq = tl.program_id(0)
        chunk_offset = tl.program_id(1) + token_block_start

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    # base of the sequence
    x_base = (
        x_ptr
        + sequence_start_index.to(tl.int64) * stride_x_token
        + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states. Cast to x's dtype so col* keep a single
            # dtype across the whole kernel: when x is fp16 but the conv-state
            # cache is bf16 (e.g. MiniCPM-V GDN prefill), the chunk_offset==0
            # branch would otherwise produce bf16 cols while the chunk_offset>0
            # else branch (and the sliding-window reassignment) produce fp16,
            # which trips Triton's if/else phi type check on col0.
            x_elem_ty = x_ptr.dtype.element_ty
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
        else:
            # prior-tokens are zeros (same x dtype as every other col* source)
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + (
                    (sequence_start_index + idx_tokens_last).to(tl.int64)
                    * stride_x_token
                )[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            if UPDATE_CONV_STATES:
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                if UPDATE_CONV_STATES:
                    tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                if UPDATE_CONV_STATES:
                    tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset.to(tl.int64) * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
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
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

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
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        output_token = (sequence_start_index + token_offset + idx_token).to(tl.int64)
        if OUTPUT_SPLIT_SIZE:
            # Store each channel group as a contiguous [tokens, channels] plane.
            # KDA can consume Q/K/V directly without three materialization copies.
            o_ptrs = (
                o_ptr
                + (idx_feats // OUTPUT_SPLIT_SIZE).to(tl.int64)
                * total_tokens
                * OUTPUT_SPLIT_SIZE
                + output_token * OUTPUT_SPLIT_SIZE
                + idx_feats % OUTPUT_SPLIT_SIZE
            )
        else:
            o_ptrs = o_ptr + output_token * stride_o_token + idx_feats * stride_o_dim

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    block_table: Optional[torch.Tensor] = None,
    output_split_size: int = 0,
    out: Optional[torch.Tensor] = None,
    token_block_range: Optional[tuple[int, int]] = None,
    update_conv_states: bool = True,
    **kwargs,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    seq_lens_cpu: (batch) int32
        The sequence lengths of the sequences in the batch
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3

    output_split_size: channel width of each independently contiguous output
        plane, or zero for the original layout. Must divide dim.
    out: same shape as `x`, or (dim // output_split_size, cu_seq_len,
        output_split_size) when output_split_size is nonzero.
        Optional destination shared by disjoint token-block launches.
    token_block_range: optional [begin, end) range of 8-token blocks per sequence.
        Block zero owns the initial-state read and final conv-state write.
        Interior blocks only read local input, permitting halo exchange overlap.
    update_conv_states: whether to write final history back to conv_states.
        Disable for read-only halo views into a projected input buffer.
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    dim, cu_seqlen = x.shape
    if output_split_size:
        if output_split_size < 0 or dim % output_split_size:
            raise ValueError("output_split_size must be positive and divide dim")
        output_shape = (dim // output_split_size, cu_seqlen, output_split_size)
    else:
        output_shape = x.shape
    if out is None:
        out = (
            torch.empty(output_shape, dtype=x.dtype, device=x.device)
            if output_split_size
            else torch.empty_like(x)
        )
    elif out.shape != output_shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError(
            "Convolution output must match the output shape, dtype and device"
        )
    if block_table is not None:
        if token_block_range is not None:
            raise ValueError("Token block ranges cannot be combined with a block table")
        block_start = block_end = 0
    else:
        num_blocks = triton.cdiv(
            max(seq_lens_cpu, default=0), CAUSAL_CONV1D_FWD_BLOCK_M
        )
        block_start, block_end = token_block_range or (0, num_blocks)
        if not 0 <= block_start <= block_end <= num_blocks:
            raise ValueError("Convolution token block range exceeds the input")
        if block_start == block_end:
            return out

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (
            num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2]
        )
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        # assert stride_istate_dim == 1
    if output_split_size:
        # The plane pitch uses runtime total_tokens in the kernel; do not
        # specialize the launch on every ragged prefill's token count.
        stride_o_seq = 0
        stride_o_dim = 1
        stride_o_token = output_split_size
    elif out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        padded_batch = query_start_loc.size(0) - 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert conv_states is not None, (
                "ERROR: `has_initial_state` is used, which needs also `conv_states`"
            )
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    def grid(META):
        if block_table is not None:
            # (num_blocks, 1, feature blocks): no host-side sequence lengths.
            return (block_table.shape[0], 1, triton.cdiv(dim, META["BLOCK_N"]))
        return (
            len(seq_lens_cpu),  # batch_size
            block_end - block_start,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        block_table,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        UPDATE_CONV_STATES=update_conv_states,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # launch_cooperative_grid=True
        BLOCK_M=CAUSAL_CONV1D_FWD_BLOCK_M,
        BLOCK_N=256,
        USE_BLOCK_TABLE=block_table is not None,
        OUTPUT_SPLIT_SIZE=output_split_size,
        token_block_start=block_start,
        num_stages=2,
    )
    return out


def causal_conv1d_with_halo(
    x: torch.Tensor,
    halo: torch.Tensor,
    weight: torch.Tensor,
    *,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    has_initial_state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    out: Optional[torch.Tensor] = None,
    token_block_range: Optional[tuple[int, int]] = None,
):
    """Convolve valid tokens with read-only per-sequence predecessor rows.

    x is [channels, sum(seq_lens_cpu)]; halo is [sequences, channels, width-1]
    in chronological order. Both may be strided views of one projection buffer.
    query_start_loc and seq_lens_cpu describe only valid tokens, excluding halo
    and physical padding. has_initial_state selects halo versus zero history.
    Output contains only valid tokens; out must not alias x or halo. The caller
    retains ownership of any persistent convolution state and final-history store.
    """
    if x.ndim != 2 or weight.ndim != 2 or weight.shape[0] != x.shape[0]:
        raise ValueError("Convolution expects [channels, tokens] and [channels, width]")
    if weight.shape[1] not in (2, 3, 4):
        raise ValueError("Read-only halo convolution supports widths 2, 3 and 4")
    sequences = len(seq_lens_cpu)
    if halo.shape != (sequences, x.shape[0], weight.shape[1] - 1):
        raise ValueError("Halo must contain width-1 predecessor rows per sequence")
    if halo.dtype != x.dtype or halo.device != x.device:
        raise ValueError("Halo must match the input dtype and device")
    if (
        not sequences
        or any(length < 0 for length in seq_lens_cpu)
        or sum(seq_lens_cpu) != x.shape[1]
        or query_start_loc.shape != (sequences + 1,)
        or has_initial_state.shape != (sequences,)
        or query_start_loc.device != x.device
        or has_initial_state.device != x.device
        or query_start_loc.dtype not in (torch.int32, torch.int64)
        or has_initial_state.dtype != torch.bool
    ):
        raise ValueError("Convolution metadata must describe the valid input tokens")
    if activation not in (None, "silu", "swish"):
        raise ValueError("Convolution activation must be None, silu or swish")
    return causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states=halo,
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens_cpu,
        has_initial_state=has_initial_state,
        activation=activation,
        out=out,
        token_block_range=token_block_range,
        update_conv_states=False,
    )


# HAS_EAGLE_TREE_CUSTOM_ATTN_MASK is added to support eagle tree attention mask
# retrieve_next_token_ptr: [N, NP2_T], retrieve_next_sibling_ptr: [N, NP2_T]
# e.g. for a sequence of length 4, the eagle tree attention structure is:
# retrieve_next_token=[1, 3, -1, -1] -> retrieve_next_token[i]: the 1st child token of token i
# retrieve_next_sibling=[-1, 2, -1, -1] -> retrieve_next_sibling[i]: the 1st tree sibling token of token i
# retrieve_parent_token=[n/a, 0, 0, 1] -> retrieve_parent_token[i]: the parent token of token i
# Tree:
#    0
#   / \
#  1   2
# /
# 3
# When calculating token 3's convolution, it should conv to token 1 (parent) and token 0 (grand-parent)
# When calculating token 2's convolution, it should conv to token 0 (parent)
# This kernel is a fused kernel which will also produce retrieve_parent_token based on retrieve_next_token & retrieve_next_sibling
@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accept_tokens_ptr,
    intermediate_conv_window_ptr,
    intermediate_state_indices_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    retrieve_parent_token_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
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
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_intermediate_state_indices: tl.constexpr,
    stride_retrieve_next_token_seq: tl.constexpr,
    stride_retrieve_next_token_token: tl.constexpr,
    stride_retrieve_next_sibling_seq: tl.constexpr,
    stride_retrieve_next_sibling_token: tl.constexpr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    DEDUP_CONV_WINDOW: tl.constexpr = False,
    NP2_WINDOW_COLS: tl.constexpr = 1,
):
    # ruff: noqa: E501
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
        if SAVE_INTERMEDIATE:
            intermediate_state_batch_coord = tl.load(
                intermediate_state_indices_ptr
                + idx_seq * stride_intermediate_state_indices
            ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = tl.load(num_accept_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # The conv_state updates works in a sliding window manner,
    # at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            :, None
        ]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        idx_tokens = tl.arange(0, NP2_SEQLEN)  # [BLOCK_M]
        # Update parent mapping for all tokens at once using vectorized operations
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

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    if SAVE_INTERMEDIATE and DEDUP_CONV_WINDOW:
        tl.static_assert(KERNEL_WIDTH == 4)
        tl.static_assert(not HAS_EAGLE_TREE_CUSTOM_ATTN_MASK)
        # Deduplicated window (KERNEL_WIDTH == 4, linear chain): step t's window is
        # columns t..t+2 of one shared per-channel row
        # [state[1], state[2], x_0, ..., x_{seqlen-1}]. Write those seqlen + 2
        # columns as one [BLOCK_N, NP2_WINDOW_COLS] tile instead of three
        # lane-strided stores per token; the bytes are the same.
        idx_c = tl.arange(0, NP2_WINDOW_COLS)
        x_win = tl.load(
            x_base[:, None] + ((idx_c - 2) * stride_x_token)[None, :],
            mask_w[:, None] & ((idx_c >= 2) & (idx_c < seqlen + 2))[None, :],
            other=0.0,
        )
        win = tl.where(
            (idx_c == 0)[None, :],
            col1[:, None],
            tl.where((idx_c == 1)[None, :], col2[:, None], x_win),
        )
        tl.store(
            intermediate_conv_window_ptr
            + intermediate_state_batch_coord * stride_inter_seq
            + (idx_feats * stride_inter_dim)[:, None]
            + (idx_c * stride_inter_win)[None, :],
            win,
            mask=mask_w[:, None] & (idx_c < seqlen + 2)[None, :],
        )

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # set the parent index of the next token in the eagle tree
            # next token's parent is the current token
            retrieve_next_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_tokens, 0)
            )
            if retrieve_next_token_idx != -1:  # pad slot id
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_next_token_idx,
                    idx_token,
                    parent_idx_tokens,
                )
            # next token's parent is the parent of the current token
            retrieve_sibling_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_siblings, 0)
            )
            if retrieve_sibling_token_idx != -1:  # pad slot id
                parent_idx_token = tl.sum(
                    tl.where(idx_tokens == idx_token, parent_idx_tokens, 0)
                )
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_sibling_token_idx,
                    parent_idx_token,
                    parent_idx_tokens,
                )

            _idx_token = idx_token
            x_ptrs_1d = x_base_1d + _idx_token * stride_x_token  # [BLOCK_N]
            matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            # convolution operation: itself * wcol[-1] + parent * wcol[-2] + grand-parent * wcol[-3] + ...
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

                if SAVE_INTERMEDIATE:
                    # Save the window state after consuming this token
                    # Layout: [seq(cache line), step, dim, win(K-1)]
                    base_ptr = (
                        intermediate_conv_window_ptr
                        + intermediate_state_batch_coord * stride_inter_seq
                        + idx_token * stride_inter_step
                        + idx_feats * stride_inter_dim
                    )

                    # store itself in KERNEL_WIDTH-2 slot, parent in KERNEL_WIDTH-3 slot, grand-parent in KERNEL_WIDTH-4 slot, ...
                    if KERNEL_WIDTH - j - 2 >= 0:
                        tl.store(
                            base_ptr + (KERNEL_WIDTH - j - 2) * stride_inter_win,
                            matrix_x,
                            mask=mask_w,
                        )

                acc += matrix_x * matrix_w

                # move to parent for next iteration
                if _idx_token > 0:
                    _idx_token = tl.sum(
                        tl.where(idx_tokens == _idx_token, parent_idx_tokens, 0)
                    )
                    x_ptrs_1d = x_base_1d + _idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                else:
                    # no parent within the current chunk, load from prev conv state: col[-1] (idx 0's parent), col[-2] (idx 0's grand parent), ...
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
        else:
            matrix_w = w_col0
            matrix_x = col0

            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 1:  # KERNEL_WIDTH-1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
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
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

                acc += matrix_x * matrix_w  # [BLOCK_N]

            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x

            if SAVE_INTERMEDIATE and not DEDUP_CONV_WINDOW:
                # Save the window state after consuming this token
                # Layout: [seq(cache line), step, dim, win(K-1)]
                base_ptr = (
                    intermediate_conv_window_ptr
                    + intermediate_state_batch_coord * stride_inter_seq
                    + idx_token * stride_inter_step
                    + idx_feats * stride_inter_dim
                )
                if KERNEL_WIDTH >= 2:
                    tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
                if KERNEL_WIDTH >= 3:
                    tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
                if KERNEL_WIDTH >= 4:
                    tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)

        # fuse: store calculated retrieve_parent_token to tensor
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            tl.store(
                retrieve_parent_token_ptr
                + idx_seq * stride_retrieve_parent_token_seq
                + idx_tokens * stride_retrieve_parent_token_token,
                parent_idx_tokens,
                mask=mask_retrieve,
            )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accept_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        )
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape
            assert intermediate_state_indices is not None
            assert (batch,) == intermediate_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = torch.empty_like(x)
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    stride_intermediate_state_indices = (
        intermediate_state_indices.stride(0)
        if intermediate_state_indices is not None
        else 0
    )
    if num_accept_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    np2_seqlen = triton.next_power_of_2(seqlen)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0
    # In a width-4 linear chain, step t's window is columns t..t+2 of the row
    # [state[1], state[2], x_0, ..., x_{seqlen-1}]. When the window view's step
    # stride equals its window stride s, the per-token stores write column t + w
    # of that row at (t + w) * s, and one store of the whole row writes column c
    # at c * s: the same addresses and values for any layout with these strides
    # (MambaPool's deduplicated view is one). The kernel then writes the row as
    # one tile. Tree verify and other widths keep the per-token stores.
    dedup_conv_window = (
        intermediate_conv_window is not None
        and retrieve_next_token is None
        and width == 4
        and stride_inter_step == stride_inter_win
    )

    # prepare retrieve next token buffer strides if provided
    if retrieve_next_token is not None:
        stride_retrieve_next_token_seq, stride_retrieve_next_token_token = (
            retrieve_next_token.stride(0),
            retrieve_next_token.stride(1),
        )
    else:
        stride_retrieve_next_token_seq = stride_retrieve_next_token_token = 0

    # prepare retrieve next sibling buffer strides if provided
    if retrieve_next_sibling is not None:
        stride_retrieve_next_sibling_seq, stride_retrieve_next_sibling_token = (
            retrieve_next_sibling.stride(0),
            retrieve_next_sibling.stride(1),
        )
    else:
        stride_retrieve_next_sibling_seq = stride_retrieve_next_sibling_token = 0

    # prepare retrieve parent token buffer strides if provided
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token = (
            retrieve_parent_token.stride(0),
            retrieve_parent_token.stride(1),
        )
    else:
        stride_retrieve_parent_token_seq = stride_retrieve_parent_token_token = 0

    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    _causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accept_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
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
        stride_intermediate_state_indices,
        stride_retrieve_next_token_seq,
        stride_retrieve_next_token_token,
        stride_retrieve_next_sibling_seq,
        stride_retrieve_next_sibling_token,
        stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accept_tokens is not None,
        NP2_STATELEN=np2_statelen,
        NP2_SEQLEN=np2_seqlen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_next_token is not None,
        DEDUP_CONV_WINDOW=dedup_conv_window,
        NP2_WINDOW_COLS=(
            triton.next_power_of_2(seqlen + width - 2) if dedup_conv_window else 1
        ),
        **pdl_kwargs,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out


@triton.jit(do_not_specialize=["N"])
def _causal_conv1d_peer_qkv_kernel(
    X,
    W,
    History,
    Peers,
    Q,
    K,
    V,
    N,
    C: tl.constexpr,
    SX: tl.constexpr,
    QD: tl.constexpr,
    KD: tl.constexpr,
    RANK: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """Load the three predecessor rows directly and emit packed Q/K/V once.

    The producer publishes only 3 * C BF16 halo values. There is no all-gather,
    intermediate convolved projection, or later layout-copy kernel. Independent
    token/channel tiles naturally overlap their remote halo loads with interior
    compute; all cross-rank lifetime ordering remains in the symmetric-memory communicator.
    """
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    channels = tl.program_id(1) * BC + tl.arange(0, BC)
    if RANK > 0:
        previous = tl.load(Peers + RANK - 1).to(tl.pointer_type(tl.bfloat16))
    else:
        previous = History
    acc = tl.full((BT, BC), 0, tl.float32)
    for j in tl.static_range(4):
        source = rows + j - 3
        local = tl.load(
            X + source[:, None] * SX + channels[None, :],
            (rows[:, None] < N) & (source[:, None] >= 0) & (channels[None, :] < C),
            0,
        )
        halo = tl.load(
            previous + channels[None, :] * 3 + (source[:, None] + 3),
            (rows[:, None] < N) & (source[:, None] < 0) & (channels[None, :] < C),
            0,
        )
        x = tl.where(source[:, None] >= 0, local, halo)
        weight = tl.load(W + channels * 4 + j, channels < C, 0)
        # Match causal_conv1d_fn's BF16 multiplicands and ordered accumulation.
        acc += x * weight[None, :]
    acc = acc / (1 + tl.exp(-acc))
    valid = (rows[:, None] < N) & (channels[None, :] < C)
    tl.store(
        Q + rows[:, None] * QD + channels[None, :],
        acc,
        valid & (channels[None, :] < QD),
    )
    tl.store(
        K + rows[:, None] * KD + channels[None, :] - QD,
        acc,
        valid & (channels[None, :] >= QD) & (channels[None, :] < QD + KD),
    )
    tl.store(
        V + rows[:, None] * (C - QD - KD) + channels[None, :] - QD - KD,
        acc,
        valid & (channels[None, :] >= QD + KD),
    )


def warmup_causal_conv1d_peer(cp_rank, channels, projection_stride, bt, bc):
    """Compile both projection strides with the selected peer convolution tactic."""
    for stride in sorted({channels, projection_stride}):
        compiled = _causal_conv1d_peer_qkv_kernel.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.uint64,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            64,
            channels,
            stride,
            2048,
            2048,
            cp_rank,
            bt,
            bc,
            num_warps=4,
            grid=(triton.cdiv(64, bt), triton.cdiv(channels, bc)),
        )
        _ = compiled.run


def supports_causal_conv1d_peer(
    raw, history, weight, *, head_geometry, bias=None, activation="silu"
):
    # Total projection width alone does not identify the Q/K/V layout.
    # The kernel below packs 16 Q/K heads and 64 V heads, each of width 128.
    return (
        head_geometry == (16, 16, 64, 128, 128, 128)
        and raw.ndim == 2
        and raw.shape[1] == 12288
        and raw.shape[0] >= 64
        and raw.shape[0] % 64 == 0
        and raw.dtype == history.dtype == weight.dtype == torch.bfloat16
        and raw.device == history.device == weight.device
        and raw.stride(1) == 1
        and weight.shape == (12288, 4)
        and weight.is_contiguous()
        and history.shape == (12288, 3)
        and history.is_contiguous()
        and bias is None
        and activation == "silu"
    )


def causal_conv1d_peer(
    raw,
    history,
    weight,
    peer_ptrs,
    *,
    rank,
    head_geometry,
    bias=None,
    activation="silu",
    token_tile=8,
    channel_tile=256,
):
    """Convolve equal contiguous shards using a published predecessor halo.

    Peer buffers contain [channels, 3] BF16 values. The caller publishes and
    protects them; halo rows do not become output tokens or update cached state.
    """
    if not supports_causal_conv1d_peer(
        raw,
        history,
        weight,
        head_geometry=head_geometry,
        bias=bias,
        activation=activation,
    ):
        raise ValueError("Unsupported peer convolution shape, layout, or dtype")
    if (
        not raw.is_cuda
        or not 0 <= rank < peer_ptrs.numel()
        or peer_ptrs.dtype != torch.uint64
        or peer_ptrs.device != raw.device
        or not peer_ptrs.is_contiguous()
    ):
        raise ValueError("Invalid convolution peer pointers or rank")
    if token_tile not in (4, 8, 16) or channel_tile not in (128, 256):
        raise ValueError("Unsupported peer convolution tile")
    n = raw.shape[0]
    q = torch.empty((1, n, 16, 128), device=raw.device, dtype=raw.dtype)
    k = torch.empty_like(q)
    v = torch.empty((1, n, 64, 128), device=raw.device, dtype=raw.dtype)
    _causal_conv1d_peer_qkv_kernel[
        (triton.cdiv(n, token_tile), triton.cdiv(raw.shape[1], channel_tile))
    ](
        raw,
        weight,
        history,
        peer_ptrs,
        q,
        k,
        v,
        n,
        raw.shape[1],
        raw.stride(0),
        2048,
        2048,
        rank,
        token_tile,
        channel_tile,
        num_warps=4,
    )
    return [q, k, v]


def warmup_causal_conv1d_context_parallel(
    device, dtype, channels, kernel_size, projection_stride
):
    """Compile contiguous and projected halos, ragged and empty local shards."""
    weights = torch.zeros((channels, kernel_size), device=device, dtype=dtype)
    indices = torch.zeros(1, device=device, dtype=torch.int32)
    initial = torch.ones(1, device=device, dtype=torch.bool)
    for count in (0, 1, 17, 64):
        for stride in sorted({channels, projection_stride}):
            raw = torch.zeros((count, stride), device=device, dtype=dtype)[:, :channels]
            history = torch.zeros(
                (1, channels, kernel_size - 1), device=device, dtype=dtype
            )
            arguments = dict(
                x=raw.T,
                weight=weights,
                bias=None,
                conv_states=history,
                query_start_loc=torch.tensor(
                    [0, count], device=device, dtype=torch.int32
                ),
                seq_lens_cpu=[count],
                cache_indices=indices,
                has_initial_state=initial,
                activation="silu",
            )
            convolved = causal_conv1d_fn(**arguments)
            causal_conv1d_with_halo(
                raw.T,
                history,
                weights,
                query_start_loc=arguments["query_start_loc"],
                seq_lens_cpu=[count],
                has_initial_state=initial,
            )
            # A producer can prepend projected halo rows without a history copy.
            # Its strides differ from the contiguous cached-state layout above.
            halo_width = kernel_size - 1
            projected = torch.zeros(
                (count + halo_width, stride), device=device, dtype=dtype
            )
            causal_conv1d_with_halo(
                projected[halo_width:, :channels].T,
                projected[:halo_width, :channels].T.unsqueeze(0),
                weights,
                query_start_loc=arguments["query_start_loc"],
                seq_lens_cpu=[count],
                has_initial_state=initial,
            )
            if count > 8:
                out = causal_conv1d_fn(
                    **arguments, token_block_range=(1, (count + 7) // 8)
                )
                causal_conv1d_fn(**arguments, out=out, token_block_range=(0, 1))


_CONV_PEER_TACTICS = {}


def causal_conv1d_peer_with_halo(
    raw,
    history,
    weight,
    workspace,
    *,
    head_geometry,
    bias=None,
    activation="silu",
    tuning=None,
):
    """Publish the predecessor halo and return QKV plus independent final history."""
    communicator = workspace.communicator
    if len(workspace.peers) != 4:
        raise ValueError("Peer convolution requires four contiguous shards")
    if not supports_causal_conv1d_peer(
        raw,
        history,
        weight,
        head_geometry=head_geometry,
        bias=bias,
        activation=activation,
    ):
        raise ValueError("Unsupported peer convolution shape, layout, or dtype")
    tuning = tuning or _CONV_PEER_TACTICS.get(
        (raw.device.index, raw.shape[1]), {"bt": 8, "bc": 256}
    )
    communicator.publish(workspace, raw[-3:].T.contiguous())
    rank = communicator.rank
    qkv = causal_conv1d_peer(
        raw,
        history,
        weight,
        workspace.peer_ptrs,
        rank=rank,
        head_geometry=head_geometry,
        bias=bias,
        activation=activation,
        token_tile=tuning["bt"],
        channel_tile=tuning["bc"],
    )
    preceding = history if rank == 0 else workspace.peers[rank - 1]
    return qkv, preceding, workspace.peers[-1].clone()


def autotune_causal_conv1d_peer(raw, history, weight, workspace, *, head_geometry):
    import torch.distributed as dist

    from sglang.srt.layers.cp.utils import select_cp_tactic

    rank, group = workspace.communicator.rank, workspace.communicator.group
    for bt in (4, 8, 16):
        for bc in (128, 256):
            warmup_causal_conv1d_peer(rank, raw.shape[1], raw.stride(0), bt, bc)
    dist.barrier(group=group)
    tuning = select_cp_tactic(
        [{"bt": bt, "bc": bc} for bt in (4, 8, 16) for bc in (128, 256)],
        lambda tactic: causal_conv1d_peer_with_halo(
            raw, history, weight, workspace, head_geometry=head_geometry, tuning=tactic
        ),
        group,
    )
    _CONV_PEER_TACTICS[(raw.device.index, raw.shape[1])] = tuning
    return tuning
