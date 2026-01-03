# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import triton
import triton.language as tl


@triton.jit
def rotate_input_ids_kernel(
    input_ids_ptr,
    extend_start_loc_ptr,
    extend_seq_lens_ptr,
    topk_index_ptr,
    select_index_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    start_loc = tl.load(extend_start_loc_ptr + pid)
    seq_len = tl.load(extend_seq_lens_ptr + pid)
    new_token = tl.load(topk_index_ptr + pid)

    num_elements_to_shift = seq_len - 1

    for off in range(0, num_elements_to_shift, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements_to_shift

        read_ptr = input_ids_ptr + start_loc + offsets + 1
        val = tl.load(read_ptr, mask=mask)
        tl.debug_barrier()

        write_ptr = input_ids_ptr + start_loc + offsets
        tl.store(write_ptr, val, mask=mask)
        tl.debug_barrier()

    if seq_len > 0:
        if select_index_ptr is not None:
            last_pos_ptr = input_ids_ptr + tl.load(select_index_ptr + pid)
        else:
            last_pos_ptr = input_ids_ptr + start_loc + seq_len - 1
        tl.store(last_pos_ptr, new_token)


def rotate_input_ids_triton(
    input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index=None
):
    batch_size = extend_seq_lens.shape[0]
    BLOCK_SIZE = 4096 if select_index is not None else 8
    grid = (batch_size,)

    rotate_input_ids_kernel[grid](
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        topk_index,
        select_index,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return input_ids


@triton.jit
def assign_new_state_kernel(
    # Source pointers
    old_input_ids_ptr,
    old_positions_ptr,
    old_hidden_states_ptr,
    old_out_cache_loc_ptr,
    old_extend_seq_lens_ptr,
    old_extend_start_loc_ptr,
    # Destination pointers
    input_ids_ptr,
    positions_ptr,
    hidden_states_ptr,
    out_cache_loc_ptr,
    extend_seq_lens_ptr,
    extend_start_loc_ptr,
    # Auxiliary data pointers
    next_token_ids_ptr,
    seq_lens_ptr,
    padding_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    req_to_hidden_states_pool_ptr,
    # Scalars and Strides
    step,
    stride_hidden_seq,
    stride_hidden_dim,  # hidden_states strides
    stride_pool_req,
    stride_pool_step,
    stride_pool_dim,  # pool strides
    stride_req_token_0,
    stride_req_token_1,  # req_to_token strides
    # Meta-parameters
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)

    seq_len: tl.tensor = tl.load(seq_lens_ptr + pid)
    old_extend_len = tl.load(old_extend_seq_lens_ptr + pid)
    old_start = tl.load(old_extend_start_loc_ptr + pid)
    new_extend_len = old_extend_len + 1
    new_start = old_start + pid

    tl.store(extend_seq_lens_ptr + pid, new_extend_len)
    tl.store(extend_start_loc_ptr + pid, new_start)

    offs_seq = tl.arange(0, BLOCK_SEQ)
    mask_seq = offs_seq < old_extend_len

    old_ids = tl.load(old_input_ids_ptr + old_start + offs_seq, mask=mask_seq)
    tl.store(input_ids_ptr + new_start + offs_seq, old_ids, mask=mask_seq)
    padding_len = tl.load(padding_lens_ptr + pid)
    tl.store(
        input_ids_ptr + new_start + old_extend_len - padding_len,
        tl.load(next_token_ids_ptr + pid),
    )

    old_pos = tl.load(old_positions_ptr + old_start + offs_seq, mask=mask_seq)
    tl.store(positions_ptr + new_start + 1 + offs_seq, old_pos, mask=mask_seq)
    tl.store(
        positions_ptr + new_start, max(tl.load(old_positions_ptr + old_start) - 1, 0)
    )

    old_cache = tl.load(old_out_cache_loc_ptr + old_start + offs_seq, mask=mask_seq)
    tl.store(out_cache_loc_ptr + new_start + 1 + offs_seq, old_cache, mask=mask_seq)

    req_idx = tl.load(req_pool_indices_ptr + pid)
    token_idx_col = seq_len - old_extend_len - 1
    if token_idx_col >= 0:
        req_token_ptr_loc = (
            req_to_token_ptr
            + (req_idx * stride_req_token_0)
            + (token_idx_col * stride_req_token_1)
        )
        last_cache_loc = tl.load(req_token_ptr_loc)
        tl.store(out_cache_loc_ptr + new_start, last_cache_loc)

    pool_vec_offset_base = ((req_idx + 1) * stride_pool_req) + (
        -(step + 1) * stride_pool_step
    )

    for off_h in range(0, HIDDEN_DIM, BLOCK_HID):
        offs_h = off_h + tl.arange(0, BLOCK_HID)
        mask_h = offs_h < HIDDEN_DIM

        for i in range(BLOCK_SEQ):
            if i < old_extend_len:
                old_h_ptr = (
                    old_hidden_states_ptr
                    + (old_start + i) * stride_hidden_seq
                    + (offs_h * stride_hidden_dim)
                )
                new_h_ptr = (
                    hidden_states_ptr
                    + (new_start + 1 + i) * stride_hidden_seq
                    + (offs_h * stride_hidden_dim)
                )

                chunk_old = tl.load(old_h_ptr, mask=mask_h)
                tl.store(new_h_ptr, chunk_old, mask=mask_h)

        pool_ptrs = (
            req_to_hidden_states_pool_ptr
            + pool_vec_offset_base
            + (offs_h * stride_pool_dim)
        )
        pool_val = tl.load(pool_ptrs, mask=mask_h)

        new_h_start_ptrs = (
            hidden_states_ptr
            + (new_start * stride_hidden_seq)
            + (offs_h * stride_hidden_dim)
        )
        tl.store(new_h_start_ptrs, pool_val, mask=mask_h)


def assign_new_state_triton(
    next_token_ids: torch.Tensor,
    old_input_ids: torch.Tensor,
    old_positions: torch.Tensor,
    old_hidden_states: torch.Tensor,
    old_out_cache_loc: torch.Tensor,
    old_extend_seq_lens: torch.Tensor,
    old_extend_start_loc: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    padding_lens: torch.Tensor,
    num_seqs: int,
    step: int,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
):
    """
    Wrapper function to calculate offsets and launch the Triton kernel.
    """
    hidden_dim = hidden_states.shape[1]

    BLOCK_SEQ = 8
    BLOCK_HID = 64

    grid = (num_seqs,)

    assign_new_state_kernel[grid](
        # Pointers
        old_input_ids,
        old_positions,
        old_hidden_states,
        old_out_cache_loc,
        old_extend_seq_lens,
        old_extend_start_loc,
        input_ids,
        positions,
        hidden_states,
        out_cache_loc,
        extend_seq_lens,
        extend_start_loc,
        next_token_ids,
        seq_lens,
        padding_lens,
        req_pool_indices,
        req_to_token,
        req_to_hidden_states_pool,
        # Constants/Strides
        step,
        old_hidden_states.stride(0),
        old_hidden_states.stride(1),
        req_to_hidden_states_pool.stride(0),
        req_to_hidden_states_pool.stride(1),
        req_to_hidden_states_pool.stride(2),
        req_to_token.stride(0),
        req_to_token.stride(1),
        # Meta
        HIDDEN_DIM=hidden_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HID=BLOCK_HID,
    )


@triton.jit
def assign_hidden_states_pool_kernel(
    hidden_states_ptr,
    req_pool_indices_ptr,
    req_to_hidden_states_pool_ptr,
    extend_seq_lens_ptr,
    extend_start_loc_ptr,
    stride_hidden_seq,
    stride_hidden_dim,
    stride_pool_req,
    stride_pool_step,
    stride_pool_dim,
    HIDDEN_DIM: tl.constexpr,
    pool_size: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)

    extend_len = tl.load(extend_seq_lens_ptr + pid)
    start_loc = tl.load(extend_start_loc_ptr + pid)
    end_loc = start_loc + extend_len

    req_idx = tl.load(req_pool_indices_ptr + pid)
    pool_vec_offset_base = req_idx * stride_pool_req

    for i in range(pool_size):
        for off_h in range(0, HIDDEN_DIM, BLOCK_HID):
            offs_h = off_h + tl.arange(0, BLOCK_HID)
            mask_h = offs_h < HIDDEN_DIM

            hid_ptr = (
                hidden_states_ptr
                + (end_loc - pool_size + i) * stride_hidden_seq
                + offs_h * stride_hidden_dim
            )
            hid_val = tl.load(hid_ptr, mask=mask_h)

            pool_ptr = (
                req_to_hidden_states_pool_ptr
                + pool_vec_offset_base
                + i * stride_pool_step
                + offs_h * stride_pool_dim
            )
            tl.store(pool_ptr, hid_val, mask=mask_h)


def assign_hidden_states_pool_triton(
    hidden_states: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
    pool_size: int,
    num_seqs: int,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
):
    grid = (num_seqs,)
    assign_hidden_states_pool_kernel[grid](
        hidden_states,
        req_pool_indices,
        req_to_hidden_states_pool,
        extend_seq_lens,
        extend_start_loc,
        hidden_states.stride(0),
        hidden_states.stride(1),
        req_to_hidden_states_pool.stride(0),
        req_to_hidden_states_pool.stride(1),
        req_to_hidden_states_pool.stride(2),
        HIDDEN_DIM=hidden_states.shape[1],
        pool_size=pool_size,
        BLOCK_HID=64,
    )


def assign_hidden_states_pool_torch(
    hidden_states: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
    pool_size: int,
    num_seqs: int,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
):
    for req in range(num_seqs):
        pool_idx = req_pool_indices[req]
        extend_len = extend_seq_lens[req]
        start_loc = extend_start_loc[req]
        end_loc = start_loc + extend_len
        req_to_hidden_states_pool[pool_idx, :pool_size, :].copy_(
            hidden_states[end_loc - pool_size : end_loc, :]
        )
