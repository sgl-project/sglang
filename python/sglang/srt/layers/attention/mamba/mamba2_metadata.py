# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from .constants import PAD_SLOT_ID


def _query_start_loc_to_chunk_indices_offsets(
    query_start_loc: torch.Tensor, chunk_size: int, total_seqlens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        query_start_loc (torch.Tensor): 1D tensor of cumulative sequence
            lengths, shape (num_seqs + 1,).
            The first element should be 0. Each entry represents the starting
            index of a sequence in the flattened token array.
        chunk_size (int): The size of each physical mamba chunk
            (number of tokens per chunk).
        total_seqlens (int): The total number of tokens in the batch.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - chunk_indices (torch.Tensor): 1D tensor of indices
                indicating the physical chunk for each logical chunk.
            - chunk_offsets (torch.Tensor): 1D tensor of offsets
                indicating the starting index of each logical chunk within
                its physical chunk.

    This function computes the chunk indices and offsets for the given
    query_start_loc and chunk_size. Both are tensors of integers with length N,
    where N is the number of logical (pseudo) chunks.
    A logical chunk is a sequence of tokens that are all part of the same
    sequence and are all in the same physical mamba chunk.
    In other words, a logical chunk changes every time we cross a sequence
    boundary or a physical mamba chunk boundary.
    Logical chunks are needed to handle batched requests with initial states
    (see _state_passing_fwd and _chunk_scan_fwd).
    The chunk_indices tensor contains the index of the physical chunk for each
    logical chunk.
    The chunk_offsets tensor contains the offset (AKA starting index) of the
    logical chunk in the physical chunk.

    Example:
    query_start_loc = [0, 5, 10]
    chunk_size = 8
    total_seqlens = 10
    -> chunk_indices = [0, 0, 1]
    -> chunk_offsets = [0, 5, 0]

    In this example, we have 2 sequences, each with 5 tokens. The physical
    chunk size is 8 tokens.
    We have three logical chunks:
    - the first logical chunk starts at token 0 in the first physical chunk
        and contains all 5 tokens from the first sequence
    - the second logical chunk starts at token 5 in the first physical chunk
        and contains first 3 tokens from the second sequence
    - the third logical chunk starts at token 0 in the second physical chunk
        and contains the remaining 2 tokens from the second sequence
    """

    cu_seqlens = query_start_loc[1:]  # remove prepended 0

    # outputs will have length expansion of chunks that do not divide
    # chunk_size
    N = math.ceil(total_seqlens / chunk_size) + (cu_seqlens[:-1] % chunk_size > 0).sum()
    chunk_indices = torch.arange(N, dtype=torch.int, device=query_start_loc.device)
    chunk_offsets = torch.zeros((N,), dtype=torch.int, device=query_start_loc.device)

    p = 0  # num of insertions
    for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):

        # if does not divide chunk_size, then there is one chunk insertion
        p += s % chunk_size > 0

        # get the dimensions
        # - the + 1 for _e is to shift the boundary by one chunk
        # - this shifting is not needed if chunk_size divides e
        _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size > 0)

        # adjust indices and offsets
        chunk_indices[_s:_e] -= p
        chunk_offsets[_s] = s % chunk_size

    return chunk_indices, chunk_offsets


@dataclass(kw_only=True)
class Mamba2Metadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    has_initial_states: torch.Tensor
    prep_initial_states: bool

    chunk_size: int
    seq_idx: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    """
    With continuous batching layout of `x` in vLLM, to enable a Triton program
    to handle a request in parallel, two supporting tensors are used
    (batch_ptr, token_chunk_offset_ptr)
    BLOCK_M = the # tokens to be handled by a Triton program
              (can be customized for different hardware)

    nums_dict:
       tracks the data associated with a given value of BLOCK_M
       BLOCK_M = #tokens handled by a Triton program
    cu_seqlen: total tokens per batch
           (used as flag to update other data at each new input)
    batch_ptr: tracks batch-id handled by the Triton program
    token_chunk_offset_ptr: tracks token group_idx handled by the Triton program
           (Triton implementation of causal_conv1d handles parallelism in 3-axes
           - feature-axis
           - batch-axis
           - sequence-axis)
    """
    nums_dict: Optional[dict] = None
    cu_seqlen: Optional[int] = None
    batch_ptr: Optional[torch.tensor] = None
    token_chunk_offset_ptr: Optional[torch.tensor] = None


def prepare_mamba2_metadata(
    forward_batch: ForwardBatch, *, chunk_size: int
) -> Mamba2Metadata:
    """stable metadata across all mamba2 layers in the forward pass"""
    seq_idx = None
    chunk_indices, chunk_offsets = None, None
    # Need flags to indicate if there are initial states
    # currently we really only support the FlashAttention backend
    has_initial_states = None
    prep_initial_states = False

    # Compute seq_idx, chunk_indices and chunk_offsets for prefill only
    # TODO: I think this might be right?
    num_prefills = (
        len(forward_batch.extend_seq_lens)
        if forward_batch.extend_seq_lens is not None
        else 0
    )
    num_prefill_tokens = forward_batch.extend_num_tokens or 0
    num_decode_tokens = forward_batch.seq_lens_sum - num_prefill_tokens
    context_lens_tensor = forward_batch.extend_prefix_lens
    query_start_loc = forward_batch.query_start_loc(
        device=str(forward_batch.seq_lens.device)
    )
    if num_prefills > 0:
        if context_lens_tensor is not None:
            # precompute flag to avoid device syncs later in mamba2 layer
            # forwards
            # prep is only needed for mamba2 ssd prefill processing
            has_initial_states = context_lens_tensor > 0
            prep_initial_states = torch.any(has_initial_states[:num_prefills]).item()
        assert query_start_loc is not None
        query_start_loc = query_start_loc[: num_prefills + 1]
        seq_idx = torch.repeat_interleave(
            torch.arange(
                num_prefills, dtype=torch.int32, device=query_start_loc.device
            ),
            query_start_loc.diff(),
            output_size=num_prefill_tokens,
        )
        seq_idx.unsqueeze_(0)

        # We compute metadata for chunked prefill once at the top level model
        # forward and reuse them in mamba layers. If not needed, they will be
        # ignored inside mamba kernels.
        if prep_initial_states:
            chunk_indices, chunk_offsets = _query_start_loc_to_chunk_indices_offsets(
                query_start_loc, chunk_size, num_prefill_tokens
            )

    return Mamba2Metadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        query_start_loc=query_start_loc,
        has_initial_states=has_initial_states,
        prep_initial_states=prep_initial_states,
        chunk_size=chunk_size,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )


def update_metadata(
    x: torch.Tensor, query_start_loc: torch.Tensor, mamba2_metadata: Mamba2Metadata
):
    """
    this is triggered upon handling a new input at the first layer
    """
    dim, cu_seqlen = x.shape
    mamba2_metadata.cu_seqlen = cu_seqlen
    seqlens = np.diff(query_start_loc.to("cpu"))
    nums_dict = {}  # type: ignore
    for BLOCK_M in [8]:  # cover all BLOCK_M values
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(nums_dict[BLOCK_M]["mlist"])
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if mamba2_metadata.batch_ptr is None:
            # Update default value after class definition
            # mamba2_metadata.MAX_NUM_PROGRAMS *= 2
            mamba2_metadata.batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device="cuda"
            )
            mamba2_metadata.token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device="cuda"
            )
        else:
            if mamba2_metadata.batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                mamba2_metadata.batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                mamba2_metadata.token_chunk_offset_ptr.resize_(  # type: ignore
                    MAX_NUM_PROGRAMS
                ).fill_(PAD_SLOT_ID)

        mamba2_metadata.batch_ptr[0:mlist_len].copy_(mlist)
        mamba2_metadata.token_chunk_offset_ptr[0:mlist_len].copy_(  # type: ignore
            offsetlist
        )
        nums_dict[BLOCK_M]["batch_ptr"] = mamba2_metadata.batch_ptr
        nums_dict[BLOCK_M][
            "token_chunk_offset_ptr"
        ] = mamba2_metadata.token_chunk_offset_ptr  # type: ignore
    mamba2_metadata.nums_dict = nums_dict
    return mamba2_metadata
