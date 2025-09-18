# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from .constants import PAD_SLOT_ID


@dataclass(kw_only=True)
class Mamba2Metadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    has_initial_states: torch.Tensor
    chunk_size: int
    seq_idx: torch.Tensor
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
    # Need flags to indicate if there are initial states
    # currently we really only support the FlashAttention backend
    has_initial_states = None

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
    query_start_loc = forward_batch.extend_start_loc
    if num_prefills > 0:
        if context_lens_tensor is not None:
            # precompute flag to avoid device syncs later in mamba2 layer
            # forwards
            # prep is only needed for mamba2 ssd prefill processing
            has_initial_states = context_lens_tensor > 0
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

    return Mamba2Metadata(
        has_initial_states=has_initial_states,
        chunk_size=chunk_size,
        seq_idx=seq_idx,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        query_start_loc=query_start_loc,
    )


def update_metadata(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    mamba2_metadata: Mamba2Metadata,
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
