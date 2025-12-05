# Copyright 2025 SGLang Team
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
# Adapted from https://github.com/vllm-project/vllm/blob/2c58742dff8613a3bd7496f2008ce927e18d38d1/vllm/model_executor/layers/mamba/mamba2_metadata.py


import math
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(kw_only=True)
class ForwardMetadata:
    query_start_loc: torch.Tensor
    mamba_cache_indices: torch.Tensor
    retrieve_next_token: Optional[torch.Tensor] = None
    retrieve_next_sibling: Optional[torch.Tensor] = None
    retrieve_parent_token: Optional[torch.Tensor] = None
    is_target_verify: bool = False
    draft_token_num: int = 1


@dataclass(kw_only=True)
class Mamba2Metadata(ForwardMetadata):
    """stable metadata across all mamba2 layers in the forward pass"""

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int

    @dataclass(kw_only=True, frozen=True)
    class MixedMetadata:
        has_initial_states: torch.Tensor
        prep_initial_states: bool

        chunk_size: int
        seq_idx: torch.Tensor
        chunk_indices: torch.Tensor
        chunk_offsets: torch.Tensor

        extend_seq_lens_cpu: list[int]

    mixed_metadata: MixedMetadata | None = None
    """`mixed_metadata` is used for extend/mixed requests"""

    @staticmethod
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
        N = (
            math.ceil(total_seqlens / chunk_size)
            + (cu_seqlens[:-1] % chunk_size > 0).sum()
        )
        chunk_indices = torch.arange(N, dtype=torch.int, device=query_start_loc.device)
        chunk_offsets = torch.zeros(
            (N,), dtype=torch.int, device=query_start_loc.device
        )

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

    @staticmethod
    def prepare_decode(
        forward_metadata: ForwardMetadata,
        seq_lens: torch.Tensor,
        *,
        is_target_verify: bool,
        draft_token_num: int,
    ) -> "Mamba2Metadata":
        """This path is run during CUDA graph capture, i.e. decode only, so `num_prefills` is 0"""
        return Mamba2Metadata(
            query_start_loc=forward_metadata.query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_decodes=len(seq_lens),
            num_prefills=0,
            num_prefill_tokens=0,
            is_target_verify=is_target_verify,
            draft_token_num=draft_token_num,
        )

    @classmethod
    def prepare_mixed(
        cls,
        forward_metadata: ForwardMetadata,
        chunk_size: int,
        forward_batch: ForwardBatch,
    ) -> "Mamba2Metadata":
        """This path cannot run with CUDA graph, as it contains extend requests."""
        if forward_batch.extend_num_tokens is None:
            draft_token_num = (
                forward_batch.spec_info.draft_token_num
                if forward_batch.spec_info is not None
                else 1
            )
            return cls.prepare_decode(
                forward_metadata,
                forward_batch.seq_lens,
                is_target_verify=forward_batch.forward_mode.is_target_verify(),
                draft_token_num=draft_token_num,
            )
        num_prefills = len(forward_batch.extend_seq_lens)
        num_prefill_tokens = forward_batch.extend_num_tokens
        num_decodes = len(forward_batch.seq_lens) - num_prefills
        context_lens_tensor = forward_batch.extend_prefix_lens
        assert context_lens_tensor is not None
        # precompute flag to avoid device syncs later
        has_initial_states = context_lens_tensor > 0
        prep_initial_states = torch.any(has_initial_states[:num_prefills]).item()

        query_start_loc = forward_metadata.query_start_loc[: num_prefills + 1]
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
        chunk_offsets, chunk_indices = None, None
        if prep_initial_states:
            chunk_indices, chunk_offsets = (
                cls._query_start_loc_to_chunk_indices_offsets(
                    query_start_loc, chunk_size, num_prefill_tokens
                )
            )

        draft_token_num = (
            getattr(forward_batch.spec_info, "draft_token_num", 1)
            if forward_batch.spec_info is not None
            else 1
        )
        return Mamba2Metadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
            mixed_metadata=cls.MixedMetadata(
                has_initial_states=has_initial_states,
                prep_initial_states=prep_initial_states,
                chunk_size=chunk_size,
                seq_idx=seq_idx,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ),
        )
