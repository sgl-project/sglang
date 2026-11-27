# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional

import msgspec
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.linear.kernels.state_pool_io import PoolStaging


class Mamba2MISMetadata(msgspec.Struct, frozen=True, kw_only=True):
    """Compact query/item segment descriptors for Mamba-2 multi-item scoring."""

    chunk_size: int
    reset_cache_indices: torch.Tensor
    query_token_indices: torch.Tensor
    query_cu_seqlens: torch.Tensor
    query_seq_lens_cpu: list[int]
    query_seq_idx: torch.Tensor
    query_cache_indices: torch.Tensor
    query_cache_indices_i64: torch.Tensor
    query_has_initial_states: torch.Tensor
    item_token_indices: torch.Tensor
    item_cu_seqlens: torch.Tensor
    item_seq_lens_cpu: list[int]
    item_seq_idx: torch.Tensor
    item_chunk_indices: Optional[torch.Tensor]
    item_chunk_offsets: Optional[torch.Tensor]
    item_ssm_indices_i64: torch.Tensor
    item_conv_indices: torch.Tensor
    item_has_initial_states: torch.Tensor


class ForwardStateSlots(NamedTuple):
    """A forward's per-request state slots, resolved during its metadata preparation.
    Work past its shared-read end uses these: the scheduler may rewrite the tables."""

    working: torch.Tensor  # one slot per request
    track: Optional[torch.Tensor] = None  # prefix-cache checkpoint slots
    # A request-scoped ReplaySSM ring's row per request: the request row, or its
    # request sub-pool unit under unified memory.
    ring: Optional[torch.Tensor] = None


@dataclass(kw_only=True)
class ForwardMetadata:
    query_start_loc: torch.Tensor
    # int64 copy of query_start_loc for extend kernels that need it; None otherwise.
    query_start_loc_i64: Optional[torch.Tensor] = None
    mamba_cache_indices: torch.Tensor
    conv_block_table: Optional[torch.Tensor] = None
    logical_num_tokens: Optional[int] = None
    mamba_cache_indices_gdn: Optional[torch.Tensor] = None
    # Mamba track DESTINATION slots (PHYSICAL, length == batch). Like
    # mamba_cache_indices: a backend-owned static buffer under cuda-graph (translated
    # in-place each replay), eager sets the translated decode tensor. The decode
    # track-save reads THIS, never forward_batch.mamba_track_indices.
    mamba_track_indices: Optional[torch.Tensor] = None
    # GDN ReplaySSM (slice 1a): per-decode-row snapshot of the ring write
    # cursor for THIS decode step (gathered from the persistent per-slot
    # buffer, then advanced once for the next step). int32, length == batch.
    replayssm_write_pos: Optional[torch.Tensor] = None
    # GDN ReplaySSM (slice 2b): per-decode-row int32 flush flag for THIS decode
    # step. !=0 forces the kernel to fold the partial ring + current token into
    # the checkpoint (temporal[slot]) so the radix cache reads an up-to-date
    # state. Fires on EXACTLY the rows the radix track snapshots, i.e. the same
    # condition the track uses: seq_lens_cpu % mamba_track_interval == 0.
    replayssm_force_flush: Optional[torch.Tensor] = None
    # KDA host-only ReplaySSM: ring rows are live requests (req_pool_idx),
    # not mamba slots. None keeps the ring indexed by mamba_cache_indices.
    replayssm_ring_indices: Optional[torch.Tensor] = None
    # For topk > 1 eagle
    retrieve_next_token: Optional[torch.Tensor] = None
    retrieve_next_sibling: Optional[torch.Tensor] = None
    retrieve_parent_token: Optional[torch.Tensor] = None
    # For prefill radix cache
    track_conv_indices: Optional[torch.Tensor] = None
    track_ssm_h_src: Optional[torch.Tensor] = None
    track_ssm_h_dst: Optional[torch.Tensor] = None
    track_ssm_final_src: Optional[torch.Tensor] = None
    track_ssm_final_dst: Optional[torch.Tensor] = None
    track_chunk_idx: Optional[torch.Tensor] = None
    # Batch rows of the chunk-unaligned tracked seqs; indexes the fp32
    # h_track_buf snapshot (KDA path) with plain integer indexing, so the
    # copy into the track slots does not nonzero()-sync the stream.
    track_ssm_h_batch_src: Optional[torch.Tensor] = None
    state_checkpoint_cu_starts: Optional[torch.Tensor] = None
    num_state_checkpoints: int = 0
    state_checkpoint_every_n_tokens: int = 0
    state_checkpoint_indices: Optional[torch.Tensor] = None
    track_ssm_seq_idx: Optional[torch.Tensor] = None
    track_ssm_end_locs: Optional[torch.Tensor] = None
    track_ssm_recompute_dst: Optional[torch.Tensor] = None

    is_target_verify: bool = False
    draft_token_num: int = 1

    # KDA fused-accept: the [N, T] slot-indexed scratch rows and the per-request
    # accept length that seed the verify kernel. Every KDA layer of a forward
    # sees the same slots and draft window, so these are built once and shared:
    # a cuda-graph capture then holds one build instead of one per layer.
    fused_accept_state_indices: Optional[torch.Tensor] = None
    fused_accept_num_accepted: Optional[torch.Tensor] = None
    # GDN prefill on packed copies of a strided state pool: built by the first
    # layer that stages and shared by the rest of the forward.
    prefill_state_staging: Optional["PoolStaging"] = None

    has_mamba_track_mask: bool = False
    mamba_track_mask_indices: Optional[torch.Tensor] = None
    conv_states_mask_indices: Optional[torch.Tensor] = None


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
    mis_metadata: Mamba2MISMetadata | None = None

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
            mamba_track_indices=forward_metadata.mamba_track_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            track_conv_indices=forward_metadata.track_conv_indices,
            track_ssm_h_src=forward_metadata.track_ssm_h_src,
            track_ssm_h_dst=forward_metadata.track_ssm_h_dst,
            track_ssm_final_src=forward_metadata.track_ssm_final_src,
            track_ssm_final_dst=forward_metadata.track_ssm_final_dst,
            track_ssm_seq_idx=forward_metadata.track_ssm_seq_idx,
            track_ssm_end_locs=forward_metadata.track_ssm_end_locs,
            track_ssm_recompute_dst=forward_metadata.track_ssm_recompute_dst,
            has_mamba_track_mask=forward_metadata.has_mamba_track_mask,
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
        if (
            forward_batch.extend_num_tokens is None
            or forward_batch.logical_forward_mode.is_decode()
        ):
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
        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        if extend_seq_lens_cpu is None:
            num_prefills = len(forward_batch.extend_seq_lens)
        else:
            num_prefills = len(extend_seq_lens_cpu)
        if extend_seq_lens_cpu is not None:
            num_prefill_tokens = int(sum(extend_seq_lens_cpu))
        else:
            num_prefill_tokens = int(forward_batch.extend_num_tokens)
        num_decodes = len(forward_batch.seq_lens) - num_prefills
        context_lens_tensor = forward_batch.extend_prefix_lens
        assert context_lens_tensor is not None
        has_initial_states = context_lens_tensor > 0
        prep_initial_states = torch.any(has_initial_states[:num_prefills]).item()

        query_start_loc = forward_metadata.query_start_loc[: num_prefills + 1]
        _seq_idx_output_size = (
            num_prefill_tokens if extend_seq_lens_cpu is not None else None
        )
        seq_idx = torch.repeat_interleave(
            torch.arange(
                num_prefills, dtype=torch.int32, device=query_start_loc.device
            ),
            query_start_loc.diff(),
            output_size=_seq_idx_output_size,
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
        # Resolve the tracked-row selection once per forward
        mamba_track_mask_indices = None
        conv_states_mask_indices = None
        if forward_metadata.has_mamba_track_mask:
            mamba_track_mask_indices = forward_batch.mamba_track_mask.nonzero(
                as_tuple=True
            )[0]
            conv_states_mask_indices = forward_batch.mamba_track_indices[
                mamba_track_mask_indices
            ]
        return Mamba2Metadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            mamba_track_indices=forward_metadata.mamba_track_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            track_conv_indices=forward_metadata.track_conv_indices,
            track_ssm_h_src=forward_metadata.track_ssm_h_src,
            track_ssm_h_dst=forward_metadata.track_ssm_h_dst,
            track_ssm_final_src=forward_metadata.track_ssm_final_src,
            track_ssm_final_dst=forward_metadata.track_ssm_final_dst,
            track_ssm_seq_idx=forward_metadata.track_ssm_seq_idx,
            track_ssm_end_locs=forward_metadata.track_ssm_end_locs,
            track_ssm_recompute_dst=forward_metadata.track_ssm_recompute_dst,
            has_mamba_track_mask=forward_metadata.has_mamba_track_mask,
            mamba_track_mask_indices=mamba_track_mask_indices,
            conv_states_mask_indices=conv_states_mask_indices,
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
                extend_seq_lens_cpu=extend_seq_lens_cpu,
            ),
        )

    @classmethod
    def prepare_mis(
        cls,
        *,
        cache_indices: torch.Tensor,
        replayssm_ring_indices: Optional[torch.Tensor],
        extend_prefix_lens_cpu: Optional[Sequence[int] | torch.Tensor],
        extend_seq_lens_cpu: Optional[Sequence[int] | torch.Tensor],
        multi_item_delimiter_indices: Optional[Sequence[torch.Tensor]],
        num_input_tokens: int,
        chunk_size: int,
        device: torch.device,
        is_prefill_only: bool = True,
        is_extend: bool = True,
        is_target_verify: bool = False,
        mamba_prefill_track_mask_cpu: Optional[Sequence[bool]] = None,
        has_mamba_track_mask: bool = False,
        has_mamba_track_indices: bool = False,
    ) -> "Mamba2Metadata":
        """Build prefill-only MIS metadata without touching prepare_mixed sync paths."""
        if (
            (
                mamba_prefill_track_mask_cpu is not None
                and any(mamba_prefill_track_mask_cpu)
            )
            or (mamba_prefill_track_mask_cpu is None and has_mamba_track_mask)
            or has_mamba_track_indices
        ):
            raise ValueError(
                "Mamba-2 multi-item scoring requires --disable-radix-cache"
            )
        if not is_prefill_only or not is_extend or is_target_verify:
            raise ValueError("Mamba2 MIS is only supported for prefill-only requests")
        if extend_seq_lens_cpu is None or extend_prefix_lens_cpu is None:
            raise ValueError(
                "Mamba2 MIS requires extend_seq_lens_cpu and extend_prefix_lens_cpu"
            )

        seq_lens = _to_int_list(values=extend_seq_lens_cpu)
        num_prefills = len(seq_lens)
        prefill_cache_indices = cache_indices[:num_prefills]
        prefill_ring_indices = (
            None
            if replayssm_ring_indices is None
            else replayssm_ring_indices[:num_prefills]
        )
        mis_metadata = build_mamba2_mis_metadata(
            is_prefill_only=is_prefill_only,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=seq_lens,
            multi_item_delimiter_indices=multi_item_delimiter_indices,
            num_input_tokens=num_input_tokens,
            cache_indices=prefill_cache_indices,
            chunk_size=chunk_size,
            device=device,
        )
        return cls(
            query_start_loc=torch.empty(0, dtype=torch.int32, device=device),
            logical_num_tokens=sum(seq_lens),
            mamba_cache_indices=prefill_cache_indices,
            replayssm_ring_indices=prefill_ring_indices,
            num_prefills=num_prefills,
            num_prefill_tokens=sum(seq_lens),
            num_decodes=0,
            mis_metadata=mis_metadata,
        )


def build_mamba2_mis_metadata(
    *,
    is_prefill_only: bool,
    extend_prefix_lens_cpu: Sequence[int] | torch.Tensor,
    extend_seq_lens_cpu: Sequence[int] | torch.Tensor,
    multi_item_delimiter_indices: Optional[Sequence[torch.Tensor]],
    num_input_tokens: int,
    cache_indices: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> Mamba2MISMetadata:
    """Build compact query and item segments from request-local MIS delimiters."""
    seq_lens, validated_delimiters = _validate_mis_batch_inputs(
        is_prefill_only=is_prefill_only,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        multi_item_delimiter_indices=multi_item_delimiter_indices,
        num_input_tokens=num_input_tokens,
        cache_indices=cache_indices,
        chunk_size=chunk_size,
        device=device,
    )
    (
        query_token_indices,
        query_seq_lens_cpu,
        query_seq_idx_flat,
        query_request_indices,
        item_token_indices,
        item_seq_lens_cpu,
        item_seq_idx_flat,
        item_request_indices,
    ) = _collect_mis_segments(
        seq_lens=seq_lens,
        multi_item_delimiter_indices=validated_delimiters,
    )

    query_cu_seqlens_cpu = _cu_seqlens_cpu(lengths=query_seq_lens_cpu)
    item_cu_seqlens_cpu = _cu_seqlens_cpu(lengths=item_seq_lens_cpu)
    if item_token_indices:
        chunk_idx_cpu, chunk_off_cpu = (
            Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(
                query_start_loc=item_cu_seqlens_cpu,
                chunk_size=chunk_size,
                total_seqlens=len(item_token_indices),
            )
        )
        item_chunk_indices = chunk_idx_cpu.to(device=device, non_blocking=True)
        item_chunk_offsets = chunk_off_cpu.to(device=device, non_blocking=True)
    else:
        item_chunk_indices = None
        item_chunk_offsets = None

    # Clone so later caller mutations to cache_indices cannot alias batch metadata.
    reset_cache_indices = cache_indices.to(dtype=torch.int64).clone()
    query_req_t = _to_device_non_blocking(
        values=query_request_indices, dtype=torch.int64, device=device
    )
    item_req_t = _to_device_non_blocking(
        values=item_request_indices, dtype=torch.int64, device=device
    )
    query_cache_i64 = reset_cache_indices.index_select(0, query_req_t)
    item_ssm_i64 = reset_cache_indices.index_select(0, item_req_t)

    return Mamba2MISMetadata(
        chunk_size=chunk_size,
        reset_cache_indices=reset_cache_indices,
        query_token_indices=_to_device_non_blocking(
            values=query_token_indices, dtype=torch.int64, device=device
        ),
        query_cu_seqlens=query_cu_seqlens_cpu.to(device=device, non_blocking=True),
        query_seq_lens_cpu=query_seq_lens_cpu,
        query_seq_idx=_to_device_non_blocking(
            values=query_seq_idx_flat, dtype=torch.int32, device=device
        ).unsqueeze(0),
        query_cache_indices=query_cache_i64.to(dtype=torch.int32),
        query_cache_indices_i64=query_cache_i64,
        query_has_initial_states=torch.zeros(
            len(query_seq_lens_cpu), dtype=torch.bool, device=device
        ),
        item_token_indices=_to_device_non_blocking(
            values=item_token_indices, dtype=torch.int64, device=device
        ),
        item_cu_seqlens=item_cu_seqlens_cpu.to(device=device, non_blocking=True),
        item_seq_lens_cpu=item_seq_lens_cpu,
        item_seq_idx=_to_device_non_blocking(
            values=item_seq_idx_flat, dtype=torch.int32, device=device
        ).unsqueeze(0),
        item_chunk_indices=item_chunk_indices,
        item_chunk_offsets=item_chunk_offsets,
        item_ssm_indices_i64=item_ssm_i64,
        item_conv_indices=torch.arange(
            len(item_seq_lens_cpu), dtype=torch.int32, device=device
        ),
        item_has_initial_states=torch.ones(
            len(item_seq_lens_cpu), dtype=torch.bool, device=device
        ),
    )


def _validate_mis_batch_inputs(
    *,
    is_prefill_only: bool,
    extend_prefix_lens_cpu: Sequence[int] | torch.Tensor,
    extend_seq_lens_cpu: Sequence[int] | torch.Tensor,
    multi_item_delimiter_indices: Optional[Sequence[torch.Tensor]],
    num_input_tokens: int,
    cache_indices: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> tuple[list[int], Sequence[torch.Tensor]]:
    if not is_prefill_only:
        raise ValueError("Mamba2 MIS is only supported for prefill-only requests")

    prefix_lens = _to_int_list(values=extend_prefix_lens_cpu)
    if any(prefix_len != 0 for prefix_len in prefix_lens):
        raise ValueError("Mamba2 MIS does not support cached prefixes")

    seq_lens = _to_int_list(values=extend_seq_lens_cpu)
    if multi_item_delimiter_indices is None or len(multi_item_delimiter_indices) != len(
        seq_lens
    ):
        raise ValueError("Mamba2 MIS requires delimiter indices for every request")
    if sum(seq_lens) > num_input_tokens:
        raise ValueError("Mamba2 MIS sequence lengths exceed the input tokens")
    if chunk_size <= 0:
        raise ValueError("Mamba2 MIS requires a positive chunk_size")
    if (
        cache_indices.is_meta
        or cache_indices.ndim != 1
        or cache_indices.numel() != len(seq_lens)
        or cache_indices.device != device
        or cache_indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Mamba2 MIS requires 1D integer cache_indices on the batch device for every request"
        )
    return seq_lens, multi_item_delimiter_indices


def _validate_mis_delimiters(
    *, seq_len: int, request_delimiters: torch.Tensor
) -> list[int]:
    delimiters = [int(index) for index in request_delimiters.tolist()]
    if len(delimiters) < 2:
        raise ValueError("Mamba2 MIS requires at least two delimiters per request")
    if any(
        current >= following for current, following in zip(delimiters, delimiters[1:])
    ):
        raise ValueError("Mamba2 MIS delimiter indices must be strictly increasing")
    if delimiters[0] < 0 or delimiters[-1] >= seq_len:
        raise ValueError("Mamba2 MIS delimiter index is outside the request")
    if delimiters[-1] != seq_len - 1:
        raise ValueError("Mamba2 MIS final delimiter must be the last request token")
    if any(
        following - current == 1
        for current, following in zip(delimiters, delimiters[1:])
    ):
        raise ValueError("Mamba2 MIS does not support empty items")
    return delimiters


def _collect_mis_segments(
    *,
    seq_lens: list[int],
    multi_item_delimiter_indices: Sequence[torch.Tensor],
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
]:
    query_token_indices: list[int] = []
    query_seq_lens_cpu: list[int] = []
    query_seq_idx_flat: list[int] = []
    query_request_indices: list[int] = []
    item_token_indices: list[int] = []
    item_seq_lens_cpu: list[int] = []
    item_seq_idx_flat: list[int] = []
    item_request_indices: list[int] = []

    request_start = 0
    for request_idx, (seq_len, request_delimiters) in enumerate(
        zip(seq_lens, multi_item_delimiter_indices)
    ):
        delimiters = _validate_mis_delimiters(
            seq_len=seq_len, request_delimiters=request_delimiters
        )
        query_len = delimiters[0]
        if query_len > 0:
            q_seg_idx = len(query_seq_lens_cpu)
            query_token_indices.extend(range(request_start, request_start + query_len))
            query_seq_lens_cpu.append(query_len)
            query_seq_idx_flat.extend([q_seg_idx] * query_len)
            query_request_indices.append(request_idx)

        # Skip delimiter tokens themselves; items start right after each delimiter
        # and stop before the next delimiter (leaving the trailing delimiter excluded).
        for branch_start, branch_end in zip(delimiters, delimiters[1:]):
            item_start = branch_start + 1
            item_len = branch_end - item_start
            assert item_len > 0
            it_seg_idx = len(item_seq_lens_cpu)
            item_token_indices.extend(
                range(request_start + item_start, request_start + branch_end)
            )
            item_seq_lens_cpu.append(item_len)
            item_seq_idx_flat.extend([it_seg_idx] * item_len)
            item_request_indices.append(request_idx)

        request_start += seq_len

    return (
        query_token_indices,
        query_seq_lens_cpu,
        query_seq_idx_flat,
        query_request_indices,
        item_token_indices,
        item_seq_lens_cpu,
        item_seq_idx_flat,
        item_request_indices,
    )


def _cu_seqlens_cpu(*, lengths: list[int]) -> torch.Tensor:
    if not lengths:
        return torch.zeros(1, dtype=torch.int32)
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)


def _to_int_list(*, values: Sequence[int] | torch.Tensor) -> list[int]:
    raw_values = values.tolist() if isinstance(values, torch.Tensor) else values
    return [int(value) for value in raw_values]


def _to_device_non_blocking(
    *, values: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).to(device=device, non_blocking=True)
