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


from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import is_pin_memory_available


@dataclass(kw_only=True)
class ForwardMetadata:
    query_start_loc: torch.Tensor
    mamba_cache_indices: torch.Tensor
    mamba_cache_indices_gdn: Optional[torch.Tensor] = None
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

    is_target_verify: bool = False
    draft_token_num: int = 1

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
        cu_chunk_seqlens: torch.Tensor
        seq_idx: torch.Tensor
        last_chunk_indices: torch.Tensor

        extend_seq_lens_cpu: list[int]

    mixed_metadata: MixedMetadata | None = None
    """`mixed_metadata` is used for extend/mixed requests"""

    @staticmethod
    def _async_tensor_h2d(data: list[int], device: torch.device) -> torch.Tensor:
        t = torch.tensor(
            data,
            dtype=torch.int32,
            device="cpu",
            pin_memory=is_pin_memory_available(device),
        )
        return t.to(device=device, non_blocking=True)

    @staticmethod
    def _seq_lens_to_logical_chunks(
        extend_seq_lens_cpu: list[int],
        chunk_size: int,
        device: torch.device,
        extend_prefix_lens_cpu: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cu_chunk_seqlens: list[int] = [0]
        seq_idx: list[int] = []
        last_chunk_indices: list[int] = []

        if extend_prefix_lens_cpu is None:
            extend_prefix_lens_cpu = [0] * len(extend_seq_lens_cpu)

        start = 0
        for seq_i, (prefix_len, seq_len) in enumerate(
            zip(extend_prefix_lens_cpu, extend_seq_lens_cpu, strict=True)
        ):
            remaining_len = seq_len
            chunk_start = start
            first_chunk_len = chunk_size - (prefix_len % chunk_size)
            if first_chunk_len == 0:
                first_chunk_len = chunk_size

            while remaining_len > 0:
                chunk_len = min(first_chunk_len, remaining_len)
                seq_idx.append(seq_i)
                chunk_start += chunk_len
                remaining_len -= chunk_len
                first_chunk_len = chunk_size
                cu_chunk_seqlens.append(chunk_start)

            last_chunk_indices.append(len(cu_chunk_seqlens) - 2)
            start += seq_len

        return (
            Mamba2Metadata._async_tensor_h2d(cu_chunk_seqlens, device),
            Mamba2Metadata._async_tensor_h2d(seq_idx, device),
            Mamba2Metadata._async_tensor_h2d(last_chunk_indices, device),
        )

    @staticmethod
    def _to_cpu_len_list(seq_lens: list[int] | torch.Tensor | None) -> list[int]:
        assert seq_lens is not None
        if isinstance(seq_lens, torch.Tensor):
            assert seq_lens.device.type == "cpu"
            return seq_lens.tolist()
        return seq_lens

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
        extend_seq_lens_cpu = cls._to_cpu_len_list(forward_batch.extend_seq_lens_cpu)
        extend_prefix_lens_cpu = cls._to_cpu_len_list(
            forward_batch.extend_prefix_lens_cpu
        )
        # precompute flag to avoid device syncs later
        has_initial_states = context_lens_tensor > 0
        prep_initial_states = any(
            prefix_len > 0 for prefix_len in extend_prefix_lens_cpu
        )

        query_start_loc = forward_metadata.query_start_loc[: num_prefills + 1]
        cu_chunk_seqlens, seq_idx, last_chunk_indices = cls._seq_lens_to_logical_chunks(
            extend_seq_lens_cpu,
            chunk_size,
            query_start_loc.device,
            extend_prefix_lens_cpu,
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
                cu_chunk_seqlens=cu_chunk_seqlens,
                seq_idx=seq_idx,
                last_chunk_indices=last_chunk_indices,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
            ),
        )
