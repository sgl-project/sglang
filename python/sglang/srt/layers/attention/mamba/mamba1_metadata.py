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
"""Metadata for Mamba1 layers (e.g., Jamba).

Analogous to Mamba2Metadata but simpler since Mamba1 doesn't use
chunked processing or chunk indices.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(kw_only=True)
class Mamba1Metadata(ForwardMetadata):
    """Stable metadata across all Mamba1 layers in the forward pass."""

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int

    # For prefix caching: which prefill requests have initial states to load
    has_initial_states: Optional[torch.Tensor] = field(default=None)
    prep_initial_states: bool = False

    @staticmethod
    def prepare_decode(
        forward_metadata: ForwardMetadata,
        seq_lens: torch.Tensor,
    ) -> "Mamba1Metadata":
        """Prepare metadata for decode-only batch (used in CUDA graph path)."""
        return Mamba1Metadata(
            query_start_loc=forward_metadata.query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=len(seq_lens),
        )

    @classmethod
    def prepare_mixed(
        cls,
        forward_metadata: ForwardMetadata,
        forward_batch: ForwardBatch,
    ) -> "Mamba1Metadata":
        """Prepare metadata for mixed prefill/decode batch."""
        if forward_batch.extend_num_tokens is None:
            return cls.prepare_decode(forward_metadata, forward_batch.seq_lens)

        num_prefills = len(forward_batch.extend_seq_lens)
        num_prefill_tokens = forward_batch.extend_num_tokens
        num_decodes = len(forward_batch.seq_lens) - num_prefills

        # Compute has_initial_states for prefix caching
        has_initial_states = None
        prep_initial_states = False
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens
            if prefix_lens.any():
                has_initial_states = (prefix_lens > 0)
                prep_initial_states = True

        return Mamba1Metadata(
            query_start_loc=forward_metadata.query_start_loc[: num_prefills + 1],
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            has_initial_states=has_initial_states,
            prep_initial_states=prep_initial_states,
        )
