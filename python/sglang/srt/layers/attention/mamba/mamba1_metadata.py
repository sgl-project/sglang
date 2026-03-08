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

    # For chunked prefill: which prefill requests have initial states to load
    has_initial_states: Optional[torch.Tensor] = field(default=None)

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
        if forward_batch.extend_num_tokens is None:
            return cls.prepare_decode(forward_metadata, forward_batch.seq_lens)

        num_prefills = len(forward_batch.extend_seq_lens)
        num_prefill_tokens = forward_batch.extend_num_tokens
        num_decodes = len(forward_batch.seq_lens) - num_prefills

        has_initial_states = None
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens
            if prefix_lens.any():
                has_initial_states = prefix_lens > 0

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
        )
