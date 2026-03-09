from dataclasses import dataclass
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

    @dataclass(kw_only=True, frozen=True)
    class MixedMetadata:
        has_initial_states: Optional[torch.Tensor]
        extend_seq_lens_cpu: list[int]

    mixed_metadata: MixedMetadata | None = None
    """`mixed_metadata` is used for extend/mixed requests"""

    @staticmethod
    def prepare_decode(
        forward_metadata: ForwardMetadata,
        seq_lens: torch.Tensor,
    ) -> "Mamba1Metadata":
        """Prepare metadata for decode-only batch (used in CUDA graph path)."""
        return Mamba1Metadata(
            query_start_loc=forward_metadata.query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
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
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            mixed_metadata=cls.MixedMetadata(
                has_initial_states=has_initial_states,
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ),
        )
