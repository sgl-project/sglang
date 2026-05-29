from dataclasses import dataclass

import torch

from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(kw_only=True)
class BailingLinearMetadata(ForwardMetadata):
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    batch_size: int
    has_initial_states: torch.Tensor
    q_lengths: torch.Tensor

    @staticmethod
    def prepare_decode(
        query_start_loc: torch.Tensor,
        mamba_cache_indices: torch.Tensor,
        bs: int,
        seq_lens: torch.Tensor,
    ) -> "BailingLinearMetadata":
        """This path is run during CUDA graph capture, i.e. decode only, so `num_prefills` is 0"""
        return BailingLinearMetadata(
            batch_size=bs,
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            num_decodes=seq_lens.shape[0],
            num_prefills=0,
            num_prefill_tokens=0,
            has_initial_states=torch.ones_like(seq_lens),
            q_lengths=query_start_loc.diff(),
        )

    @classmethod
    def prepare_mixed(
        cls,
        query_start_loc: torch.Tensor,
        mamba_cache_indices: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> "BailingLinearMetadata":
        """This path cannot run with CUDA graph, as it contains extend requests."""
        if forward_batch.extend_num_tokens is None:
            return cls.prepare_decode(
                query_start_loc=query_start_loc,
                mamba_cache_indices=mamba_cache_indices,
                bs=forward_batch.batch_size,
                seq_lens=forward_batch.seq_lens,
            )
        num_prefills = len(forward_batch.extend_seq_lens)
        num_prefill_tokens = forward_batch.extend_num_tokens
        num_decodes = len(forward_batch.seq_lens) - num_prefills
        context_lens_tensor = forward_batch.extend_prefix_lens
        assert context_lens_tensor is not None
        has_initial_states = context_lens_tensor > 0

        query_start_loc = query_start_loc[: num_prefills + 1]

        return BailingLinearMetadata(
            batch_size=forward_batch.batch_size,
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            has_initial_states=has_initial_states,
            q_lengths=query_start_loc.diff(),
        )
