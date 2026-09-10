from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, TypeVar

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata


class CandidateMetadata:
    """Base of an implementation's published state on
    ``DSV4Metadata.candidate_metadata``."""


@dataclass(frozen=True)
class IndexerInputs:
    """One index-source layer's operands on the paged fp4 decode path (one query
    row per request, or per draft token under verify)."""

    q_fp4: torch.Tensor  # [rows, 1, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, 1, heads] int32, packed ue8m0
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool
    weights: torch.Tensor  # [rows, heads] bf16/fp32 head weights
    metadata: PagedIndexerMetadata  # this ratio's lengths, page table and plans

    @property
    def num_rows(self) -> int:
        return self.q_fp4.shape[0]


T = TypeVar("T", bound=CandidateMetadata)


# TODO(dark): support publish prefill/select prefill
# TODO(dark): support fusion of publish + topk of publish layer
class CandidateIndexer(Protocol, Generic[T]):
    def publish_decode(
        self,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> T: ...
    def select_decode(
        self,
        candidate_metadata: T,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> None: ...


def make_candidate_indexer(topk_blocks: int, block_size: int) -> CandidateIndexer:
    """Decided once, at backend init: DeepGEMM's sparse indexer when
    ``SGLANG_DSV41_DEEP_GEMM_CANDIDATE_INDEXER`` is set (the user vouches for the
    kernel being there), the torch algorithm otherwise. A model without a
    candidate source gets the torch one too; it is never asked to publish or
    select."""
    from sglang.srt.layers.attention.dsv4.candidate_deep_gemm import (
        DeepGemmCandidateIndexer,
    )
    from sglang.srt.layers.attention.dsv4.candidate_torch import (
        TorchCandidateIndexer,
    )

    if envs.SGLANG_DSV41_DEEP_GEMM_CANDIDATE_INDEXER.get():
        return DeepGemmCandidateIndexer(topk_blocks, block_size)
    return TorchCandidateIndexer(topk_blocks, block_size)
