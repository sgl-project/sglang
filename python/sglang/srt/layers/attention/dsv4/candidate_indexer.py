from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, TypeVar

import torch

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
    # [rows] int, one request id per query row, the rows of one request
    # consecutive (verify: its draft tokens); None = every row its own request
    request_ids: Optional[torch.Tensor] = None

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
    """Use the sparse indexer when the installed DeepGEMM provides its APIs."""
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_SPARSE_INDEXER,
    )

    if DEEPGEMM_SPARSE_INDEXER and block_size == 8 and topk_blocks > 0:
        from sglang.srt.layers.attention.dsv4.candidate_deep_gemm import (
            DeepGemmCandidateIndexer,
        )

        return DeepGemmCandidateIndexer(topk_blocks, block_size)
    from sglang.srt.layers.attention.dsv4.candidate_torch import (
        TorchCandidateIndexer,
    )

    return TorchCandidateIndexer(topk_blocks, block_size)
