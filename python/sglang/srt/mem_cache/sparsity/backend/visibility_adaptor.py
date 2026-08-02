"""Metadata adaptors for HBM-resident visibility-only sparsity."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache.sparsity.contracts import Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MetadataAdaptor(ABC):
    """Translate logical selections into backend-owned metadata."""

    def bind_attention_backend(self, attention_backend: Any) -> None:
        """Bind optional backend services after attention backend creation."""

    @abstractmethod
    def capture_dense_metadata(self, metadata: Any) -> None: ...

    @abstractmethod
    def apply(
        self,
        result: SelectionResult,
        metadata: Any,
        forward_batch: ForwardBatch,
    ) -> Any: ...

    @abstractmethod
    def restore_dense_metadata(self, metadata: Any) -> None: ...


class HBMResidentPlacement:
    """Map request-relative logical pages to their resident HBM page IDs."""

    def __init__(self, req_to_token: torch.Tensor, page_size: int):
        self.req_to_token = req_to_token
        self.page_size = page_size

    def logical_to_physical_pages(
        self,
        logical_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        if logical_pages.ndim != 2:
            raise ValueError("logical_pages must have shape [batch, capacity]")
        if req_pool_indices.ndim != 1:
            raise ValueError("req_pool_indices must have shape [batch]")
        if logical_pages.shape[0] != req_pool_indices.shape[0]:
            raise ValueError("selection and request batch sizes must match")

        token_positions = (logical_pages * self.page_size).clamp(min=0).long()
        rows = req_pool_indices.long().unsqueeze(1).expand_as(token_positions)
        physical_tokens = self.req_to_token[rows, token_positions]
        physical_pages = torch.div(
            physical_tokens, self.page_size, rounding_mode="floor"
        )
        return torch.where(
            logical_pages >= 0,
            physical_pages,
            torch.zeros_like(physical_pages),
        ).to(torch.int32)


class FlashAttentionVisibilityAdaptor(MetadataAdaptor):
    """Rewrite FA3 page-table metadata without moving or freeing KV data."""

    def __init__(self, placement: HBMResidentPlacement):
        self.placement = placement
        self._dense_page_table = None
        self._dense_cache_seqlens = None
        self._dense_cu_seqlens_k = None
        self._dense_max_seq_len_k = None
        self._dense_scheduler_metadata = None
        self._sparse_scheduler_metadata = None
        self._scheduler_metadata_prepared = False
        self._scheduler_metadata_builder = None

    def bind_attention_backend(self, attention_backend: Any) -> None:
        builder = getattr(attention_backend, "_compute_scheduler_metadata", None)
        if builder is None:
            raise TypeError(
                "FA3 visibility requires an attention backend with scheduler "
                "metadata support"
            )
        self._scheduler_metadata_builder = builder

    def capture_dense_metadata(self, metadata: Any) -> None:
        required = (
            "page_table",
            "cache_seqlens_int32",
            "cu_seqlens_k",
            "max_seq_len_k",
        )
        if metadata is None or not all(hasattr(metadata, name) for name in required):
            raise TypeError("FA3 sparse visibility requires FlashAttention metadata")
        self._dense_page_table = metadata.page_table.clone()
        self._dense_cache_seqlens = metadata.cache_seqlens_int32.clone()
        self._dense_cu_seqlens_k = metadata.cu_seqlens_k.clone()
        self._dense_max_seq_len_k = metadata.max_seq_len_k
        self._dense_scheduler_metadata = getattr(metadata, "scheduler_metadata", None)
        self._sparse_scheduler_metadata = None
        self._scheduler_metadata_prepared = False

    def apply(
        self,
        result: SelectionResult,
        metadata: Any,
        forward_batch: ForwardBatch,
    ) -> Any:
        if result.granularity != Granularity.PAGE:
            raise ValueError("FA3 visibility adaptor requires page selections")
        if self._dense_page_table is None:
            raise RuntimeError("capture_dense_metadata must be called before apply")
        # During warmup or short decode, no row is sparse and FA3 may publish a
        # page table narrower than the policy's fixed capacity. Truncating is
        # safe: if any row were sparse, its sequence would already be wider
        # than the retained capacity and so would the batch page table.
        visible_capacity = min(result.capacity, metadata.page_table.shape[1])

        physical_pages = self.placement.logical_to_physical_pages(
            result.logical_indices[:, :visible_capacity],
            forward_batch.req_pool_indices,
        )
        columns = torch.arange(
            visible_capacity, device=physical_pages.device
        ).unsqueeze(0)
        valid = columns < result.valid_lengths.unsqueeze(1)
        update = result.sparse_mask.unsqueeze(1) & valid

        dense_prefix = self._dense_page_table[:, :visible_capacity]
        metadata.page_table[:, :visible_capacity].copy_(
            torch.where(update, physical_pages, dense_prefix)
        )
        metadata.cache_seqlens_int32.copy_(
            torch.where(
                result.sparse_mask,
                result.visible_kv_lens.to(torch.int32),
                self._dense_cache_seqlens,
            )
        )
        metadata.cu_seqlens_k[0].zero_()
        metadata.cu_seqlens_k[1:].copy_(
            torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
        )
        sparse_max_seq_len_k = self._dense_max_seq_len_k
        if result.max_visible_kv_len is not None:
            sparse_max_seq_len_k = min(sparse_max_seq_len_k, result.max_visible_kv_len)
        metadata.max_seq_len_k = sparse_max_seq_len_k
        if hasattr(metadata, "scheduler_metadata"):
            # A schedule precomputed for dense lengths is invalid. Recompute it
            # once per decode step and reuse it across every sparse layer.
            if (
                not self._scheduler_metadata_prepared
                and self._scheduler_metadata_builder is not None
            ):
                self._sparse_scheduler_metadata = self._scheduler_metadata_builder(
                    metadata.cache_seqlens_int32.shape[0],
                    max(sparse_max_seq_len_k, 1),
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_q,
                )
                self._scheduler_metadata_prepared = True
            metadata.scheduler_metadata = self._sparse_scheduler_metadata
        return metadata

    def restore_dense_metadata(self, metadata: Any) -> None:
        if self._dense_page_table is None:
            return
        metadata.page_table.copy_(self._dense_page_table)
        metadata.cache_seqlens_int32.copy_(self._dense_cache_seqlens)
        metadata.cu_seqlens_k.copy_(self._dense_cu_seqlens_k)
        metadata.max_seq_len_k = self._dense_max_seq_len_k
        if hasattr(metadata, "scheduler_metadata"):
            metadata.scheduler_metadata = self._dense_scheduler_metadata
