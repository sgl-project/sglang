import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import nvtx
import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class BackendAdaptor(ABC):
    """Base class for attention backend adaptors."""

    def __init__(self, device: torch.device, sparse_mode):
        self.device = device
        self.sparse_mode = sparse_mode
        self._original_metadata = None

    def save_original_metadata(self, metadata: Any) -> None:
        """Save original metadata (called once at layer 0)."""
        pass

    @abstractmethod
    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Any:
        """Adapt attention metadata for sparse attention."""
        pass


class NSABackendAdaptor(BackendAdaptor):
    """Adaptor for NSA (Native Sparse Attention) backend."""

    def __init__(
        self,
        device: torch.device,
        sparse_mode,
        req_to_token_pool,
        kvcache_manager,
    ):
        super().__init__(device, sparse_mode)
        self.req_to_token_pool = req_to_token_pool
        self.kvcache_manager = kvcache_manager

    @nvtx.annotate("NSABackendAdaptor.adapt_for_attn_metadata", color="green")
    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Transform NSA topk indices to physical device indices."""
        req_pool_indices = forward_batch.req_pool_indices
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
        page_table = self.req_to_token_pool.req_to_token[:, :max_seqlen_k]
        transformed_indices = (
            self.kvcache_manager.transfer_sparse_top_k_cache_v2(
                req_pool_indices=req_pool_indices,
                top_k_result=selected_indices,
                out_cache_loc=forward_batch.out_cache_loc,
                seq_lens=forward_batch.seq_lens,
                sparse_mask=sparse_mask,
                page_table=page_table,
                layer_id=layer_id,
                page_size=1,
            )
            .detach()
            .clone()
        ).to(torch.int32)
        return transformed_indices


class FlashAttentionAdaptor(BackendAdaptor):
    """Adaptor for FlashAttention backend."""

    def save_original_metadata(self, metadata: Any) -> None:
        self._original_metadata = {
            "page_table": metadata.page_table.clone(),
            "cache_seqlens_int32": metadata.cache_seqlens_int32.clone(),
            "cu_seqlens_k": metadata.cu_seqlens_k.clone(),
            "max_seq_len_k": metadata.max_seq_len_k,
        }

    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Any:
        from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import SparseMode

        if self.sparse_mode == SparseMode.PAGE_WISE:
            return self._adapt_for_page_wise(
                selected_indices,
                valid_lengths,
                sparse_mask,
                current_metadata,
                forward_batch,
                req_to_token,
                page_size,
                layer_id,
            )
        elif self.sparse_mode == SparseMode.TOKEN_WISE:
            return self._adapt_for_token_wise(
                selected_indices,
                valid_lengths,
                sparse_mask,
                current_metadata,
                forward_batch,
                req_to_token,
                page_size,
                layer_id,
            )
        else:
            return current_metadata

    def _adapt_for_page_wise(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
    ) -> Any:
        if self._original_metadata is None:
            return current_metadata

        if not sparse_mask.any():
            return current_metadata

        current_metadata.page_table.copy_(self._original_metadata["page_table"])
        current_metadata.cache_seqlens_int32.copy_(
            self._original_metadata["cache_seqlens_int32"]
        )

        physical_pages = self._logical_to_physical_pages_batch(
            selected_indices,
            forward_batch.req_pool_indices,
            req_to_token,
            page_size,
        )

        # Only update valid pages based on valid_lengths
        max_selected = physical_pages.shape[1]
        valid_mask = torch.arange(max_selected, device=physical_pages.device).unsqueeze(
            0
        ) < valid_lengths.unsqueeze(1)
        update_mask = sparse_mask.unsqueeze(1) & valid_mask

        current_metadata.page_table[:, :max_selected] = torch.where(
            update_mask, physical_pages, current_metadata.page_table[:, :max_selected]
        )

        seq_lens = forward_batch.seq_lens
        positions_in_page = (seq_lens - 1) % page_size
        diff = page_size - positions_in_page - 1
        sparse_seq_lens = (valid_lengths * page_size - diff).to(torch.int32)

        if layer_id == 0 and sparse_mask.any():
            logger.info(
                f"[DEBUG] adapt_for_page_wise called: layer_id={layer_id}, original_seq={seq_lens}, sparse_seq={sparse_seq_lens}, page_table={current_metadata.page_table}"
            )

        current_metadata.cache_seqlens_int32 = torch.where(
            sparse_mask, sparse_seq_lens, self._original_metadata["cache_seqlens_int32"]
        )

        current_metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(
                current_metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
            ),
            (1, 0),
        )
        current_metadata.max_seq_len_k = int(current_metadata.cache_seqlens_int32.max())

        return current_metadata

    def _adapt_for_token_wise(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
    ) -> Any:
        return current_metadata

    def _logical_to_physical_pages_batch(
        self,
        logical_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        bs, max_pages = logical_pages.shape

        page_starts = logical_pages * page_size
        page_starts_clamped = page_starts.clamp(min=0)

        req_indices_expanded = req_pool_indices.unsqueeze(1).expand(-1, max_pages)
        first_tokens = req_to_token[req_indices_expanded, page_starts_clamped]

        physical_pages = first_tokens // page_size
        physical_pages = torch.where(
            logical_pages >= 0, physical_pages, torch.zeros_like(physical_pages)
        )

        return physical_pages.to(torch.int32)
