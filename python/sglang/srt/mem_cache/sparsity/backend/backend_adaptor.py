import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_QUEST_METADATA_KERNEL_MODULE = (
    "sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata"
)


@lru_cache(maxsize=1)
def _load_quest_metadata_kernel():
    try:
        module_spec = find_spec(_QUEST_METADATA_KERNEL_MODULE)
    except ModuleNotFoundError as exc:
        if not _QUEST_METADATA_KERNEL_MODULE.startswith(f"{exc.name}."):
            raise
        return None
    if module_spec is None:
        return None
    return import_module(_QUEST_METADATA_KERNEL_MODULE)


class BackendAdaptor(ABC):
    """Base class for attention backend adaptors."""

    supports_prepared_metadata = False

    def __init__(self, device: torch.device):
        self.device = device
        self._original_metadata = None

    def save_original_metadata(self, metadata: Any) -> None:
        """Save original metadata in the beginning of the forward pass."""
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
        """
        Adapt attention metadata for sparse KVCache access.

        Transforms sparse retrieval results (logical indices of important KV pages/tokens)
        into backend-specific attention metadata format.

        Returns:
            Modified attention metadata compatible with the backend
        """
        pass


class DSABackendAdaptor(BackendAdaptor):
    """Adaptor for DSA (DeepSeek Sparse Attention) backend."""

    def __init__(
        self,
        device: torch.device,
        req_to_token_pool,
    ):
        super().__init__(device)
        self.req_to_token_pool = req_to_token_pool

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
        """
        Transform logical page indices to physical device indices for DSA backend.
        """
        # TODO: Implement DSA backend adaptor logic
        pass


class FlashAttentionAdaptor(BackendAdaptor):
    """Adaptor for FlashAttention backend."""

    supports_prepared_metadata = True

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._metadata_prepared = False
        self._max_selected = None

    def _reset_forward_state(self) -> None:
        self._metadata_prepared = False
        self._max_selected = None

    def save_original_metadata(self, metadata: Any) -> None:
        self._reset_forward_state()
        required_attrs = (
            "page_table",
            "cache_seqlens_int32",
            "cu_seqlens_k",
            "max_seq_len_k",
        )
        if metadata is None or not all(
            hasattr(metadata, attr) for attr in required_attrs
        ):
            self._original_metadata = None
            return
        self._original_metadata = {
            "cache_seqlens_int32": metadata.cache_seqlens_int32.clone(),
            "max_seq_len_k": metadata.max_seq_len_k,
        }
        if hasattr(metadata, "scheduler_metadata"):
            metadata.scheduler_metadata = None

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
        """
        Adapt FlashAttention metadata for sparse KVCache access.

        Modifies page_table, cache_seqlens, and related metadata to redirect
        FlashAttention to only process selected sparse pages.

        # TODO: Optimize performance
        """
        if self._original_metadata is None:
            return current_metadata

        max_selected = selected_indices.shape[1]
        if not self._metadata_prepared:
            self._max_selected = max_selected
            current_metadata.max_seq_len_k = max(
                self._original_metadata["max_seq_len_k"],
                max_selected * page_size,
            )
            self._metadata_prepared = True
        elif max_selected != self._max_selected:
            raise ValueError("Sparse selection width changed within one forward")

        if kwargs.get("metadata_prepared", False):
            return current_metadata

        kernel_module = _load_quest_metadata_kernel()
        use_metadata_kernel = kernel_module is not None and all(
            tensor.is_cuda
            for tensor in (
                selected_indices,
                valid_lengths,
                sparse_mask,
                forward_batch.seq_lens,
                forward_batch.req_pool_indices,
                req_to_token,
                current_metadata.page_table,
                current_metadata.cache_seqlens_int32,
                current_metadata.cu_seqlens_k,
            )
        )
        if use_metadata_kernel:
            kernel_module.quest_update_flashattention_metadata_(
                selected_indices=selected_indices,
                valid_lengths=valid_lengths,
                sparse_mask=sparse_mask,
                seq_lens=forward_batch.seq_lens,
                req_pool_indices=forward_batch.req_pool_indices,
                req_to_token=req_to_token,
                page_table=current_metadata.page_table,
                cache_seqlens_int32=current_metadata.cache_seqlens_int32,
                cu_seqlens_k=current_metadata.cu_seqlens_k,
                page_size=page_size,
                update_lengths=True,
            )
            return current_metadata

        physical_pages = self._logical_to_physical_pages_batch(
            selected_indices,
            forward_batch.req_pool_indices,
            req_to_token,
            page_size,
        )

        active_sparse_mask = sparse_mask & (valid_lengths > 0)
        valid_mask = torch.arange(max_selected, device=physical_pages.device).unsqueeze(
            0
        ) < valid_lengths.unsqueeze(1)
        page_table_update_mask = active_sparse_mask.unsqueeze(1) & valid_mask

        seq_lens = forward_batch.seq_lens
        positions_in_page = (seq_lens - 1) % page_size
        sparse_seq_lens = (
            valid_lengths * page_size - (page_size - positions_in_page - 1)
        ).to(torch.int32)
        current_metadata.cache_seqlens_int32.copy_(
            torch.where(
                active_sparse_mask,
                sparse_seq_lens,
                self._original_metadata["cache_seqlens_int32"],
            )
        )
        current_metadata.cu_seqlens_k[0].zero_()
        current_metadata.cu_seqlens_k[1:].copy_(
            torch.cumsum(
                current_metadata.cache_seqlens_int32,
                dim=0,
                dtype=torch.int32,
            )
        )

        page_table = current_metadata.page_table[:, :max_selected]
        page_table.copy_(
            torch.where(page_table_update_mask, physical_pages, page_table)
        )
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
