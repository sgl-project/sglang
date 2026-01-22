import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch
import triton
from sglang.srt.utils.common import is_cuda, is_hip
from sglang.srt.mem_cache.sparsity.kernel.flashattn_metadata_kernels import (
    update_page_table_triton,
    compute_sparse_seqlens_triton,
)

from sgl_kernel import invoke_sparse_diff_cuda_kernel, update_sparse_metadata


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class BackendAdaptor(ABC):
    """Base class for attention backend adaptors."""

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


class NSABackendAdaptor(BackendAdaptor):
    """Adaptor for NSA (Native Sparse Attention) backend."""

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
        Transform logical page indices to physical device indices for NSA backend.
        """
        # TODO: Implement NSA backend adaptor logic
        pass


class FlashAttentionAdaptor(BackendAdaptor):
    """Adaptor for FlashAttention backend."""

    def __init__(
        self, device: torch.device, req_to_token_pool, sparse_kv_cache_manager
    ):
        super().__init__(device)
        self.req_to_token_pool = req_to_token_pool
        self.sparse_kv_cache_manager = sparse_kv_cache_manager

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
        """
        Adapt FlashAttention metadata for sparse KVCache access.

        Modifies page_table, cache_seqlens, and related metadata to redirect
        FlashAttention to only process selected sparse pages.

        # TODO: Optimize performance
        """
        if self._original_metadata is None:
            return current_metadata

        if not sparse_mask.any():
            return current_metadata

        req_states = self.sparse_kv_cache_manager.req_states
        batch_size = sparse_mask.shape[0]

        topk_tokens_cnt = req_states.topk_tokens_cnt
        topk_pages = topk_tokens_cnt // page_size
        physical_pages = req_states.curr_device_indices[:batch_size, :topk_pages].contiguous()
        topk_page_indices = selected_indices[:, :topk_pages].to(torch.int32).contiguous()

        invoke_sparse_diff_cuda_kernel(
            current_metadata.page_table,
            req_states.last_top_k_result.contiguous(),
            req_states.last_device_indices.contiguous(),
            topk_page_indices,
            forward_batch.req_pool_indices.to(torch.int32).contiguous(),
            forward_batch.seq_lens.to(torch.int32).contiguous(),
            valid_lengths.to(torch.int32).contiguous(),
            sparse_mask.to(torch.int32).contiguous(),
            req_states.req_to_tokens_host,
            physical_pages,
            req_states.should_load_device_indices,
            req_states.should_load_host_indices,
            current_metadata.cache_seqlens_int32,
            self._original_metadata["cache_seqlens_int32"],
            layer_id,
            page_size
        )

        # Data Loading
        swap_target_device_slots = req_states.should_load_device_indices[:batch_size, :topk_tokens_cnt]
        swap_source_host_slots = req_states.should_load_host_indices[:batch_size, :topk_tokens_cnt]

        flat_target = swap_target_device_slots.reshape(-1)
        flat_source = swap_source_host_slots.reshape(-1)
        valid_pos = torch.nonzero(
            flat_target.ne(-1) & flat_source.ne(-1), as_tuple=False
        ).squeeze(1)

        if valid_pos.numel() > 0:
            target_valid = flat_target.index_select(0, valid_pos)
            source_valid = flat_source.index_select(0, valid_pos)
            self.sparse_kv_cache_manager.mem_pool_host.load_to_device_per_layer(
                self.sparse_kv_cache_manager.mem_pool_device,
                source_valid,
                target_valid,
                layer_id,
                "kernel"
            )

        update_sparse_metadata(
            current_metadata.page_table,
            physical_pages,
            valid_lengths.to(torch.int32).contiguous(),
            sparse_mask.to(torch.int32).contiguous(),
            current_metadata.cache_seqlens_int32,
            forward_batch.seq_lens.to(torch.int32).contiguous(),
            self._original_metadata["cache_seqlens_int32"],
            page_size
        )

        current_metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(
                current_metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
            ),
            (1, 0),
        )
        current_metadata.max_seq_len_k = int(current_metadata.cache_seqlens_int32.max())
        return current_metadata
