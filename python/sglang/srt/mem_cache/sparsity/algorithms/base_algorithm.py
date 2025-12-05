import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class SparseMode(Enum):
    """Sparse attention granularity mode."""

    PAGE_WISE = "page_wise"
    TOKEN_WISE = "token_wise"
    ORIGINAL_WISE = "original_wise"


class BaseSparseAlgorithm(ABC):
    """Abstract base class for sparse attention algorithms."""

    def __init__(self, config, device: torch.device, **kwargs):
        self.config = config
        self.device = device
        self.repr_pool = None
        self.req_to_token = None

    def set_pools(self, repr_pool, req_to_token_pool):
        """Set representation pool for zero-copy access."""
        self.repr_pool = repr_pool
        self.req_to_token_pool = req_to_token_pool

    @abstractmethod
    def get_sparse_mode(self) -> SparseMode:
        """Return the sparsity granularity mode."""
        pass

    @abstractmethod
    def get_representation_storage_shape(self, token_to_kv_pool) -> Dict[str, tuple]:
        """
        Register storage slots needed by this algorithm.

        Returns:
            Dict mapping storage_name to (shape, dtype).
            Example: {
                "page_repr": ((num_heads * head_dim,), torch.float32),
                "centroids": ((n_subvec, n_centroids, subvec_dim), torch.float32),
                "token_codes": ((page_size, n_subvec), torch.long)
            }
        """
        pass

    def should_construct_representations(
        self,
        forward_batch: "ForwardBatch",
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        repr_constructed: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if the representations pool should be constructed for the given requests.
        Args:
            forward_batch: Forward batch info
            layer_id: Current layer
            req_pool_indices: [bs]
            seq_lens: [bs]
            repr_constructed: [max_pool_size] global state
            prompt_lens: [max_pool_size] global state

        Returns:
            [bs] bool mask
        """
        return torch.zeros_like(req_pool_indices, dtype=torch.bool)

    def construct_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        construct_mask: torch.Tensor,
        k_buffer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct representations for the given requests.
        Args:
            layer_id: Current layer
            req_pool_indices: [bs]
            seq_lens: [bs]
            construct_mask: [bs] bool
            k_buffer: KV cache buffer

        Returns:
            [bs] bool success mask
        """
        return torch.zeros_like(req_pool_indices, dtype=torch.bool)

    @abstractmethod
    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        attn_metadata: Optional[Any],
        **kwargs,
    ) -> tuple:
        """
        Retrieve topk indices for the given requests.
        Args:
            queries: [bs, repr_dim]
            layer_id: layer index
            req_pool_indices: [bs]
            seq_lens: [bs] sequence lengths
            sparse_mask: [bs] bool, which requests need to be retrieved

        Returns:
            selected_indices: [bs, max_selected] padded with -1
            valid_lengths: [bs] actual selected count per request
        """
        pass

    def should_update_represetations(
        self,
        forward_batch: Any,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        is_decode: bool,
        repr_constructed: torch.Tensor,
        decode_steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if the representations pool should be updated for the given requests.

        Args:
            forward_batch: Forward batch info
            layer_id: Current layer
            req_pool_indices: [bs]
            is_decode: Whether in decode mode
            repr_constructed: [max_pool_size] global state
            decode_steps: [max_pool_size] global state

        Returns:
            [bs] bool mask
        """
        return torch.zeros_like(req_pool_indices, dtype=torch.bool)

    def update_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        update_mask: torch.Tensor,
        k_buffer: torch.Tensor,
        last_extracted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update representations for the given requests.

        Args:
            layer_id: Current layer
            req_pool_indices: [bs]
            seq_lens: [bs]
            update_mask: [bs] bool
            k_buffer: KV cache buffer
            req_to_token: Mapping tensor
            last_extracted_tokens: [max_pool_size] global state

        Returns:
            success_mask: [bs] bool, which requests successfully updated
        """
        return torch.zeros_like(req_pool_indices, dtype=torch.bool)


class PageMeanPoolingAlgorithm(BaseSparseAlgorithm):
    """
    Experimental: Page-wise sparse attention with mean pooling and TopK selection.

    This is an example implementation demonstrating page-wise sparse attention where:
    - Pages are represented by mean-pooled key vectors
    - TopK pages are selected based on query-page similarity
    - Recent pages are always included
    """

    PAGE_REPR_STORAGE_NAME = "page_repr"

    def __init__(
        self, config, device: torch.device, start_layer: int, end_layer: int, **kwargs
    ):
        super().__init__(config, device, **kwargs)
        self.sparse_ratio = 0.9
        self.page_size = getattr(config, "page_size", 64)
        self.num_recent_pages = 4
        self.min_sparse_prompt_len = self.page_size * (self.num_recent_pages + 2)
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_heads = None
        self.head_dim = None

    def get_sparse_mode(self) -> SparseMode:
        return SparseMode.PAGE_WISE

    def get_representation_storage_shape(self, token_to_kv_pool) -> Dict[str, tuple]:
        k_sample = token_to_kv_pool.get_key_buffer(self.start_layer)[0]
        if k_sample.dim() == 2:
            self.num_heads, self.head_dim = k_sample.shape
        repr_dim = self.num_heads * self.head_dim
        return {self.PAGE_REPR_STORAGE_NAME: ((repr_dim,), k_sample.dtype)}

    def should_construct_representations(
        self,
        forward_batch,
        layer_id,
        req_pool_indices,
        seq_lens,
        repr_constructed,
        prompt_lens,
    ) -> torch.Tensor:
        if not forward_batch.forward_mode.is_extend():
            return torch.zeros_like(req_pool_indices, dtype=torch.bool)

        return ~repr_constructed[req_pool_indices] & (
            seq_lens >= prompt_lens[req_pool_indices]
        )

    def construct_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        construct_mask,
        k_buffer,
    ) -> torch.Tensor:
        bs = req_pool_indices.shape[0]
        device = k_buffer.device
        success_mask = torch.zeros(bs, dtype=torch.bool, device=device)

        if not construct_mask.any():
            return success_mask

        num_pages_per_req = seq_lens // self.page_size
        max_pages = torch.max(num_pages_per_req[construct_mask])

        if max_pages == 0:
            return success_mask

        valid_mask = construct_mask & (num_pages_per_req > 0)
        if not valid_mask.any():
            return success_mask

        self._batch_update_pages(
            layer_id=layer_id,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            valid_mask=valid_mask,
            start_pages=torch.zeros_like(num_pages_per_req),
            end_pages=num_pages_per_req,
            k_buffer=k_buffer,
            req_to_token=self.req_to_token_pool.req_to_token,
        )

        success_mask[valid_mask] = True
        return success_mask

    def should_update_represetations(
        self,
        forward_batch,
        layer_id,
        req_pool_indices,
        is_decode,
        repr_constructed,
        decode_steps,
    ) -> torch.Tensor:
        if not is_decode:
            return torch.zeros_like(req_pool_indices, dtype=torch.bool)

        mask = repr_constructed[req_pool_indices] & (
            decode_steps[req_pool_indices] >= self.page_size
        )
        return mask

    def update_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        update_mask,
        k_buffer,
        last_extracted_tokens,
    ) -> torch.Tensor:
        bs = req_pool_indices.shape[0]
        device = k_buffer.device
        success_mask = torch.zeros(bs, dtype=torch.bool, device=device)

        if not update_mask.any():
            return success_mask

        last_extracted = last_extracted_tokens[req_pool_indices]
        new_tokens = seq_lens - last_extracted
        has_new_pages_mask = update_mask & (new_tokens >= self.page_size)

        if not has_new_pages_mask.any():
            return success_mask

        start_pages = last_extracted // self.page_size
        end_pages = seq_lens // self.page_size
        valid_update_mask = has_new_pages_mask & (start_pages < end_pages)

        if valid_update_mask.any():
            self._batch_update_pages(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                valid_mask=valid_update_mask,
                start_pages=start_pages,
                end_pages=end_pages,
                k_buffer=k_buffer,
                req_to_token=self.req_to_token_pool.req_to_token,
            )
            success_mask[valid_update_mask] = True

        return success_mask

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        attn_metadata: Optional[Any],
        **kwargs,
    ) -> tuple:
        seq_lens = (
            attn_metadata.cache_seqlens_int32 if attn_metadata is not None else None
        )
        bs, device = queries.shape[0], queries.device
        queries = queries.view(bs, -1)
        num_pages = (seq_lens + self.page_size - 1) // self.page_size
        max_pages = max(
            int(torch.max(num_pages).item()) if num_pages.numel() > 0 else 1, 1
        )

        selected_indices = torch.full(
            (bs, max_pages), -1, dtype=torch.int32, device=device
        )
        valid_lengths = torch.zeros(bs, dtype=torch.int32, device=device)

        effective_mask = sparse_mask & (num_pages > self.num_recent_pages)
        if not effective_mask.any():
            return selected_indices, valid_lengths

        storage = self.repr_pool.get_layer_storage(
            layer_id, self.PAGE_REPR_STORAGE_NAME
        )
        repr_dim = storage.shape[1]

        if queries.shape[1] != repr_dim:
            queries = (
                queries.view(bs, self.num_heads, -1, self.head_dim)
                .mean(dim=2)
                .reshape(bs, -1)
            )

        page_indices = torch.arange(max_pages, device=device).unsqueeze(0)
        page_starts = page_indices * self.page_size

        first_tokens = self.req_to_token_pool.req_to_token[
            req_pool_indices.unsqueeze(1).expand(bs, max_pages),
            page_starts.clamp(0, self.req_to_token_pool.req_to_token.shape[1] - 1),
        ]
        phys_pages = (first_tokens // self.page_size).clamp(0, storage.shape[0] - 1)
        page_reprs = storage[phys_pages].to(queries.dtype)

        recent_start = (num_pages - self.num_recent_pages).clamp(min=0)
        history_mask = page_indices < recent_start.unsqueeze(1)

        scores = (queries.unsqueeze(1) @ page_reprs.transpose(1, 2)).squeeze(1)
        scores = torch.where(history_mask, scores, float("-inf"))

        num_select = (recent_start.float() * self.sparse_ratio).int().clamp(min=1)
        max_select = int(num_select.max().item())

        topk_indices = torch.topk(scores, k=max_select, dim=1, sorted=False)[1]
        topk_mask = torch.arange(max_select, device=device).unsqueeze(
            0
        ) < num_select.unsqueeze(1)
        topk_indices = torch.where(topk_mask, topk_indices, -1)

        recent_indices = recent_start.unsqueeze(1) + torch.arange(
            self.num_recent_pages, device=device
        )
        recent_indices = torch.where(
            recent_indices < num_pages.unsqueeze(1), recent_indices, -1
        )

        combined = torch.cat([topk_indices, recent_indices], dim=1)
        combined = torch.sort(combined, dim=1)[0]

        valid_lengths[:] = torch.where(
            effective_mask, (combined >= 0).sum(dim=1).int(), 0
        )
        selected_indices[:, : combined.shape[1]] = torch.where(
            effective_mask.unsqueeze(1), combined, -1
        )

        return selected_indices, valid_lengths

    def _batch_update_pages(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        valid_mask: torch.Tensor,
        start_pages: torch.Tensor,
        end_pages: torch.Tensor,
        k_buffer: torch.Tensor,
        req_to_token: torch.Tensor,
    ):
        if not valid_mask.any():
            return

        device = k_buffer.device
        storage = self.repr_pool.get_layer_storage(
            layer_id, self.PAGE_REPR_STORAGE_NAME
        )

        valid_reqs = req_pool_indices[valid_mask]
        valid_lens = seq_lens[valid_mask]
        valid_starts = start_pages[valid_mask]
        valid_ends = end_pages[valid_mask]
        num_valid = valid_reqs.shape[0]

        max_pages = int(torch.max(valid_ends - valid_starts))

        page_offsets = torch.arange(max_pages, device=device).unsqueeze(0)
        page_ids = valid_starts.unsqueeze(1) + page_offsets
        page_mask = page_ids < valid_ends.unsqueeze(1)

        token_starts = page_ids * self.page_size
        token_ends = torch.clamp(
            token_starts + self.page_size, max=valid_lens.unsqueeze(1)
        )

        token_offsets = torch.arange(self.page_size, device=device).view(1, 1, -1)
        token_pos = token_starts.unsqueeze(2) + token_offsets
        token_mask = (token_pos < token_ends.unsqueeze(2)) & page_mask.unsqueeze(2)

        req_exp = valid_reqs.view(num_valid, 1, 1).expand(
            num_valid, max_pages, self.page_size
        )
        token_pos_safe = torch.clamp(token_pos, 0, req_to_token.shape[1] - 1)
        phys_tokens = req_to_token[req_exp, token_pos_safe]
        phys_tokens_safe = torch.clamp(phys_tokens, 0, k_buffer.shape[0] - 1)

        k_values = k_buffer[phys_tokens_safe] * token_mask.unsqueeze(-1).unsqueeze(-1)
        token_counts = token_mask.sum(dim=2, keepdim=True).unsqueeze(-1).clamp(min=1)
        page_reprs = (k_values.sum(dim=2) / token_counts).reshape(
            num_valid, max_pages, -1
        )

        first_tokens = req_to_token[
            valid_reqs.unsqueeze(1).expand(num_valid, max_pages),
            torch.clamp(token_starts, 0, req_to_token.shape[1] - 1),
        ]
        phys_pages = first_tokens // self.page_size

        valid_indices = page_mask.nonzero(as_tuple=False)
        if valid_indices.numel() > 0:
            storage[phys_pages[valid_indices[:, 0], valid_indices[:, 1]]] = page_reprs[
                valid_indices[:, 0], valid_indices[:, 1]
            ]
