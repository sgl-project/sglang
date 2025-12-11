# SPDX-License-Identifier: Apache-2.0
import logging
from abc import abstractmethod
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithm,
    SparseMode,
)

logger = logging.getLogger(__name__)


class BasePageWiseAlgorithm(BaseSparseAlgorithm):
    """
    Base class for page-wise sparse attention algorithms.

    Provides common infrastructure for algorithms that operate at page/chunk granularity:
    - Generic construct/update flow with state tracking
    - TopK retrieval with recent page retention (can be overridden)

    Subclasses need to implement:
    - _initialize_representation_pools(): Initialize algorithm-specific representation pools
    - _compute_page_representations(): Compute page scores/representations
    - _retrieve_page_scores(): Retrieve page scores for TopK selection

    Subclasses can optionally override:
    - retrieve_topk(): For query-dependent retrieval logic
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.compression_ratio = getattr(config, "compression_ratio", 0.2)
        self.page_size = getattr(config, "page_size", 64)
        self.num_recent_pages = getattr(config, "num_recent_pages", 4)

    def get_sparse_mode(self) -> SparseMode:
        return SparseMode.PAGE_WISE

    def initialize_representation_pool(
        self,
        start_layer: int,
        end_layer: int,
        token_to_kv_pool,
        req_to_token_pool,
        states,
    ):
        super().initialize_representation_pool(
            start_layer, end_layer, token_to_kv_pool, req_to_token_pool, states
        )
        self.start_layer = start_layer
        self.end_layer = end_layer

        total_num_tokens = token_to_kv_pool.get_key_buffer(start_layer).shape[0]
        total_num_pages = (total_num_tokens + self.page_size - 1) // self.page_size

        self._initialize_representation_pools(start_layer, end_layer, total_num_pages)

    def construct_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ) -> torch.Tensor:
        if not forward_batch.forward_mode.is_extend():
            return

        num_pages = seq_lens // self.page_size
        valid_mask = (
            ~self.states.repr_constructed[req_pool_indices]
            & (seq_lens >= self.states.prompt_lens[req_pool_indices])
            & (num_pages > 0)
        )

        if not valid_mask.any():
            return

        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            0,
            num_pages[valid_mask],
            k_buffer,
        )

        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.repr_constructed[success_indices] = True
            self.states.last_extracted_token[success_indices] = (
                seq_lens[valid_mask] // self.page_size * self.page_size
            )

    def update_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ) -> torch.Tensor:
        if not forward_batch.forward_mode.is_decode_or_idle():
            return

        start_page = (
            self.states.last_extracted_token[req_pool_indices] // self.page_size
        )
        end_page = seq_lens // self.page_size
        valid_mask = self.states.repr_constructed[req_pool_indices] & (
            start_page < end_page
        )

        if not valid_mask.any():
            return

        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            start_page[valid_mask],
            end_page[valid_mask],
            k_buffer,
        )

        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.last_extracted_token[success_indices] = (
                seq_lens[valid_mask] // self.page_size * self.page_size
            )

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
        Default TopK retrieval: score-based selection + recent pages.
        Subclasses can override for query-dependent retrieval.
        """
        bs, device = queries.shape[0], queries.device
        seq_lens = attn_metadata.cache_seqlens_int32
        num_pages = (seq_lens + self.page_size - 1) // self.page_size
        max_pages = max(int(num_pages.max().item()), 1)

        out_indices = torch.full((bs, max_pages), -1, dtype=torch.int32, device=device)
        out_lengths = torch.zeros(bs, dtype=torch.int32, device=device)

        mask = sparse_mask & (num_pages > self.num_recent_pages)
        if not mask.any():
            return out_indices, out_lengths

        page_idx = torch.arange(max_pages, device=device).unsqueeze(0)
        page_start_token = self.req_to_token_pool.req_to_token[
            req_pool_indices.unsqueeze(1).expand(bs, max_pages),
            (page_idx * self.page_size).clamp(
                0, self.req_to_token_pool.req_to_token.shape[1] - 1
            ),
        ]
        phys_pages = page_start_token // self.page_size

        scores = self._retrieve_page_scores(
            layer_id, phys_pages, req_pool_indices, queries
        )

        recent_start = (num_pages - self.num_recent_pages).clamp(min=0)
        scores.masked_fill_(page_idx >= recent_start.unsqueeze(1), float("-inf"))

        k = max(
            int((recent_start.float() * (1 - self.compression_ratio)).max().item()), 1
        )
        topk_idx = torch.topk(scores, k=k, dim=1, sorted=False)[1]
        topk_mask = torch.arange(k, device=device).unsqueeze(0) < (
            recent_start * (1 - self.compression_ratio)
        ).int().clamp(min=1).unsqueeze(1)

        recent_idx = recent_start.unsqueeze(1) + torch.arange(
            self.num_recent_pages, device=device
        )
        recent_mask = recent_idx < num_pages.unsqueeze(1)

        combined = torch.cat(
            [
                torch.where(topk_mask, topk_idx, -1),
                torch.where(recent_mask, recent_idx, -1),
            ],
            dim=1,
        ).sort(dim=1)[0]

        out_lengths[:] = torch.where(mask, (combined >= 0).sum(dim=1).int(), 0)
        out_indices[:, : combined.shape[1]] = torch.where(
            mask.unsqueeze(1), combined, -1
        )

        return out_indices, out_lengths

    @abstractmethod
    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        """
        Initialize algorithm-specific representation pools for all layers.

        Subclasses define their own representation format based on algorithm needs.
        Examples:
        - Knorm: self.page_scores[layer_id] = torch.zeros((total_num_pages, 1))
        - Quest: self.page_reprs[layer_id] = torch.zeros((total_num_pages, head_dim))
        """
        pass

    @abstractmethod
    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        """
        Compute and store page representations for given page range.

        Args:
            layer_id: Current layer index
            reqs: [n] Request pool indices
            seq_lens: [n] Current sequence lengths
            start_page: Starting page index (int or [n] tensor)
            end_page: [n] Ending page indices (exclusive)
            k_buffer: Key buffer for the layer
        """
        pass

    @abstractmethod
    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve page scores for TopK selection.

        Args:
            layer_id: Current layer index
            phys_pages: [bs, max_pages] Physical page indices
            req_pool_indices: [bs] Request pool indices
            queries: [bs, num_heads, head_dim] Query vectors

        Returns:
            scores: [bs, max_pages] Page scores for ranking
        """
        pass


class KnormPageAlgorithm(BasePageWiseAlgorithm):
    """
    L2-norm based page-wise sparse attention (ChunkKV-style).

    Pages are scored based on key L2 norms aggregated across tokens.
    TopK pages are selected based on pre-computed scores with recent pages always included.

    Based on ChunkKV (https://arxiv.org/abs/2502.00299).

    Note: This is an experimental/example implementation for demonstrating
    how to integrate algorithms into the sparse framework.
    Not production-ready - use for reference and testing purposes only.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.page_scores = {}

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        for layer_id in range(start_layer, end_layer):
            self.page_scores[layer_id] = torch.zeros(
                (total_num_pages, 1), dtype=torch.float32, device=self.device
            )
        logger.info(
            f"Initialized page representation pools: {total_num_pages} pages, "
            f"{end_layer - start_layer} layers"
        )

    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        if isinstance(start_page, int):
            start_page = torch.full_like(end_page, start_page)

        device = k_buffer.device
        req_to_token = self.req_to_token_pool.req_to_token
        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())

        pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
        pg_id = start_page.unsqueeze(1) + pg_off
        pg_mask = pg_id < end_page.unsqueeze(1)

        tok_start = pg_id * self.page_size
        tok_off = torch.arange(self.page_size, device=device).view(1, 1, -1)
        tok_pos = tok_start.unsqueeze(2) + tok_off
        tok_mask = (
            tok_pos
            < (tok_start + self.page_size).clamp(max=seq_lens.unsqueeze(1)).unsqueeze(2)
        ) & pg_mask.unsqueeze(2)

        phys_tok = req_to_token[
            reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
            tok_pos.clamp(0, req_to_token.shape[1] - 1),
        ].clamp(0, k_buffer.shape[0] - 1)

        tok_score = k_buffer[phys_tok].norm(dim=-1).sum(dim=-1)
        pg_score = (tok_score * tok_mask).sum(dim=2) / tok_mask.sum(dim=2).clamp(min=1)

        phys_pg = (
            req_to_token[
                reqs.unsqueeze(1).expand(n, max_pages),
                tok_start.clamp(0, req_to_token.shape[1] - 1),
            ]
            // self.page_size
        )
        idx = pg_mask.nonzero(as_tuple=False)
        if idx.numel() > 0:
            scores_to_store = (
                pg_score[idx[:, 0], idx[:, 1]]
                .unsqueeze(-1)
                .to(self.page_scores[layer_id].dtype)
            )
            self.page_scores[layer_id][phys_pg[idx[:, 0], idx[:, 1]]] = scores_to_store

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        phys_pages_clamped = phys_pages.clamp(
            0, self.page_scores[layer_id].shape[0] - 1
        )
        return self.page_scores[layer_id][phys_pages_clamped].squeeze(-1)
