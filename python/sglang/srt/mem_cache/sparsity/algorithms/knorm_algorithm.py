# SPDX-License-Identifier: Apache-2.0
import logging

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


class KnormPageAlgorithm(BaseSparseAlgorithmImpl):
    """
    L2-norm based page-wise sparse attention.

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
