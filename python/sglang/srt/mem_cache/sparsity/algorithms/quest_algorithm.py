"""
Quest sparse attention algorithm.

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection. For each KV page, it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.
"""

import logging

import torch
import triton

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_kernels import quest_page_rep_kernel

from sgl_kernel import retrieval_score_and_combine_indices

logger = logging.getLogger(__name__)


class QuestAlgorithm(BaseSparseAlgorithmImpl):
    """Quest page-wise sparse attention using bounding-box criticality."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.page_k_min = {}
        self.page_k_max = {}
        self.page_valid = {}

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]

        for layer_id in range(start_layer, end_layer):
            self.page_k_min[layer_id] = torch.zeros(
                (total_num_pages, head_num, head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.page_k_max[layer_id] = torch.zeros_like(self.page_k_min[layer_id])
            self.page_valid[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.bool, device=self.device
            )

        logger.info(
            "Initialized Quest page reps: %d pages, %d layers, head_num=%d, head_dim=%d",
            total_num_pages,
            end_layer - start_layer,
            head_num,
            head_dim,
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

        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())
        if max_pages <= 0:
            return

        req_to_token = self.req_to_token_pool.req_to_token
        head_num = k_buffer.shape[1]
        head_dim = k_buffer.shape[2]

        BLOCK_DIM = triton.next_power_of_2(head_dim)

        page_k_min = self.page_k_min[layer_id]
        page_k_max = self.page_k_max[layer_id]
        page_valid = self.page_valid[layer_id]

        grid = (n, max_pages, head_num)

        quest_page_rep_kernel[grid](
            page_k_min,
            page_k_max,
            page_valid,
            reqs,
            seq_lens,
            start_page,
            end_page,
            req_to_token,
            k_buffer,
            # Strides
            req_to_token.stride(0),
            req_to_token.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            k_buffer.stride(2),
            page_k_min.stride(0),
            page_k_min.stride(1),
            page_k_min.stride(2),
            # Shapes
            req_to_token.shape[1],
            k_buffer.shape[0],
            # Constants
            PAGE_SIZE=self.page_size,
            HEAD_NUM=head_num,
            HEAD_DIM=head_dim,
            BLOCK_DIM=BLOCK_DIM,
        )

        if layer_id == 0:
            logger.info(
                f"Computed page representations for layer {layer_id}, start_page={start_page}, end_page={end_page}"
            )

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        # Clamp pages to valid storage range
        phys_pages_clamped = phys_pages.clamp(0, self.page_k_min[layer_id].shape[0] - 1)

        k_min = self.page_k_min[layer_id][phys_pages_clamped]
        k_max = self.page_k_max[layer_id][phys_pages_clamped]
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]
        # Align query shape to KV heads.
        head_dim = k_min.shape[-1]
        if queries.dim() == 2:
            bs, hidden = queries.shape
            if hidden % head_dim != 0:
                raise ValueError(
                    f"Quest query hidden size {hidden} not divisible by head_dim {head_dim}"
                )
            q_heads = hidden // head_dim
            q = queries.view(bs, q_heads, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")

        kv_heads = k_min.shape[-2]
        q_heads = q.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads
            # Average grouped query heads to align with KV heads (approximation for MQA/GQA).
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)

        q = q.to(k_min.dtype).unsqueeze(1)  # [bs, 1, kv_heads, head_dim]

        criticality = torch.where(q >= 0, q * k_max, q * k_min).sum(dim=(2, 3))
        criticality = torch.where(
            valid_mask, criticality, torch.full_like(criticality, float("-inf"))
        )

        return criticality
