"""
Quest sparse attention algorithm.

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection. For each KV page, it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.
"""

import logging

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
    load_optional_quest_kernel,
)

logger = logging.getLogger(__name__)


class QuestAlgorithm(BaseSparseAlgorithmImpl):
    """Quest page-wise sparse attention using bounding-box criticality."""

    supports_fixed_cuda_graph_capacity = True

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.enable_cuda_graph_retrieval = config.sparse_extra_config.get(
            "enable_cuda_graph_retrieval", True
        )
        self.page_k_min = {}
        self.page_k_max = {}
        self.page_valid = {}

    def update_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        if not forward_batch.forward_mode.is_decode():
            return super().update_representations(
                layer_id,
                req_pool_indices,
                seq_lens,
                k_buffer,
                forward_batch,
            )
        if not self.should_update_representations(forward_batch):
            return
        return self._update_decode_representations(
            layer_id,
            req_pool_indices,
            seq_lens,
            k_buffer,
            forward_batch,
        )

    def _update_decode_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        kernel_module = None
        if (
            req_pool_indices.numel() >= 4
            and k_buffer.is_cuda
            and torch.version.hip is None
            and k_buffer.ndim == 3
            and k_buffer.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and 0 < self.page_size <= 128
            and 0 < k_buffer.shape[-1] <= 256
        ):
            kernel_module = load_optional_quest_kernel(
                "sglang.srt.mem_cache.sparsity.kernels.quest_page_update"
            )
        if kernel_module is not None:
            tensors = (
                req_pool_indices,
                seq_lens,
                self.req_to_token_pool.req_to_token,
                self.states.repr_constructed,
                self.states.last_constructed_page,
            )
            if all(tensor.device == k_buffer.device for tensor in tensors):
                kernel_module.quest_update_page_representations_(
                    req_pool_indices,
                    seq_lens,
                    self.req_to_token_pool.req_to_token,
                    k_buffer,
                    self.states.repr_constructed,
                    self.states.last_constructed_page,
                    self.page_k_min[layer_id],
                    self.page_k_max[layer_id],
                    self.page_valid[layer_id],
                    self.page_size,
                    advance_trackers=layer_id == self.end_layer - 1,
                )
                return

        return super().update_representations(
            layer_id,
            req_pool_indices,
            seq_lens,
            k_buffer,
            forward_batch,
        )

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

        device = k_buffer.device
        req_to_token = self.req_to_token_pool.req_to_token
        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())
        if max_pages <= 0:
            return

        pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
        pg_id = start_page.unsqueeze(1) + pg_off
        pg_mask = pg_id < end_page.unsqueeze(1)

        tok_start = pg_id * self.page_size
        tok_off = torch.arange(self.page_size, device=device).view(1, 1, -1)
        tok_pos = tok_start.unsqueeze(2) + tok_off

        phys_tok = req_to_token[
            reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
            tok_pos.clamp(0, req_to_token.shape[1] - 1),
        ].clamp(0, k_buffer.shape[0] - 1)

        keys = k_buffer[phys_tok].to(self.page_k_min[layer_id].dtype)
        page_min = keys.amin(dim=2)
        page_max = keys.amax(dim=2)

        phys_pg = (
            req_to_token[
                reqs.unsqueeze(1).expand(n, max_pages),
                tok_start.clamp(0, req_to_token.shape[1] - 1),
            ]
            // self.page_size
        )

        idx = pg_mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return

        target_pages = phys_pg[idx[:, 0], idx[:, 1]].clamp(
            0, self.page_k_min[layer_id].shape[0] - 1
        )
        self.page_k_min[layer_id][target_pages] = page_min[idx[:, 0], idx[:, 1]]
        self.page_k_max[layer_id][target_pages] = page_max[idx[:, 0], idx[:, 1]]
        self.page_valid[layer_id][target_pages] = True

    def _optional_score_kernel(self, queries: torch.Tensor):
        if (
            not queries.is_cuda
            or torch.version.hip is not None
            or not self.page_k_min
            or next(iter(self.page_k_min.values())).shape[-1] > 256
        ):
            return None
        return load_optional_quest_kernel(
            "sglang.srt.mem_cache.sparsity.kernels.quest_score"
        )

    def _retrieve_page_scores_batched(self, layer_id, queries, plan) -> torch.Tensor:
        kernel_module = self._optional_score_kernel(queries)
        if kernel_module is not None:
            return kernel_module.quest_page_scores(
                queries,
                self.page_k_min[layer_id],
                self.page_k_max[layer_id],
                self.page_valid[layer_id],
                plan.physical_pages,
                active_mask=plan.active_mask,
                history_page_counts=plan.recent_start,
            )
        return super()._retrieve_page_scores_batched(layer_id, queries, plan)

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        physical_pages = phys_pages
        kernel_module = self._optional_score_kernel(queries)
        if kernel_module is not None:
            return kernel_module.quest_page_scores(
                queries,
                self.page_k_min[layer_id],
                self.page_k_max[layer_id],
                self.page_valid[layer_id],
                physical_pages,
            )

        # Clamp pages to valid storage range for the portable fallback.
        phys_pages_clamped = phys_pages.clamp(0, self.page_k_min[layer_id].shape[0] - 1)

        k_min = self.page_k_min[layer_id][phys_pages_clamped].to(torch.float32)
        k_max = self.page_k_max[layer_id][phys_pages_clamped].to(torch.float32)
        valid_mask = self.page_valid[layer_id][phys_pages_clamped] & (
            (physical_pages >= 0)
            & (physical_pages < self.page_k_min[layer_id].shape[0])
        )
        # Align query shape to KV heads.
        head_dim = k_min.shape[-1]
        if queries.dim() == 2:
            bs, hidden = queries.shape
            if hidden % head_dim != 0:
                raise ValueError(
                    f"Quest query hidden size {hidden} not divisible by head_dim {head_dim}"
                )
            q_heads = hidden // head_dim
            q = queries.reshape(bs, q_heads, head_dim)
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
