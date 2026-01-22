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

from sgl_kernel import quest_retrieval_score_and_combine_indices

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

    def construct_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ) -> torch.Tensor:
        num_pages = seq_lens // self.page_size
        prompt_lens = self.states.prompt_lens[req_pool_indices]
        valid_mask = (
            ~self.states.repr_constructed[req_pool_indices]
            & (prompt_lens >= self.states.device_buffer_cnt)
            & (num_pages > 0)
        )

        if not valid_mask.any():
            return

        # Compute page representations by subclass
        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            0,
            num_pages[valid_mask],
            k_buffer,
        )

        # Update tracking states
        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.repr_constructed[success_indices] = True
            self.states.last_constructed_page[success_indices] = num_pages[valid_mask]


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

        start_page = self.states.last_constructed_page[req_pool_indices]
        end_page = seq_lens // self.page_size
        valid_mask = self.states.repr_constructed[req_pool_indices] & (
            start_page < end_page
        )

        if not valid_mask.any():
            return

        # Compute page representations by subclass
        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            start_page[valid_mask],
            end_page[valid_mask],
            k_buffer,
        )

        # Update tracking states
        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.last_constructed_page[success_indices] = end_page[valid_mask]

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        bs, device = queries.shape[0], queries.device
        
        seq_lens_source = kwargs.get("forward_batch", None)
        if seq_lens_source is None or not hasattr(seq_lens_source, "seq_lens"):
            raise ValueError("forward_batch with seq_lens is required for TopK retrieval")
        seq_lens = seq_lens_source.seq_lens.to(device)
        
        # Calculate max_out roughly
        max_seq_len = torch.max(seq_lens).item()
        max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        
        k_val = 0
        if self.fixed_topk_page_cnt is not None:
            k_val = self.fixed_topk_page_cnt
        else:
            k_val = int(max_pages * self.sparsity_ratio) + self.num_recent_pages
        
        # Clamp k_val
        if k_val > max_pages:
            k_val = max_pages
            
        # Add buffer for safety and recent pages overlap
        max_out = k_val + self.num_recent_pages + 32
        
        out_indices = torch.empty((bs, max_out), dtype=torch.int32, device=device)
        out_lengths = torch.empty((bs,), dtype=torch.int32, device=device)
        
        """
            bs=<class 'int'>
            seq_lens=<class 'torch.Tensor'>
            page_size=<class 'int'>
            req_to_token=<class 'torch.Tensor'>
            self.page_k_min[layer_id]=<class 'torch.Tensor'>
            self.page_k_max[layer_id]=<class 'torch.Tensor'>
            queries=<class 'torch.Tensor'>
            req_pool_indices=<class 'torch.Tensor'>
            self.num_recent_pages=<class 'int'>
            self.fixed_topk_page_cnt=<class 'int'>
            self.sparsity_ratio=<class 'float'>
            sparse_mask=<class 'torch.Tensor'>
            out_indices=<class 'torch.Tensor'>
            out_lengths=<class 'torch.Tensor'>
            
            === para info ===
            bs: Python type=<class 'int'>, value=1
            page_size: Python type=<class 'int'>, value=64
            num_recent_pages: Python type=<class 'int'>, value=4
            fixed_topk_page_cnt: Python type=<class 'int'>, value=16
            sparsity_ratio: Python type=<class 'float'>, value=0.7
            layer_id: Python type=<class 'int'>, value=0

            seq_lens: Python type=<class 'torch.Tensor'>, tensor dtype=torch.int64, shape=torch.Size([1])
            req_to_token: Python type=<class 'torch.Tensor'>, tensor dtype=torch.int32, shape=torch.Size([4096, 40964])
            page_k_min[layer_id]: Python type=<class 'torch.Tensor'>, tensor dtype=torch.float32, shape=torch.Size([6886, 8, 128])
            page_k_max[layer_id]: Python type=<class 'torch.Tensor'>, tensor dtype=torch.float32, shape=torch.Size([6886, 8, 128])
            queries: Python type=<class 'torch.Tensor'>, tensor dtype=torch.bfloat16, shape=torch.Size([1, 4096])
            req_pool_indices: Python type=<class 'torch.Tensor'>, tensor dtype=torch.int64, shape=torch.Size([1])
            sparse_mask: Python type=<class 'torch.Tensor'>, tensor dtype=torch.bool, shape=torch.Size([1])
            out_indices: Python type=<class 'torch.Tensor'>, tensor dtype=torch.int32, shape=torch.Size([1, 37])
            out_lengths: Python type=<class 'torch.Tensor'>, tensor dtype=torch.int32, shape=torch.Size([1])

        """

        quest_retrieval_score_and_combine_indices(
            bs,
            seq_lens.to(torch.int32),
            self.page_size,
            self.req_to_token_pool.req_to_token,
            self.page_k_min[layer_id].to(queries.dtype),
            self.page_k_max[layer_id].to(queries.dtype),
            queries,
            req_pool_indices.to(torch.int32),
            self.num_recent_pages,
            self.fixed_topk_page_cnt,
            self.sparsity_ratio,
            sparse_mask.to(torch.int32),
            out_indices,
            out_lengths,
        )
        
        return out_indices, out_lengths
