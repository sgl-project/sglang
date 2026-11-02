"""Native LIM plans shared by fused short, sparse and chain-MTP attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.npu_lim import (
    fused_lightning_indexer_manage,
    load_fused_lim_operator,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    prepare_mtp_rows,
    sparse_query_width,
    validate_mtp_chain,
)


@dataclass
class LIMSelection:
    source_ids: torch.Tensor
    device_slots: torch.Tensor
    topk_misses: torch.Tensor
    miss_sources: torch.Tensor
    miss_slots: torch.Tensor
    miss_counts: torch.Tensor
    prefix_lens: torch.Tensor
    sparse_rows: torch.Tensor
    valid_rows: torch.Tensor
    request_slots: torch.Tensor
    seq_lens: torch.Tensor
    query_width: int = 1


class LIMDecodeAdapter:
    def __init__(self, manager):
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_copy_sfa_metadata import (
            fused_lim_copy_sfa_metadata,
        )
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_input_metadata import (
            fused_prepare_lim_metadata,
        )

        self.capacity = manager.cache_capacity
        self.rows = manager.size
        self.batch_capacity = manager.batch_capacity
        self.max_query_width = manager.max_query_width
        self.top_k = manager.sparse_context_len
        self.full_visible = not manager.separate_mtp_tail
        self.source_capacity = manager.max_context_len
        self.operator = load_fused_lim_operator()
        try:
            supported = torch.ops.npu.lim_full_visible_max_query_width.default()
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "Install the matching dynamic-query LIM kernel wheel."
            ) from error
        if supported < self.max_query_width:
            raise ValueError(
                "Configured MTP query width exceeds the loaded LIM kernel."
            )
        self._fused_prepare_lim_metadata = fused_prepare_lim_metadata
        self._fused_copy_sfa_metadata = fused_lim_copy_sfa_metadata
        device = manager.device
        tokens = self.batch_capacity * self.max_query_width
        self.source_ids = torch.empty(
            (tokens, 1, self.top_k), dtype=torch.int32, device=device
        )
        self.device_slots = torch.empty_like(self.source_ids)
        self.topk_misses = torch.empty(tokens, dtype=torch.int32, device=device)
        self.miss_sources = torch.empty(
            (self.batch_capacity, 32768), dtype=torch.int32, device=device
        )
        self.miss_slots = torch.empty_like(self.miss_sources)
        self.miss_counts = torch.empty(
            self.batch_capacity, dtype=torch.int32, device=device
        )
        self.query_scale = torch.empty((tokens, 64), dtype=torch.float32, device=device)
        self.query_ends = (
            torch.arange(1, self.batch_capacity + 1, dtype=torch.int32, device=device)[
                None
            ]
            * torch.arange(
                1, self.max_query_width + 1, dtype=torch.int32, device=device
            )[:, None]
        )
        self.block_table = torch.empty(
            (self.batch_capacity, self.source_capacity // 128),
            dtype=torch.int32,
            device=device,
        )
        self.last_prefix = torch.full(
            (manager.layer_num, self.rows + 1), -1, dtype=torch.int32, device=device
        )
        self._metadata_prefix_scratch = torch.full(
            (self.rows + 1,), -1, dtype=torch.int32, device=device
        )
        self._pre_safe_slots = torch.empty(
            self.batch_capacity, dtype=torch.int32, device=device
        )
        self._pre_lengths = torch.empty_like(self._pre_safe_slots)
        self._pre_prefixes = torch.empty_like(self._pre_safe_slots)
        self._pre_states = torch.empty_like(self._pre_safe_slots)
        self._pre_next_prefix = torch.empty_like(self._pre_safe_slots)
        self._pre_cache_sizes = torch.empty_like(self._pre_safe_slots)
        self._pre_sparse = torch.empty(
            self.batch_capacity, dtype=torch.bool, device=device
        )
        self._pre_valid = torch.empty_like(self._pre_sparse)
        key = manager.paged_kv_cache.get_index_k_buffer(manager.start_layer)
        if key.dtype != torch.bfloat16 or manager.store_dtype != torch.bfloat16:
            raise ValueError("Native LIM requires BF16 index and KV caches.")
        if key.ndim != 4 or key.shape[1:] != (128, 1, 128):
            raise ValueError("Native LIM requires index keys [pages,128,1,128].")
        # Charged to the index-cache pool coefficient rather than this adapter.
        self.key_scale = torch.empty(key.shape[:3], dtype=torch.float32, device=device)

    def reset_requests(self, request_slots):
        self.last_prefix[:, request_slots] = -1

    def select(
        self, manager, *, query, weights, key, block_table, forward_batch, layer_id
    ):
        batch = forward_batch.batch_size
        width = sparse_query_width(forward_batch)
        rows = batch * width
        if (
            not 1 <= width <= self.max_query_width
            or query.shape[0] < rows
            or batch > self.batch_capacity
        ):
            raise ValueError(
                "LIM query geometry exceeds the configured native buffers."
            )
        if block_table.shape[0] != batch:
            raise ValueError("LIM source table must match the padded request batch.")
        if width > 1:
            from sglang.srt.environ import envs

            if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
                _, positions, live = prepare_mtp_rows(
                    forward_batch, query.shape[0], self.rows, self.source_capacity
                )
                validate_mtp_chain(forward_batch, positions, live, self.rows)
        columns = min(block_table.shape[1], self.block_table.shape[1])
        layer_index = layer_id - manager.start_layer
        self._fused_prepare_lim_metadata(
            seq_lens=forward_batch.seq_lens[:batch],
            positions=forward_batch.positions[:rows],
            request_slots=forward_batch.req_pool_indices,
            last_prefix=self.last_prefix[layer_index],
            block_table=block_table[:, :columns],
            out_block_table=self.block_table[:batch],
            safe_slots=self._pre_safe_slots[:batch],
            lengths=self._pre_lengths[:batch],
            prefixes=self._pre_prefixes[:batch],
            sparse_rows=self._pre_sparse[:batch],
            valid_rows=self._pre_valid[:batch],
            request_states=self._pre_states[:batch],
            next_prefixes=self._pre_next_prefix[:batch],
            cache_sizes=self._pre_cache_sizes[:batch],
            capacity=self.capacity,
            request_rows=self.rows,
            full_visible=self.full_visible,
            query_width=width,
        )
        scale = self.query_scale.view(-1)[: rows * query.shape[1]].view(
            rows, query.shape[1]
        )
        sources = self.source_ids[:rows]
        fused_lightning_indexer_manage(
            weights=weights[:rows].contiguous(),
            query_scale=scale,
            query=query[:rows].contiguous(),
            key_scale=self.key_scale,
            key=key,
            block_table=self.block_table[:batch],
            query_ends=self.query_ends[width - 1, :batch],
            seq_lens=self._pre_lengths[:batch],
            prefix_lens=self._pre_prefixes[:batch],
            cache_sizes=self._pre_cache_sizes[:batch],
            request_state=self._pre_states[:batch],
            request_slots=self._pre_safe_slots[:batch],
            slot_pool=manager.device_slot_map[layer_index],
            source_ids=sources,
            device_slots=self.device_slots[:rows],
            topk_misses=self.topk_misses[:rows],
            miss_sources=self.miss_sources[:batch],
            miss_slots=self.miss_slots[:batch],
            miss_counts=self.miss_counts[:batch],
            operator=self.operator,
            full_visible=self.full_visible,
        )
        self.last_prefix[layer_index].index_copy_(
            0, self._pre_safe_slots[:batch].long(), self._pre_next_prefix[:batch]
        )
        return LIMSelection(
            sources[:, 0],
            self.device_slots[:rows, 0],
            self.topk_misses[:rows],
            self.miss_sources[:batch],
            self.miss_slots[:batch],
            self.miss_counts[:batch],
            self._pre_prefixes[:batch],
            self._pre_sparse[:batch],
            self._pre_valid[:batch],
            self._pre_safe_slots[:batch],
            self._pre_lengths[:batch],
            width,
        )

    def copy_sfa_metadata(self, manager, plan):
        batch = plan.request_slots.numel()
        self._fused_copy_sfa_metadata(
            source_ids=plan.source_ids,
            topk_misses=plan.topk_misses,
            miss_counts=plan.miss_counts,
            valid_rows=plan.valid_rows,
            sparse_rows=plan.sparse_rows,
            request_slots=plan.request_slots,
            next_prefix=plan.prefix_lens,
            last_prefix=self._metadata_prefix_scratch,
            actual_seq_lengths_kv=manager.planner_actual_seq_lengths_kv[:batch],
            num_cache_tokens=manager.planner_num_cache_tokens[:batch],
            hbm_block_table=manager.planner_hbm_block_table[:batch],
            dram_block_table=manager.planner_dram_block_table[:batch],
            capacity=self.capacity,
            host_context_len=manager.max_context_len,
            request_rows=self.rows,
            query_width=plan.query_width,
            tail_tokens=manager.tail_tokens,
            tail_seq_lens=plan.seq_lens,
            host_block_table=self.block_table[:batch],
            full_visible=self.full_visible,
        )
        return self.query_ends[plan.query_width - 1, :batch]
