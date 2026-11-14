"""Native LIM plans shared by fused short, sparse and chain-MTP attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.npu_lim import (
    fused_lightning_indexer_manage,
    load_fused_lim_operator,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    sparse_query_width,
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
    request_safe_slots: torch.Tensor
    query_width: int = 1


class LIMDecodeAdapter:
    def __init__(self, manager):
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_copy_sfa_metadata import (
            prepare_copy_sfa_metadata,
        )
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_input_metadata import (
            fused_prepare_lim_metadata,
        )

        self.capacity = manager.cache_capacity
        self.rows = manager.size
        self.batch_capacity = manager.batch_capacity
        self.max_query_width = manager.max_query_width
        self.operator = load_fused_lim_operator()
        self._fused_prepare_lim_metadata = fused_prepare_lim_metadata
        self._prepare_copy_sfa_metadata = prepare_copy_sfa_metadata
        device = manager.device
        for name, spec in manager.budget.adapter_buffers.items():
            if name == "query_ends":
                value = (
                    torch.arange(
                        1, self.max_query_width + 1, dtype=spec.dtype, device=device
                    )[:, None]
                    * torch.arange(
                        1, self.batch_capacity + 1, dtype=spec.dtype, device=device
                    )[None, :]
                )
            elif name == "last_prefix":
                value = spec.allocate(device, fill_value=-1)
            else:
                value = spec.allocate(device)
            setattr(self, name, value)

    def reset_requests(self, request_slots):
        self.last_prefix[:, request_slots] = -1

    def select(
        self, manager, *, query, weights, key, block_table, forward_batch, layer_id
    ):
        batch = forward_batch.batch_size
        width = sparse_query_width(forward_batch)
        rows = batch * width
        columns = min(block_table.shape[1], self.block_table.shape[1])
        layer_index = layer_id - manager.start_layer
        self._fused_prepare_lim_metadata(
            seq_lens=forward_batch.seq_lens[:batch],
            positions=forward_batch.positions[:rows],
            request_slots=forward_batch.req_pool_indices,
            last_prefix=self.last_prefix[layer_index],
            block_table=block_table[:, :columns],
            out_block_table=self.block_table[:batch],
            safe_slots=self.request_safe_slots[:batch],
            lengths=self._pre_lengths[:batch],
            prefixes=self._pre_prefixes[:batch],
            sparse_rows=self._pre_sparse[:batch],
            valid_rows=self._pre_valid[:batch],
            request_states=self._pre_states[:batch],
            next_prefixes=self._pre_next_prefix[:batch],
            cache_sizes=self._pre_cache_sizes[:batch],
            capacity=self.capacity,
            request_rows=self.rows,
            query_width=width,
        )
        sources = self.source_ids[:rows]
        fused_lightning_indexer_manage(
            weights=weights[:rows].contiguous(),
            query=query[:rows].contiguous(),
            key=key,
            block_table=self.block_table[:batch],
            query_ends=self.query_ends[width - 1, :batch],
            seq_lens=self._pre_lengths[:batch],
            prefix_lens=self._pre_prefixes[:batch],
            cache_sizes=self._pre_cache_sizes[:batch],
            request_state=self._pre_states[:batch],
            request_slots=self.request_safe_slots[:batch],
            slot_pool=manager.device_slot_map[layer_index],
            source_ids=sources,
            device_slots=self.device_slots[:rows],
            topk_misses=self.topk_misses[:rows],
            miss_sources=self.miss_sources[:batch],
            miss_slots=self.miss_slots[:batch],
            miss_counts=self.miss_counts[:batch],
            operator=self.operator,
            full_visible=True,
        )
        self.last_prefix[layer_index].index_copy_(
            0, self.request_safe_slots[:batch].long(), self._pre_next_prefix[:batch]
        )
        return LIMSelection(
            sources,
            self.device_slots[:rows],
            self.topk_misses[:rows],
            self.miss_sources[:batch],
            self.miss_slots[:batch],
            self.miss_counts[:batch],
            self._pre_prefixes[:batch],
            self._pre_sparse[:batch],
            self._pre_valid[:batch],
            self.request_safe_slots[:batch],
            width,
        )

    def prepare_copy_sfa_metadata(self, manager, plan) -> None:
        batch = plan.request_safe_slots.numel()
        self._prepare_copy_sfa_metadata(
            source_ids=plan.source_ids,
            valid_rows=plan.valid_rows,
            sparse_rows=plan.sparse_rows,
            request_slots=plan.request_safe_slots,
            actual_seq_lengths_kv=manager.planner_actual_seq_lengths_kv[:batch],
            hbm_block_table=manager.planner_hbm_block_table[:batch],
            capacity=self.capacity,
            query_width=plan.query_width,
            seq_lens=plan.prefix_lens,
        )
        manager.planner_query_ends = self.query_ends[plan.query_width - 1, :batch]
