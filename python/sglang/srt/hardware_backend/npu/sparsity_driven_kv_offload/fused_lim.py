"""Native LIM plans shared by fused short, sparse and chain-MTP attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.npu_lim import (
    fused_lightning_indexer_manage,
    load_fused_lim_operator,
)

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
        BudgetPlan,
    )


def sparse_query_width(forward_batch):
    """Decode has one query per request; verify uses its chain width."""
    if forward_batch.forward_mode.is_decode():
        return 1
    return forward_batch.spec_info.num_tokens_per_req


@dataclass
class LIMSelection:
    source_ids: torch.Tensor
    device_slots: torch.Tensor
    topk_misses: torch.Tensor
    miss_sources: torch.Tensor
    miss_slots: torch.Tensor
    miss_counts: torch.Tensor
    sparse_rows: torch.Tensor
    valid_rows: torch.Tensor
    request_safe_slots: torch.Tensor
    resident_safe_slots: torch.Tensor
    query_width: int = 1


class LIMDecodeAdapter:
    def __init__(
        self,
        budget: BudgetPlan,
        *,
        device: str | torch.device,
        last_prefix: torch.Tensor,
        request_to_resident: torch.Tensor | None,
    ) -> None:
        from sgl_kernel_npu.operators.indexer.fused_prepare_lim_metadata import (
            fused_prepare_lim_metadata,
        )

        self.capacity = budget.cache_tokens
        self.rows = budget.request_rows
        self.resident_rows = budget.resident_rows
        self.batch_capacity = budget.batch_capacity
        self.max_query_width = budget.max_query_width
        self.last_prefix = last_prefix
        self.request_to_resident = request_to_resident
        self.operator = load_fused_lim_operator()
        self._fused_prepare_lim_metadata = fused_prepare_lim_metadata
        self._prepared_batch = None
        for name, spec in budget.adapter_buffers.items():
            if name == "query_ends":
                value = (
                    torch.arange(
                        1, self.max_query_width + 1, dtype=spec.dtype, device=device
                    )[:, None]
                    * torch.arange(
                        1, self.batch_capacity + 1, dtype=spec.dtype, device=device
                    )[None, :]
                )
            else:
                value = spec.allocate(device)
            setattr(self, name, value)

    def _prepare_forward(self, block_table, forward_batch, width):
        batch = forward_batch.batch_size
        self._fused_prepare_lim_metadata(
            seq_lens=forward_batch.seq_lens[:batch],
            positions=forward_batch.positions[: batch * width],
            request_slots=forward_batch.req_pool_indices,
            last_prefix=self.last_prefix,
            block_table=block_table,
            out_block_table=self.block_table[:batch],
            safe_slots=self.request_safe_slots[:batch],
            lengths=self._pre_lengths[:batch],
            sparse_rows=self._pre_sparse[:batch],
            valid_rows=self._pre_valid[:batch],
            request_states=self._pre_states[:batch],
            next_prefixes=self._pre_next_prefix[:batch],
            cache_sizes=self._pre_cache_sizes[:batch],
            capacity=self.capacity,
            request_rows=self.rows,
            query_width=width,
            resident_safe_slots=self.resident_safe_slots[:batch],
            request_to_resident=self.request_to_resident,
            resident_rows=self.resident_rows,
        )
        self._prepared_batch = forward_batch

    def select(
        self,
        *,
        query,
        weights,
        key,
        block_table,
        forward_batch,
        slot_pool: torch.Tensor,
        reuse_metadata=False,
    ):
        batch = forward_batch.batch_size
        width = sparse_query_width(forward_batch)
        rows = batch * width
        # Interleaved batches share the adapter scratch. Rebuild it when a
        # different batch has used it since this batch's preceding owner.
        if not reuse_metadata or self._prepared_batch is not forward_batch:
            self._prepare_forward(block_table, forward_batch, width)
        sources = self.source_ids[:rows]
        fused_lightning_indexer_manage(
            weights=weights[:rows].contiguous(),
            query=query[:rows].contiguous(),
            key=key,
            block_table=self.block_table[:batch],
            query_ends=self.query_ends[width - 1, :batch],
            seq_lens=self._pre_lengths[:batch],
            prefix_lens=self._pre_lengths[:batch],
            cache_sizes=self._pre_cache_sizes[:batch],
            request_state=self._pre_states[:batch],
            request_slots=self.resident_safe_slots[:batch],
            slot_pool=slot_pool,
            source_ids=sources,
            device_slots=self.device_slots[:rows],
            topk_misses=self.topk_misses[:rows],
            miss_sources=self.miss_sources[:batch],
            miss_slots=self.miss_slots[:batch],
            miss_counts=self.miss_counts[:batch],
            operator=self.operator,
            full_visible=True,
        )
        return LIMSelection(
            sources,
            self.device_slots[:rows],
            self.topk_misses[:rows],
            self.miss_sources[:batch],
            self.miss_slots[:batch],
            self.miss_counts[:batch],
            self._pre_sparse[:batch],
            self._pre_valid[:batch],
            self.request_safe_slots[:batch],
            self.resident_safe_slots[:batch],
            width,
        )

    def commit_forward(self, batch: int) -> None:
        """Publish the visible lengths after the last layer has copied its KV."""
        self.last_prefix.index_copy_(
            0, self.resident_safe_slots[:batch].long(), self._pre_next_prefix[:batch]
        )
        self._prepared_batch = None
