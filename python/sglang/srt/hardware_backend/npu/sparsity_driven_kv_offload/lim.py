"""Q1 framework bridge from the DSA indexer to native LIM cache plans.

LIM selects TopK from a stable page-aligned prefix. Both experimental modes
append the same visible tail to attention; the default legacy selection policy
is unchanged. Runtime state belongs to this manager-owned adapter, while the
current plan travels on Ascend ForwardMetadata for IndexShare consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.npu_lim import (
    fused_lightning_indexer_manage,
    load_fused_lim_operator,
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


def prepare_lim_rows(seq_lens, request_slots, last_prefix, capacity, request_rows):
    """Derive native states without host scalar reads, including padding/reset."""
    valid = (request_slots > 0) & (request_slots < request_rows) & (seq_lens > 0)
    safe_slots = torch.where(valid, request_slots, request_rows).to(torch.int32)
    lengths = torch.where(valid, seq_lens, 0).to(torch.int32)
    prefix = ((lengths - 1).clamp(min=0) // 128) * 128
    sparse = valid & (prefix >= capacity)
    previous = last_prefix[safe_slots.long()]
    warm = sparse & (previous >= 0) & (prefix >= previous)
    state = torch.where(sparse, torch.where(warm, -1, -2), -3).to(torch.int32)
    # Native invalid rows take its seq_len < Q guard before touching pool state.
    next_prefix = torch.where(sparse, prefix, -1)
    return safe_slots, lengths, prefix, sparse, valid, state, next_prefix


class LIMDecodeAdapter:
    def __init__(self, manager, *, library=None):
        self.impl = manager.indexer_impl
        self.is_native = self.impl == "lim"
        self.capacity = manager.device_cache_size
        self.rows = manager.size
        self.top_k = manager.sparse_context_len
        self.source_capacity = (manager.max_context_len + 127) // 128 * 128
        self.width = self.top_k + 128
        device = manager.device
        self.operator = load_fused_lim_operator(library) if self.is_native else None
        self.source_ids = torch.empty(
            (self.rows, 1, self.top_k), dtype=torch.int32, device=device
        )
        self.device_slots = torch.empty_like(self.source_ids)
        self.topk_misses = torch.empty(self.rows, dtype=torch.int32, device=device)
        self.miss_sources = torch.empty(
            (self.rows, 32768), dtype=torch.int32, device=device
        )
        self.miss_slots = torch.empty_like(self.miss_sources)
        self.miss_counts = torch.empty_like(self.topk_misses)
        self.query_ends = torch.arange(
            1, self.rows + 1, dtype=torch.int32, device=device
        )
        self.query_scale = torch.empty(
            (self.rows, 64), dtype=torch.float32, device=device
        )
        self.block_table = torch.empty(
            (self.rows, self.source_capacity // 128), dtype=torch.int32, device=device
        )
        self.last_prefix = torch.full(
            (manager.layer_num, self.rows + 1), -1, dtype=torch.int32, device=device
        )
        self.tail_offsets = torch.arange(128, dtype=torch.int32, device=device)
        self.combined_kv = torch.empty(
            (self.rows, self.width, manager.head_num, manager.head_dim),
            dtype=manager.store_dtype,
            device=device,
        )
        self.combined_ids = torch.empty(
            (self.rows, self.width), dtype=torch.int32, device=device
        )
        # One scale workspace is shared by all indexer layers. Native BF16 LIM
        # does not read its values, but its ABI requires the complete PA shape.
        # Its variable 4 bytes/index-token are charged to the indexer pool coeff.
        key = manager.paged_kv_cache.get_index_k_buffer(manager.start_layer)
        if key.dtype != torch.bfloat16 or manager.store_dtype != torch.bfloat16:
            raise ValueError(
                "LIM/reference framework integration requires BF16 indexer and KV caches."
            )
        if key.ndim != 4 or key.shape[1:] != (128, 1, 128):
            raise ValueError("LIM/reference requires PA index keys [pages,128,1,128].")
        self.key_scale = (
            torch.empty(key.shape[:3], dtype=torch.float32, device=device)
            if self.is_native
            else None
        )

    def reset_requests(self, request_slots):
        self.last_prefix[:, request_slots] = -1

    def select(
        self,
        manager,
        *,
        query,
        weights,
        key,
        block_table,
        forward_batch,
        layer_id,
        reference_indexer=None,
    ):
        batch = query.shape[0]
        if batch != forward_batch.req_pool_indices.numel() or batch > self.rows:
            raise ValueError(
                "LIM Q1 requires one query row per padded request and batch <= request capacity."
            )
        if (
            block_table.shape[0] != batch
            or block_table.shape[1] > self.block_table.shape[1]
        ):
            raise ValueError(
                "LIM index block-table geometry exceeds its fixed source capacity."
            )
        layer_index = layer_id - manager.start_layer
        slots, lengths, prefix, sparse, valid, state, next_prefix = prepare_lim_rows(
            forward_batch.seq_lens[:batch],
            forward_batch.req_pool_indices,
            self.last_prefix[layer_index],
            self.capacity,
            self.rows,
        )
        table = self.block_table[:batch]
        table.zero_()
        table[:, : block_table.shape[1]].copy_(block_table)
        sources = self.source_ids[:batch]
        if self.is_native:
            # [:, :heads] would be strided for H32; view a contiguous prefix.
            scale = self.query_scale.view(-1)[: batch * query.shape[1]].view(
                batch, query.shape[1]
            )
            fused_lightning_indexer_manage(
                weights=weights.contiguous(),
                query_scale=scale,
                query=query.contiguous(),
                key_scale=self.key_scale,
                key=key,
                block_table=table,
                query_ends=self.query_ends[:batch],
                seq_lens=lengths,
                prefix_lens=prefix,
                cache_sizes=torch.where(sparse, self.capacity, 0).to(torch.int32),
                request_state=state,
                request_slots=slots,
                slot_pool=manager.device_slot_map[layer_index],
                source_ids=sources,
                device_slots=self.device_slots[:batch],
                topk_misses=self.topk_misses[:batch],
                miss_sources=self.miss_sources[:batch],
                miss_slots=self.miss_slots[:batch],
                miss_counts=self.miss_counts[:batch],
                operator=self.operator,
            )
        else:
            result = reference_indexer(
                query=query.contiguous(),
                key=key,
                weights=weights.contiguous(),
                actual_seq_lengths_query=self.query_ends[:batch],
                actual_seq_lengths_key=torch.where(
                    sparse, prefix, lengths.clamp(min=1)
                ),
                block_table=table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.top_k,
                sparse_mode=3,
            )
            indices = result[0] if isinstance(result, (tuple, list)) else result
            sources.copy_(indices.view(batch, 1, self.top_k))
        # The native invalid-query fast path can leave reused output rows
        # untouched during graph replay. Never publish a previous request's IDs.
        sources.masked_fill_(~valid[:, None, None], -1)
        self.last_prefix[layer_index].index_copy_(0, slots.long(), next_prefix)
        return LIMSelection(
            sources[:, 0],
            self.device_slots[:batch, 0],
            self.topk_misses[:batch],
            self.miss_sources[:batch],
            self.miss_slots[:batch],
            self.miss_counts[:batch],
            prefix,
            sparse,
            valid,
            slots,
            lengths,
        )

    def materialize(self, manager, plan, layer_index, selected, stream, *, copy):
        """Consume native plans directly; no Python lookup, sort or LRU rebuild."""
        batch = selected.shape[0]
        with torch.npu.stream(stream):
            req = plan.request_slots.long()
            safe_req = req.clamp(max=self.rows - 1)
            miss_src = plan.miss_sources[:, : self.capacity].long()
            miss_dst = plan.miss_slots[:, : self.capacity].long()
            offsets = torch.arange(self.capacity, device=selected.device)
            misses = (offsets[None] < plan.miss_counts[:, None]) & plan.sparse_rows[
                :, None
            ]
            # First fill can contain C misses, not just K. Consume the complete
            # request-level plan before any selected slot is read.
            copy(
                manager.host_kv_buffer[layer_index],
                manager.device_kv_buffer[layer_index],
                (safe_req[:, None] * manager.max_context_len + miss_src)
                .reshape(-1)
                .contiguous(),
                (safe_req[:, None] * self.capacity + miss_dst).reshape(-1).contiguous(),
                misses.reshape(-1).contiguous(),
                2,
                2,
                block_dim=24,
                src_ptr=manager.dev_ptr_list[layer_index],
            )
            valid = (plan.source_ids >= 0) & plan.valid_rows[:, None]
            dst = torch.arange(batch * self.top_k, device=selected.device)
            copy(
                manager.device_kv_buffer[layer_index],
                selected,
                (safe_req[:, None] * self.capacity + plan.device_slots.long())
                .reshape(-1)
                .contiguous(),
                dst,
                (valid & plan.sparse_rows[:, None]).reshape(-1).contiguous(),
                2,
                2,
                block_dim=24,
            )
            copy(
                manager.host_kv_buffer[layer_index],
                selected,
                (safe_req[:, None] * manager.max_context_len + plan.source_ids.long())
                .reshape(-1)
                .contiguous(),
                dst,
                (valid & ~plan.sparse_rows[:, None]).reshape(-1).contiguous(),
                2,
                2,
                block_dim=24,
                src_ptr=manager.dev_ptr_list[layer_index],
            )
            stream.record_event(manager.miss_done)
            stream.record_event(manager.refill_done)

    def append_tail(self, manager, plan, layer_index, selected, *, copy):
        batch = selected.shape[0]
        combined = self.combined_kv[:batch]
        combined.zero_()
        combined[:, : self.top_k].copy_(selected)
        tail = plan.prefix_lens[:, None] + self.tail_offsets[None]
        valid = plan.sparse_rows[:, None] & (tail < plan.seq_lens[:, None])
        ids = self.combined_ids[:batch]
        ids[:, : self.top_k].copy_(plan.source_ids)
        ids[:, self.top_k :].copy_(torch.where(valid, tail, -1))
        request = plan.request_slots.long().clamp(max=self.rows - 1)
        dst = (
            torch.arange(batch, device=selected.device)[:, None] * self.width
            + self.top_k
            + self.tail_offsets[None]
        )
        copy(
            manager.host_kv_buffer[layer_index],
            combined,
            (request[:, None] * manager.max_context_len + tail.long())
            .reshape(-1)
            .contiguous(),
            dst.long().reshape(-1).contiguous(),
            valid.reshape(-1).contiguous(),
            2,
            2,
            block_dim=24,
            src_ptr=manager.dev_ptr_list[layer_index],
        )
        return combined, ids
