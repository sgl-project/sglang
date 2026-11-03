"""Q1 framework bridge from the DSA indexer to native LIM cache plans.

By default LIM selects a page-aligned prefix and appends its visible tail.
With Copy-SFA, Q1 and K=2048 select over the complete visible sequence;
the resident cache capacity C can be larger than K.
Runtime state belongs to this manager-owned adapter, while the current plan
travels on Ascend ForwardMetadata for IndexShare consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.npu_lim import (
    fused_lightning_indexer_manage,
    load_fused_lim_operator,
)
from sglang.srt.environ import envs


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


def prepare_lim_rows(
    seq_lens, request_slots, last_prefix, capacity, request_rows, *, full_visible=False
):
    """Derive native states without host scalar reads, including padding/reset."""
    valid = (request_slots > 0) & (request_slots < request_rows) & (seq_lens > 0)
    safe_slots = torch.where(valid, request_slots, request_rows).to(torch.int32)
    lengths = torch.where(valid, seq_lens, 0).to(torch.int32)
    prefix = lengths if full_visible else ((lengths - 1).clamp(min=0) // 128) * 128
    sparse = valid & (prefix >= capacity)
    previous = last_prefix[safe_slots.long()]
    warm = sparse & (previous >= 0) & (prefix >= previous)
    state = torch.where(sparse, torch.where(warm, -1, -2), -3).to(torch.int32)
    # Native invalid rows take its seq_len < Q guard before touching pool state.
    next_prefix = torch.where(sparse, prefix, -1)
    return safe_slots, lengths, prefix, sparse, valid, state, next_prefix


class LIMDecodeAdapter:
    def __init__(self, manager):
        self.impl = manager.indexer_impl
        self.is_native = self.impl == "lim"
        self.capacity = manager.device_cache_size
        self.rows = manager.size
        self.batch_capacity = manager.batch_capacity
        self.top_k = manager.sparse_context_len
        self.full_visible = manager.copy_sfa_enabled
        self.fused_copy_metadata = (
            self.is_native
            and self.full_visible
            and envs.SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA.get()
        )
        self.fused_pre_metadata = (
            self.is_native
            and self.full_visible
            and self.top_k == 2048
            and envs.SGLANG_NPU_FUSED_LIM_PRE_METADATA.get()
        )
        self.source_capacity = (manager.max_context_len + 127) // 128 * 128
        self.width = self.top_k if self.full_visible else self.top_k + 128
        device = manager.device
        self.operator = load_fused_lim_operator() if self.is_native else None
        self.source_ids = torch.empty(
            (self.batch_capacity, 1, self.top_k), dtype=torch.int32, device=device
        )
        self.device_slots = torch.empty_like(self.source_ids)
        self.topk_misses = torch.empty(
            self.batch_capacity, dtype=torch.int32, device=device
        )
        self.miss_sources = torch.empty(
            (self.batch_capacity, 32768), dtype=torch.int32, device=device
        )
        self.miss_slots = torch.empty_like(self.miss_sources)
        self.miss_counts = torch.empty_like(self.topk_misses)
        self.query_ends = torch.arange(
            1, self.batch_capacity + 1, dtype=torch.int32, device=device
        )
        self.query_scale = torch.empty(
            (self.batch_capacity, 64), dtype=torch.float32, device=device
        )
        self.block_table = torch.empty(
            (self.batch_capacity, self.source_capacity // 128),
            dtype=torch.int32,
            device=device,
        )
        self.last_prefix = torch.full(
            (manager.layer_num, self.rows + 1), -1, dtype=torch.int32, device=device
        )
        if self.fused_pre_metadata:
            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_input_metadata import (
                fused_prepare_lim_metadata,
            )

            self._fused_prepare_lim_metadata = fused_prepare_lim_metadata
            self._pre_safe_slots = torch.empty(
                self.batch_capacity, dtype=torch.int32, device=device
            )
            self._pre_lengths = torch.empty_like(self._pre_safe_slots)
            self._pre_prefixes = torch.empty_like(self._pre_safe_slots)
            self._pre_sparse = torch.empty(
                self.batch_capacity, dtype=torch.bool, device=device
            )
            self._pre_valid = torch.empty_like(self._pre_sparse)
            self._pre_states = torch.empty_like(self._pre_safe_slots)
            self._pre_next_prefix = torch.empty_like(self._pre_safe_slots)
            self._pre_cache_sizes = torch.empty_like(self._pre_safe_slots)
        if self.fused_copy_metadata:
            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_copy_sfa_metadata import (
                fused_lim_copy_sfa_metadata,
            )

            self._fused_copy_sfa_metadata = fused_lim_copy_sfa_metadata
            # The fused kernel writes prefix state. Keep that write private: select
            # has already published the real state, even for short/fallback rows.
            self._metadata_prefix_scratch = torch.full(
                (self.rows + 1,), -1, dtype=torch.int32, device=device
            )
            self._metadata_next_scratch = torch.full(
                (self.batch_capacity,), -1, dtype=torch.int32, device=device
            )
        self.mtp_full_visible = envs.SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE.get()
        if self.mtp_full_visible:
            # The diagnostic integration supports the target-verify Q4 shape.
            rows = self.batch_capacity * 4
            self.mtp_source_ids = torch.empty(
                (rows, 1, self.top_k), dtype=torch.int32, device=device
            )
            self.mtp_device_slots = torch.empty_like(self.mtp_source_ids)
            self.mtp_topk_misses = torch.empty(rows, dtype=torch.int32, device=device)
            self.mtp_query_scale = torch.empty(
                (rows, 64), dtype=torch.float32, device=device
            )
            self.mtp_query_ends = torch.arange(
                4, rows + 1, 4, dtype=torch.int32, device=device
            )
        self.tail_offsets = torch.arange(128, dtype=torch.int32, device=device)
        self.combined_kv = self.combined_ids = None
        if not self.full_visible:
            self.combined_kv = torch.empty(
                (self.batch_capacity, self.width, manager.head_num, manager.head_dim),
                dtype=manager.store_dtype,
                device=device,
            )
            self.combined_ids = torch.empty(
                (self.batch_capacity, self.width), dtype=torch.int32, device=device
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
        use_fused_pre_metadata = self.fused_pre_metadata and query.shape[1] == 1
        if (
            batch != forward_batch.req_pool_indices.numel()
            or batch > self.batch_capacity
        ):
            raise ValueError(
                "LIM Q1 requires one query row per padded request and batch <= graph capacity."
            )
        if (
            block_table.shape[0] != batch
            or block_table.shape[1] > self.block_table.shape[1]
        ):
            raise ValueError(
                "LIM index block-table geometry exceeds its fixed source capacity."
            )
        layer_index = layer_id - manager.start_layer
        if use_fused_pre_metadata:
            table = self.block_table[:batch]
            self._fused_prepare_lim_metadata(
                seq_lens=forward_batch.seq_lens[:batch],
                request_slots=forward_batch.req_pool_indices,
                last_prefix=self.last_prefix[layer_index],
                block_table=block_table,
                out_block_table=table,
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
            )
            slots = self._pre_safe_slots[:batch]
            lengths = self._pre_lengths[:batch]
            prefix = self._pre_prefixes[:batch]
            sparse = self._pre_sparse[:batch]
            valid = self._pre_valid[:batch]
            state = self._pre_states[:batch]
            next_prefix = self._pre_next_prefix[:batch]
        else:
            slots, lengths, prefix, sparse, valid, state, next_prefix = (
                prepare_lim_rows(
                    forward_batch.seq_lens[:batch],
                    forward_batch.req_pool_indices,
                    self.last_prefix[layer_index],
                    self.capacity,
                    self.rows,
                    full_visible=self.full_visible,
                )
            )
        if self.full_visible and not getattr(
            forward_batch, "sparse_kv_copy_sfa_capture", False
        ):
            cpu_lengths = forward_batch.seq_lens_cpu
            # Mixed/short batches use a host gather without populating resident
            # KV. Force the following fused step to cold-fill its cache.
            if cpu_lengths is None or any(
                0 < int(length) < self.capacity for length in cpu_lengths.tolist()
            ):
                if use_fused_pre_metadata:
                    next_prefix.fill_(-1)
                else:
                    next_prefix = torch.full_like(next_prefix, -1)
        if not use_fused_pre_metadata:
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
                cache_sizes=(
                    self._pre_cache_sizes[:batch]
                    if use_fused_pre_metadata
                    else torch.where(sparse, self.capacity, 0).to(torch.int32)
                ),
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

    def select_mtp(
        self, manager, *, query, weights, key, block_table, forward_batch, layer_id
    ):
        """Build one Q4 union plan, retained through all IndexShare layers."""
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
            prepare_mtp_rows,
            validate_mtp_chain,
        )

        batch = forward_batch.batch_size
        rows = batch * 4
        if not self.mtp_full_visible or not self.is_native or query.shape[0] < rows:
            raise ValueError("Experimental native MTP requires Q4 native LIM.")
        if forward_batch.out_cache_loc is None:
            raise ValueError("Native MTP verify requires per-token KV locations.")
        _, positions, live = prepare_mtp_rows(
            forward_batch, query.shape[0], manager.size, manager.logical_max_context_len
        )
        if positions.shape[1] != 4 or batch > self.batch_capacity:
            raise ValueError(
                "Experimental native MTP requires Q4 within graph capacity."
            )
        validate_mtp_chain(forward_batch, positions, live, self.rows)
        valid = live.all(dim=1)
        lengths = torch.where(valid, positions[:, -1] + 1, 0).to(torch.int32)
        sparse = valid & (lengths >= self.capacity)
        slots = torch.where(valid, forward_batch.req_pool_indices, self.rows).to(
            torch.int32
        )
        layer_index = layer_id - manager.start_layer
        previous = self.last_prefix[layer_index][slots.long()]
        warm = sparse & (previous >= 0) & (lengths >= previous)
        states = torch.where(sparse, torch.where(warm, -1, -2), -3).to(torch.int32)
        if block_table.shape[0] != batch:
            raise ValueError("MTP LIM block-table batch must match the request batch.")
        # Graph metadata may reserve extra speculative pages beyond the logical
        # host context. prepare_mtp_rows rejects positions outside that context.
        table = self.block_table[:batch]
        table.zero_()
        columns = min(block_table.shape[1], table.shape[1])
        if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
            from sglang.srt.utils.async_probe import maybe_assert_async

            maybe_assert_async(
                ((~valid) | (lengths <= columns * 128)).all(),
                "Native MTP index block table does not cover the live context.",
            )
        table[:, :columns].copy_(block_table[:, :columns])
        sources = self.mtp_source_ids[:rows]
        fused_lightning_indexer_manage(
            weights=weights[:rows].contiguous(),
            query_scale=self.mtp_query_scale.view(-1)[: rows * query.shape[1]].view(
                rows, query.shape[1]
            ),
            query=query[:rows].contiguous(),
            key_scale=self.key_scale,
            key=key,
            block_table=table,
            query_ends=self.mtp_query_ends[:batch],
            seq_lens=lengths,
            prefix_lens=lengths,
            cache_sizes=torch.where(sparse, self.capacity, 0).to(torch.int32),
            request_state=states,
            request_slots=slots,
            slot_pool=manager.device_slot_map[layer_index],
            source_ids=sources,
            device_slots=self.mtp_device_slots[:rows],
            topk_misses=self.mtp_topk_misses[:rows],
            miss_sources=self.miss_sources[:batch],
            miss_slots=self.miss_slots[:batch],
            miss_counts=self.miss_counts[:batch],
            operator=self.operator,
            full_visible=True,
        )
        sources.masked_fill_(~valid[:, None].expand(batch, 4).reshape(rows, 1, 1), -1)
        next_prefix = torch.where(sparse, lengths, -1)
        self.last_prefix[layer_index].index_copy_(0, slots.long(), next_prefix)
        return LIMSelection(
            sources[:, 0],
            self.mtp_device_slots[:rows, 0],
            self.mtp_topk_misses[:rows],
            self.miss_sources[:batch],
            self.miss_slots[:batch],
            self.miss_counts[:batch],
            lengths,
            sparse,
            valid,
            slots,
            lengths,
        )

    def copy_sfa_metadata(self, manager, plan):
        """Pass graph-stable native LIM buffers directly, without replanning."""
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.block_tables import (
            fill_request_block_table,
        )

        batch = plan.source_ids.shape[0]
        if self.fused_copy_metadata:
            # Rebuild shared manager buffers on every consumer. IndexShare or
            # another path may have reused them since the preceding call.
            self._fused_copy_sfa_metadata(
                source_ids=plan.source_ids,
                topk_misses=plan.topk_misses,
                miss_counts=plan.miss_counts,
                valid_rows=plan.valid_rows,
                sparse_rows=plan.sparse_rows,
                request_slots=plan.request_slots,
                next_prefix=self._metadata_next_scratch[:batch],
                last_prefix=self._metadata_prefix_scratch,
                actual_seq_lengths_kv=manager.planner_actual_seq_lengths_kv[:batch],
                num_cache_tokens=manager.planner_num_cache_tokens[:batch],
                hbm_block_table=manager.planner_hbm_block_table[:batch],
                dram_block_table=manager.planner_dram_block_table[:batch],
                capacity=self.capacity,
                host_context_len=manager.max_context_len,
                request_rows=self.rows,
            )
        else:
            valid = plan.valid_rows & plan.sparse_rows
            # Invalid native queries may leave old output contents. Zero counts
            # prevent either copy phase from reading their unused source/slot rows.
            plan.topk_misses.masked_fill_(~valid, 0)
            plan.miss_counts.masked_fill_(~valid, 0)
            # Preserve the full contiguous [B,32768] rows. A [:,:K] slice would
            # retain the native row stride and fail the Copy-SFA layout contract.
            lengths = valid.to(torch.int32) * self.capacity
            manager.planner_actual_seq_lengths_kv[:batch].copy_(lengths)
            manager.planner_num_cache_tokens[:batch].copy_(lengths)
            for pages, offsets, output in (
                (
                    self.capacity // 128,
                    manager.hbm_page_offsets,
                    manager.planner_hbm_block_table,
                ),
                (
                    manager.max_context_len // 128,
                    manager.dram_page_offsets,
                    manager.planner_dram_block_table,
                ),
            ):
                fill_request_block_table(
                    plan.request_slots, valid, offsets, pages, output[:batch]
                )
        return (
            plan.source_ids.unsqueeze(1),
            plan.device_slots.unsqueeze(1),
            plan.topk_misses,
            plan.miss_sources,
            plan.miss_slots,
            plan.miss_counts,
            manager.planner_actual_seq_lengths_query[:batch],
            manager.planner_actual_seq_lengths_kv[:batch],
            manager.planner_num_cache_tokens[:batch],
            manager.planner_hbm_block_table[:batch],
            manager.planner_dram_block_table[:batch],
        )

    def gather_full_visible(self, manager, plan, layer_index, selected, *, copy):
        """Short/mixed fallback: gather the selected IDs, preserving resident KV."""
        batch = selected.shape[0]
        sources = plan.source_ids.long()
        valid = (
            plan.valid_rows[:, None]
            & (sources >= 0)
            & (sources < plan.seq_lens[:, None])
        )
        rows = plan.request_slots.long().clamp(max=self.rows - 1)
        copy(
            manager.host_kv_buffer[layer_index],
            selected,
            (rows[:, None] * manager.max_context_len + sources)
            .reshape(-1)
            .contiguous(),
            torch.arange(batch * self.top_k, device=selected.device),
            valid.reshape(-1).contiguous(),
            2,
            2,
            block_dim=24,
            src_ptr=manager.dev_ptr_list[layer_index],
        )
        stream = torch.npu.current_stream(manager.device)
        stream.record_event(manager.miss_done)
        stream.record_event(manager.refill_done)

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
                block_dim=48,
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
