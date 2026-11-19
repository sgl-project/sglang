"""Ascend950 payload and FP8 indexer adapters for the shared offload manager."""

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

from sglang.kernels.ops.kvcache.npu_sparse_kv import prepare_sparse_kv_rows
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    C8_ROW_BYTES,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
    LIMDecodeAdapter,
    sparse_query_width,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    FusedSparseKVCacheManager,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.memfabric_pool import (
    MemFabricKVPool,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_copy_sfa_metadata import (
    prepare_copy_sfa_metadata,
)
from sglang.srt.runtime_context import get_parallel


def _copy_packed(source, destination, src, dst, valid):
    unidex_copy_inplace(
        source.view(-1, C8_ROW_BYTES),
        destination.view(-1, C8_ROW_BYTES),
        src.reshape(-1).long(),
        dst.reshape(-1).long(),
        valid.reshape(-1),
        1,
        1,
        block_dim=48,
        src_ptr=source.data_ptr() if source.is_cpu else None,
        dst_ptr=destination.data_ptr() if destination.is_cpu else None,
    )


class PackedMemFabricKVPool(MemFabricKVPool):
    def copy_rows(self, sources, destinations, source_rows, destination_rows, valid):
        (source,), (destination,) = sources, destinations
        if source_rows.numel():
            _copy_packed(source, destination, source_rows, destination_rows, valid)


class C8DecodeAdapter(LIMDecodeAdapter):
    @staticmethod
    def _load_operator():
        operator = getattr(
            torch.ops.npu, "fused_quant_lightning_indexer_manage_full_visible", None
        )
        if operator is None:
            raise RuntimeError("A5 C8 requires the paired fully visible LIM wheel.")
        return operator

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
        query_scale,
        key_scale,
    ):
        layer = layer_id - manager.start_layer
        batch, width = self.prepare(manager, block_table, forward_batch, layer)
        rows = batch * width
        self.operator(
            query=query[:rows].contiguous(),
            weights=weights[:rows],
            key=key,
            query_scale=query_scale[:rows].contiguous(),
            key_scale=key_scale,
            block_table=self.block_table[:batch],
            query_ends=self.query_ends[width - 1, :batch],
            lengths=self._pre_lengths[:batch],
            cache_tokens=self._pre_cache_sizes[:batch],
            request_state=self._pre_states[:batch],
            pool_entries=self.resident_safe_slots[:batch],
            slot_pool=manager.device_slot_map[layer],
            topk_sources=self.source_ids[:rows],
            topk_slots=self.device_slots[:rows],
            topk_miss_counts=self.topk_misses[:rows],
            miss_sources=self.miss_sources[:batch],
            miss_slots=self.miss_slots[:batch],
            miss_counts=self.miss_counts[:batch],
        )
        return self.finish(layer, batch, width)


class C8SparseKVCacheManager(FusedSparseKVCacheManager):
    packed_c8 = True

    @staticmethod
    def _load_copy_sfa_operator():
        if not (
            hasattr(torch.ops.npu, "a5_copy_sfa_capabilities")
            and torch.ops.npu.a5_copy_sfa_capabilities().get("full_visible") == 1
        ):
            raise RuntimeError("A5 C8 requires the paired fully visible CopySFA wheel.")
        return torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2

    def _make_decode_adapter(self):
        return C8DecodeAdapter(self)

    def _init_host_pool(self):
        self.host_pool = self.mf_host_pool = PackedMemFabricKVPool(
            tokens=self.host_token_capacity,
            layers=self.layer_num,
            page_size=self.page_size,
            head_num=1,
            widths=(C8_ROW_BYTES,),
            dtype=torch.int8,
            device=self.device,
            group=get_parallel().attn_tp_group,
        )
        (self.host_kv_buffer,) = self.host_pool.buffers
        self.dev_ptr_list = [x.data_ptr() for x in self.host_kv_buffer]

    def invalidate_packed_request(self, request_id):
        resident_id = (
            request_id
            if self.request_to_resident is None
            else self._resident_rows_by_request[request_id]
        )
        if resident_id:
            self.reset_requests([request_id], [resident_id])

    def offload_v2(self, k, k_rope, layer, forward_batch):
        layer_index = layer.layer_id - self.start_layer
        current = self.paged_kv_cache._pack_dsa_fp8_kv_cache(k, k_rope).view(torch.int8)
        prefill = forward_batch.forward_mode.is_extend_without_speculative()
        src, dst, positions, valid = prepare_sparse_kv_rows(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            None if prefill else forward_batch.positions,
            forward_batch.out_cache_loc,
            tokens=current.shape[0],
            device=current.device,
            request_rows=self.size,
            context_len=self.logical_max_context_len,
            host_capacity=self.host_token_capacity,
            query_width=1 if prefill else sparse_query_width(forward_batch),
            extend_seq_lens=forward_batch.extend_seq_lens if prefill else None,
            extend_prefix_lens=forward_batch.extend_prefix_lens if prefill else None,
            compact_prefill=prefill
            and sum(forward_batch.extend_seq_lens_cpu[: forward_batch.batch_size])
            == current.shape[0],
        )
        if self.host_pool.rank == 0:
            _copy_packed(current, self.host_kv_buffer[layer_index], src, dst, valid)
        self.host_pool.publish()
        if prefill:
            requests = forward_batch.req_pool_indices.long()
            self.resident_end[layer_index].index_fill_(0, requests, -1)
            if self.request_to_resident is None:
                self.lim_adapter.last_prefix[layer_index].index_fill_(0, requests, -1)
        return current, positions, valid

    def _restore_short_and_refresh(self, layer, plan, lengths, current_inputs):
        """Restore short identity rows and refresh any newly rewritten resident."""
        current, positions, valid = current_inputs
        batch, width = lengths.numel(), plan.query_width
        capacity = self.cache_capacity
        reqs = torch.where(plan.valid_rows, plan.request_safe_slots, 0).long()
        base = (plan.resident_safe_slots.long() - 1) * capacity
        short = plan.valid_rows & ~plan.sparse_rows
        first = lengths - width
        old_end = self.resident_end[layer].index_select(0, reqs).clamp_min(0)
        table = self.req_to_token_pool.req_to_token
        history = torch.arange(min(capacity, table.shape[1]), device=self.device)
        host_sources = table[reqs[:, None], history]
        present = short[:, None] & (history >= torch.minimum(old_end, first)[:, None])
        present = present & (history < first[:, None])
        present = (
            present & (host_sources >= 0) & (host_sources < self.host_token_capacity)
        )
        cache = self.device_kv_buffer[layer]
        _copy_packed(
            self.host_kv_buffer[layer],
            cache,
            host_sources,
            base[:, None] + history,
            present,
        )

        rows = batch * width
        pos = positions[:rows].reshape(batch, width)
        resident = plan.resident_safe_slots.long()[:, None]
        # IndexShare consumers must use the mapping produced at the owner layer.
        source_map = self.device_slot_map[plan.owner_layer]
        mapped = source_map[resident, pos.clamp(0, source_map.shape[1] - 1)]
        mapped_ok = (
            (pos >= 0)
            & (pos < source_map.shape[1])
            & (mapped >= 0)
            & (mapped < capacity)
        )
        valid_now = valid[:rows].reshape_as(pos) & plan.valid_rows[:, None]
        valid_now = valid_now & torch.where(
            short[:, None], (pos >= 0) & (pos < capacity), mapped_ok
        )
        slots = torch.where(short[:, None], pos, mapped)
        _copy_packed(
            current,
            cache,
            torch.arange(rows, device=self.device),
            base[:, None] + slots,
            valid_now,
        )
        self.resident_end[layer].index_copy_(0, reqs, torch.where(short, lengths, -1))

    def copy_sfa_attention(self, layer, query_nope, query_rope, plan, current_inputs):
        layer_index = layer.layer_id - self.start_layer
        batch, width = plan.valid_rows.numel(), plan.query_width
        rows, heads = batch * width, query_nope.shape[1]
        padded_heads = self.budget.copy_sfa_query_heads
        lengths = self.lim_adapter._pre_lengths[:batch]
        self._restore_short_and_refresh(layer_index, plan, lengths, current_inputs)
        hbm_table = self.planner_hbm_block_table[:batch]
        prepare_copy_sfa_metadata(
            source_ids=plan.source_ids,
            valid_rows=plan.valid_rows,
            sparse_rows=plan.sparse_rows,
            request_slots=plan.resident_safe_slots,
            actual_seq_lengths_kv=lengths,
            hbm_block_table=hbm_table,
            capacity=self.cache_capacity,
            query_width=width,
            hbm_row_offset=1,
        )
        if heads < padded_heads:
            padding = (0, 0, 0, padded_heads - heads)
            query_nope = torch.nn.functional.pad(query_nope, padding)
            query_rope = torch.nn.functional.pad(query_rope, padding)
        output = query_nope.new_empty((rows, padded_heads, 512))
        self._copy_sfa_operator(
            query_nope.contiguous(),
            query_rope.contiguous(),
            self.device_kv_buffer[layer_index].view(
                -1, self.page_size, 1, C8_ROW_BYTES
            ),
            self.host_kv_buffer[layer_index],
            self.dev_ptr_list[layer_index],
            hbm_table,
            self.lim_adapter.block_table[:batch],
            plan.device_slots.unsqueeze(1),
            plan.source_ids.unsqueeze(1),
            plan.topk_misses,
            self.lim_adapter.query_ends[width - 1, :batch],
            lengths,
            self.lim_adapter._pre_cache_sizes[:batch],
            plan.miss_sources,
            plan.miss_slots,
            plan.miss_counts,
            layer.scaling,
            output,
        )
        valid = plan.valid_rows[:, None].expand(batch, width).reshape(rows)
        output.masked_fill_(~valid[:, None, None], 0)
        return output[:, :heads].reshape(rows, -1)
