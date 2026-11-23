"""Ascend950 payload and FP8 indexer adapters for the shared offload manager."""

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    C8_ROW_BYTES,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
    LIMDecodeAdapter,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    FusedSparseKVCacheManager,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.host_pool import (
    HostKVPool,
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


class PackedMemFabricKVPool(HostKVPool):
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

    def _run_operator(
        self,
        manager,
        layer_index,
        batch,
        width,
        *,
        query,
        weights,
        key,
        query_scale,
        key_scale,
    ):
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
            slot_pool=manager.device_slot_map[layer_index],
            topk_sources=self.source_ids[:rows],
            topk_slots=self.device_slots[:rows],
            topk_miss_counts=self.topk_misses[:rows],
            miss_sources=self.miss_sources[:batch],
            miss_slots=self.miss_slots[:batch],
            miss_counts=self.miss_counts[:batch],
        )


class C8SparseKVCacheManager(FusedSparseKVCacheManager):
    packed_c8 = True

    @staticmethod
    def _load_copy_sfa_operator():
        if not (
            hasattr(torch.ops.npu, "a5_copy_sfa_capabilities")
            and torch.ops.npu.a5_copy_sfa_capabilities().get("resident_source_gather")
            == 1
        ):
            raise RuntimeError(
                "A5 C8 requires the paired native-LIM source-gather CopySFA wheel."
            )
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

    def offload_v2(self, k, k_rope, layer, forward_batch, *, offload_rows=None):
        layer_index = layer.layer_id - self.start_layer
        current = self.paged_kv_cache._pack_dsa_fp8_kv_cache(k, k_rope).view(torch.int8)
        if offload_rows is None:
            offload_rows = self.prepare_offload_rows(k.shape[0], forward_batch)
        src, dst, positions, valid = offload_rows
        if self.host_pool.rank == 0:
            _copy_packed(current, self.host_kv_buffer[layer_index], src, dst, valid)
        self.host_pool.publish()
        if forward_batch.forward_mode.is_extend_without_speculative():
            requests = forward_batch.req_pool_indices.long()
            self.resident_end[layer_index].index_fill_(0, requests, -1)
            if self.request_to_resident is None and layer_index == self.layer_num - 1:
                self.lim_adapter.reset_requests(requests)
        return current, positions, valid

    def copy_sfa_attention(self, layer, query_nope, query_rope, plan, current_inputs):
        layer_index = layer.layer_id - self.start_layer
        batch, width = plan.valid_rows.numel(), plan.query_width
        rows, heads = batch * width, query_nope.shape[1]
        padded_heads = self.budget.copy_sfa_query_heads
        lengths = self.lim_adapter._copy_lengths[:batch]
        current, positions, valid = current_inputs
        hbm_table = self.planner_hbm_block_table[:batch]
        if heads < padded_heads:
            padding = (0, 0, 0, padded_heads - heads)
            query_nope = torch.nn.functional.pad(query_nope, padding)
            query_rope = torch.nn.functional.pad(query_rope, padding)
        output = query_nope.new_empty((rows, padded_heads, 512))
        self._copy_sfa_operator(
            query_nope.contiguous(),
            query_rope.contiguous(),
            self.device_k_buffer[layer_index].view(-1, self.page_size, 1, C8_ROW_BYTES),
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
            current=current[:rows],
            positions=positions[:rows],
            current_valid=valid[:rows],
            request_slots=plan.request_safe_slots,
            req_to_token=self.req_to_token_pool.req_to_token,
            resident_end=self.resident_end[layer_index, : self.actual_request_rows],
            native_lim_plan=True,
        )
        # Invalid requests have positive TND query width and zero KV length.
        # CopySFA initializes their output before skipping them in attention.
        return output[:, :heads].reshape(rows, -1)
