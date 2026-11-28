"""Ascend950 packed-C8 payload and quantized LIM adapters."""

from __future__ import annotations

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
    """Copy opaque 656-byte C8 records without interpreting their fields."""
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
        try:
            from sgl_kernel_npu.operators.indexer.fused_quant_lightning_indexer_manage_full_visible import (
                fused_quant_lightning_indexer_manage_full_visible as operator,
            )

            torch.ops.npu.fused_quant_lightning_indexer_manage_full_visible.default
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "A5 C8 requires the paired fully visible LIM wheel."
            ) from error
        return operator

    def _run_operator(
        self,
        batch,
        width,
        *,
        query,
        weights,
        key,
        slot_pool,
        query_scale,
        key_scale,
    ):
        rows = batch * width
        self.operator(
            query=query[:rows].contiguous(),
            weights=weights[:rows].contiguous(),
            key=key,
            query_scale=query_scale[:rows].contiguous(),
            key_scale=key_scale,
            block_table=self.block_table[:batch],
            query_ends=self.query_ends[width - 1, :batch],
            lengths=self._pre_lengths[:batch],
            cache_tokens=self._pre_cache_sizes[:batch],
            request_state=self._pre_states[:batch],
            pool_entries=self.resident_safe_slots[:batch],
            slot_pool=slot_pool,
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
        capabilities = getattr(torch.ops.npu, "a5_copy_sfa_capabilities", None)
        if capabilities is None or capabilities().get("resident_source_gather") != 1:
            raise RuntimeError(
                "A5 C8 requires the paired native-LIM source-gather CopySFA wheel."
            )
        try:
            from sgl_kernel_npu.operators.attention.fused_quant_scatter_copy_sparse_flash_attention_v2 import (
                fused_quant_scatter_copy_sparse_flash_attention_v2 as operator,
            )

            torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2.default
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "A5 C8 requires the paired native-LIM source-gather CopySFA wheel."
            ) from error
        return operator

    def _make_decode_adapter(self):
        return C8DecodeAdapter(
            self.budget,
            device=self.device,
            last_prefix=self.resident_pool.last_prefix,
            request_to_resident=self.resident_pool.request_to_resident,
        )

    def _init_host_pool(self) -> None:
        self.host_pool = PackedMemFabricKVPool(
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
        self.dev_ptr_list = [buffer.data_ptr() for buffer in self.host_kv_buffer]

    def offload_v2(self, k, k_rope, layer, forward_batch, *, offload_rows=None):
        from sglang.srt.hardware_backend.npu.kv_capability import (
            DSA_KV_QUANT_BLOCK_SIZE,
        )
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            pack_dsa_quant_kv_cache,
        )

        layer_index = layer.layer_id - self.start_layer
        packed = pack_dsa_quant_kv_cache(
            k,
            k_rope,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            quant_block_size=DSA_KV_QUANT_BLOCK_SIZE,
            dtype=torch.float8_e4m3fn,
        )
        current = packed.view(torch.int8)
        if offload_rows is None:
            offload_rows = self.prepare_offload_rows(k.shape[0], forward_batch)
        src, dst, positions, valid = offload_rows
        if self.host_pool.rank == 0:
            _copy_packed(current, self.host_kv_buffer[layer_index], src, dst, valid)
        self.host_pool.publish()

        if forward_batch.forward_mode.is_extend_without_speculative():
            requests = forward_batch.req_pool_indices[: forward_batch.batch_size].to(
                torch.long
            )
            self.resident_pool.short_resident_end[layer_index].index_fill_(
                0, requests, -1
            )
            if (
                self.resident_pool.request_to_resident is None
                and layer_index == self.layer_num - 1
            ):
                # Mixed/recomputed prefill can reuse an identity request row.
                # Force the next LIM call through first-fill so an old slot map
                # cannot be interpreted as warm for the rewritten Host payload.
                self.resident_pool.last_prefix.index_fill_(0, requests, -1)
        return current, positions, valid

    def copy_sfa_attention(self, layer, query_nope, query_rope, plan, current_inputs):
        layer_index = layer.layer_id - self.start_layer
        batch, width = plan.valid_rows.numel(), plan.query_width
        rows, query_heads = batch * width, query_nope.shape[1]
        padded_heads = self.budget.copy_sfa_query_heads
        current, positions, valid = current_inputs
        if query_heads < padded_heads:
            padding = (0, 0, 0, padded_heads - query_heads)
            query_nope = torch.nn.functional.pad(query_nope, padding)
            query_rope = torch.nn.functional.pad(query_rope, padding)
        output = query_nope.new_empty((rows, padded_heads, self.kv_lora_rank))
        resident_pool = self.resident_pool
        self._copy_sfa_operator(
            query_nope.contiguous(),
            query_rope.contiguous(),
            resident_pool.device_k_buffer[layer_index].view(
                -1, self.page_size, 1, C8_ROW_BYTES
            ),
            self.host_kv_buffer[layer_index],
            self.dev_ptr_list[layer_index],
            self.planner_hbm_block_table[:batch],
            self.lim_adapter.block_table[:batch],
            plan.device_slots.unsqueeze(1),
            plan.source_ids.unsqueeze(1),
            plan.topk_misses,
            self.lim_adapter.query_ends[width - 1, :batch],
            self.lim_adapter.actual_seq_lengths_kv[:batch],
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
            resident_end=resident_pool.short_resident_end[
                layer_index, : self.actual_request_rows
            ],
            native_lim_plan=True,
        )
        # Invalid requests retain positive TND width but have zero KV length.
        return output[:, :query_heads].reshape(rows, -1)
