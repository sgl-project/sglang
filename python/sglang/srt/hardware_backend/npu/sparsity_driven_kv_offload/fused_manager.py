"""Sparsity-driven KV offload manager for the Ascend backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace
from sglang.kernels.ops.attention.npu_copy_sfa import (
    fused_scatter_copy_sparse_flash_attention,
    load_copy_sfa_operator,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.prefill_kv_plan import (
    make_fused_prefill_plan,
    prepare_prefill_gather_rows,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.resident_pool import (
    ResidentKVPool,
)
from sglang.srt.layers.dp_attention import get_eager_max_batch_size
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.host_pool import (
        HostKVPool,
    )
    from sglang.srt.layers.radix_attention import RadixAttention


class FusedSparseKVCacheManager:
    host_pool: HostKVPool
    packed_c8 = False

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool | DecodeReqToTokenPool,
        sparse_topk: int,
        *,
        index_k_pool: MLATokenToKVPool,
    ) -> None:
        self.actual_request_rows = req_to_token_pool.req_to_token.shape[0]
        self.logical_max_context_len = req_to_token_pool.req_to_token.shape[1]
        self.req_to_token_pool = req_to_token_pool
        self.sparse_topk = sparse_topk
        self.device = req_to_token_pool.device
        self.start_layer = index_k_pool.start_layer
        # MLA params
        self.page_size = index_k_pool.page_size
        self.kv_lora_rank = index_k_pool.kv_lora_rank
        self.qk_rope_head_dim = index_k_pool.qk_rope_head_dim
        self.store_dtype = index_k_pool.store_dtype
        self.layer_num = index_k_pool.layer_num
        self.budget = index_k_pool.sparse_kv_budget_plan
        self.budget.validate_batch_size(
            get_eager_max_batch_size(req_to_token_pool.size), "eager"
        )
        self.head_num = self.budget.kv_head_num
        self.size = self.budget.request_rows
        self.max_context_len = self.budget.context_len
        self.host_token_capacity = self.budget.host_token_capacity
        self._copy_sfa_operator = self._load_copy_sfa_operator()

        self.resident_pool = ResidentKVPool(self.budget, self.device)
        try:
            for name, spec in self.budget.manager_buffers.items():
                setattr(self, name, spec.allocate(self.device))
        except Exception as e:  # noqa: BLE001 - preserve the allocation context
            self._raise_buffer_allocation_error(name, e)

        self._init_host_pool()
        self.lim_adapter = self._make_decode_adapter()

        if self.resident_pool.request_to_resident is None:
            self._install_req_alloc_hook(req_to_token_pool)
        else:
            decode_pool = cast("DecodeReqToTokenPool", req_to_token_pool)
            decode_pool.resident_row_lifecycle = self.resident_pool

    @staticmethod
    def _load_copy_sfa_operator():
        return load_copy_sfa_operator()

    def _init_host_pool(self) -> None:
        """Allocate the default split CKV/KPE Host payload pool."""
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.host_pool import (
            HostKVPool,
        )

        self.host_pool = HostKVPool(
            tokens=self.host_token_capacity,
            layers=self.layer_num,
            page_size=self.page_size,
            head_num=self.head_num,
            widths=(self.kv_lora_rank, self.qk_rope_head_dim),
            dtype=self.store_dtype,
            device=self.device,
            group=get_parallel().attn_tp_group,
        )
        self.dev_ptr_list, self.rope_dev_ptr_list = (
            [x.data_ptr() for x in component] for component in self.host_pool.buffers
        )
        self.host_k_buffer, self.host_v_buffer = self.host_pool.buffers

    def _make_decode_adapter(self):
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
            LIMDecodeAdapter,
        )

        return LIMDecodeAdapter(
            self.budget,
            device=self.device,
            last_prefix=self.resident_pool.last_prefix,
            request_to_resident=self.resident_pool.request_to_resident,
        )

    def _raise_buffer_allocation_error(
        self,
        buffer_name: str,
        exc: Exception,
    ) -> None:
        raise RuntimeError(
            "Failed to allocate sparse KV buffer "
            f"{buffer_name}: req_capacity={self.size}, "
            f"max_context_len={self.max_context_len}, "
            f"sparse_topk={self.sparse_topk}. "
            "The sparse KV request capacity may be too large; set a smaller "
            "--max-running-requests for sparse KV offload."
        ) from exc

    def host_source_indices(self, request_ids, positions):
        """Resolve logical positions through the allocator's shared token map."""
        # Invalid/padded entries are masked by the caller. Clamp only their
        # speculative address calculation, never an active token's mapping.
        rows = request_ids.long().clamp(0, self.actual_request_rows - 1)
        columns = positions.long().clamp(0, self.logical_max_context_len - 1)
        return self.req_to_token_pool.req_to_token[rows, columns].long()

    def prepare_copy_sfa_plan(self, plan) -> None:
        from sgl_kernel_npu.operators.attention.prepare_copy_sfa_metadata import (
            prepare_copy_sfa_metadata,
        )

        batch = plan.request_safe_slots.numel()
        adapter = self.lim_adapter
        prepare_copy_sfa_metadata(
            source_ids=plan.source_ids,
            valid_rows=plan.valid_rows,
            sparse_rows=plan.sparse_rows,
            request_slots=plan.resident_safe_slots,
            seq_lens=adapter._pre_lengths[:batch],
            actual_seq_lengths_kv=adapter.actual_seq_lengths_kv[:batch],
            hbm_block_table=self.planner_hbm_block_table[:batch],
            capacity=adapter.capacity,
            query_width=plan.query_width,
        )

    def copy_sfa_attention(
        self,
        layer,
        query_nope,
        query_rope,
        plan,
        current_inputs,
    ):
        batch = plan.request_safe_slots.numel()
        rows = batch * plan.query_width
        adapter = self.lim_adapter
        resident_pool = self.resident_pool
        query_heads = query_nope.shape[1]
        current, positions, valid = current_inputs
        padded_heads = self.budget.copy_sfa_query_heads
        if query_nope.shape[1] < padded_heads:
            padding = padded_heads - query_nope.shape[1]
            query_nope = torch.nn.functional.pad(query_nope, (0, 0, 0, padding))
            query_rope = torch.nn.functional.pad(query_rope, (0, 0, 0, padding))
        output = torch.empty(
            (rows, padded_heads, self.kv_lora_rank),
            dtype=query_nope.dtype,
            device=query_nope.device,
        )
        layer_idx = layer.layer_id - self.start_layer
        cache = resident_pool.device_k_buffer[layer_idx].view(
            -1, self.page_size, self.head_num, self.kv_lora_rank
        )
        rope_cache = resident_pool.device_v_buffer[layer_idx].view(
            -1, self.page_size, self.head_num, self.qk_rope_head_dim
        )
        fused_scatter_copy_sparse_flash_attention(
            query=query_nope.contiguous(),
            query_rope=query_rope.contiguous(),
            hbm_ckv=cache,
            hbm_rope=rope_cache,
            actual_seq_lengths_query=adapter.query_ends[plan.query_width - 1, :batch],
            actual_seq_lengths_kv=adapter.actual_seq_lengths_kv[:batch],
            num_cache_tokens=adapter._pre_cache_sizes[:batch],
            topk_src_ids=plan.source_ids,
            topk_dst_slots=plan.device_slots,
            topk_miss_counts=plan.topk_misses,
            miss_src_ids=plan.miss_sources,
            miss_dst_slots=plan.miss_slots,
            miss_counts=plan.miss_counts,
            hbm_block_table=self.planner_hbm_block_table[:batch],
            dram_block_table=self.lim_adapter.block_table[:batch],
            dram_kv_dev_ptr=self.dev_ptr_list[layer_idx],
            attention_out=output,
            scale_value=layer.scaling,
            operator=self._copy_sfa_operator,
            full_visible=True,
            resident_refresh_tail=plan.query_width > 1,
            current=current,
            positions=positions,
            valid=valid,
            request_slots=plan.request_safe_slots,
            req_to_token=self.req_to_token_pool.req_to_token,
            resident_end=resident_pool.short_resident_end[
                layer_idx, : self.actual_request_rows
            ],
            host_token_capacity=self.host_token_capacity,
            dram_rope_dev_ptr=self.rope_dev_ptr_list[layer_idx],
        )
        return output[:, :query_heads].reshape(rows, -1)

    def _install_req_alloc_hook(
        self, req_to_token_pool: ReqToTokenPool | DecodeReqToTokenPool
    ) -> None:
        req_to_token_pool.register_on_alloc_rows(self.resident_pool.reset_rows)

    def prepare_offload_rows(
        self, tokens: int, forward_batch: ForwardBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from sgl_kernel_npu.operators.mem_cache.prepare_sparse_kv_offload_rows import (
            prepare_sparse_kv_offload_rows,
        )
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
            sparse_query_width,
        )

        is_prefill = forward_batch.forward_mode.is_extend_without_speculative()
        compact_prefill = is_prefill and (
            sum(forward_batch.extend_seq_lens_cpu[: forward_batch.batch_size]) == tokens
        )
        return prepare_sparse_kv_offload_rows(
            tokens,
            self.device,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            None if is_prefill else forward_batch.positions,
            forward_batch.out_cache_loc,
            request_rows=self.size,
            context_len=(
                self.max_context_len if is_prefill else self.logical_max_context_len
            ),
            host_capacity=self.host_token_capacity,
            query_width=1 if is_prefill else sparse_query_width(forward_batch),
            extend_seq_lens=forward_batch.extend_seq_lens if is_prefill else None,
            extend_prefix_lens=forward_batch.extend_prefix_lens if is_prefill else None,
            compact_prefill=compact_prefill,
        )

    def offload_v2(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        *,
        offload_rows: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ):
        """Publish this layer's KV and return its current rows for attention."""
        layer_idx = layer.layer_id - self.start_layer
        is_prefill = forward_batch.forward_mode.is_extend_without_speculative()
        if offload_rows is None:
            offload_rows = self.prepare_offload_rows(k.shape[0], forward_batch)
        src, dst, positions, valid = offload_rows
        current = (k, k_rope) if is_prefill else torch.cat((k, k_rope), dim=-1)
        if self.host_pool.rank == 0:
            self.host_pool.copy_rows(
                (k, k_rope),
                (self.host_k_buffer[layer_idx], self.host_v_buffer[layer_idx]),
                src,
                dst,
                valid,
            )
        self.host_pool.publish()

        if is_prefill:
            # A prefill can rewrite an existing request's host prefix.
            # Invalidate only this layer's resident extent.
            requests = forward_batch.req_pool_indices.to(torch.long)
            self.resident_pool.short_resident_end[layer_idx].index_fill_(
                0, requests, -1
            )

        return current, positions, valid

    def prepare_forward_kv_plan(self, forward_batch, current=None):
        """Use a fixed budgeted workspace on the current compute stream."""
        capacity = self.budget.prefill_workspace_tokens
        if not capacity:
            return None
        prepared = prepare_prefill_gather_rows(
            forward_batch.req_pool_indices_cpu,
            forward_batch.seq_lens_cpu,
        )
        _, total = prepared
        if not total or total > capacity:
            return None
        current_valid = None if current is None else current[2]
        return make_fused_prefill_plan(
            forward_batch,
            self,
            prepared=prepared,
            current_valid=current_valid,
        )

    def get_forward_kv(
        self,
        layer: RadixAttention | int,
        forward_batch: ForwardBatch,
        plan=None,
        current=None,
    ):
        """Gather full request KV from sparse storage as compact TND tensors.

        This helper is intended for the prefill/extend sparse path. It returns
        KV in the same request order as forward_batch.req_pool_indices:
        [req0 tokens][req1 tokens]... . The SFA caller should pair this with
        actual_seq_lengths_kv = cumsum(forward_batch.seq_lens).

        TODO: Add a prefill-resident device KV cache shaped like
        [max_prefill_parallel_reqs, max_prefill_len, nhead, head_dim]. Reuse a
        slot while the same req_id continues chunked prefill, and fully
        overwrite it when a new req_id takes that slot. This mirrors the decode
        device cache idea and avoids repeatedly copying the full prefix KV from
        host for every chunk.
        """
        layer_id = layer.layer_id if hasattr(layer, "layer_id") else layer
        layer_idx = layer_id - self.start_layer
        device = self.device
        if plan is not None:
            total = plan.total_tokens
            nope = self.prefill_nope[:total]
            rope = self.prefill_rope[:total]
            mapping = None
            if current is not None:
                current_payload, _, _ = current
                mapping = plan.current_mapping
            host_mask = plan.valid if mapping is None else mapping[3]
            self.host_pool.copy_rows(
                (self.host_k_buffer[layer_idx], self.host_v_buffer[layer_idx]),
                (nope, rope),
                plan.source,
                plan.destination,
                host_mask,
            )
            if mapping is not None:
                src, dst, valid, _ = mapping
                for component, output in zip(
                    current_payload,
                    (nope, rope),
                ):
                    unidex_copy_inplace(
                        component,
                        output,
                        src,
                        dst,
                        valid,
                        1,
                        1,
                        block_dim=48,
                    )
            return nope, rope

        req_ids = forward_batch.req_pool_indices.to(device=device, dtype=torch.long)
        seq_lens = forward_batch.seq_lens.to(device=device, dtype=torch.long)

        keep_cpu, total_tokens = prepare_prefill_gather_rows(
            forward_batch.req_pool_indices_cpu,
            forward_batch.seq_lens_cpu,
        )
        if total_tokens == 0:
            empty_nope = torch.empty(
                (0, self.head_num, self.kv_lora_rank),
                dtype=self.store_dtype,
                device=device,
            )
            empty_pe = torch.empty(
                (0, self.head_num, self.qk_rope_head_dim),
                dtype=self.store_dtype,
                device=device,
            )
            return empty_nope, empty_pe

        if keep_cpu.numel() != req_ids.numel():
            keep = keep_cpu.to(device=device, dtype=torch.long, non_blocking=True)
            req_ids = req_ids.index_select(0, keep)
            seq_lens = seq_lens.index_select(0, keep)

        seq_starts = torch.cumsum(seq_lens, dim=0) - seq_lens
        src_req_ids = torch.repeat_interleave(
            req_ids, seq_lens, output_size=total_tokens
        )
        src_seq_starts = torch.repeat_interleave(
            seq_starts, seq_lens, output_size=total_tokens
        )
        token_pos = (
            torch.arange(total_tokens, device=device, dtype=torch.long) - src_seq_starts
        )
        src_index = self.host_source_indices(src_req_ids, token_pos).contiguous()
        dst_index = torch.arange(total_tokens, device=device, dtype=torch.long)
        valid_mask = torch.ones(total_tokens, device=device, dtype=torch.bool)

        outputs = [
            torch.empty(
                (total_tokens, self.head_num, width),
                dtype=self.store_dtype,
                device=device,
            )
            for width in (self.kv_lora_rank, self.qk_rope_head_dim)
        ]
        self.host_pool.copy_rows(
            (self.host_k_buffer[layer_idx], self.host_v_buffer[layer_idx]),
            outputs,
            src_index,
            dst_index,
            valid_mask,
        )
        return tuple(outputs)
