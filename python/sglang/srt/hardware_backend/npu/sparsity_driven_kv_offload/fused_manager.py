"""Sparsity-driven KV offload manager for the Ascend backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace
from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
    create_fused_shm_tensor as create_shm_tensor,
)

from sglang.kernels.ops.attention.npu_copy_sfa import (
    fused_scatter_copy_sparse_flash_attention,
    load_copy_sfa_operator,
)
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    use_pd_split_kv,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.prefill_kv_plan import (
    make_fused_prefill_plan,
    prepare_prefill_gather_rows,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.memfabric_pool import (
        MemFabricKVPool,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class ShmSplitKVPool:
    """Per-rank SHM storage with the same component copy contract as MF."""

    rank = 0
    supports_pd_export = False

    def __init__(self, *, shape, widths, layers, dtype, device_id, name):
        self.buffers = [[], []]
        self.dev_ptrs = [[], []]
        self._device_pointers = {}
        for component, width in enumerate(widths):
            for layer in range(layers):
                tensor, _, pointer = create_shm_tensor(
                    shape=(*shape[:-1], width),
                    dtype=dtype,
                    device_id=device_id,
                    name=f"{name}_component_{component}_layer_{layer}",
                )
                self.buffers[component].append(tensor)
                self.dev_ptrs[component].append(pointer)
                self._device_pointers[tensor.untyped_storage().data_ptr()] = pointer

    def publish(self):
        pass

    def _device_pointer(self, tensor):
        if tensor.device.type != "cpu":
            return None
        base = tensor.untyped_storage().data_ptr()
        pointer = self._device_pointers.get(base)
        return None if pointer is None else pointer + tensor.data_ptr() - base

    def copy_rows(self, sources, destinations, source_rows, destination_rows, valid):
        if len(sources) != len(destinations) or not sources:
            raise ValueError(
                "Sparse KV copies require matching nonempty component lists"
            )
        for source, destination in zip(sources, destinations):
            src_ptr = self._device_pointer(source)
            dst_ptr = self._device_pointer(destination)
            kwargs = {}
            if src_ptr is not None:
                kwargs["src_ptr"] = src_ptr
            if dst_ptr is not None:
                kwargs["dst_ptr"] = dst_ptr
            unidex_copy_inplace(
                source,
                destination,
                source_rows,
                destination_rows,
                valid,
                source.ndim - 2,
                destination.ndim - 2,
                block_dim=48,
                **kwargs,
            )


class FusedSparseKVCacheManager:
    host_pool: Optional[Union[MemFabricKVPool, ShmSplitKVPool]] = None

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        sparse_topk: int,
        paged_kv_cache: Optional[MLATokenToKVPool] = None,
        budget_plan=None,
    ) -> None:
        enable_memory_saver = False
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.actual_request_rows = int(req_to_token_pool.req_to_token.shape[0])
        self.logical_max_context_len = int(req_to_token_pool.req_to_token.shape[1])
        self.req_to_token_pool = req_to_token_pool
        self.sparse_topk = int(sparse_topk)
        self.device = req_to_token_pool.device
        # EAGLE shares the target allocator's token IDs while owning a
        # separate draft KV pool. The allocator's pool is not the draft pool.
        if paged_kv_cache is None:
            paged_kv_cache = token_to_kv_pool_allocator.get_kvcache()
        self.paged_kv_cache = paged_kv_cache
        self.start_layer = paged_kv_cache.start_layer
        # MLA params
        self.head_num = 1
        self.kv_lora_rank = self.paged_kv_cache.kv_lora_rank
        self.qk_rope_head_dim = self.paged_kv_cache.qk_rope_head_dim
        # kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self.head_dim = (
            self.paged_kv_cache.kv_lora_rank + self.paged_kv_cache.qk_rope_head_dim
        )
        self.store_dtype = self.paged_kv_cache.store_dtype
        self.layer_num = self.paged_kv_cache.layer_num
        self.budget = budget_plan or getattr(
            paged_kv_cache, "sparse_kv_budget_plan", None
        )
        if self.budget is None:
            raise ValueError("Native KV pool has no sparse KV budget plan.")
        self.size = self.budget.request_rows
        self.batch_capacity = self.budget.batch_capacity
        self.max_context_len = self.budget.context_len
        self.max_query_width = self.budget.max_query_width
        self.host_token_capacity = self.budget.host_token_capacity
        self.device_cache_size = self.budget.cache_tokens
        logger.info(
            "Sparse KV per rank: K=%d C=%d, managed HBM=%.2f MiB, host KV=%.2f MiB, host_layout=paged_split",
            self.sparse_topk,
            self.device_cache_size,
            self.budget.managed_hbm_bytes / 1024**2,
            self.budget.host_kv_bytes / 1024**2,
        )
        logger.info(
            "Sparse KV prefill workspace: enabled=%s capacity_tokens=%d persistent_bytes=%d",
            bool(self.budget.prefill_workspace_tokens),
            self.budget.prefill_workspace_tokens,
            self.budget.prefill_workspace_bytes,
        )
        self.cache_capacity = self.device_cache_size
        self._copy_sfa_operator = load_copy_sfa_operator(split_hbm=True)
        split_host = use_pd_split_kv()

        # Allocate exactly the persistent shapes reserved by the plan.
        try:
            with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                for name, spec in self.budget.manager_buffers.items():
                    fill_value = (
                        -1
                        if name in ("device_slot_map", "resident_begin", "resident_end")
                        else None
                    )
                    value = (
                        [
                            spec.allocate(self.device, fill_value)
                            for _ in range(spec.count)
                        ]
                        if name
                        in ("device_kv_buffer", "device_rope_buffer", "device_slot_map")
                        else spec.allocate(self.device, fill_value)
                    )
                    setattr(self, name, value)
        except Exception as e:
            self._raise_buffer_allocation_error(name, e)

        # Host KV buffer
        # [bs, ctx_len, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.dev_ptr_list: list[int] = []

        # Radix prefixes use native physical token IDs. Keep host payloads in that
        # same address space; request slots own only bounded HBM residency maps.
        host_kv_shape = self.budget.host_kv_spec.shape
        device_id = torch.npu.current_device()

        if split_host:
            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.memfabric_pool import (
                MemFabricKVPool,
            )

            self.host_pool = MemFabricKVPool(
                tokens=self.host_token_capacity,
                layers=self.layer_num,
                widths=(self.kv_lora_rank, self.qk_rope_head_dim),
                dtype=self.store_dtype,
                device=self.device,
                group=get_parallel().attn_tp_group,
            )
            self.host_k_buffer, self.host_v_buffer = self.host_pool.buffers
            self.dev_ptr_list = [x.data_ptr() for x in self.host_k_buffer]
        else:
            self.host_pool = ShmSplitKVPool(
                shape=host_kv_shape,
                widths=(self.kv_lora_rank, self.qk_rope_head_dim),
                layers=self.layer_num,
                dtype=self.store_dtype,
                device_id=device_id,
                name=f"host_kv_pool_{id(paged_kv_cache)}_rank_{device_id}",
            )
            self.host_k_buffer, self.host_v_buffer = self.host_pool.buffers
            self.dev_ptr_list = self.host_pool.dev_ptrs[0]
        self.rope_dev_ptr_list = (
            [x.data_ptr() for x in self.host_v_buffer]
            if split_host
            else self.host_pool.dev_ptrs[1]
        )
        logger.info(
            "Sparse KV host buffers: layout=split, CKV=%s, KPE=%s, pool=%s",
            tuple(self.host_k_buffer[0].shape),
            tuple(self.host_v_buffer[0].shape),
            type(self.host_pool).__name__,
        )
        for component, width in zip(
            self.host_pool.buffers, (self.kv_lora_rank, self.qk_rope_head_dim)
        ):
            if len(component) != self.layer_num or any(
                t.ndim != 4
                or t.shape[0] * t.shape[1] != self.host_token_capacity
                or tuple(t.shape[2:]) != (self.head_num, width)
                or t.dtype != self.store_dtype
                or not t.is_contiguous()
                for t in component
            ):
                raise ValueError("Sparse split host KV does not match its budget.")
        if (
            sum(t.nbytes for component in self.host_pool.buffers for t in component)
            != self.budget.host_kv_bytes
        ):
            raise ValueError("Sparse split host KV bytes do not match its budget.")
        # Log allocation metadata once, before any graph capture or forward.
        logger.info(
            "Sparse split C allocation: capacity=%d layers=%d bytes=%d "
            "ckv_spec=%s rope_spec=%s",
            self.device_cache_size,
            self.layer_num,
            self.budget.device_kv_bytes,
            self.budget.manager_buffers["device_kv_buffer"],
            self.budget.manager_buffers["device_rope_buffer"],
        )
        for layer_idx in range(self.layer_num):
            ckv = self.device_kv_buffer[layer_idx]
            rope = self.device_rope_buffer[layer_idx]
            if ckv.untyped_storage().data_ptr() == rope.untyped_storage().data_ptr():
                raise ValueError("Split HBM components must have independent storage.")
            host_ckv = self.host_k_buffer[layer_idx]
            host_rope = self.host_v_buffer[layer_idx]
            logger.debug(
                "Sparse split layout layer=%d host_ckv_shape=%s host_ckv_stride=%s "
                "host_ckv_registered_ptr=%#x host_rope_shape=%s host_rope_stride=%s "
                "host_rope_registered_ptr=%#x hbm_ckv_shape=%s hbm_ckv_stride=%s "
                "hbm_ckv_storage_ptr=%#x hbm_rope_shape=%s hbm_rope_stride=%s "
                "hbm_rope_storage_ptr=%#x hbm_bytes=%d",
                layer_idx,
                tuple(host_ckv.shape),
                tuple(host_ckv.stride()),
                self.dev_ptr_list[layer_idx],
                tuple(host_rope.shape),
                tuple(host_rope.stride()),
                self.rope_dev_ptr_list[layer_idx],
                tuple(ckv.shape),
                tuple(ckv.stride()),
                ckv.untyped_storage().data_ptr(),
                tuple(rope.shape),
                tuple(rope.stride()),
                rope.untyped_storage().data_ptr(),
                ckv.nbytes + rope.nbytes,
            )
        logger.info(
            "Sparse split HBM verified: start_layer=%d layers=%d C=%d "
            "CKV=%d RoPE=%d independent_storage=True",
            self.start_layer,
            self.layer_num,
            self.device_cache_size,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
        )
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
            LIMDecodeAdapter,
        )

        self.lim_adapter = LIMDecodeAdapter(self)

        self._install_req_alloc_hook(req_to_token_pool)

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

    def reset_requests(self, req_ids: List[int]) -> None:
        if not req_ids:
            return

        req_ids_tensor = torch.tensor(
            req_ids, dtype=torch.long, device=self.device
        ).contiguous()
        for layer_idx in range(self.layer_num):
            self.resident_begin[layer_idx].index_fill_(0, req_ids_tensor, -1)
            self.resident_end[layer_idx].index_fill_(0, req_ids_tensor, -1)
            self.device_slot_map[layer_idx].index_fill_(0, req_ids_tensor, -1)
        self.lim_adapter.reset_requests(req_ids_tensor)

    def copy_sfa_attention(
        self,
        layer,
        query_nope,
        query_rope,
        query_heads,
        scale_value,
        plan,
        current_inputs,
    ):
        batch = plan.request_safe_slots.numel()
        rows = batch * plan.query_width
        self.lim_adapter.copy_sfa_metadata(self, plan)
        padded_heads = self.fused_attn_out.shape[1]
        if query_nope.shape[1] < padded_heads:
            padding = padded_heads - query_nope.shape[1]
            query_nope = torch.nn.functional.pad(query_nope, (0, 0, 0, padding))
            query_rope = torch.nn.functional.pad(query_rope, (0, 0, 0, padding))
        output = self.fused_attn_out[:rows]
        layer_idx = layer.layer_id - self.start_layer
        cache = self.device_kv_buffer[layer_idx].view(
            -1, 128, self.head_num, self.kv_lora_rank
        )
        rope_cache = self.device_rope_buffer[layer_idx].view(
            -1, 128, self.head_num, self.qk_rope_head_dim
        )
        fused_scatter_copy_sparse_flash_attention(
            query_nope.contiguous(),
            query_rope.contiguous(),
            cache,
            self.planner_query_ends,
            self.planner_actual_seq_lengths_kv[:batch],
            self.planner_num_cache_tokens[:batch],
            plan.source_ids,
            plan.device_slots,
            plan.topk_misses,
            plan.miss_sources,
            plan.miss_slots,
            plan.miss_counts,
            self.planner_hbm_block_table[:batch],
            self.planner_dram_block_table[:batch],
            self.dev_ptr_list[layer_idx],
            output,
            scale_value,
            operator=self._copy_sfa_operator,
            split_hbm_rope=rope_cache,
            full_visible=True,
            resident_refresh_tail=plan.query_width > 1,
            resident_inputs={
                "current": current_inputs[0],
                "positions": current_inputs[1],
                "valid": current_inputs[2],
                "request_slots": plan.request_safe_slots,
                "req_to_token": self.req_to_token_pool.req_to_token,
                "resident_begin": self.resident_begin[layer_idx],
                "resident_end": self.resident_end[layer_idx],
                "host_token_capacity": self.host_token_capacity,
                "dram_rope_dev_ptr": self.rope_dev_ptr_list[layer_idx],
            },
        )
        valid = (
            plan.valid_rows[:, None].expand(batch, plan.query_width).reshape(rows, 1)
        )
        return output[:, :query_heads].reshape(rows, -1).masked_fill(~valid, 0)

    def _install_req_alloc_hook(self, req_to_token_pool: ReqToTokenPool) -> None:
        # Preserve the target manager's hook when adding the draft manager.
        original_alloc = req_to_token_pool.alloc

        def alloc_with_sparse_reset(reqs: list[Req]) -> Optional[List[int]]:
            newly_allocated = [not req.kv.holds_kv for req in reqs]
            req_pool_indices = original_alloc(reqs)
            if req_pool_indices is not None:
                self.reset_requests(
                    [
                        req_pool_indices[i]
                        for i, is_new in enumerate(newly_allocated)
                        if is_new
                    ]
                )
            return req_pool_indices

        setattr(req_to_token_pool, "alloc", alloc_with_sparse_reset)

    def offload_v2(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """Publish KV to paged host storage and return decode rows for CopySFA."""
        layer_idx = layer.layer_id - self.start_layer
        device = k.device

        # k:         [num_new_tokens, nhead, kv_lora_rank]
        # k_rope:    [num_new_tokens, nhead, qk_rope_head_dim]
        # kv_device: [num_new_tokens, nhead, kv_lora_rank + qk_rope_head_dim]
        src_tensor = torch.cat([k, k_rope], dim=-1).contiguous()

        num_src_kvs = int(src_tensor.shape[0])
        fused_mtp_rows = not forward_batch.forward_mode.is_extend_without_speculative()
        if not fused_mtp_rows:
            src_index = torch.arange(num_src_kvs, device=device, dtype=torch.long)

        if fused_mtp_rows:
            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.row_metadata import (
                mtp_publication_rows,
            )

            src_index, dst_index, token_positions, valid_mask = mtp_publication_rows(
                forward_batch,
                num_src_kvs,
                self.size,
                self.logical_max_context_len,
                self.host_token_capacity,
            )
        else:
            if (
                forward_batch.extend_seq_lens is None
                or forward_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "Sparse v2 prefill offload requires extend_seq_lens and "
                    "extend_prefix_lens in ForwardBatch."
                )

            req_ids = forward_batch.req_pool_indices.to(torch.long)
            extend_seq_lens = forward_batch.extend_seq_lens.to(torch.long)
            extend_prefix_lens = forward_batch.extend_prefix_lens.to(torch.long)

            batch_size = int(req_ids.shape[0])
            if batch_size <= 0:
                return

            if int(extend_seq_lens.shape[0]) != batch_size:
                raise RuntimeError(
                    "Sparse v2 prefill offload expects extend_seq_lens to be "
                    f"padded to batch size {batch_size}, got "
                    f"{int(extend_seq_lens.shape[0])}."
                )

            prefix_len_size = int(extend_prefix_lens.shape[0])
            if prefix_len_size < batch_size:
                extend_prefix_lens = torch.cat(
                    [
                        extend_prefix_lens,
                        torch.zeros(
                            batch_size - prefix_len_size,
                            device=device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )
            elif prefix_len_size > batch_size:
                raise RuntimeError(
                    "Sparse v2 prefill offload expects extend_prefix_lens length "
                    f"<= batch size {batch_size}, got {prefix_len_size}."
                )

            if forward_batch.extend_seq_lens_cpu is not None:
                extend_seq_lens_sum = int(
                    sum(forward_batch.extend_seq_lens_cpu[:batch_size])
                )
            else:
                extend_seq_lens_sum = int(extend_seq_lens.sum().item())
            if extend_seq_lens_sum == num_src_kvs:
                # Chunk prefill emits compact rows as [req0 tokens][req1 tokens]...
                # instead of graph-captured padded [B, tokens_per_req] rows.
                seq_starts = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens
                flat_req_ids = torch.repeat_interleave(
                    req_ids, extend_seq_lens, output_size=num_src_kvs
                )
                flat_seq_starts = torch.repeat_interleave(
                    seq_starts, extend_seq_lens, output_size=num_src_kvs
                )
                flat_prefix_lens = torch.repeat_interleave(
                    extend_prefix_lens, extend_seq_lens, output_size=num_src_kvs
                )
                token_pos = (
                    flat_prefix_lens
                    + torch.arange(num_src_kvs, device=device, dtype=torch.long)
                    - flat_seq_starts
                )
                token_positions = token_pos
                valid_mask = (
                    (flat_req_ids > 0)
                    & (token_pos >= 0)
                    & (token_pos < self.max_context_len)
                )
            else:
                if num_src_kvs % batch_size != 0:
                    raise RuntimeError(
                        "Sparse v2 prefill offload expects either compact ragged "
                        "layout with rows=sum(extend_seq_lens) or graph-style "
                        f"row-major layout [B, tokens_per_req], got rows={num_src_kvs}, "
                        f"batch={batch_size}, extend_seq_lens_sum={extend_seq_lens_sum}."
                    )

                # Graph-friendly static layout:
                # compact rows are interpreted as [batch_size, tokens_per_req].
                # Invalid padded columns are masked by local_offsets < extend_seq_lens.
                tokens_per_req = num_src_kvs // batch_size

                local_offsets = (
                    torch.arange(tokens_per_req, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_size, tokens_per_req)
                )
                req_ids_2d = req_ids.unsqueeze(1).expand(batch_size, tokens_per_req)
                token_pos_2d = extend_prefix_lens.unsqueeze(1) + local_offsets

                token_positions = token_pos_2d.reshape(-1).contiguous()
                valid_mask = (
                    (local_offsets < extend_seq_lens.unsqueeze(1))
                    & (req_ids_2d > 0)
                    & (token_pos_2d >= 0)
                    & (token_pos_2d < self.max_context_len)
                ).reshape(-1)

            if (
                forward_batch.out_cache_loc is not None
                and int(forward_batch.out_cache_loc.numel()) == num_src_kvs
            ):
                valid_mask = valid_mask & (
                    forward_batch.out_cache_loc.to(torch.long) >= 0
                )
            valid_mask = valid_mask.contiguous()

        if not fused_mtp_rows:
            dst_index = forward_batch.out_cache_loc.to(torch.long).reshape(-1)
            valid_mask = (
                valid_mask & (dst_index >= 0) & (dst_index < self.host_token_capacity)
            )
        assert src_index.shape == dst_index.shape == valid_mask.shape

        if self.host_pool.rank == 0:
            self.host_pool.copy_rows(
                (k.contiguous(), k_rope.contiguous()),
                (self.host_k_buffer[layer_idx], self.host_v_buffer[layer_idx]),
                src_index,
                dst_index,
                valid_mask,
            )
        self.host_pool.publish()

        if forward_batch.forward_mode.is_extend_without_speculative():
            # A prefill can rewrite an existing request's host prefix.
            # Invalidate only this layer's resident extent.
            requests = forward_batch.req_pool_indices.to(torch.long)
            self.resident_begin[layer_idx].index_fill_(0, requests, -1)
            self.resident_end[layer_idx].index_fill_(0, requests, -1)

        return src_tensor, token_positions, valid_mask

    def prepare_forward_kv_plan(self, forward_batch, current=None):
        """Use a fixed budgeted workspace on the current compute stream."""
        capacity = getattr(self.budget, "prefill_workspace_tokens", 0)
        if not capacity:
            return None
        prepared = prepare_prefill_gather_rows(
            forward_batch.req_pool_indices_cpu,
            forward_batch.seq_lens_cpu,
            self.size,
            self.max_context_len,
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
        layer: Union[RadixAttention, int],
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
        layer_id = layer.layer_id if hasattr(layer, "layer_id") else int(layer)
        layer_idx = layer_id - self.start_layer
        if layer_idx < 0 or layer_idx >= self.layer_num:
            raise RuntimeError(
                f"Invalid sparse KV layer id {layer_id}; start_layer="
                f"{self.start_layer}, layer_num={self.layer_num}."
            )

        if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
            raise RuntimeError(
                "get_forward_kv requires req_pool_indices and seq_lens in ForwardBatch."
            )

        device = (
            forward_batch.req_pool_indices.device
            if forward_batch.req_pool_indices.device.type == "npu"
            else self.device
        )
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
                    current_payload.split(
                        (self.kv_lora_rank, self.qk_rope_head_dim), dim=-1
                    ),
                    (nope, rope),
                ):
                    unidex_copy_inplace(
                        component.contiguous(),
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

        if int(req_ids.numel()) != int(seq_lens.numel()):
            raise RuntimeError(
                "get_forward_kv expects req_pool_indices and seq_lens to have the "
                f"same length, got {int(req_ids.numel())} and "
                f"{int(seq_lens.numel())}."
            )

        keep_cpu, total_tokens = prepare_prefill_gather_rows(
            forward_batch.req_pool_indices_cpu,
            forward_batch.seq_lens_cpu,
            self.size,
            self.max_context_len,
        )
        if forward_batch.req_pool_indices_cpu.numel() != req_ids.numel():
            raise RuntimeError(
                "Prefill CPU and device metadata must have matching sizes"
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
