"""Sparsity-driven KV offload manager for the Ascend backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    slot_map_lookup,
    unidex_copy_inplace,
)

from sglang.kernels.ops.attention.npu_copy_sfa import (
    fused_scatter_copy_sparse_flash_attention,
    load_copy_sfa_operator,
)
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.block_tables import (
    fill_request_block_table,
    prepare_prefill_gather_rows,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.budget import (
    sparse_batch_capacity,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    get_sparse_kv_budget,
    get_sparse_kv_indexer_impl,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.copy_sfa_buffers import (
    copy_sfa_host_row_stride,
    view_copy_sfa_output,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.lru import (
    plan_sparse_lru,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    is_sparse_mtp_batch,
    prepare_mtp_rows,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import get_eager_max_batch_size
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    import torch.npu

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def _record_stream_event(stream, event) -> None:
    if hasattr(stream, "record_event"):
        stream.record_event(event)
    else:
        event.record(stream)


def _wait_stream_event(stream, event) -> None:
    if hasattr(stream, "wait_event"):
        stream.wait_event(event)
    else:
        event.wait(stream)


def normalize_batch_topk_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    """Normalize DSA top-k indices to [batch, topk] for compact KV copies."""
    if topk_indices.dim() == 2:
        return topk_indices
    if topk_indices.dim() == 3 and topk_indices.shape[1] == 1:
        return topk_indices[:, 0, :]
    if (
        topk_indices.dim() == 4
        and topk_indices.shape[1] == 1
        and topk_indices.shape[2] == 1
    ):
        return topk_indices[:, 0, 0, :]
    raise RuntimeError(
        "Sparsity-driven KV offload expects DSA top-k indices with shape "
        f"[batch, topk], [batch, 1, topk], or [batch, 1, 1, topk], got "
        f"{tuple(topk_indices.shape)}."
    )


class SparseKVCacheManager:
    copy_stream = None
    miss_shm_cpu_tensor: list = []
    miss_shm_dev_ptr: Optional[int] = None
    miss_shm_shape: list = []
    miss_shm_dtype: list = []

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        sparse_context_len: int,
        copy_sfa_padded_query_heads: Optional[int] = None,
        graph_batch_capacity: int = 0,
        paged_kv_cache: Optional[MLATokenToKVPool] = None,
    ) -> None:
        enable_memory_saver = False
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        # Include the padding row because real request IDs can equal the
        # configured capacity when row 0 is reserved for graph padding.
        self.size = int(req_to_token_pool.req_to_token.shape[0])
        self.batch_capacity = sparse_batch_capacity(
            self.size,
            max(graph_batch_capacity, get_eager_max_batch_size(self.size - 1)),
        )
        self.last_decode_path = "unselected"
        self.logical_max_context_len = int(req_to_token_pool.max_context_len)
        self.copy_sfa_enabled = envs.SGLANG_NPU_ENABLE_COPY_SFA.get()
        self.max_context_len = copy_sfa_host_row_stride(
            self.logical_max_context_len, self.copy_sfa_enabled
        )
        self.sparse_context_len = int(sparse_context_len)
        if self.sparse_context_len <= 0:
            raise ValueError(
                "SparseKVCacheManager requires a positive sparse_context_len, "
                f"got {self.sparse_context_len}."
            )
        self.device = req_to_token_pool.device
        # EAGLE shares the target allocator's token IDs while owning a
        # separate draft KV pool. The allocator's pool is not the draft pool.
        if paged_kv_cache is None:
            paged_kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(paged_kv_cache, MLATokenToKVPool):
            raise TypeError(
                "SparseKVCacheManager requires an MLATokenToKVPool, "
                f"got {type(paged_kv_cache).__name__}"
            )
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
        self.budget = get_sparse_kv_budget(
            request_rows=self.size,
            context_len=self.max_context_len,
            num_layers=self.layer_num,
            top_k=self.sparse_context_len,
            kv_bytes_per_token=self.head_dim * self.store_dtype.itemsize,
            copy_sfa_query_heads=copy_sfa_padded_query_heads or 128,
            graph_batch_capacity=self.batch_capacity,
        )
        self.device_cache_size = self.budget.cache_tokens
        self.indexer_impl = get_sparse_kv_indexer_impl()
        self.use_lru = (
            self.device_cache_size > self.sparse_context_len
            and self.indexer_impl != "lim"
        )
        self._slot_map_width = (
            (self.max_context_len + 127) // 128 * 128
            if self.indexer_impl == "lim"
            else (self.max_context_len // 8 + 1) * 8
        )
        logger.info(
            "Sparse KV per rank: K=%d C=%d, managed HBM=%.2f MiB, host KV=%.2f MiB, LRU=%s",
            self.sparse_context_len,
            self.device_cache_size,
            self.budget.managed_hbm_bytes / 1024**2,
            self.budget.host_kv_bytes / 1024**2,
            self.use_lru,
        )
        self.cache_capacity = self.device_cache_size
        if self.copy_sfa_enabled and (
            not 2048 <= self.cache_capacity <= 32640
            or self.cache_capacity % 128
            or copy_sfa_padded_query_heads is None
        ):
            raise ValueError(
                "Copy-SFA requires C in [2048,32640], aligned to 128, and a padded query head count."
            )
        if self.copy_sfa_enabled:
            if (self.kv_lora_rank, self.qk_rope_head_dim) != (512, 64):
                raise ValueError(
                    "Copy-SFA supports only MLA KV dimensions CKV=512 and KPE=64; "
                    f"got CKV={self.kv_lora_rank}, KPE={self.qk_rope_head_dim}."
                )
            if self.store_dtype not in (torch.bfloat16, torch.float16):
                raise ValueError(
                    "Copy-SFA supports only BF16 or FP16 KV storage; "
                    f"got {self.store_dtype}."
                )
            from sgl_kernel_npu.sparsity_driven_kv_offload import plan_sparse_kv_cache

            self._plan_sparse_kv_cache = plan_sparse_kv_cache
            self._copy_sfa_operator = load_copy_sfa_operator()
        else:
            self._plan_sparse_kv_cache = None
            self._copy_sfa_operator = None
        # Main stream handles hits and attention. One side stream handles host
        # misses, cache refill and publication; refill overlaps attention.
        self._materialize_h2d_miss_stream = torch.npu.Stream()
        self.copy_ready = torch.npu.Event()
        self.hit_done = torch.npu.Event()
        self.miss_done = torch.npu.Event()
        self.refill_done = torch.npu.Event()

        # device KV buffer
        try:
            with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                # [bs, ctx_len, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.device_kv_buffer: list[torch.Tensor] = [
                    torch.empty(
                        (
                            self.size,
                            self.device_cache_size,
                            self.head_num,
                            self.head_dim,
                        ),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
        except Exception as e:
            self._raise_buffer_allocation_error("device_kv_buffer", e)

        try:
            with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                # Reserve the last row for padded requests and ensure token index
                # `max_context_len` is a valid sentinel column for masked writes.
                # The row width is also aligned to eight int32 values (32 bytes).
                self.device_slot_map: list[torch.Tensor] = [
                    torch.full(
                        (
                            self.size + 1,
                            self._slot_map_width,
                        ),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                # One compact workspace is reused across layers and steps. The
                # attention path joins refill_done before the next reuse.
                self.selected_kv_buffer = torch.empty(
                    (
                        self.batch_capacity,
                        self.sparse_context_len,
                        self.head_num,
                        self.head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                if self.use_lru:
                    self._lru_init = torch.arange(
                        self.device_cache_size, dtype=torch.int64, device=self.device
                    )
                    self.cache_tokens = [
                        torch.full(
                            (self.size + 1, self.device_cache_size),
                            -1,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.lru_order = [
                        self._lru_init.expand(self.size + 1, -1).clone()
                        for _ in range(self.layer_num)
                    ]
                if self.copy_sfa_enabled:
                    self.device_slot_owner: list[torch.Tensor] = [
                        torch.full(
                            (self.size + 1, self.cache_capacity),
                            -1,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.planner_topk_src_ids = torch.full(
                        (self.batch_capacity, 1, self.sparse_context_len),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.planner_topk_dst_slots = torch.full_like(
                        self.planner_topk_src_ids, -1
                    )
                    self.planner_topk_miss_counts = torch.zeros(
                        (self.batch_capacity,), dtype=torch.int32, device=self.device
                    )
                    self.planner_miss_src_ids = torch.full(
                        (self.batch_capacity, self.cache_capacity),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.planner_miss_dst_slots = torch.full_like(
                        self.planner_miss_src_ids, -1
                    )
                    self.planner_miss_counts = torch.zeros(
                        (self.batch_capacity,), dtype=torch.int32, device=self.device
                    )
                    self.planner_actual_seq_lengths_query = torch.arange(
                        1,
                        self.batch_capacity + 1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.planner_actual_seq_lengths_kv = torch.empty_like(
                        self.planner_actual_seq_lengths_query
                    )
                    self.planner_num_cache_tokens = torch.empty_like(
                        self.planner_actual_seq_lengths_query
                    )
                    self.hbm_page_offsets = torch.arange(
                        self.cache_capacity // 128,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.dram_page_offsets = torch.arange(
                        self.max_context_len // 128,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.planner_hbm_block_table = torch.empty(
                        (self.batch_capacity, self.cache_capacity // 128),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.planner_dram_block_table = torch.empty(
                        (self.batch_capacity, self.max_context_len // 128),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.planner_attn_out = torch.empty(
                        (
                            self.batch_capacity,
                            copy_sfa_padded_query_heads,
                            self.kv_lora_rank,
                        ),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
        except Exception as e:
            self._raise_buffer_allocation_error("device_slot_map", e)

        # Host KV buffer
        # [bs, ctx_len, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.host_kv_buffer: list[torch.Tensor] = []
        self.host_ptr_list: list[int] = []
        self.dev_ptr_list: list[int] = []

        host_kv_shape = (
            self.size,
            self.max_context_len,
            self.head_num,
            self.head_dim,
        )
        logger.info("Sparse KV host buffer shape: %s", host_kv_shape)
        device_id = torch.npu.current_device()

        try:
            for layer_idx in range(self.layer_num):
                shm_cpu_tensor, host_ptr, dev_ptr = create_shm_tensor(
                    shape=host_kv_shape,
                    dtype=self.store_dtype,
                    device_id=device_id,
                    name=f"host_kv_pool_{id(paged_kv_cache)}_layer_{layer_idx}_rank_{device_id}",
                )
                self.host_kv_buffer.append(shm_cpu_tensor)
                self.host_ptr_list.append(host_ptr)
                self.dev_ptr_list.append(dev_ptr)
        except Exception as e:
            self._raise_buffer_allocation_error("host_kv_buffer", e)
        self.host_kv_ctx_len = torch.zeros(
            (self.size, self.max_context_len), dtype=torch.int32, device="cpu"
        )
        self.topk_indices_cpu = None
        self.token_on_device_cpu = None
        self.device_token_pos_cpu = None
        self.current_req_indices_cpu = None

        self._device_cache_slot_ids = torch.arange(
            self.sparse_context_len, dtype=torch.long, device=self.device
        )
        self.lim_adapter = None
        if self.indexer_impl != "legacy":
            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.lim import (
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
            f"sparse_context_len={self.sparse_context_len}. "
            "The sparse KV request capacity may be too large; set a smaller "
            "--max-running-requests for sparse KV offload."
        ) from exc

    def init_req(self, req: Req) -> None:
        if req.inflight_middle_chunks > 0:
            return
        rid = req.kv.req_pool_idx
        if rid is None:
            raise RuntimeError(
                "Cannot initialize sparse KV state before allocating a request pool slot"
            )
        current_len = len(req.origin_input_ids)
        self.host_kv_ctx_len[rid] = current_len
        self.reset_requests([rid])

    def reset_requests(self, req_ids: List[int]) -> None:
        if not req_ids:
            return

        req_ids_tensor = torch.tensor(
            req_ids, dtype=torch.long, device=self.device
        ).contiguous()
        for layer_idx in range(self.layer_num):
            self.device_slot_map[layer_idx].index_fill_(0, req_ids_tensor, -1)
            if self.copy_sfa_enabled:
                self.device_slot_owner[layer_idx].index_fill_(0, req_ids_tensor, -1)
            if self.use_lru:
                self.cache_tokens[layer_idx].index_fill_(0, req_ids_tensor, -1)
                self.lru_order[layer_idx][req_ids_tensor] = self._lru_init
        if self.lim_adapter is not None:
            self.lim_adapter.reset_requests(req_ids_tensor)

    def invalidate_mtp_cache(self, layer, request_slots):
        """Discard residency before publishing a potentially rewritten draft."""
        layer_index = layer.layer_id - self.start_layer
        slots = request_slots.long().clamp(0, self.size - 1)
        self.device_slot_map[layer_index].index_fill_(0, slots, -1)
        if self.copy_sfa_enabled:
            self.device_slot_owner[layer_index].index_fill_(0, slots, -1)
        if self.lim_adapter is not None:
            self.lim_adapter.last_prefix[layer_index].index_fill_(0, slots, -1)

    def plan_decode_cache(
        self,
        layer: RadixAttention,
        req_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        topk_src_ids: torch.Tensor,
    ):
        """Update stable source/slot maps and fill graph-stable Copy-SFA metadata."""
        layer_idx = layer.layer_id - self.start_layer
        batch_size = req_ids.shape[0]
        req_ids_i32 = req_ids.to(dtype=torch.int32).contiguous()
        topk_src_ids_i32 = (
            normalize_batch_topk_indices(topk_src_ids)
            .to(dtype=torch.int32)
            .contiguous()
        )
        valid_req = (
            (req_ids_i32 >= 0) & (req_ids_i32 < self.size) & (seq_lens[:batch_size] > 0)
        )
        # Graph padding uses request ID zero, which is also a valid live row.
        # Convert padded rows to the manager's reserved request sentinel so the
        # native planner returns before reading either cache map.
        planner_req_ids = torch.where(
            valid_req, req_ids_i32, torch.full_like(req_ids_i32, self.size)
        )
        self._plan_sparse_kv_cache(
            self.device_slot_map[layer_idx],
            self.device_slot_owner[layer_idx],
            planner_req_ids,
            topk_src_ids_i32,
            self.planner_topk_src_ids[:batch_size],
            self.planner_topk_dst_slots[:batch_size],
            self.planner_topk_miss_counts[:batch_size],
            self.planner_miss_src_ids[:batch_size],
            self.planner_miss_dst_slots[:batch_size],
            self.planner_miss_counts[:batch_size],
            max_context_len=self.logical_max_context_len,
            cache_capacity=self.cache_capacity,
        )
        # Keep dense Q row positions stable; padded K/cache rows have length 0.
        valid_req_i32 = valid_req.to(torch.int32)
        kv_lengths = valid_req_i32 * self.cache_capacity
        self.planner_actual_seq_lengths_kv[:batch_size].copy_(kv_lengths)
        self.planner_num_cache_tokens[:batch_size].copy_(kv_lengths)
        hbm_block_table = self.planner_hbm_block_table[:batch_size]
        fill_request_block_table(
            req_ids_i32,
            valid_req,
            self.hbm_page_offsets,
            self.cache_capacity // 128,
            hbm_block_table,
        )
        dram_block_table = self.planner_dram_block_table[:batch_size]
        fill_request_block_table(
            req_ids_i32,
            valid_req,
            self.dram_page_offsets,
            self.max_context_len // 128,
            dram_block_table,
        )
        return (
            self.planner_topk_src_ids[:batch_size],
            self.planner_topk_dst_slots[:batch_size],
            self.planner_topk_miss_counts[:batch_size],
            self.planner_miss_src_ids[:batch_size],
            self.planner_miss_dst_slots[:batch_size],
            self.planner_miss_counts[:batch_size],
            self.planner_actual_seq_lengths_query[:batch_size],
            self.planner_actual_seq_lengths_kv[:batch_size],
            self.planner_num_cache_tokens[:batch_size],
            self.planner_hbm_block_table[:batch_size],
            self.planner_dram_block_table[:batch_size],
        )

    def copy_sfa_decode(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        query_nope: torch.Tensor,
        query_rope: torch.Tensor,
        query_heads: int,
        scale_value: float,
        topk_src_ids: torch.Tensor,
        lim_plan=None,
    ) -> torch.Tensor:
        """Plan cache residency and run the fused scatter/copy/SFA operator."""
        batch_size = forward_batch.batch_size
        if lim_plan is not None:
            if self.lim_adapter is None or not self.lim_adapter.is_native:
                raise RuntimeError(
                    "Copy-SFA received a native plan without native LIM."
                )
            # LIM already owns the residency update. Do not invoke a second
            # planner against its encoded slot map.
            metadata = self.lim_adapter.copy_sfa_metadata(self, lim_plan)
        else:
            metadata = self.plan_decode_cache(
                layer,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                topk_src_ids,
            )
        (
            topk_src,
            topk_dst,
            topk_miss_counts,
            miss_src,
            miss_dst,
            miss_counts,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
            num_cache_tokens,
            hbm_block_table,
            dram_block_table,
        ) = metadata
        query_nope = query_nope.squeeze(1).contiguous()
        query_rope = query_rope.squeeze(1).contiguous()
        output_buffer = self.planner_attn_out
        if (
            query_nope.shape[1] > output_buffer.shape[1]
            or query_nope.shape[2] != output_buffer.shape[2]
            or query_nope.dtype != output_buffer.dtype
            or batch_size > output_buffer.shape[0]
        ):
            raise RuntimeError(
                "Copy-SFA query exceeds its preallocated output contract: "
                f"query={tuple(query_nope.shape)} {query_nope.dtype}, "
                f"output={tuple(output_buffer.shape)} {output_buffer.dtype}."
            )
        attention_out = view_copy_sfa_output(
            output_buffer,
            batch_size,
            query_nope.shape[1],
            query_nope.shape[2],
        )

        layer_idx = layer.layer_id - self.start_layer
        combined_hbm_cache = self.device_kv_buffer[layer_idx].view(
            self.size * (self.cache_capacity // 128), 128, self.head_num, self.head_dim
        )
        fused_scatter_copy_sparse_flash_attention(
            query_nope,
            query_rope,
            combined_hbm_cache,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
            num_cache_tokens,
            topk_src,
            topk_dst,
            topk_miss_counts,
            miss_src,
            miss_dst,
            miss_counts,
            hbm_block_table,
            dram_block_table,
            self.dev_ptr_list[layer_idx],
            attention_out,
            scale_value,
            operator=self._copy_sfa_operator,
        )
        return attention_out[:, :query_heads].reshape(batch_size, -1)

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

    def offload(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        stream: torch.npu.Stream,
    ):
        layer_idx = layer.layer_id - self.start_layer
        device = k.device

        # k:        [total_token_slots, nhead, dim]
        # k_rope:   [total_token_slots, nhead, dim]
        # kv_device: [total_token_slots, nhead, 2*dim]
        kv_device = torch.cat([k, k_rope], dim=-1)

        # Source row indices into kv_device.
        # In graph mode this tensor is expected to have a static shape.
        src_token_indices = forward_batch.out_cache_loc.to(torch.long).contiguous()
        static_token_slots = int(src_token_indices.numel())

        if forward_batch.forward_mode.is_decode():
            # decode graph mode:
            # one token slot per request, padded requests are masked out
            req_ids = forward_batch.req_pool_indices.to(torch.long)
            token_pos = (forward_batch.seq_lens - 1).to(torch.long)

            dst_token_indices = (
                req_ids * self.max_context_len + token_pos
            ).contiguous()

            # Existing graph decode convention:
            # padded decode requests carry seq_len == 1.
            valid_mask = (
                (forward_batch.seq_lens != 1)
                & (src_token_indices >= 0)
                & (req_ids >= 0)
            ).contiguous()

        else:
            # prefill graph mode:
            # assume out_cache_loc is laid out as [B, TOKENS_PER_REQ] flattened row-major
            if (
                forward_batch.extend_seq_lens is None
                or forward_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "Sparse graph prefill offload requires extend_seq_lens and "
                    "extend_prefix_lens in ForwardBatch."
                )

            batch_size = int(forward_batch.req_pool_indices.shape[0])
            if batch_size <= 0:
                return

            if static_token_slots % batch_size != 0:
                raise RuntimeError(
                    f"out_cache_loc length {static_token_slots} is not divisible by "
                    f"batch size {batch_size}. Cannot infer graph static token layout."
                )

            tokens_per_req = static_token_slots // batch_size

            req_ids = forward_batch.req_pool_indices.to(torch.long)
            extend_seq_lens = forward_batch.extend_seq_lens.to(torch.long)
            extend_prefix_lens = forward_batch.extend_prefix_lens.to(torch.long)

            local_offsets = (
                torch.arange(
                    tokens_per_req,
                    device=device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, tokens_per_req)
            )

            req_ids_2d = req_ids.unsqueeze(1).expand(batch_size, tokens_per_req)
            dst_pos_2d = extend_prefix_lens.unsqueeze(1) + local_offsets

            dst_token_indices = (
                (req_ids_2d * self.max_context_len + dst_pos_2d)
                .reshape(-1)
                .contiguous()
            )

            valid_mask = (
                (local_offsets < extend_seq_lens.unsqueeze(1)).reshape(-1)
                & (src_token_indices >= 0)
                & (req_ids_2d.reshape(-1) >= 0)
            ).contiguous()

        # Layout check:
        # kv_device rows:                 [token_slot]
        # host_kv_buffer[layer] rows:     [req_id, seq_pos]
        # block dims must match on [nhead, 2*dim]
        assert kv_device.shape[1:] == self.host_kv_buffer[layer_idx].shape[2:]
        # torch.npu.synchronize()
        actual_stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(actual_stream):
            unidex_copy_inplace(
                kv_device,
                self.host_kv_buffer[layer_idx],
                src_token_indices,
                dst_token_indices,
                valid_mask,
                1,  # kv_device: [token_slot, nhead, 2*dim]
                2,  # host_kv_buffer: [num_req, max_context_len, nhead, 2*dim]
                block_dim=48,
                dst_ptr=self.dev_ptr_list[layer_idx],
            )
        # torch.npu.synchronize()

    def offload_v2(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        stream: torch.npu.Stream,
    ):
        """Offload compact per-forward KV rows into the sparse host KV buffer.

        v1 expects k/k_rope to be full native KV-cache views and therefore uses
        forward_batch.out_cache_loc as source rows. v2 expects k/k_rope to be
        compact rows produced by the current forward pass, so source rows are
        simply [0, num_new_tokens). The native cache slot is kept only as
        validity metadata.

        Keep src_tensor/dst_tensor/src_index/dst_index/valid_mask explicit so
        the final copy can be swapped to a custom kernel without changing the
        graph-friendly index construction.
        """
        layer_idx = layer.layer_id - self.start_layer
        device = k.device

        # k:         [num_new_tokens, nhead, kv_lora_rank]
        # k_rope:    [num_new_tokens, nhead, qk_rope_head_dim]
        # kv_device: [num_new_tokens, nhead, kv_lora_rank + qk_rope_head_dim]
        src_tensor = torch.cat([k, k_rope], dim=-1).contiguous()
        dst_tensor = self.host_kv_buffer[layer_idx]
        src_index = torch.arange(src_tensor.shape[0], device=device, dtype=torch.long)
        num_src_rows = int(src_tensor.shape[0])

        if is_sparse_mtp_batch(forward_batch):
            req_ids, token_pos, valid_mask = prepare_mtp_rows(
                forward_batch, num_src_rows, self.size, self.logical_max_context_len
            )
            dst_index = (req_ids * self.max_context_len + token_pos).reshape(-1)
            valid_mask = valid_mask.reshape(-1).contiguous()
            # TP token padding is a suffix, independent of the request window.
            padding = num_src_rows - dst_index.numel()
            dst_index = torch.nn.functional.pad(dst_index, (0, padding), value=0)
            valid_mask = torch.nn.functional.pad(valid_mask, (0, padding), value=False)
        elif forward_batch.forward_mode.is_decode():
            req_ids = forward_batch.req_pool_indices.to(torch.long)
            token_pos = (forward_batch.seq_lens - 1).to(torch.long)
            cache_loc = forward_batch.out_cache_loc.to(torch.long)

            if int(req_ids.shape[0]) != num_src_rows:
                raise RuntimeError(
                    "Sparse v2 decode offload expects compact KV rows to match "
                    f"batch size, got {num_src_rows} and {int(req_ids.shape[0])}."
                )
            if int(cache_loc.shape[0]) != num_src_rows:
                raise RuntimeError(
                    "Sparse v2 decode offload expects out_cache_loc rows to match "
                    f"compact KV rows, got {int(cache_loc.shape[0])} and "
                    f"{num_src_rows}."
                )

            dst_index = (req_ids * self.max_context_len + token_pos).contiguous()
            valid_mask = (
                (forward_batch.seq_lens != 1)
                & (cache_loc >= 0)
                & (req_ids >= 0)
                & (token_pos >= 0)
                & (token_pos < self.max_context_len)
            ).contiguous()
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
            if extend_seq_lens_sum == num_src_rows:
                # Chunk prefill emits compact rows as [req0 tokens][req1 tokens]...
                # instead of graph-captured padded [B, tokens_per_req] rows.
                seq_starts = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens
                flat_req_ids = torch.repeat_interleave(
                    req_ids, extend_seq_lens, output_size=num_src_rows
                )
                flat_seq_starts = torch.repeat_interleave(
                    seq_starts, extend_seq_lens, output_size=num_src_rows
                )
                flat_prefix_lens = torch.repeat_interleave(
                    extend_prefix_lens, extend_seq_lens, output_size=num_src_rows
                )
                token_pos = (
                    flat_prefix_lens
                    + torch.arange(num_src_rows, device=device, dtype=torch.long)
                    - flat_seq_starts
                )
                dst_index = (
                    flat_req_ids * self.max_context_len + token_pos
                ).contiguous()
                valid_mask = (
                    (flat_req_ids >= 0)
                    & (token_pos >= 0)
                    & (token_pos < self.max_context_len)
                )
            else:
                if num_src_rows % batch_size != 0:
                    raise RuntimeError(
                        "Sparse v2 prefill offload expects either compact ragged "
                        "layout with rows=sum(extend_seq_lens) or graph-style "
                        f"row-major layout [B, tokens_per_req], got rows={num_src_rows}, "
                        f"batch={batch_size}, extend_seq_lens_sum={extend_seq_lens_sum}."
                    )

                # Graph-friendly static layout:
                # compact rows are interpreted as [batch_size, tokens_per_req].
                # Invalid padded columns are masked by local_offsets < extend_seq_lens.
                tokens_per_req = num_src_rows // batch_size

                local_offsets = (
                    torch.arange(tokens_per_req, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_size, tokens_per_req)
                )
                req_ids_2d = req_ids.unsqueeze(1).expand(batch_size, tokens_per_req)
                token_pos_2d = extend_prefix_lens.unsqueeze(1) + local_offsets

                dst_index = (
                    (req_ids_2d * self.max_context_len + token_pos_2d)
                    .reshape(-1)
                    .contiguous()
                )
                valid_mask = (
                    (local_offsets < extend_seq_lens.unsqueeze(1))
                    & (req_ids_2d >= 0)
                    & (token_pos_2d >= 0)
                    & (token_pos_2d < self.max_context_len)
                ).reshape(-1)

            if (
                forward_batch.out_cache_loc is not None
                and int(forward_batch.out_cache_loc.numel()) == num_src_rows
            ):
                valid_mask = valid_mask & (
                    forward_batch.out_cache_loc.to(torch.long) >= 0
                )
            valid_mask = valid_mask.contiguous()

        assert src_tensor.shape[1:] == dst_tensor.shape[2:]
        assert src_index.shape == dst_index.shape == valid_mask.shape

        actual_stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(actual_stream):
            unidex_copy_inplace(
                src_tensor,
                dst_tensor,
                src_index,
                dst_index,
                valid_mask,
                1,  # src_tensor: [num_new_tokens, nhead, head_dim]
                2,  # dst_tensor: [num_req, max_context_len, nhead, head_dim]
                block_dim=48,
                dst_ptr=self.dev_ptr_list[layer_idx],
            )

    def get_forward_kv(
        self,
        layer: Union[RadixAttention, int],
        forward_batch: ForwardBatch,
        stream: Optional[torch.npu.Stream] = None,
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

        kv_cat = torch.empty(
            (total_tokens, self.head_num, self.head_dim),
            dtype=self.store_dtype,
            device=device,
        )

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
        src_index = (src_req_ids * self.max_context_len + token_pos).contiguous()
        dst_index = torch.arange(total_tokens, device=device, dtype=torch.long)
        valid_mask = torch.ones(total_tokens, device=device, dtype=torch.bool)

        actual_stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(actual_stream):
            unidex_copy_inplace(
                self.host_kv_buffer[layer_idx],
                kv_cat,
                src_index,
                dst_index,
                valid_mask,
                2,  # host_kv_buffer: [num_req, max_context_len, nhead, head_dim]
                1,  # kv_cat: [total_tokens, nhead, head_dim]
                block_dim=48,
                src_ptr=self.dev_ptr_list[layer_idx],
            )

        k_nope, k_pe = kv_cat.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return k_nope.contiguous(), k_pe.contiguous()

    def materialize_selected_kv(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
        selected_kv_buffer: torch.Tensor,
        stream: torch.npu.Stream,
    ) -> None:
        """Gather current selections, then refill and publish on one side stream.

        The caller waits for miss_done before attention and refill_done after
        attention, protecting the shared workspace and the next cache lookup.
        """
        layer_idx = layer.layer_id - self.start_layer
        stream = stream if stream is not None else torch.npu.current_stream()

        with torch.npu.stream(stream):
            # Route invalid requests to sentinel rows without changing graph shape.
            # slot_map_row_indices: invalid -> self.size (reserved slot-map row)
            # device_cache_row_indices: invalid -> 0 (masked by valid_topk_mask)
            req_pool_indices = forward_batch.req_pool_indices
            req_pool_indices = req_pool_indices.to(torch.long).contiguous()
            valid_req_mask = (
                (req_pool_indices > 0)
                & (req_pool_indices < self.size)
                & (forward_batch.seq_lens[: req_pool_indices.numel()] > 0)
            )
            slot_map_row_indices = torch.where(
                valid_req_mask,
                req_pool_indices,
                torch.full_like(req_pool_indices, self.size),
            )
            device_cache_row_indices = torch.where(
                valid_req_mask,
                req_pool_indices,
                torch.zeros_like(req_pool_indices),
            )

            # Normalize top-k indices and mask invalid requests and token IDs.
            topk_indices = normalize_batch_topk_indices(topk_indices)
            batch_size, topk_len = topk_indices.shape
            if selected_kv_buffer.shape[:2] != (batch_size, topk_len):
                raise ValueError(
                    "Selected KV workspace must match padded [batch, top_k]."
                )
            if topk_len > self.sparse_context_len:
                raise RuntimeError(
                    "DSA top-k length exceeds sparse KV device cache capacity: "
                    f"topk_len={topk_len}, sparse_context_len={self.sparse_context_len}."
                )
            valid_topk_mask = (
                (topk_indices >= 0)
                & (topk_indices < self.max_context_len)
                & valid_req_mask.unsqueeze(1)
            )

            # Query the slot map for device-cache hits and their slot positions.
            slot_lookup_req_indices = slot_map_row_indices.to(
                dtype=torch.int32
            ).contiguous()
            slot_lookup_topk_indices = (
                torch.where(valid_topk_mask, topk_indices, self.max_context_len)
                .to(dtype=torch.int32)
                .contiguous()
            )
            token_on_device, device_token_pos = slot_map_lookup(
                self.device_slot_map[layer_idx],
                slot_lookup_req_indices,
                slot_lookup_topk_indices,
            )
            token_on_device = token_on_device.to(torch.bool) & valid_topk_mask

            # Build copy indices on the main stream, then protect their use on
            # the hit and miss streams with copy_ready.
            hit_src_index, hit_dst_index, hit_valid_mask = _build_hit_src_dst_index(
                token_on_device,
                device_token_pos,
                device_cache_row_indices,
                self.device_cache_size,
            )

            host_miss_mask = (~token_on_device) & valid_topk_mask
            miss_src_index, miss_dst_index, miss_valid_mask = _build_miss_src_dst_index(
                host_miss_mask,
                topk_indices,
                device_cache_row_indices,
                self.max_context_len,
            )

            cache_slot_ids = self._device_cache_slot_ids[:topk_len]
            plan = None
            if self.use_lru:
                plan = plan_sparse_lru(
                    hit_mask=token_on_device,
                    hit_slots=device_token_pos,
                    miss_mask=host_miss_mask,
                    cached_tokens=self.cache_tokens[layer_idx][slot_map_row_indices],
                    lru_order=self.lru_order[layer_idx][slot_map_row_indices],
                )
                refill_slots = plan.miss_slots
                refill_valid_mask = host_miss_mask.reshape(-1).contiguous()
            else:
                # Preserve the K-slot path without adding an LRU sort.
                refill_slots = cache_slot_ids.expand(batch_size, -1)
                refill_valid_mask = valid_topk_mask.reshape(-1).contiguous()
            request_cache_offsets = (
                device_cache_row_indices.unsqueeze(1) * self.device_cache_size
            )
            refill_src_index = torch.arange(
                batch_size * topk_len,
                dtype=torch.long,
                device=topk_indices.device,
            )
            refill_dst_index = (
                (request_cache_offsets + refill_slots).reshape(-1).contiguous()
            )
            # These tensors were allocated on the producer stream. Keep their
            # storage alive until the side stream has finished using them.
            side_inputs = [
                topk_indices,
                slot_map_row_indices,
                host_miss_mask,
                miss_src_index,
                miss_dst_index,
                miss_valid_mask,
                refill_src_index,
                refill_dst_index,
                refill_slots,
                refill_valid_mask,
            ]
            if plan is not None:
                side_inputs.extend(plan)
            for tensor in side_inputs:
                tensor.record_stream(self._materialize_h2d_miss_stream)
            _record_stream_event(stream, self.copy_ready)

        # Copy device-cache hits into the selected KV buffer.
        with torch.npu.stream(stream):
            unidex_copy_inplace(
                self.device_kv_buffer[layer_idx],
                selected_kv_buffer,
                hit_src_index,
                hit_dst_index,
                hit_valid_mask,
                2,
                2,  #
                block_dim=24,
            )
            _record_stream_event(stream, self.hit_done)

        # Copy host shared-memory misses into the selected KV buffer.
        with torch.npu.stream(self._materialize_h2d_miss_stream):
            _wait_stream_event(self._materialize_h2d_miss_stream, self.copy_ready)
            unidex_copy_inplace(
                self.host_kv_buffer[layer_idx],
                selected_kv_buffer,
                miss_src_index,
                miss_dst_index,
                miss_valid_mask,
                2,
                2,
                block_dim=24,
                src_ptr=self.dev_ptr_list[layer_idx],
            )
            _record_stream_event(self._materialize_h2d_miss_stream, self.miss_done)

            # miss_done deliberately precedes refill and metadata publication:
            # attention can consume the workspace while this stream refills.
            _wait_stream_event(self._materialize_h2d_miss_stream, self.hit_done)
            unidex_copy_inplace(
                selected_kv_buffer,
                self.device_kv_buffer[layer_idx],
                refill_src_index,
                refill_dst_index,
                refill_valid_mask,
                2,
                2,
                block_dim=24,
            )
            # No full-pool reset: inactive requests keep their resident mappings.
            # In LRU mode invalidate only victims, after their hit copies finish.
            if plan is None:
                self.device_slot_map[layer_idx].index_fill_(0, slot_map_row_indices, -1)
            else:
                evicted = plan.evicted_tokens
                evict_mask = host_miss_mask & (evicted >= 0)
                evict_indices = (
                    (
                        slot_map_row_indices.unsqueeze(1) * self._slot_map_width
                        + evicted.long().clamp(min=0)
                    )
                    .reshape(-1)
                    .contiguous()
                )
                unidex_copy_inplace(
                    torch.full_like(evict_indices.view(-1, 1), -1, dtype=torch.int32),
                    self.device_slot_map[layer_idx].view(-1, 1),
                    refill_src_index,
                    evict_indices,
                    evict_mask.reshape(-1).contiguous(),
                    1,
                    1,
                    block_dim=48,
                )
                unidex_copy_inplace(
                    topk_indices.to(torch.int32).reshape(-1, 1).contiguous(),
                    self.cache_tokens[layer_idx].view(-1, 1),
                    refill_src_index,
                    refill_dst_index,
                    refill_valid_mask,
                    1,
                    1,
                    block_dim=48,
                )
                self.lru_order[layer_idx].index_copy_(
                    0, slot_map_row_indices, plan.next_order
                )

            slot_map_token_indices = torch.where(
                refill_valid_mask.view(batch_size, topk_len),
                topk_indices.to(torch.long),
                torch.full_like(topk_indices, self.max_context_len, dtype=torch.long),
            )
            slot_map_slot_values = torch.where(
                refill_valid_mask.view(batch_size, topk_len),
                refill_slots.to(torch.int32),
                -1,
            )
            slot_map_flat_indices = (
                slot_map_row_indices.unsqueeze(1) * self._slot_map_width
                + slot_map_token_indices
            ).reshape(-1)
            unidex_copy_inplace(
                slot_map_slot_values.reshape(-1, 1).contiguous(),
                self.device_slot_map[layer_idx].view(-1, 1),
                torch.arange(
                    slot_map_slot_values.numel(),
                    dtype=torch.long,
                    device=topk_indices.device,
                ),
                slot_map_flat_indices,
                refill_valid_mask,
                1,
                1,
                block_dim=48,
            )
            _record_stream_event(self._materialize_h2d_miss_stream, self.refill_done)


_global_sparse_kv_manager: Optional[SparseKVCacheManager] = None


def register_sparse_kv_manager(manager: SparseKVCacheManager) -> None:
    global _global_sparse_kv_manager
    _global_sparse_kv_manager = manager


def get_sparse_kv_manager() -> Optional[SparseKVCacheManager]:
    return _global_sparse_kv_manager


def _build_hit_src_dst_index(
    token_on_device: torch.Tensor,
    device_token_pos: torch.Tensor,
    current_req_indices: torch.Tensor,
    sparse_context_len: int,
):
    """
    token_on_device: [bs, topk], bool
    device_token_pos: [bs, topk], int64 or int32
    current_req_indices: [bs], int64

    Return:
        src_index_full: [bs * topk], int64
        dst_index_full: [bs * topk], int64
        valid_mask: [bs * topk], bool

    Flattening rule:
        src row = req_id * sparse_context_len + device_token_pos
        dst row = batch_id * topk + topk_pos
    """
    if token_on_device.dim() != 2 or device_token_pos.dim() != 2:
        raise RuntimeError(
            f"token_on_device and device_token_pos must be 2-D, got "
            f"{token_on_device.dim()} and {device_token_pos.dim()}"
        )
    if token_on_device.shape != device_token_pos.shape:
        raise RuntimeError(
            f"token_on_device and device_token_pos must have the same shape, got "
            f"{tuple(token_on_device.shape)} and {tuple(device_token_pos.shape)}"
        )
    if current_req_indices.dim() != 1:
        raise RuntimeError(
            f"current_req_indices must be 1-D, got {current_req_indices.dim()}"
        )

    bs, topk = token_on_device.shape
    if current_req_indices.numel() != bs:
        raise RuntimeError(
            f"current_req_indices length mismatch: "
            f"{current_req_indices.numel()} vs batch {bs}"
        )
    if sparse_context_len <= 0:
        raise RuntimeError(
            f"sparse_context_len must be positive, got {sparse_context_len}"
        )

    device = token_on_device.device

    valid_mask = token_on_device.reshape(-1).contiguous()

    flat_dst_index_all = torch.arange(
        bs * topk,
        device=device,
        dtype=torch.int64,
    )

    req_offsets = current_req_indices.to(torch.int64).unsqueeze(1) * sparse_context_len
    src_index_2d = req_offsets + device_token_pos.to(torch.int64)
    flat_src_index_all = src_index_2d.reshape(-1).contiguous()

    return flat_src_index_all, flat_dst_index_all, valid_mask


def _build_miss_src_dst_index(
    token_from_host: torch.Tensor,
    topk_indices: torch.Tensor,
    current_req_indices: torch.Tensor,
    max_context_len: int,
):
    if token_from_host.dim() != 2 or topk_indices.dim() != 2:
        raise RuntimeError(
            f"token_from_host and topk_indices must be 2-D, got "
            f"{token_from_host.dim()} and {topk_indices.dim()}"
        )
    if token_from_host.shape != topk_indices.shape:
        raise RuntimeError(
            f"token_from_host and topk_indices must have the same shape, got "
            f"{tuple(token_from_host.shape)} and {tuple(topk_indices.shape)}"
        )
    if current_req_indices.dim() != 1:
        raise RuntimeError(
            f"current_req_indices must be 1-D, got {current_req_indices.dim()}"
        )
    if current_req_indices.numel() != token_from_host.shape[0]:
        raise RuntimeError(
            f"current_req_indices length mismatch: "
            f"{current_req_indices.numel()} vs batch {token_from_host.shape[0]}"
        )

    bs, topk = token_from_host.shape
    device = token_from_host.device

    valid_2d = token_from_host & (topk_indices >= 0) & (topk_indices < max_context_len)
    valid_mask = valid_2d.reshape(-1).contiguous()

    flat_dst_index_all = torch.arange(
        bs * topk,
        device=device,
        dtype=torch.int64,
    )

    req_offsets = current_req_indices.to(torch.int64).unsqueeze(1) * max_context_len
    src_index_2d = req_offsets + topk_indices.to(torch.int64)
    flat_src_index_all = src_index_2d.reshape(-1).contiguous()

    return flat_src_index_all, flat_dst_index_all, valid_mask
