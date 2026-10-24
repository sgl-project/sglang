"""Sparsity-driven KV offload manager for the Ascend backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import msgspec
import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    slot_map_lookup,
    unidex_copy_inplace,
)

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    get_sparsity_driven_kv_offload_device_cache_capacity,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import (
    get_cuda_graph_max_batch_size,
    get_eager_max_batch_size,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    import torch.npu

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class KVRefillPlan(msgspec.Struct, frozen=True, kw_only=True):
    victim_slots: torch.Tensor
    src_indices: torch.Tensor
    request_offsets: torch.Tensor
    valid_mask: torch.Tensor


class _KVMaterializationInputs(msgspec.Struct, frozen=True, kw_only=True):
    slot_map_rows: torch.Tensor
    cache_rows: torch.Tensor
    lookup_req_indices: torch.Tensor
    topk_indices: torch.Tensor
    request_offsets: torch.Tensor
    copy_indices: torch.Tensor
    valid_mask: torch.Tensor
    device_token_pos: torch.Tensor
    hit_position_mask: Optional[torch.Tensor]
    hit_mask: torch.Tensor
    miss_mask: torch.Tensor


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
    ) -> None:
        enable_memory_saver = False
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

        # Include the padding row because real request IDs can equal the
        # configured capacity when row 0 is reserved for graph padding.
        self.size = int(req_to_token_pool.req_to_token.shape[0])
        self.max_context_len = req_to_token_pool.max_context_len
        self.sparse_context_len = int(sparse_context_len)
        if self.sparse_context_len <= 0:
            raise ValueError(
                "SparseKVCacheManager requires a positive sparse_context_len, "
                f"got {self.sparse_context_len}."
            )
        self.enable_lru = envs.SGLANG_NPU_SPARSE_KV_ENABLE_LRU.get()
        self.device_cache_capacity = (
            get_sparsity_driven_kv_offload_device_cache_capacity(
                sparse_context_len=self.sparse_context_len,
                enable_lru=self.enable_lru,
            )
        )
        if self.enable_lru:
            from sgl_kernel_npu.sparsity_driven_kv_offload import (
                fused_timestamp_lru_metadata_update_with_probation,
                parallel_lru_metadata_write,
            )

            self._lru_metadata_update = (
                fused_timestamp_lru_metadata_update_with_probation
            )
            self._lru_metadata_write = parallel_lru_metadata_write
        # Bind allocations to this rank's device. The pool's device string may
        # be bare "npu", which can resolve to NPU 0 in the PD receive thread.
        self.device = req_to_token_pool.req_to_token.device
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
        self._probation_age = envs.SGLANG_NPU_SPARSE_KV_PROBATION_AGE.get()

        # Hit and miss copies overlap on independent 24-AIV streams. Metadata
        # update runs on a third stream after both copies, while refill remains
        # on the caller stream to keep selected_kv_buffer stream-local after
        # the copy join.
        self._materialize_d2d_hit_stream = torch.npu.Stream()
        self._materialize_h2d_miss_stream = torch.npu.Stream()
        self._materialize_metadata_update_stream = torch.npu.Stream()
        # Decode offload is submitted to a persistent side stream. Keeping the
        # stream and events alive on the manager is required by NPU graph
        # capture: creating either object from the captured forward would make
        # replay depend on Python-side state that is not part of the graph.
        self._decode_offload_stream = torch.npu.Stream()
        self._decode_offload_done = [torch.npu.Event() for _ in range(self.layer_num)]
        self._materialize_hit_done = torch.npu.Event()
        self._materialize_miss_done = torch.npu.Event()
        self._materialize_victim_slot_select_done = torch.npu.Event()
        self._materialize_metadata_update_done = torch.npu.Event()

        # device KV buffer
        try:
            with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                # Physical cache slots are stable. LRU ordering is maintained
                # separately in device_lru_slots so hits never need a writeback.
                # Request IDs start at row 0. Invalid requests are masked before
                # any device-cache row is accessed.
                self.device_kv_buffer: list[torch.Tensor] = [
                    torch.empty(
                        (
                            self.size,
                            self.device_cache_capacity,
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
                # Reserve the last row for invalid requests and ensure token index
                # `max_context_len` is a valid sentinel column for masked writes.
                # The row width is also aligned to eight int32 values (32 bytes).
                self.device_slot_map: list[torch.Tensor] = [
                    torch.full(
                        (
                            self.size + 1,
                            (self.max_context_len // 8 + 1) * 8,
                        ),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                if self.enable_lru:
                    # Reverse metadata from stable physical slot to token position.
                    self.device_slot_tokens: list[torch.Tensor] = [
                        torch.full(
                            (self.size, self.device_cache_capacity),
                            -1,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

                    # Physical-slot permutation paired with descending timestamps.
                    # Initial equal-stamp ties may use any slot order.
                    self._initial_lru_slot_order = torch.arange(
                        self.device_cache_capacity,
                        dtype=torch.int32,
                        device=self.device,
                    ).unsqueeze(0)
                    self.device_lru_slots: list[torch.Tensor] = [
                        self._initial_lru_slot_order.expand(
                            self.size, self.device_cache_capacity
                        ).clone()
                        for _ in range(self.layer_num)
                    ]
                    # Timestamp is aligned with device_lru_slots: larger means
                    # older. The fused AIV kernel increments with saturation and
                    # resets hit/newly-filled slots to zero.
                    self.device_lru_slot_stamps: list[torch.Tensor] = [
                        torch.zeros(
                            (self.size, self.device_cache_capacity),
                            dtype=torch.int32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
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
                    name=f"host_kv_layer_{layer_idx}_rank_{device_id}",
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

        # Graph/eager padding can repeat request row 0 beyond the request-pool
        # capacity. Only compute templates need the padded batch capacity.
        self._decode_batch_capacity = max(
            self.size,
            get_cuda_graph_max_batch_size(req_to_token_pool.size),
            get_eager_max_batch_size(req_to_token_pool.size),
        )
        # Static flattened row addresses for selected_kv_buffer. The same
        # addresses are used as the hit/miss copy destinations and the refill
        # copy sources. Keeping the full padded-batch template avoids three
        # arange/add/reshape sequences on every decode step; materialization
        # only takes a view covering the current graph batch.
        self._selected_kv_copy_indices = torch.arange(
            self._decode_batch_capacity * self.sparse_context_len,
            dtype=torch.long,
            device=self.device,
        )
        # Static sparse-attention metadata templates. Materialization returns
        # the dynamic validity mask/counts; attention applies them to these
        # initialization-time tensors without rebuilding arange/full/ones.
        self._compact_sparse_indices = torch.arange(
            self.sparse_context_len,
            dtype=torch.int32,
            device=self.device,
        ).view(1, 1, 1, self.sparse_context_len)
        self._invalid_sparse_indices = torch.full(
            (1, 1, 1, self.sparse_context_len),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._decode_query_seq_lengths = torch.ones(
            self._decode_batch_capacity,
            dtype=torch.int32,
            device=self.device,
        )
        self._zero_sparse_index = torch.zeros(
            (self._decode_batch_capacity, 1, self.head_num),
            dtype=torch.int32,
            device=self.device,
        )
        self._slot_map_sentinel_req_indices = torch.full(
            (self._decode_batch_capacity,),
            self.size,
            dtype=torch.long,
            device=self.device,
        )
        self._zero_req_indices = torch.zeros(
            self._decode_batch_capacity, dtype=torch.long, device=self.device
        )
        self._slot_map_width = (self.max_context_len // 8 + 1) * 8
        self.pd_decode_k_staging: Optional[list[torch.Tensor]] = None
        self.pd_decode_v_staging: Optional[list[torch.Tensor]] = None
        self._pd_decode_staging_token_capacity = self.max_context_len
        self._pd_decode_copy_stream = torch.npu.Stream()
        self._pd_room_to_req_pool_idx: dict[int, int] = {}
        self._pd_room_to_input_len: dict[int, int] = {}
        self._pd_req_pool_idx_to_room: dict[int, int] = {}

        self._install_req_lifecycle_hooks(req_to_token_pool)

    def _raise_buffer_allocation_error(
        self,
        buffer_name: str,
        exc: Exception,
    ) -> None:
        raise RuntimeError(
            "Failed to allocate sparse KV buffer "
            f"{buffer_name}: req_capacity={self.size}, "
            f"max_context_len={self.max_context_len}, "
            f"sparse_context_len={self.sparse_context_len}, "
            f"device_cache_capacity={self.device_cache_capacity}. "
            "The sparse KV request capacity may be too large; set a smaller "
            "--max-running-requests for sparse KV offload."
        ) from exc

    def _pd_decode_k_buffer_shape(self, slot_count: int) -> tuple[int, int, int, int]:
        return (
            int(slot_count),
            int(self._pd_decode_staging_token_capacity),
            self.head_num,
            self.kv_lora_rank,
        )

    def _pd_decode_v_buffer_shape(self, slot_count: int) -> tuple[int, int, int, int]:
        return (
            int(slot_count),
            int(self._pd_decode_staging_token_capacity),
            self.head_num,
            self.qk_rope_head_dim,
        )

    def _alloc_pd_device_buffers(
        self,
        slot_count: int,
        buffer_name: str,
        shape: tuple[int, int, int, int],
        fill_value: float,
    ) -> list[torch.Tensor]:
        enable_memory_saver = False
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        try:
            with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                return [
                    torch.full(
                        shape,
                        fill_value,
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
        except Exception as e:
            self._raise_buffer_allocation_error(buffer_name, e)

    def ensure_pd_decode_staging_buffers(
        self,
        slot_count: int = 1,
        page_size: Optional[int] = None,
    ) -> None:
        slot_count = int(slot_count)
        if slot_count <= 0:
            raise ValueError(
                f"Sparse KV PD decode slot_count must be > 0: {slot_count}"
            )

        transfer_page_size = int(page_size or self.paged_kv_cache.page_size)
        if transfer_page_size <= 0:
            raise ValueError(
                "Sparse KV PD decode transfer page_size must be positive, "
                f"got {transfer_page_size}."
            )
        aligned_capacity = (
            (self.max_context_len + transfer_page_size - 1)
            // transfer_page_size
            * transfer_page_size
        )
        if self.pd_decode_k_staging is not None or self.pd_decode_v_staging is not None:
            if self.pd_decode_k_staging is None or self.pd_decode_v_staging is None:
                raise RuntimeError(
                    "Sparse KV PD decode staging buffers are partially initialized."
                )
            current_slots = int(self.pd_decode_k_staging[0].shape[0])
            current_capacity = int(self.pd_decode_k_staging[0].shape[1])
            if current_slots < slot_count or current_capacity < aligned_capacity:
                raise RuntimeError(
                    "Sparse KV PD decode staging was already initialized with "
                    f"shape={tuple(self.pd_decode_k_staging[0].shape)}, cannot "
                    f"grow to slots={slot_count}, capacity={aligned_capacity}."
                )
            return

        self._pd_decode_staging_token_capacity = aligned_capacity
        logger.info(
            "Sparse KV PD decode HBM staging buffer shapes: k=%s, v=%s",
            self._pd_decode_k_buffer_shape(slot_count),
            self._pd_decode_v_buffer_shape(slot_count),
        )
        self.pd_decode_k_staging = self._alloc_pd_device_buffers(
            slot_count,
            "pd_decode_k_staging",
            self._pd_decode_k_buffer_shape(slot_count),
            4444.44,
        )
        self.pd_decode_v_staging = self._alloc_pd_device_buffers(
            slot_count,
            "pd_decode_v_staging",
            self._pd_decode_v_buffer_shape(slot_count),
            5555.55,
        )

    def get_pd_decode_transfer_buf_infos(
        self,
        page_size: Optional[int] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        if self.pd_decode_k_staging is None or self.pd_decode_v_staging is None:
            raise RuntimeError("Sparse KV PD decode staging buffer is not initialized.")
        transfer_page_size = int(page_size or self.paged_kv_cache.page_size)
        if transfer_page_size <= 0:
            raise ValueError(
                "Sparse KV PD decode transfer page_size must be positive, "
                f"got {transfer_page_size}."
            )
        return (
            [buf.data_ptr() for buf in self.pd_decode_k_staging]
            + [buf.data_ptr() for buf in self.pd_decode_v_staging],
            [buf.nbytes for buf in self.pd_decode_k_staging]
            + [buf.nbytes for buf in self.pd_decode_v_staging],
            [buf[0, 0].nbytes * transfer_page_size for buf in self.pd_decode_k_staging]
            + [
                buf[0, 0].nbytes * transfer_page_size
                for buf in self.pd_decode_v_staging
            ],
        )

    def get_pd_decode_staging_token_capacity(self) -> int:
        return int(self._pd_decode_staging_token_capacity)

    def get_pd_decode_pages_per_slot(self, page_size: Optional[int] = None) -> int:
        transfer_page_size = int(page_size or self.paged_kv_cache.page_size)
        if transfer_page_size <= 0:
            raise ValueError(
                "Sparse KV PD decode transfer page_size must be positive, "
                f"got {transfer_page_size}."
            )
        return self.get_pd_decode_staging_token_capacity() // transfer_page_size

    def record_pd_request_metadata(self, req: Req) -> None:
        bootstrap_room = getattr(req, "bootstrap_room", None)
        req_pool_idx = req.kv.req_pool_idx
        if bootstrap_room is None or req_pool_idx is None:
            return
        room = int(bootstrap_room)
        pool_idx = int(req_pool_idx)
        input_len = int(len(req.origin_input_ids))
        old_room = self._pd_req_pool_idx_to_room.get(pool_idx)
        if old_room is not None and old_room != room:
            self.clear_pd_request_metadata(bootstrap_room=old_room)
        self._pd_room_to_req_pool_idx[room] = pool_idx
        self._pd_room_to_input_len[room] = input_len
        self._pd_req_pool_idx_to_room[pool_idx] = room

    def clear_pd_request_metadata(
        self,
        bootstrap_room: Optional[int] = None,
        req_pool_idx: Optional[int] = None,
    ) -> None:
        if bootstrap_room is None and req_pool_idx is not None:
            bootstrap_room = self._pd_req_pool_idx_to_room.pop(int(req_pool_idx), None)
        if bootstrap_room is None:
            return
        room = int(bootstrap_room)
        pool_idx = self._pd_room_to_req_pool_idx.pop(room, None)
        self._pd_room_to_input_len.pop(room, None)
        if pool_idx is not None:
            self._pd_req_pool_idx_to_room.pop(int(pool_idx), None)

    def clear_all_pd_request_metadata(self) -> None:
        self._pd_room_to_req_pool_idx.clear()
        self._pd_room_to_input_len.clear()
        self._pd_req_pool_idx_to_room.clear()

    def get_pd_copy_metadata(
        self,
        bootstrap_room: int,
        decode_prefix_len: int = 0,
    ) -> tuple[int, int]:
        room = int(bootstrap_room)
        if room not in self._pd_room_to_req_pool_idx:
            raise RuntimeError(
                f"Sparse KV PD metadata for bootstrap_room={room} is not recorded."
            )
        req_pool_idx = self._pd_room_to_req_pool_idx[room]
        input_len = self._pd_room_to_input_len[room]
        token_count = input_len - int(decode_prefix_len)
        if token_count < 0:
            raise RuntimeError(
                "Sparse KV PD got negative token_count from metadata: "
                f"input_len={input_len}, decode_prefix_len={decode_prefix_len}."
            )
        return req_pool_idx, token_count

    def offload_pd_decode_staging_to_host(
        self,
        slot_id: int,
        req_pool_idx: int,
        token_count: int,
        stream: Optional[torch.npu.Stream] = None,
    ) -> None:
        if self.pd_decode_k_staging is None or self.pd_decode_v_staging is None:
            raise RuntimeError("Sparse KV PD decode staging buffer is not initialized.")
        slot_id = int(slot_id)
        req_pool_idx = int(req_pool_idx)
        token_count = int(token_count)
        slot_count = int(self.pd_decode_k_staging[0].shape[0])
        if slot_id < 0 or slot_id >= slot_count:
            raise RuntimeError(
                f"Sparse KV PD decode got slot_id={slot_id}, outside slot_count "
                f"{slot_count}."
            )
        if req_pool_idx < 0 or req_pool_idx >= self.size:
            raise RuntimeError(
                f"Sparse KV PD decode got req_pool_idx={req_pool_idx}, outside "
                f"sparse pool size {self.size}."
            )
        if token_count < 0 or token_count > self.max_context_len:
            raise RuntimeError(
                f"Sparse KV PD decode got token_count={token_count}, outside "
                f"max_context_len {self.max_context_len}."
            )

        self.host_kv_ctx_len[req_pool_idx] = token_count
        actual_stream = stream if stream is not None else self._pd_decode_copy_stream
        with torch.npu.stream(actual_stream):
            # The receive thread must finish both metadata reset and KV copies
            # before publishing transfer success, including empty transfers.
            self.reset_requests([req_pool_idx])
            for layer_idx in range(self.layer_num):
                dst = self.host_kv_buffer[layer_idx][req_pool_idx, :token_count]
                dst[..., : self.kv_lora_rank].copy_(
                    self.pd_decode_k_staging[layer_idx][slot_id, :token_count],
                    non_blocking=False,
                )
                dst[..., self.kv_lora_rank :].copy_(
                    self.pd_decode_v_staging[layer_idx][slot_id, :token_count],
                    non_blocking=False,
                )
        actual_stream.synchronize()

    def init_req(self, req: Req) -> None:
        if getattr(req, "inflight_middle_chunks", getattr(req, "is_chunked", 0)) > 0:
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
            if self.enable_lru:
                self.device_slot_tokens[layer_idx].index_fill_(0, req_ids_tensor, -1)
                self.device_lru_slots[layer_idx].index_copy_(
                    0,
                    req_ids_tensor,
                    self._initial_lru_slot_order.expand(
                        len(req_ids), self.device_cache_capacity
                    ).contiguous(),
                )
                self.device_lru_slot_stamps[layer_idx].index_fill_(0, req_ids_tensor, 0)

    def _install_req_lifecycle_hooks(self, req_to_token_pool: ReqToTokenPool) -> None:
        original_alloc = getattr(
            req_to_token_pool, "_sparse_kv_original_alloc", req_to_token_pool.alloc
        )
        setattr(req_to_token_pool, "_sparse_kv_original_alloc", original_alloc)
        original_free = getattr(
            req_to_token_pool, "_sparse_kv_original_free", req_to_token_pool.free
        )
        setattr(req_to_token_pool, "_sparse_kv_original_free", original_free)
        original_clear = getattr(
            req_to_token_pool, "_sparse_kv_original_clear", req_to_token_pool.clear
        )
        setattr(req_to_token_pool, "_sparse_kv_original_clear", original_clear)

        def alloc_with_sparse_reset(reqs: list[Req]) -> Optional[List[int]]:
            newly_allocated = [req.kv.req_pool_idx is None for req in reqs]
            req_pool_indices = original_alloc(reqs)
            if req_pool_indices is not None:
                self.reset_requests(
                    [
                        req_pool_indices[i]
                        for i, is_new in enumerate(newly_allocated)
                        if is_new
                    ]
                )
                for req in reqs:
                    self.record_pd_request_metadata(req)
            return req_pool_indices

        def free_with_sparse_clear(req: Req) -> None:
            req_pool_idx = req.kv.req_pool_idx
            try:
                if req_pool_idx is not None:
                    self.reset_requests([req_pool_idx])
            finally:
                self.clear_pd_request_metadata(req_pool_idx=req_pool_idx)
                original_free(req)

        def clear_with_sparse_clear():
            self.clear_all_pd_request_metadata()
            return original_clear()

        setattr(req_to_token_pool, "alloc", alloc_with_sparse_reset)
        setattr(req_to_token_pool, "free", free_with_sparse_clear)
        setattr(req_to_token_pool, "clear", clear_with_sparse_clear)

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

        if forward_batch.forward_mode.is_decode():
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
            global_num_token_non_padded_cpu = (
                forward_batch.global_num_token_non_padded_cpu
            )
            unpadded_prefill_rows = (
                int(global_num_token_non_padded_cpu)
                if global_num_token_non_padded_cpu is not None
                else None
            )
            has_tail_padding = (
                unpadded_prefill_rows == extend_seq_lens_sum
                and extend_seq_lens_sum < num_src_rows
            )

            if extend_seq_lens_sum == num_src_rows or has_tail_padding:
                # Chunk prefill emits compact rows as [req0 tokens][req1 tokens]...
                # instead of graph-captured padded [B, tokens_per_req] rows. MLP sync
                # can append tail padding tokens; keep static rows but mask them out.
                seq_starts = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens
                flat_req_ids = torch.repeat_interleave(
                    req_ids, extend_seq_lens, output_size=extend_seq_lens_sum
                )
                flat_seq_starts = torch.repeat_interleave(
                    seq_starts, extend_seq_lens, output_size=extend_seq_lens_sum
                )
                flat_prefix_lens = torch.repeat_interleave(
                    extend_prefix_lens, extend_seq_lens, output_size=extend_seq_lens_sum
                )
                token_pos = (
                    flat_prefix_lens
                    + torch.arange(extend_seq_lens_sum, device=device, dtype=torch.long)
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
                if has_tail_padding:
                    pad_rows = num_src_rows - extend_seq_lens_sum
                    dst_index = torch.cat(
                        [
                            dst_index,
                            torch.zeros(pad_rows, device=device, dtype=torch.long),
                        ],
                        dim=0,
                    )
                    valid_mask = torch.cat(
                        [
                            valid_mask,
                            torch.zeros(pad_rows, device=device, dtype=torch.bool),
                        ],
                        dim=0,
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
                and int(forward_batch.out_cache_loc.numel()) >= num_src_rows
            ):
                valid_mask = valid_mask & (
                    forward_batch.out_cache_loc[:num_src_rows].to(torch.long) >= 0
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

    def offload_v2_decode_async(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        producer_stream: torch.npu.Stream,
    ) -> torch.npu.Event:
        """Submit decode KV offload on the graph-capturable side stream.

        The returned persistent event represents host-KV readiness for this
        layer. Consumers that may read the just-written token must wait for it;
        independent decode preparation can continue on ``producer_stream``.
        """
        if not forward_batch.forward_mode.is_decode():
            raise RuntimeError("Async sparse KV offload is only valid for decode.")

        layer_idx = layer.layer_id - self.start_layer
        self._decode_offload_stream.wait_stream(producer_stream)
        with torch.npu.stream(self._decode_offload_stream):
            self.offload_v2(
                k,
                k_rope,
                layer,
                forward_batch,
                self._decode_offload_stream,
            )
            _record_stream_event(
                self._decode_offload_stream,
                self._decode_offload_done[layer_idx],
            )
        # These tensors were allocated on the producer stream but are consumed
        # asynchronously. This also keeps error paths safe if the caller exits
        # before reaching the host-miss join below.
        k.record_stream(self._decode_offload_stream)
        k_rope.record_stream(self._decode_offload_stream)
        return self._decode_offload_done[layer_idx]

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

        valid_reqs = (seq_lens > 0) & (req_ids >= 0)
        req_ids = req_ids[valid_reqs].contiguous()
        seq_lens = seq_lens[valid_reqs].contiguous()

        if int(seq_lens.numel()) == 0:
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

        if bool((req_ids >= self.size).any().item()):
            raise RuntimeError(
                f"get_forward_kv got req id outside sparse pool size {self.size}."
            )
        if bool((seq_lens > self.max_context_len).any().item()):
            raise RuntimeError(
                "get_forward_kv got seq_len larger than max_context_len "
                f"{self.max_context_len}."
            )

        total_tokens = int(seq_lens.sum().item())
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
        host_kv_ready_event: Optional[torch.npu.Event] = None,
    ) -> tuple[KVRefillPlan, torch.Tensor, torch.Tensor]:
        """Materialize selected KV and return its refill plan, mask and counts."""
        layer_idx = layer.layer_id - self.start_layer
        stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(stream):
            inputs = self._prepare_materialization_inputs(
                layer_idx=layer_idx,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                topk_indices=topk_indices,
                selected_kv_buffer=selected_kv_buffer,
            )
            valid_topk_counts = inputs.valid_mask.sum(dim=1, dtype=torch.int32)

        self._copy_selected_kv(
            layer_idx=layer_idx,
            inputs=inputs,
            selected_kv_buffer=selected_kv_buffer,
            stream=stream,
            host_kv_ready_event=host_kv_ready_event,
        )
        plan = self._prepare_refill_plan(layer_idx=layer_idx, inputs=inputs)

        # Attention input preparation needs the copies, but can overlap metadata.
        _wait_stream_event(stream, self._materialize_hit_done)
        _wait_stream_event(stream, self._materialize_miss_done)
        return plan, inputs.valid_mask, valid_topk_counts

    def _get_materialization_request_rows(
        self, *, req_pool_indices: torch.Tensor, seq_lens: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        req_pool_indices = req_pool_indices.to(torch.long).contiguous()
        request_count = req_pool_indices.numel()
        if request_count > self._decode_batch_capacity:
            raise RuntimeError(
                "Materialize batch exceeds the initialized copy-index capacity: "
                f"batch_size={request_count}, capacity={self._decode_batch_capacity}."
            )
        valid_reqs = (req_pool_indices >= 0) & (req_pool_indices < self.size)
        if seq_lens is not None:
            valid_reqs = valid_reqs & (seq_lens[:request_count] > 0)
        # Invalid rows use the slot-map sentinel; KV copies to row 0 are masked.
        slot_map_rows = torch.where(
            valid_reqs,
            req_pool_indices,
            self._slot_map_sentinel_req_indices[:request_count],
        )
        cache_rows = torch.where(
            valid_reqs, req_pool_indices, self._zero_req_indices[:request_count]
        )
        return slot_map_rows, cache_rows, valid_reqs

    def _prepare_materialization_inputs(
        self,
        *,
        layer_idx: int,
        req_pool_indices: torch.Tensor,
        seq_lens: Optional[torch.Tensor],
        topk_indices: torch.Tensor,
        selected_kv_buffer: torch.Tensor,
    ) -> _KVMaterializationInputs:
        slot_map_rows, cache_rows, valid_reqs = self._get_materialization_request_rows(
            req_pool_indices=req_pool_indices, seq_lens=seq_lens
        )
        topk_indices = normalize_batch_topk_indices(topk_indices)
        batch_size, topk_len = topk_indices.shape
        if batch_size != req_pool_indices.numel():
            raise RuntimeError(
                "Top-k and request batch sizes differ: "
                f"topk_batch={batch_size}, request_batch={req_pool_indices.numel()}."
            )
        if topk_len > self.sparse_context_len:
            raise RuntimeError(
                "DSA top-k length exceeds sparse attention window: "
                f"topk_len={topk_len}, sparse_context_len={self.sparse_context_len}."
            )
        if self.enable_lru and topk_len != self.sparse_context_len:
            raise RuntimeError("The fused timestamp-LRU kernel requires topk_len=2048.")
        if (
            selected_kv_buffer.dim() != 4
            or selected_kv_buffer.shape[0] != batch_size
            or selected_kv_buffer.shape[1] != self.sparse_context_len
        ):
            raise RuntimeError(
                "Current KV buffer must have shape "
                "[batch, sparse_context_len, head_num, head_dim], got "
                f"{tuple(selected_kv_buffer.shape)} with batch={batch_size} and "
                f"sparse_context_len={self.sparse_context_len}."
            )
        valid_mask = (
            (topk_indices >= 0)
            & (topk_indices < self.max_context_len)
            & valid_reqs.unsqueeze(1)
        )
        lookup_req_indices = slot_map_rows.to(dtype=torch.int32).contiguous()
        lookup_topk_indices = topk_indices.to(dtype=torch.int32).contiguous()
        hit_position_mask = None
        if self.enable_lru:
            token_on_device, device_token_pos, hit_position_mask = slot_map_lookup(
                self.device_slot_map[layer_idx],
                lookup_req_indices,
                lookup_topk_indices,
                pos_mask_size=self.device_cache_capacity,
            )
        else:
            token_on_device, device_token_pos = slot_map_lookup(
                self.device_slot_map[layer_idx], lookup_req_indices, lookup_topk_indices
            )
        hit_mask = token_on_device.to(torch.bool) & valid_mask
        return _KVMaterializationInputs(
            slot_map_rows=slot_map_rows,
            cache_rows=cache_rows,
            lookup_req_indices=lookup_req_indices,
            topk_indices=lookup_topk_indices,
            request_offsets=cache_rows.unsqueeze(1) * self.device_cache_capacity,
            copy_indices=self._selected_kv_copy_indices[: batch_size * topk_len],
            valid_mask=valid_mask,
            device_token_pos=device_token_pos,
            hit_position_mask=hit_position_mask,
            hit_mask=hit_mask,
            miss_mask=(~hit_mask) & valid_mask,
        )

    def _copy_selected_kv(
        self,
        *,
        layer_idx: int,
        inputs: _KVMaterializationInputs,
        selected_kv_buffer: torch.Tensor,
        stream: torch.npu.Stream,
        host_kv_ready_event: Optional[torch.npu.Event],
    ) -> None:
        with torch.npu.stream(stream):
            hit_src, hit_dst, hit_mask = _build_hit_src_dst_index(
                inputs.hit_mask,
                inputs.device_token_pos,
                inputs.request_offsets,
                inputs.copy_indices,
            )
            miss_src, miss_dst, miss_mask = _build_miss_src_dst_index(
                inputs.miss_mask,
                inputs.topk_indices,
                inputs.cache_rows,
                self.max_context_len,
                inputs.copy_indices,
            )

        # Disjoint hit/miss masks allow both 24-AIV copies to run concurrently.
        self._materialize_d2d_hit_stream.wait_stream(stream)
        with torch.npu.stream(self._materialize_d2d_hit_stream):
            unidex_copy_inplace(
                self.device_kv_buffer[layer_idx],
                selected_kv_buffer,
                hit_src,
                hit_dst,
                hit_mask,
                2,
                2,
                block_dim=24,
            )
            _record_stream_event(
                self._materialize_d2d_hit_stream, self._materialize_hit_done
            )

        self._materialize_h2d_miss_stream.wait_stream(stream)
        with torch.npu.stream(self._materialize_h2d_miss_stream):
            # Only host misses depend on the asynchronous decode host write.
            if host_kv_ready_event is not None:
                _wait_stream_event(
                    self._materialize_h2d_miss_stream, host_kv_ready_event
                )
            unidex_copy_inplace(
                self.host_kv_buffer[layer_idx],
                selected_kv_buffer,
                miss_src,
                miss_dst,
                miss_mask,
                2,
                2,
                block_dim=24,
                src_ptr=self.dev_ptr_list[layer_idx],
            )
            _record_stream_event(
                self._materialize_h2d_miss_stream, self._materialize_miss_done
            )

    def _prepare_refill_plan(
        self, *, layer_idx: int, inputs: _KVMaterializationInputs
    ) -> KVRefillPlan:
        with torch.npu.stream(self._materialize_metadata_update_stream):
            _wait_stream_event(
                self._materialize_metadata_update_stream, self._materialize_hit_done
            )
            _wait_stream_event(
                self._materialize_metadata_update_stream, self._materialize_miss_done
            )
            if self.enable_lru:
                victim_slots, miss_counts = self._lru_metadata_update(
                    inputs.lookup_req_indices,
                    inputs.topk_indices,
                    inputs.device_token_pos,
                    inputs.hit_position_mask,
                    self.device_lru_slots[layer_idx],
                    self.device_lru_slot_stamps[layer_idx],
                    max_context_len=self.max_context_len,
                    probation_age=self._probation_age,
                )
                _record_stream_event(
                    self._materialize_metadata_update_stream,
                    self._materialize_victim_slot_select_done,
                )
                self._lru_metadata_write(
                    self.device_slot_map[layer_idx],
                    inputs.lookup_req_indices,
                    inputs.topk_indices,
                    victim_slots,
                    miss_counts,
                    self.device_slot_tokens[layer_idx],
                    max_context_len=self.max_context_len,
                )
                refill_mask = inputs.miss_mask
            else:
                victim_slots = self._compact_sparse_indices.view(1, -1).expand(
                    *inputs.topk_indices.shape
                )
                _record_stream_event(
                    self._materialize_metadata_update_stream,
                    self._materialize_victim_slot_select_done,
                )
                # These caller-stream temporaries feed torch ops on this stream.
                inputs.slot_map_rows.record_stream(
                    self._materialize_metadata_update_stream
                )
                inputs.topk_indices.record_stream(
                    self._materialize_metadata_update_stream
                )
                self._replace_window_metadata(
                    layer_idx,
                    inputs.slot_map_rows,
                    inputs.topk_indices,
                    inputs.valid_mask,
                    victim_slots,
                    inputs.copy_indices,
                )
                refill_mask = inputs.valid_mask
            _record_stream_event(
                self._materialize_metadata_update_stream,
                self._materialize_metadata_update_done,
            )
        return KVRefillPlan(
            victim_slots=victim_slots,
            src_indices=inputs.copy_indices,
            request_offsets=inputs.request_offsets,
            valid_mask=refill_mask.reshape(-1).contiguous(),
        )

    def _replace_window_metadata(
        self,
        layer_idx: int,
        req_indices: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_mask: torch.Tensor,
        cache_slots: torch.Tensor,
        copy_indices: torch.Tensor,
    ) -> None:
        """Replace the slot map with the current dynamic top-k window."""
        self.device_slot_map[layer_idx].fill_(-1)
        token_indices = torch.where(
            valid_mask, topk_indices.to(torch.long), self.max_context_len
        )
        slot_values = torch.where(valid_mask, cache_slots, -1)
        flat_indices = (
            req_indices.unsqueeze(1) * self._slot_map_width + token_indices
        ).reshape(-1)
        unidex_copy_inplace(
            slot_values.reshape(-1, 1).contiguous(),
            self.device_slot_map[layer_idx].view(-1, 1),
            copy_indices,
            flat_indices,
            valid_mask.reshape(-1),
            1,
            1,
            block_dim=48,
        )

    def refill_selected_kv(
        self,
        *,
        layer: RadixAttention,
        selected_kv_buffer: torch.Tensor,
        plan: KVRefillPlan,
        stream: torch.npu.Stream,
    ) -> None:
        """Refill slots and order all cache metadata before subsequent attention."""
        layer_idx = layer.layer_id - self.start_layer
        stream = stream if stream is not None else torch.npu.current_stream()

        with torch.npu.stream(stream):
            _wait_stream_event(stream, self._materialize_victim_slot_select_done)
            refill_dst_index = (
                (plan.request_offsets + plan.victim_slots.to(torch.long))
                .reshape(-1)
                .contiguous()
            )
            unidex_copy_inplace(
                selected_kv_buffer,
                self.device_kv_buffer[layer_idx],
                plan.src_indices,
                refill_dst_index,
                plan.valid_mask,
                2,
                2,
                block_dim=48,
            )
            # Metadata writes can overlap refill, but must precede attention.
            _wait_stream_event(stream, self._materialize_metadata_update_done)


_global_sparse_kv_manager: Optional[SparseKVCacheManager] = None


def register_sparse_kv_manager(manager: SparseKVCacheManager) -> None:
    global _global_sparse_kv_manager
    _global_sparse_kv_manager = manager


def get_sparse_kv_manager() -> Optional[SparseKVCacheManager]:
    return _global_sparse_kv_manager


def _build_hit_src_dst_index(
    token_on_device: torch.Tensor,
    device_token_pos: torch.Tensor,
    request_cache_offsets: torch.Tensor,
    flat_dst_index: torch.Tensor,
):
    """
    token_on_device: [bs, topk], bool
    device_token_pos: [bs, topk], int64 or int32
    request_cache_offsets: [bs, 1], int64
    flat_dst_index: [bs * topk], int64, initialized by the manager

    Return:
        src_index_full: [bs * topk], int64
        dst_index_full: [bs * topk], int64
        valid_mask: [bs * topk], bool

    Flattening rule:
        src row = request_cache_offsets[batch_id] + device_token_pos
        dst row = flat_dst_index[batch_id, topk_pos]
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
    if request_cache_offsets.dim() != 2 or request_cache_offsets.shape[1] != 1:
        raise RuntimeError(
            "request_cache_offsets must have shape [batch, 1], got "
            f"{tuple(request_cache_offsets.shape)}"
        )

    bs, topk = token_on_device.shape
    if request_cache_offsets.shape[0] != bs:
        raise RuntimeError(
            "request_cache_offsets batch mismatch: "
            f"{request_cache_offsets.shape[0]} vs batch {bs}"
        )
    if flat_dst_index.dim() != 1 or flat_dst_index.numel() != bs * topk:
        raise RuntimeError(
            "flat_dst_index must contain batch * topk entries, got "
            f"shape={tuple(flat_dst_index.shape)}, expected={bs * topk}"
        )

    valid_mask = token_on_device.reshape(-1).contiguous()
    src_index_2d = request_cache_offsets + device_token_pos.to(torch.int64)
    flat_src_index_all = src_index_2d.reshape(-1).contiguous()

    return flat_src_index_all, flat_dst_index, valid_mask


def _build_miss_src_dst_index(
    token_from_host: torch.Tensor,
    topk_indices: torch.Tensor,
    current_req_indices: torch.Tensor,
    max_context_len: int,
    flat_dst_index: torch.Tensor,
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
    if flat_dst_index.dim() != 1 or flat_dst_index.numel() != bs * topk:
        raise RuntimeError(
            "flat_dst_index must contain batch * topk entries, got "
            f"shape={tuple(flat_dst_index.shape)}, expected={bs * topk}"
        )

    # materialize_selected_kv has already masked invalid requests and token
    # positions before constructing token_from_host.
    valid_mask = token_from_host.reshape(-1).contiguous()

    req_offsets = current_req_indices.to(torch.int64).unsqueeze(1) * max_context_len
    src_index_2d = req_offsets + topk_indices.to(torch.int64)
    flat_src_index_all = src_index_2d.reshape(-1).contiguous()

    return flat_src_index_all, flat_dst_index, valid_mask
