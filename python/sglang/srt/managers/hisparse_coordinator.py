# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from typing import List, NamedTuple, Union

import torch

from sglang.jit_kernel.hisparse import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    MLATokenToKVPoolHost,
)
from sglang.srt.utils import get_device_module, is_hip, is_npu

device_module = get_device_module()

_is_hip = is_hip()
_is_npu = is_npu()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


class HiSparseDecodeAllocationRequirements(NamedTuple):
    logical_need: int
    device_need: int


class _DeviceBufferGrowthPlanEntry(NamedTuple):
    first_index: int
    req_pool_idx: int
    old_cap: int
    new_cap: int
    grow_size: int
    net_extra: int


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        real_host_token_count = self.host_token_len(req.extend_range.end)
        if real_host_token_count <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(
                req, real_host_token_count=real_host_token_count
            )
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def next_decode_allocation_requirements(
        self, reqs: List[Req]
    ) -> HiSparseDecodeAllocationRequirements:
        allocator = self.token_to_kv_pool_allocator
        logical_page_size = allocator.page_size
        assert not _is_npu
        assert logical_page_size > 1
        assert allocator.supports_page_aligned_alloc

        logical_need = 0
        device_need = 0
        for req in reqs:
            assert req.kv is not None
            assert 0 <= req.kv_committed_len <= req.kv.kv_allocated_len
            if req.kv_committed_len % logical_page_size != 0:
                continue

            assert req.kv_committed_len == req.kv.kv_allocated_len
            logical_need += logical_page_size
            if self.is_dsv4_hisparse:
                assert logical_page_size % self.compress_ratio == 0
                device_need += logical_page_size // self.compress_ratio
                continue

            growth = self._get_device_buffer_growth(
                first_index=-1,
                req_pool_idx=req.req_pool_idx,
                seq_len=req.kv_committed_len + 1,
            )
            device_need += logical_page_size
            if growth is not None:
                device_need += growth.net_extra

        return HiSparseDecodeAllocationRequirements(
            logical_need=logical_need,
            device_need=device_need,
        )

    def _preload_to_device_buffer(
        self, req: Req, *, real_host_token_count: int
    ) -> None:
        """Preload all tokens from host pool into the device buffer."""
        assert real_host_token_count >= 0
        assert real_host_token_count <= int(
            self.req_to_host_pool_allocated_len[req.req_pool_idx]
        )
        assert real_host_token_count <= int(
            self.req_device_buffer_size[req.req_pool_idx]
        )
        assert real_host_token_count <= self.req_to_host_pool.shape[1]
        assert real_host_token_count <= self.req_to_device_buffer.shape[1]

        host_indices = self.req_to_host_pool[req.req_pool_idx, :real_host_token_count]
        device_locs = self.req_to_device_buffer[
            req.req_pool_idx, :real_host_token_count
        ]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        real_len = req.extend_range.end
        allocated_len = req.kv.kv_allocated_len
        row_width = self.req_to_token_pool.req_to_token.shape[1]
        assert 0 <= real_len <= allocated_len <= row_width

        if self.is_dsv4_hisparse:
            alloc_size = self.padded_buffer_size
        else:
            page_size = self.token_to_kv_pool_allocator.hisparse_device_page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size
        assert 0 <= alloc_size <= self.req_to_device_buffer.shape[1]

        real_full_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :real_len
        ].to(dtype=torch.int64, copy=True)
        allocated_full_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ].to(dtype=torch.int64, copy=True)
        ordered_real_mapping_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                real_full_indices
            )
        )
        allocated_mapping_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                allocated_full_indices
            )
        )

        assert int(self.req_device_buffer_size[req.req_pool_idx]) == 0
        buffer_owner_row = self.req_to_device_buffer[req.req_pool_idx]
        torch._assert_async(
            torch.all(buffer_owner_row == 0),
            "HiSparse device buffer must be unowned before admission",
        )

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            ordered_real_mapping_indices=ordered_real_mapping_indices,
            allocated_mapping_indices=allocated_mapping_indices,
            need_size=alloc_size,
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(real_mapping_len=%d, allocated_mapping_len=%d, alloc_size=%d)",
                req.rid,
                len(ordered_real_mapping_indices),
                len(allocated_mapping_indices),
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )
        assert int(self.req_device_buffer_size[req.req_pool_idx]) == alloc_size
        torch._assert_async(
            torch.all(self.req_to_device_buffer[req.req_pool_idx, :alloc_size] > 0),
            "HiSparse admission must publish positive device coordinates",
        )
        torch._assert_async(
            torch.all(self.req_to_device_buffer[req.req_pool_idx, alloc_size:] == 0),
            "HiSparse admission must not publish owners beyond its capacity",
        )

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def map_latest_cache_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        allocator = self.token_to_kv_pool_allocator
        if allocator.page_size > 1 and allocator.supports_page_aligned_alloc:
            self._map_page_aligned_latest_cache_loc_to_buffer(
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                req_pool_indices=req_pool_indices,
                seq_lens_cpu=seq_lens_cpu,
                req_pool_indices_cpu=req_pool_indices_cpu,
            )
            return

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest cache slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_cache_locs = self.token_to_kv_pool_allocator.translate_latest_cache_locs_to_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_cache_locs
                )
                torch._assert_async(
                    torch.all(previous_locs >= 0),
                    "HiSparse decode mapping must not contain negative coordinates",
                )
                stale_mask = (previous_locs > 0) & (
                    previous_locs != reserved_buffer_loc
                )
                if self.token_to_kv_pool_allocator.hisparse_device_page_size == 1:
                    stale_mapping_indices = compressed_cache_locs[stale_mask]
                    stale_page_ids = (
                        self.token_to_kv_pool_allocator.collect_owned_hisparse_page_ids(
                            mapping_indices=stale_mapping_indices
                        )
                    )
                    self.token_to_kv_pool_allocator.clear_hisparse_mapping(
                        mapping_indices=stale_mapping_indices
                    )
                    self.token_to_kv_pool_allocator.release_owned_hisparse_pages(
                        owned_page_ids=stale_page_ids
                    )
                else:
                    torch._assert_async(
                        torch.all(~stale_mask),
                        "HiSparse page-sized temporary owners require the op29d "
                        "whole-page retirement lifecycle",
                    )

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_cache_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_cache_locs = (
            self.token_to_kv_pool_allocator.translate_latest_cache_locs_to_compressed(
                active_out_cache_loc
            )
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            compressed_cache_locs
        ] = reserved_buffer_loc

    def _map_page_aligned_latest_cache_loc_to_buffer(
        self,
        *,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        allocator = self.token_to_kv_pool_allocator
        logical_page_size = allocator.page_size
        device_page_size = allocator.hisparse_device_page_size
        assert not _is_npu
        assert logical_page_size > 1
        assert allocator.supports_page_aligned_alloc
        assert device_page_size > 1
        assert seq_lens.ndim == 1
        assert out_cache_loc.shape == seq_lens.shape
        assert req_pool_indices.shape == seq_lens.shape
        assert seq_lens_cpu.shape == seq_lens.shape
        assert req_pool_indices_cpu.shape == seq_lens.shape

        if self.is_dsv4_hisparse:
            assert logical_page_size % self.compress_ratio == 0
            assert device_page_size == logical_page_size // self.compress_ratio
            mapped_batch_indices_cpu = [
                batch_index
                for batch_index in range(seq_lens_cpu.numel())
                if int(seq_lens_cpu[batch_index]) % self.compress_ratio == 0
            ]
        else:
            assert device_page_size == logical_page_size
            mapped_batch_indices_cpu = list(range(seq_lens_cpu.numel()))

        if not mapped_batch_indices_cpu:
            return

        mapped_batch_indices = torch.tensor(
            mapped_batch_indices_cpu,
            dtype=torch.int64,
            device=seq_lens.device,
        )
        mapped_seq_lens = seq_lens[mapped_batch_indices]
        mapped_out_cache_loc = out_cache_loc[mapped_batch_indices]
        mapped_req_pool_indices = req_pool_indices[mapped_batch_indices]
        semantic_positions = mapped_seq_lens - 1
        if self.is_dsv4_hisparse:
            semantic_positions = mapped_seq_lens // self.compress_ratio - 1

        first_indices_cpu = [
            mapped_index
            for mapped_index, batch_index in enumerate(mapped_batch_indices_cpu)
            if (
                (
                    int(seq_lens_cpu[batch_index]) // self.compress_ratio - 1
                    if self.is_dsv4_hisparse
                    else int(seq_lens_cpu[batch_index]) - 1
                )
                % device_page_size
                == 0
            )
        ]
        if not first_indices_cpu:
            self._validate_current_page_aligned_mappings(
                semantic_positions=semantic_positions,
                mapped_out_cache_loc=mapped_out_cache_loc,
                mapped_req_pool_indices=mapped_req_pool_indices,
            )
            return

        first_indices = torch.tensor(
            first_indices_cpu,
            dtype=torch.int64,
            device=seq_lens.device,
        )
        first_semantic_positions = semantic_positions[first_indices]
        first_mapping_indices = allocator.translate_latest_cache_locs_to_compressed(
            mapped_out_cache_loc[first_indices]
        ).to(dtype=torch.int64)
        first_req_pool_indices = mapped_req_pool_indices[first_indices]
        offsets = torch.arange(
            device_page_size,
            dtype=torch.int64,
            device=seq_lens.device,
        )
        mapping_index_blocks = first_mapping_indices[:, None] + offsets
        semantic_position_blocks = first_semantic_positions[:, None] + offsets
        buffer_positions = semantic_position_blocks.clamp(max=self.device_buffer_size)
        mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
        full_semantic_position_blocks = semantic_position_blocks
        if self.is_dsv4_hisparse:
            full_semantic_position_blocks = (
                semantic_position_blocks * self.compress_ratio + self.compress_ratio - 1
            )

        torch._assert_async(
            torch.all(first_mapping_indices % device_page_size == 0),
            "HiSparse first-remap mapping blocks must be page aligned",
        )
        torch._assert_async(
            torch.all(mapping_index_blocks >= 0),
            "HiSparse first-remap mapping indices must be non-negative",
        )
        torch._assert_async(
            torch.all(mapping_index_blocks < mapping.numel() - 1),
            "HiSparse first-remap mapping blocks must stay within the mapping domain",
        )
        torch._assert_async(
            torch.all(semantic_position_blocks >= 0),
            "HiSparse first-remap semantic positions must be non-negative",
        )
        torch._assert_async(
            torch.all(
                full_semantic_position_blocks
                < self.req_to_token_pool.req_to_token.shape[1]
            ),
            "HiSparse first-remap semantic positions must stay within request rows",
        )
        full_logical_blocks = self.req_to_token_pool.req_to_token[
            first_req_pool_indices[:, None], full_semantic_position_blocks
        ]
        expected_mapping_index_blocks = (
            allocator.translate_latest_cache_locs_to_compressed(full_logical_blocks).to(
                dtype=torch.int64
            )
        )
        torch._assert_async(
            torch.all(expected_mapping_index_blocks == mapping_index_blocks),
            "HiSparse first-remap key blocks must match allocated request rows",
        )

        actual_owner_rows = self.req_to_device_buffer[first_req_pool_indices]
        predicted_owner_rows = actual_owner_rows.clone()
        actual_destination_vectors = torch.gather(
            actual_owner_rows,
            dim=1,
            index=buffer_positions,
        )
        mapped_coordinates = mapping[mapping_index_blocks]
        torch._assert_async(
            torch.all(mapped_coordinates >= 0),
            "HiSparse decode mapping must not contain negative coordinates",
        )
        current_coordinates = mapped_coordinates[:, 0]
        torch._assert_async(
            torch.all(current_coordinates > 0),
            "HiSparse first-remap owners must contain positive coordinates",
        )
        temporary_page_ids = current_coordinates // device_page_size
        expected_temporary_blocks = (
            temporary_page_ids[:, None] * device_page_size + offsets
        )

        growth_plans = self._plan_device_buffer_growth(
            first_indices_cpu=first_indices_cpu,
            mapped_batch_indices_cpu=mapped_batch_indices_cpu,
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices_cpu=req_pool_indices_cpu,
        )
        growth_mask_values = [False] * len(first_indices_cpu)
        for growth in growth_plans:
            growth_mask_values[growth.first_index] = True
            mapped_index = first_indices_cpu[growth.first_index]
            batch_index = mapped_batch_indices_cpu[mapped_index]
            assert growth.req_pool_idx == int(req_pool_indices_cpu[batch_index])
            assert 0 <= growth.old_cap < growth.new_cap <= self.padded_buffer_size
            torch._assert_async(
                torch.all(actual_owner_rows[growth.first_index, : growth.old_cap] > 0),
                "HiSparse existing buffer ownership must be positive before growth",
            )
            torch._assert_async(
                torch.all(
                    actual_owner_rows[
                        growth.first_index, growth.old_cap : growth.new_cap
                    ]
                    == 0
                ),
                "HiSparse buffer growth must target an unowned range",
            )
        growth_mask = torch.tensor(
            growth_mask_values,
            dtype=torch.bool,
            device=seq_lens.device,
        )

        replay_mask = (~growth_mask) & torch.all(
            mapped_coordinates == actual_destination_vectors,
            dim=1,
        )
        transaction_mask = ~replay_mask
        valid_transaction_blocks = torch.all(
            mapped_coordinates == expected_temporary_blocks,
            dim=1,
        )
        torch._assert_async(
            torch.all(replay_mask | valid_transaction_blocks),
            "HiSparse first-remap found a partial or shared temporary owner",
        )
        torch._assert_async(
            torch.all(actual_destination_vectors[replay_mask] > 0),
            "HiSparse first-remap replay destinations must remain owned",
        )
        torch._assert_async(
            torch.all(actual_destination_vectors[~growth_mask] > 0),
            "HiSparse release-only destination vectors must remain owned",
        )

        transfer_page_ids = temporary_page_ids[growth_mask]
        sorted_transfer_page_ids = torch.sort(transfer_page_ids).values
        torch._assert_async(
            torch.all(sorted_transfer_page_ids[1:] != sorted_transfer_page_ids[:-1]),
            "HiSparse buffer growth requires one unique temporary page per request",
        )
        if transfer_page_ids.numel() > 0:
            transfer_page_blocks = allocator.materialize_owned_hisparse_page_blocks(
                owned_page_ids=transfer_page_ids
            ).reshape(-1, device_page_size)
        else:
            transfer_page_blocks = torch.empty(
                (0, device_page_size),
                dtype=torch.int64,
                device=seq_lens.device,
            )
        torch._assert_async(
            torch.all(mapped_coordinates[growth_mask] == transfer_page_blocks),
            "HiSparse transfer owners must match their complete temporary pages",
        )

        release_mapping_indices = mapping_index_blocks[
            transaction_mask & ~growth_mask
        ].reshape(-1)
        if release_mapping_indices.numel() > 0:
            release_page_ids = allocator.collect_owned_hisparse_page_ids(
                mapping_indices=release_mapping_indices
            )
        else:
            release_page_ids = torch.empty(
                (0,),
                dtype=torch.int64,
                device=seq_lens.device,
            )
        torch._assert_async(
            torch.all(~torch.isin(transfer_page_ids, release_page_ids)),
            "HiSparse transfer and release owner sets must be disjoint",
        )
        transaction_page_ids = temporary_page_ids[transaction_mask]
        buffer_page_ids = torch.div(
            actual_owner_rows[transaction_mask],
            device_page_size,
            rounding_mode="floor",
        )
        torch._assert_async(
            torch.all(~torch.isin(buffer_page_ids, transaction_page_ids)),
            "HiSparse temporary owners must not alias existing device buffers",
        )
        for growth in growth_plans:
            torch._assert_async(
                torch.all(
                    actual_owner_rows[growth.first_index, : growth.old_cap]
                    // device_page_size
                    != temporary_page_ids[growth.first_index]
                ),
                "HiSparse temporary transfer owners must not alias existing buffers",
            )
        release_destination_page_ids = (
            actual_destination_vectors[transaction_mask & ~growth_mask]
            // device_page_size
        )
        torch._assert_async(
            torch.all(~torch.isin(release_destination_page_ids, release_page_ids)),
            "HiSparse release-only destinations must not alias retired owners",
        )

        total_net_extra = sum(growth.net_extra for growth in growth_plans)
        if total_net_extra > 0:
            extra_indices = allocator.hisparse_attn_allocator.alloc(total_net_extra)
            if extra_indices is None:
                logger.error(
                    "HiSparse device buffer net allocation failed (total_net_extra=%d)",
                    total_net_extra,
                )
                raise RuntimeError(
                    "HiSparse device buffer net allocation failed "
                    f"(total_net_extra={total_net_extra})"
                )
            self._validate_page_blocks(
                indices=extra_indices,
                expected_size=total_net_extra,
                page_size=device_page_size,
            )
        else:
            extra_indices = torch.empty(
                (0,),
                dtype=torch.int64,
                device=seq_lens.device,
            )

        transfer_offset = 0
        extra_offset = 0
        for growth in growth_plans:
            transfer_block = transfer_page_blocks[transfer_offset]
            transfer_offset += 1
            predicted_owner_rows[
                growth.first_index,
                growth.old_cap : growth.old_cap + device_page_size,
            ] = transfer_block
            if growth.net_extra > 0:
                extra_end = extra_offset + growth.net_extra
                predicted_owner_rows[
                    growth.first_index,
                    growth.old_cap + device_page_size : growth.new_cap,
                ] = extra_indices[extra_offset:extra_end]
                extra_offset = extra_end
        assert transfer_offset == len(growth_plans)
        assert extra_offset == total_net_extra

        predicted_destination_vectors = torch.gather(
            predicted_owner_rows,
            dim=1,
            index=buffer_positions,
        )
        torch._assert_async(
            torch.all(predicted_destination_vectors > 0),
            "HiSparse first-remap destination vectors must remain fully owned",
        )
        transaction_mapping_indices = mapping_index_blocks[transaction_mask].reshape(-1)
        if transaction_mapping_indices.numel() > 0:
            allocator.clear_hisparse_mapping(
                mapping_indices=transaction_mapping_indices
            )
            mapping_page_ids = torch.div(
                mapping,
                device_page_size,
                rounding_mode="floor",
            )
            torch._assert_async(
                torch.all(~torch.isin(mapping_page_ids, transaction_page_ids)),
                "HiSparse temporary pages must have no mapping aliases before conversion",
            )

        for growth in growth_plans:
            committed_chunk = predicted_owner_rows[
                growth.first_index, growth.old_cap : growth.new_cap
            ]
            self.req_to_device_buffer[
                growth.req_pool_idx, growth.old_cap : growth.new_cap
            ] = committed_chunk
            self.req_device_buffer_token_locs[
                :, growth.req_pool_idx, growth.old_cap : growth.new_cap
            ] = committed_chunk.to(torch.int32)
            self.req_device_buffer_size[growth.req_pool_idx] = growth.new_cap

        committed_owner_rows = self.req_to_device_buffer[first_req_pool_indices]
        committed_destination_vectors = torch.gather(
            committed_owner_rows,
            dim=1,
            index=buffer_positions,
        )
        torch._assert_async(
            torch.all(committed_destination_vectors == predicted_destination_vectors),
            "HiSparse committed destination vectors must match their prediction",
        )
        if release_page_ids.numel() > 0:
            allocator.release_owned_hisparse_pages(owned_page_ids=release_page_ids)
        mapping[mapping_index_blocks[transaction_mask]] = committed_destination_vectors[
            transaction_mask
        ]

        self._validate_current_page_aligned_mappings(
            semantic_positions=semantic_positions,
            mapped_out_cache_loc=mapped_out_cache_loc,
            mapped_req_pool_indices=mapped_req_pool_indices,
        )

    def _validate_current_page_aligned_mappings(
        self,
        *,
        semantic_positions: torch.Tensor,
        mapped_out_cache_loc: torch.Tensor,
        mapped_req_pool_indices: torch.Tensor,
    ) -> None:
        current_buffer_positions = semantic_positions.clamp(max=self.device_buffer_size)
        current_destination_coordinates = self.req_to_device_buffer[
            mapped_req_pool_indices, current_buffer_positions
        ]
        current_mapping_indices = (
            self.token_to_kv_pool_allocator.translate_latest_cache_locs_to_compressed(
                mapped_out_cache_loc
            ).to(dtype=torch.int64)
        )
        torch._assert_async(
            torch.all(current_destination_coordinates > 0),
            "HiSparse current decode destinations must remain owned",
        )
        torch._assert_async(
            torch.all(
                self.mem_pool_device.full_to_hisparse_device_index_mapping[
                    current_mapping_indices
                ]
                == current_destination_coordinates
            ),
            "HiSparse current decode mappings must target their reserved destinations",
        )
        self.req_device_buffer_token_locs[
            :, mapped_req_pool_indices, self.device_buffer_size
        ] = current_destination_coordinates.to(torch.int32)

    def _plan_device_buffer_growth(
        self,
        *,
        first_indices_cpu: List[int],
        mapped_batch_indices_cpu: List[int],
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> List[_DeviceBufferGrowthPlanEntry]:
        if self.is_dsv4_hisparse:
            return []

        growth_plans: List[_DeviceBufferGrowthPlanEntry] = []
        for first_index, mapped_index in enumerate(first_indices_cpu):
            batch_index = mapped_batch_indices_cpu[mapped_index]
            growth = self._get_device_buffer_growth(
                first_index=first_index,
                req_pool_idx=int(req_pool_indices_cpu[batch_index]),
                seq_len=int(seq_lens_cpu[batch_index]),
            )
            if growth is not None:
                growth_plans.append(growth)
        return growth_plans

    def _get_device_buffer_growth(
        self,
        *,
        first_index: int,
        req_pool_idx: int,
        seq_len: int,
    ) -> _DeviceBufferGrowthPlanEntry | None:
        current_cap = int(self.req_device_buffer_size[req_pool_idx])
        if seq_len > self.device_buffer_size or seq_len <= current_cap:
            return None

        page_size = self.mem_pool_device.page_size
        assert seq_len - 1 == current_cap
        new_cap = min(
            ((seq_len + page_size - 1) // page_size) * page_size,
            self.device_buffer_size,
        )
        if new_cap == self.device_buffer_size:
            new_cap = self.padded_buffer_size
        grow_size = new_cap - current_cap
        assert grow_size >= page_size
        assert grow_size % page_size == 0
        return _DeviceBufferGrowthPlanEntry(
            first_index=first_index,
            req_pool_idx=req_pool_idx,
            old_cap=current_cap,
            new_cap=new_cap,
            grow_size=grow_size,
            net_extra=grow_size - page_size,
        )

    def _validate_page_blocks(
        self,
        *,
        indices: torch.Tensor,
        expected_size: int,
        page_size: int,
    ) -> None:
        assert indices.ndim == 1
        assert indices.dtype == torch.int64
        assert indices.device == self.req_to_device_buffer.device
        assert indices.numel() == expected_size
        assert expected_size % page_size == 0
        page_rows = indices.reshape(-1, page_size)
        offsets = torch.arange(
            page_size,
            dtype=torch.int64,
            device=indices.device,
        )
        torch._assert_async(
            torch.all(page_rows[:, 0] % page_size == 0),
            "HiSparse extra owner pages must be aligned",
        )
        torch._assert_async(
            torch.all(page_rows == page_rows[:, :1] + offsets),
            "HiSparse extra owner pages must be complete and consecutive",
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        allocated_len = req.kv.kv_allocated_len
        assert 0 <= req.extend_range.end <= allocated_len
        assert allocated_len <= self.req_to_token_pool.req_to_token.shape[1]
        assert int(self.req_device_buffer_size[req.req_pool_idx]) == 0
        torch._assert_async(
            torch.all(self.req_to_device_buffer[req.req_pool_idx] == 0),
            "HiSparse staging abort must not observe a published device buffer",
        )

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        allocated_mapping_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(allocated_locs)
        )
        owned_page_ids = (
            self.token_to_kv_pool_allocator.collect_owned_hisparse_page_ids(
                mapping_indices=allocated_mapping_indices
            )
        )
        self.token_to_kv_pool_allocator.clear_hisparse_mapping(
            mapping_indices=allocated_mapping_indices
        )
        self.token_to_kv_pool_allocator.release_owned_hisparse_pages(
            owned_page_ids=owned_page_ids
        )

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self._skip_first_backup[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        allocated_len = req.kv.kv_allocated_len
        assert 0 <= allocated_len <= self.req_to_token_pool.req_to_token.shape[1]

        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        assert 0 <= current_cap <= self.req_to_device_buffer.shape[1]
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        allocated_mapping_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(allocated_locs)
        )
        owned_page_ids = (
            self.token_to_kv_pool_allocator.collect_owned_hisparse_page_ids(
                mapping_indices=allocated_mapping_indices,
                extra_owned_coordinates=self.req_to_device_buffer[
                    req.req_pool_idx, :current_cap
                ],
            )
        )

        self.token_to_kv_pool_allocator.clear_hisparse_mapping(
            mapping_indices=allocated_mapping_indices
        )

        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        assert int(self.req_device_buffer_size[req.req_pool_idx]) == 0
        torch._assert_async(
            torch.all(self.req_to_device_buffer[req.req_pool_idx] == 0),
            "HiSparse terminal cleanup must clear device buffer owners",
        )
        torch._assert_async(
            torch.all(self.req_device_buffer_tokens[:, req.req_pool_idx, :] == -1),
            "HiSparse terminal cleanup must clear device buffer tokens",
        )
        torch._assert_async(
            torch.all(self.req_device_buffer_token_locs[:, req.req_pool_idx, :] == -1),
            "HiSparse terminal cleanup must clear device buffer locations",
        )

        self.token_to_kv_pool_allocator.release_owned_hisparse_pages(
            owned_page_ids=owned_page_ids
        )

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices."""
        num_reqs = req_pool_indices.size(0)

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
        )
        return top_k_indices
