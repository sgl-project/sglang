# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from typing import List, Literal, NamedTuple, Optional, Union

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
from sglang.srt.utils import get_device_module, is_hip

device_module = get_device_module()

_is_hip = is_hip()

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
        host_allocator_type: str = "default",
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
        # unified-KV HiSparse (ROCm) keeps the compressed C4 hot/cold data in
        # the bf16 unified layout, so swap-in/backup use the generic linear MLA
        # path instead of the FP8 page-padded dsv4 path.
        self.is_unified_hisparse = False
        if self.is_dsv4_hisparse:
            from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
                HiSparseUnifiedC4DevicePool,
            )

            self.is_unified_hisparse = isinstance(
                self.token_to_kv_pool_allocator.hisparse_kvcache,
                HiSparseUnifiedC4DevicePool,
            )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            # Host cold-pool sizing (unified-KV / dsv4 path): the host pool is
            # bound to size_full/compress_ratio -- a mirror of the GPU full-token
            # c4 budget, not an independent expansion into host RAM.
            # host_to_device_ratio (== c4_shrink_factor, >= 1) sets how much of
            # that budget stays GPU-resident: ratio=1 keeps the full c4 pool
            # on-GPU (host == device, a 1:1 mirror); ratio>1 shrinks the GPU c4
            # pool (freeing memory for a larger overall token budget) while the
            # host mirror still tracks the full budget. This is orthogonal to the
            # per-request device_buffer_size hot window (the swap trigger), which
            # is never scaled by ratio.
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
                allocator_type=host_allocator_type,
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # Extra slots beyond device_buffer_size: 1 reserved slot for the newest
        # decode token, plus draft slots for speculative MTP tokens. When
        # speculative decoding is active, get_draft_device_slots allocates from
        # device_buffer_size+1 onwards, needing up to double_alloc extra slots.
        from sglang.srt.mem_cache.common import get_alloc_reserve_per_decode
        spec_extra = get_alloc_reserve_per_decode()
        self.padded_buffer_size = (
            self.device_buffer_size
            + max(self.mem_pool_device.page_size, 1 + spec_extra)
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
            self.padded_buffer_size, dtype=torch.int32, device=device
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
        self._pending_draft_extend_backup = None

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

        host_len = self.host_token_len(req.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
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

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32[: self.device_buffer_size]
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )
        self._skip_first_backup[req.req_pool_idx] = True

        # HiSparse x MTP fix: alloc_device_buffer zeroed the mapping for the
        # resident (staged) prefill compressed locs, since decode reads them via
        # req_to_device_buffer (swap-in). But the EAGLE TARGET_VERIFY forward is
        # not decode, so the indexer reads C4 through the mapping
        # (translate_loc_to_hisparse_device) -- which then resolves the whole
        # prefill prefix to sink row 0, corrupting verification and collapsing
        # accuracy. Re-point the mapping at the resident device-buffer rows so
        # the mapping read matches the swap-in read. Only valid while the prefix
        # fits the device buffer (alloc_device_buffer reorders/clamps beyond it);
        # longer sequences keep the prior behavior.
        if self.is_dsv4_hisparse and compressed_len <= self.device_buffer_size:
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_logical_indices
            ] = buffer_indices[:compressed_len].to(torch.int64)


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
                    hot_end = min(new_cap, self.device_buffer_size)
                    if current_cap < hot_end:
                        self.req_device_buffer_tokens[
                            :, req_idx, current_cap:hot_end
                        ] = self._device_buffer_arange_i32[current_cap:hot_end]
                    if hot_end < new_cap:
                        self.req_device_buffer_tokens[:, req_idx, hot_end:new_cap] = -1
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

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu=None,
    ) -> None:
        if req_pool_indices_cpu is None:
            req_pool_indices_cpu = req_pool_indices.cpu()

        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)
            self.req_device_buffer_tokens[
                :, req_pool_indices, self.device_buffer_size
            ] = (seq_lens.to(torch.int32) - 1)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
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
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

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

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
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


    def _backup_device_locs_to_host(
        self, host_locs: torch.Tensor, device_locs: torch.Tensor
    ) -> None:
        if host_locs.numel() == 0:
            return
        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        device_locs = device_locs.contiguous()
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
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        event = device_module.Event()
        event.record(self.decode_backup_stream)
        device_module.current_stream().wait_event(event)

    def finish_pending_draft_extend_backup(self) -> None:
        pending = self._pending_draft_extend_backup
        if pending is None:
            return
        self._pending_draft_extend_backup = None
        host_locs, device_locs, logical_locs_to_clear = pending
        self._backup_device_locs_to_host(host_locs, device_locs)
        if logical_locs_to_clear.numel() > 0:
            full_to_device_mapping = (
                self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
            )
            full_to_device_mapping[logical_locs_to_clear] = 0

    def clear_pending_draft_extend_backup(self) -> None:
        pending = self._pending_draft_extend_backup
        if pending is None:
            return
        self._pending_draft_extend_backup = None
        _, _, logical_locs_to_clear = pending
        if logical_locs_to_clear.numel() > 0:
            full_to_device_mapping = (
                self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
            )
            full_to_device_mapping[logical_locs_to_clear] = 0

    def supports_hisparse_draft_slots(self) -> bool:
        return True

    def _ensure_padded_buffer(self, req_pool_indices: torch.Tensor) -> None:
        """Grow under-sized buffers to padded_buffer_size so extra-page slots are valid."""
        req_indices_cpu = req_pool_indices.cpu().tolist()
        grow_reqs = []
        total_grow = 0
        for req_idx in req_indices_cpu:
            current_cap = int(self.req_device_buffer_size[req_idx])
            if current_cap >= self.padded_buffer_size:
                continue
            grow_reqs.append((req_idx, current_cap))
            total_grow += self.padded_buffer_size - current_cap

        if total_grow == 0:
            return

        all_new = self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
            total_grow
        )
        if all_new is None:
            raise RuntimeError(
                f"HiSparse: failed to grow buffers for draft slots (need {total_grow})"
            )

        offset = 0
        for req_idx, current_cap in grow_reqs:
            grow_size = self.padded_buffer_size - current_cap
            chunk = all_new[offset : offset + grow_size]
            offset += grow_size
            self.req_to_device_buffer[
                req_idx, current_cap : self.padded_buffer_size
            ] = chunk
            hot_end = min(self.padded_buffer_size, self.device_buffer_size)
            if current_cap < hot_end:
                self.req_device_buffer_tokens[:, req_idx, current_cap:hot_end] = (
                    self._device_buffer_arange_i32[current_cap:hot_end]
                )
            if hot_end < self.padded_buffer_size:
                self.req_device_buffer_tokens[
                    :, req_idx, hot_end : self.padded_buffer_size
                ] = -1
            self.req_device_buffer_token_locs[
                :, req_idx, current_cap : self.padded_buffer_size
            ] = chunk
            self.req_device_buffer_size[req_idx] = self.padded_buffer_size

    def get_draft_device_slots(
        self,
        req_pool_indices: torch.Tensor,
        num_tokens_per_req: int,
        start_positions_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Return device slots for a uniform draft allocation.

        Draft tokens that still fit in the hot buffer reuse their logical hot
        slots. Later draft tokens use the per-request extra-page area.
        """
        assert self.supports_hisparse_draft_slots()
        start = self.device_buffer_size + 1
        end = start + num_tokens_per_req
        if end > self.padded_buffer_size:
            raise ValueError(
                f"Requested {num_tokens_per_req} draft slots but extra page only "
                f"has {self.padded_buffer_size - self.device_buffer_size - 1} "
                f"available (padded_buffer_size={self.padded_buffer_size}, "
                f"device_buffer_size={self.device_buffer_size})."
            )
        self._ensure_padded_buffer(req_pool_indices)
        self.req_device_buffer_tokens[
            :, req_pool_indices, start:end
        ] = self._device_buffer_arange_i32[start:end]
        num_reqs = req_pool_indices.size(0)
        slots = self.req_to_device_buffer[req_pool_indices, start:end]
        return slots.reshape(num_reqs * num_tokens_per_req)

    def finalize_accepted_tokens(
        self,
        req_pool_indices: torch.Tensor,
        accepted_cache_locs: torch.Tensor,
        draft_cache_locs: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        num_correct_drafts_cpu: torch.Tensor,
        accepted_token_positions: torch.Tensor,
    ) -> None:
        """Commit accepted draft tokens into HiSparse device/host storage.

        For each request, the last accepted token is placed into the reserved
        slot (device_buffer_size position) of the hot buffer while earlier
        accepted tokens are backed up to host.
        """
        assert self.supports_hisparse_draft_slots()
        if accepted_cache_locs.numel() == 0:
            return

        full_to_device_mapping = (
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )

        # Clear device mapping for all draft cache locs
        full_to_device_mapping[draft_cache_locs] = 0

        total_accepted = accepted_cache_locs.numel()
        num_reqs = req_pool_indices.size(0)

        offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=num_correct_drafts.device),
            (num_correct_drafts + 1).cumsum(0)
        ])
        last_offsets = (offsets[1:] - 1).to(torch.int64)

        accepted_device_locs = self.req_to_device_buffer[
            req_pool_indices.repeat_interleave(num_correct_drafts + 1),
            (self.device_buffer_size + 1) + (
                torch.arange(total_accepted, device=accepted_cache_locs.device)
                - torch.repeat_interleave(offsets[:-1], num_correct_drafts + 1)
            ),
        ]

        needs_backup = torch.ones(total_accepted, dtype=torch.bool, device=accepted_cache_locs.device)
        needs_backup[last_offsets] = False
        backup_count = int(needs_backup.sum().item())

        if backup_count > 0:
            backup_device_locs = accepted_device_locs[needs_backup]
            host_pool_indices = req_pool_indices.repeat_interleave(num_correct_drafts + 1)[needs_backup]
            backup_positions = accepted_token_positions[needs_backup]

            host_locs_list = []
            for i in range(backup_count):
                ridx = int(host_pool_indices[i].item())
                pos = int(backup_positions[i].item()) // self.compress_ratio
                allocated = int(self.req_to_host_pool_allocated_len[ridx])
                if pos > allocated:
                    gap = pos - allocated
                    self.mem_pool_host.alloc_paged_token_slots(
                        self.req_to_host_pool,
                        self.req_to_host_pool_allocated_len,
                        ridx,
                        allocated,
                        gap,
                    )
                host_loc = self.mem_pool_host.alloc_paged_token_slots(
                    self.req_to_host_pool,
                    self.req_to_host_pool_allocated_len,
                    ridx,
                    pos,
                    1,
                )
                host_locs_list.append(host_loc)
            host_locs = torch.cat(host_locs_list)

        last_positions = accepted_token_positions[last_offsets]
        last_slots = accepted_device_locs[last_offsets]
        newest_slots = accepted_cache_locs[last_offsets]

        reserved_positions = self.device_buffer_size
        self.req_device_buffer_token_locs[:, req_pool_indices, reserved_positions] = (
            newest_slots.to(torch.int32)
        )
        self.req_device_buffer_tokens[:, req_pool_indices, reserved_positions] = (
            last_positions.to(torch.int32).unsqueeze(0)
        )
        for idx in req_pool_indices.tolist():
            self._skip_first_backup[idx] = True

        same_slot = last_slots == newest_slots
        if torch.any(~same_slot):
            self.mem_pool_device.transfer_values_on_device(
                dst_indices=newest_slots[~same_slot],
                src_indices=last_slots[~same_slot],
            )
        full_to_device_mapping[accepted_cache_locs[last_offsets]] = newest_slots

        if backup_count > 0:
            backup_positions_in_needs = (
                torch.cumsum(needs_backup.to(torch.int64), dim=0) - 1
            )
            last_needs_backup = needs_backup[last_offsets]
            post_backup_device_locs = backup_device_locs.clone()
            if torch.any(last_needs_backup):
                last_backup_offsets = backup_positions_in_needs[
                    last_offsets[last_needs_backup]
                ]
                post_backup_device_locs[last_backup_offsets] = newest_slots[
                    last_needs_backup
                ]

            logical_locs_to_clear_mask = needs_backup.clone()
            logical_locs_to_clear_mask[last_offsets] = False
            self._pending_draft_extend_backup = (
                host_locs,
                post_backup_device_locs,
                accepted_cache_locs[logical_locs_to_clear_mask],
            )

    def finalize_accepted_tokens_spec_v2(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_cache_locs: torch.Tensor,
        accept_index: torch.Tensor,
    ) -> None:
        """Commit accepted spec-v2 target-verify KV into HiSparse storage."""
        assert self.supports_hisparse_draft_slots()
        if verify_cache_locs.numel() == 0:
            return

        counts = (accept_index != -1).sum(dim=1).to(torch.int64)
        total_accepted = int(counts.sum().item())
        if total_accepted == 0:
            full_to_device_mapping = (
                self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
            )
            full_to_device_mapping[verify_cache_locs] = 0
            return

        flat_accept_index = accept_index.reshape(-1)
        accepted_offsets = flat_accept_index[flat_accept_index >= 0].to(torch.int64)
        if accepted_offsets.numel() != total_accepted:
            raise ValueError(
                "HiSparse spec-v2 accepted index mismatch: "
                f"expected {total_accepted} accepted slots, "
                f"got {accepted_offsets.numel()}."
            )

        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0)]
        )
        pos_in_segment = torch.arange(
            total_accepted, dtype=torch.int64, device=counts.device
        ) - torch.repeat_interleave(offsets[:-1], counts)
        accepted_token_positions = (
            torch.repeat_interleave(seq_lens.to(torch.int64), counts) + pos_in_segment
        )

        self.finalize_accepted_tokens(
            req_pool_indices=req_pool_indices,
            accepted_cache_locs=verify_cache_locs[accepted_offsets],
            draft_cache_locs=verify_cache_locs,
            num_correct_drafts=counts - 1,
            num_correct_drafts_cpu=(counts - 1).cpu(),
            accepted_token_positions=accepted_token_positions,
        )

    def get_front_topk_tokens(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        top_k_indices = self.req_to_device_buffer[req_pool_indices, : self.top_k].to(
            torch.int32
        )
        topk_col_indices = torch.arange(self.top_k, device=self.device).unsqueeze(0)
        mask = topk_col_indices >= seq_lens.unsqueeze(1)
        top_k_indices[mask] = -1
        return top_k_indices

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA/HIP kernel) instead.
        Used as a kernel oracle in test_hisparse_unit.py.

        Both the non-dsv4 and the dsv4 (DeepSeek V4 separate-KV C4) layouts are
        supported. The per-request loc-selection logic below is layout-agnostic:
        it indexes req_to_device_buffer / req_to_host_pool and delegates the
        host->device byte copy to mem_pool_host.load_to_device_per_layer. For the
        dsv4 path mem_pool_host is a DeepSeekV4PagedHostPool (which implements
        load_to_device_per_layer for the token-granular, page-padded C4 value/
        scale layout), and seq_lens / top_k_tokens are expressed in compressed C4
        space (matching swap_in_selected_pages, called with compressed_seq_lens).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
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

        prefill_len = req.extend_range.end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.req_to_host_pool[req.req_pool_idx]
        host_indices = host_indices[host_indices >= 0]
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

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv_allocated_len

        # release memory -- only free actually-allocated buffer indices
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        _hs_map = self.mem_pool_device.full_to_hisparse_device_index_mapping
        _valid = (compressed_locs >= 0) & (compressed_locs < _hs_map.shape[0])
        _hs_map[compressed_locs[_valid]] = 0

        host_indices = self.req_to_host_pool[req.req_pool_idx]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
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
        token_position_space: str = "compressed",
        num_steps: int = 1,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        When num_steps > 1 (speculative multi-step):
            seq_lens shape: [num_reqs * num_steps] flat req-major
            top_k_result shape: [num_reqs, num_steps, top_k]
            returns: [num_reqs, num_steps, top_k]
        """
        num_reqs = req_pool_indices.size(0)
        needed = num_reqs * num_steps

        if needed > self.top_k_device_locs_buffer.shape[0]:
            self.top_k_device_locs_buffer = torch.full(
                (needed, self.top_k), -1, dtype=torch.int32, device=self.device,
            )

        if num_steps > 1:
            top_k_indices = self.top_k_device_locs_buffer[:needed].view(
                num_reqs, num_steps, -1
            )
        else:
            top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)

        swap_seq_lens = compressed_seq_lens
        swap_top_k_result = top_k_result[:num_reqs] if top_k_result.size(0) > needed else top_k_result
        if token_position_space == "full" and self.is_dsv4_hisparse:
            if num_steps > 1:
                seq_lens_for_compare = compressed_seq_lens.view(
                    num_reqs, num_steps
                ).unsqueeze(2)
            else:
                seq_lens_for_compare = compressed_seq_lens.unsqueeze(1)
            valid_compressed_token = (
                (swap_top_k_result >= 0)
                & (swap_top_k_result < seq_lens_for_compare)
                & ((swap_top_k_result + 1) % self.compress_ratio == 0)
            )
            swap_top_k_result = torch.where(
                valid_compressed_token,
                swap_top_k_result // self.compress_ratio,
                torch.full_like(swap_top_k_result, -1),
            )
            if num_steps > 1:
                swap_seq_lens = (
                    compressed_seq_lens.view(num_reqs, num_steps) // self.compress_ratio
                ).reshape(-1)
            else:
                swap_seq_lens = compressed_seq_lens // self.compress_ratio
        elif token_position_space != "compressed":
            assert (
                token_position_space == "full"
            ), f"Unsupported token_position_space={token_position_space}"

        if self.is_dsv4_hisparse and not self.is_unified_hisparse:
            swap_in_fn = load_cache_to_device_buffer_dsv4_mla
        else:
            swap_in_fn = load_cache_to_device_buffer_mla
        swap_in_fn(
            top_k_tokens=swap_top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=swap_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=self.mem_pool_device.page_size if num_steps > 1 else 1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            num_steps=num_steps,
        )
        return top_k_indices
