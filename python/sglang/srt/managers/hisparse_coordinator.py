import logging
from typing import List, NamedTuple

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.hisparse_memory_pool import (
    DeepSeekV4SingleKVPoolHost,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.utils import get_device_module

device_module = get_device_module()

from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mla
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

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
        token_to_kv_pool_allocator: HiSparseTokenToKVPoolAllocator,
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
        host_size = self.token_to_kv_pool_allocator.size_full // self.compress_ratio
        self.mem_pool_host = DeepSeekV4SingleKVPoolHost(
            self.mem_pool_device, host_size, 1
        )
        self.item_size_bytes = (
            self.mem_pool_host.kv_cache_total_dim * self.mem_pool_host.dtype.itemsize
        )

        max_num_reqs = req_to_token_pool.size
        max_context_len = req_to_token_pool.max_context_len

        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_reqs, self.padded_buffer_size), dtype=torch.int64, device=device
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_reqs, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_reqs, max_context_len // self.compress_ratio),
            -1,
            dtype=torch.int64,
            device=device,
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_reqs, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_reqs, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_reqs, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )
        self._top_k_arange = torch.arange(self.top_k, device=device).unsqueeze(0)

        self.top_k_device_locs_buffer = torch.full(
            (max_num_reqs, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_reqs, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        self._skip_first_backup = [False] * max_num_reqs

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

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
            req.req_pool_idx, : len(req.fill_ids)
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc(prefill_len)
        if host_indices is None:
            logger.error(
                "HiSparse: host mem pool alloc failed for %d tokens (req %s)",
                prefill_len,
                req.rid,
            )
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {prefill_len} tokens"
            )
        host_indices = host_indices.to(device=self.device)
        self.req_to_host_pool[req.req_pool_idx, :prefill_len] = host_indices

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def alloc_device_buffer(self, req: Req) -> None:
        prefill_len = len(req.fill_ids)
        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :prefill_len]
            )
        )
        compressed_len = len(compressed_logical_indices)
        alloc_size = self.padded_buffer_size

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
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

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
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
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
    ) -> None:
        req_pool_indices_cpu = req_pool_indices.cpu()

        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

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

        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs = self.mem_pool_host.alloc(len(device_locs))
        if host_locs is None:
            logger.error(
                "HiSparse: host mem pool alloc failed for %d decode backup tokens",
                len(device_locs),
            )
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {len(device_locs)} decode backup tokens"
            )
        host_locs = host_locs.to(device=self.device)
        self.req_to_host_pool[backup_req_indices, actual_compressed_pos] = host_locs

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

    def get_front_topk_tokens(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        compressed_seq_lens = seq_lens // self.compress_ratio
        top_k_indices = self.req_to_device_buffer[req_pool_indices, : self.top_k].to(
            torch.int32
        )
        mask = self._top_k_arange >= compressed_seq_lens.unsqueeze(1)
        top_k_indices[mask] = -1
        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        self.write_staging_stream.synchronize()

        prefill_len = len(req.fill_ids)
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        compressed_len = prefill_len // self.compress_ratio
        host_indices = self.req_to_host_pool[req.req_pool_idx, :compressed_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self._skip_first_backup[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        compressed_len = req.seqlen // self.compress_ratio

        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.seqlen
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        host_indices = self.req_to_host_pool[req.req_pool_idx, :compressed_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        num_reqs = req_pool_indices.size(0)

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)

        block_size = 1024
        load_cache_to_device_buffer_mla(
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
            block_size=block_size,
            num_real_reqs=self.num_real_reqs,
        )
        return top_k_indices
