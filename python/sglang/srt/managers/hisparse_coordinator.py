# to be combined with the sparse coordinator class and sparse algorithm family

from typing import List, NamedTuple

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseTokenToKVPoolAllocator

try:
    from sglang.srt.mem_cache.hisparse_memory_pool import DeepSeekV4SingleKVPoolHost
except ImportError:
    DeepSeekV4SingleKVPoolHost = None
from sglang.srt.utils import get_device_module

device_module = get_device_module()

from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mla
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


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
        self.mem_pool_host = DeepSeekV4SingleKVPoolHost(
            self.mem_pool_device,
            self.token_to_kv_pool_allocator.size_full // self.compress_ratio,
            1,
        )
        self.item_size_bytes = (
            self.mem_pool_host.kv_cache_total_dim * self.mem_pool_host.dtype.itemsize
        )

        max_num_reqs = req_to_token_pool.size
        max_context_len = req_to_token_pool.max_context_len

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_reqs, self.padded_buffer_size), dtype=torch.int64, device=device
        )
        self.req_to_host_pool = torch.zeros(
            (max_num_reqs, max_context_len // self.compress_ratio),
            dtype=torch.int64,
            device=device,
        )

        self.write_staging_stream = device_module.Stream()
        self.write_decoding_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.ack_decoding_queue: List[HiSparseAct] = []

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (max_num_reqs, layer_num, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (max_num_reqs, layer_num, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.bitmap = torch.full(
            (max_num_reqs, max_context_len // self.compress_ratio),
            -1,
            dtype=torch.int16,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(max_num_reqs, layer_num, 1)
            .contiguous()
        )
        self.transfer_tasks_src = torch.full(
            (max_num_reqs * (self.top_k + 1),),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.transfer_tasks_dst = torch.full(
            (max_num_reqs * (self.top_k + 1),),
            -1,
            dtype=torch.int64,
            device=device,
        )

    def admit_request_into_staging(self, req: Req) -> None:
        req.staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )
        # req.c4_indices = device_indices

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc(prefill_len).to(device=self.device)
        assert host_indices is not None, "Host mem pool alloc failed"
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
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def alloc_device_buffer(self, req: Req) -> None:
        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, : req.seqlen]
            )
        )
        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, self.padded_buffer_size
        ).to(torch.int32)
        assert (
            len(buffer_indices) == self.padded_buffer_size
        ), "Device buffer alloc failed"
        self.req_to_device_buffer[req.req_pool_idx, : self.padded_buffer_size] = (
            buffer_indices
        )
        # initialize the token locs for the device buffer
        self.req_device_buffer_tokens[
            req.req_pool_idx, :, : self.device_buffer_size
        ] = torch.arange(self.device_buffer_size, device=self.device)
        self.req_device_buffer_token_locs[
            req.req_pool_idx, :, : self.padded_buffer_size
        ] = buffer_indices[: self.padded_buffer_size]

    def testing_backup(self, req):
        device_indices = req.c4_indices
        host_indices = self.req_to_host_pool[req.req_pool_idx, : len(device_indices)]

        self.mem_pool_host.testing_backup_to_device_all_layer(
            self.mem_pool_device, host_indices, device_indices
        )
        torch.cuda.current_stream().synchronize()

    def collect_ready_batch(self) -> List[Req]:
        ready_batch = None
        if len(self.ack_staging_queue) == 0:
            return ready_batch

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
            req.staging = False
            finish_count -= 1
            if len(self.ack_staging_queue) == 0:
                ready_batch = req.batch
            elif self.ack_staging_queue[0][2].batch != req.batch:
                ready_batch = req.batch
            # to break the circular reference
            req.batch = None
            # self.testing_backup(req)
        return ready_batch

    def map_last_loc_to_buffer(
        self,
        out_cache_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> None:
        active_reqs = seq_lens % self.compress_ratio == 0
        new_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        # point output locations to the reserved buffer locations
        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            new_out_cache_loc
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, self.device_buffer_size
        ]
        # todo, maybe clear the prior mapping as well
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )
        # proceed only if the backup is finished for new generated tokens
        self.wait_for_decode_writes()

    def get_front_topk_tokens(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        raw_indices: torch.Tensor,
    ) -> torch.Tensor:
        # a dummy selection for testing
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )
        for i in range(num_reqs):
            top_n = min(
                seq_lens[i] // self.compress_ratio,
                self.top_k,
            )
            if top_n == 0:
                continue
            top_k_indices[i, :top_n] = self.req_to_device_buffer[req_pool_indices[i]][
                raw_indices[i, :top_n]
            ]
        return top_k_indices

    def wait_for_decode_writes(self) -> None:
        if len(self.ack_decoding_queue) == 0:
            return
        _, finish_event, _ = self.ack_decoding_queue.pop(0)
        finish_event.synchronize()

    def retract_req(self, req: Req) -> None:
        # todo
        raise NotImplementedError

    def update_requests_after_decode(self, reqs: List[Req]) -> None:
        reqs_to_backup = [r for r in reqs if r.seqlen % self.compress_ratio == 0]
        if len(reqs_to_backup) == 0:
            return

        req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in reqs_to_backup], device=self.device
        )
        req_seq_lens = torch.tensor(
            [r.seqlen // self.compress_ratio for r in reqs_to_backup],
            device=self.device,
        )
        buffer_indices = self.req_to_device_buffer[
            req_pool_indices, self.device_buffer_size
        ]

        # for short requests, copy the new token from reserved buffer to normal buffer
        short_reqs = req_seq_lens <= self.device_buffer_size
        if torch.any(short_reqs):
            new_token_buffer_indices = self.req_to_device_buffer[
                req_pool_indices[short_reqs], req_seq_lens[short_reqs] - 1
            ]
            # todo, need to do the same transfer after prefill as well
            self.mem_pool_device.transfer_values_on_device(
                buffer_indices[short_reqs], new_token_buffer_indices
            )

        # for all requests, backup the new token to host for future use
        host_indices = self.mem_pool_host.alloc(len(buffer_indices)).to(
            device=self.device
        )
        assert host_indices is not None, "Host mem pool alloc failed"
        self.req_to_host_pool[req_pool_indices, req_seq_lens - 1] = host_indices

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_decoding_stream):
            start_event.wait(self.write_decoding_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                buffer_indices.contiguous(),
            )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_decoding_stream)
            if buffer_indices.is_cuda:
                buffer_indices.record_stream(self.write_decoding_stream)

        self.ack_decoding_queue.append(HiSparseAct(start_event, finish_event, None))

    def request_finished(self, req: Req):
        compressed_len = req.seqlen // self.compress_ratio
        # release memory
        buffer_indices = self.req_to_device_buffer[req.req_pool_idx]
        self.token_to_kv_pool_allocator.free_hisparse_indices(buffer_indices)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :compressed_len]
        self.mem_pool_host.free(host_indices)
        # clear req info
        self.req_device_buffer_tokens[req.req_pool_idx, :, :] = -1
        self.req_device_buffer_token_locs[req.req_pool_idx, :, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = 0
        self.lru_slots[req.req_pool_idx].copy_(self._lru_init)

    def swap_in_selected_pages(
        self,
        req_pool_indices,
        top_k_result,
        top_k_device_locs,
        seq_lens,
        layer_id,
    ):
        """
        Swap in selected top-k pages/tokens from host to device memory.
        First step: Using diff kernel to identify the top-k pages/tokens that need to be swapped in.
        Second step: Using the io kernel to load the pages/tokens from host to device.
        Returns:
            Device indices of the selected pages/tokens
        """
        block_size = 512
        load_cache_to_device_buffer_mla(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens,
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs,
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_device_locs,
            diff_map=self.bitmap,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens // self.compress_ratio,
            lru_slots=self.lru_slots,
            transfer_tasks_src=self.transfer_tasks_src,
            transfer_tasks_dst=self.transfer_tasks_dst,
            page_size=1,
            layer_id=layer_id,
            item_size_bytes=self.item_size_bytes,
            block_size=block_size,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
        )
        return top_k_device_locs
