
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import nvtx
import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostKVCache,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.sparsity2.core.sparse_coordinator import RequestTrackers
from sglang.srt.sparsity2.ops.triton_kernel import invoke_nsa_sparse_diff_kernel

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class SparseKVCacheManager:

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        req_states: RequestTrackers,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.request_counter = 0
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                1,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                1,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=server_args.hicache_storage_backend_extra_config,
        )

        self.sparse_decode_ongoing_offload = {}
        self.sparse_prefill_ongoing_offload = {}

        max_pool_size = self.req_to_token_pool.req_to_token.shape[0]
        self.bitmap = torch.full(
            (max_pool_size, server_args.model_config.context_len),
            -1,
            dtype=torch.int16,
            device=server_args.device,
        )
        self.req_states = req_states


    @nvtx.annotate(
        "DecodeKVCacheOffloadManager.transform_sparse_top_k_cache", color="red"
    )
    def transfer_sparse_top_k_cache(
        self,
        req_pool_indices,
        curr_top_k_result,
        out_cache_loc,
        seq_lens,
        layer_id,
    ):
        bs = curr_top_k_result.shape[0]
        top_k = curr_top_k_result.shape[1]

        # need to register buffer for cuda graph
        curr_device_indices = torch.full(
            (bs, top_k + 1), -1, dtype=torch.int64, device=self.device
        )
        should_load_device_indices = torch.full(
            (bs, top_k), -1, dtype=torch.int64, device=self.device
        )
        should_load_host_indices = torch.full(
            (bs, top_k), -1, dtype=torch.int64, device=self.device
        )

        full_host_indices = [self.req_states.full_host_indices[idx] for idx in req_pool_indices]
        invoke_nsa_sparse_diff_kernel(
            self.req_states.prev_top_k_result,
            curr_top_k_result,
            self.req_states.prev_device_indices,
            curr_device_indices,
            self.bitmap,
            full_host_indices,
            should_load_device_indices,
            should_load_host_indices,
            out_cache_loc,
            seq_lens,
            req_pool_indices,
            layer_id,
        )

        should_load_device_indices = should_load_device_indices[should_load_device_indices != -1]
        should_load_host_indices = should_load_host_indices[should_load_host_indices != -1]
        assert len(should_load_device_indices) == len(should_load_host_indices)
        if len(should_load_device_indices) > 0:
            # load cache from cpu
            self.host_mem_pool.load_to_device_per_layer(
                self.host_mem_pool.device_pool,
                should_load_host_indices,
                should_load_device_indices,
                layer_id,
                "kernel",
            )

        return curr_device_indices[:, :-1]


    def offload_sparse_decode_req_tokens(self, req_pool_indices, out_alloc_len):
        """Offload incremental token KV cache for sparse attention."""

        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=out_alloc_len.long(),
            node_id=ack_id,
        )
        assert host_indices is not None, "Host out of memory"
        self.sparse_decode_ongoing_offload[ack_id] = (host_indices, req_pool_indices)
        return ack_id

    def check_sparse_offload_progress(self):
        """Check the progress of offload from device to host for sparse schedule every step"""
        if len(self.sparse_decode_ongoing_offload) == 0:
            return

        cc = self.cache_controller
        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        finish_count = qsizes.tolist()[0]
        assert finish_count == 1

        _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
        finish_event.synchronize()

        # update full_host_indices
        host_indices, req_pool_indices = self.sparse_decode_ongoing_offload.pop(
            ack_list[0]
        )
        for i in range(len(req_pool_indices)):
            full_host_indices = self.req_states.full_host_indices[req_pool_indices[i]]
            if len(host_indices) > 0:
                host_idx_i = host_indices[i].to(self.req_states.device).reshape(-1)
                self.req_states.full_host_indices[req_pool_indices[i]] = torch.cat(
                    [full_host_indices, host_idx_i]
                )
            else:
                logger.warning(
                    f"Host indices is empty for request {req_pool_indices[i]}"
                )

    def offload_prefill_full_kv_cache(self, req):
        offloaded_len = len(req.origin_input_ids)
        token_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :offloaded_len
        ].long()

        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=token_indices,
            node_id=ack_id,
        )
        assert host_indices is not None, "Host out of memory"
        self.sparse_prefill_ongoing_offload[ack_id] = (host_indices, req)
        logger.info(
            f"Offloaded prefill full KV cache for request {req.rid}, offloaded len:{offloaded_len}, host len:{len(host_indices)}"
        )
        return ack_id

    def check_prefill_offload_progress(self):
        if len(self.sparse_prefill_ongoing_offload) == 0:
            return

        cc = self.cache_controller
        while True:
            qsizes = torch.tensor(
                [
                    len(cc.ack_write_queue),
                ],
                dtype=torch.int,
            )
            if self.tp_world_size > 1:
                torch.distributed.all_reduce(
                    qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
            finish_count = qsizes.tolist()[0]
            if finish_count > 0:
                break

        _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
        finish_event.synchronize()

        (host_indices, req) = self.sparse_prefill_ongoing_offload.pop(ack_list[0])
        self.req_states.full_host_indices[req.req_pool_idx] = host_indices.to(
            self.req_states.device
        )
