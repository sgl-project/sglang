import logging
import threading
import time
from functools import partial

import torch

from sglang.srt.managers.eic_cache_controller import (
    EICCacheController,
    get_content_hash,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.eic_memory_pool import (
    EICMHATokenToKVPoolHost,
    EICMLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def mha_pool_get_flat_data(self: MHATokenToKVPool, indices: torch.Tensor):
    flatten = torch.stack(
        [
            torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
            torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
        ]
    )
    return flatten


def mha_pool_transfer(
    self: MHATokenToKVPool, indices: torch.Tensor, flat_data: torch.Tensor
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    k_data, v_data = flat_data[0], flat_data[1]
    for i in range(self.layer_num):
        self.k_buffer[i][indices] = k_data[i]
        self.v_buffer[i][indices] = v_data[i]


def mla_pool_get_flat_data(self: MLATokenToKVPool, indices: torch.Tensor):
    return torch.stack([self.kv_buffer[i][indices] for i in range(self.layer_num)])


def mla_pool_transfer(
    self: MLATokenToKVPool, indices: torch.Tensor, flat_data: torch.Tensor
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    for i in range(self.layer_num):
        self.kv_buffer[i][indices] = flat_data[i]


class EICChunkCache(ChunkCache):
    def __init__(
        self,
        params: CacheInitParams,
        server_args: ServerArgs,
    ):
        super().__init__(params)
        self.tp_group = params.tp_cache_group
        self.tp_size = self.tp_group.size()
        self.rank = self.tp_group.rank()
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        self.page_size = params.page_size
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = EICMHATokenToKVPoolHost(
                self.kv_cache,
                4.0,
                0,
                "cpu",
                self.page_size,
                self.rank,
                extra_info=self.get_extra_info(server_args),
            )
            self.kv_cache.get_flat_data = partial(mha_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(mha_pool_transfer, self.kv_cache)
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = EICMLATokenToKVPoolHost(
                self.kv_cache,
                4.0,
                0,
                "cpu",
                self.page_size,
                self.rank,
                extra_info=self.get_extra_info(server_args),
            )
            self.kv_cache.get_flat_data = partial(mla_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(mla_pool_transfer, self.kv_cache)
        else:
            raise ValueError(f"EICChunkCache only supports MHA and MLA yet")

        self.load_cache_event = threading.Event()
        self.cache_controller = EICCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            load_cache_event=self.load_cache_event,
            write_policy="write_through",
            server_args=server_args,
        )
        self.ongoing_writing_queue = dict()
        self.background_thread = threading.Thread(
            target=self.background_thread, daemon=True
        )
        self.background_thread.start()
        self._evictable_size = 0
        self.save_docode_cache = True

    def get_extra_info(self, server_args: ServerArgs):
        extra_info = {
            "model_path": server_args.model_path,
            "world_size": self.tp_size,
            "tp_rank": self.rank,
            "framework": "sglang",
        }
        return extra_info

    def write_backup(self, req: Req, save_decode_cache: bool = True):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
        ]
        if (
            len(self.ongoing_writing_queue) >= 20
            or len(kv_indices) < self.page_size
            or not save_decode_cache
            or isinstance(req.finished_reason, FINISH_ABORT)
        ):
            self.token_to_kv_pool_allocator.free(kv_indices)
            return
        page_aligned_len = len(kv_indices) // self.page_size * self.page_size
        logger.debug(f"page aligned length: {page_aligned_len}")
        paged_kv_indices = kv_indices[:page_aligned_len]
        token_ids = (req.origin_input_ids + req.output_ids)[:page_aligned_len]
        content_hash = get_content_hash(token_ids, self.page_size)
        # filter decode
        decode_hash_offset = len(req.origin_input_ids) // self.page_size
        decode_content_hash = content_hash[decode_hash_offset:]
        decode_paged_len = decode_hash_offset * self.page_size
        if decode_paged_len < self.page_size:
            self.token_to_kv_pool_allocator.free(kv_indices)
            return
        decode_device_indices = paged_kv_indices[decode_paged_len:]
        logger.debug(f"write decode cache len: {len(decode_device_indices)}")
        host_indices = self.cache_controller.write_page(
            device_indices=decode_device_indices,
            priority=None,
            node_id=req.rid,
            content_hash=decode_content_hash,
            data_copy=True,
        )
        if host_indices is not None:
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.ongoing_writing_queue[req.rid] = decode_device_indices.clone()
            logger.debug(
                f"cache request {req.rid} started, kvcache indices: {len(decode_device_indices)}"
            )
        else:
            self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, is_decode: bool = False
    ):
        save_cache = is_insert and is_decode and self.save_docode_cache
        self.write_backup(req, save_decode_cache=save_cache)
        self.req_to_token_pool.free(req.req_pool_idx)

    def writing_check(self):
        while not self.cache_controller.ack_write_queue.empty():
            try:
                rid, success = self.cache_controller.ack_write_queue.get_nowait()
                kv_indices = self.ongoing_writing_queue.get(rid)
                while not self.token_to_kv_pool_allocator.is_not_in_free_group:
                    time.sleep(0.1)
                logger.debug(
                    f"cache request {rid} complete, kvcache indices: {len(kv_indices)} "
                )
                del self.ongoing_writing_queue[rid]
            except Exception as e:
                logger.error(f"Error in writing check: {e}")
                continue

    def background_thread(self):
        while True:
            time.sleep(0.1)
            self.writing_check()

    def evictable_size(self):
        return self._evictable_size

    def reset(self):
        logger.info("Reset EICChunkCache")
        while len(self.ongoing_writing_queue) > 0:
            time.sleep(0.1)
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.ongoing_writing_queue = dict()
        self._evictable_size = 0


class EICSWAChunkCache(EICChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def __init__(
        self,
        params: CacheInitParams,
        server_args: ServerArgs,
    ):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        super().__init__(
            params,
            server_args,
        )

    def evict_swa(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
    ):
        if prelen >= req.evicted_seqlen_local + attention_chunk_size:
            new_evicted_seqlen_local = attention_chunk_size * (
                prelen // attention_chunk_size
            )
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.evicted_seqlen_local : new_evicted_seqlen_local
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.evicted_seqlen_local = new_evicted_seqlen_local

    def evict(self, num_tokens: int):
        pass
