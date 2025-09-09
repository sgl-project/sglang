import logging
import threading

import torch

from sglang import ServerArgs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.request_counter = 0
        self.tree_cache = tree_cache
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.decode_cache_controller = HiCacheController(
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

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        logger.info("Enable offload kv cache for decode side")

    def offload_kv_cache(self, req) -> bool:
        """Offload a finished request's KV cache to storage."""

        if self.decode_cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx == -1:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            logger.debug(
                f"Request {req.rid} has invalid token_indices: {token_indices}"
            )
            return False

        tokens = req.origin_input_ids + req.output_ids
        aligned_len = (len(tokens) // self.page_size) * self.page_size
        if aligned_len == 0:
            return False

        token_indices = token_indices[:aligned_len]
        tokens = tokens[:aligned_len]

        # Asynchronously offload KV cache from device to host by cache controller
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.decode_cache_controller.write(
            device_indices=token_indices.long(),
            node_id=ack_id,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        self.ongoing_offload[ack_id] = (req, host_indices, tokens)
        return True

    def check_offload_progress(self):
        self._check_offload_progress()
        self._check_backup_progress()

    def _check_offload_progress(self):
        """Check the progress of offload from device to host, and trigger backup from host to storage."""
        queue_size = torch.tensor(
            self.decode_cache_controller.ack_write_queue.qsize(), dtype=torch.int
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        for _ in range(queue_size.item()):
            ack_id = self.decode_cache_controller.ack_write_queue.get()
            req, host_indices, tokens = self.ongoing_offload[ack_id]

            # Release device
            self.tree_cache.cache_finished_req(req)

            # Trigger async backup from host to storage by cache controller
            self._trigger_backup(req.rid, host_indices, tokens)

            del self.ongoing_offload[ack_id]

    def _trigger_backup(self, req_id, host_indices, tokens):
        """Trigger async backup from host to storage by cache controller."""

        # Generate page hashes and write to storage
        page_hashes = []
        last_hash = ""
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.decode_cache_controller.get_hash_str(
                page_tokens, last_hash
            )
            page_hashes.append(last_hash)

        ack_id = self.decode_cache_controller.write_storage(
            host_indices,
            tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req_id, host_indices)

    def _check_backup_progress(self):
        """Check the progress of backup from host to storage."""

        queue_size = torch.tensor(
            self.decode_cache_controller.ack_backup_queue.qsize(), dtype=torch.int
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        for _ in range(queue_size.item()):
            storage_operation = self.decode_cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices = self.ongoing_backup[ack_id]

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)
            del self.ongoing_backup[ack_id]

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}"
            )
