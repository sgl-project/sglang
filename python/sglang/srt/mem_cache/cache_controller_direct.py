import logging
from queue import Empty
from typing import List, Optional

import torch
from torch import Tensor

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, get_hash_str
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.storage import StorageBackendFactory

logger = logging.getLogger(__name__)


def get_hash_list(
    token_ids: List[int], prior_hash: str = None, page_size: int = 128
) -> List[str]:
    assert len(token_ids) % page_size == 0
    hashes = []
    last_hash = prior_hash
    token_groups = (
        token_ids[i : i + page_size] for i in range(0, len(token_ids), page_size)
    )
    for group in token_groups:
        last_hash = get_hash_str(group, last_hash)
        hashes.append(last_hash)
    return hashes


class LoadStorageOperation:
    counter = 0

    def __init__(
        self,
        request_id: str,
        device_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        page_size: int = 128,
    ):
        self.request_id = request_id
        self.device_indices = device_indices

        self.token_ids = token_ids
        self.last_hash = last_hash
        self.hash_keys = get_hash_list(token_ids, last_hash, page_size)

        self.id = LoadStorageOperation.counter
        LoadStorageOperation.counter += 1


class HiCacheControllerDirect:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        storage_backend: str,
        device_id: int = 0,
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        if self.mem_pool_device_allocator:
            self.device = self.mem_pool_device_allocator.device
        else:
            self.device = torch.device("cpu")
        # self.kv_layer_ptrs: every layer ptr
        # self.kv_layer_nbytes: the byte length of each layer
        # self.kv_page_nbytes: the page byte length of each layer
        self.kv_layer_ptrs, self.kv_layer_nbytes, self.kv_page_nbytes = (
            self.mem_pool_device.get_contiguous_buf_infos()
        )

        self.page_size = page_size
        self.device_id = device_id
        self.is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = self.is_mla_model and self.tp_rank != 0

        self.storage_config = HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            is_mla_model=self.is_mla_model,
            is_page_first_layout=False,
            model_name=None,
            extra_config={"device_id": device_id},
        )
        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, None
            )
        except ValueError as e:
            raise ValueError(f"Failed to create storage backend: {e}") from e

        self.storage_backend.register_mem_pool_device(self.mem_pool_device)

        self.load_tokens_threshold = 128
        # granularity of batch storage IO operations, in number of pages
        self.storage_batch_size = 256

        # create a new communication group for synchronizing storage operations across TP workers
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if self.tp_world_size > 1:
            group_ranks = torch.distributed.get_process_group_ranks(tp_group)
            self.load_tp_group = torch.distributed.new_group(
                group_ranks, backend="gloo"
            )

        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        self.load_queue: List[LoadStorageOperation] = []

    def reset(self):
        self.load_queue.clear()

    def write(self, hash_keys: List[str], device_indices: torch.Tensor) -> int:
        if self.backup_skip:
            return 0

        try:
            succ_pages_num = self._memcpy_between_device_and_storage(
                hash_keys, device_indices, "write"
            )
            if self.tp_world_size > 1 and self.is_mla_model is False:
                # only mha model need all reduce
                succ_pages_num = self._allreduce_results(succ_pages_num)

            logger.debug(f"success write {succ_pages_num} pages")
            return succ_pages_num * self.page_size
        except Empty:
            return 0

    def load(
        self,
        rid,
        new_input_tokens,
        device_indices,
        last_hash: Optional[str] = None,
    ):
        """
        Load KV caches from L3 storage to device memory.
        """
        self.load_queue.append(
            LoadStorageOperation(rid, device_indices, new_input_tokens, last_hash)
        )
        device_indices, free_device_indices = self.start_loading()
        return device_indices, free_device_indices

    def start_loading(self) -> tuple[Tensor | None, Tensor | None]:
        if len(self.load_queue) == 0:
            return None, None

        assert len(self.load_queue) == 1
        # producer_id = self.layer_done_counter.update_producer()
        op = self.load_queue[0]
        self.load_queue.clear()
        # producer_event = self.layer_done_counter.events[producer_id]
        # producer_event.start_event.record()

        try:
            hit_hash_len = self._storage_hit_query(op)
            if self.tp_world_size > 1:
                hit_hash_len = self._allreduce_results(hit_hash_len)

            hit_token_len = hit_hash_len * self.page_size
            if hit_token_len < self.load_tokens_threshold:
                # not to load storage if not enough benefits
                logger.debug(
                    f"Revoking Load operation for request {op.request_id} due to insufficient hits ({hit_token_len})."
                )
                return None, op.device_indices
            else:
                hit_hash_keys = op.hash_keys[:hit_hash_len]
                device_indices = op.device_indices[:hit_token_len]
                succ_pages_num = self._memcpy_between_device_and_storage(
                    hit_hash_keys, device_indices, "load"
                )
                if self.tp_world_size > 1:
                    succ_pages_num = self._allreduce_results(succ_pages_num)

                token_len = succ_pages_num * self.page_size
                hit_device_indices = op.device_indices[:token_len]
                free_device_indices = op.device_indices[token_len:]

                logger.debug(
                    f"success load {token_len} tokens for request {op.request_id}"
                )
                return hit_device_indices, free_device_indices
        except Empty:
            logger.error(
                f"Failed load storage {len(op.hash_keys)} pages for request {op.request_id}"
            )
            return None, op.device_indices

    def _allreduce_results(self, result: int) -> int:
        result_tensor = torch.tensor(result, dtype=torch.int)
        torch.distributed.all_reduce(
            result_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.load_tp_group,
        )

        return result_tensor.item()

    def _storage_hit_query(self, operation: LoadStorageOperation) -> int:
        if not operation.hash_keys:
            return 0

        total_len = len(operation.hash_keys)
        total_hit_num = 0
        for start in range(0, total_len, self.storage_batch_size):
            end = min(start + self.storage_batch_size, total_len)
            batch_hashes = operation.hash_keys[start:end]
            hit_num = self.storage_backend.batch_exists(batch_hashes)
            total_hit_num += hit_num
            if hit_num < len(batch_hashes):
                break

        return total_hit_num

    def _memcpy_between_device_and_storage(
        self,
        hash_keys: List[str],
        device_indices: torch.Tensor,
        direction: str,
    ) -> int:
        assert hash_keys

        batch_memcpy = None
        if direction == "write":
            batch_memcpy = self.storage_backend.batch_set
        elif direction == "load":
            batch_memcpy = self.storage_backend.batch_get
        assert batch_memcpy is not None

        total_elements = len(hash_keys)
        ptr_list, element_size_list = self._get_page_buffer_meta(device_indices)
        assert total_elements == len(ptr_list)
        assert total_elements == len(element_size_list)
        total_succ_num = 0
        for start in range(0, total_elements, self.storage_batch_size):
            end = min(start + self.storage_batch_size, total_elements)
            batch_hashes = hash_keys[start:end]
            target_locations = ptr_list[start:end]
            target_sizes = element_size_list[start:end]
            succ_num = batch_memcpy(
                keys=batch_hashes,
                target_locations=target_locations,
                target_sizes=target_sizes,
            )
            total_succ_num += succ_num
            if succ_num < len(batch_hashes):
                break

        return total_succ_num

    def _parse_success_hashes_from_l3_results(
        self,
        hash_keys: List[str],
        results: List[int],
    ) -> int:
        # for each key
        hit_hash_len = 0
        for h, r in zip(hash_keys, results):
            if r == 1:
                hit_hash_len += 1
            else:
                break

        return hit_hash_len

    def _get_page_buffer_meta(
        self, device_indices: torch.Tensor
    ) -> tuple[List[List[int]], List[List[int]]]:
        # 1. concatenate device index tensors
        token_len = device_indices.shape[0]
        assert token_len % self.page_size == 0

        # 2. compute page indices
        group_first_indices = device_indices[:: self.page_size]
        page_indices = group_first_indices // self.page_size

        # 3. Translate layer_ptrs and page_nbytes into tensors
        kv_layer_ptrs_tensor = torch.tensor(
            self.kv_layer_ptrs, dtype=torch.int64, device=device_indices.device
        )
        kv_page_nbytes_tensor = torch.tensor(
            self.kv_page_nbytes, dtype=torch.int64, device=device_indices.device
        )

        # 4. compute the pointers and sizes for all layers
        # Expand page_indices to shape [M, 1] and broadcast with [L] to shape [M, L]
        page_indices_expanded = page_indices.unsqueeze(1)
        ptr_tensor = (
            kv_layer_ptrs_tensor + page_indices_expanded * kv_page_nbytes_tensor
        )
        element_size_tensor = kv_page_nbytes_tensor.unsqueeze(0).expand_as(ptr_tensor)

        # 6. translate to a list
        ptr_list = ptr_tensor.tolist()
        element_size_list = element_size_tensor.tolist()

        return ptr_list, element_size_list
