from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation
from sglang.srt.managers.cache_controller import (
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _pool_name_key(pool_name) -> str:
    return pool_name.value if hasattr(pool_name, "value") else str(pool_name)


class CacheOperation(BaseCacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.pool_transfers = pool_transfers

    @staticmethod
    def merge_pool_transfers(
        ops: List["CacheOperation"],
    ) -> Optional[list[PoolTransfer]]:
        grouped: dict[str, list[PoolTransfer]] = {}
        for op in ops:
            for transfer in op.pool_transfers or []:
                grouped.setdefault(_pool_name_key(transfer.name), []).append(transfer)
        if not grouped:
            return None

        def cat_or_none(tensors):
            parts = [x for x in tensors if x is not None]
            return torch.cat(parts) if parts else None

        merged = []
        for transfers in grouped.values():
            first = transfers[0]
            merged.append(
                PoolTransfer(
                    name=first.name,
                    host_indices=cat_or_none(t.host_indices for t in transfers),
                    device_indices=cat_or_none(t.device_indices for t in transfers),
                    keys=[k for t in transfers if t.keys for k in t.keys] or None,
                    hit_policy=first.hit_policy,
                    use_anchor_host_indices=first.use_anchor_host_indices,
                    use_anchor_device_indices=first.use_anchor_device_indices,
                )
            )
        return merged

    @staticmethod
    def merge_ops(ops: List["CacheOperation"]) -> "CacheOperation":
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation.merge_pool_transfers(ops),
        )
        merged.node_ids = node_ids
        return merged


class StorageOperation(BaseStorageOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()
        super().__init__(
            host_indices,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
    ):
        self.transfer_layer_num = transfer_layer_num
        startup_storage_backend = storage_backend
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            pp_rank=pp_rank,
            pp_size=pp_size,
            enable_storage_metrics=enable_storage_metrics,
        )
        self.transfer_layer_num = self.transfer_layer_num or self.layer_num
        if self.transfer_layer_num != self.layer_done_counter.num_layers:
            self.layer_done_counter = LayerDoneCounter(self.transfer_layer_num)
            self.mem_pool_device.register_layer_transfer_counter(
                self.layer_done_counter
            )
        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        for entry in host_pools or []:
            self.storage_backend.register_mem_host_pool_v2(
                entry.host_pool, _pool_name_key(entry.name)
            )
        logger.info("Using pool-based interface for hybrid storage operations")

    def reset(self):
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools, alloc_host=True
        )
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None

        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(op)
        self.write_queue.clear()
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                self.io_backend,
                pool_transfers=op.pool_transfers,
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        need_load_kv = host_indices.numel() > 0
        if need_load_kv:
            device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None
        else:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)

        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools, alloc_host=False
        )
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                self.mem_pool_device_allocator.free(device_indices)
            return None

        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(op)
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.transfer_layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                    pool_transfers=op.pool_transfers,
                )
                producer_event.complete(i)
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)
        self.ack_load_queue.append(
            HiCacheAck(
                producer_event.start_event,
                producer_event.finish_event,
                op.node_ids,
            )
        )
        return producer_id

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _populate_transfer_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        for transfer in pool_transfers:
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                transfer.keys = all_hashes[:kv_hit_pages]
            else:
                trailing_n = len(transfer.keys) if transfer.keys else 1
                transfer.keys = all_hashes[
                    max(0, kv_hit_pages - trailing_n) : kv_hit_pages
                ]

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        hash_value = []
        for start in range(0, len(operation.token_ids), self.page_size):
            last_hash = self.get_hash_str(
                operation.token_ids[start : start + self.page_size], last_hash
            )
            hash_value.append(last_hash)

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None
        )
        if operation.pool_transfers:
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        if kv_hit_pages > 0 and operation.pool_transfers:
            self._populate_transfer_keys(
                operation.pool_transfers, hash_value, kv_hit_pages
            )

        return hash_value[:kv_hit_pages], kv_hit_pages * self.page_size

    def _materialize_storage_transfers(
        self,
        pool_transfers: Optional[list[PoolTransfer]],
        anchor_host_indices: torch.Tensor,
        anchor_device_indices: Optional[torch.Tensor],
        kv_pages: int,
    ) -> Optional[list[PoolTransfer]]:
        if not pool_transfers:
            return None
        materialized: list[PoolTransfer] = []
        for transfer in pool_transfers:
            entry = self.mem_pool_host.entry_map.get(_pool_name_key(transfer.name))
            if entry is None:
                continue
            page_size = getattr(entry.host_pool, "page_size", self.page_size) or 1
            keys = transfer.keys or []
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                keys = keys[:kv_pages]
            item_count = len(keys) * page_size
            host_indices = transfer.host_indices
            if host_indices is None and transfer.use_anchor_host_indices:
                host_indices = anchor_host_indices
            if host_indices is not None:
                host_indices = host_indices[:item_count]
            device_indices = transfer.device_indices
            if device_indices is None and transfer.use_anchor_device_indices:
                device_indices = anchor_device_indices
            if device_indices is not None:
                device_indices = device_indices[:item_count]
            materialized.append(
                PoolTransfer(
                    name=transfer.name,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    keys=keys,
                    hit_policy=transfer.hit_policy,
                    use_anchor_host_indices=transfer.use_anchor_host_indices,
                    use_anchor_device_indices=transfer.use_anchor_device_indices,
                )
            )
        return materialized

    def _page_transfer(self, operation):
        super()._page_transfer(operation)
        kv_pages = operation.completed_tokens // self.page_size
        transfers = self._materialize_storage_transfers(
            operation.pool_transfers,
            operation.host_indices,
            None,
            kv_pages,
        )
        if transfers and not operation.is_terminated():
            results = self.storage_backend.batch_get_v2(transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

    def _page_backup(self, operation):
        super()._page_backup(operation)
        kv_pages = operation.completed_tokens // self.page_size
        transfers = self._materialize_storage_transfers(
            operation.pool_transfers,
            operation.host_indices,
            None,
            kv_pages,
        )
        if transfers:
            results = self.storage_backend.batch_set_v2(transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

    def get_usable_prefetch_token_count(self, operation: PrefetchOperation) -> int:
        usable_pages = operation.completed_tokens // self.page_size
        for transfer in operation.pool_transfers or []:
            if transfer.hit_policy != PoolHitPolicy.ALL_PAGES:
                continue
            usable_pages = min(
                usable_pages,
                operation.pool_storage_result.extra_pool_hit_pages.get(
                    _pool_name_key(transfer.name), 0
                ),
            )
        return usable_pages * self.page_size

    def _resolve_pool_transfers_allocation(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
    ) -> Optional[list[PoolTransfer]]:
        if not extra_pools:
            return None
        newly_allocated: list[tuple[PoolTransfer, Any, torch.Tensor]] = []
        for pool in extra_pools:
            entry = self.mem_pool_host.entry_map.get(_pool_name_key(pool.name))
            if entry is None:
                continue
            if alloc_host:
                if (
                    pool.host_indices is not None
                    or pool.device_indices is None
                    or pool.use_anchor_host_indices
                ):
                    continue
                entry_pool, evict_fn, size = (
                    entry.host_pool,
                    entry.host_evict_fn,
                    len(pool.device_indices),
                )
            else:
                if (
                    pool.device_indices is not None
                    or pool.host_indices is None
                    or pool.use_anchor_device_indices
                ):
                    continue
                entry_pool, evict_fn, size = (
                    entry.device_pool,
                    entry.device_evict_fn,
                    len(pool.host_indices),
                )
            indices = entry_pool.alloc(size)
            if indices is None and evict_fn:
                evict_fn(size)
                indices = entry_pool.alloc(size)
            if indices is None:
                for prev_pool, prev_entry_pool, prev_indices in newly_allocated:
                    prev_entry_pool.free(prev_indices)
                    if alloc_host:
                        prev_pool.host_indices = None
                    else:
                        prev_pool.device_indices = None
                return None
            if alloc_host:
                pool.host_indices = indices
            else:
                pool.device_indices = indices
            newly_allocated.append((pool, entry_pool, indices))
        return extra_pools
