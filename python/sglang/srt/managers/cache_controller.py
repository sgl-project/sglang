from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING, Callable, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    count_pool_hits,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.pool_host import HostKVCache

from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None
        self.pool_transfers = pool_transfers

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def _merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[List[PoolTransfer]]:
        grouped: dict[tuple[PoolName, Optional[PoolName]], List[PoolTransfer]] = {}
        for op in ops:
            for transfer in op.pool_transfers or []:
                grouped.setdefault(
                    (transfer.name, transfer.indices_from_pool), []
                ).append(transfer)
        if not grouped:
            return None

        def cat_or_none(tensors):
            parts = [tensor for tensor in tensors if tensor is not None]
            return torch.cat(parts) if parts else None

        return [
            PoolTransfer(
                name=transfers[0].name,
                host_indices=cat_or_none(t.host_indices for t in transfers),
                device_indices=cat_or_none(t.device_indices for t in transfers),
                keys=[key for t in transfers if t.keys for key in t.keys] or None,
                hit_policy=transfers[0].hit_policy,
                indices_from_pool=transfers[0].indices_from_pool,
            )
            for transfers in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert ops
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation._merge_pool_transfers(ops),
        )
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]
    num_tokens: int = 0
    timing_enabled: bool = False
    # Tokens transferred per host pool (PoolName value -> count).
    num_tokens_by_pool: Optional[dict[str, int]] = None
    # Total bytes moved by the op across all pools, including draft piggyback
    # and sidecar transfers that the per-pool token counts exclude.
    num_bytes: int = 0


@dataclass
class PrefetchAck:
    """ACK for prefetch operation.

    A sequence of PrefetchAck is sent to the scheduler thread via ack_prefetch_queue,
    indicating progress or completion of the prefetch operation.

    For example, a prefetch operation may results into the following sequence of PrefetchAck:

    1. PrefetchAck(completed_tokens = 128)
    2. PrefetchAck(completed_tokens = 256)
    3. PrefetchAck(pool_hits={INDEXER: 256})
    4. PrefetchAck(completed_req = True)

    The last PrefetchAck always specifies completed_req = True.
    """

    rid: str
    operation: PrefetchOperation
    # Number of hits in KV pool.
    completed_tokens: Optional[int] = None
    # Number of hits in extra pools.
    pool_hits: Optional[dict[str, int]] = None
    completed_req: Optional[bool] = None


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: Optional[torch.Tensor],
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys
        # Full queried page-hash chain, set by _storage_hit_query before
        # hash_value is truncated to the hit boundary; the tail is the
        # absence signal that invalidates buffer-mode existence beliefs.
        self.all_hash_values: Optional[List[str]] = None
        # Prefetch-outcome accounting, set at enqueue by the tree cache.
        self.stats_requested_tokens = 0
        self.stats_total_tokens = 0

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: StorageOperation):
        return self.id < other.id


# Buffer-mode staging budgets. Prefetch staging is latency-critical
# (wait_complete gates TTFT), so loads may fill the pool up to this fraction
# before new prefetches are declined.
HICACHE_LOAD_POOL_USAGE_FRACTION = 0.9
# Write-staging floor: writes are deferrable, so the flush gate grows the
# write window dynamically into whatever load staging is not using.
HICACHE_WRITE_STAGING_POOL_FRACTION = 0.2


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.storage_hit_count = 0
        self.start_time = time.monotonic()

        super().__init__(None, token_ids, last_hash, prefix_keys=prefix_keys)

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        with self._lock:
            return self._terminated_flag


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        enable_storage_metrics: bool = False,
        host_memory_mode: str = "cache",
    ):
        self.tp_group = tp_group
        self.host_memory_mode = host_memory_mode
        self.attn_cp_group = attn_cp_group
        self.attn_tp_group = attn_tp_group
        self.pp_group = pp_group
        self.prefetch_hits_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.prefetch_completion_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.storage_host_pool = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage_metrics = enable_storage_metrics
        # Buffer mode: wired by the tree cache after attach; the load rate
        # limiter subtracts write staging from actual pool usage.
        self.host_write_staged_tokens_fn: Optional[Callable[[], int]] = None

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        self.storage_stop_event = threading.Event()

        # Storage control queues, (re)created whenever the storage threads start.
        self.prefetch_buffer: Optional[Queue[PrefetchOperation]] = None
        self.prefetch_sync_queue: Optional[Queue[PrefetchAck]] = None
        self.prefetch_hit_queue: Optional[Queue[StorageOperation]] = None
        self.ack_prefetch_queue = Queue[PrefetchAck]()
        self.ack_backup_queue: Optional[Queue[StorageOperation]] = None
        self.host_mem_release_queue: Optional[Queue[torch.Tensor]] = None

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        # Set by the scheduler to the forward stream; gates load-back H2D
        # behind in-flight forwards (see start_loading).
        self.load_fence_stream = None
        self.ack_write_queue: List[HiCacheAck] = []

        self.l2_transfer_engine = L2TransferEngine(io_backend)

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group."""
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_sync_groups(self) -> List[torch.distributed.ProcessGroup]:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        groups: List[torch.distributed.ProcessGroup] = []
        seen_rank_sets = set()

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]
        if self.pp_group is not None:
            base_groups.append(self.pp_group)

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )
        return groups

    def _destroy_sync_groups(
        self, groups: List[torch.distributed.ProcessGroup]
    ) -> None:
        for group in groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass

    def _all_reduce(
        self,
        tensor: torch.Tensor,
        op,
        groups: List[torch.distributed.ProcessGroup],
    ) -> None:
        for group in groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_sync_thread = threading.Thread(
            target=self.prefetch_sync_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()
        self.prefetch_buffer = Queue()
        self.prefetch_sync_queue = Queue()
        self.prefetch_hit_queue = Queue()
        self.ack_prefetch_queue = Queue()
        self.ack_backup_queue = Queue()
        self.host_mem_release_queue = Queue()

        self.prefetch_thread.start()
        self.prefetch_io_aux_thread.start()
        self.prefetch_sync_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear storage_stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
            if hasattr(self, "prefetch_sync_queue"):
                self.prefetch_sync_queue.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)
        if hasattr(self, "prefetch_sync_thread"):
            threads.append(self.prefetch_sync_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.storage_host_pool
            )
            self.storage_backend.register_mem_pool_host(self.storage_host_pool)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            if self.host_memory_mode == "buffer_only":
                # The whole pool is transient staging; loads may fill it up
                # to this fraction, and the tree's write flush gate yields
                # to live fetch demand (the write fraction is a floor).
                self.prefetch_capacity_limit = int(
                    HICACHE_LOAD_POOL_USAGE_FRACTION * self.mem_pool_host.size
                )
            else:
                # Budget speculative prefetch at half the host pool, leaving the rest for the write-back staging path.
                self.prefetch_capacity_limit = int(0.5 * self.mem_pool_host.size)
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            self.prefetch_hits_sync_groups = self._create_sync_groups()
            self.prefetch_completion_sync_groups = self._create_sync_groups()

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm", "mori"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_sync_groups(self.prefetch_hits_sync_groups)
            self._destroy_sync_groups(self.prefetch_completion_sync_groups)
            self.prefetch_hits_sync_groups = []
            self.prefetch_completion_sync_groups = []
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_sync_groups(
            self.prefetch_hits_sync_groups + self.prefetch_completion_sync_groups
        )
        self.prefetch_hits_sync_groups = []
        self.prefetch_completion_sync_groups = []

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        if is_dp_attention_enabled():
            self.tp_rank = get_parallel().attn_tp_rank
            self.tp_size = get_parallel().attn_tp_size
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_parallel().tp_rank
            self.tp_size = get_parallel().tp_size
            self.dp_rank = 0

        self.pp_rank = get_parallel().pp_rank
        self.pp_size = get_parallel().pp_size

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        # DeepSeekV4TokenToKVPool has compressed MLA-style rank-replicated cache
        # data. storage only needs rank 0 to write it back.
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)
        is_compressed_mla_model = isinstance(
            self.mem_pool_device, DeepSeekV4TokenToKVPool
        )
        is_rank_replicated = is_mla_model or is_compressed_mla_model
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_rank_replicated
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            # TODO(hzh): Rename is_mla_model to is_rank_replicated.
            is_mla_model=is_rank_replicated,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.prefetch_io_aux_thread.join()
            self.prefetch_sync_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_buffer.queue.clear()
            self.prefetch_sync_queue.queue.clear()
            self.prefetch_hit_queue.queue.clear()
            self.ack_prefetch_queue.queue.clear()
            self.ack_backup_queue.queue.clear()
            self.host_mem_release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.prefetch_io_aux_thread = threading.Thread(
                target=self.prefetch_io_aux_func, daemon=True
            )
            self.prefetch_sync_thread = threading.Thread(
                target=self.prefetch_sync_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.prefetch_io_aux_thread.start()
            self.prefetch_sync_thread.start()
            self.backup_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices, pool_transfers = self._move_write_operation(op)
        self.write_queue.clear()

        completion = self.l2_transfer_engine.submit_device_to_host(
            self._l2_transfers(host_indices, device_indices, pool_transfers)
        )

        self.ack_write_queue.append(
            HiCacheAck(
                start_event=completion.start_event,
                finish_event=completion.finish_event,
                node_ids=op.node_ids,
                num_tokens=len(op.device_indices),
                timing_enabled=completion.timing_enabled,
                num_tokens_by_pool=self._num_tokens_by_pool(op),
                num_bytes=self._transfer_num_bytes(op),
            )
        )

    def _transfer_num_bytes(self, op: CacheOperation) -> int:
        return len(op.device_indices) * self.mem_pool_host.size_per_token

    def _num_tokens_by_pool(self, op: CacheOperation) -> dict[str, int]:
        return {PoolName.KV.value: len(op.device_indices)}

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def move_indices(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def _move_write_operation(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[List[PoolTransfer]]]:
        """Keep CPU host indices only for page-first staged write-back."""
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            return op.host_indices, op.device_indices, op.pool_transfers
        return self._move_op_indices(op)

    def _move_op_indices(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[List[PoolTransfer]]]:
        return (*self.move_indices(op.host_indices, op.device_indices), None)

    def _l2_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        transfers = [
            L2Transfer(
                host_pool=self.mem_pool_host,
                device_pool=self.mem_pool_device,
                host_indices=host_indices,
                device_indices=device_indices,
            )
        ]
        return transfers

    def _l2_load_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        return self._l2_transfers(host_indices, device_indices, pool_transfers)

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, pool_transfers = self._move_op_indices(op)
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        if self.load_fence_stream is not None:
            # in overlap scheduling, reclaimed pages might still be written by the forward thread
            # therefore a fence is needed for loading thread to prevent memory corruption
            # todo: it's possible to use a finer-grained fence
            self.l2_transfer_engine.host_to_device_stream.wait_stream(
                self.load_fence_stream
            )

        completion = self.l2_transfer_engine.submit_host_to_device(
            self._l2_load_transfers(host_indices, device_indices, pool_transfers),
            start_event=producer_event.start_event,
            on_layer_done=producer_event.complete,
            layer_num=self.layer_num,
        )

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=completion.start_event,
                finish_event=completion.finish_event,
                node_ids=op.node_ids,
                num_tokens=len(op.device_indices),
                timing_enabled=completion.timing_enabled,
                num_tokens_by_pool=self._num_tokens_by_pool(op),
                num_bytes=self._transfer_num_bytes(op),
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def prefetch(
        self,
        request_id: str,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        """
        Request to terminate a prefetch operation.

        Must be called in the scheduler thread.

        Asynchronous prefetch tasks may be running in background threads.  When all prefetch
        tasks are terminated, a PrefetchAck with completed_req=True will be sent to ack_prefetch_queue.
        """
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ) -> int:
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += 1
        return inc

    # todo: deprecate
    def _generic_page_get(
        self, operation, hash_values, host_indices, extra_info=None
    ) -> int:
        dummy_page_dst = [
            self.storage_host_pool.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return 0
        count = 0
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            if operation.is_terminated():
                break
            self.storage_host_pool.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            count += 1
        return count

    def _page_transfer(self, operation: PrefetchOperation) -> int:
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        kv_derived_transfers = [
            transfer
            for transfer in getattr(operation, "pool_transfers", None) or []
            if transfer.indices_from_pool == PoolName.KV
        ]
        all_success = True
        completed_pages = 0
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            # When an error is occurred, we should keep looping and produce the same number of
            # PrefetchAck as other ranks do, because prefetch_sync_thread (i.e. consumer of
            # prefetch_sync_queue) perform reduce on the results.  This is so tricky.
            if all_success and operation.is_terminated():
                all_success = False
            if all_success:
                batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
                batch_host_indices = operation.host_indices[
                    i * self.page_size : (i + len(batch_hashes)) * self.page_size
                ]

                # Get one batch token, and update the completed_tokens if succeed
                extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)

                hit_pages = self._page_transfer_kv_batch(
                    operation,
                    batch_hashes,
                    batch_host_indices,
                    extra_info,
                    kv_derived_transfers,
                )
                # Check termination
                if hit_pages != len(batch_hashes):
                    all_success = False
                if prefix_keys and len(prefix_keys) > 0:
                    prefix_keys += batch_hashes
                completed_pages += hit_pages
            ack = PrefetchAck(
                rid=operation.request_id,
                completed_tokens=completed_pages * self.page_size,
                operation=operation,
            )
            self.prefetch_sync_queue.put(ack)
        return completed_pages

    def _page_transfer_kv_batch(
        self,
        operation: PrefetchOperation,
        batch_hashes: List[str],
        batch_host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo,
        kv_derived_transfers: List[PoolTransfer],
    ) -> int:
        """Read a single batch from KV and KV-derived pools (e.g. indexer pool).

        Return the number of hit pages.  If the hits from KV and KV-derived pools differ,
        clamp to the minimal number of hits.

        Here, "batch" means a single unit of L3 read, not a "batch" in model forward.
        """
        # Read from KV pool.
        kv_hits = self.page_get_func(
            operation, batch_hashes, batch_host_indices, extra_info
        )

        # Read from KV-derived sidecar pools, if any.
        sidecar_hits: dict[str, int] = {}
        if len(kv_derived_transfers) > 0:
            current_kv_derived_transfers = [
                PoolTransfer(
                    name=transfer.name,
                    host_indices=batch_host_indices,
                    keys=batch_hashes,
                )
                for transfer in kv_derived_transfers
            ]
            sidecar_results = self.storage_backend.batch_get_v2(
                current_kv_derived_transfers
            )
            sidecar_hits = count_pool_hits(sidecar_results)

        # Clamp to minimal number of hits.
        return min([kv_hits, *sidecar_hits.values()])

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_transfer(operation)

                self.prefetch_sync_queue.put(
                    PrefetchAck(
                        rid=operation.request_id,
                        completed_req=True,
                        operation=operation,
                    )
                )
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        if self.host_memory_mode == "buffer_only":
            # Gate on real pool usage: buffer mode allocates hit-sized, so
            # prefetch_tokens_occupied's requested spans overstate it. Pool
            # state mutates only at scheduler-thread lockstep points, so this
            # stays TP-deterministic. Write staging is the write budget's
            # usage; charging it here would park hits behind its storage drain.
            used = self.mem_pool_host.size - self.mem_pool_host.available_size()
            if self.host_write_staged_tokens_fn is not None:
                used -= self.host_write_staged_tokens_fn()
            return max(0, used) >= self.prefetch_capacity_limit
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []
        page_hashes = self.get_hash_str(
            tokens_to_fetch, last_hash, page_size=self.page_size
        )
        operation.all_hash_values = page_hashes

        for start in range(0, len(page_hashes), STORAGE_BATCH_SIZE):
            batch_hashes = page_hashes[start : start + STORAGE_BATCH_SIZE]
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                if operation.is_terminated():
                    hash_value, storage_hit_count = [], 0
                else:
                    hash_value, storage_hit_count = self._storage_hit_query(operation)
                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce(
                    storage_hit_count_tensor,
                    torch.distributed.ReduceOp.MIN,
                    self.prefetch_hits_sync_groups,
                )
                storage_hit_count = storage_hit_count_tensor.item()

                # Record the TP-synced hit count; the scheduler thread decides
                # at drain time whether to revoke (below threshold) or allocate.
                operation.hash_value = hash_value[
                    : (storage_hit_count // self.page_size)
                ]
                operation.storage_hit_count = storage_hit_count
                self.prefetch_hit_queue.put(operation)

            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.storage_host_pool.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                continue

    def prefetch_sync_thread_func(self):
        """Synchronize prefetch results across all PP and TP ranks."""
        while not self.storage_stop_event.is_set():
            try:
                ack = self.prefetch_sync_queue.get(block=True, timeout=1)
                if ack is None:
                    continue
                self._reduce_prefetch_ack(ack)
                self.ack_prefetch_queue.put(ack)
            except Empty:
                continue

    def _reduce_prefetch_ack(self, ack: PrefetchAck) -> None:
        """Synchronize all ranks to agree on a PrefetchAck."""
        if ack.completed_tokens is not None:
            # Determine the minimal successful prefix of tokens.
            completed_tokens_tensor = torch.tensor(
                ack.completed_tokens, dtype=torch.int
            )
            self._all_reduce(
                completed_tokens_tensor,
                torch.distributed.ReduceOp.MIN,
                self.prefetch_completion_sync_groups,
            )
            ack.completed_tokens = completed_tokens_tensor.item()
