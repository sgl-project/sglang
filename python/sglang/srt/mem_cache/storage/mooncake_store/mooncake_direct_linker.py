from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from queue import Empty, Queue

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import (
    get_memory,
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            raise RuntimeError("Mooncake layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()


class MooncakeDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        storage=None,
    ):
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers

        tp_rank = 0
        tp_size = get_parallel().tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=rank_replicated,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(storage_config, mem_pool=None)
        else:
            self.storage = storage
        self.storage.mem_pool_host = self.pool_group
        self.storage.registered_pools = self.pools
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        self.storage.mla_suffix = storage_suffix
        self.storage.mha_suffix = storage_suffix
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
        )

        self.register_buffers()
        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.load_sessions: dict[str, set[str]] = {}
        self.load_session_refcounts: dict[str, int] = {}
        self.load_session_lock = threading.Lock()
        self.gc_frozen = False
        self.lookup_queue: Queue[tuple[str, list[PoolTransfer]] | None] = Queue()
        self.completed_lookups: Queue[tuple[str, list[int]]] = Queue()
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        self.completed_loads: Queue[list[str]] = Queue()
        self.offload_queue: Queue[tuple[list[PoolTransfer], int, object] | None] = (
            Queue()
        )
        self.offload_results: Queue[bool] = Queue()
        self.started_at = time.monotonic()
        self.stats: dict[str, int | float] = {
            "lookup": 0,
            "lookup_pages": 0,
            "lookup_hit_pages": 0,
            "lookup_seconds": 0.0,
            "load": 0,
            "load_seconds": 0.0,
            "offload": 0,
            "offload_tokens": 0,
            "offload_seconds": 0.0,
            "reserve_lock_seconds": 0.0,
            "reserve_lock_max_seconds": 0.0,
            "reserve_rpc_count": 0,
            "reserve_rpc_seconds": 0.0,
            "reserve_rpc_max_seconds": 0.0,
            "load_ready_count": 0,
            "load_ready_seconds": 0.0,
            "load_ready_max_seconds": 0.0,
        }
        # Metadata-only accounting: do not inspect GPU indices or synchronize
        # just to measure cache admission. These are requested logical page
        # bytes, not actual puts: storage skips keys that already exist.
        self.offload_page_bytes = {
            name: sum(
                size for component in entry.buffer_meta for _, _, size in component
            )
            for name, entry in self.pool_group.entry_map.items()
        }
        for name in self.offload_page_bytes:
            self.stats[f"offload_requested_pages_{name}"] = 0
            self.stats[f"offload_requested_bytes_{name}"] = 0
        self.lookup_thread = threading.Thread(
            target=self.lookup_thread_func,
            daemon=True,
            name=f"mooncake-lookup-tp{tp_rank}",
        )
        self.lookup_thread.start()
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def register_buffers(self) -> None:
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = self.storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register GPU KV buffer with Mooncake, "
                        f"error code: {result}."
                    )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> None:
        self.lookup_queue.put((rid, transfers))

    def lookup_thread_func(self) -> None:
        while True:
            task = self.lookup_queue.get()
            try:
                if task is None:
                    return
                rid, transfers = task
                try:
                    restorable = self._lookup_now(rid, transfers)
                except BaseException:
                    logger.exception("Mooncake lookup failed: rid=%s", rid)
                    restorable = []
                self.completed_lookups.put((rid, restorable))
            finally:
                self.lookup_queue.task_done()

    def _lookup_now(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        started = time.perf_counter()
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            return []
        result = self.storage.batch_exists_v2(page_keys, expanded)
        restorable = result.restorable_prefix_pages or []
        self.stats["lookup"] += 1
        self.stats["lookup_pages"] += len(page_keys)
        self.stats["lookup_hit_pages"] += restorable[-1] if restorable else 0
        self.stats["lookup_seconds"] += time.perf_counter() - started
        logger.debug(
            "Mooncake direct linker lookup: rid=%s first_key=%s "
            "queried_pages=%d pages=%d candidates=%d",
            rid,
            page_keys[0],
            len(page_keys),
            restorable[-1] if restorable else 0,
            len(restorable),
        )
        return restorable

    def num_completed_lookups(self) -> int:
        return self.completed_lookups.qsize()

    def pop_completed_lookup(self) -> tuple[str, list[int]]:
        return self.completed_lookups.get_nowait()

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            self._release_load_session(rid)
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        actual_keys = set(self._component_keys(expanded))
        with self.load_session_lock:
            reserved_keys = self.load_sessions.get(rid)
            if reserved_keys is None or not actual_keys <= reserved_keys:
                self._release_load_session_locked(rid)
                return False
            self._release_load_session_locked(rid, keep=actual_keys)
        self.pending_loads[rid] = expanded
        return True

    def _component_keys(self, transfers: list[PoolTransfer]) -> list[str]:
        keys = []
        for transfer in transfers:
            component_keys, _ = self.storage._get_hybrid_page_component_keys(
                list(transfer.keys), transfer
            )
            keys.extend(self.storage._tag_keys(component_keys))
        return list(dict.fromkeys(keys))

    def reserve_load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        keys = self._component_keys(expanded)
        if not keys:
            return False

        lock_started = time.perf_counter()
        with self.load_session_lock:
            lock_seconds = time.perf_counter() - lock_started
            self.stats["reserve_lock_seconds"] += lock_seconds
            self.stats["reserve_lock_max_seconds"] = max(
                self.stats["reserve_lock_max_seconds"], lock_seconds
            )
            if rid in self.load_sessions:
                raise RuntimeError(
                    f"Mooncake load session for rid={rid} already exists."
                )
            new_keys = [
                key for key in keys if self.load_session_refcounts.get(key, 0) == 0
            ]
            results = []
            if new_keys:
                rpc_started = time.perf_counter()
                try:
                    results = list(self.storage.store.batch_get_session_start(new_keys))
                finally:
                    rpc_seconds = time.perf_counter() - rpc_started
                    self.stats["reserve_rpc_count"] += 1
                    self.stats["reserve_rpc_seconds"] += rpc_seconds
                    self.stats["reserve_rpc_max_seconds"] = max(
                        self.stats["reserve_rpc_max_seconds"], rpc_seconds
                    )
            if len(results) != len(new_keys) or any(result != 0 for result in results):
                started = [key for key, result in zip(new_keys, results) if result == 0]
                self._end_load_sessions(started)
                logger.info(
                    "Mooncake load reservation missed after lookup: rid=%s, "
                    "keys=%d, failed=%d",
                    rid,
                    len(keys),
                    len(new_keys) - len(started),
                )
                return False

            reserved = set(keys)
            self.load_sessions[rid] = reserved
            for key in reserved:
                self.load_session_refcounts[key] = (
                    self.load_session_refcounts.get(key, 0) + 1
                )
        return True

    def _end_load_sessions(self, keys: list[str]) -> None:
        if not keys:
            return
        result = self.storage.store.batch_get_session_end(keys)
        if result not in (0, None):
            logger.warning(
                "Mooncake get session cleanup failed: keys=%d, result=%s",
                len(keys),
                result,
            )

    def _release_load_session_locked(
        self, rid: str, *, keep: set[str] | None = None
    ) -> None:
        keys = self.load_sessions.pop(rid, None)
        if keys is None:
            return
        keep = keep or set()
        retained = keys & keep
        if retained:
            self.load_sessions[rid] = retained

        ending = []
        for key in keys - retained:
            refs = self.load_session_refcounts[key] - 1
            if refs == 0:
                del self.load_session_refcounts[key]
                ending.append(key)
            else:
                self.load_session_refcounts[key] = refs
        self._end_load_sessions(ending)

    def _release_load_session(self, rid: str) -> None:
        with self.load_session_lock:
            self._release_load_session_locked(rid)

    def _release_all_load_sessions(self) -> None:
        with self.load_session_lock:
            for rid in list(self.load_sessions):
                self._release_load_session_locked(rid)

    def cancel_queued_load(self, rid: str) -> bool:
        if rid not in self.pending_loads:
            return False
        del self.pending_loads[rid]
        self._release_load_session(rid)
        return True

    def num_completed_loads(self) -> int:
        return self.completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        return self.completed_loads.get_nowait()

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        counter_index = self.layer_done_counter.update_producer()
        ready_event = device_module.Event()
        ready_event.record()
        self.load_queue.put((counter_index, pending, ready_event))
        self.stats["load"] += len(pending)
        return counter_index

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event = task
                try:
                    ready_started = time.perf_counter()
                    try:
                        ready_event.synchronize()
                    finally:
                        ready_seconds = time.perf_counter() - ready_started
                        self.stats["load_ready_count"] += 1
                        self.stats["load_ready_seconds"] += ready_seconds
                        self.stats["load_ready_max_seconds"] = max(
                            self.stats["load_ready_max_seconds"], ready_seconds
                        )
                    self.load_layer_wise(counter_index, pending)
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
                finally:
                    for rid in pending:
                        self._release_load_session(rid)
                    self.completed_loads.put(list(pending))
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self,
        counter_index: int,
        pending: dict[str, list[PoolTransfer]],
    ) -> None:
        started_at = time.perf_counter()
        try:
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            for transfers in pending.values():
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, _ = self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                    keys.extend(self.storage._tag_keys(component_keys))
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )
            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    result = self.storage.store.batch_get_into_multi_buffer_ranges(
                        keys,
                        ptrs,
                        sizes,
                        offsets,
                    )
                    expected = [sum(item) for item in sizes]
                    if (
                        result is None
                        or isinstance(result, int)
                        or list(result) != expected
                    ):
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}: transferred={result}, "
                            f"expected={expected}"
                        )
                self.layer_done_counter.complete(counter_index, layer)
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            self.stats["load_seconds"] += time.perf_counter() - started_at

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        if not self.offload_owner:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_queue.put((expanded, tokens, ready_event))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                expanded, tokens, ready_event = task
                ready_event.synchronize()
                for transfer in expanded:
                    pages = len(transfer.keys)
                    self.stats[f"offload_requested_pages_{transfer.name}"] += pages
                    self.stats[f"offload_requested_bytes_{transfer.name}"] += (
                        pages * self.offload_page_bytes[transfer.name]
                    )
                started_at = time.perf_counter()
                results = self.storage.batch_set_v2(expanded)
                self.stats["offload_seconds"] += time.perf_counter() - started_at
                success = all(all(pool_results) for pool_results in results.values())
                if success:
                    self.stats["offload"] += 1
                    self.stats["offload_tokens"] += tokens
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def debug_snapshot(self) -> dict[str, int | float]:
        return {
            "uptime_s": round(time.monotonic() - self.started_at, 1),
            "lookup_queue": self.lookup_queue.qsize(),
            "lookup_done": self.completed_lookups.qsize(),
            "load_queue": self.load_queue.qsize(),
            "load_done": self.completed_loads.qsize(),
            "offload_queue": self.offload_queue.qsize(),
            "offload_done": self.offload_results.qsize(),
            **self.stats,
            **self.storage.hybrid_put_stats,
        }

    def reset(self) -> None:
        self.pending_loads.clear()
        self.lookup_queue.join()
        self.load_queue.join()
        self.offload_queue.join()
        self._release_all_load_sessions()
        while True:
            try:
                self.completed_lookups.get_nowait()
            except Empty:
                break
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        while True:
            try:
                self.completed_loads.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.lookup_queue.put(None)
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.lookup_thread.join()
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
