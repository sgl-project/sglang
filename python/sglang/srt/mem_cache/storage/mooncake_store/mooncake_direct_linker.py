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
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import (
    get_memory,
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _get_mooncake_storage_metrics_dp_rank(server_args, params) -> int:
    if getattr(server_args, "enable_dp_attention", False):
        from sglang.srt.layers.dp_attention import get_attention_dp_rank

        return get_attention_dp_rank()
    return getattr(params, "dp_rank", None) or 0


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
        self.enable_page_wise_load = server_args.mooncake_enable_page_wise_load
        self.page_wise_load_threshold = (
            server_args.mooncake_page_wise_load_threshold
        )
        if self.page_wise_load_threshold <= 0:
            raise ValueError(
                "--mooncake-page-wise-load-threshold must be positive, got "
                f"{self.page_wise_load_threshold}."
            )
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
            dp_rank=getattr(params, "dp_rank", None),
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(
                storage_config,
                mem_pool=None,
                enable_client_http_server=params.enable_metrics,
            )
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

        self.storage_metrics_collector = None
        if params.enable_metrics:
            labels = {
                "storage_backend": "mooncake_direct",
                "tp_rank": tp_rank,
                "dp_rank": _get_mooncake_storage_metrics_dp_rank(
                    server_args, params
                ),
                "pp_rank": params.pp_rank,
                "pp_size": params.pp_size,
                "attn_cp_rank": params.attn_cp_rank,
                "attn_cp_size": params.attn_cp_size,
            }
            if server_args.extra_metric_labels:
                labels.update(server_args.extra_metric_labels)
            collector_cls = resolve_collector_class(
                STAT_LOGGER_ROLE_STORAGE, StorageMetricsCollector
            )
            self.storage_metrics_collector = collector_cls(labels=labels)

        dfs_replica_num = getattr(
            self.storage,
            "dfs_replica_num",
            (extra_config or {}).get("dfs_replica_num", 1),
        )
        self.backup_metric_source = (
            "dfs" if int(dfs_replica_num) > 0 else "local_disk"
        )
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
        self.pending_load_metrics: dict[str, tuple[int, dict[str, int], float]] = {}
        self.request_source_callbacks: dict[str, object] = {}
        self.request_time_stats: dict[str, object] = {}
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        self.completed_loads: Queue[list[str]] = Queue()
        self.offload_queue: Queue[
            tuple[list[PoolTransfer], int, str, float, object] | None
        ] = Queue()
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
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

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
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
        if restorable:
            logger.info(
                "Mooncake direct linker lookup hit: rid=%s pages=%d "
                "candidates=%d duration=%.2fms",
                rid,
                restorable[-1],
                len(restorable),
                (time.perf_counter() - started) * 1000,
            )
        return restorable

    def set_request_time_stats(self, rid: str, time_stats) -> None:
        self.request_time_stats[rid] = time_stats

    def set_request_storage_source_callback(self, rid: str, callback) -> None:
        self.request_source_callbacks[rid] = callback

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        logical_pages = {
            page_key for transfer in expanded for page_key in transfer.keys
        }
        self.pending_load_metrics[rid] = (
            len(logical_pages) * self.page_size,
            {},
            time.perf_counter(),
        )
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # Already-published loads cannot be safely canceled without tree rollback.
        return False

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

        started = time.perf_counter()
        for rid in pending:
            time_stats = getattr(self, "request_time_stats", {}).get(rid)
            if time_stats is not None:
                time_stats.set_direct_load_start_time(started)

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
                    ready_event.synchronize()
                    self.load_layer_wise(counter_index, list(pending.items()))
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
                finally:
                    finished = time.perf_counter()
                    for rid in pending:
                        time_stats = getattr(self, "request_time_stats", {}).pop(
                            rid, None
                        )
                        if time_stats is not None:
                            time_stats.set_direct_load_finish_time(finished)
                    self.completed_loads.put(list(pending))
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[tuple[str, list[PoolTransfer]]],
    ) -> None:
        started = []
        request_success = {rid: False for rid, _ in request_transfers}
        try:
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            batch_rids: dict[PoolName, list[str]] = {}
            batch_page_refs: dict[PoolName, list[tuple[str, str]]] = {}
            for rid, transfers in request_transfers:
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, key_multiplier = (
                        self.storage._get_hybrid_page_component_keys(
                            list(transfer.keys), transfer
                        )
                    )
                    tagged_keys = self.storage._tag_keys(component_keys)
                    keys.extend(tagged_keys)
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )
                    batch_rids.setdefault(transfer.name, []).extend(
                        [rid] * len(tagged_keys)
                    )
                    refs = batch_page_refs.setdefault(transfer.name, [])
                    for page_key in transfer.keys:
                        refs.extend([(rid, page_key)] * key_multiplier)

            request_page_sources: dict[str, dict[str, set[str | None]]] = {
                rid: {} for rid, _ in request_transfers
            }
            for name, (keys, _) in batches.items():
                start_with_sources = getattr(
                    self.storage.store,
                    "batch_get_session_start_with_sources",
                    None,
                )
                if start_with_sources is None:
                    result = list(self.storage.store.batch_get_session_start(keys))
                    sources: list[str | None] = [None] * len(keys)
                else:
                    result, sources = start_with_sources(keys)
                    result = list(result)
                    sources = list(sources)
                    if len(sources) != len(keys):
                        sources = [None] * len(keys)
                if result != [0] * len(keys):
                    raise RuntimeError(
                        f"Mooncake get session start failed: keys={len(keys)}, "
                        f"results={result}"
                    )
                started.append(keys)
                refs = batch_page_refs.get(name, ())
                if len(refs) != len(keys):
                    sources = [None] * len(keys)
                for (rid, page_key), source in zip(refs, sources):
                    request_page_sources[rid].setdefault(page_key, set()).add(source)

            for rid, page_sources in request_page_sources.items():
                metric = getattr(self, "pending_load_metrics", {}).get(rid)
                if metric is None:
                    continue
                total_tokens, _, metric_started = metric
                source_tokens: dict[str, int] = {}
                for sources in page_sources.values():
                    source = (
                        "dfs"
                        if "dfs" in sources
                        else "local_disk"
                        if "local_disk" in sources
                        else None
                    )
                    if source is not None:
                        source_tokens[source] = (
                            source_tokens.get(source, 0) + self.page_size
                        )
                self.pending_load_metrics[rid] = (
                    total_tokens,
                    source_tokens,
                    metric_started,
                )
                source = (
                    "dfs"
                    if source_tokens.get("dfs", 0) > 0
                    else "local_disk"
                    if source_tokens.get("local_disk", 0) > 0
                    else None
                )
                callback = getattr(self, "request_source_callbacks", {}).get(rid)
                if callback is not None:
                    try:
                        callback(source, total_tokens if source is not None else 0)
                    except BaseException:
                        logger.warning(
                            "Failed to publish Mooncake cache source for rid=%s.",
                            rid,
                            exc_info=True,
                        )

            if self.enable_page_wise_load and any(
                len(keys) >= self.page_wise_load_threshold
                for keys, _ in batches.values()
            ):
                request_success = self._load_page_wise(
                    counter_index, request_transfers, batches, batch_rids
                )
                if not all(request_success.values()):
                    request_success = {rid: False for rid, _ in request_transfers}
                    raise RuntimeError(
                        "Mooncake page-wise load failed for one or more requests."
                    )
                return

            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    rids = batch_rids.get(name, [None] * len(keys))
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
                        transferred = (
                            None
                            if result is None or isinstance(result, int)
                            else list(result)
                        )
                        failed_objects = []
                        for index, key in enumerate(keys):
                            actual = (
                                result
                                if result is None or isinstance(result, int)
                                else transferred[index]
                                if index < len(transferred)
                                else None
                            )
                            wanted = expected[index]
                            if actual != wanted:
                                failed_objects.append(
                                    {
                                        "key": key,
                                        "rid": (
                                            rids[index]
                                            if index < len(rids)
                                            else None
                                        ),
                                        "transferred": actual,
                                        "expected": wanted,
                                    }
                                )
                        logger.error(
                            "Mooncake lookup/session succeeded but range get "
                            "failed: rids=%s pool=%s layer=%d failed_objects=%s",
                            [rid for rid, _ in request_transfers],
                            name,
                            layer,
                            failed_objects,
                        )
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}, failed_objects={len(failed_objects)}."
                        )
                self.layer_done_counter.complete(counter_index, layer)
            request_success = {rid: True for rid, _ in request_transfers}
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            for rid, _ in request_transfers:
                self._finish_l4_metric(
                    "prefetch", rid, request_success.get(rid, False)
                )
            for keys in started:
                try:
                    self.storage.store.batch_get_session_end(keys)
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load session cleanup failed")
            finished = time.perf_counter()
            for rid, _ in request_transfers:
                time_stats = getattr(self, "request_time_stats", {}).pop(rid, None)
                if time_stats is not None:
                    time_stats.set_direct_load_finish_time(finished)

    def _load_page_wise(
        self,
        counter_index: int,
        request_transfers: list[tuple[str, list[PoolTransfer]]],
        batches: dict[PoolName, tuple[list[str], list[int]]],
        batch_rids: dict[PoolName, list[str]],
    ) -> dict[str, bool]:
        """Load complete pages before releasing their layers to the consumer."""
        request_success = {rid: True for rid, _ in request_transfers}

        # Keep failure attribution conservative.  The batch lists are expected
        # to stay aligned with ``batch_rids``; if a backend/component violates
        # that contract, a failed object cannot be safely assigned to one rid.
        # Treat the whole batch as failed instead of accidentally completing the
        # layer counter for an affected request.
        mapping_valid = all(
            len(batch_rids.get(name, ())) == len(keys)
            for name, (keys, _) in batches.items()
        )
        if not mapping_valid:
            logger.error(
                "Mooncake page-wise request attribution mismatch; "
                "marking the whole batch failed."
            )
            raise ValueError("Mooncake page-wise request attribution mismatch.")

        all_keys: list[str] = []
        all_ptrs: list[list[int]] = []
        all_sizes: list[list[int]] = []
        all_offsets: list[list[int]] = []
        all_rids: list[str | None] = []
        all_pools: list[PoolName] = []
        for name, (keys, locations) in batches.items():
            ptrs: list[list[int]] = [[] for _ in keys]
            sizes: list[list[int]] = [[] for _ in keys]
            offsets: list[list[int]] = [[] for _ in keys]

            for layer in range(self.num_layers):
                meta = self.pools[name].get_prepared_layer_range_meta(locations, layer)
                if meta is None:
                    continue
                layer_ptrs, layer_sizes, layer_offsets = meta
                if not (
                    len(layer_ptrs)
                    == len(layer_sizes)
                    == len(layer_offsets)
                    == len(keys)
                ):
                    raise ValueError(
                        f"Mooncake pool={name} layer={layer} produced "
                        f"{len(layer_ptrs)} range entries for {len(keys)} keys."
                    )
                for index in range(len(keys)):
                    ptrs[index].extend(layer_ptrs[index])
                    sizes[index].extend(layer_sizes[index])
                    offsets[index].extend(layer_offsets[index])

            rids = batch_rids.get(name, [None] * len(keys))
            all_keys.extend(keys)
            all_ptrs.extend(ptrs)
            all_sizes.extend(sizes)
            all_offsets.extend(offsets)
            all_rids.extend(rids)
            all_pools.extend([name] * len(keys))

        lengths = {
            "keys": len(all_keys),
            "ptrs": len(all_ptrs),
            "sizes": len(all_sizes),
            "offsets": len(all_offsets),
            "rids": len(all_rids),
            "pools": len(all_pools),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Mooncake page-wise aggregated metadata mismatch: {lengths}."
            )

        # Mooncake's range API is key-major and does not take a pool argument,
        # so differently suffixed physical-pool objects can share one call.
        pool_counts = {
            str(name): len(keys) for name, (keys, _) in batches.items()
        }
        unique_rids = sorted({rid for rid in all_rids if rid is not None})
        result = self.storage.store.batch_get_into_multi_buffer_ranges(
            all_keys, all_ptrs, all_sizes, all_offsets
        )
        expected = [sum(item) for item in all_sizes]
        transferred = (
            None if result is None or isinstance(result, int) else list(result)
        )
        if transferred is None or transferred != expected:
            failed_objects = []
            for index, key in enumerate(all_keys):
                actual = (
                    result
                    if result is None or isinstance(result, int)
                    else transferred[index]
                    if index < len(transferred)
                    else None
                )
                wanted = expected[index] if index < len(expected) else None
                if actual != wanted:
                    rid = all_rids[index] if index < len(all_rids) else None
                    pool = all_pools[index] if index < len(all_pools) else None
                    if rid is None:
                        request_success = {
                            request_rid: False
                            for request_rid, _ in request_transfers
                        }
                    else:
                        request_success[rid] = False
                    failed_objects.append(
                        {
                            "key": key,
                            "rid": rid,
                            "pool": pool,
                            "transferred": actual,
                            "expected": wanted,
                        }
                    )
            logger.error(
                "Mooncake page-wise aggregated range get failed: "
                "failed_objects=%s",
                failed_objects,
            )

        # Page-wise loading intentionally gives up layer overlap: no layer is
        # made visible to the model until every complete-page read succeeds.
        if all(request_success.values()):
            for layer in range(self.num_layers):
                self.layer_done_counter.complete(counter_index, layer)
        return request_success

    def _finish_l4_metric(self, operation: str, rid: str, success: bool) -> None:
        metric = getattr(self, "pending_load_metrics", {}).pop(rid, None)
        callback = getattr(self, "request_source_callbacks", {}).pop(rid, None)
        if not success and callback is not None:
            try:
                callback(None, 0)
            except BaseException:
                logger.warning(
                    "Failed to clear Mooncake cache source for rid=%s.",
                    rid,
                    exc_info=True,
                )
        if metric is None:
            return
        total_tokens, source_tokens, started = metric
        if success and total_tokens > 0:
            recorder = getattr(
                getattr(self.storage, "store", None),
                "record_prefetched_tokens",
                None,
            )
            if recorder is not None:
                try:
                    recorder(total_tokens)
                except BaseException:
                    logger.warning(
                        "Failed to record Mooncake prefetched token metric.",
                        exc_info=True,
                    )
        duration = time.perf_counter() - started
        for source, tokens in source_tokens.items():
            self._log_l4_metric(operation, source, tokens, duration, success)

    def _log_l4_metric(
        self,
        operation: str,
        source: str,
        tokens: int,
        duration: float,
        success: bool,
    ) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None:
            return
        try:
            if operation == "prefetch":
                collector.log_l4_prefetch(source, tokens, duration, success)
            else:
                collector.log_l4_backup(source, tokens, duration, success)
        except BaseException:
            logger.warning(
                "Failed to record SGLang L4 %s metrics.",
                operation,
                exc_info=True,
            )

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
        source = self.backup_metric_source
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_queue.put(
            (expanded, tokens, source, time.perf_counter(), ready_event)
        )
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            metric_recorded = False
            try:
                if task is None:
                    return
                expanded, tokens, source, started, ready_event = task
                ready_event.synchronize()
                results = self.storage.batch_set_v2(expanded)
                success = all(all(pool_results) for pool_results in results.values())
                self._log_l4_metric(
                    "backup",
                    source,
                    tokens,
                    time.perf_counter() - started,
                    success,
                )
                metric_recorded = True
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                if task is not None and not metric_recorded:
                    _, tokens, source, started, _ = task
                    self._log_l4_metric(
                        "backup",
                        source,
                        tokens,
                        time.perf_counter() - started,
                        False,
                    )
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        self.pending_loads.clear()
        self.pending_load_metrics.clear()
        self.request_source_callbacks.clear()
        self.load_queue.join()
        self.offload_queue.join()
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
        self.request_time_stats.clear()
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
