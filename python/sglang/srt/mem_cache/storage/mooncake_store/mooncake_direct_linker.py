from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import Future
from copy import deepcopy
from queue import Empty, Queue

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.runtime_context import (
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


def _shared_pool_suffix(
    *,
    tp_shared: bool,
    tp_rank: int,
    attn_cp_rank: int,
    pp_rank: int,
    dcp_sharded: bool,
    dcp_size: int,
    dcp_rank: int,
) -> str:
    """Key suffix for a pool whose pages are byte-identical across attention TP.

    Dropping the TP axis makes one object serve every attention-TP rank, so
    the store holds one copy instead of one per rank. The dropped axis adds its
    own namespace term so a shared object can never alias an older per-rank one.
    """
    parts = ["tpshared_v1" if tp_shared else f"tp{tp_rank}"]
    if dcp_sharded:
        parts.append(f"dcp{dcp_size}r{dcp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


def _storage_sharing(
    pools,
    *,
    rank_replicated: bool,
    tp_rank: int,
    tp_size: int,
    attn_cp_rank: int,
    pp_rank: int,
    offload_owner: bool,
    dcp_size: int,
    dcp_rank: int,
):
    """Per-pool key suffixes and the pools this rank offloads.

    A pool replicated across attention TP is written by attention-TP rank 0
    only; every rank still reads it, with its own indices, under the shared key.
    A rank-replicated group has no sharded pool, so its group-level suffix
    already drops the tp term and nothing is overridden here.
    A DCP-sharded pool is written by the first DCP block of the TP group,
    one object per DCP rank.
    """
    dcp_sharded = {name for name, pool in pools.items() if pool.dcp_sharded}
    tp_shared = {
        name
        for name, pool in pools.items()
        if not rank_replicated and tp_size > 1 and pool.tp_replicated
    }
    suffixes = {
        name: _shared_pool_suffix(
            tp_shared=name in tp_shared,
            tp_rank=tp_rank,
            attn_cp_rank=attn_cp_rank,
            pp_rank=pp_rank,
            dcp_sharded=name in dcp_sharded,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )
        for name in tp_shared | dcp_sharded
    }
    offload_pools = {
        name
        for name in pools
        if offload_owner
        and (
            name not in tp_shared or tp_rank < (dcp_size if name in dcp_sharded else 1)
        )
    }
    return suffixes, offload_pools, tp_shared


class MooncakeDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        extra_config=None,
        storage=None,
        forward_stream=None,
    ):
        self.forward_stream = forward_stream
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

        parallel = get_parallel()
        tp_rank = 0
        tp_size = parallel.tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        extra_config = deepcopy(extra_config or {})
        self.load_mode = extra_config.pop("load_mode", "layer_wise")
        # TODO: unlike `HiCacheController._generate_storage_config`, this sets no
        # `dp_rank`. Aligning the storage-config producers changes persisted
        # storage keys, so it waits for a storage-key format change.
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
        # An MLA-family pool holds identical pages on every attention-TP rank.
        # KDA state stays in its own rank namespace.
        self.storage.pool_suffixes, self.offload_pools, tp_shared = _storage_sharing(
            self.pools,
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            tp_size=tp_size,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
            offload_owner=self.offload_owner,
            dcp_size=parallel.attn_dcp_size,
            dcp_rank=parallel.attn_dcp_rank,
        )
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s, "
            "tp_shared=%s, offload_pools=%s, pool_suffixes=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
            sorted(tp_shared),
            sorted(self.offload_pools),
            dict(sorted(self.storage.pool_suffixes.items())),
        )

        self.register_buffers()
        self.mla_broadcaster = None
        self.tp_group = tp_group
        self.broadcast_loads = {}
        if (
            self.load_mode == "layer_wise"
            and server_args.enable_linker_mla_dedup
            and rank_replicated
            and tp_size > 1
        ):
            self.mla_broadcaster = LinkerMLADedupBroadcaster.build(
                self.pool_group, params.tp_cache_group, params.attn_tp_cache_group
            )
            logger.info(
                "MLA linker rank-0 loading enabled: tp_rank=%d/%d", tp_rank, tp_size
            )
        self.layer_done_counter = LayerWiseLoadCounter(
            self.num_layers,
            self._broadcast_loaded_layers if self.mla_broadcaster else None,
        )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        # rid -> when load() queued it; filled only with log_load_timing.
        self.load_queued_at: dict[str, float] = {}
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object, dict[str, float]]
            | tuple[Future[bool], list[PoolTransfer], object]
            | None
        ] = Queue()
        self.offload_queue: Queue[
            tuple[list[PoolTransfer], int, list[object], str | None] | None
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
                "Mooncake direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                restorable[-1],
                len(restorable),
            )
        return restorable

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
        if self.log_load_timing:
            self.load_queued_at[rid] = time.monotonic()
        return True

    def prefetch(self, transfers: list[PoolTransfer]) -> Future[bool] | None:
        if self.load_mode != "prefetch" or not self.load_thread.is_alive():
            return None
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return None
        self.freeze_gc_once()
        ready = self.record_transfer_ready()
        future = Future()
        self.load_queue.put((future, expanded, ready))
        return future

    def load_all_layers(self, transfers: list[PoolTransfer]) -> bool:
        for transfer in transfers:
            keys, multiplier = self.storage._get_hybrid_page_component_keys(
                list(transfer.keys), transfer
            )
            keys = self.storage._tag_keys(keys)
            ptrs, sizes = self.storage._build_pool_buffer_meta(
                self.pools[transfer.name],
                transfer.host_indices,
                keys,
                multiplier,
                None,
            )
            received = self.storage._get_batch_zero_copy_impl(keys, ptrs, sizes)
            # Require every byte of every layer, including short reads.
            if list(received) != [sum(s) if isinstance(s, list) else s for s in sizes]:
                return False
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        self.load_queued_at.pop(rid, None)
        return self.pending_loads.pop(rid, None) is not None

    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        # An MLA dedup peer's layers complete at start; its data arrives through
        # the broadcast on the forward stream, ordered before any later reuse.
        return self.layer_done_counter.finish(counter_index)

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        broadcaster = self.mla_broadcaster
        if broadcaster is not None:
            # Compare the logical load order (not local slots), including empty
            # batches, before deciding collectively whether this batch can
            # broadcast.
            plan = [
                (rid, [(t.name, list(t.keys)) for t in transfers])
                for rid, transfers in self.pending_loads.items()
            ]
            digest = hashlib.sha256(repr(plan).encode()).digest()
            digests = [None] * torch.distributed.get_world_size(self.tp_group)
            torch.distributed.all_gather_object(digests, digest, group=self.tp_group)
            if any(other != digest for other in digests):
                logger.warning("MLA linker load plans differ; using all-rank reads.")
                broadcaster = None
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}
        queued_at = {
            rid: self.load_queued_at.pop(rid)
            for rid in pending
            if rid in self.load_queued_at
        }

        counter_index = self.layer_done_counter.update_producer()
        if broadcaster is not None:
            indices = {}
            for transfers in pending.values():
                for transfer in transfers:
                    indices.setdefault(transfer.name, []).append(transfer.host_indices)
            prepared = broadcaster.prepare_broadcast(
                {name: torch.cat(parts) for name, parts in indices.items()},
                device_module.current_stream(),
            )
            if not broadcaster.is_src:
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
        ready = self.record_transfer_ready()
        if broadcaster is not None:
            self.broadcast_loads[counter_index] = (0, prepared, ready)
        if broadcaster is None or broadcaster.is_src:
            self.load_queue.put((counter_index, pending, ready, queued_at))
        self.stats["load"] += len(pending)
        return counter_index

    def _broadcast_loaded_layers(self, index: int, threshold: int) -> None:
        batch = self.broadcast_loads.get(index)
        if batch is None:
            return
        first, prepared, ready = batch
        if first == 0:
            ready.wait(device_module.current_stream())
        # KV access may wait several times per layer, or skip sparse layers.
        # Launch each broadcast exactly once, in order, on the forward stream.
        for layer in range(first, threshold + 1):
            self.mla_broadcaster.broadcast_loaded_layer(layer, prepared)
        if threshold == self.num_layers - 1:
            del self.broadcast_loads[index]
        else:
            self.broadcast_loads[index] = (
                max(first, threshold + 1),
                prepared,
                ready,
            )

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                # Layer-wise tasks also carry each request's queue time.
                completion, pending, ready = task[:3]
                prefetch = isinstance(completion, Future)
                if prefetch and not completion.set_running_or_notify_cancel():
                    continue
                try:
                    ready.synchronize()
                    if prefetch:
                        completion.set_result(self.load_all_layers(pending))
                    else:
                        queued_at = task[3]
                        self.load_layer_wise(
                            completion,
                            list(pending.values()),
                            [(rid, queued_at.get(rid)) for rid in pending],
                        )
                except BaseException as error:
                    if prefetch:
                        completion.set_result(False)
                    else:
                        self.layer_done_counter.fail(completion, error)
                    logger.exception(
                        "Mooncake %s load batch failed",
                        "prefetch" if prefetch else "layer-wise",
                    )
            finally:
                self.load_queue.task_done()

    def _get_layer_ranges(self, keys, ptrs, sizes, offsets):
        store = self.storage.store
        result = store.batch_get_into_multi_buffer_ranges(keys, ptrs, sizes, offsets)
        if result is None or isinstance(result, int):
            return result
        result = list(result)
        if len(result) != len(keys):
            return result
        expected = [sum(item) for item in sizes]
        # Mooncake ErrorCode::LEASE_EXPIRED. Session leases can expire between
        # layers of a long prefix. The synchronous read has drained its writers;
        # reacquire only those immutable content-addressed objects and retry
        # their current layer once. Other failures still abort the request.
        expired = [i for i, value in enumerate(result) if value == -707]
        if not expired or any(
            value not in (want, -707) for value, want in zip(result, expected)
        ):
            return result
        expired_keys = [keys[i] for i in expired]
        renewed = list(store.batch_get_session_start(expired_keys))
        if renewed != [0] * len(expired_keys):
            raise RuntimeError(
                f"Mooncake get session renewal failed: keys={len(expired_keys)}, "
                f"result_sample={renewed[:8]}"
            )
        retried = store.batch_get_into_multi_buffer_ranges(
            expired_keys,
            [ptrs[i] for i in expired],
            [sizes[i] for i in expired],
            [offsets[i] for i in expired],
        )
        if retried is None or isinstance(retried, int):
            return retried
        retried = list(retried)
        if len(retried) != len(expired):
            raise RuntimeError("Mooncake renewed range get returned wrong result count")
        for index, value in zip(expired, retried):
            result[index] = value
        logger.debug("Refreshed Mooncake read leases for %d objects", len(expired))
        return result

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[list[PoolTransfer]],
        requests: list[tuple[str, float | None]] | None = None,
    ) -> None:
        """``requests`` names each entry of ``request_transfers`` and when its
        load was queued; with ``log_load_timing`` each one is logged once all
        layers land."""
        started = []
        began = time.monotonic()
        timed = self.log_load_timing and requests is not None
        # Per request: the key range each pool's batch holds for it.
        spans: list[list[tuple[PoolName, int, int]]] = []
        try:
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            for transfers in request_transfers:
                span = []
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, _ = self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                    first = len(keys)
                    keys.extend(self.storage._tag_keys(component_keys))
                    span.append((transfer.name, first, len(keys)))
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )
                spans.append(span)
            loaded = [0] * len(request_transfers)
            for keys, _ in batches.values():
                # Even a partial start failure may have installed some sessions.
                started.append(keys)
                result = self.storage.store.batch_get_session_start(keys)
                if list(result) != [0] * len(keys):
                    raise RuntimeError(
                        f"Mooncake get session start failed: keys={len(keys)}, "
                        f"result_sample={list(result)[:8]}"
                    )

            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    result = self._get_layer_ranges(keys, ptrs, sizes, offsets)
                    expected = [sum(item) for item in sizes]
                    if (
                        result is None
                        or isinstance(result, int)
                        or list(result) != expected
                    ):
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}: objects={len(keys)}, "
                            f"transferred_sample={result[:8] if isinstance(result, list) else result}, "
                            f"expected_sample={expected[:8]}"
                        )
                    if timed:
                        for index, span in enumerate(spans):
                            for pool, first, last in span:
                                if pool == name:
                                    loaded[index] += sum(expected[first:last])
                self.layer_done_counter.complete(counter_index, layer)
            if timed:
                self._log_load_timing(
                    request_transfers, requests, loaded, began, time.monotonic()
                )
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            for keys in started:
                try:
                    self.storage.store.batch_get_session_end(keys)
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load session cleanup failed")

    @staticmethod
    def _log_load_timing(request_transfers, requests, loaded, began, landed):
        """``wait``: queued until this thread started reading, which includes
        the rest of admission and the batch launch; ``xfer``: until the last
        layer landed. Bytes are this rank's share only."""
        rate = sum(loaded) / max(landed - began, 1e-9) / 1e9
        for transfers, (rid, queued_at), nbytes in zip(
            request_transfers, requests, loaded
        ):
            pages = next((len(t.keys) for t in transfers if t.name == PoolName.KV), 0)
            queued_at = began if queued_at is None else queued_at
            logger.info(
                "Mooncake linker load rid=%s pages=%d bytes=%d wait=%.0fms "
                "xfer=%.0fms total=%.0fms batch_reqs=%d batch_rate=%.2fGB/s",
                rid,
                pages,
                nbytes,
                (began - queued_at) * 1e3,
                (landed - began) * 1e3,
                (landed - queued_at) * 1e3,
                len(request_transfers),
                rate,
            )

    def offload(
        self, transfers: list[PoolTransfer], session_id: str | None = None
    ) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        expanded = [t for t in expanded if t.name in self.offload_pools]
        if not expanded:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        ready = self.record_transfer_ready()
        self.offload_queue.put((expanded, tokens, ready, session_id))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                expanded, tokens, ready, session_id = task
                ready.synchronize()
                extra_info = (
                    HiCacheStorageExtraInfo(extra_info={"session_id": session_id})
                    if session_id
                    else None
                )
                results = self.storage.batch_set_v2(expanded, extra_info)
                success = all(all(pool_results) for pool_results in results.values())
                if success:
                    self.stats["offload"] += 1
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

    def reset(self) -> None:
        self.pending_loads.clear()
        self.load_queued_at.clear()
        # Both load modes must stop writing before any destination is recycled.
        self.load_queue.join()
        self.offload_queue.join()
        self.broadcast_loads.clear()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
        if self.mla_broadcaster is not None:
            self.mla_broadcaster.destroy()
