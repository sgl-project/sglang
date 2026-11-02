"""LMCache MP backend for the UnifiedRadixCache external linker."""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import Future
from itertools import chain
from typing import Any, Optional

import torch
from lmcache.integration.sglang.lmcache_mp_metadata import (
    LMCacheLinkerCollectedPrefix,
    LMCacheLinkerPendingStore,
    LMCacheLinkerQueuedLoad,
    LMCacheLinkerStartedLoadBatch,
)
from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    UnifiedLMCacheMPConnector,
)

from sglang.srt.arg_groups.model_override_base import model_config_of
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.runtime_context import get_memory, get_parallel
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


class _LookupFuture(Future[list[int]]):
    """Poll an LMCache lookup without blocking the scheduler thread."""

    def __init__(self, linker, rid: str, operation: Any) -> None:
        super().__init__()
        self._linker = linker
        self._rid = rid
        self._operation = operation
        self._discard = False

    def cancel(self) -> bool:
        self._discard = True
        if super().done():
            self._linker._discard_completed_lookup(self._rid, self._operation)
            return False
        # No device transfer has started, so the client can retire its session
        # immediately. The server releases any late lookup locks with it.
        self._linker._finish_lookup(self._rid, self._operation)
        self._linker._lookup_futures.pop(self._rid, None)
        return super().cancel()

    def done(self) -> bool:
        if not super().done():
            try:
                total_hit = self._linker.connector.poll_lookup(self._operation)
            except Exception:
                logger.exception("LMCache linker lookup failed for %s", self._rid)
                self._complete(None)
            else:
                if total_hit is not None:
                    self._complete(total_hit)
        return super().done()

    def result(self, timeout=None) -> list[int]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.done():
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning("LMCache linker lookup timed out for %s", self._rid)
                self._complete(None)
                break
            time.sleep(0.001)
        return super().result(timeout=0)

    def _complete(self, total_hit: Optional[int]) -> None:
        if super().done():
            return
        if self._discard or total_hit is None:
            self._linker._finish_lookup(self._rid, self._operation)
            restorable = []
        else:
            restorable = self._linker._consume_lookup_result(
                self._rid, self._operation, total_hit
            )
        self._linker._lookup_futures.pop(self._rid, None)
        self.set_result(restorable)


class _LoadFuture(Future[bool]):
    """Adapt an LMCache retrieve operation to the linker Future contract."""

    def __init__(self, connector: Any, rid: str, operation: Any, on_done) -> None:
        super().__init__()
        self._connector = connector
        self._rid = rid
        self._operation = operation
        self._on_done = on_done

    def cancel(self) -> bool:
        # The H2D is already submitted and still owns its destination slots.
        return False

    def done(self) -> bool:
        if not super().done():
            try:
                if self._operation.query():
                    self._complete()
            except Exception:
                logger.exception(
                    "LMCache linker prefetch polling failed for %s", self._rid
                )
                self._complete(success=False)
        return super().done()

    def result(self, timeout=None) -> bool:
        if not super().done():
            try:
                self._operation.future.result(timeout=timeout)
                self._complete()
            except Exception:
                logger.exception("LMCache linker prefetch failed for %s", self._rid)
                self._complete(success=False)
        return super().result(timeout=0)

    def _complete(self, success: Optional[bool] = None) -> None:
        if super().done():
            return
        if success is None:
            try:
                success = self._connector.complete_load(
                    self._operation, synchronize=False
                )
            except Exception:
                logger.exception("LMCache linker prefetch failed for %s", self._rid)
                success = False
        self._connector.finish_request(self._rid)
        self._on_done(self._rid)
        self.set_result(bool(success))


class LMCacheDirectLinker(UnifiedCacheLinker):
    """Connect UnifiedRadixCache device pools to an LMCache MP server."""

    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        cache,
        extra_config=None,
        _storage=None,
        forward_stream=None,
    ) -> None:
        if cache.tree_core.is_eagle:
            raise NotImplementedError(
                "The LMCache external linker does not support EAGLE/bigram trees"
            )
        parallel = get_parallel()
        self.cache = cache
        self.forward_stream = forward_stream
        self._mamba_component = cache.components.get(ComponentType.MAMBA)
        if (
            self._mamba_component is not None
            and params.req_to_token_pool.mamba_ckpt_pool is not None
        ):
            raise NotImplementedError(
                "The LMCache external linker does not support int8 Mamba checkpoints"
            )
        self.connector = UnifiedLMCacheMPConnector(
            config_file=get_memory().lmcache_config_file,
            model_config=model_config_of(server_args),
            tp_size=parallel.tp_size,
            tp_rank=parallel.tp_rank,
            tp_group=params.tp_cache_group,
            pp_size=params.pp_size,
            pp_rank=params.pp_rank,
            pp_group=params.pp_cache_group,
            page_size=cache.page_size,
            token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
            req_to_token_pool=params.req_to_token_pool,
            tree_components=cache.tree_components,
            mamba_component=self._mamba_component,
            sliding_window_size=cache.sliding_window_size,
        )

        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pools = {PoolName.KV: kvcache}
        if ComponentType.SWA in cache.tree_components:
            self.pools[PoolName.SWA] = kvcache
        if self._mamba_component is not None:
            self.pools[PoolName.MAMBA] = params.req_to_token_pool.mamba_pool
        self.num_layers = self._resolve_num_layers(kvcache, params.req_to_token_pool)
        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)

        self._lookups: dict[str, Any] = {}
        self._lookup_futures: dict[str, _LookupFuture] = {}
        self._queued_loads: dict[str, LMCacheLinkerQueuedLoad] = {}
        self._load_batches: dict[int, LMCacheLinkerStartedLoadBatch] = {}
        self._prefetch_futures: dict[str, Future[bool]] = {}
        self._pending_stores: deque[LMCacheLinkerPendingStore] = deque()
        self._completed_stores: deque[bool] = deque()
        self._offload_sequence = 0
        self._cached_prefixes: dict[int, list[int]] = {}

    def lookup(
        self,
        rid: str,
        key: RadixKey,
        device_hit_len: int,
        transfers: list[PoolTransfer],
    ) -> list[int]:
        del transfers
        if rid in self._lookup_futures or rid in self._prefetch_futures:
            return []
        previous = self._lookups.pop(rid, None)
        if previous is not None:
            self._finish_lookup(rid, previous)

        token_ids = list(key.raw_token_ids()[: len(key)])
        operation = self.connector.submit_lookup(
            rid,
            token_ids,
            local_hit_tokens=device_hit_len,
            cache_salt=self.connector.build_cache_salt(key.cache_salt, key.extra_key),
        )
        deadline = time.monotonic() + self.connector.operation_timeout
        total_hit = self.connector.poll_lookup(operation)
        while total_hit is None:
            timed_out = torch.tensor(
                [int(time.monotonic() >= deadline)], dtype=torch.int32, device="cpu"
            )
            self.connector.parallel_all_reduce(
                timed_out, torch.distributed.ReduceOp.MAX
            )
            if timed_out.item():
                break
            time.sleep(0.001)
            total_hit = self.connector.poll_lookup(operation)
        if total_hit is None:
            logger.warning("LMCache linker lookup timed out for %s", rid)
            self._finish_lookup(rid, operation)
            return []

        return self._consume_lookup_result(rid, operation, total_hit)

    def submit_lookup(
        self,
        rid: str,
        key: RadixKey,
        device_hit_len: int,
        transfers: list[PoolTransfer],
    ) -> Future[list[int]] | None:
        del transfers
        if rid in self._lookup_futures or rid in self._prefetch_futures:
            return None
        previous = self._lookups.pop(rid, None)
        if previous is not None:
            self._finish_lookup(rid, previous)

        token_ids = list(key.raw_token_ids()[: len(key)])
        try:
            operation = self.connector.submit_lookup(
                rid,
                token_ids,
                local_hit_tokens=device_hit_len,
                cache_salt=self.connector.build_cache_salt(
                    key.cache_salt, key.extra_key
                ),
            )
        except Exception:
            logger.exception("LMCache linker lookup submission failed for %s", rid)
            return None

        future = _LookupFuture(self, rid, operation)
        self._lookup_futures[rid] = future
        return future

    def _consume_lookup_result(
        self, rid: str, operation: Any, total_hit: int
    ) -> list[int]:
        local_hit = operation.local_hit_tokens
        release_end = min(
            total_hit,
            local_hit // self.connector.chunk_size * self.connector.chunk_size,
        )
        if release_end > operation.lock_start:
            self.connector.free_lookup_locks(
                rid, start=operation.lock_start, end=release_end
            )

        external_tokens = total_hit - local_hit
        if external_tokens <= 0:
            self._finish_lookup(rid, operation)
            return []
        if external_tokens % self.cache.page_size:
            self._finish_lookup(rid, operation)
            return []

        self._lookups[rid] = operation
        return [external_tokens // self.cache.page_size]

    def prefetch(self, rid: str, transfers: list[PoolTransfer]) -> Future[bool] | None:
        operation = self._lookups.get(rid)
        if operation is None:
            return None
        prepared = self._prepare_load(rid, operation, transfers)
        if prepared is None:
            self._lookups.pop(rid, None)
            self._finish_lookup(rid, operation)
            return None

        stream = self.forward_stream or device_module.current_stream()
        try:
            load = self.connector.submit_load(
                operation,
                prepared.group_indices,
                local_hit_tokens=prepared.local_hit_tokens,
                owned_device_indices=prepared.owned_device_indices,
                producer_stream=stream,
            )
        except Exception:
            logger.exception("LMCache linker retrieve submission failed for %s", rid)
            self._lookups.pop(rid, None)
            self._finish_lookup(rid, operation)
            return None

        self._lookups.pop(rid, None)
        future = _LoadFuture(
            self.connector,
            rid,
            load,
            lambda rid: self._prefetch_futures.pop(rid, None),
        )
        self._prefetch_futures[rid] = future
        return future

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        operation = self._lookups.get(rid)
        if operation is None:
            return False
        if rid in self._queued_loads:
            raise RuntimeError(f"LMCache load for rid={rid} is already queued")
        prepared = self._prepare_load(rid, operation, transfers)
        if prepared is None:
            self._lookups.pop(rid, None)
            self._finish_lookup(rid, operation)
            return False
        self._queued_loads[rid] = prepared
        return True

    def start_layer_wise_loading(self) -> int:
        if not self._queued_loads:
            return -1

        pending, self._queued_loads = self._queued_loads, {}
        counter_index = self.layer_done_counter.update_producer()
        stream = self.forward_stream or device_module.current_stream()
        operations = []
        submitted = True
        error: Optional[BaseException] = None

        for rid, queued in pending.items():
            try:
                operation = self.connector.submit_load(
                    queued.lookup,
                    queued.group_indices,
                    local_hit_tokens=queued.local_hit_tokens,
                    owned_device_indices=queued.owned_device_indices,
                    producer_stream=stream,
                )
                operations.append(operation)
                if not self.connector.prepare_load_on_stream(operation, stream):
                    submitted = False
                    error = RuntimeError(f"LMCache server rejected retrieve for {rid}")
            except BaseException as exc:
                logger.exception("LMCache retrieve submission failed for %s", rid)
                submitted = False
                error = exc

        if submitted:
            for layer in range(self.num_layers):
                self.layer_done_counter.complete(counter_index, layer)
        else:
            assert error is not None
            self.layer_done_counter.fail(counter_index, error)

        self._load_batches[counter_index] = LMCacheLinkerStartedLoadBatch(
            rids=list(pending), operations=operations, submitted=submitted
        )
        return counter_index

    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        batch = self._load_batches.pop(counter_index, None)
        if batch is None:
            return counter_index == -1

        success = batch.submitted
        for operation in batch.operations:
            success = self.connector.complete_load(operation) and success
        for rid in batch.rids:
            self._lookups.pop(rid, None)
            self.connector.finish_request(rid)
        return self.layer_done_counter.finish(counter_index) and success

    def cancel_queued_load(self, rid: str) -> bool:
        queued = self._queued_loads.pop(rid, None)
        if queued is None:
            return False
        self._lookups.pop(rid, None)
        self._finish_lookup(rid, queued.lookup)
        return True

    def offload(
        self,
        node_id: int,
        transfers: list[PoolTransfer],
        session_id: Optional[str] = None,
    ) -> bool:
        del session_id
        chunk_size = self.connector.chunk_size
        collected = self._collect_prefix(node_id)
        if collected is None:
            return False
        by_pool = {transfer.name: transfer for transfer in transfers}
        full = by_pool.get(PoolName.KV)
        if full is None or full.device_indices is None:
            return False
        store_start = collected.node_start // chunk_size * chunk_size
        aligned_end = len(collected.token_ids) // chunk_size * chunk_size
        if aligned_end <= store_start:
            return False

        mamba = by_pool.get(PoolName.MAMBA)
        if self._mamba_component is not None:
            node_end = collected.node_start + len(full.device_indices)
            if mamba is None or mamba.device_indices is None or aligned_end != node_end:
                return False

        full_indices = self.cache.path_device_indices(node_id)[store_start:aligned_end]
        if len(full_indices) != aligned_end - store_start:
            return False

        self._offload_sequence += 1
        request_id = f"sglang-linker-{self._offload_sequence}-{collected.terminal_hash}"
        try:
            stream = device_module.current_stream()
            self.record_transfer_ready().wait(stream)
            operation = self.connector.submit_store(
                request_id,
                collected.token_ids,
                self.connector.device_indices_by_group(
                    full_indices,
                    mamba_value=None if mamba is None else mamba.device_indices,
                    mamba_transfer_tokens=aligned_end - store_start,
                ),
                device_indices_start=store_start,
                store_start=store_start,
                cache_salt=self.connector.build_cache_salt(
                    collected.cache_salt, collected.extra_key
                ),
            )
        except Exception:
            logger.exception(
                "LMCache linker store submission failed for %s", request_id
            )
            self.connector.finish_request(request_id)
            return False
        if operation is None:
            self.connector.finish_request(request_id)
            return False

        self._pending_stores.append(LMCacheLinkerPendingStore(request_id, operation))
        return True

    def num_completed_offloads(self) -> int:
        if self._pending_stores:
            ready = self.connector.ready_prefix_count(
                [pending.operation for pending in self._pending_stores]
            )
            for _ in range(ready):
                pending = self._pending_stores.popleft()
                success = self.connector.complete_store(pending.operation)
                self.connector.finish_request(pending.request_id)
                self._completed_stores.append(success)
        return len(self._completed_stores)

    def pop_completed_offload(self) -> bool:
        return self._completed_stores.popleft()

    def reset(self) -> None:
        for future in list(self._lookup_futures.values()):
            if not future.cancel():
                future.result(timeout=self.connector.operation_timeout)
        self._lookup_futures.clear()

        for rid, queued in self._queued_loads.items():
            self._lookups.pop(rid, None)
            self._finish_lookup(rid, queued.lookup)
        self._queued_loads.clear()

        for batch in self._load_batches.values():
            for operation in batch.operations:
                try:
                    operation.future.result(timeout=self.connector.operation_timeout)
                except Exception:
                    pass
                self.connector.complete_load(operation, synchronize=False)
            for rid in batch.rids:
                self._lookups.pop(rid, None)
                self.connector.finish_request(rid)
        self._load_batches.clear()

        for future in list(self._prefetch_futures.values()):
            try:
                future.result(timeout=self.connector.operation_timeout)
            except Exception:
                pass
        self._prefetch_futures.clear()

        while self._pending_stores:
            pending = self._pending_stores.popleft()
            try:
                pending.operation.future.result(
                    timeout=self.connector.operation_timeout
                )
            except Exception:
                pass
            self.connector.complete_store(pending.operation, synchronize=False)
            self.connector.finish_request(pending.request_id)

        for rid, operation in list(self._lookups.items()):
            self._finish_lookup(rid, operation)
        self._lookups.clear()
        self._completed_stores.clear()
        self.layer_done_counter.reset()
        self._cached_prefixes.clear()

    def close(self) -> None:
        self.reset()
        self.connector.close()

    def _prepare_load(
        self, rid: str, operation: Any, transfers: list[PoolTransfer]
    ) -> Optional[LMCacheLinkerQueuedLoad]:
        if operation.total_hit_tokens is None:
            return None
        by_pool = {transfer.name: transfer for transfer in transfers}
        full = by_pool.get(PoolName.KV)
        if full is None or full.device_indices is None:
            return None
        fresh_tokens = len(full.device_indices)
        local_hit = operation.total_hit_tokens - fresh_tokens
        if local_hit < operation.local_hit_tokens:
            return None

        release_end = min(
            operation.total_hit_tokens,
            local_hit // self.connector.chunk_size * self.connector.chunk_size,
        )
        if release_end > operation.lock_start:
            self.connector.free_lookup_locks(
                rid, start=operation.lock_start, end=release_end
            )

        swa = by_pool.get(PoolName.SWA)
        mamba = by_pool.get(PoolName.MAMBA)
        if self._mamba_component is not None and (
            mamba is None or mamba.device_indices is None
        ):
            return None
        load_start = local_hit // self.connector.chunk_size * self.connector.chunk_size
        group_indices = self.connector.device_indices_for_linker_load(
            full.device_indices,
            swa_indices=None if swa is None else swa.device_indices,
            mamba_indices=None if mamba is None else mamba.device_indices,
            mamba_transfer_tokens=operation.total_hit_tokens - load_start,
        )
        return LMCacheLinkerQueuedLoad(
            lookup=operation,
            group_indices=group_indices,
            local_hit_tokens=local_hit,
            owned_device_indices=full.device_indices,
        )

    def _collect_prefix(self, node_id: int) -> Optional[LMCacheLinkerCollectedPrefix]:
        current_node = self.cache.tree_core.snapshot_buffer_backup(
            node_id, pass_prefix_keys=False
        )
        if (
            current_node is None
            or current_node.key.is_bigram
            or not current_node.hash_values
        ):
            return None

        node_tokens = list(current_node.key.raw_token_ids())
        cached_tokens = self._cached_prefixes.pop(current_node.parent_node_id, None)
        if cached_tokens is not None:
            prefix_token_ids = cached_tokens + node_tokens
        else:
            chunks = [node_tokens]
            current = current_node.parent_node_id
            while not self.cache.tree_core.is_root(current):
                snapshot = self.cache.tree_core.snapshot_buffer_backup(
                    current, pass_prefix_keys=False
                )
                if snapshot is None or snapshot.key.is_bigram:
                    return None
                chunks.append(snapshot.key.raw_token_ids())
                current = snapshot.parent_node_id
            chunks.reverse()
            prefix_token_ids = list(chain.from_iterable(chunks))

        # TODO: Bound this cache so interleaved request paths can be retained.
        self._cached_prefixes.clear()
        self._cached_prefixes[node_id] = prefix_token_ids
        total_len = len(prefix_token_ids)
        return LMCacheLinkerCollectedPrefix(
            token_ids=prefix_token_ids,
            node_start=total_len - len(node_tokens),
            extra_key=current_node.key.extra_key,
            cache_salt=current_node.key.cache_salt,
            terminal_hash=current_node.hash_values[-1],
        )

    def _finish_lookup(self, rid: str, operation: Any) -> None:
        if self._lookups.get(rid) is operation:
            self._lookups.pop(rid, None)
        if operation.total_hit_tokens is not None and operation.locks_held:
            self.connector.free_lookup_locks(
                rid, start=operation.lock_start, end=operation.total_hit_tokens
            )
        self.connector.finish_request(rid)

    def _discard_completed_lookup(self, rid: str, operation: Any) -> None:
        if self._lookups.get(rid) is operation:
            self._finish_lookup(rid, operation)

    def _resolve_num_layers(self, kvcache: Any, req_to_token_pool: Any) -> int:
        if self._mamba_component is None:
            return kvcache.layer_num

        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        if isinstance(kvcache, HybridLinearKVPool):
            attention_layers = kvcache.full_attention_layer_id_mapping
        elif isinstance(kvcache, SWAKVPool):
            attention_layers = kvcache.layers_mapping
        else:
            raise NotImplementedError(
                "LMCache Mamba linker does not support KV pool "
                f"{type(kvcache).__name__}"
            )

        layer_ids = {
            *attention_layers,
            *req_to_token_pool.mamba_map,
            *req_to_token_pool.short_conv_pool.layer_map,
        }
        if not layer_ids:
            raise ValueError("LMCache Mamba linker has no local model layers")
        return max(layer_ids) - kvcache.start_layer + 1
