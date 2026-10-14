"""LMCache MP backend for the UnifiedRadixCache external linker."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from itertools import chain, islice
from typing import Any, Optional

import torch
from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    UnifiedLMCacheMPConnector,
)

from sglang.srt.arg_groups.model_override_base import model_config_of
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCachePrefetchParams,
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.runtime_context import get_memory, get_parallel
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


@dataclass
class _QueuedLoad:
    lookup: Any
    group_indices: list[torch.Tensor]
    local_hit_tokens: int
    owned_device_indices: torch.Tensor


@dataclass
class _StartedLoadBatch:
    rids: list[str]
    operations: list[Any]
    submitted: bool


@dataclass
class _PendingStore:
    request_id: str
    operation: Any


@dataclass(frozen=True)
class _CollectedPrefix:
    token_ids: list[int]
    node_start: int
    extra_key: Optional[str]
    cache_salt: Optional[str]
    terminal_hash: str


class LMCacheUnifiedCacheLinker(UnifiedCacheLinker):
    """Connect UnifiedRadixCache device pools to the LMCache MP server."""

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
        if self._mamba_component is not None:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self._prefetches: dict[str, Any] = {}
        self._queued_loads: dict[str, _QueuedLoad] = {}
        self._load_batches: dict[int, _StartedLoadBatch] = {}
        self._pending_stores: deque[_PendingStore] = deque()
        self._completed_stores: deque[bool] = deque()
        self._offload_sequence = 0

    def support_prefetch(self) -> bool:
        return True

    def prefetch(self, params: ExternalCachePrefetchParams) -> bool:
        if params.rid in self._prefetches:
            return False
        token_ids = list(params.key.raw_token_ids()[: len(params.key)])
        operation = self.connector.submit_lookup(
            params.rid,
            token_ids,
            local_hit_tokens=params.device_hit_len,
            cache_salt=self.connector.build_cache_salt(
                params.key.cache_salt, params.key.extra_key
            ),
        )
        self._prefetches[params.rid] = operation
        return True

    def poll_prefetch(self, rid: str) -> Optional[list[int]]:
        operation = self._prefetches.get(rid)
        if operation is None:
            return []
        total_hit = self.connector.poll_lookup(operation)
        if total_hit is None:
            return None

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
            self._prefetches.pop(rid, None)
            self._finish_lookup(rid, operation)
            return []
        return [external_tokens // self.cache.page_size]

    def cancel_prefetch(self, rid: str) -> None:
        operation = self._prefetches.pop(rid, None)
        if operation is not None:
            self._finish_lookup(rid, operation)

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        del rid, transfers
        return []

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        operation = self._prefetches.get(rid)
        if operation is None or operation.total_hit_tokens is None:
            return False
        if rid in self._queued_loads:
            raise RuntimeError(f"LMCache load for rid={rid} is already queued")

        by_pool = {transfer.name: transfer for transfer in transfers}
        full = by_pool.get(PoolName.KV)
        if full is None or full.device_indices is None:
            return False
        fresh_tokens = len(full.device_indices)
        local_hit = operation.total_hit_tokens - fresh_tokens
        if local_hit < operation.local_hit_tokens:
            return False

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
            return False
        load_start = local_hit // self.connector.chunk_size * self.connector.chunk_size
        group_indices = self.connector.device_indices_for_linker_load(
            full.device_indices,
            swa_indices=None if swa is None else swa.device_indices,
            mamba_indices=None if mamba is None else mamba.device_indices,
            mamba_transfer_tokens=operation.total_hit_tokens - load_start,
        )
        self._queued_loads[rid] = _QueuedLoad(
            lookup=operation,
            group_indices=group_indices,
            local_hit_tokens=local_hit,
            owned_device_indices=full.device_indices,
        )
        return True

    def start_layer_wise_loading(self) -> int:
        if not self._queued_loads:
            return -1

        pending, self._queued_loads = self._queued_loads, {}
        counter_index = self.layer_done_counter.update_producer()
        stream = device_module.current_stream()
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

        self._load_batches[counter_index] = _StartedLoadBatch(
            rids=list(pending),
            operations=operations,
            submitted=submitted,
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
            self._prefetches.pop(rid, None)
            self.connector.finish_request(rid)
        return self.layer_done_counter.finish(counter_index) and success

    def cancel_queued_load(self, rid: str) -> bool:
        queued = self._queued_loads.pop(rid, None)
        if queued is None:
            return False
        self._prefetches.pop(rid, None)
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
        collected = self._collect_prefix(node_id, alignment=chunk_size)
        if collected is None:
            return False
        by_pool = {transfer.name: transfer for transfer in transfers}
        full = by_pool.get(PoolName.KV)
        if full is None or full.device_indices is None:
            return False
        store_start = collected.node_start // chunk_size * chunk_size
        aligned_end = len(collected.token_ids)
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
            current_stream = device_module.current_stream()
            self.record_transfer_ready().wait(current_stream)
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

        self._pending_stores.append(_PendingStore(request_id, operation))
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
        for rid, queued in self._queued_loads.items():
            self._prefetches.pop(rid, None)
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
                self._prefetches.pop(rid, None)
                self.connector.finish_request(rid)
        self._load_batches.clear()

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

        for rid, operation in list(self._prefetches.items()):
            self._finish_lookup(rid, operation)
        self._prefetches.clear()
        self._completed_stores.clear()
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.connector.close()

    def _collect_prefix(
        self, node_id: int, *, alignment: int
    ) -> Optional[_CollectedPrefix]:
        chunks = []
        leaf = None
        node_len = 0
        total_len = 0
        current = node_id
        while not self.cache.tree_core.is_root(current):
            snapshot = self.cache.tree_core.snapshot_buffer_backup(
                current, pass_prefix_keys=False
            )
            if snapshot is None or snapshot.key.is_bigram:
                return None
            if leaf is None:
                leaf = snapshot
                node_len = len(snapshot.key)
            token_ids = snapshot.key.raw_token_ids()
            chunks.append(token_ids)
            total_len += len(token_ids)
            current = snapshot.parent_node_id

        if leaf is None or not leaf.hash_values:
            return None
        chunks.reverse()
        aligned_len = total_len // alignment * alignment
        token_ids = list(islice(chain.from_iterable(chunks), aligned_len))
        return _CollectedPrefix(
            token_ids=token_ids,
            node_start=total_len - node_len,
            extra_key=leaf.key.extra_key,
            cache_salt=leaf.key.cache_salt,
            terminal_hash=leaf.hash_values[-1],
        )

    def _finish_lookup(self, rid: str, operation: Any) -> None:
        if operation.total_hit_tokens is not None and operation.locks_held:
            self.connector.free_lookup_locks(
                rid,
                start=operation.lock_start,
                end=operation.total_hit_tokens,
            )
        self.connector.finish_request(rid)

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
