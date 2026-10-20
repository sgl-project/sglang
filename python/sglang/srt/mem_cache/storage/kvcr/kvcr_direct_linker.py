"""Layer-wise direct delivery through a single KVCR API owner."""

from __future__ import annotations

import copy
import json
import logging
import socket
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from queue import Empty, Queue

import msgspec
import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.kvcr.device_layout import KVCRLayout
from sglang.srt.mem_cache.storage.kvcr.endpoints import rank_endpoint
from sglang.srt.mem_cache.storage.kvcr.runtime import KVCRRuntime
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.runtime_context import get_memory, get_model, get_parallel

logger = logging.getLogger(__name__)


@dataclass
class _Load:
    rid: str
    scope: str | None
    transfers: tuple
    claims: list = field(default_factory=list)
    done: bool = False
    error: BaseException | None = None


@dataclass
class _Batch:
    index: int
    loads: tuple[_Load, ...]
    ready: object
    work: object
    remaining: list[int]
    active: int = 0
    exhausted: bool = False
    error: BaseException | None = None
    started: float = field(default_factory=time.perf_counter)
    completion: Future = field(default_factory=Future)


@dataclass
class _Offload:
    transfers: tuple
    ready: object
    dependencies: tuple[_Load, ...]
    keys: set[bytes]
    work: object = None
    active: int = 0
    exhausted: bool = False
    success: bool = True


@dataclass
class _Confirm:
    claims: dict
    active: int = 0


@dataclass
class _Operation:
    owner: _Batch | _Offload | _Confirm
    keys: set[bytes]
    layer: int | None = None
    result: object = None


def _offset_source_ports(hints, offset):
    """Offset the first fetch action in an explicitly homogeneous deployment."""
    for action in hints.get("actions", ()):
        if not isinstance(action, dict) or action.get("action_type") != "kv.fetch":
            continue
        payload = action.get("payload")
        if isinstance(payload, dict):
            endpoint = payload.get("source_control_endpoint")
            endpoint = rank_endpoint(endpoint, offset)
            if endpoint is not None:
                payload["source_control_endpoint"] = endpoint
        break
    return hints


class KVCRDirectLinker(UnifiedCacheLinker):
    loaded_path_is_stored = False
    local_storage_medium = StorageMedium.CPU
    local_storage_ownership = "kvcr"

    def __init__(self, server_args, params, *, components, runtime_factory=None):
        if get_memory().enable_linker_mla_dedup:
            raise ValueError("KVCR does not support --enable-linker-mla-dedup")
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=params.token_to_kv_pool_allocator.get_kvcache(),
            page_size=params.page_size,
            params=params,
            components=components,
        )
        self.requires_lookup_confirmation = any(
            entry.indices_from_pool == PoolName.SWA for entry in self.pool_group.entries
        )
        tp_rank, tp_size, rank = 0, get_parallel().tp_size, 0
        if torch.distributed.is_initialized():
            group = params.attn_tp_cache_group or params.tp_cache_group
            tp_rank = torch.distributed.get_rank(group)
            tp_size = torch.distributed.get_world_size(group)
            rank = torch.distributed.get_rank()
        self._remote_hints_supported = params.attn_cp_size == params.pp_size == 1 and (
            tp_size == 1 or self.pool_group.rank_replicated
        )
        if not self._remote_hints_supported:
            logger.warning(
                "KVCR remote hints disabled: this layout requires shard-specific "
                "source endpoints. Local DRAM reuse remains available."
            )
        self.layout = KVCRLayout(
            self.pool_group,
            model_name=json.dumps(
                (
                    get_model().model_path,
                    get_model().revision,
                    get_model().quantization,
                    get_model().kv_cache_dtype,
                )
            ),
            endpoint_name=f"sglang-{socket.gethostname()}-{rank}-{uuid.uuid4().hex}",
            tp_rank=tp_rank,
            tp_size=tp_size,
            cp_rank=params.attn_cp_rank,
            cp_size=params.attn_cp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
        )
        options, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        route_replicated = options.pop("replicated_source_rank_routing", False)
        if type(route_replicated) is not bool:
            raise ValueError("KVCR replicated_source_rank_routing must be a boolean")
        if route_replicated and (
            not self.pool_group.rank_replicated
            or not self._remote_hints_supported
            or get_parallel().nnodes != 1
        ):
            raise ValueError(
                "KVCR replicated_source_rank_routing requires replicated pools "
                "with single-node CP1/PP1 serving"
            )
        # Opt-in asserts identical source attention-TP widths and rank-zero base ports.
        self._source_rank_offset = tp_rank if route_replicated else 0
        self._make_runtime = lambda: (runtime_factory or KVCRRuntime)(
            self.layout, options, rank=rank
        )
        self.layer_done_counter = LayerWiseLoadCounter(self.pool_group.num_layers)
        self.pools = self.pool_group.entry_map
        self._batch_results = {}
        self._commands = Queue()
        self._command_lock = threading.Lock()
        self._completed_loads = Queue()
        self._completed_offloads = Queue()
        self._loads = {}
        self._queued = []
        self._batches = deque()
        self._offloads = deque()
        self._unacknowledged_keys = Counter()
        self._operations = {}
        self._hazards = set()
        self._scopes = {}
        self._request_scopes = {}
        self._lookup_claims = {}
        self._released_scopes = set()
        self._fatal = None
        self._closed = False
        self._runtime = None
        self._started = Future()
        self._thread = threading.Thread(
            target=self._run, name=f"kvcr-owner-{rank}", daemon=True
        )
        self._thread.start()
        self._started.result()

    def _call(self, function, *args):
        future = Future()
        with self._command_lock:
            if self._closed:
                raise RuntimeError("KVCR linker is closed")
            self._commands.put((function, args, future))
        return future.result()

    def _run(self):
        try:
            buffer = self.pool_group.entries[0].kv_buffer[0]
            if buffer.is_cuda:
                torch.cuda.set_device(buffer.device)
            self._runtime = self._make_runtime()
            self._started.set_result(None)
        except BaseException as error:
            self._fatal = error
            self._started.set_exception(error)
            # KVCRStartupError may retain registrations and DMA; this owner
            # keeps the exception, runtime and framework tensors alive.
            if not hasattr(error, "kvcr_runtime"):
                self._closed = True
                return
            self._runtime = error.kvcr_runtime
        while not self._closed:
            active = self._fatal is None and (self._batches or self._offloads)
            try:
                function, args, future = self._commands.get(
                    timeout=0 if active else 0.001
                )
            except Empty:
                if active:
                    # Yield the GIL without pacing active transfers at 1 ms.
                    time.sleep(0)
            else:
                try:
                    if self._fatal is not None and function != self._close:
                        raise self._fatal
                    future.set_result(function(*args))
                except BaseException as error:
                    if function in (self._close, self._reset):
                        if hasattr(error, "kvcr_runtime"):
                            self._runtime = error.kvcr_runtime
                        self._fatal = error
                        for batch in self._batches:
                            self._fail_batch(batch, error, fatal=True)
                    future.set_exception(error)
            if self._fatal is None and not self._closed:
                try:
                    self._progress()
                    self._runtime.log_stats()
                except BaseException as error:
                    self._fatal = error
                    for batch in self._batches:
                        self._fail_batch(batch, error, fatal=True)
                    logger.exception(
                        "KVCR owner failed; retaining transfer ownership until close"
                    )
        while True:
            try:
                _, _, future = self._commands.get_nowait()
            except Empty:
                break
            future.set_exception(RuntimeError("KVCR linker is closed"))

    def set_request_context(self, handle, kv_hints):
        hints = (
            msgspec.to_builtins(kv_hints)
            if isinstance(kv_hints, msgspec.Struct)
            else copy.deepcopy(kv_hints)
        )
        self._call(self._set_context, handle, hints)

    def _set_context(self, handle, hints):
        scope = f"{len(handle.rid)}:{handle.rid}:{handle.attempt_id}"
        self._scopes[handle] = scope
        self._request_scopes[handle.rid] = scope
        if hints is not None and self._remote_hints_supported:
            if self._source_rank_offset:
                hints = _offset_source_ports(hints, self._source_rank_offset)
            try:
                self._runtime.client.submit_hint(hints, request_id=scope)
            except ValueError as error:
                # SGLang envelopes may carry actions that KVCR does not consume.
                logger.warning(
                    "Ignoring unsupported KVCR hint for %s: %s", handle.rid, error
                )

    def release_request_context(self, handle):
        if not self._closed:
            self._call(self._release_context, handle)

    def _release_context(self, handle):
        scope = self._scopes.pop(handle, None)
        if scope is not None:
            self._discard_lookup(scope)
            if self._request_scopes.get(handle.rid) == scope:
                self._request_scopes.pop(handle.rid)
            self._released_scopes.add(scope)
            self._cleanup_scopes()

    def _cleanup_scopes(self):
        in_use = {load.scope for load in self._loads.values()}
        for scope in self._released_scopes - in_use:
            self._runtime.client.discard_hint(scope)
            self._released_scopes.remove(scope)

    def lookup(self, rid, transfers):
        prepared = self.layout.group.resolve_transfers(transfers)
        keys = (
            next((tuple(t.keys) for t in transfers if t.name == PoolName.KV), ())
            if prepared
            else ()
        )
        return self._call(self._lookup, rid, keys, prepared)

    def _lookup(self, rid, keys, prepared):
        self._discard_lookup(self._request_scopes.get(rid))
        valid = set(range(1, len(keys) + 1))
        for transfer in prepared:
            encoded = [self.layout.encode_key(key, transfer.name) for key in keys]
            result = self._runtime.client.query(
                encoded, request_id=self._request_scopes.get(rid)
            )
            if len(result) != len(encoded):
                raise RuntimeError("KVCR query returned an incomplete result")
            hits = [
                (status.value, tier.value if tier is not None else None)
                in (("HIT", "DRAM"), ("FETCHABLE", "REMOTE_G2"))
                for status, tier in result
            ]
            missing = [0]
            for hit in hits:
                missing.append(missing[-1] + (not hit))
            window = max(1, len(transfer.keys))
            valid = {
                end
                for end in valid
                if missing[end]
                == missing[
                    0
                    if transfer.hit_policy == PoolHitPolicy.ALL_PAGES
                    else max(0, end - window)
                ]
            }
        return sorted(valid)

    def _release_claims(self, claims):
        if claims:
            released = self._runtime.client.release(claims)
            if len(released) != len(claims) or not all(ok for _, ok in released):
                raise RuntimeError("KVCR claim release failed")

    def _discard_lookup(self, scope):
        claims = self._lookup_claims.get(scope, {})
        self._release_claims([claim for claim in claims.values() if claim is not None])
        self._lookup_claims.pop(scope, None)

    def confirm_lookup(self, rid, transfers, hit_pages):
        prepared = self.layout.group.resolve_transfers(transfers) if hit_pages else ()
        keys = next((tuple(t.keys) for t in transfers if t.name == PoolName.KV), ())
        return self._call(self._confirm_lookup, rid, keys, prepared, hit_pages)

    def _confirm_lookup(self, rid, keys, prepared, hit_pages):
        scope = self._request_scopes.get(rid)
        if not hit_pages:
            self._discard_lookup(scope)
            return False
        windows = {
            t.name: [
                self.layout.encode_key(key, t.name)
                for key in keys[max(0, hit_pages - max(1, len(t.keys))) : hit_pages]
            ]
            for t in prepared
            if t.hit_policy != PoolHitPolicy.ALL_PAGES
        }
        if not windows:
            return True
        if scope is None:
            raise RuntimeError("KVCR trailing confirmation requires a request context")
        required = {key for window in windows.values() for key in window}
        claims = self._lookup_claims.setdefault(scope, {})
        stale = claims.keys() - required
        self._release_claims([claims[key] for key in stale if claims[key] is not None])
        for key in stale:
            del claims[key]
        # Keep only this candidate's window; None remembers misses while retrying.
        owner = _Confirm(claims)
        for name, window in windows.items():
            pending = [key for key in window if key not in claims]
            if not pending:
                continue
            labels = [
                label for component in self.layout._labels[name] for label in component
            ]
            handle = self._runtime.client.fetch(
                pending, request_id=scope, expected_layout=labels
            )
            self._operations[handle] = _Operation(owner, set(pending))
            owner.active += 1
        while owner.active:
            self._progress()
            time.sleep(0)
        return all(claims.get(key) is not None for key in required)

    def load(self, rid, transfers):
        prepared = self.layout.prepare(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        return self._call(self._load, rid, prepared)

    def _load(self, rid, prepared):
        scope = self._request_scopes.get(rid)
        if not prepared:
            self._discard_lookup(scope)
            return False
        if rid in self._loads:
            raise RuntimeError(f"KVCR load already accepted for {rid}")
        claims = self._lookup_claims.pop(scope, {})
        load = _Load(
            rid,
            scope,
            prepared,
            [claim for claim in claims.values() if claim is not None],
        )
        self._loads[rid] = load
        self._queued.append(load)
        return True

    def cancel_queued_load(self, rid):
        return self._call(self._cancel_queued_load, rid)

    def _cancel_queued_load(self, rid):
        load = self._loads.get(rid)
        if load is None or load not in self._queued:
            return False
        self._queued.remove(load)
        self._loads.pop(rid)
        self._release_claims(load.claims)
        self._cleanup_scopes()
        return True

    def finish_layer_wise_loading(self, counter_index):
        if counter_index < 0:
            return True
        # Layer readiness alone is not terminal native ownership: wait for the
        # whole batch to drain, including failed operations and source claims.
        completion = self._call(self._batch_results.pop, counter_index)
        landed = completion.result()
        self.layer_done_counter.finish(counter_index)
        return landed

    def start_layer_wise_loading(self):
        if self._closed:
            raise RuntimeError("KVCR linker is closed")
        if self._fatal is not None:
            raise self._fatal
        # Synchronous load/start/reset calls keep this queue visible to the scheduler.
        if not self._queued:
            return -1
        ready = torch.cuda.Event()
        ready.record()
        return self._call(self._start, ready)

    def _start(self, ready):
        if not self._queued:
            return -1
        loads, self._queued = tuple(self._queued), []
        index = self.layer_done_counter.update_producer()
        batch = _Batch(
            index,
            loads,
            ready,
            self._layer_work(loads),
            [1] * self.pool_group.num_layers,
        )
        self._batch_results[index] = batch.completion
        self._batches.append(batch)
        return index

    def _layer_work(self, loads):
        for layer in range(self.pool_group.num_layers):
            for load in loads:
                for blocks in self.layout.mappings(load.transfers, layer):
                    yield layer, load.scope, blocks
            # The sentinel accounts for work not yet submitted in this layer.
            yield layer, None, None

    def offload(self, transfers):
        # Row conversion copies indices to CPU and waits for queued GPU work.
        # Snapshot on the producer stream; prepare only after its event is ready.
        keys = self.layout.encoded_keys(transfers, allow_partial=True)
        if not keys:
            return False
        transfers = tuple(
            replace(
                transfer,
                keys=list(transfer.keys) if transfer.keys is not None else None,
                device_indices=(
                    transfer.device_indices.clone()
                    if isinstance(transfer.device_indices, torch.Tensor)
                    else transfer.device_indices
                ),
                host_indices=(
                    transfer.host_indices.clone()
                    if isinstance(transfer.host_indices, torch.Tensor)
                    else transfer.host_indices
                ),
            )
            for transfer in transfers
        )
        ready = torch.cuda.Event()
        ready.record()
        return self._call(self._offload, transfers, ready, keys)

    def _offload(self, transfers, ready, keys):
        self._unacknowledged_keys.update(keys)
        self._offloads.append(
            _Offload(transfers, ready, tuple(self._loads.values()), keys)
        )
        return True

    def _prepare_offload(self, transfers):
        prepared = self.layout.prepare(list(transfers), allow_partial=True)
        return self.layout.mappings(prepared)

    def _fail_batch(self, batch, error, *, fatal=False):
        batch.error = error
        batch.exhausted = True
        batch.work = None
        self.layer_done_counter.fail(batch.index, error)
        if fatal and not batch.completion.done():
            # Unblock the scheduler with a fatal error, not a reusable failed
            # load. Native registrations and private slots remain retained.
            batch.completion.set_exception(error)

    def _progress(self):
        results = list(self._runtime.client.poll_completed())
        while self._runtime.events:
            event = self._runtime.events.popleft()
            if getattr(event, "state", None) == "uncertain":
                self._hazards.add(event.op_handle)
            elif getattr(event, "state", None) == "quiesced":
                self._hazards.discard(event.op_handle)
            else:
                raise event
        for handle, result in results:
            operation = self._operations[handle]
            operation.result = result
            if set(result) != operation.keys or not all(
                entry.success for entry in result.values()
            ):
                if isinstance(operation.owner, _Batch):
                    self._fail_batch(
                        operation.owner, RuntimeError(f"KVCR delivery {handle} failed")
                    )
                elif isinstance(operation.owner, _Offload):
                    operation.owner.success = False
                    operation.owner.exhausted = True
        # Lifecycle handles may name internal copies rather than public ops.
        # Conservatively retain every accepted operation while DMA is uncertain.
        if self._hazards:
            return
        for handle, operation in list(self._operations.items()):
            if operation.result is None:
                continue
            owner = operation.owner
            owner.active -= 1
            if isinstance(owner, _Batch):
                owner.remaining[operation.layer] -= 1
                self._complete_layer(owner, operation.layer)
            elif isinstance(owner, _Confirm):
                owner.claims.update(dict.fromkeys(operation.keys))
                owner.claims.update(
                    (key, entry.release_handle)
                    for key, entry in operation.result.items()
                    if entry.success
                )
            del self._operations[handle]
        deliveries = sum(
            isinstance(op.owner, _Batch) for op in self._operations.values()
        )
        for batch in self._batches:
            if not batch.ready.query():
                continue
            while not batch.exhausted and deliveries < 4:
                try:
                    layer, scope, blocks = next(batch.work)
                except StopIteration:
                    batch.exhausted = True
                    break
                if blocks is None:
                    batch.remaining[layer] -= 1
                    self._complete_layer(batch, layer)
                    continue
                started = time.perf_counter()
                handle = self._runtime.client.deliver(blocks, request_id=scope)
                logger.debug(
                    "KVCR submission batch=%d layer=%d keys=%d cpu_ms=%.3f",
                    batch.index,
                    layer,
                    len(blocks),
                    (time.perf_counter() - started) * 1000,
                )
                self._operations[handle] = _Operation(batch, set(blocks), layer)
                batch.active += 1
                batch.remaining[layer] += 1
                deliveries += 1
        # Completion queues preserve admission order even if transport does not.
        while (
            self._batches and self._batches[0].exhausted and not self._batches[0].active
        ):
            batch = self._batches.popleft()
            logger.debug(
                "KVCR batch=%d drained_ms=%.3f success=%s",
                batch.index,
                (time.perf_counter() - batch.started) * 1000,
                batch.error is None,
            )
            for load in batch.loads:
                self._release_claims(load.claims)
                load.done, load.error = True, batch.error
                del self._loads[load.rid]
            self._completed_loads.put([load.rid for load in batch.loads])
            self._cleanup_scopes()
            batch.completion.set_result(batch.error is None)
        if self._offloads:
            task = self._offloads[0]
            if not task.ready.query() or any(
                not load.done for load in task.dependencies
            ):
                return
            if any(load.error for load in task.dependencies):
                task.success, task.exhausted = False, True
            if not task.exhausted and not task.active:
                if not self._commands.empty():
                    return
                if task.work is None:
                    try:
                        task.work = iter(self._prepare_offload(task.transfers))
                    except Exception:
                        logger.exception("KVCR offload preparation failed")
                        task.success, task.exhausted = False, True
                if not task.exhausted:
                    blocks = next(task.work, None)
                    if blocks is None:
                        task.exhausted = True
                    else:
                        handle = self._runtime.client.deposit(blocks)
                        self._operations[handle] = _Operation(task, set(blocks))
                        task.active += 1
            if task.exhausted and not task.active:
                self._completed_offloads.put((task.success, task.keys))
                self._offloads.popleft()

    def _complete_layer(self, batch, layer):
        if not batch.error and batch.remaining[layer] == 0:
            logger.debug(
                "KVCR batch=%d layer=%d ready_ms=%.3f",
                batch.index,
                layer,
                (time.perf_counter() - batch.started) * 1000,
            )
            self.layer_done_counter.complete(batch.index, layer)

    def num_completed_loads(self):
        return self._completed_loads.qsize()

    def pop_completed_load(self):
        return self._completed_loads.get_nowait()

    def num_completed_offloads(self):
        return self._completed_offloads.qsize()

    def pop_completed_offload(self):
        return self._call(self._pop_completed_offload)

    def _pop_completed_offload(self):
        success, keys = self._completed_offloads.get_nowait()
        for key in keys:
            self._unacknowledged_keys[key] -= 1
            if not self._unacknowledged_keys[key]:
                del self._unacknowledged_keys[key]
        return success

    def pop_storage_removals(self):
        return self._call(self._pop_storage_removals)

    def _pop_storage_removals(self):
        # A delayed rank-agreed acknowledgement must not resurrect an eviction.
        return [
            self._runtime.removals.pop(key)
            for key in list(self._runtime.removals)
            if key not in self._unacknowledged_keys
        ]

    def reset(self):
        self._call(self._reset)

    def _reset(self):
        self._runtime.close()
        removals = self._runtime.removals
        for batch in self._batches:
            self._fail_batch(batch, RuntimeError("KVCR linker reset"))
        self._loads.clear()
        self._queued.clear()
        self._batches.clear()
        for completion in self._batch_results.values():
            if not completion.done():
                completion.set_result(False)
        self._batch_results.clear()
        self._offloads.clear()
        self._unacknowledged_keys.clear()
        self._operations.clear()
        self._hazards.clear()
        self._scopes.clear()
        self._request_scopes.clear()
        self._lookup_claims.clear()
        self._released_scopes.clear()
        self._completed_loads = Queue()
        self._completed_offloads = Queue()
        self.layer_done_counter.reset()
        self._runtime = self._make_runtime()
        # Engine reset must not drop pending removals from retained inventory.
        self._runtime.removals.update(removals)

    def close(self):
        if not self._closed:
            self._call(self._close)
            self._thread.join()

    def _close(self):
        self._runtime.close()
        self._lookup_claims.clear()
        for batch in self._batches:
            self._fail_batch(batch, RuntimeError("KVCR linker closed"))
            if not batch.completion.done():
                batch.completion.set_result(False)
        with self._command_lock:
            self._closed = True
