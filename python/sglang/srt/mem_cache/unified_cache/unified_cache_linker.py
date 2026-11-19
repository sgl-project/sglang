"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. The transport contract and tree-side wrapper live here, while
each backend owns its device-pool layout and physical I/O.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``_prefetch_kvcache`` -> full-layer device prefetch before admission
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``check_hicache_events`` -> full-layer prefetch publication before admission
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

A layer-wise load lands in device slots private to the loading request and joins the tree
only when that request inserts it like any computed KV. A failed load therefore
reaches nothing but its own requests, which the scheduler aborts.
Prefetch publishes complete KV before admission and recomputes failed reads.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefParams,
    InsertParams,
    MatchResult,
)
from sglang.srt.mem_cache.capacity import shares_memory
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand, largest_fitting_chunk
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.session.streaming_session import is_virtual_node
from sglang.srt.utils import broadcast_pyobj, get_device_module

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)
device_module = get_device_module()

# The token and state components the linker stores, besides the tree's window
# slots (which it finds by family, whatever ids the tree gives them).
_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
        ComponentType.MAMBA,
    }
)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook.

    A failed layer does not raise into the forward that waits on it: the
    forward runs on whatever the slots hold, and :meth:`finish` reports the
    failure afterwards so the scheduler can abort the loading requests before
    their output is used.
    """

    def __init__(
        self,
        num_layers: int,
        on_layer_ready: Optional[Callable[[int, int], None]] = None,
    ):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
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
        # Blocks until the layer resolves; a failure is reported by finish().
        futures[threshold].exception()
        # Runs on failure too: an MLA dedup source must still issue the layer
        # broadcast its peers are waiting for.
        if self.on_layer_ready is not None:
            self.on_layer_ready(index, threshold)

    def finish(self, index: int) -> bool:
        """Wait for every layer of batch ``index``; True when all of them landed."""
        futures = self.futures.get(index)
        if futures is None:
            return True
        # Wait before dropping the entry: the load thread still completes the
        # layers the forward never waited on through it.
        landed = all(future.exception() is None for future in futures)
        del self.futures[index]
        return landed

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()


class TransferReady(NamedTuple):
    events: list[torch.cuda.Event]

    def synchronize(self) -> None:
        for event in self.events:
            event.synchronize()

    def wait(self, stream: torch.cuda.Stream) -> None:
        for event in self.events:
            stream.wait_event(event)


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools.

    Out-of-tree linkers are constructed per worker as
    ``Linker(server_args, params, components=..., extra_config=..., forward_stream=...)``.
    Store ``forward_stream`` and use ``record_transfer_ready()`` to gate loads
    and offloads. ``layer_done_counter`` provides ``set_consumer(index)`` and
    ``wait_until(layer)``. Report completion only after device copies finish.
    """

    layer_done_counter: object
    pools: dict[PoolName, object]
    load_mode = "layer_wise"
    prefetch_policy = "timeout"
    # None when the current stream is the only one using the pages.
    forward_stream: Optional[torch.cuda.Stream] = None

    def record_transfer_ready(self) -> TransferReady:
        """Record what a transfer waits on before it touches device pages.

        The current stream computed the page indices; under read-done the
        forward in flight may still use those pages, including freed ones.
        """
        indices_ready = device_module.Event()
        indices_ready.record()
        events = [indices_ready]
        if self.forward_stream is not None:
            forward_done = device_module.Event()
            forward_done.record(self.forward_stream)
            events.append(forward_done)
        return TransferReady(events)

    def prefetch(self, transfers: list[PoolTransfer]) -> Future[bool] | None:
        """Read all layers; the future completes after all device writes stop."""
        return None

    @abstractmethod
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        Local to this rank; the tree intersects the sets across ranks.
        """

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        """Wait until the batch started as ``counter_index`` stops writing to
        device memory; True when every layer landed."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    @abstractmethod
    def offload(
        self, transfers: list[PoolTransfer], session_id: Optional[str] = None
    ) -> bool:
        """Queue every transfer for atomic persistence.

        ``session_id`` names the session whose insert produced the transfers,
        when known; backends may attribute the stored objects to it.
        """

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``tail_hashes`` are the per-page storage hashes of the restorable tail,
    starting at ``device_hit_len``.
    """

    tail_hashes: list[str]
    device_hit_len: int


@dataclass
class _ExternalCachePrefetch:
    key: RadixKey
    anchor_node_id: NodeId
    anchor_lock: DecLockRefParams | None = None
    component_transfers: list[tuple[TreeComponent, PoolTransfer]] = field(
        default_factory=list
    )
    future: Future[bool] | None = None
    timeout_s: float = 0.0
    wait_started: float | None = None
    discard: bool = False


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        # The linker keeps every window slot of the tree, each as its own pool.
        windows = cache.window_slots()
        supported = _EXTERNAL_LINKER_SUPPORTED_COMPONENTS | set(windows)
        unsupported = set(cache.tree_components) - supported
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full, window and Mamba tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        self.load_mode = cache_linker.load_mode
        self._windows = tuple(cache.components[slot] for slot in windows)
        # A window whose pages the store lacks tombstones restored tails instead.
        self._skipped_windows = tuple(
            window
            for window in self._windows
            if window.pool_name not in cache_linker.pools
        )
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not any(component is window for window in self._skipped_windows)
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        self.prefetches: dict[CacheRequestHandle, _ExternalCachePrefetch] = {}
        # Requests loading in the batch being built, then its counter index once
        # the batch starts; finish_loads consumes both after its forward.
        self.inflight_load_rids: list[str] = []
        self.rejected_pp_loads: set[str] = set()
        # Loaded tails stay outside the tree until their forward completes.
        self.private_load_indices: dict[str, torch.Tensor] = {}
        self.inflight_load_index = -1
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []
        # Page-aligned radix-key prefix excluded from offload during an insert.
        self.offload_skip_prefix_len = 0

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1
        self.pp_tickets = (
            _LinkerPPTickets(self)
            if cache.pp_size > 1 and cache.pp_group is not None
            else None
        )

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        cache = self.cache
        key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
        page = cache.page_size
        device_hit_len = result.device_prefix_len
        if self.load_mode == "prefetch":
            return result
        self.hit_markers.pop(req.rid, None)

        if cache.pp_size > 1:
            # Only the ticket worker queries storage under PP. Before its
            # verdict arrives ordinary matches may expose the local L1 only.
            end = req.external_cache_hit_length or 0
            if len(key) < end:
                # A trailing checkpoint was proved at end, not at a shorter
                # candidate that another caller chose to probe.
                return result
            key = key[:end]

        if device_hit_len >= len(key):
            if self.pp_tickets is not None and req.external_cache_hit_length:
                self._finish_prefetch(req.cache_request_handle, publish=False)
                req.external_cache_hit_length = 0
            return result

        tail_hashes = self._tail_hashes(key, result, device_hit_len)
        if not tail_hashes:
            return result

        if self.pp_tickets is not None:
            # A longer L1 match frees surplus reserved rows before admission.
            reserved = self._trim_pp_reservation(
                req.cache_request_handle, len(key), tail_hashes
            )
            if reserved is None:
                return result
            lookup_transfers = [t for _, t in reserved.component_transfers]
        else:
            lookup_transfers = self._lookup_transfers(tail_hashes)
            if not lookup_transfers:
                return result
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}

        # Page 0 of tail_hashes is the first uncached page.
        hit_pages = len(tail_hashes)
        if cache.pp_size == 1:
            hit_pages = self._sync_restorable_prefix(
                self.cache_linker.lookup(req.rid, lookup_transfers),
                num_pages=len(tail_hashes),
                device_hit_pages=0,
            )
        if hit_pages == 0:
            return result
        hit_tokens = hit_pages * page

        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if PoolName.MAMBA in by_pool else 0

        self.hit_markers[req.rid] = ExternalCacheHitMarker(
            tail_hashes=list(tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        window_hits = dict(result.window_host_hit_lengths)
        for window in self._windows:
            transfer = by_pool.get(window.pool_name)
            hit = min(len(transfer.keys), hit_pages) * page if transfer else 0
            name = window.window_sub_pool
            window_hits[name] = max(window_hits.get(name, 0), hit)
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            window_host_hit_lengths=window_hits,
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _lookup_transfers(self, hashes: list[str]) -> list[PoolTransfer]:
        transfers = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, hashes
            )
            if transfer is None:
                return []
            transfers.append(transfer)
        return transfers

    def _prefetch_key(self, req: Req) -> RadixKey:
        tokens = req.full_untruncated_fill_ids
        return RadixKey(
            tokens,
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
            is_bigram=self.cache.tree_core.is_eagle,
            limit=req._compute_max_prefix_len(len(tokens)),
        ).page_aligned(self.cache.page_size)

    def _lookup_prefetch(self, rid: str, hashes: list[str]) -> list[int]:
        try:
            transfers = self._lookup_transfers(hashes)
            if not transfers:
                return []
            hits = self.cache_linker.lookup(rid, transfers)
            return [pages for pages in hits if 0 < pages <= len(hashes)]
        except Exception:
            logger.warning(
                "External linker prefetch lookup failed for %s", rid, exc_info=True
            )
            return []

    def prefetch(self, req: Req) -> None:
        """Stage a queued request; publish only after all ranks succeed."""
        if self.pp_tickets is not None:
            # Arrival issued the original ticket. A retraction starts a new
            # attempt locally on every stage without another inbound message.
            if req.is_retracted and not req.finished():
                self.pp_tickets.bind(req, expected=True, retracted=True)
            return
        if self.load_mode != "prefetch":
            return
        request = req.cache_request_handle
        for previous in self.prefetches:
            if previous.rid == request.rid and previous != request:
                self._cancel_prefetch(previous)
        if self.cache.disable or request in self.prefetches:
            return
        req.init_next_round_input(self.cache, cow_mamba=False)
        if is_virtual_node(req.last_node) or req.positional_embed_overrides is not None:
            return
        page = self.cache.page_size
        key = self._prefetch_key(req)
        start = req.prefix_len
        last_hash = self.cache.get_last_hash_value(req.last_node)
        if len(key) <= start or (start and last_hash is None):
            return
        hashes = get_storage_hash_str(key[start:], last_hash, page_size=page)
        pages = self._sync_restorable_prefix(
            self._lookup_prefetch(req.rid, hashes),
            num_pages=len(hashes),
            device_hit_pages=0,
        )
        if not pages:
            return
        end = start + pages * page
        state = _ExternalCachePrefetch(
            key[:end],
            req.last_node,
            self.cache.inc_lock_ref(req.last_node).to_dec_params(),
            timeout_s=self.cache.prefetch_timeout_base
            + len(hashes) * self.cache.prefetch_timeout_per_page,
        )
        self.prefetches[request] = state
        try:
            if self._allocate_transfers(state.component_transfers, hashes[:pages]):
                state.future = self.cache_linker.prefetch(
                    [t for _, t in state.component_transfers]
                )
        except Exception:
            logger.warning("External linker prefetch submission failed", exc_info=True)
        accepted = torch.tensor([state.future is not None], dtype=torch.int)
        self.cache._all_reduce_attn_groups(accepted, torch.distributed.ReduceOp.MIN)
        if not accepted.item():
            self._cancel_prefetch(request)
            self.check_prefetch_events()

    def _allocate_transfers(
        self, transfers, hashes, phase: LinkerTransferPhase = LinkerTransferPhase.LOAD
    ) -> bool:
        """Append allocated pools in order so the caller can unwind a partial take."""
        for component in self._components:
            transfer = component.build_external_linker_transfer(phase, None, hashes)
            if transfer is None:
                return False
            transfers.append((component, transfer))
        return True

    def _pp_prefetch_fits(self, pages: int) -> bool:
        """Whether all pools can satisfy this reservation without tree eviction."""
        page = self.cache.page_size
        tokens = pages * page
        allocator = self.cache.token_to_kv_pool_allocator
        window_tokens = {
            w.window_sub_pool: min(
                tokens, (w.sliding_window_size + page - 1) // page * page
            )
            for w in self._windows
            if w not in self._skipped_windows
        }
        has_state = ComponentType.MAMBA in self.cache.components
        if shares_memory(allocator):
            # An owner, window and state can fit separately in the chain and
            # still exceed the shared arena when their bytes are priced together.
            demand = KVDemand(tokens, window_tokens, int(has_state))
            return allocator.ensure_capacity(
                demand.completed(allocator.window_pool_names())
            )
        storage = self.cache.kv_storage()
        owner, _, _ = allocator.sub_pool_roles()
        return (
            storage.units_fit(owner, tokens)
            and all(storage.units_fit(name, n) for name, n in window_tokens.items())
            and (not has_state or allocator.state_slots.slots_fit(1))
        )

    def reserve_pp_prefetch(
        self,
        request: CacheRequestHandle,
        anchor_node_id: NodeId,
        key: RadixKey,
        start: int,
        hashes: list[str],
    ) -> int:
        """Reserve free slots below the query bound and return the absolute end page."""
        page = self.cache.page_size
        start_page = start // page
        pages = len(hashes)
        if not pages or request in self.prefetches:
            return start_page
        # The ticket holds the anchor through publication or discard on this stage.
        state = _ExternalCachePrefetch(key, anchor_node_id)
        self.prefetches[request] = state
        try:
            if not self._pp_prefetch_fits(pages):
                pages = largest_fitting_chunk(self._pp_prefetch_fits, pages - 1, 1)
            if pages:
                state.key = key[: start + pages * page]
                if self._allocate_transfers(
                    state.component_transfers,
                    hashes[:pages],
                    LinkerTransferPhase.RESERVE,
                ):
                    return start_page + pages
        except Exception:
            logger.warning("PP linker prefetch reservation failed", exc_info=True)
        self._finish_prefetch(request, publish=False)
        return start_page

    def _trim_pp_reservation(self, request, end, hashes=None):
        """Retarget FULL, trailing windows and state to the agreed prefix or a later L1 match."""
        state = self.prefetches.get(request)
        if state is None or state.future is not None:
            return state
        full = state.component_transfers[0][1]
        start = len(state.key) - len(full.device_indices)
        if state.discard or end <= start:
            self._finish_prefetch(request, publish=False)
            return None
        if hashes is None:
            page = self.cache.page_size
            assert end <= len(state.key) and (end - start) % page == 0
            hashes = full.keys[: (end - start) // page]
        if full.keys == hashes:
            return state
        lookups = self._lookup_transfers(hashes)
        # A smaller boundary changes the window tail and the Mamba
        # checkpoint key even when both need the same number of slots.
        for (component, transfer), lookup in zip(
            state.component_transfers, lookups, strict=True
        ):
            indices = transfer.device_indices
            keep = len(lookup.keys) * (len(indices) // len(transfer.keys))
            if keep < len(indices):
                component.update_external_linker_load(
                    ExternalLinkerLoadPhase.ABORT,
                    None,
                    full,
                    replace(transfer, device_indices=indices[keep:]),
                    end,
                )
                transfer.device_indices = indices[:keep]
            transfer.keys = lookup.keys
        state.key = state.key[:end]
        return state

    def submit_pp_prefetch(
        self, request: CacheRequestHandle, end: int
    ) -> Future[bool] | None:
        try:
            state = self._trim_pp_reservation(request, end)
            if state is None:
                return None
            if state.future is None:
                state.future = self.cache_linker.prefetch(
                    [t for _, t in state.component_transfers]
                )
            if state.future is not None:
                return state.future
        except Exception:
            logger.warning("PP linker prefetch submission failed", exc_info=True)
        self._finish_prefetch(request, publish=False)
        return None

    def check_prefetch_progress(self, request: CacheRequestHandle) -> bool:
        if self.pp_tickets is not None:
            return self.pp_tickets.can_admit(request)
        state = self.prefetches.get(request)
        if state is None or state.discard:
            return True
        policy = self.cache.prefetch_stop_policy
        if policy == "wait_complete":
            return False
        if policy == "timeout":
            now = time.perf_counter()
            expired = torch.tensor(
                state.wait_started is not None
                and now - state.wait_started > state.timeout_s,
                dtype=torch.int,
            )
            self.cache._all_reduce_attn_groups(expired, torch.distributed.ReduceOp.MAX)
            if not expired.item():
                if state.wait_started is None:
                    state.wait_started = now
                return False
        # Consume any fully completed read before stopping. Running reads keep
        # their private slots until every rank has stopped writing to them.
        self.check_prefetch_events()
        self._cancel_prefetch(request)
        return True

    def _cancel_prefetch(self, request: CacheRequestHandle) -> None:
        state = self.prefetches.get(request)
        if state is not None:
            state.discard = True
            if state.future is not None:
                state.future.cancel()

    def prefetch_size(self, pool: PoolName) -> int:
        """Device slots still owned by prefetch, including discarded writes."""
        return sum(
            len(transfer.device_indices)
            for state in self.prefetches.values()
            for _, transfer in state.component_transfers
            if transfer.name == pool and transfer.device_indices is not None
        )

    def check_prefetch_events(self) -> None:
        if self.pp_tickets is not None:
            self.cache._bound_async_work()
            self.pp_tickets.poll()
            return
        if not self.prefetches:
            return
        pending = list(self.prefetches.items())
        progress = []
        for _, state in pending:
            future = state.future
            done = future is None or future.done()
            landed = False
            try:
                if done and future is not None and not future.cancelled():
                    landed = future.result() is True
            except Exception:
                logger.warning("External linker prefetch failed", exc_info=True)
            progress.append([done, landed and not state.discard])
        # Keep the same operation order on every rank, including discarded
        # reads. Local cancellation must not change collective participation.
        verdicts = torch.tensor(progress, dtype=torch.int)
        self.cache._all_reduce_attn_groups(verdicts, torch.distributed.ReduceOp.MIN)
        for (request, _), (done, landed) in zip(pending, verdicts.tolist()):
            if done:
                self._finish_prefetch(request, publish=bool(landed))

    def has_pending_prefetch(self) -> bool:
        return bool(self.prefetches or (self.pp_tickets and self.pp_tickets.pending))

    def _finish_prefetch(self, request: CacheRequestHandle, *, publish: bool) -> None:
        state = self.prefetches.pop(request, None)
        if state is None:
            return
        # Event polling or reset/close has already quiesced every device writer.
        if state.future is not None:
            assert state.future.done()
        end = len(state.key)
        try:
            if not publish:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT, None, state.component_transfers, end
                )
                return
            by_pool = {t.name: t for _, t in state.component_transfers}
            storage = self.cache.kv_storage()
            full = by_pool[PoolName.KV].device_indices
            start = end - len(full)
            evicted = {}
            page = self.cache.page_size
            for window in self._windows:
                transfer = by_pool.get(window.pool_name)
                if transfer is not None:
                    storage.bind_window(
                        window.window_sub_pool,
                        full[-len(transfer.device_indices) :],
                        transfer.device_indices,
                    )
                skipped = any(window is w for w in self._skipped_windows)
                kept = (window.sliding_window_size + page - 1) // page * page
                evicted[window.component_type] = end if skipped else max(0, end - kept)
            prefix = self.cache.path_device_indices(state.anchor_node_id)
            assert len(prefix) == start
            # The normal insert protects the borrowed prefix and frees duplicate
            # fetched slots. Avoid writing the just-fetched data back to storage.
            mamba = by_pool.get(PoolName.MAMBA)
            if mamba is not None:
                sub_pool = storage.tree_sub_pool(ComponentType.MAMBA.name)
                # A restored checkpoint has no pending ReplaySSM updates.
                storage.restart_replay(sub_pool, mamba.device_indices)
            inserted = self.cache.insert(
                InsertParams(
                    key=state.key,
                    value=torch.cat([prefix, full]),
                    mamba_value=mamba.device_indices if mamba is not None else None,
                    prev_prefix_len=start,
                    inserted_len=end,
                    component_evicted_seqlens=evicted,
                )
            )
            if mamba is not None and inserted.mamba_exist:
                storage.free_state_slots(sub_pool, mamba.device_indices)
            self.cache.prefetch_loaded_tokens_by_reqid[request] = len(full)
            self.cache.prefetch_loaded_storage_start_by_reqid[request] = start
        finally:
            if state.anchor_lock is not None:
                self.cache.dec_lock_ref(state.anchor_node_id, state.anchor_lock)

    def _sync_restorable_prefix(
        self, restorable: list[int], *, num_pages: int, device_hit_pages: int
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection.
        """
        mask = torch.zeros(num_pages + 1, dtype=torch.int)
        for pages in restorable:
            if device_hit_pages < pages <= num_pages:
                mask[pages] = 1
        self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        common = mask.nonzero()
        if common.numel() == 0:
            return 0
        return int(common[-1].item())

    def _tail_hashes(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        return get_storage_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> request-private device slots ----

    def load_back(self, req: Req) -> tuple[int, NodeId]:
        """Queue the external hit into device slots owned by ``req``.

        The loaded tail extends the request's prefix but stays out of the
        tree (``req.last_node`` is unchanged): the request inserts it like
        computed KV after its forward, once :meth:`finish_loads` has confirmed
        the load.
        """
        cache = self.cache
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return 0, req.last_node

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        reserved = (
            self.prefetches.pop(req.cache_request_handle, None)
            if self.pp_tickets
            else None
        )
        if reserved is not None:
            req.external_cache_hit_length = 0
            component_transfers = reserved.component_transfers
        else:
            component_transfers = []
            if not self._allocate_transfers(component_transfers, tail_hashes):
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
                )
                return 0, req.last_node
        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        # PP already agreed on the reserved shape. A local backend rejection
        # must follow the same retirement path as an asynchronous read failure.
        try:
            queued = self.cache_linker.load(
                req.rid, [transfer for _, transfer in component_transfers]
            )
        except BaseException as exc:
            if reserved is None or not isinstance(exc, Exception):
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
                )
                raise
            logger.exception("Could not queue PP linker load for %s", req.rid)
            queued = False
        if not queued:
            if reserved is None:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
                )
                return 0, req.last_node
            self.rejected_pp_loads.add(req.rid)
        self._update_load(
            ExternalLinkerLoadPhase.PREPARE,
            req,
            component_transfers,
            prefix_len,
        )

        # Components omitted from the linker do not run their PREPARE hook.
        # Keep a non-restorable window range as tombstones instead of rebuilding
        # it from an uninitialized FULL-to-window mapping when the tail is inserted.
        for window in self._skipped_windows:
            name = window.window_sub_pool
            req.kv.set_evicted_seqlen(
                name, max(req.kv.get_evicted_seqlen(name), prefix_len)
            )

        # match placed the SWA branch point before this tail was known; a point
        # the tail covers is gone, since the request inserts the whole tail.
        if (
            req.swa_branching_seqlen is not None
            and req.swa_branching_seqlen <= prefix_len
        ):
            req.swa_branching_seqlen = None
        self.private_load_indices[req.rid] = full_transfer.device_indices
        self.inflight_load_rids.append(req.rid)
        if cache.pp_size > 1 and req.kv is not None and req.kv.holds_kv:
            # A later PP chunk already owns a row; the extend allocator reads
            # its prefix from that row instead of private_load_indices.
            # Allocation settles the KV watermarks after consuming this row.
            cache.req_to_token_pool.write(
                (req.kv.req_pool_idx, slice(device_hit_len, prefix_len)),
                full_transfer.device_indices,
            )
        return len(full_transfer.device_indices), req.last_node

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req | None,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            transfer = component.update_external_linker_load(
                phase, req, full, transfer, prefix_len
            )
            if transfer is not None:
                result.append(transfer)
        return result

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            transfers = self.cache.tree_core.build_external_linker_offload_transfers(
                node_id
            )
            if transfers is None:
                continue
            if self._skipped_windows:
                skipped = {window.pool_name for window in self._skipped_windows}
                transfers = [t for t in transfers if t.name not in skipped]
            if self.offload_skip_prefix_len:
                transfers = self._drop_offload_skip_prefix(node_id, transfers)
                if transfers is None:
                    continue
            self._offload_node(node_id, transfers)

    @contextmanager
    def skip_offload_prefix(self, length: int):
        """Skip offloading the first ``length`` radix-key units during an insert."""
        self.offload_skip_prefix_len = length
        try:
            yield
        finally:
            self.offload_skip_prefix_len = 0

    def _drop_offload_skip_prefix(
        self, node_id: NodeId, transfers: list[PoolTransfer]
    ) -> Optional[list[PoolTransfer]]:
        info = self.cache.tree_core.collect_path_info([node_id])[0][-1]
        page = self.cache.page_size
        kept = []
        for transfer in transfers:
            first = info.path_end - len(transfer.keys) * page
            drop = min(
                len(transfer.keys),
                max(0, (self.offload_skip_prefix_len - first) // page),
            )
            if drop == len(transfer.keys):
                if transfer.name == PoolName.KV:
                    # The caller delegates storage of this prefix to another writer.
                    # Mark it stored locally to suppress uploads, without a persistence ACK.
                    self.cache.tree_core.mark_external_cache_stored_path(
                        node_id, info.parent_id
                    )
                    return None
                continue
            if drop:
                transfer = replace(
                    transfer,
                    keys=transfer.keys[drop:],
                    device_indices=transfer.device_indices[drop * page :],
                )
            kept.append(transfer)
        return kept

    def _offload_node(self, node_id: NodeId, transfers: list[PoolTransfer]) -> None:
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(
                transfers, session_id=getattr(cache, "insert_session_id", None)
            )
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        cache.tree_core.mark_external_linker_offload_pending(node_id)
        self.pending_offloads.append(_PendingOffload(node_id, lock_params, [node_id]))

    def replace_pending_offload_node(
        self, ack_id: NodeId, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        for index, pending in enumerate(self.pending_offloads):
            if pending.lock_node_id != ack_id:
                continue
            publish_node_ids = []
            for node_id in pending.publish_node_ids:
                if node_id == old_node_id:
                    publish_node_ids.extend(new_node_ids)
                else:
                    publish_node_ids.append(node_id)
            self.pending_offloads[index] = pending._replace(
                publish_node_ids=publish_node_ids
            )
            return

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, success
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)

    def start_layer_wise_loading(self) -> int:
        self.inflight_load_index = self.cache_linker.start_layer_wise_loading()
        return self.inflight_load_index

    def finish_loads(self, reqs: list[Req] | None = None) -> list[str]:
        """Wait for local writes and agree on failed rows within this stage.

        A PP stage can satisfy the same ticket from L1, or its backend can
        refuse to queue a row. All attention ranks vote in forward row order
        for a PP load, including ranks which had no I/O for that batch.
        """
        rids, self.inflight_load_rids = self.inflight_load_rids, []
        if reqs is None and not rids:
            return []
        index, self.inflight_load_index = self.inflight_load_index, -1
        failed = set(rids) & self.rejected_pp_loads
        self.rejected_pp_loads.difference_update(failed)
        if index >= 0 and not self.cache_linker.finish_layer_wise_loading(index):
            failed.update(rids)
        for rid in rids:
            self.private_load_indices.pop(rid, None)
        rows = rids if reqs is None else [req.rid for req in reqs]
        landed = torch.tensor([rid not in failed for rid in rows], dtype=torch.int)
        self.cache._all_reduce_attn_groups(landed, torch.distributed.ReduceOp.MIN)
        failures = [rid for rid, ok in zip(rows, landed.tolist()) if not ok]
        if failures:
            logger.error("External linker load failed for %s", failures)
        return failures

    # ---- lifecycle ----

    def reset(self) -> None:
        if self.pp_tickets is not None:
            self.pp_tickets.close()
        self.cache_linker.reset()
        self._clear_state()
        if self.pp_tickets is not None:
            self.pp_tickets = _LinkerPPTickets(self)

    def _clear_state(self) -> None:
        for request in list(self.prefetches):
            self._finish_prefetch(request, publish=False)
        self.hit_markers.clear()
        self.inflight_load_rids.clear()
        self.rejected_pp_loads.clear()
        self.private_load_indices.clear()
        self.inflight_load_index = -1
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()

    def release_request(self, request: CacheRequestHandle) -> None:
        rid = request.rid
        self.hit_markers.pop(rid, None)
        if self.pp_tickets is not None:
            self.pp_tickets.release(request)
        else:
            self._cancel_prefetch(request)
        # A queued load targets slots the request owns and is about to free.
        # A started one is finished by finish_loads before anything is freed.
        if rid in self.inflight_load_rids and (
            self.cache_linker.cancel_queued_load(rid) or rid in self.rejected_pp_loads
        ):
            self.rejected_pp_loads.discard(rid)
            self.inflight_load_rids.remove(rid)
            self.private_load_indices.pop(rid, None)

    def close(self) -> None:
        if self.pp_tickets is not None:
            self.pp_tickets.close()
        self.cache_linker.close()
        self._clear_state()


class _PPTicket(NamedTuple):
    handle: CacheRequestHandle
    key: RadixKey
    start: int
    last_hash: str | None


@dataclass
class _LocalPPTicket:
    req: Req | None = None
    ticket: _PPTicket | None = None
    pin: tuple[NodeId, DecLockRefParams] | None = None
    discard: bool = False
    awaiting_turn: bool = False
    wait_started: float | None = None
    # Worker-to-scheduler query bound, scheduler-to-worker reserved capacity,
    # then the agreed boundary and the actual GET result.
    hashes: list[str] = field(default_factory=list)
    lookup: int | None = None
    capacity: int | None = None
    end: int | None = None
    read: Future[bool] | bool | None = None
    landed: bool | None = None


class _LinkerPPTickets:
    """Agree on storage boundaries off-thread. Only the scheduler owns device rows."""

    def __init__(self, linker: UnifiedCacheLinkerWrapper):
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.linker = linker
        self.cache = cache = linker.cache
        self.page = cache.page_size
        self.prefetch = linker.load_mode == "prefetch"
        self.arrival = threading.Condition()
        self.pending: dict[CacheRequestHandle, _LocalPPTicket] = {}
        self.layer_admission: _LocalPPTicket | None = None
        self.stopping = False
        self.failure: BaseException | None = None
        self.commands = SimpleQueue()
        self.completions = SimpleQueue()
        self.query_groups = []
        self.ack_groups = []
        self.command_thread = self.ack_thread = None

        pp_ranks = dist.get_process_group_ranks(cache.pp_group)
        attention = [
            g for g in (cache.attn_cp_group, cache.attn_tp_group) if g is not None
        ]
        rank_sets = []
        for group in attention or [cache.tp_group]:
            if group is not None:
                ranks = dist.get_process_group_ranks(group)
                if len(ranks) > 1 and ranks != pp_ranks and ranks not in rank_sets:
                    rank_sets.append(ranks)
        rank_sets.append(pp_ranks)
        try:
            for groups in (
                [self.query_groups, self.ack_groups]
                if self.prefetch
                else [self.query_groups]
            ):
                for ranks in rank_sets:
                    groups.append(
                        create_custom_parallel_group(group_ranks=ranks, backend="gloo")
                    )
            if self.prefetch:
                self.ack_thread = self._start(
                    self._complete_reads, "linker-pp-completion"
                )
            self.command_thread = self._start(self._prepare_tickets, "linker-pp-ticket")
        except BaseException:
            self.close()
            raise

    def _start(self, target, name):
        def run():
            try:
                target()
            except BaseException as error:
                self.failure = error
                logger.exception("%s exited", name)

        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return thread

    def _register(self, handle, state):
        with self.arrival:
            self.pending[handle] = state
            self.arrival.notify_all()

    def bind(self, req: Req, expected: bool, *, retracted: bool = False) -> bool:
        handle = req.cache_request_handle
        if handle in self.pending or (
            self.layer_admission is not None
            and self.layer_admission.ticket.handle == handle
        ):
            return True
        req.external_cache_hit_length = 0
        source = self.cache.pp_rank == 0
        if not source and not expected:
            return False
        ticket, pin = self._local_ticket(req, pin=self.prefetch)
        if (
            source
            and not retracted
            and (not len(ticket.key) or req.positional_embed_overrides is not None)
        ):
            return False
        self._register(
            handle, _LocalPPTicket(req, ticket, pin, awaiting_turn=not self.prefetch)
        )
        if source:
            self.commands.put(ticket)
        return True

    def _local_ticket(self, req: Req, *, pin: bool):
        req.init_next_round_input(self.cache, cow_mamba=False)
        key = self.linker._prefetch_key(req)
        start = req.prefix_len
        valid = (
            bool(len(key))
            and req.positional_embed_overrides is None
            and not is_virtual_node(req.last_node)
            and start <= len(key)
            and start % self.page == 0
        )
        prior = self.cache.get_last_hash_value(req.last_node) if valid else None
        lock = None
        if pin and valid and (not start or prior):
            lock = (
                req.last_node,
                self.cache.inc_lock_ref(req.last_node).to_dec_params(),
            )
        return _PPTicket(req.cache_request_handle, key, start, prior), lock

    def skip_unbound(self, rid: str) -> None:
        handle = CacheRequestHandle(rid, attempt_id=0)
        if handle not in self.pending:
            self._register(handle, _LocalPPTicket(discard=True))

    def _lookup(self, ticket):
        if ticket.start < len(ticket.key) and (not ticket.start or ticket.last_hash):
            try:
                hashes = get_storage_hash_str(
                    ticket.key[ticket.start :], ticket.last_hash, page_size=self.page
                )
                return hashes, self.linker._lookup_prefetch(ticket.handle.rid, hashes)
            except Exception:
                logger.exception("PP linker key lookup failed for %s", ticket.handle)
        return [], []

    def _prepare_tickets(self):
        command_group = self.query_groups[-1]
        rank = dist.get_rank()
        src = dist.get_process_group_ranks(command_group)[0]
        while True:
            commands = []
            if rank == src:
                try:
                    commands = [self.commands.get(timeout=60)]
                except Empty:
                    pass  # Complete idle broadcasts before the group times out.
            commands = broadcast_pyobj(commands, rank, command_group, src=src)
            if not commands:
                continue
            ticket = commands[0]
            if ticket is None:
                return
            with self.arrival:
                arrived = ticket.handle in self.pending or self.stopping
            # The first query can run before the follower receives its Req.
            lookup = None if arrived else self._lookup(ticket)
            with self.arrival:
                self.arrival.wait_for(
                    lambda: (
                        self.stopping
                        or (
                            ticket.handle in self.pending
                            and not self.pending[ticket.handle].awaiting_turn
                        )
                    )
                )
                state = self.pending.get(ticket.handle)
            mask = torch.zeros(len(ticket.key) // self.page + 1, dtype=torch.int)
            if state is not None and state.pin is not None and not self.stopping:
                local = state.ticket
                if (
                    lookup is None
                    or local.start != ticket.start
                    or local.last_hash != ticket.last_hash
                ):
                    lookup = self._lookup(local)
                state.hashes, hits = lookup
                start = local.start // self.page
                pages = [start] + [start + p for p in hits]
                state.lookup = max(pages)
                with self.arrival:
                    self.arrival.wait_for(
                        lambda: state.capacity is not None or self.stopping
                    )
                if state.capacity is not None:
                    mask[[p for p in pages if p <= state.capacity]] = 1
            elif state is not None:
                state.lookup = 0
            for group in self.query_groups:
                dist.all_reduce(mask, op=dist.ReduceOp.MIN, group=group)
            common = mask.nonzero()
            end = int(common[-1].item()) * self.page if common.numel() else 0
            if state is not None:
                state.end = end
                if self.prefetch:
                    self.completions.put((state, end))

    def _complete_reads(self):
        # This FIFO has independent groups so an unfinished GET does not stop
        # the next ticket from being queried, reserved, and submitted.
        while (item := self.completions.get()) is not None:
            state, end = item
            landed = False
            if end:
                with self.arrival:
                    self.arrival.wait_for(
                        lambda: state.read is not None or self.stopping
                    )
                read = state.read
                try:
                    landed = read.result() if isinstance(read, Future) else read
                except Exception:
                    landed = False
                ack = torch.tensor(int(bool(landed) and not state.discard))
                for group in self.ack_groups:
                    dist.all_reduce(ack, op=dist.ReduceOp.MIN, group=group)
                landed = bool(ack.item())
            state.landed = landed

    def _reserve(self, state, upper):
        if state.discard or state.pin is None:
            return -1
        ticket = state.ticket
        start = ticket.start // self.page
        if upper <= start:
            return start
        # PP0 can ask us to reserve before our lookup worker returns.
        hashes = state.hashes or get_storage_hash_str(
            ticket.key[ticket.start : upper * self.page],
            ticket.last_hash,
            page_size=self.page,
        )
        return self.linker.reserve_pp_prefetch(
            ticket.handle,
            state.pin[0],
            ticket.key,
            ticket.start,
            hashes[: upper - start],
        )

    def _ready_word(self, state):
        if state.capacity is None:
            value = state.lookup
        elif state.read is None:
            value = state.end
        else:
            value = state.landed
        # Followers can poll before their worker records the same collective result.
        return 0 if value is None else value + 1

    def _keep_waiting(self, state, now):
        if state.discard:
            return False
        if state.wait_started is None or (
            state.read is not None and state.landed is not None
        ):
            return True
        if self.cache.prefetch_stop_policy == "best_effort":
            return False
        pages = (len(state.ticket.key) - state.ticket.start) // self.page
        budget = (
            self.cache.prefetch_timeout_base
            + pages * self.cache.prefetch_timeout_per_page
        )
        return now - state.wait_started <= budget

    def poll(self) -> None:
        if self.failure is not None:
            raise RuntimeError("PP linker ticket worker failed") from self.failure
        # A layer-wise reservation is private until admission. The synchronized
        # ticket verdict gives each stage one scheduling turn to consume it.
        # A deferred request can recompute; it must not keep other requests out.
        if self.layer_admission is not None:
            self.release(self.layer_admission.ticket.handle)
        if not self.pending:
            return
        # Poll in arrival order on every stage, including misses.
        items = self.pending.items()
        if not self.prefetch:
            # Only the head may pin a rank-local L1 prefix against admission.
            handle, state = next(iter(items))
            items = [(handle, state)]
            if state.awaiting_turn:
                if not state.discard:
                    state.ticket, state.pin = self._local_ticket(state.req, pin=True)
                with self.arrival:
                    state.awaiting_turn = False
                    self.arrival.notify_all()
        else:
            items = list(items)
        now = time.perf_counter()
        ready = torch.tensor(
            [(self._ready_word(s), self._keep_waiting(s, now)) for _, s in items],
            dtype=torch.int,
        )
        self.cache._all_reduce(ready, torch.distributed.ReduceOp.MIN)
        for (flag, keep), (handle, state) in zip(ready.tolist(), items):
            if not keep and not state.discard:
                self.release(handle)
            if not flag:
                continue
            if state.capacity is None:
                capacity = self._reserve(state, flag - 1)
                with self.arrival:
                    state.capacity = capacity
                    state.hashes = []
                    self.arrival.notify_all()
            elif self.prefetch and state.read is None:
                end = flag - 1
                read = self.linker.submit_pp_prefetch(
                    handle, 0 if state.discard else end
                )
                local = state.pin is not None and state.ticket.start == end
                with self.arrival:
                    state.read = read or (local and not state.discard)
                    self.arrival.notify_all()
            else:
                if self.prefetch:
                    self.linker._finish_prefetch(
                        handle, publish=bool(flag - 1) and not state.discard
                    )
                else:
                    end = 0 if state.discard else flag - 1
                    if end:
                        state.req.external_cache_hit_length = end
                        state.req.pp_linker_verdict_end = end
                        self.layer_admission = state
                    self.linker._trim_pp_reservation(handle, end)
                if state.pin is not None and state is not self.layer_admission:
                    self.cache.dec_lock_ref(*state.pin)
                with self.arrival:
                    del self.pending[handle]

    def can_admit(self, handle: CacheRequestHandle) -> bool:
        state = self.pending.get(handle)
        if state is None or state.discard:
            return True
        if (
            self.cache.pp_rank == 0
            and self.prefetch
            and self.cache.prefetch_stop_policy != "wait_complete"
            and state.wait_started is None
        ):
            state.wait_started = time.perf_counter()
        return False

    def release(self, handle: CacheRequestHandle) -> None:
        state = self.pending.get(handle)
        if state is not None:
            state.discard = True
            self.linker._cancel_prefetch(handle)
        ready = self.layer_admission
        if ready is not None and ready.ticket.handle == handle:
            if ready.req.cache_request_handle == handle:
                ready.req.external_cache_hit_length = 0
                self.linker.hit_markers.pop(handle.rid, None)
            self.linker._finish_prefetch(handle, publish=False)
            self.cache.dec_lock_ref(*ready.pin)
            self.layer_admission = None

    def close(self) -> None:
        with self.arrival:
            if self.stopping:
                return
            self.stopping = True
            self.arrival.notify_all()
        if self.cache.pp_rank == 0:
            self.commands.put(None)
        # The producer enqueues the last ACK before the completion worker stops.
        if self.command_thread is not None:
            self.command_thread.join()
        if self.ack_thread is not None:
            self.completions.put(None)
            self.ack_thread.join()
        for groups in (self.ack_groups, self.query_groups):
            for group in reversed(groups):
                dist.destroy_process_group(group)
        for state in self.pending.values():
            if self.failure is not None and isinstance(state.read, Future):
                # A failed ACK worker may have left later GETs in flight.
                try:
                    state.read.result()
                except Exception:
                    pass
            if state.pin is not None:
                self.cache.dec_lock_ref(*state.pin)
        if self.layer_admission is not None:
            self.release(self.layer_admission.ticket.handle)
        self.pending.clear()
        if self.failure is not None:
            raise RuntimeError("PP linker ticket worker failed") from self.failure
