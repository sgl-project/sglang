"""Buffer-only mode transfer pipelines for the unified radix cache.

``BufferModePipeline`` owns all buffer-mode state and the two pipelines that
move KV through the transient host staging buffer:

- backup (write path): admission-gated FIFO intents, head-of-line D2H
  staging launches, storage writes at the D2H ack, staging freed at the
  storage ack;
- load back (read path): completed storage fetches parked as op-owned host
  bounces, consumed at prefill admission via a device alloc + layer-gated
  H2D + plain tree insert, bounce freed at the H2D ack.

The pipeline is an intimate collaborator of ``UnifiedRadixCache``: it is
constructed by ``init_hicache`` only when ``--hicache-host-memory-mode
buffer_only`` is active, and it drives tree/controller operations (insert,
match, evict, lock refs, cache actions) through the owning cache. All
buffer-mode-only state lives here; the cache dispatches to this object at
its mode branches.

TP-lockstep contract: every mutation runs on the scheduler thread at
rank-synchronized points (insert walks, rank-MIN-reduced drains, ack
drains), so per-rank state never diverges. There is no runtime
verification; a violation surfaces as an unexplained collective hang.
"""

from __future__ import annotations

import logging
from array import array
from collections import deque
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HICACHE_WRITE_STAGING_POOL_FRACTION
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefParams,
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StagedPrefetchPlan
from sglang.srt.mem_cache.unified_cache.cache_action import RebuildFullToSWAMapping
from sglang.srt.mem_cache.unified_cache.components import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
    BufferBackupState,
    NodeId,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.pool_host import HostPoolGroup
    from sglang.srt.mem_cache.unified_cache.components import SWAComponent
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


def _minimum_full_transfer_tokens(
    host_pool: HostKVCache, prefetch_threshold: int
) -> int:
    page_size = host_pool.page_size
    return max(page_size, -(-prefetch_threshold // page_size) * page_size)


class _UnifiedBackupIntent(msgspec.Struct):
    """One pool of one node queued for a buffer-mode write (unpinned while
    queued); a derived sidecar rides its source pool's intent.

    Snapshots node identity at enqueue time: a split rewrites the node's
    key/hash in place while these copies stay intact, so a key-length change
    detects a split (repaired during queue refresh) and a missing FULL device
    value detects eviction (``_validate_backup_intent``). ``keys`` is the
    admission view (it sizes drop metrics); the D2H launch re-reads rows and
    keys off the node.
    """

    snapshot: BufferBackupSnapshot
    pool: PoolName
    keys: list[str]


class _UnifiedBufferBackupEntry(msgspec.Struct):
    """A buffer-mode backup after its D2H launch: intent + staging slots.

    ``host_indices`` are the KV staging slots (empty for another pool's
    intent); ``aux_xfers`` the staged transfers and riding sidecars; all are
    freed at the storage-write ack (host memory is never a cache tier).
    ``keys_by_pool`` are the keys the write stores, learned at the ack.
    """

    intent: _UnifiedBackupIntent
    host_indices: torch.Tensor
    aux_xfers: list[PoolTransfer]
    lock_params: DecLockRefParams
    occupied_units: int
    keys_by_pool: dict[PoolName, list[str]]


class _StagedPrefetch(msgspec.Struct):
    """A completed buffer-mode fetch parked until prefill admission: only
    the op-owned host bounce exists (no device state, nothing in the tree).
    """

    request: CacheRequestHandle
    key_tokens: array
    extra_key: Optional[str]
    cache_salt: Optional[str]
    matched_len: int
    num_tokens: int
    occupied_tokens: int
    host_indices: torch.Tensor
    aux_xfers: list[PoolTransfer]
    hash_values: list[str]
    operation_id: int


class _OngoingBufferLoadBack(msgspec.Struct):
    """A buffer-mode load-back awaiting its H2D ack: the span is already
    tree-resident. The host bounce and any redundant auxiliary device slots
    remain owned here until the copy completes.
    """

    request: CacheRequestHandle
    num_tokens: int
    occupied_tokens: int
    aux_xfers: list[PoolTransfer]
    host_indices: torch.Tensor
    hash_values: list[str]
    aux_device_releases: list[tuple[PoolName, torch.Tensor]]


class _AnchorLock(msgspec.Struct):
    """Pins a staged prefetch's FULL device anchor until consumption."""

    node_id: NodeId
    tokens: int


def validate_buffer_only_stack(
    sidecar_pool_specs: list[SidecarPoolSpec],
    host_pool_group: HostPoolGroup,
    swa_component: Optional[SWAComponent],
    storage_prefetch_threshold: int = 256,
) -> None:
    """Post-assembly buffer-mode fences.

    Sidecars reuse their source pool's transient slot ids, so every sidecar
    host pool must expose the full source slot namespace.  unified_kv SWA
    (device-only ring, never offloaded) still has no staging path.
    """
    entry_map = host_pool_group.entry_map
    for spec in sidecar_pool_specs:
        source = entry_map.get(spec.indices_from_pool)
        sidecar = entry_map.get(spec.pool_name)
        if source is None or sidecar is None:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only sidecar pool mapping "
                f"is incomplete: pool={spec.pool_name}, "
                f"indices_from_pool={spec.indices_from_pool}."
            )
        source_size = source.host_pool.logical_size
        sidecar_size = sidecar.host_pool.logical_size
        if sidecar_size < source_size:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only sidecar host pool is "
                "smaller than its index source: "
                f"pool={spec.pool_name}, host_slots={sidecar_size}, "
                f"source={spec.indices_from_pool}, source_slots={source_size}."
            )
    swa = swa_component
    if swa is not None and swa._swa_kv_pool_host is None:
        # Only reachable on SWA models with the unified_kv layout (SWA as
        # a device-only ring): without a host pool the window can neither
        # stage for writes nor fetch for load-backs.
        raise ValueError(
            "--hicache-host-memory-mode buffer_only on SWA models "
            "requires an SWA host staging pool; the unified_kv layout "
            "keeps SWA as a device-only ring."
        )
    if swa is not None and swa._swa_kv_pool_host is not None:
        # Below two windows the pool cannot hold a staging write AND the
        # loads-priority reserve (_aux_loads_margin floors at one
        # window), so every window-carrying intent would be dropped as
        # oversize and SWA storage coverage would silently be zero.
        window_tokens = swa.full_window_pages * swa._swa_kv_pool_host.page_size
        shared_domain = (
            swa._swa_kv_pool_host.shared_allocation_domain
            if isinstance(swa._swa_kv_pool_host, HostKVCache)
            else None
        )
        if shared_domain is not None:
            full_host_pool = (
                swa.cache.cache_controller.mem_pool_host.anchor_entry.host_pool
            )
            min_full_tokens = _minimum_full_transfer_tokens(
                full_host_pool, storage_prefetch_threshold
            )
            one_transfer_bytes = (
                window_tokens * swa._swa_kv_pool_host.size_per_token
                + min_full_tokens * full_host_pool.size_per_token
            )
            one_transfer = [
                (full_host_pool.pool_label, min_full_tokens),
                (swa._swa_kv_pool_host.pool_label, window_tokens),
            ]
            enough_capacity = shared_domain.can_fit_many_then(
                one_transfer, one_transfer, empty=True
            )
            capacity = f"{shared_domain.capacity_bytes} shared bytes"
            requirement = f"{2 * one_transfer_bytes} shared bytes"
        else:
            enough_capacity = swa._swa_kv_pool_host.size >= 2 * window_tokens
            capacity = f"{swa._swa_kv_pool_host.size} SWA tokens"
            requirement = f"{2 * window_tokens} SWA tokens"
        if not enough_capacity:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only requires a host arena "
                "large enough for two minimum Full/SWA transfers "
                f"({requirement}; got {capacity}): one staging a write "
                "while one stays reserved for prefetch window allocs."
            )


class BufferModePipeline:
    """All buffer-mode state plus the backup and load-back pipelines.

    Constructed by ``UnifiedRadixCache.init_hicache`` when host memory mode
    is ``buffer_only``; ``cache.buffer_pipeline is None`` elsewhere, which
    the cache's mode branches use as the dispatch test.
    """

    def __init__(
        self,
        cache: UnifiedRadixCache,
        swa_window_pages: int,
        write_backlog_cap: int,
        max_context_len: int = 0,
    ):
        self._cache = cache
        # SWA window size in KV pages when the SWA component stages through
        # a host pool (0 = KV-only: no trailing window staged). Static after
        # pool assembly.
        self._swa_window_pages = swa_window_pages
        # Metadata-only pending-write backlog cap, measured in unique FULL
        # snapshot spans; beyond it new spans are dropped at admission.
        self.write_backlog_cap = write_backlog_cap
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        full_pool = kvcache.full_kv_pool if isinstance(kvcache, SWAKVPool) else kvcache
        # Clamp by admission headroom: pins must leave room for the largest
        # allowed request, else a queued hold can wedge admission permanently
        # (pool full, nothing retractable). No-headroom pools take no pins.
        self.anchor_lock_cap_tokens = max(
            0,
            min(
                int(envs.SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP.get() * full_pool.size),
                full_pool.size - max_context_len,
            ),
        )
        if self.anchor_lock_cap_tokens == 0:
            logger.warning(
                "BufferModePipeline anchor_lock_cap_tokens=0 (pool=%d, "
                "max_context_len=%d): every prefetch launches with its splice "
                "base unpinned. Shrink --context-length or grow the KV pool.",
                full_pool.size,
                max_context_len,
            )
        else:
            logger.info(
                "BufferModePipeline anchor_lock_cap_tokens=%d",
                self.anchor_lock_cap_tokens,
            )
        self.reset()

    # Arbitrary: a deferral means the admission budget and the allocator disagree
    # about free slots, which waiting on decode rarely fixes.
    max_staged_admission_defers: int = 32

    def reset(self) -> None:
        # Load pipeline: hits awaiting a staging grant (park-and-retry),
        # enqueue-time prefix context, completed prefetches staged until
        # prefill admission, and load-backs in flight (keyed by synthetic
        # negative ack id).
        self.pending_hit_allocs: deque = deque()
        self._staged_admission_defers: dict[CacheRequestHandle, int] = {}
        self._prefetch_prefix_ctx: dict[
            CacheRequestHandle, tuple[list[int], Optional[str], Optional[str]]
        ] = {}
        self.staged_prefetches: dict[CacheRequestHandle, _StagedPrefetch] = {}
        self.ongoing_buffer_load_back: dict[int, _OngoingBufferLoadBack] = {}
        # FIFO intents (one per pool of a node) awaiting a D2H slot, and the
        # pools in flight per node (admission to storage ack).
        self.pending_write_queue: deque[_UnifiedBackupIntent] = deque()
        self._queued_span_refs: dict[tuple[NodeId, int], int] = {}
        self.inflight_backup_pools: dict[NodeId, set[PoolName]] = {}
        # Nodes admitted during the current insert walk (None outside one).
        self._walk_admitted: Optional[set[NodeId]] = None
        # Launched writes: D2H in flight (by ack id), then the storage write
        # in flight (by operation id).
        self.ongoing_write_through: dict[int, _UnifiedBufferBackupEntry] = {}
        self.ongoing_backup: dict[int, _UnifiedBufferBackupEntry] = {}
        self._backup_ack_seq = 0
        self.write_staged_tokens_ = 0
        self.write_backlog_tokens_ = 0
        self._backlog_cap_hits = 0
        # Attempt-keyed locks stop stale completions from unlocking retries.
        self.anchor_locks: dict[CacheRequestHandle, _AnchorLock] = {}
        self.anchor_locked_tokens_ = 0
        self._anchor_lock_cap_skips = 0

    def _shared_host_domain(self):
        cc = self._cache.cache_controller
        anchor = cc.mem_pool_host.anchor_entry.host_pool
        if not isinstance(anchor, HostKVCache):
            return None
        return anchor.shared_allocation_domain

    def _transfer_tokens(self, transfer: PoolTransfer) -> int:
        if transfer.host_indices is not None:
            return len(transfer.host_indices)
        if transfer.keys is None:
            return 0
        entry = self._cache.cache_controller.mem_pool_host.entry_map.get(transfer.name)
        return 0 if entry is None else len(transfer.keys) * entry.host_pool.page_size

    def host_allocation_units(
        self,
        host_indices: Optional[torch.Tensor],
        aux_xfers: Optional[list[PoolTransfer]],
    ) -> int:
        """Host usage expressed in anchor-token units for scheduler accounting."""
        if host_indices is None:
            return 0
        return self._host_request_units(len(host_indices), aux_xfers)

    def _shared_host_requests(
        self, kv_tokens: int, aux_xfers: Optional[list[PoolTransfer]]
    ) -> Optional[list[tuple[str, int]]]:
        if self._shared_host_domain() is None:
            return None
        return [
            (pool.pool_label, tokens)
            for pool, tokens in self._host_staging_sizes(kv_tokens, aux_xfers)
        ]

    def _host_staging_sizes(self, kv_tokens, aux_xfers):
        cc = self._cache.cache_controller
        yield cc.mem_pool_host.anchor_entry.host_pool, kv_tokens
        for transfer in aux_xfers or ():
            if transfer.indices_from_pool is not None:
                continue
            entry = cc.mem_pool_host.entry_map.get(transfer.name)
            if entry is not None:
                yield entry.host_pool, self._transfer_tokens(transfer)

    def _host_request_units(
        self, kv_tokens: int, aux_xfers: Optional[list[PoolTransfer]]
    ) -> int:
        """Host staging in anchor-token accounting units."""
        cc = self._cache.cache_controller
        anchor = cc.mem_pool_host.anchor_entry.host_pool
        if self._shared_host_domain() is None:
            # Aux transfers staged through the anchor's own host pool (an
            # entry aliasing it) occupy anchor tokens as well.
            return kv_tokens + self._anchor_pool_aux_tokens(aux_xfers or [])
        num_bytes = sum(
            tokens * pool.size_per_token
            for pool, tokens in self._host_staging_sizes(kv_tokens, aux_xfers)
        )
        return (num_bytes + anchor.size_per_token - 1) // anchor.size_per_token

    def _shared_backup_fits(self, requests, *, empty: bool = False) -> bool:
        return self._shared_host_domain().can_fit_many_then(
            requests, self._shared_load_reserve_requests(), empty=empty
        )

    def _shared_load_reserve_requests(self) -> list[tuple[str, int]]:
        cc = self._cache.cache_controller
        anchor = cc.mem_pool_host.anchor_entry.host_pool
        swa_entry = cc.mem_pool_host.entry_map.get(PoolName.SWA)
        min_full_tokens = _minimum_full_transfer_tokens(
            anchor, self._cache.prefetch_threshold
        )
        requests = [(anchor.pool_label, min_full_tokens)]
        if swa_entry is not None and self._swa_window_pages:
            requests.append(
                (
                    swa_entry.host_pool.pool_label,
                    self._swa_window_pages * swa_entry.host_pool.page_size,
                )
            )
        return requests

    @property
    def inflight_backup_node_ids(self):
        """Nodes with a write of any pool queued, staged or awaiting its
        storage ack."""
        return self.inflight_backup_pools.keys()

    def backup_pending(self, node_id: NodeId) -> bool:
        """A write of ``node_id`` is queued, staged or awaiting its storage
        ack. A component that drops rows a pending write still reads asks
        here first; the pipeline pins the node only from D2H launch to ack."""
        return node_id in self.inflight_backup_pools

    def is_idle(self) -> bool:
        """No queued, staged, or in-flight buffer-mode work or anchor pins."""
        return not (
            self.pending_hit_allocs
            or self.staged_prefetches
            or self.ongoing_buffer_load_back
            or self.pending_write_queue
            or self.inflight_backup_node_ids
            or self.ongoing_write_through
            or self.ongoing_backup
            or self.anchor_locks
        )

    def swa_transient_size(self) -> int:
        """SWA destinations kept alive only until an in-flight H2D completes."""
        return sum(
            len(device_indices)
            for load_back in self.ongoing_buffer_load_back.values()
            for pool_name, device_indices in load_back.aux_device_releases
            if pool_name == PoolName.SWA
        )

    # ---- backup pipeline (device -> staging -> storage) ----

    def _anchor_pool_aux_tokens(self, aux_xfers: list[PoolTransfer]) -> int:
        """Staging tokens aux transfers hold in the anchor's host pool."""
        group = self._cache.cache_controller.mem_pool_host
        anchor_pool = group.anchor_entry.host_pool
        tokens = 0
        for t in aux_xfers:
            if t.indices_from_pool is not None:
                continue
            entry = group.entry_map.get(t.name)
            if entry is not None and entry.host_pool is anchor_pool:
                tokens += self._transfer_tokens(t)
        return tokens

    def _backup_parent_covered(self, state: BufferBackupState, node_id: NodeId) -> bool:
        """Only admit a node whose parent is stored/in-flight: writing above
        a dropped parent creates a permanent longest-prefix hole. A component
        may vouch for the parent through a pool of its own."""
        if state.parent_is_root or PoolName.KV in self.inflight_backup_pools.get(
            state.parent_node_id, ()
        ):
            return True
        if (
            state.parent_last_hash is not None
            and self._cache.storage_existence_cache.pool(PoolName.KV).contains(
                state.parent_last_hash
            )
        ):
            return True
        return any(
            component.buffer_backup_parent_covered(state.parent_node_id, node_id)
            for component in self._cache.components.values()
        )

    def _log_backup_dropped(self, num_tokens: int) -> None:
        cache = self._cache
        if cache.enable_storage_metrics and cache.storage_metrics_collector is not None:
            cache.storage_metrics_collector.log_backup_dropped_tokens(num_tokens)

    def _pool_covered(self, pool: PoolName, keys: list[str]) -> bool:
        """Beliefs plus content past a D2H launch (republished content must
        not re-write while the original write drains)."""
        return self._cache.storage_existence_cache.pool(pool).covered(keys)

    def _transfer_keys(
        self, transfer: PoolTransfer, hash_values: list[str]
    ) -> Optional[list[str]]:
        """The builder's keys, else the chain's trailing pages, one per staged
        page (the SWA convention); None when the transfer stages nothing."""
        if transfer.keys is not None:
            return list(transfer.keys)
        if transfer.indices_from_pool is not None or transfer.device_indices is None:
            return None
        entry = self._cache.cache_controller.mem_pool_host.entry_map.get(transfer.name)
        if entry is None:
            return None
        num_pages = len(transfer.device_indices) // entry.host_pool.page_size
        if num_pages == 0 or num_pages > len(hash_values):
            return None
        return list(hash_values[-num_pages:])

    def _backup_pool_keys(
        self, snapshot: BufferBackupSnapshot
    ) -> dict[PoolName, list[str]]:
        """Admission view of the keys a backup writes per pool: the KV chain,
        each aux component's BACKUP_HOST transfers, component overrides on
        top, every sidecar under its source's keys."""
        hash_values = snapshot.hash_values
        tree_core = self._cache.tree_core
        pool_keys: dict[PoolName, list[str]] = {PoolName.KV: list(hash_values)}
        overrides = tree_core.buffer_backup_pool_keys(snapshot.node_id, hash_values)
        for ct in self._cache.components:
            # A component that names its pools' keys spares the transfer build.
            if ct == ComponentType.FULL or ct in overrides:
                continue
            transfers = tree_core.build_hicache_transfers(
                ct, snapshot.node_id, CacheTransferPhase.BACKUP_HOST
            )
            for transfer in transfers or ():
                keys = self._transfer_keys(transfer, hash_values)
                if keys:
                    pool_keys.setdefault(transfer.name, []).extend(keys)
        for named in overrides.values():
            for pool, keys in named.items():
                if keys:
                    pool_keys[pool] = list(keys)
                else:
                    pool_keys.pop(pool, None)
        for spec in self._cache.sidecar_pool_specs:
            source = pool_keys.get(spec.indices_from_pool)
            if source:
                pool_keys[spec.pool_name] = list(source)
        return pool_keys

    @staticmethod
    def _pool_order(pool: PoolName) -> tuple[bool, str]:
        # KV first, then a fixed order: every rank must admit identically.
        return (pool != PoolName.KV, pool.value)

    def _write_intents(
        self, snapshot: BufferBackupSnapshot
    ) -> list[tuple[PoolName, list[str]]]:
        """(pool, keys) of the pools not believed stored or in flight; a
        sidecar riding a covered pool drives that pool too."""
        pool_keys = self._backup_pool_keys(snapshot)
        sidecars = {
            spec.pool_name: spec.indices_from_pool
            for spec in self._cache.sidecar_pool_specs
        }
        intents: list[tuple[PoolName, list[str]]] = []
        for pool in sorted(pool_keys, key=self._pool_order):
            keys = pool_keys[pool]
            if pool in sidecars or not keys:
                continue
            riders = [
                name
                for name, source in sidecars.items()
                if source == pool and pool_keys.get(name)
            ]
            if all(
                self._pool_covered(name, pool_keys[name]) for name in (pool, *riders)
            ):
                continue
            intents.append((pool, keys))
        return intents

    def _sizing(
        self, pool: PoolName, keys: list[str]
    ) -> tuple[int, list[PoolTransfer]]:
        """(KV tokens, keys-only aux transfers) an intent stages, for the
        oversize and budget gates."""
        if pool == PoolName.KV:
            return len(keys) * self._cache.page_size, []
        return 0, [PoolTransfer(name=pool, keys=list(keys))]

    def _finish_inflight(self, node_id: NodeId, pool: PoolName) -> None:
        pools = self.inflight_backup_pools.get(node_id)
        if pools is None:
            return
        pools.discard(pool)
        if not pools:
            del self.inflight_backup_pools[node_id]

    @staticmethod
    def _queued_span_key(intent: _UnifiedBackupIntent) -> tuple[NodeId, int]:
        snapshot = intent.snapshot
        return snapshot.node_id, len(snapshot.key)

    def _retain_queued_span(self, intent: _UnifiedBackupIntent) -> None:
        span = self._queued_span_key(intent)
        refs = self._queued_span_refs.get(span, 0)
        if refs == 0:
            self.write_backlog_tokens_ += span[1]
        self._queued_span_refs[span] = refs + 1

    def _release_queued_span(self, intent: _UnifiedBackupIntent) -> None:
        span = self._queued_span_key(intent)
        refs = self._queued_span_refs[span]
        if refs == 1:
            del self._queued_span_refs[span]
            self.write_backlog_tokens_ -= span[1]
        else:
            self._queued_span_refs[span] = refs - 1

    def begin_insert_walk(self) -> None:
        """Admit each node once per insert walk: the trigger chains every
        unbacked ancestor, so a depth-d path is presented d(d+1)/2 times."""
        self._walk_admitted = set()

    def end_insert_walk(self) -> None:
        self._walk_admitted = None

    def enqueue_backup_intent(self, node_id: NodeId) -> None:
        """Queue one intent per pool with something to write; rejected intents
        are counted and the node re-triggers on a later hit."""
        if not self._cache.enable_storage:
            return
        admitted = self._walk_admitted
        if admitted is not None:
            if node_id in admitted:
                return
            admitted.add(node_id)
        snapshot = self._cache.tree_core.snapshot_buffer_backup(
            node_id, self._cache.hicache_storage_pass_prefix_keys
        )
        if snapshot is None:
            return
        intents = self._write_intents(snapshot)
        if not intents:
            return
        pending = self.inflight_backup_pools.get(node_id, set())
        parent_covered = self._backup_parent_covered(
            BufferBackupState(
                parent_node_id=snapshot.parent_node_id,
                parent_is_root=snapshot.parent_is_root,
                parent_last_hash=snapshot.parent_last_hash,
            ),
            node_id,
        )
        page_size = self._cache.page_size
        span = (snapshot.node_id, len(snapshot.key))
        for pool, keys in intents:
            if pool in pending:
                continue
            intent_tokens = len(keys) * page_size
            if (
                span not in self._queued_span_refs
                and self.write_backlog_tokens_ >= self.write_backlog_cap
            ):
                # The cap sits at 2x the intrinsic live-backlog ceiling (see
                # init_hicache), so reaching it means leaked accounting or a
                # broken stale sweep — a bug, not load.
                self._backlog_cap_hits += 1
                if self._backlog_cap_hits <= 3 or self._backlog_cap_hits % 1000 == 0:
                    logger.error(
                        "HiCache write backlog cap hit (occurrence %d): "
                        "backlog=%d cap=%d queue=%d. Live backlog is bounded "
                        "by the device pool span, so this indicates a "
                        "stale-sweep or accounting leak.",
                        self._backlog_cap_hits,
                        self.write_backlog_tokens_,
                        self.write_backlog_cap,
                        len(self.pending_write_queue),
                    )
                self._log_backup_dropped(intent_tokens)
                continue
            # A span larger than the pool's whole staging capacity can never
            # stage; admitting it would wedge the head-of-line queue forever.
            if not parent_covered or self._backup_oversize(*self._sizing(pool, keys)):
                self._log_backup_dropped(intent_tokens)
                continue
            intent = _UnifiedBackupIntent(snapshot=snapshot, pool=pool, keys=list(keys))
            self.pending_write_queue.append(intent)
            self.inflight_backup_pools.setdefault(node_id, set()).add(pool)
            self._retain_queued_span(intent)

    def _backup_oversize(self, kv_tokens: int, aux_xfers: list[PoolTransfer]) -> bool:
        """True if any pool's staging need exceeds that pool's write-usable
        capacity (total for KV, total minus the loads-priority margin for aux
        pools — matching ``_aux_budget_blocked``'s admission ceiling): such an
        intent could never stage and would wedge the FIFO head. Aux pools
        staged through the anchor's host pool count against the KV total."""
        cc = self._cache.cache_controller
        shared_requests = self._shared_host_requests(kv_tokens, aux_xfers)
        if shared_requests is not None:
            pool_tokens = cc.mem_pool_host.size
            max_write_units = pool_tokens - pool_tokens // 10
            if self._host_request_units(kv_tokens, aux_xfers) > max_write_units:
                return True
            return not self._shared_backup_fits(shared_requests, empty=True)
        anchor_pool = cc.mem_pool_host.anchor_entry.host_pool
        anchor_need = kv_tokens
        for t in aux_xfers:
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is None:
                continue
            need = len(t.keys) * entry.host_pool.page_size
            if entry.host_pool is anchor_pool:
                anchor_need += need
            elif need > entry.host_pool.size - self._aux_loads_margin(entry.host_pool):
                return True
        return anchor_need > cc.mem_pool_host.size

    def _aux_loads_margin(self, host_pool) -> int:
        """Aux-pool tokens reserved for loads: at least one trailing window
        (prepare_prefetch allocates its window here and a failed alloc
        forfeits the whole prefetch), plus a 10% burst absorber mirroring
        live_cap."""
        return max(
            self._swa_window_pages * host_pool.page_size,
            host_pool.size // 10,
        )

    def _validate_backup_intent(
        self, intent: _UnifiedBackupIntent
    ) -> Optional[BufferBackupState]:
        # Arena-lookup failure = deleted, key-length mismatch vs the snapshot
        # = split, a None FULL device value = evicted.
        snapshot = intent.snapshot
        return self._cache.tree_core.validate_buffer_backup(
            snapshot.node_id, len(snapshot.key)
        )

    def _split_pieces(
        self, snapshot: BufferBackupSnapshot
    ) -> Optional[list[BufferBackupSnapshot]]:
        """Resolve a split span, parents first, or return None if it is gone.

        The original node retains the tail; its ancestors must cover exactly
        the admitted span and end at the original parent.
        """
        tree_core = self._cache.tree_core
        pass_prefix_keys = self._cache.hicache_storage_pass_prefix_keys
        pieces: list[BufferBackupSnapshot] = []
        node_id, remaining = snapshot.node_id, len(snapshot.key)
        while remaining > 0:
            piece = tree_core.snapshot_buffer_backup(node_id, pass_prefix_keys)
            if piece is None or len(piece.key) > remaining:
                return None
            pieces.append(piece)
            remaining -= len(piece.key)
            node_id = piece.parent_node_id
        if node_id != snapshot.parent_node_id:
            return None
        pieces.reverse()
        return pieces

    def _refresh_pending_backup_intents(self) -> dict[NodeId, BufferBackupState]:
        """Drop dead spans and replace split intents in FIFO order.

        The earliest intent for each (node, pool) wins, including split pieces.
        """
        if not self.pending_write_queue:
            return {}
        page_size = self._cache.page_size
        queued = {(i.snapshot.node_id, i.pool) for i in self.pending_write_queue}
        survivors: dict[tuple[NodeId, PoolName], _UnifiedBackupIntent] = {}
        states: dict[NodeId, BufferBackupState] = {}
        swept_tokens = 0
        for intent in self.pending_write_queue:
            snapshot = intent.snapshot
            key = (snapshot.node_id, intent.pool)
            if key in survivors:
                # The replacement retains this intent's inflight ownership.
                self._release_queued_span(intent)
                continue
            state = self._validate_backup_intent(intent)
            if state is not None:
                survivors[key] = intent
                states[snapshot.node_id] = state
                continue
            self._finish_inflight(snapshot.node_id, intent.pool)
            self._release_queued_span(intent)
            pieces = self._split_pieces(snapshot)
            if pieces is None:
                swept_tokens += len(intent.keys) * page_size
                continue
            for piece in pieces:
                key = (piece.node_id, intent.pool)
                if key in survivors:
                    continue
                if (
                    intent.pool in self.inflight_backup_pools.get(piece.node_id, ())
                    and key not in queued
                ):
                    continue  # staged or awaiting its storage ack
                keys = dict(self._write_intents(piece)).get(intent.pool)
                if not keys:
                    continue  # covered since admission
                repaired = _UnifiedBackupIntent(
                    snapshot=piece, pool=intent.pool, keys=list(keys)
                )
                survivors[key] = repaired
                states[piece.node_id] = BufferBackupState(
                    parent_node_id=piece.parent_node_id,
                    parent_is_root=piece.parent_is_root,
                    parent_last_hash=piece.parent_last_hash,
                )
                self.inflight_backup_pools.setdefault(piece.node_id, set()).add(
                    intent.pool
                )
                self._retain_queued_span(repaired)
        self.pending_write_queue = deque(survivors.values())
        self._log_backup_dropped(swept_tokens)
        return states

    def flush_pending_writes(self) -> None:
        """Stage admitted intents, then submit their D2H as one operation.

        Preparation must not allocate or free L1 KV slots: it only allocates
        host staging, and buffer-mode evict_host is a no-op. Flush before
        returning to scheduler admission, where L1 slots can be reused.
        Each staged source stays locked until its D2H ack.
        """
        if not self.pending_write_queue:
            return
        cc = self._cache.cache_controller
        states = self._refresh_pending_backup_intents()
        # Loads have priority (writes are deferrable): the write window is
        # the pool minus prefetch occupancy minus a 10% margin, floored at
        # the configured fraction.
        pool_tokens = cc.mem_pool_host.size
        live_cap = max(
            int(HICACHE_WRITE_STAGING_POOL_FRACTION * pool_tokens),
            pool_tokens - cc.prefetch_tokens_occupied - pool_tokens // 10,
        )
        while self.pending_write_queue:
            intent = self.pending_write_queue[0]
            snapshot = intent.snapshot
            node_id = snapshot.node_id
            state = states[node_id]
            intent_tokens = len(intent.keys) * self._cache.page_size
            if not self._backup_parent_covered(state, node_id):
                # Cascade a dropped parent down the chain rather than creating
                # a permanent storage hole.
                self._drop_queue_head(intent_tokens, dropped=True)
                continue
            if (
                self.write_staged_tokens_ >= live_cap
                and self._host_request_units(*self._sizing(intent.pool, intent.keys))
                > 0
            ):
                # Wait for current copies to free staging before rebuilding the head.
                break
            launch = self._launch_transfers(intent)
            # The pool may have become covered since admission (an earlier
            # launch this round, a sibling's ack): nothing left to write
            # retires the intent, not a drop.
            if launch is None or all(
                self._pool_covered(pool, keys) for pool, keys in launch[1].items()
            ):
                self._drop_queue_head(intent_tokens, dropped=False)
                continue
            device_value, keys_by_pool, transfers = launch
            sizing_xfers = [t for t in transfers if t.indices_from_pool is None]
            kv_tokens = len(device_value)
            if self._backup_oversize(kv_tokens, sizing_xfers):
                # A permanently unstageable head must not block the queue.
                self._drop_queue_head(intent_tokens, dropped=True)
                continue
            shared_domain = self._shared_host_domain()
            staging_at_limit = (
                self.write_staged_tokens_
                + self._host_request_units(kv_tokens, sizing_xfers)
                > live_cap
                if shared_domain is not None
                else self.write_staged_tokens_ >= live_cap
                and self._host_request_units(kv_tokens, sizing_xfers) > 0
            )
            if staging_at_limit:
                # Yield to live fetch demand; retry next round.
                break
            if self._aux_budget_blocked(kv_tokens, sizing_xfers):
                # An aux pool lacks staging headroom: yield at the gate
                # instead of failing the alloc inside cc.write; acks free
                # aux staging, retry next round.
                break
            if not self._stage_backup_intent(
                intent, device_value, transfers, keys_by_pool
            ):
                # Pool full of in-flight staging and nothing reclaimable
                # (the tree never holds host values in buffer mode):
                # defer, head-of-line; pending acks will free slots.
                break
            self.pending_write_queue.popleft()

        # Submit earlier successes even if a later intent ran out of staging.
        # Do not leave prepared copies deferred across scheduler admission.
        cc.start_writing()

    def _drop_queue_head(self, intent_tokens: int, *, dropped: bool) -> None:
        """Retire the FIFO head without a D2H launch; ``dropped`` feeds the
        dropped-tokens metric (a pool with nothing left to write is not a
        drop)."""
        intent = self.pending_write_queue.popleft()
        self._finish_inflight(intent.snapshot.node_id, intent.pool)
        self._release_queued_span(intent)
        if dropped:
            self._log_backup_dropped(intent_tokens)

    def _build_backup_transfers(
        self, node_id: NodeId, *, kv_only: bool = False
    ) -> tuple[
        torch.Tensor, dict[ComponentType, list[PoolTransfer]], Optional[list[str]]
    ]:
        """The node's D2H spec. A FULL component's BACKUP_HOST KV transfer
        replaces the KV rows (its keys name their pages); its other transfers
        ride as aux pools. A KV intent skips the other components' transfers."""
        tree_core = self._cache.tree_core
        if kv_only:
            device_value = tree_core.get_component_device_value(
                node_id, ComponentType.FULL
            )
            comp_xfers = {}
        else:
            device_value, comp_xfers = tree_core.build_backup_spec(node_id)
        full_xfers = tree_core.build_hicache_transfers(
            ComponentType.FULL, node_id, CacheTransferPhase.BACKUP_HOST
        )
        kv_keys = None
        aux: list[PoolTransfer] = []
        for transfer in full_xfers or ():
            if transfer.name == PoolName.KV:
                device_value = (
                    transfer.device_indices
                    if transfer.device_indices is not None
                    else device_value[:0]
                )
                kv_keys = list(transfer.keys or ())
            else:
                aux.append(transfer)
        if aux:
            comp_xfers[ComponentType.FULL] = aux
        return device_value, comp_xfers, kv_keys

    def _launch_transfers(
        self, intent: _UnifiedBackupIntent
    ) -> Optional[tuple[torch.Tensor, dict[PoolName, list[str]], list[PoolTransfer]]]:
        """(KV rows, keys per written pool, transfers to stage) of one intent,
        off the node's fresh spec; None when nothing is left to write."""
        snapshot = intent.snapshot
        hash_values = snapshot.hash_values
        device_value, comp_xfers, kv_keys = self._build_backup_transfers(
            snapshot.node_id, kv_only=intent.pool == PoolName.KV
        )
        transfers: list[PoolTransfer] = []
        if intent.pool == PoolName.KV:
            keys = hash_values if kv_keys is None else kv_keys
            if not keys or len(device_value) == 0:
                return None
            comp_xfers = {}
        else:
            device_value = device_value[:0]
            keys = []
            for ct, xfers in comp_xfers.items():
                kept = []
                for transfer in xfers:
                    if transfer.name != intent.pool or transfer.indices_from_pool:
                        continue
                    t_keys = self._transfer_keys(transfer, hash_values)
                    if not t_keys:
                        continue
                    if transfer.keys is None:
                        transfer.keys = t_keys
                        transfer.hit_policy = PoolHitPolicy.TRAILING_PAGES
                    kept.append(transfer)
                    keys.extend(t_keys)
                if kept:
                    transfers.extend(kept)
                    comp_xfers[ct] = kept
            if not transfers:
                return None
            comp_xfers = {ct: xfers for ct, xfers in comp_xfers.items() if xfers}
        keys_by_pool = {intent.pool: list(keys)}
        # Sidecars ride their source pool's slots and D2H operation; they
        # allocate no staging of their own.
        for sidecar in self._cache._build_backup_sidecar(device_value, comp_xfers):
            if sidecar.indices_from_pool != intent.pool:
                continue
            sidecar.keys = list(keys)
            keys_by_pool[sidecar.name] = list(keys)
            transfers.append(sidecar)
        return device_value, keys_by_pool, transfers

    def _stage_backup_intent(
        self,
        intent: _UnifiedBackupIntent,
        device_value: torch.Tensor,
        transfers: list[PoolTransfer],
        keys_by_pool: dict[PoolName, list[str]],
    ) -> bool:
        """Allocate host staging and pin one admitted intent's source.

        The caller removes successful intents from pending_write_queue and
        submits them together before returning. Return False when staging
        cannot be allocated.
        """
        cache = self._cache
        cc = cache.cache_controller
        snapshot = intent.snapshot
        # One ack id per write: a node may have several pools in flight.
        self._backup_ack_seq += 1
        ack_id = -(1 << 40) - self._backup_ack_seq
        host_indices = cc.write(
            device_value,
            node_id=ack_id,
            extra_pools=transfers or None,
            flush=False,
        )
        if host_indices is None:
            self._backup_ack_seq -= 1
            return False
        beliefs = cache.storage_existence_cache
        for pool, keys in keys_by_pool.items():
            beliefs.pool(pool).track_inflight(keys)
        # NOTE: no commit_backup — the node must never appear
        # host-resident in buffer mode; staging slots live in the entry.
        lock_params = cache.inc_lock_ref(snapshot.node_id).to_dec_params()
        occupied_units = self.host_allocation_units(host_indices, transfers)
        self.ongoing_write_through[ack_id] = _UnifiedBufferBackupEntry(
            intent=intent,
            host_indices=host_indices,
            aux_xfers=transfers,
            lock_params=lock_params,
            occupied_units=occupied_units,
            keys_by_pool=keys_by_pool,
        )
        self.write_staged_tokens_ += occupied_units
        self._release_queued_span(intent)
        return True

    def _aux_budget_blocked(self, kv_tokens: int, aux: list[PoolTransfer]) -> bool:
        """True when an aux pool cannot stage this intent right now (free
        minus the loads-priority margin falls short of the need): defer at
        the gate instead of failing the alloc inside cc.write and blocking
        pure-KV intents behind an unallocatable head. The margin enforces
        loads-have-priority on aux pools the way live_cap does on the KV
        pool; avail already reflects prefetch-held slots, so no occupancy
        subtraction here. Aux pools staged through the anchor's host pool
        are gated by live_cap and the allocation itself."""
        cc = self._cache.cache_controller
        shared_requests = self._shared_host_requests(kv_tokens, aux)
        if shared_requests is not None:
            return not self._shared_backup_fits(shared_requests)
        anchor_pool = cc.mem_pool_host.anchor_entry.host_pool
        for t in aux:
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is None or entry.host_pool is anchor_pool:
                continue
            need = len(t.keys) * entry.host_pool.page_size
            headroom = entry.host_pool.available_size() - self._aux_loads_margin(
                entry.host_pool
            )
            if need > headroom:
                return True
        return False

    def finish_backup_ack(self, ack_id: int) -> None:
        """D2H confirmed: drop the device lock and enqueue the storage write
        (which reads from the staging copy, so device eviction may proceed)."""
        entry = self.ongoing_write_through.pop(ack_id)
        intent = entry.intent
        snapshot = intent.snapshot
        self._cache.dec_lock_ref(snapshot.node_id, entry.lock_params)

        storage_xfers = self._storage_transfers(entry.aux_xfers)
        # Another pool's write carries no KV pages; its keys ride the transfer.
        hash_value = (
            list(entry.keys_by_pool[PoolName.KV]) if intent.pool == PoolName.KV else []
        )
        operation_id = self._cache.cache_controller.write_storage(
            entry.host_indices,
            snapshot.key.token_ids,
            hash_value,
            snapshot.prefix_keys,
            extra_pools=storage_xfers or None,
        )
        self.ongoing_backup[operation_id] = entry

    @staticmethod
    def _storage_transfers(staged: list[PoolTransfer]) -> list[PoolTransfer]:
        """One storage transfer per pool: a pool staged in several transfers
        becomes one contiguous key span over their slots; a derived sidecar
        writes its source's key span over the source's slots."""
        merged: dict[PoolName, PoolTransfer] = {}
        for transfer in staged:
            if transfer.indices_from_pool is not None:
                merged[transfer.name] = PoolTransfer(
                    name=transfer.name,
                    keys=list(transfer.keys),
                    hit_policy=transfer.hit_policy,
                    indices_from_pool=transfer.indices_from_pool,
                )
                continue
            earlier = merged.get(transfer.name)
            if earlier is None:
                merged[transfer.name] = PoolTransfer(
                    name=transfer.name,
                    host_indices=transfer.host_indices,
                    keys=list(transfer.keys),
                    hit_policy=transfer.hit_policy,
                )
                continue
            earlier.keys.extend(transfer.keys)
            earlier.host_indices = torch.cat(
                [earlier.host_indices, transfer.host_indices]
            )
        return list(merged.values())

    def finish_storage_write_ack(self, operation_id: int) -> None:
        """Storage ack (rank-synced drain): free the staging and feed each
        written pool's beliefs unconditionally, keeping admission
        TP-deterministic under per-rank backend failures."""
        entry = self.ongoing_backup.pop(operation_id, None)
        if entry is None:
            return
        beliefs = self._cache.storage_existence_cache
        for pool, keys in entry.keys_by_pool.items():
            pool_beliefs = beliefs.pool(pool)
            pool_beliefs.add(keys)
            pool_beliefs.untrack_inflight(keys)
        self._free_staging_now(entry.host_indices, entry.aux_xfers)
        self.write_staged_tokens_ -= entry.occupied_units
        self._finish_inflight(entry.intent.snapshot.node_id, entry.intent.pool)

    def _free_staging_now(
        self, host_indices: torch.Tensor, aux_xfers: list[PoolTransfer]
    ) -> None:
        """Synchronously free a staging span (KV + aux pools) on the
        scheduler thread; buffer-mode acks/drops all run here, so frees
        land before the tick's next gate reads pool availability."""
        cc = self._cache.cache_controller
        if host_indices is not None and host_indices.numel() > 0:
            cc.mem_pool_host.free(host_indices)
        for t in aux_xfers or ():
            if (
                t.host_indices is None
                or t.host_indices.numel() == 0
                or t.indices_from_pool is not None
            ):
                continue
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is not None:
                entry.host_pool.free(t.host_indices)

    # ---- load back pipeline (storage -> staging -> device) ----

    def try_lock_anchor(
        self, request: CacheRequestHandle, remaining_full_tokens: int
    ) -> tuple[str, int]:
        """Pin the staged prefetch's device anchor so eviction cannot
        invalidate the splice, finding it by re-matching the live tree
        (carried node ids go stale via splits and eviction; the walk is
        O(request hit span)). Returns "locked", "no_anchor" (nothing to pin),
        "cap_skip" (bigger than the whole cap; launches unlocked), "cap_busy"
        (fits, but the budget is taken; the caller parks), or "anchor_lost"
        (splice base gone -- the caller re-plans)."""
        assert request not in self.anchor_locks, (
            f"prefetch anchor already locked: {request.rid}"
        )
        prefix_tokens, extra_key, cache_salt = self._prefetch_prefix_ctx[request]
        matched_len = len(prefix_tokens)
        assert matched_len + remaining_full_tokens > 0, (
            f"empty prefetch span: {request.rid}"
        )
        full_key_tokens = array("q", prefix_tokens)
        if remaining_full_tokens or self._cache.tree_core.is_eagle:
            info = self._cache.ongoing_prefetch[request]
            raw_len = remaining_full_tokens + int(info.prefetch_key.is_bigram)
            full_key_tokens.extend(info.prefetch_key.token_ids[:raw_len])
        cache = self._cache
        matched_full, anchor_node, anchor_tokens = (
            cache.tree_core.match_full_device_prefix(
                RadixKey(
                    full_key_tokens,
                    extra_key=extra_key,
                    is_bigram=cache.tree_core.is_eagle,
                    cache_salt=cache_salt,
                )
            )
        )
        if matched_full < matched_len:
            return "anchor_lost", matched_full
        if anchor_tokens == 0:
            return "no_anchor", matched_full
        if self.anchor_locked_tokens_ + anchor_tokens > self.anchor_lock_cap_tokens:
            # Parking for a pin no drain can ever satisfy deadlocks the
            # request, so only a pin that still fits the cap is worth a wait.
            over_cap = anchor_tokens > self.anchor_lock_cap_tokens
            self._anchor_lock_cap_skips += 1
            if (
                self._anchor_lock_cap_skips <= 3
                or self._anchor_lock_cap_skips % 1000 == 0
            ):
                logger.warning(
                    "HiCache anchor-lock cap reached (skip %d): locked=%d "
                    "want=%d cap=%d; %s.",
                    self._anchor_lock_cap_skips,
                    self.anchor_locked_tokens_,
                    anchor_tokens,
                    self.anchor_lock_cap_tokens,
                    "launching unlocked" if over_cap else "parking",
                )
            return ("cap_skip" if over_cap else "cap_busy"), matched_full
        cache.tree_core.inc_full_pin(anchor_node)
        self.anchor_locks[request] = _AnchorLock(
            node_id=anchor_node,
            tokens=anchor_tokens,
        )
        self.anchor_locked_tokens_ += anchor_tokens
        return "locked", matched_full

    def release_anchor_lock(self, request: CacheRequestHandle) -> None:
        """Drop a staged prefetch's anchor lock (idempotent; called at every
        consume/drop/abort exit)."""
        lock = self.anchor_locks.pop(request, None)
        if lock is None:
            return
        self._cache.tree_core.dec_full_pin(lock.node_id)
        self.anchor_locked_tokens_ -= lock.tokens
        assert self.anchor_locked_tokens_ >= 0, (
            f"anchor-lock accounting corrupted: locked={self.anchor_locked_tokens_} "
            f"after releasing {request.rid}"
        )

    def staged_span_covered(
        self, request: CacheRequestHandle, span_tokens: int
    ) -> bool:
        """True when the live device tree already covers the fetch's whole
        would-be span (prefix + the storage-hit tokens): nothing would be
        left to splice at consumption, so the IO-commit caller cancels
        before the bounce alloc and the storage read."""
        info = self._cache.ongoing_prefetch.get(request)
        if info is None or span_tokens <= 0:
            return False
        prefix_tokens, _, _ = self._prefetch_prefix_ctx[request]
        span_key = info.prefetch_key
        full_tokens = array("q", prefix_tokens)
        full_tokens.extend(span_key[:span_tokens].token_ids)
        key = RadixKey(
            full_tokens,
            extra_key=span_key.extra_key,
            is_bigram=self._cache.tree_core.is_eagle,
            cache_salt=span_key.cache_salt,
        )
        match = self._cache.match_prefix(MatchPrefixParams(key=key))
        return match.device_prefix_len >= len(key)

    def set_prefix_ctx(
        self,
        request: CacheRequestHandle,
        matched_prefix_tokens,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ) -> None:
        """Record the device-matched prefix (and its tree-key namespace) at
        prefetch enqueue; consumed at staging commit to build the full-span
        tree key, and by try_lock_anchor to re-match a stale anchor."""
        self._prefetch_prefix_ctx[request] = (
            list(matched_prefix_tokens or []),
            extra_key,
            cache_salt,
        )

    def pop_prefix_ctx(self, request: CacheRequestHandle) -> None:
        self._prefetch_prefix_ctx.pop(request, None)

    def has_staged(self, request: CacheRequestHandle) -> bool:
        return request in self.staged_prefetches

    def prepare_staged_prefetch(self, req: Req) -> bool:
        """Rebuild the admission plan from this pass's joint FULL/SWA match."""
        req.staged_prefetch_plan = None
        f = self.staged_prefetches.get(req.cache_request_handle)
        if f is None:
            if not (req.host_hit_is_storage and req.host_loaded_length > 0):
                self._clip_storage_hit(req)
            return True
        joint_len = req.prefix_len
        # Preserve the completed L3 span, including peer-covered KV. An aux
        # fetch can also make resident FULL before that span reusable.
        req.storage_hit_start = min(f.matched_len, joint_len)
        req.storage_hit_length = f.matched_len + f.num_tokens - req.storage_hit_start
        if joint_len >= f.matched_len + f.num_tokens:
            # The joint match already covers the staged span; a shorter FULL-only
            # prefix would strand the slots recomputed below cache_protected_len.
            self._resolve_device_covered(req)
            return True
        key = RadixKey(
            f.key_tokens,
            extra_key=f.extra_key,
            is_bigram=self._cache.tree_core.is_eagle,
            cache_salt=f.cache_salt,
        )
        matched_len, node_id, _ = self._cache.tree_core.match_full_device_prefix(key)
        if matched_len < f.matched_len:
            logger.warning(
                "HiCache staged prefetch deferred req=%s reason=shrunk "
                "matched=%d now=%d tokens=%d",
                req.rid,
                f.matched_len,
                matched_len,
                f.num_tokens,
            )
            self._refetch_staged(f)
            return False
        req.prefix_len = matched_len
        req.last_node = node_id
        req.kv.cache_protected_len = matched_len
        full_tokens = max(0, f.matched_len + f.num_tokens - matched_len)
        swa_tokens = sum(
            len(t.host_indices)
            for t in f.aux_xfers
            if t.name == PoolName.SWA and t.host_indices is not None
        )
        if full_tokens == 0 and swa_tokens == 0:
            self._resolve_device_covered(req)
            return True
        req.host_hit_length = full_tokens
        req.swa_host_hit_length = swa_tokens
        req.host_hit_is_storage = True
        req.staged_prefetch_plan = StagedPrefetchPlan(
            f.operation_id, key, matched_len, full_tokens, swa_tokens
        )
        return True

    def _resolve_device_covered(self, req: Req) -> None:
        req.host_hit_length = 0
        req.swa_host_hit_length = 0
        self._clip_storage_hit(req)
        self.release_staged_hold(req.cache_request_handle, reason="device_covered")

    @staticmethod
    def _clip_storage_hit(req: Req) -> None:
        req.storage_hit_length = req.fulfilled_storage_hit_len(req.prefix_len)
        req.host_hit_is_storage = False

    def _refetch_staged(self, f: _StagedPrefetch) -> None:
        self.release_staged_hold(f.request, reason="shrunk")
        self._cache.storage_prefetch_retries.refetch(
            f.request.rid, f.matched_len + f.num_tokens
        )

    def stage_completed_prefetch(
        self,
        request: CacheRequestHandle,
        num_tokens: int,
        hash_value: list[str],
    ) -> bool:
        """Park the completed fetch as a held bounce; the scheduler surfaces
        it as host_hit_length and the adder consumes it via init_load_back.
        Always returns True (ready is a stable, revisited state)."""
        cache = self._cache
        info = cache.ongoing_prefetch.pop(request)
        prefetch_key = info.prefetch_key
        host_indices = info.host_indices
        operation = info.operation
        comp_xfers = info.comp_xfers
        cc = cache.cache_controller
        prefix_ctx = self._prefetch_prefix_ctx.pop(request, None)
        prefix_tokens = prefix_ctx[0] if prefix_ctx is not None else None
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        # Component transfers are already present in comp_xfers.  Preserve the
        # derived sidecars from the storage operation as well; cc.load resolves
        # them against the freshly allocated source host/device indices.
        aux_xfers.extend(
            transfer
            for transfer in operation.pool_transfers or ()
            if transfer.indices_from_pool is not None
        )
        occupied_tokens = operation.buffer_host_occupied_units
        if occupied_tokens is None:
            occupied_tokens = self.host_allocation_units(host_indices, aux_xfers)

        has_aux = any(
            t.host_indices is not None and t.host_indices.numel() > 0 for t in aux_xfers
        )
        if (num_tokens == 0 and not has_aux) or prefix_tokens is None:
            # Nothing usable fetched: recompute.
            cache.discard_storage_prefetch_accounting(request)
            self.release_anchor_lock(request)
            cc.append_host_mem_release(
                host_indices[:num_tokens], extra_pools=aux_xfers or None
            )
            cc.prefetch_tokens_occupied -= occupied_tokens
            cache.prefetch_loaded_tokens_by_reqid[request] = 0
            cache.prefetch_loaded_storage_start_by_reqid.pop(request, None)
            return True

        staged_pages = num_tokens // cache.page_size
        staged_hashes = hash_value[:staged_pages]
        staged_kv = host_indices[:num_tokens]
        # The fetch is evidence for every pool it delivered (sound even if the
        # staged prefetch is later dropped): KV-derived sidecars came with the
        # staged pages, other pools with the keys their transfer carries.
        beliefs = cache.storage_existence_cache
        beliefs.pool(PoolName.KV).add(staged_hashes)
        for transfer in aux_xfers:
            if transfer.indices_from_pool == PoolName.KV:
                beliefs.pool(transfer.name).add(staged_hashes)
            elif transfer.keys:
                beliefs.pool(transfer.name).add(transfer.keys)
        self.staged_prefetches[request] = _StagedPrefetch(
            request=request,
            key_tokens=array(
                "q",
                prefix_tokens
                + list(
                    prefetch_key.token_ids[: num_tokens + int(prefetch_key.is_bigram)]
                ),
            ),
            extra_key=prefetch_key.extra_key,
            cache_salt=prefetch_key.cache_salt,
            matched_len=len(prefix_tokens),
            num_tokens=num_tokens,
            occupied_tokens=occupied_tokens,
            host_indices=staged_kv,
            aux_xfers=aux_xfers,
            hash_values=staged_hashes,
            operation_id=operation.id,
        )
        cache.prefetch_loaded_tokens_by_reqid[request] = num_tokens
        cache.prefetch_loaded_storage_start_by_reqid[request] = operation.storage_start
        return True

    def init_load_back(
        self, params: InitLoadBackParams
    ) -> Optional[tuple[int, NodeId]]:
        """Materialize a selected prefill under the caller's prefix lock.

        The caller has finished selecting its prefill shape and must acquire
        the request lock after success, without further admission gates. The
        prefix lock protects allocation-time eviction. None retains staging
        and its anchor for the next admission attempt.

        Ownership contract: cc.load queues the H2D before insert adjudicates
        ownership, so the prepared boundary must ensure the insert can only
        ADD nodes — a dedup would free slots the in-flight copy still
        targets (queued use-after-free)."""
        cache = self._cache
        req = params.req
        assert req is not None
        request = req.cache_request_handle
        unchanged = (0, req.last_node)
        f = self.staged_prefetches.get(request)
        if f is None:
            self.release_anchor_lock(request)
            return unchanged
        cc = cache.cache_controller
        plan = req.staged_prefetch_plan
        assert plan is not None, f"staged prefetch was not planned for {req.rid}"
        assert f.operation_id == plan.operation_id
        assert (f.extra_key, f.cache_salt) == (req.extra_key, req.cache_salt)
        assert (req.host_hit_length, req.swa_host_hit_length) == (
            plan.full_tokens,
            plan.swa_tokens,
        ), f"staged load-back budget changed for {req.rid}"

        def _defer_for_capacity(pool: str) -> None:
            defers = self._staged_admission_defers.get(request, 0) + 1
            self._staged_admission_defers[request] = defers
            cache._log_storage_prefetch_deferred(f.num_tokens, "device_capacity")
            if defers < self.max_staged_admission_defers:
                logger.warning(
                    "HiCache staged prefetch deferred at admission req=%s "
                    "reason=device_capacity pool=%s tokens=%d defers=%d",
                    req.rid,
                    pool,
                    f.num_tokens,
                    defers,
                )
                return
            # Still unmaterializable: drop the hold so the admission loop stops
            # breaking on this request, which recomputes on its next pass.
            logger.warning(
                "HiCache staged prefetch dropped after %d device_capacity "
                "deferrals req=%s pool=%s tokens=%d",
                defers,
                req.rid,
                pool,
                f.num_tokens,
            )
            self.release_staged_hold(request, reason="device_capacity")
            req.staged_prefetch_plan = None

        splice_base = plan.device_prefix_len
        assert req.prefix_len == splice_base
        prefix_indices = self._cache.prefix_device_indices(req)
        trim_tokens = splice_base - f.matched_len
        assert trim_tokens % cache.page_size == 0, (
            f"staged splice trim not page-aligned req={req.rid}: "
            f"matched={f.matched_len} splice_base={splice_base}"
        )

        key = plan.key
        span_end = f.matched_len + f.num_tokens
        load_tokens = plan.full_tokens

        # Evict-before-alloc (mirrors _load_back_transfers): the budget gate
        # counts evictable pages, but cc.load draws from free slots only.
        if cache.supports_swa():
            avail = cache.token_to_kv_pool_allocator.full_available_size()
        else:
            avail = cache.token_to_kv_pool_allocator.available_size()
        if avail < load_tokens:
            needed = load_tokens - avail
            cache.evict_for_alloc(EvictParams(num_tokens=needed))
            if cache.supports_swa():
                avail = cache.token_to_kv_pool_allocator.full_available_size()
            else:
                avail = cache.token_to_kv_pool_allocator.available_size()
            if avail < load_tokens:
                return _defer_for_capacity("full")

        load_back_id = -(f.operation_id) - 1
        # The full trailing-window aux transfer is independent of the shorter
        # FULL suffix and may remain nonempty for an aux-only load.
        load_xfers = list(f.aux_xfers)
        staged_swa = next(
            (
                len(t.host_indices)
                for t in load_xfers
                if t.name == PoolName.SWA and t.host_indices is not None
            ),
            0,
        )

        device_indices = cc.load(
            host_indices=f.host_indices[trim_tokens:],
            node_id=load_back_id,
            extra_pools=load_xfers or None,
        )
        if device_indices is None:
            # load() allocates all pools atomically before queueing H2D, so the
            # staged host buffers remain reusable after either pool is short.
            return _defer_for_capacity("full_or_aux")
        del self.staged_prefetches[request]
        self._staged_admission_defers.pop(request, None)
        req.staged_prefetch_plan = None
        cache._settle_storage_prefetch_hit(
            request, credited_tokens=req.storage_hit_length
        )

        swa_dev = next(
            (
                t.device_indices
                for t in load_xfers
                if t.name == PoolName.SWA
                and t.device_indices is not None
                and t.device_indices.numel() > 0
            ),
            None,
        )
        aux_device_releases: list[tuple[PoolName, torch.Tensor]] = []
        if swa_dev is not None:
            # SWA loads into fresh reservations. Register the window's FULL->SWA
            # translation now (attention reads through it); slots another request
            # still holds are kept, and their redundant destinations freed at ack.
            full_window = torch.cat([prefix_indices, device_indices])[-len(swa_dev) :]
            allocator = cache.token_to_kv_pool_allocator
            old_swa = allocator.translate_swa_indices_for_transfer(full_window)
            missing = old_swa <= 0
            window_start = span_end - len(swa_dev)
            repair_end = min(splice_base, span_end)
            tree_missing = torch.zeros_like(missing)
            tail_start = max(splice_base, window_start)
            tree_missing[tail_start - window_start :] = True
            repair_ranges = []
            if window_start < repair_end:
                repair_ranges = cache.tree_core.swa_tombstone_ranges(
                    key, window_start, repair_end
                )
                for repair_start, repair_end_ in repair_ranges:
                    repair_slice = slice(
                        repair_start - window_start, repair_end_ - window_start
                    )
                    tree_missing[repair_slice] = True
            assert torch.equal(missing, tree_missing), (
                "SWA tree and allocator residency disagree for restored window "
                f"[{window_start}, {span_end})"
            )
            for repair_start, repair_end_ in repair_ranges:
                repair_slice = slice(
                    repair_start - window_start, repair_end_ - window_start
                )
                for action in cache.tree_core.attach_swa_window(
                    key,
                    repair_start,
                    repair_end_,
                    swa_dev[repair_slice],
                ):
                    cache._apply_cache_action(action)
            if bool(missing.any()):
                cache._apply_cache_action(
                    RebuildFullToSWAMapping(
                        [full_window[missing]],
                        [swa_dev[missing]],
                    )
                )
            if bool((~missing).any()):
                aux_device_releases.append((PoolName.SWA, swa_dev[~missing]))

        # Publish via a plain insert under the admission lock choreography;
        # the caller's request lock then pins the span (load_back pattern).
        # prev_prefix_len covers the already-device-resident head.
        insert_result = cache.insert(
            InsertParams(
                key=key,
                value=torch.cat([prefix_indices, device_indices]),
                prev_prefix_len=splice_base,
                component_evicted_seqlens={
                    ComponentType.SWA: (span_end - staged_swa) if staged_swa else 0
                },
            )
        )
        self.ongoing_buffer_load_back[load_back_id] = _OngoingBufferLoadBack(
            request=f.request,
            num_tokens=load_tokens,
            occupied_tokens=f.occupied_tokens,
            aux_xfers=f.aux_xfers,
            # The full staged bounce (not the trimmed H2D source): the ack
            # frees it whole, trimmed head included.
            host_indices=f.host_indices,
            hash_values=f.hash_values,
            aux_device_releases=aux_device_releases,
        )
        match = cache.match_prefix(MatchPrefixParams(key=key))
        canonical = cache.path_device_indices(match.last_device_node)[
            splice_base:span_end
        ]
        self.release_anchor_lock(request)
        owned = match.device_prefix_len >= span_end and torch.equal(
            canonical, device_indices
        )
        if not owned:
            # Fail-stop: the insert freed or replaced slots the in-flight H2D
            # still targets; continuing risks silent KV corruption.
            raise RuntimeError(
                "HiCache buffer load-back ownership violation "
                f"req={f.request.rid}: "
                f"insert prefix_len={insert_result.prefix_len} "
                f"expected={splice_base}, matched={match.device_prefix_len} "
                f"span_end={span_end} splice_base={splice_base}; "
                f"in-flight H2D targets freed slots"
            )
        return len(canonical), match.last_device_node

    def try_finish_load_back(self, ack_id: int) -> bool:
        """Fill ack: free the host bounce and return True when the ack id is
        a buffer-mode load-back. The span was published at admission; the
        ack never touches the tree (existence beliefs were fed from the
        storage-fetched pages at staging commit)."""
        f = self.ongoing_buffer_load_back.pop(ack_id, None)
        if f is None:
            return False
        cache = self._cache
        cc = cache.cache_controller

        # The H2D consumed the bounce buffers; free them outright.
        self._free_staging_now(f.host_indices, f.aux_xfers)
        for pool_name, device_indices in f.aux_device_releases:
            entry = cc.mem_pool_host.entry_map[pool_name]
            from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
                UnifiedSWAAllocatorBase,
            )

            allocator = cache.token_to_kv_pool_allocator
            if pool_name == PoolName.SWA and isinstance(
                allocator, UnifiedSWAAllocatorBase
            ):
                # H2D has completed; these redundant slots are no longer pending.
                allocator.swa_attn_allocator.free_physical(device_indices)
            else:
                free_fn = entry.device_free_fn or entry.device_pool.free
                free_fn(device_indices)

        cc.prefetch_tokens_occupied -= f.occupied_tokens
        logger.info(
            "HiCache prefetch fill committed req=%s filled=%d occupied=%d locked=%d",
            f.request.rid,
            f.num_tokens,
            cc.prefetch_tokens_occupied,
            self.anchor_locked_tokens_,
        )
        cache._finish_storage_prefetch(
            f.request, fulfilled_tokens=f.num_tokens, reason=None
        )
        return True

    def release_staged_hold(
        self, request: CacheRequestHandle, reason: Optional[str] = None
    ) -> bool:
        """Free a staged hold outright — anchor pin, host bounce (KV + aux),
        occupancy grant; nothing device-side exists yet. Called for aborts
        and for holds that can no longer splice. Returns True when a hold
        existed."""
        self.release_anchor_lock(request)
        self._staged_admission_defers.pop(request, None)
        staged = self.staged_prefetches.pop(request, None)
        if staged is None:
            return False
        self._cache._finish_storage_prefetch(request, fulfilled_tokens=0, reason=reason)
        self._free_staging_now(staged.host_indices, staged.aux_xfers)
        self._cache.cache_controller.prefetch_tokens_occupied -= staged.occupied_tokens
        return True
