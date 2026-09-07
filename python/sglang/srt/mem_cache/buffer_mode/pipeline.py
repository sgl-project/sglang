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
from sglang.srt.mem_cache.radix_cache import RadixKey
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
    from sglang.srt.mem_cache.pool_host import HostPoolGroup
    from sglang.srt.mem_cache.unified_cache.components import SWAComponent
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class _UnifiedBackupIntent(msgspec.Struct):
    """Buffer-mode backup intent, unpinned while queued.

    Snapshots node identity at enqueue time: a split rewrites the node's
    key/hash in place while these copies stay intact, so a key-length change
    detects a split and a missing FULL device value detects eviction
    (``_validate_backup_intent``).
    """

    snapshot: BufferBackupSnapshot


class _UnifiedBufferBackupEntry(msgspec.Struct):
    """A buffer-mode backup after its D2H launch: intent + staging slots.

    ``host_indices`` are FULL-pool staging slots; ``aux_xfers`` carry the
    staged aux-pool slots (e.g. the SWA window). All are freed at the
    storage-write ack — host memory is never retained as a cache tier.
    """

    intent: _UnifiedBackupIntent
    host_indices: torch.Tensor
    aux_xfers: list[PoolTransfer]
    lock_params: DecLockRefParams


class _StagedPrefetch(msgspec.Struct):
    """A completed buffer-mode fetch parked until prefill admission: only
    the op-owned host bounce exists (no device state, nothing in the tree).
    """

    req_id: str
    key_tokens: list[int]
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
    tree-resident; only the host bounce remains to free.
    """

    req_id: str
    num_tokens: int
    occupied_tokens: int
    aux_xfers: list[PoolTransfer]
    host_indices: torch.Tensor
    hash_values: list[str]


class _AnchorLock(msgspec.Struct):
    """Pins a staged prefetch's device anchor from IO commit to consumption."""

    node_id: NodeId
    lock_params: DecLockRefParams
    tokens: int


def _track_content_refs(refs: dict[str, int], hash_values: list[str]) -> None:
    """Add one content ref per page hash (at D2H launch). Refcounted,
    not a flag: several launched entries can carry the same content
    (duplicate staging of republished spans)."""
    for h in hash_values:
        refs[h] = refs.get(h, 0) + 1


def _untrack_content_refs(refs: dict[str, int], hash_values: list[str]) -> None:
    """Drop one content ref per page hash (at storage-ack)."""
    for h in hash_values:
        n = refs.get(h, 0) - 1
        if n <= 0:
            refs.pop(h, None)
        else:
            refs[h] = n


def staged_splice_tokens(f: _StagedPrefetch, device_prefix_len: int) -> int:
    """Tokens a staged prefetch can still splice beyond the live device
    prefix; 0 = unusable hold (prefix shrunk below the span, span fully
    device-resident, or the trim would cut into a staged aux trailing
    window — aux pools splice whole or not at all)."""
    span_end = f.matched_len + f.num_tokens
    if device_prefix_len < f.matched_len or device_prefix_len >= span_end:
        return 0
    splice_tokens = span_end - device_prefix_len
    for t in f.aux_xfers:
        if t.host_indices is not None and t.host_indices.numel() > splice_tokens:
            return 0
    return splice_tokens


def validate_buffer_only_stack(
    sidecar_pool_specs: list[SidecarPoolSpec],
    host_pool_group: HostPoolGroup,
    swa_component: Optional[SWAComponent],
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
        if swa._swa_kv_pool_host.size < 2 * window_tokens:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only requires an SWA "
                f"host pool of at least two trailing windows "
                f"({2 * window_tokens} tokens; got "
                f"{swa._swa_kv_pool_host.size}): one staging a write "
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
        # Metadata-only pending-write backlog cap; beyond it new intents
        # are dropped at admission (re-trigger on a later hit).
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
        logger.info(
            "BufferModePipeline anchor_lock_cap_tokens=%d",
            self.anchor_lock_cap_tokens,
        )
        self.reset()

    def reset(self) -> None:
        # Load pipeline: hits awaiting a staging grant (park-and-retry),
        # enqueue-time prefix context, completed prefetches staged until
        # prefill admission, and load-backs in flight (keyed by synthetic
        # negative ack id).
        self.pending_hit_allocs: deque = deque()
        self._prefetch_prefix_ctx: dict[
            str, tuple[list[int], Optional[str], Optional[str]]
        ] = {}
        self.staged_prefetches: dict[str, _StagedPrefetch] = {}
        self.ongoing_buffer_load_back: dict[int, _OngoingBufferLoadBack] = {}
        # Backup pipeline: FIFO intents awaiting a D2H slot, node ids
        # anywhere in flight (dedupes re-triggers), and a content refcount
        # of every page hash between D2H launch and storage-ack — admission
        # skips content covered by beliefs + launched writes.
        self.pending_write_queue: deque[_UnifiedBackupIntent] = deque()
        self.inflight_backup_node_ids: set[int] = set()
        self.inflight_backup_hashes: dict[str, int] = {}
        # Backups between D2H launch and D2H ack (keyed by node id), then
        # between storage-write launch and storage ack (keyed by operation
        # id). Mirrors the cache-mode ongoing_write_through/ongoing_backup
        # stages, with buffer entries.
        self.ongoing_write_through: dict[int, _UnifiedBufferBackupEntry] = {}
        self.ongoing_backup: dict[int, _UnifiedBufferBackupEntry] = {}
        self.write_staged_tokens_ = 0
        self.write_backlog_tokens_ = 0
        self._backlog_cap_hits = 0
        # rid-keyed anchor locks; released idempotently at every exit.
        self.anchor_locks: dict[str, _AnchorLock] = {}
        self.anchor_locked_tokens_ = 0
        self._anchor_lock_cap_skips = 0

    def is_idle(self) -> bool:
        """No pending or in-flight buffer-mode transfer work."""
        return not (
            self.pending_hit_allocs
            or self.staged_prefetches
            or self.ongoing_buffer_load_back
            or self.pending_write_queue
            or self.inflight_backup_node_ids
            or self.ongoing_write_through
            or self.ongoing_backup
        )

    # ---- backup pipeline (device -> staging -> storage) ----

    def _backup_parent_covered(self, state: BufferBackupState) -> bool:
        """Only admit a node whose parent is stored/in-flight: writing above
        a dropped parent creates a permanent longest-prefix hole."""
        if (
            state.parent_is_root
            or state.parent_node_id in self.inflight_backup_node_ids
        ):
            return True
        return (
            state.parent_last_hash is not None
            and self._cache.storage_existence_cache.contains(
                PoolName.KV, state.parent_last_hash
            )
        )

    def _log_backup_dropped(self, num_tokens: int) -> None:
        cache = self._cache
        if cache.enable_storage_metrics and cache.storage_metrics_collector is not None:
            cache.storage_metrics_collector.log_backup_dropped_tokens(num_tokens)

    def enqueue_backup_intent(self, node_id: NodeId) -> None:
        """Snapshot a backup intent and commit it to the write queue.
        Admission gates: belief skip, parent-cover, backlog cap, oversize.
        Rejected intents are counted; the node re-triggers on a later hit."""
        if not self._cache.enable_storage:
            return
        if node_id in self.inflight_backup_node_ids:
            return
        snapshot = self._cache.tree_core.snapshot_buffer_backup(
            node_id, self._cache.hicache_storage_pass_prefix_keys
        )
        if snapshot is None:
            return
        # Admission cover: beliefs plus content past its D2H launch. The
        # launched cover keeps republished content (fill inserts under new
        # node ids) from re-writing while the original write drains.
        if self._cache.storage_existence_cache.covers_all(
            PoolName.KV,
            snapshot.hash_values,
            extra_cover=self.inflight_backup_hashes,
        ):
            return
        intent_tokens = len(snapshot.hash_values) * self._cache.page_size
        if self.write_backlog_tokens_ >= self.write_backlog_cap:
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
            return
        # A span larger than any pool's whole staging capacity can never
        # stage; admitting it would wedge the head-of-line queue forever.
        state = BufferBackupState(
            parent_node_id=snapshot.parent_node_id,
            parent_is_root=snapshot.parent_is_root,
            parent_last_hash=snapshot.parent_last_hash,
        )
        if not self._backup_parent_covered(state) or self._backup_oversize(
            snapshot.node_id, snapshot.hash_values, intent_tokens
        ):
            self._log_backup_dropped(intent_tokens)
            return

        intent = _UnifiedBackupIntent(snapshot=snapshot)
        self.pending_write_queue.append(intent)
        self.inflight_backup_node_ids.add(snapshot.node_id)
        self.write_backlog_tokens_ += intent_tokens

    def _build_aux_staging_transfers(
        self,
        node_id: NodeId,
        hash_values: list[str],
        comp_xfers: Optional[dict[ComponentType, list[PoolTransfer]]] = None,
    ) -> list[PoolTransfer]:
        """Keys-only aux transfers mirroring what BACKUP_STORAGE would write;
        sizes the per-pool oversize gate (beliefs do not consult these)."""
        transfers: list[PoolTransfer] = []
        if ComponentType.SWA in self._cache.components:
            current = (
                comp_xfers.get(ComponentType.SWA)
                if comp_xfers is not None
                else self._cache.tree_core.build_hicache_transfers(
                    ComponentType.SWA,
                    node_id,
                    CacheTransferPhase.BACKUP_HOST,
                )
            )
            for transfer in current or ():
                if transfer.device_indices is None:
                    continue
                num_pages = len(transfer.device_indices) // self._cache.page_size
                if num_pages > 0:
                    transfers.append(
                        PoolTransfer(
                            name=PoolName.SWA,
                            keys=hash_values[-num_pages:],
                            hit_policy=PoolHitPolicy.TRAILING_PAGES,
                        )
                    )
        return transfers

    def _backup_oversize(
        self,
        node_id: NodeId,
        hash_values: list[str],
        intent_tokens: int,
        aux_xfers: Optional[list[PoolTransfer]] = None,
    ) -> bool:
        """True if any pool's staging need exceeds that pool's write-usable
        capacity (total for KV, total minus the loads-priority margin for aux
        pools — matching ``_aux_budget_blocked``'s admission ceiling): such an
        intent could never stage and would wedge the FIFO head."""
        cc = self._cache.cache_controller
        if intent_tokens > cc.mem_pool_host.size:
            return True
        if aux_xfers is None:
            aux_xfers = self._build_aux_staging_transfers(node_id, hash_values)
        for t in aux_xfers or ():
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is not None and (
                len(t.keys) * entry.host_pool.page_size
                > entry.host_pool.size - self._aux_loads_margin(entry.host_pool)
            ):
                return True
        return False

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
        # = split, a None FULL device value = evicted. Stale
        # intents are counted as dropped; the node re-triggers on a later hit.
        snapshot = intent.snapshot
        return self._cache.tree_core.validate_buffer_backup(
            snapshot.node_id, len(snapshot.key)
        )

    def _sweep_stale_backup_intents(self) -> dict[NodeId, BufferBackupState]:
        """Cancel stale intents anywhere in the queue, not just at the head:
        a dead intent would otherwise inflate the backlog accounting and
        hold FIFO position ahead of live segments."""
        if not self.pending_write_queue:
            return {}
        page_size = self._cache.page_size
        survivors: deque[_UnifiedBackupIntent] = deque()
        states: dict[NodeId, BufferBackupState] = {}
        swept_tokens = 0
        for intent in self.pending_write_queue:
            snapshot = intent.snapshot
            state = self._validate_backup_intent(intent)
            if state is None:
                self.inflight_backup_node_ids.discard(snapshot.node_id)
                intent_tokens = len(snapshot.hash_values) * page_size
                self.write_backlog_tokens_ -= intent_tokens
                swept_tokens += intent_tokens
                continue
            survivors.append(intent)
            states[snapshot.node_id] = state
        self.pending_write_queue = survivors
        self._log_backup_dropped(swept_tokens)
        return states

    def flush_pending_writes(self) -> None:
        """Launch D2H transfers for admitted intents, head-of-line: device
        locks and staging slots are taken only here, when capacity allows."""
        if not self.pending_write_queue:
            return
        cc = self._cache.cache_controller
        states = self._sweep_stale_backup_intents()
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
            state = states[snapshot.node_id]
            intent_tokens = len(snapshot.hash_values) * self._cache.page_size
            if not self._backup_parent_covered(state):
                # Cascade a dropped parent down the chain rather than creating
                # a permanent storage hole.
                self.pending_write_queue.popleft()
                self.inflight_backup_node_ids.discard(snapshot.node_id)
                self.write_backlog_tokens_ -= intent_tokens
                self._log_backup_dropped(intent_tokens)
                continue
            if self.write_staged_tokens_ >= live_cap:
                # Yield to live fetch demand; retry next round.
                break
            device_value, comp_xfers = self._cache.tree_core.build_backup_spec(
                snapshot.node_id
            )
            sizing_xfers = self._build_aux_staging_transfers(
                snapshot.node_id, snapshot.hash_values, comp_xfers
            )
            if self._backup_oversize(
                snapshot.node_id,
                snapshot.hash_values,
                intent_tokens,
                sizing_xfers,
            ):
                # A permanently unstageable head must not block the queue.
                self.pending_write_queue.popleft()
                self.inflight_backup_node_ids.discard(snapshot.node_id)
                self.write_backlog_tokens_ -= intent_tokens
                self._log_backup_dropped(intent_tokens)
                continue
            if self._aux_budget_blocked(intent, sizing_xfers):
                # An aux pool lacks staging headroom: yield at the gate
                # instead of failing the alloc inside cc.write; acks free
                # aux staging, retry next round.
                break
            if not self._launch_backup_intent(intent, device_value, comp_xfers):
                # Pool full of in-flight staging and nothing reclaimable
                # (the tree never holds host values in buffer mode):
                # defer, head-of-line; pending acks will free slots.
                break
            self.pending_write_queue.popleft()

    def _launch_backup_intent(
        self,
        intent: _UnifiedBackupIntent,
        device_value: torch.Tensor,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> bool:
        """Launch one admitted intent's D2H (staging alloc + device lock +
        async copy); the caller removes it from pending_write_queue. Returns
        False when staging cannot be allocated. From a successful launch the
        intent always reaches its storage-ack, so its content joins the
        LAUNCHED cover consulted by admission."""
        cache = self._cache
        cc = cache.cache_controller
        snapshot = intent.snapshot
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        # Sidecars reuse the source pool's transient host/device indices.  This
        # includes both KV-derived pools and SWA-derived DSV4 state pools.  They
        # allocate no additional staging, but must ride the same D2H operation
        # so their bytes are present when the storage write starts.
        aux_xfers.extend(cache._build_backup_sidecar(device_value, comp_xfers))
        host_indices = cc.write(
            device_value,
            node_id=snapshot.node_id,
            extra_pools=aux_xfers or None,
        )
        if host_indices is None:
            return False
        _track_content_refs(self.inflight_backup_hashes, snapshot.hash_values)
        # NOTE: no commit_backup — the node must never appear
        # host-resident in buffer mode; staging slots live in the entry.
        lock_params = cache.inc_lock_ref(snapshot.node_id).to_dec_params()
        self.ongoing_write_through[snapshot.node_id] = _UnifiedBufferBackupEntry(
            intent=intent,
            host_indices=host_indices,
            aux_xfers=aux_xfers,
            lock_params=lock_params,
        )
        self.write_staged_tokens_ += len(host_indices)
        self.write_backlog_tokens_ -= len(snapshot.hash_values) * cache.page_size
        return True

    def _aux_budget_blocked(
        self,
        intent: _UnifiedBackupIntent,
        aux: Optional[list[PoolTransfer]] = None,
    ) -> bool:
        """True when an aux pool cannot stage this intent right now (free
        minus the loads-priority margin falls short of the need): defer at
        the gate instead of failing the alloc inside cc.write and blocking
        pure-KV intents behind an unallocatable head. The margin enforces
        loads-have-priority on aux pools the way live_cap does on the KV
        pool; avail already reflects prefetch-held slots, so no occupancy
        subtraction here."""
        snapshot = intent.snapshot
        if aux is None:
            aux = self._build_aux_staging_transfers(
                snapshot.node_id, snapshot.hash_values
            )
        if not aux:
            return False
        cc = self._cache.cache_controller
        for t in aux:
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is None:
                continue
            need = len(t.keys) * entry.host_pool.page_size
            headroom = entry.host_pool.available_size() - self._aux_loads_margin(
                entry.host_pool
            )
            if need > headroom:
                return True
        return False

    def _aux_window_keys(
        self, hash_values: list[str], transfer: PoolTransfer
    ) -> Optional[list[str]]:
        """Trailing KV page hashes keying an aux transfer's staged window
        (one key per aux-pool page)."""
        if transfer.host_indices is None or transfer.host_indices.numel() == 0:
            return None
        if transfer.indices_from_pool is not None:
            return None  # sidecar rides another pool's slots; nothing to key
        entry = self._cache.cache_controller.mem_pool_host.entry_map.get(transfer.name)
        if entry is None:
            return None
        num_keys = len(transfer.host_indices) // entry.host_pool.page_size
        if num_keys == 0 or num_keys > len(hash_values):
            return None
        return hash_values[-num_keys:]

    def finish_backup_ack(self, ack_id: int) -> None:
        """D2H confirmed: drop the device lock and enqueue the storage write
        (which reads from the staging copy, so device eviction may proceed)."""
        entry = self.ongoing_write_through.pop(ack_id)
        intent = entry.intent
        snapshot = intent.snapshot
        self._cache.dec_lock_ref(snapshot.node_id, entry.lock_params)

        # Every independently staged aux pool writes a trailing snapshot keyed
        # by the last KV page hashes it covers.  A derived sidecar writes the
        # exact key span of its source pool while reusing that source's slots.
        storage_xfers: list[PoolTransfer] = []
        storage_sources = {
            PoolName.KV: PoolTransfer(
                name=PoolName.KV,
                host_indices=entry.host_indices,
                keys=snapshot.hash_values,
            )
        }
        for staged in entry.aux_xfers:
            if staged.indices_from_pool is not None:
                continue
            keys = self._aux_window_keys(snapshot.hash_values, staged)
            if keys is None:
                continue
            transfer = PoolTransfer(
                name=staged.name,
                host_indices=staged.host_indices,
                keys=keys,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
            storage_xfers.append(transfer)
            storage_sources.setdefault(staged.name, transfer)
        for staged in entry.aux_xfers:
            if staged.indices_from_pool is None:
                continue
            source = storage_sources.get(staged.indices_from_pool)
            if source is None:
                raise AssertionError(
                    "Buffer-mode storage sidecar source missing: "
                    f"{staged.name} from {staged.indices_from_pool}."
                )
            storage_xfers.append(
                PoolTransfer(
                    name=staged.name,
                    keys=source.keys,
                    hit_policy=staged.hit_policy,
                    indices_from_pool=staged.indices_from_pool,
                )
            )
        operation_id = self._cache.cache_controller.write_storage(
            entry.host_indices,
            snapshot.key.token_ids,
            snapshot.hash_values,
            snapshot.prefix_keys,
            extra_pools=storage_xfers or None,
        )
        self.ongoing_backup[operation_id] = entry

    def finish_storage_write_ack(self, operation_id: int) -> None:
        """Storage write acked (rank-synced drain): free the entry's staging
        outright. Existence entries are added unconditionally
        (completed_tokens can diverge across ranks under backend failure) to
        keep admission decisions TP-deterministic. No-op for operations this
        pipeline does not own (e.g. acks for already-reset state)."""
        entry = self.ongoing_backup.pop(operation_id, None)
        if entry is None:
            return
        intent = entry.intent
        snapshot = intent.snapshot
        self._cache.storage_existence_cache.add(PoolName.KV, snapshot.hash_values)
        self._free_staging_now(entry.host_indices, entry.aux_xfers)
        self.write_staged_tokens_ -= len(entry.host_indices)
        self.inflight_backup_node_ids.discard(snapshot.node_id)
        _untrack_content_refs(self.inflight_backup_hashes, snapshot.hash_values)

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

    def try_lock_anchor(self, req_id: str) -> str:
        """Pin the staged prefetch's device anchor so eviction cannot
        invalidate the splice, finding it by re-matching the live tree
        (carried node ids go stale via splits and eviction; the walk is
        O(prefix path)). Returns "locked", "no_anchor" (nothing to pin),
        "cap_skip" (over cap; launches unlocked), or "anchor_lost" (splice
        base gone — the caller cancels the storage IO)."""
        if req_id in self.anchor_locks:
            return "locked"
        prefix_ctx = self._prefetch_prefix_ctx.get(req_id)
        if not prefix_ctx or not prefix_ctx[0]:
            return "no_anchor"  # root anchor: nothing to pin
        prefix_tokens, extra_key, cache_salt = prefix_ctx
        matched_len = len(prefix_tokens)
        if self.anchor_locked_tokens_ + matched_len > self.anchor_lock_cap_tokens:
            self._anchor_lock_cap_skips += 1
            if (
                self._anchor_lock_cap_skips <= 3
                or self._anchor_lock_cap_skips % 1000 == 0
            ):
                logger.warning(
                    "HiCache anchor-lock cap reached (skip %d): locked=%d "
                    "want=%d cap=%d; launching unlocked.",
                    self._anchor_lock_cap_skips,
                    self.anchor_locked_tokens_,
                    matched_len,
                    self.anchor_lock_cap_tokens,
                )
            return "cap_skip"
        cache = self._cache
        anchor_tokens = array("q", prefix_tokens)
        if cache.tree_core.is_eagle:
            # The suffix owns the boundary token shared with the last matched
            # bigram, so include it when rebuilding the anchor key.
            info = cache.ongoing_prefetch.get(req_id)
            if info is None or not info.prefetch_key.token_ids:
                return "anchor_lost"
            anchor_tokens.append(info.prefetch_key.token_ids[0])
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    anchor_tokens,
                    extra_key=extra_key,
                    is_bigram=cache.tree_core.is_eagle,
                    cache_salt=cache_salt,
                )
            )
        )
        if len(match.device_indices) < matched_len:
            return "anchor_lost"
        lock_params = cache.inc_lock_ref(match.last_device_node).to_dec_params()
        self.anchor_locks[req_id] = _AnchorLock(
            node_id=match.last_device_node,
            lock_params=lock_params,
            tokens=matched_len,
        )
        self.anchor_locked_tokens_ += matched_len
        return "locked"

    def release_anchor_lock(self, req_id: str) -> None:
        """Drop a staged prefetch's anchor lock (idempotent; called at every
        consume/drop/abort exit)."""
        lock = self.anchor_locks.pop(req_id, None)
        if lock is None:
            return
        self._cache.dec_lock_ref(lock.node_id, lock.lock_params)
        self.anchor_locked_tokens_ -= lock.tokens
        assert self.anchor_locked_tokens_ >= 0, (
            f"anchor-lock accounting corrupted: locked={self.anchor_locked_tokens_} "
            f"after releasing {req_id}"
        )

    def staged_span_covered(self, req_id: str, span_tokens: int) -> bool:
        """True when the live device tree already covers the fetch's whole
        would-be span (prefix + the storage-hit tokens): nothing would be
        left to splice at consumption, so the IO-commit caller cancels
        before the bounce alloc and the storage read."""
        info = self._cache.ongoing_prefetch.get(req_id)
        if info is None or span_tokens <= 0:
            return False
        prefix_tokens, _, _ = self._prefetch_prefix_ctx[req_id]
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
        return len(match.device_indices) >= len(key)

    def set_prefix_ctx(
        self,
        req_id: str,
        matched_prefix_tokens,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ) -> None:
        """Record the device-matched prefix (and its tree-key namespace) at
        prefetch enqueue; consumed at staging commit to build the full-span
        tree key, and by try_lock_anchor to re-match a stale anchor."""
        self._prefetch_prefix_ctx[req_id] = (
            list(matched_prefix_tokens or []),
            extra_key,
            cache_salt,
        )

    def pop_prefix_ctx(self, req_id: str) -> None:
        self._prefetch_prefix_ctx.pop(req_id, None)

    def has_staged(self, req_id: str) -> bool:
        return req_id in self.staged_prefetches

    @staticmethod
    def _occupied_span(host_indices) -> int:
        """Occupancy units a buffer-mode prefetch holds: granted at
        hit-alloc, sized to the allocation (0 while still querying)."""
        return len(host_indices) if host_indices is not None else 0

    def stage_completed_prefetch(
        self,
        req_id: str,
        num_tokens: int,
        hash_value: list[str],
    ) -> bool:
        """Park the completed fetch as a held bounce; the scheduler surfaces
        it as host_hit_length and the adder consumes it via init_load_back.
        Always returns True (ready is a stable, revisited state)."""
        cache = self._cache
        (
            _anchor,
            prefetch_key,
            host_indices,
            operation,
            _lock_params,
            comp_xfers,
        ) = cache.ongoing_prefetch.pop(req_id)
        cc = cache.cache_controller
        prefix_ctx = self._prefetch_prefix_ctx.pop(req_id, None)
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

        if num_tokens == 0 or prefix_tokens is None:
            # Nothing usable fetched: recompute.
            cache.discard_storage_prefetch_accounting(req_id)
            self.release_anchor_lock(req_id)
            cc.append_host_mem_release(
                host_indices[:num_tokens], extra_pools=aux_xfers or None
            )
            cc.prefetch_tokens_occupied -= self._occupied_span(host_indices)
            cache.prefetch_loaded_tokens_by_reqid[req_id] = 0
            cache.prefetch_loaded_storage_start_by_reqid.pop(req_id, None)
            return True

        staged_pages = num_tokens // cache.page_size
        staged_hashes = hash_value[:staged_pages]
        staged_kv = host_indices[:num_tokens]
        # Feed existence beliefs from the storage-fetched pages: the fetch
        # itself is the evidence, so feeding is sound even if this staged
        # prefetch is later dropped unconsumed.
        cache.storage_existence_cache.add(PoolName.KV, list(staged_hashes))
        occupied_tokens = self._occupied_span(host_indices)

        self.staged_prefetches[req_id] = _StagedPrefetch(
            req_id=req_id,
            key_tokens=prefix_tokens + list(prefetch_key[:num_tokens].token_ids),
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
        cache.prefetch_loaded_tokens_by_reqid[req_id] = num_tokens
        cache.prefetch_loaded_storage_start_by_reqid[req_id] = operation.storage_start
        return True

    def plan_staged_splice(
        self, req_id: str, device_prefix_len: int
    ) -> tuple[int, int]:
        """(kv, swa) host-hit tokens consumption will splice given the
        request's live device prefix, so admission charges no phantom
        tokens. Frees a hold that can no longer splice: surfaced as 0 but
        kept, it would leak — the adder only consumes surfaced host hits."""
        f = self.staged_prefetches.get(req_id)
        if f is None:
            return 0, 0
        splice_tokens = staged_splice_tokens(f, device_prefix_len)
        if splice_tokens == 0:
            covered_tokens = self._resolve_staged_device_coverage(f, device_prefix_len)
            logger.info(
                "HiCache staged prefetch released req=%s matched=%d "
                "device_prefix=%d tokens=%d",
                req_id,
                f.matched_len,
                device_prefix_len,
                f.num_tokens,
            )
            reason = None if covered_tokens == f.num_tokens else "shrunk"
            self.release_staged_hold(req_id, reason=reason)
            return 0, 0
        return splice_tokens, self.staged_prefetch_swa_tokens(req_id)

    def _resolve_staged_device_coverage(
        self, f: _StagedPrefetch, device_prefix_len: int
    ) -> int:
        covered_tokens = min(max(device_prefix_len - f.matched_len, 0), f.num_tokens)
        self._cache._resolve_storage_prefetch_tokens(f.req_id, covered_tokens)
        return covered_tokens

    def staged_prefetch_swa_tokens(self, req_id: str) -> int:
        """SWA device tokens consuming this staged prefetch will allocate (the
        staged trailing window); surfaced as the request's swa_host_hit_length
        so the adder's SWA gate charges the admission-time alloc."""
        f = self.staged_prefetches.get(req_id)
        if f is None:
            return 0
        return sum(
            len(t.host_indices)
            for t in f.aux_xfers
            if t.name == PoolName.SWA and t.host_indices is not None
        )

    def init_load_back(self, params: InitLoadBackParams) -> tuple[torch.Tensor, NodeId]:
        """Consume the staged prefetch at prefill admission: device alloc,
        layer-gated H2D, and a plain insert so downstream sees ordinary tree
        state. The splice base is the request's live device prefix — growth
        trims to the span tail beyond it; unusable holds drop and the
        request recomputes.

        Ownership contract: cc.load queues the H2D before insert adjudicates
        ownership, so the live pre-checks below must prove the insert can
        only ADD nodes — a dedup would free slots the in-flight copy still
        targets (queued use-after-free)."""
        cache = self._cache
        req = params.req
        assert req is not None
        empty = cache.tree_core.empty_match_result.device_indices
        unchanged = (empty, req.last_node)
        f = self.staged_prefetches.pop(req.rid, None)
        if f is None:
            self.release_anchor_lock(req.rid)
            return unchanged
        cc = cache.cache_controller

        def _drop(reason: Optional[str]) -> tuple[torch.Tensor, NodeId]:
            cache._finish_storage_prefetch(req.rid, fulfilled_tokens=0, reason=reason)
            self.release_anchor_lock(req.rid)
            self._free_staging_now(f.host_indices, f.aux_xfers)
            cc.prefetch_tokens_occupied -= f.occupied_tokens
            # Nothing spliced: keep the surfaced host-hit fields truthful.
            req.host_hit_length = 0
            req.swa_host_hit_length = 0
            req.storage_hit_length = 0
            req.storage_hit_start = None
            req.host_hit_is_storage = False
            return unchanged

        # A hold staged under a different namespace than the consuming request
        # must never splice (wrong-namespace publish = duplicate slot
        # ownership); unreachable while the prefetch key is request-derived.
        if f.extra_key != req.extra_key or f.cache_salt != req.cache_salt:
            logger.error(
                "HiCache staged prefetch dropped req=%s reason=namespace "
                "staged=%s req=%s",
                req.rid,
                (f.extra_key, f.cache_salt),
                (req.extra_key, req.cache_salt),
            )
            return _drop("dropped")

        splice_base = len(req.prefix_indices)
        splice_tokens = staged_splice_tokens(f, splice_base)
        if splice_tokens == 0:
            covered_tokens = self._resolve_staged_device_coverage(f, splice_base)
            logger.warning(
                "HiCache staged prefetch dropped req=%s matched=%d now=%d "
                "tokens_wasted=%d locked=%s",
                req.rid,
                f.matched_len,
                splice_base,
                f.num_tokens,
                req.rid in self.anchor_locks,
            )
            reason = None if covered_tokens == f.num_tokens else "shrunk"
            return _drop(reason)
        trim_tokens = splice_base - f.matched_len
        assert trim_tokens % cache.page_size == 0, (
            f"staged splice trim not page-aligned req={req.rid}: "
            f"matched={f.matched_len} splice_base={splice_base}"
        )
        cache._resolve_storage_prefetch_tokens(req.rid, trim_tokens)

        key = RadixKey(
            array("q", f.key_tokens),
            extra_key=f.extra_key,
            is_bigram=cache.tree_core.is_eagle,
            cache_salt=f.cache_salt,
        ).page_aligned(cache.page_size)
        span_end = f.matched_len + f.num_tokens

        # Live ownership pre-check at the splice base: the unified length
        # detects a stale request view (req matched before a later publish),
        # full_kv_hit_length detects FULL overlap the insert would dedup-free
        # (an SWA tombstone can mask live FULL from the unified match alone).
        live = cache.match_prefix(MatchPrefixParams(key=key))
        if (
            len(live.device_indices) != splice_base
            or live.full_kv_hit_length != splice_base
        ):
            logger.warning(
                "HiCache staged prefetch dropped req=%s reason=overlap "
                "splice_base=%d live_unified=%d live_full=%d tokens_wasted=%d "
                "locked=%s",
                req.rid,
                splice_base,
                len(live.device_indices),
                live.full_kv_hit_length,
                f.num_tokens,
                req.rid in self.anchor_locks,
            )
            available_end = min(
                span_end,
                len(live.device_indices),
                live.full_kv_hit_length,
            )
            available_overlap = max(0, available_end - splice_base)
            cache._resolve_storage_prefetch_tokens(req.rid, available_overlap)
            return _drop(None if available_overlap == splice_tokens else "shrunk")

        # Evict-before-alloc (mirrors _load_back_transfers): the budget gate
        # counts evictable pages, but cc.load draws from free slots only.
        if cache.supports_swa():
            avail = cache.token_to_kv_pool_allocator.full_available_size()
        else:
            avail = cache.token_to_kv_pool_allocator.available_size()
        if avail < splice_tokens:
            needed = splice_tokens - avail
            cache.evict_for_alloc(EvictParams(num_tokens=needed))
            if cache.supports_swa():
                avail = cache.token_to_kv_pool_allocator.full_available_size()
            else:
                avail = cache.token_to_kv_pool_allocator.available_size()
            if avail < splice_tokens:
                # Genuinely no room (locked pages): recompute.
                return _drop("device_capacity")

        load_back_id = -(f.operation_id) - 1
        device_indices = cc.load(
            host_indices=f.host_indices[trim_tokens:],
            node_id=load_back_id,
            extra_pools=f.aux_xfers or None,
        )
        if device_indices is None:
            # Transient allocator shortfall despite the evict: recompute
            # (init_load_back's degrade contract).
            return _drop("device_capacity")

        swa_dev = next(
            (
                t.device_indices
                for t in f.aux_xfers
                if t.name == PoolName.SWA
                and t.device_indices is not None
                and t.device_indices.numel() > 0
            ),
            None,
        )
        if swa_dev is not None:
            # Register the trailing window's FULL->SWA translation NOW: the
            # admitted request's attention reads the window through this
            # mapping during the layer-gated forward.
            cache._apply_cache_action(
                RebuildFullToSWAMapping([device_indices[-len(swa_dev) :]], [swa_dev])
            )

        # Publish via a plain insert under the admission lock choreography;
        # the caller's request lock then pins the span (load_back pattern).
        insert_result = cache.insert(
            InsertParams(
                key=key,
                value=torch.cat([req.prefix_indices, device_indices]),
                prev_prefix_len=splice_base,
                swa_evicted_seqlen=(
                    max(0, span_end - len(swa_dev)) if swa_dev is not None else 0
                ),
            )
        )
        self.ongoing_buffer_load_back[load_back_id] = _OngoingBufferLoadBack(
            req_id=f.req_id,
            num_tokens=splice_tokens,
            occupied_tokens=f.occupied_tokens,
            aux_xfers=f.aux_xfers,
            # The full staged bounce (not the trimmed H2D source): the ack
            # frees it whole, trimmed head included.
            host_indices=f.host_indices,
            hash_values=f.hash_values,
        )
        m = cache.match_prefix(MatchPrefixParams(key=key))
        self.release_anchor_lock(req.rid)
        canonical = m.device_indices[splice_base:span_end]
        if len(m.device_indices) < span_end or not torch.equal(
            canonical, device_indices
        ):
            # Fail-stop: the insert freed or replaced slots the in-flight H2D
            # still targets; continuing risks silent KV corruption.
            raise RuntimeError(
                f"HiCache buffer load-back ownership violation req={f.req_id}: "
                f"insert prefix_len={insert_result.prefix_len} "
                f"expected={splice_base}, adopted={len(m.device_indices)} "
                f"span_end={span_end}, canonical_matches_incoming="
                f"{len(m.device_indices) >= span_end and torch.equal(canonical, device_indices)}; "
                f"in-flight H2D targets freed slots"
            )
        # Canonical ownership: return the post-insert tree slice, never the
        # raw cc.load allocation (torch.equal here; the tree slice is truth).
        return canonical, m.last_device_node

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

        cc.prefetch_tokens_occupied -= f.occupied_tokens
        logger.info(
            "HiCache prefetch fill committed req=%s filled=%d occupied=%d locked=%d",
            f.req_id,
            f.num_tokens,
            cc.prefetch_tokens_occupied,
            self.anchor_locked_tokens_,
        )
        cache._finish_storage_prefetch(
            f.req_id, fulfilled_tokens=f.num_tokens, reason=None
        )
        return True

    def release_staged_hold(self, rid: str, reason: Optional[str] = None) -> bool:
        """Free a staged hold outright — anchor pin, host bounce (KV + aux),
        occupancy grant; nothing device-side exists yet. Called for aborts
        and for holds that can no longer splice. Returns True when a hold
        existed."""
        self.release_anchor_lock(rid)
        staged = self.staged_prefetches.pop(rid, None)
        if staged is None:
            return False
        self._cache._finish_storage_prefetch(rid, fulfilled_tokens=0, reason=reason)
        self._free_staging_now(staged.host_indices, staged.aux_xfers)
        self._cache.cache_controller.prefetch_tokens_occupied -= staged.occupied_tokens
        return True
