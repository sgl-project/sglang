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

from sglang.srt.managers.cache_controller import HICACHE_WRITE_STAGING_POOL_FRACTION
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import RebuildFullToSWAMapping
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    NodeId,
    UnifiedTreeNode,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.components import SWAComponent
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class _UnifiedBackupIntent(msgspec.Struct):
    """Buffer-mode backup intent, unpinned while queued.

    Snapshots node identity at enqueue time: a split rewrites the node's
    key/hash in place while these copies stay intact, so
    ``node.hash_value != hash_values`` doubles as split detection and a None
    FULL device value as eviction detection (``_backup_intent_stale``).
    """

    node: UnifiedTreeNode
    node_id: int
    hash_values: list[str]
    key: RadixKey
    prefix_keys: Optional[list[str]] = None


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


def validate_buffer_only_stack(
    sidecar_pool_specs: list, swa_component: Optional[SWAComponent]
) -> None:
    """Post-assembly buffer-mode fences.

    Sidecar pools (DSv4 compressed regions) and unified_kv SWA (device-only
    ring, never offloaded) have no per-pool staging path yet.
    """
    if sidecar_pool_specs:
        raise ValueError(
            "--hicache-host-memory-mode buffer_only does not support "
            "sidecar storage pools (DeepSeek-V4 compressed regions)."
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
    ):
        self._cache = cache
        # SWA window size in KV pages when the SWA component stages through
        # a host pool (0 = KV-only: no trailing window staged). Static after
        # pool assembly.
        self._swa_window_pages = swa_window_pages
        # Metadata-only pending-write backlog cap; beyond it new intents
        # are dropped at admission (re-trigger on a later hit).
        self.write_backlog_cap = write_backlog_cap
        self.reset()

    def reset(self) -> None:
        # Load pipeline: hits awaiting a staging grant (park-and-retry),
        # enqueue-time prefix context, completed prefetches staged until
        # prefill admission, and load-backs in flight (keyed by synthetic
        # negative ack id).
        self.pending_hit_allocs: deque = deque()
        self._prefetch_prefix_ctx: dict[str, list[int]] = {}
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

    def is_idle(self) -> bool:
        """No queued writes, staged prefetches, or storage writes in flight
        (all of which hold host staging or would re-trigger IO)."""
        return not (
            self.pending_write_queue or self.staged_prefetches or self.ongoing_backup
        )

    # ---- backup pipeline (device -> staging -> storage) ----

    def _backup_parent_covered(self, node: UnifiedTreeNode) -> bool:
        """Only admit a node whose parent is stored/in-flight: writing above
        a dropped parent creates a permanent longest-prefix hole."""
        parent = node.parent
        if (
            parent is self._cache.root_node
            or parent.id in self.inflight_backup_node_ids
        ):
            return True
        last_hash = parent.get_last_hash_value()
        return last_hash is not None and self._cache.storage_existence_cache.contains(
            PoolName.KV, last_hash
        )

    def _log_backup_dropped(self, num_tokens: int) -> None:
        cache = self._cache
        if cache.enable_storage_metrics and cache.storage_metrics_collector is not None:
            cache.storage_metrics_collector.log_backup_dropped_tokens(num_tokens)

    def enqueue_backup_intent(self, node: UnifiedTreeNode) -> None:
        """Snapshot a backup intent and commit it to the write queue.
        Admission gates: belief skip, parent-cover, backlog cap, oversize.
        Drops are silent; the node re-triggers on a later hit."""
        if not self._cache.enable_storage or not node.hash_value:
            return
        if node.id in self.inflight_backup_node_ids:
            return
        # Admission cover: beliefs plus content past its D2H launch. The
        # launched cover keeps republished content (fill inserts under new
        # node ids) from re-writing while the original write drains.
        if self._cache.storage_existence_cache.covers_all(
            PoolName.KV, node.hash_value, extra_cover=self.inflight_backup_hashes
        ):
            return
        intent_tokens = len(node.hash_value) * self._cache.page_size
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
        if not self._backup_parent_covered(node) or self._backup_oversize(
            node, intent_tokens
        ):
            self._log_backup_dropped(intent_tokens)
            return

        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self._cache.hicache_storage_pass_prefix_keys
            else None
        )
        intent = _UnifiedBackupIntent(
            node=node,
            node_id=node.id,
            hash_values=list(node.hash_value),
            key=node.key,
            prefix_keys=prefix_keys,
        )
        self.pending_write_queue.append(intent)
        self.inflight_backup_node_ids.add(node.id)
        self.write_backlog_tokens_ += intent_tokens

    def _build_aux_staging_transfers(
        self, node: UnifiedTreeNode
    ) -> Optional[list[PoolTransfer]]:
        """Keys-only aux transfers mirroring what BACKUP_STORAGE would write;
        sizes the per-pool oversize gate (beliefs do not consult these)."""
        transfers: list[PoolTransfer] = []
        if ComponentType.SWA in self._cache.components:
            cd = node.component_data[ComponentType.SWA]
            if cd.value is not None:
                num_pages = len(cd.value) // self._cache.page_size
                if num_pages > 0:
                    transfers.append(
                        PoolTransfer(
                            name=PoolName.SWA,
                            keys=node.hash_value[-num_pages:],
                            hit_policy=PoolHitPolicy.TRAILING_PAGES,
                        )
                    )
        return transfers or None

    def _backup_oversize(
        self,
        node: UnifiedTreeNode,
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
            aux_xfers = self._build_aux_staging_transfers(node)
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

    def _backup_intent_stale(self, intent: _UnifiedBackupIntent) -> bool:
        # Arena-lookup failure = deleted, hash mismatch vs the enqueue-time
        # snapshot = split, a None FULL device value = evicted. Stale
        # intents drop silently; the node re-triggers on a later hit.
        node = intent.node
        try:
            self._cache.tree_core.node_by_id(intent.node_id)
        except KeyError:
            return True
        return (
            node.component_data[BASE_COMPONENT_TYPE].value is None
            or node.hash_value != intent.hash_values
        )

    def _sweep_stale_backup_intents(self) -> None:
        """Cancel stale intents anywhere in the queue, not just at the head:
        a dead intent would otherwise inflate the backlog accounting and
        hold FIFO position ahead of live segments."""
        if not self.pending_write_queue:
            return
        page_size = self._cache.page_size
        survivors: deque[_UnifiedBackupIntent] = deque()
        for intent in self.pending_write_queue:
            if self._backup_intent_stale(intent):
                self.inflight_backup_node_ids.discard(intent.node_id)
                self.write_backlog_tokens_ -= len(intent.hash_values) * page_size
                continue
            survivors.append(intent)
        self.pending_write_queue = survivors

    def flush_pending_writes(self) -> None:
        """Launch D2H transfers for admitted intents, head-of-line: device
        locks and staging slots are taken only here, when capacity allows."""
        if not self.pending_write_queue:
            return
        cc = self._cache.cache_controller
        self._sweep_stale_backup_intents()
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
            intent_tokens = len(intent.hash_values) * self._cache.page_size
            if not self._backup_parent_covered(intent.node) or self._backup_oversize(
                intent.node, intent_tokens
            ):
                # Unwritable intent (dropped parent or unstageable size):
                # cascade the drop down the chain rather than creating a
                # permanent storage hole / stalling the head-of-line queue.
                self.pending_write_queue.popleft()
                self.inflight_backup_node_ids.discard(intent.node_id)
                self.write_backlog_tokens_ -= intent_tokens
                self._log_backup_dropped(intent_tokens)
                continue
            if self.write_staged_tokens_ >= live_cap:
                # Yield to live fetch demand; retry next round.
                break
            if self._aux_budget_blocked(intent):
                # An aux pool lacks staging headroom: yield at the gate
                # instead of failing the alloc inside cc.write; acks free
                # aux staging, retry next round.
                break
            if not self._launch_backup_intent(intent):
                # Pool full of in-flight staging and nothing reclaimable
                # (the tree never holds host values in buffer mode):
                # defer, head-of-line; pending acks will free slots.
                break
            self.pending_write_queue.popleft()

    def _launch_backup_intent(self, intent: _UnifiedBackupIntent) -> bool:
        """Launch one admitted intent's D2H (staging alloc + device lock +
        async copy); the caller removes it from pending_write_queue. Returns
        False when staging cannot be allocated. From a successful launch the
        intent always reaches its storage-ack, so its content joins the
        LAUNCHED cover consulted by admission."""
        cache = self._cache
        cc = cache.cache_controller
        node = intent.node
        # Build aux transfers from the node's CURRENT state: a SWA span
        # tombstoned since admission backs up FULL-only, as in cache mode.
        device_value, comp_xfers = cache.tree_core.build_backup_spec(node.id)
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        host_indices = cc.write(
            device_value,
            node_id=node.id,
            extra_pools=aux_xfers or None,
        )
        if host_indices is None:
            return False
        _track_content_refs(self.inflight_backup_hashes, intent.hash_values)
        # NOTE: no commit_backup — the node must never appear
        # host-resident in buffer mode; staging slots live in the entry.
        lock_params = cache.inc_lock_ref(node.id).to_dec_params()
        self.ongoing_write_through[node.id] = _UnifiedBufferBackupEntry(
            intent=intent,
            host_indices=host_indices,
            aux_xfers=aux_xfers,
            lock_params=lock_params,
        )
        self.write_staged_tokens_ += len(host_indices)
        self.write_backlog_tokens_ -= len(intent.hash_values) * cache.page_size
        return True

    def _aux_budget_blocked(self, intent: _UnifiedBackupIntent) -> bool:
        """True when an aux pool cannot stage this intent right now (free
        minus the loads-priority margin falls short of the need): defer at
        the gate instead of failing the alloc inside cc.write and blocking
        pure-KV intents behind an unallocatable head. The margin enforces
        loads-have-priority on aux pools the way live_cap does on the KV
        pool; avail already reflects prefetch-held slots, so no occupancy
        subtraction here."""
        aux = self._build_aux_staging_transfers(intent.node)
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
        self._cache.dec_lock_ref(intent.node_id, entry.lock_params)

        # Every aux pool writes a trailing snapshot keyed by the last KV page
        # hashes it covers: the SWA window spans page_size-sized pages, the
        # Mamba state is a single slot (host pool page_size 1 -> one key).
        storage_xfers: list[PoolTransfer] = []
        for staged in entry.aux_xfers:
            keys = self._aux_window_keys(intent.hash_values, staged)
            if keys is None:
                continue
            storage_xfers.append(
                PoolTransfer(
                    name=staged.name,
                    host_indices=staged.host_indices,
                    keys=keys,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        operation_id = self._cache.cache_controller.write_storage(
            entry.host_indices,
            intent.key.token_ids,
            intent.hash_values,
            intent.prefix_keys,
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
        self._cache.storage_existence_cache.add(PoolName.KV, intent.hash_values)
        self._free_staging_now(entry.host_indices, entry.aux_xfers)
        self.write_staged_tokens_ -= len(entry.host_indices)
        self.inflight_backup_node_ids.discard(entry.intent.node_id)
        _untrack_content_refs(self.inflight_backup_hashes, intent.hash_values)

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

    def set_prefix_ctx(self, req_id: str, matched_prefix_tokens) -> None:
        """Record the device-matched prefix at prefetch enqueue; consumed at
        staging commit to build the full-span tree key."""
        self._prefetch_prefix_ctx[req_id] = list(matched_prefix_tokens or [])

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
        prefix_tokens = self._prefetch_prefix_ctx.pop(req_id, None)
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]

        if num_tokens == 0 or prefix_tokens is None:
            # Nothing usable fetched: recompute.
            cc.append_host_mem_release(
                host_indices[:num_tokens], extra_pools=aux_xfers or None
            )
            cc.prefetch_tokens_occupied -= self._occupied_span(host_indices)
            cache.prefetch_loaded_tokens_by_reqid[req_id] = 0
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
            matched_len=len(prefix_tokens),
            num_tokens=num_tokens,
            occupied_tokens=occupied_tokens,
            host_indices=staged_kv,
            aux_xfers=aux_xfers,
            hash_values=staged_hashes,
            operation_id=operation.id,
        )
        cache.prefetch_loaded_tokens_by_reqid[req_id] = num_tokens
        return True

    def staged_prefetch_tokens(self, req_id: str) -> int:
        """Tokens a staged prefetch would splice (0 = no hold); surfaced by the
        scheduler as the request's host_hit_length."""
        f = self.staged_prefetches.get(req_id)
        return f.num_tokens if f is not None else 0

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
        """Buffer-mode branch of init_load_back: consume the staged prefetch
        at prefill admission — device alloc (evict-before-alloc), layer-gated
        H2D, and a plain insert so downstream sees ordinary tree state.
        Misaligned or alloc-failed holds drop; the request recomputes."""
        cache = self._cache
        req = params.req
        assert req is not None
        empty = cache.tree_core.empty_match_result.device_indices
        unchanged = (empty, req.last_node)
        f = self.staged_prefetches.pop(req.rid, None)
        if f is None:
            return unchanged
        cc = cache.cache_controller

        def _drop() -> tuple[torch.Tensor, NodeId]:
            self._free_staging_now(f.host_indices, f.aux_xfers)
            cc.prefetch_tokens_occupied -= f.occupied_tokens
            return unchanged

        # Splice-validity: the span only fits if the device prefix still
        # ends exactly at the enqueue-time matched_len.
        if len(req.prefix_indices) != f.matched_len:
            # Prefix moved while held (leaf eviction or sibling extension):
            # drop and recompute.
            return _drop()

        # Evict-before-alloc (mirrors _load_back_transfers): the budget gate
        # counts evictable pages, but cc.load draws from free slots only.
        if cache.supports_swa():
            avail = cache.token_to_kv_pool_allocator.full_available_size()
        else:
            avail = cache.token_to_kv_pool_allocator.available_size()
        if avail < f.num_tokens:
            needed = f.num_tokens - avail
            evicted = cache.evict(EvictParams(num_tokens=needed))
            if evicted.num_tokens_evicted < needed:
                # Genuinely no room (locked pages): recompute.
                return _drop()

        load_back_id = -(f.operation_id) - 1
        device_indices = cc.load(
            host_indices=f.host_indices,
            node_id=load_back_id,
            extra_pools=f.aux_xfers or None,
        )
        if device_indices is None:
            # Transient allocator shortfall despite the evict: recompute
            # (init_load_back's degrade contract).
            return _drop()

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
        key = RadixKey(
            array("q", f.key_tokens),
            extra_key=f.extra_key,
            is_bigram=cache.tree_core.is_eagle,
        ).page_aligned(cache.page_size)
        span_end = f.matched_len + f.num_tokens
        cache.insert(
            InsertParams(
                key=key,
                value=torch.cat([req.prefix_indices, device_indices]),
                prev_prefix_len=f.matched_len,
                swa_evicted_seqlen=(
                    max(0, span_end - len(swa_dev)) if swa_dev is not None else 0
                ),
            )
        )
        self.ongoing_buffer_load_back[load_back_id] = _OngoingBufferLoadBack(
            req_id=f.req_id,
            num_tokens=f.num_tokens,
            occupied_tokens=f.occupied_tokens,
            aux_xfers=f.aux_xfers,
            host_indices=f.host_indices,
            hash_values=f.hash_values,
        )
        m = cache.match_prefix(MatchPrefixParams(key=key))
        if len(m.device_indices) < span_end:
            # The insert walk did not adopt the full span (should not happen
            # for a locked prefix); the slots are tree-owned/evictable — do
            # not splice, the request recomputes.
            return unchanged
        return device_indices, m.last_device_node

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
            "HiCache prefetch fill committed req=%s filled=%d occupied=%d",
            f.req_id,
            f.num_tokens,
            cc.prefetch_tokens_occupied,
        )
        if cache.enable_storage_metrics and cache.storage_metrics_collector is not None:
            cache.storage_metrics_collector.log_prefetched_tokens(f.num_tokens)
        return True

    def release_aborted_staged(self, rid: str) -> bool:
        """Free an aborted request's staged prefetch (nothing device-side
        exists yet — only the bounce). Returns True when a hold existed."""
        staged = self.staged_prefetches.pop(rid, None)
        if staged is None:
            return False
        self._free_staging_now(staged.host_indices, staged.aux_xfers)
        self._cache.cache_controller.prefetch_tokens_occupied -= staged.occupied_tokens
        return True
