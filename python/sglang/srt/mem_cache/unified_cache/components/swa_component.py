from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BindCapturedSWAHost,
    FreeComponentDeviceSlot,
    FreeComponentHostSlot,
    FreeDeviceKV,
    RebuildFullToSWAMapping,
    RecoverSWAWithLockedFull,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    PreparePrefetchResult,
    TreeComponent,
    next_component_uuid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedRadixCache,
        UnifiedTreeNode,
    )

logger = logging.getLogger(__name__)
_SWA_DBG_CHECKSUM = os.environ.get("SGLANG_SWA_DBG_CHECKSUM") == "1"


# c4 / c4-indexer overlap compress-state riding.
#
# A captured SWA window is only bit-exact on reuse if the c4 overlap state at the
# reuse page boundary [B-ratio, B) is restored along with it: c4 compression reads
# the prior group's raw KV/score at every boundary, and the device state ring only
# holds the latest few groups, so a reusing request would otherwise read another
# request's slot. The non-strict path masks that with tail reprefill; the strict
# path cannot. So each SWA carrier co-owns the attn + indexer state tiles captured
# at the same (rid, B), and they ride the window's whole lifetime:
# bind -> BACKUP_HOST promote -> LOAD_BACK restore -> free.
#
# Module-level functions taking the component as first arg so fakes can drive them
# in unit tests. Each ride tuple is
#   (host_pool, device_state_pools, node_host_value_attr, node_pending_attr, li_map)

# (host_pool attr, device state-pool list attr, node host-value attr,
#  node pending attr)
_STATE_RIDE_SPECS = (
    (
        "_c4_state_host_pool",
        "_compress_state_pools",
        "_c4_state_host_value",
        "_c4_state_pending_host",
    ),
    (
        "_c4_indexer_state_host_pool",
        "_indexer_compress_state_pools",
        "_c4_indexer_state_host_value",
        "_c4_indexer_state_pending_host",
    ),
)


def _state_rides(component):
    """Active c4 overlap-state rides (attn + indexer). Empty (``[]``) unless the
    strict state offload has been wired onto ``component``."""
    rides = []
    li_map = getattr(component, "_c4_state_layer_index", None)
    if li_map is None:
        return rides
    for host_attr, pools_attr, host_value_attr, pending_attr in _STATE_RIDE_SPECS:
        hp = getattr(component, host_attr, None)
        pools = getattr(component, pools_attr, None)
        if hp is not None and pools is not None:
            rides.append((hp, pools, host_value_attr, pending_attr, li_map))
    return rides


def _bind_state_rides(component, node, rid: int, B: int) -> bool:
    """Atomically claim the state tiles captured at (rid, B) as pending refs on node
    (co-lifetime with the SWA window). Returns True if every active ride's tile
    was present (or no ride is wired); on a partial miss it rolls back the popped
    tiles and returns False.

    Callers offload the SWA window regardless of this result: a state-less window
    is kept and just excluded from the strict reuse boundary by the match
    validator. The atomicity here only prevents leaving one ride's tile popped
    while the other is missing; it does not gate offload. This keeps partial-
    prefix reuse working while still preventing the dirty read, since a state-less
    boundary is never crossed on reuse.
    """
    rides = _state_rides(component)
    if not rides:
        return True
    popped = []
    for hp, _pools, _hv_attr, _pending_attr, _li in rides:
        staging = getattr(hp, "_capture_staging", None)
        h = staging.pop((rid, int(B)), None) if staging else None
        if h is None:
            for _hp, _v in popped:
                _hp.free(_v)
            _n = getattr(hp, "_bind_miss_dbg", 0) + 1
            hp._bind_miss_dbg = _n
            if _n & (_n - 1) == 0:
                _keys = list(staging.keys())[:8] if staging else []
                logger.warning(
                    "[BIND-MISS] want=(%s,%d) miss#%d staging_n=%d sample_keys=%s",
                    rid,
                    int(B),
                    _n,
                    len(staging) if staging else 0,
                    _keys,
                )
            return False
        popped.append((hp, h.to(torch.int64)))
    # The boundary the tiles were captured at; restore needs it to address the
    # reusing request's ring rows by position.
    node._swa_state_B = int(B)
    for (hp, v), (_hp, _pools, host_value_attr, pending_attr, li_map) in zip(
        popped, rides
    ):
        setattr(node, pending_attr, v)
        if _SWA_DBG_CHECKSUM:
            crc_src = getattr(hp, "_capture_state_crc", None)
            if crc_src is not None:
                setattr(
                    node,
                    host_value_attr + "_crc",
                    {
                        li: crc_src.pop((rid, int(B), li), None)
                        for li in li_map.values()
                    },
                )
    return True


def _node_swa_page_row(component, node):
    """The node's SWA window host page row (the durable state index), or None when
    the SWA host_value is not set / no host pool wired."""
    hp = getattr(component, "_swa_kv_pool_host", None)
    if hp is None:
        return None
    cd = node.component_data[component.component_type]
    hv = getattr(cd, "host_value", None)
    if hv is None or len(hv) == 0:
        return None
    return int(hv[0].item()) // hp.slot_page_size


def _state_durable_indices(hp, swa_page_row):
    """Token indices addressing state pool durable row ``swa_page_row`` (so
    ``_restore_state_windows``'s ``host_value[0] // ring`` recovers the row)."""
    ring = hp.slot_page_size
    return torch.arange(
        swa_page_row * ring, swa_page_row * ring + ring, dtype=torch.int64
    )


def _promote_state_pending(component, node) -> None:
    """Adopt the pending state tiles as durable host values, together with the SWA
    host_value at the coordinated BACKUP_HOST commit (co-lifetime).

    When the state pool reserves a durable region, move the staged tile into the
    SWA window's coupled durable row (promote_captured_page) and record host_value
    as that durable row's indices, so the state sits in its own L3 pool at the same
    row as the SWA window and rides the same coupled key family. With no reserve,
    adopt the staged page as-is.
    """
    swa_row = None
    for hp, _pools, host_value_attr, pending_attr, _li in _state_rides(component):
        pend = getattr(node, pending_attr, None)
        if pend is None:
            continue
        reserve = int(getattr(hp, "_durable_reserve_slots", 0) or 0)
        if reserve and hasattr(hp, "promote_captured_page"):
            if swa_row is None:
                swa_row = _node_swa_page_row(component, node)
            if swa_row is None:
                # A durable region is reserved but the coupled SWA window row is unknown
                # (its host_value must be attached before promote). NEVER adopt
                # the transient slack page as a durable host_value: the allocator
                # can hand that slot to another in-flight capture -> stale/dirty
                # read on restore. Drop the binding instead (state recomputes on
                # reuse); an orphaned durable-less window is correct, just cold.
                logger.warning(
                    "[SWA-HiCache] promote skipped: no SWA window row for node "
                    "%s; dropping staged state tile (recompute on reuse).",
                    getattr(node, "id", id(node)),
                )
                hp.free(pend)
                setattr(node, pending_attr, None)
                setattr(node, host_value_attr, None)
                continue
            hp.promote_captured_page(pend, swa_row)
            setattr(node, host_value_attr, _state_durable_indices(hp, swa_row))
            setattr(node, pending_attr, None)
            continue
        # Legacy single-region pool (no durable reserve): adopt the staged page.
        setattr(node, host_value_attr, pend)
        setattr(node, pending_attr, None)


def _attach_state_durable_row(component, node, swa_slice, B: int) -> None:
    """L3 reuse: the c4/indexer state page rode this window's key family
    (independent-pool sidecar, ``indices_from_pool=SWA``) and was written into the
    coupled durable row by ``set_from_flat_data_page`` on prefetch. Point the
    carrier's state host_value at that durable row (``swa_row``, the same row the
    L3 sidecar addressed via ``_l3_page_size``) so ``restore_pending_swa_windows``
    restores it bit-exact. No-op unless a state ride with a durable region is
    wired."""
    hp0 = getattr(component, "_swa_kv_pool_host", None)
    if hp0 is None:
        return
    if swa_slice is None or len(swa_slice) == 0:
        return
    # Only pools with a reserved durable region are addressed by SWA row. Resolve
    # them before touching hp0's ring geometry: without a ride there is nothing to
    # address, and the SWA host pool need not be a row-paged one at all.
    durable = [
        (hp, host_value_attr)
        for hp, _pools, host_value_attr, _pending, _li in _state_rides(component)
        if int(getattr(hp, "_durable_reserve_slots", 0) or 0)
    ]
    if not durable:
        return
    swa_row = int(swa_slice[0].item()) // hp0.slot_page_size
    node._swa_state_B = int(B)
    for hp, host_value_attr in durable:
        setattr(node, host_value_attr, _state_durable_indices(hp, swa_row))


def _free_state_bindings(component, node) -> None:
    """Release both pending and durable state tiles back to their host pools (SWA
    carrier dropped without a durable host backup, or node removed)."""
    node._swa_state_B = None
    for hp, _pools, host_value_attr, pending_attr, _li in _state_rides(component):
        for attr in (pending_attr, host_value_attr):
            v = getattr(node, attr, None)
            if v is not None:
                hp.free(v)
                setattr(node, attr, None)
        if (
            _SWA_DBG_CHECKSUM
            and getattr(node, host_value_attr + "_crc", None) is not None
        ):
            setattr(node, host_value_attr + "_crc", None)


def _state_locs_for_window(sp, swa_chunk, swa_ring, B, ratio):
    """Device state rows holding the boundary group ``[B-ratio, B)`` of a restored
    ring block.

    These must be the rows the compressor will read, and it addresses c4 state
    from the SWA slot (see ``_c4_state_overlap_prefix``: req_to_token -> full to
    swa -> swa_loc to state_loc). At restore time the reusing request's
    req_to_token is not populated for these positions yet, so the slot comes from
    the restored block instead. The block is in ring order, so it must be indexed
    by position rather than by taking its trailing ``ratio`` slots: those coincide
    only when ``B % swa_ring == 0``, which speculative decode breaks by sizing the
    ring as sliding_window + num_draft_tokens - 1.

    If the base ever re-addresses c4 state by (request, position) -- the contract
    c128 already uses -- this is the one place to switch, and the address-contract
    test fails until it is.
    """
    pos = torch.arange(B - ratio, B, dtype=torch.int64, device=swa_chunk.device)
    return sp.translate_from_swa_loc_to_state_loc(swa_chunk[pos % swa_ring])


def _restore_state_ride_per_layer(
    node,
    hp,
    pools,
    layers,
    host_value_attr,
    swa_chunk,
    swa_ring,
    B,
    page_row,
    slot_bytes,
    off0,
) -> None:
    """One blocking H2D per layer, each layer with its own window width."""
    for layer_id, li in layers:
        sp = pools[layer_id]
        ratio = sp.ratio
        if B < ratio:
            continue
        state_locs = _state_locs_for_window(sp, swa_chunk, swa_ring, B, ratio)
        dev = sp.kv_score_buffer.kv_score
        host_tile = hp.data_refs[li][page_row]
        flat = host_tile[off0 * slot_bytes : (off0 + ratio) * slot_bytes]
        # the blocking .to() is load-bearing: an all-layer non_blocking transfer
        # keeps reading the tile after the call, and a concurrent promote / L3
        # fetch overwrites it mid-DMA -- host CRC still matches, only the device
        # rows are wrong
        window = flat.view(dev.dtype).reshape(ratio, -1).to(device=dev.device)
        dev[state_locs] = window
        if _SWA_DBG_CHECKSUM:
            _dbg_verify_state_restore(
                node, host_value_attr, hp, li, page_row, off0, flat, dev, state_locs
            )


def _restore_state_windows(component, node, swa_chunk: torch.Tensor) -> None:
    """Restore the c4 / c4-indexer overlap state for the reused window onto the
    device state ring, so the reusing request's boundary read is bit-exact.

    swa_chunk is the whole restored ring block for this request, in ring order:
    swa_chunk[j] is the slot holding every position p with p % swa_ring == j. The
    host tile packs the boundary group [B-ratio, B) in token order at off0=0, so
    the destination rows must be looked up by position, not taken from the tail of
    the block -- those coincide only when B % swa_ring == 0, which speculative
    decode breaks (it sizes the ring as sliding_window + num_draft_tokens - 1, so
    the ring no longer divides the page). Each slot's device state row is
    translate_from_swa_loc_to_state_loc(slot), so the captured window lands on the
    exact rows the compressor will read regardless of the reusing request's base.
    """
    rides = _state_rides(component)
    if not rides:
        return
    swa_ring = swa_chunk.numel()
    B = getattr(node, "_swa_state_B", None)
    for hp, pools, host_value_attr, _pending_attr, li_map in rides:
        host_value = getattr(node, host_value_attr, None)
        if host_value is None:
            continue
        if B is None:
            # Cannot address the destination rows without the boundary; restoring
            # to a guessed row would be a silent bit-exactness break, so leave the
            # ring alone and let the boundary recompute.
            logger.warning(
                "[SWA-HiCache] state restore skipped: no boundary recorded for "
                "node %s (attr=%s); boundary will recompute.",
                getattr(node, "id", id(node)),
                host_value_attr,
            )
            continue
        ring = hp.slot_page_size
        slot_bytes = hp.item_bytes // ring
        page_row = int(host_value[0].item()) // ring
        off0 = 0  # pack at tile start; must match capture off0=0
        layers = [
            (layer_id, li)
            for layer_id, li in li_map.items()
            if layer_id < len(pools) and pools[layer_id] is not None
        ]
        if not layers:
            continue
        _restore_state_ride_per_layer(
            node,
            hp,
            pools,
            layers,
            host_value_attr,
            swa_chunk,
            swa_ring,
            B,
            page_row,
            slot_bytes,
            off0,
        )


def _dbg_verify_state_restore(
    node, host_value_attr, hp, li_local, page_row, off0, flat, dev, state_locs
):
    """Gated (SGLANG_SWA_DBG_CHECKSUM) double-ended check for one c4 state ride layer.
    (a) host round-trip: the bound tile bytes still match the position-weighted CRC
    taken at capture, proving capture/bind/promote/restore kept the exact tile at
    the exact page_row/off0. (b) device landing: the rows just written match the
    host window, proving the write hit the intended state ring rows (catches
    state_locs collisions / out-of-range). Immune to model non-determinism;
    localizes any mismatch to layer/page_row/off0/state_locs.
    """
    idx = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64) + 1
    got_host = int((flat.to(torch.int64) * idx).sum().item())
    crc_all = getattr(node, host_value_attr + "_crc", None)
    exp = crc_all.get(li_local) if crc_all else None
    if exp is not None and got_host != exp:
        raise AssertionError(
            f"[C4-STATE-DBG] host round-trip CRC mismatch attr={host_value_attr} "
            f"li_local={li_local} page_row={page_row} off0={off0} "
            f"expected={exp} got={got_host}"
        )
    back = dev[state_locs].contiguous().view(torch.uint8).reshape(-1)
    bidx = torch.arange(back.numel(), device=back.device, dtype=torch.int64) + 1
    got_dev = int((back.to(torch.int64) * bidx).sum().item())
    if got_dev != got_host:
        raise AssertionError(
            f"[C4-STATE-DBG] device landing mismatch attr={host_value_attr} "
            f"li_local={li_local} page_row={page_row} off0={off0} "
            f"state_locs={state_locs.tolist()} host={got_host} dev={got_dev}"
        )
    _n = getattr(hp, "_dbg_state_verified", 0) + 1
    hp._dbg_state_verified = _n
    if _n <= 5 or _n % 200 == 0:
        logger.warning(
            "[C4-STATE-DBG] state ride bit-exact: %d layer-windows (attr=%s)",
            _n,
            host_value_attr,
        )


class SWAComponent(TreeComponent):
    """Sliding window attention component.

    Each SWA node stores translated SWA pool indices as its component
    value, independent of the full attention indices on the same tree node.
    When SWA data is evicted from an internal node the node is tombstoned
    — its SWA component value becomes None while the full attention
    value stays intact.
    """

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

        assert isinstance(
            params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        ), f"SWAComponent requires SWATokenToKVPoolAllocator, got {type(params.token_to_kv_pool_allocator)}"
        super().__init__(cache, params)
        self._session_leaf_covered_len: dict[str, dict[UnifiedTreeNode, int]] = {}
        self.sliding_window_size = params.sliding_window_size
        # HiCache state: set to host SWA pool when HiCache enabled
        self._swa_kv_pool_host = None
        # Strict bit-exact SWA HiCache (unified_kv only): when True, SWA host
        # eviction must never drop a node's SWA copy while keeping its Full
        # copy on host (that "Full-host without SWA-host" orphan would force a
        # non-bit-exact tail reprefill on reuse). Wired at pool-attach time.
        self._strict_bit_exact = False
        # req_pool_idx of the request currently being cached; used to look up
        # its prefill-captured SWA host pages during insert.
        self._capture_rid = None
        # unified_kv positional SWA ring: SWA device slots are computed as
        # req_pool_idx*ring + pos%ring (not free-list allocated). Restore on
        # reuse must be positional + deferred until req_pool_idx is known
        # (prepare_for_extend), so load_back only stashes the window on the req.
        self._unified_positional_swa = False

    component_type = ComponentType.SWA

    def needs_incremental_backup(self, node: UnifiedTreeNode) -> bool:
        return False

    def reset_session_state(self) -> None:
        super().reset_session_state()
        self._session_leaf_covered_len = {}

    def _walk_session_coverage(
        self,
        leaf: UnifiedTreeNode,
        span: int,
        delta: int,
    ) -> int:
        node = leaf
        covered = 0
        while node is not self.tree_core.root_node and covered < span:
            cd = node.component_data[self.component_type]
            if delta < 0:
                assert cd.session_ref > 0
            prev_ref = cd.session_ref
            cd.session_ref += delta
            if (prev_ref == 0) != (cd.session_ref == 0):
                self._refresh_session_partition(node)
            covered += len(node.key)
            node = node.parent
        return covered

    def _inc_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        covered_by_leaf = self._session_leaf_covered_len.setdefault(session_id, {})
        assert leaf not in covered_by_leaf
        target_span = self.sliding_window_size + self.tree_core.page_size
        covered = self._walk_session_coverage(leaf, target_span, 1)
        assert covered > 0
        covered_by_leaf[leaf] = covered

    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        covered_by_leaf = self._session_leaf_covered_len.get(session_id)
        assert covered_by_leaf is not None and leaf in covered_by_leaf
        covered_len = covered_by_leaf.pop(leaf)
        if not covered_by_leaf:
            self._session_leaf_covered_len.pop(session_id, None)
        actual = self._walk_session_coverage(leaf, covered_len, -1)
        assert actual == covered_len

    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:
        self._inc_session_coverage(session_id, leaf)
        if old_ancestor is not None:
            self._dec_session_coverage(session_id, old_ancestor)

    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:
        self._dec_session_coverage(session_id, leaf)
        if fallback is not None:
            self._inc_session_coverage(session_id, fallback)

    def validate_session_state(
        self,
        reachable_nodes: set[UnifiedTreeNode],
        report_error: Callable[[str], None],
    ) -> None:
        super().validate_session_state(reachable_nodes, report_error)
        ct = self.component_type

        for session_id, covered_by_leaf in self._session_leaf_covered_len.items():
            for leaf, covered_len in covered_by_leaf.items():
                if leaf not in self._session_leaves.get(session_id, ()):
                    report_error(
                        f"{ct} session {session_id!r} coverage leaf {leaf.id} is not indexed"
                    )
                if covered_len <= 0:
                    report_error(
                        f"{ct} session {session_id!r} leaf {leaf.id} covered_len={covered_len}"
                    )

        for session_id, leaves in self._session_leaves.items():
            covered_by_leaf = self._session_leaf_covered_len.get(session_id, {})
            for leaf in leaves:
                if leaf not in covered_by_leaf:
                    report_error(
                        f"{ct} session {session_id!r} leaf {leaf.id} has no coverage record"
                    )

    def _translate_full_to_swa(self, full_indices: torch.Tensor) -> torch.Tensor:
        return self.cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            full_indices
        )

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        match phase:
            case LRURefreshPhase.WALKDOWN:
                # Walk-down would refresh every visited ancestor to MRU,
                # but most are outside the active sliding window and must
                # stay evictable. Window-bounded refresh runs at
                # MATCH_END / INSERT_END instead.
                return
            case LRURefreshPhase.MATCH_END | LRURefreshPhase.INSERT_END:
                self.tree_core.lru_lists[
                    self.component_type
                ].reset_node_and_window_ancestors_mru(
                    node,
                    root_node,
                    self.sliding_window_size + self.tree_core.page_size,
                    self.node_has_component_data,
                )
            case _:
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        sliding_window_size = self.sliding_window_size
        ct = self.component_type
        strict_bit_exact = self._strict_bit_exact
        # Reuse correctness comes from this runtime clamp rather than an
        # insert-time SWA>=Full guard: a node without a durable SWA host copy, or
        # without its c4/c4-indexer overlap state, resets the running window
        # length so the boundary clamps to the nearest page that has both. Under
        # stride>1 a page can hold a Full host copy but no SWA window, which is a
        # normal non-reuse boundary. Empty when state riding is unwired.
        state_ride_attrs = tuple(
            hv_attr for (_hp, _pools, hv_attr, _pend, _li) in _state_rides(self)
        )
        state = {"len": float("inf")}

        # unified_kv never caches the SWA ring (per-request, not content-stable),
        # so SWA bookkeeping must not gate the match here.
        swa_device_only_hicache = (
            not self.tree_core.has_swa_host_pool and self.tree_core.enable_hicache
        )

        def validator(node: UnifiedTreeNode) -> bool:
            cd = node.component_data[ct]
            # HiCache: a host-only tombstone is a valid match boundary too
            # — load_back will restore SWA from host before use.
            if cd.value is None and (match_device_only or cd.host_value is None):
                state["len"] = 0
                if swa_device_only_hicache and (node.backuped or not node.evicted):
                    return True
                # The SWA ring only keeps the last sliding_window_size tokens, so
                # an out-of-window node has cd.value recycled to None while its
                # window is durably host-backed and its Full KV is still device
                # resident. Such a node must not truncate the Full device anchor:
                # cache_unfinished_req's self-match needs device_indices to span
                # the whole Full-resident prefix, else cache_protected_len can
                # exceed len(new_indices). Reuse over-reach is reclamped to the
                # host-gated boundary via for_reuse. Best-effort mode keeps the
                # upstream anchor, which stops at the last SWA device value.
                return bool(
                    strict_bit_exact and match_device_only and cd.host_value is not None
                )
            # The device SWA ring is per-request and is recycled when the owner's
            # req_pool_idx is reused, so it is not cross-request truth: on the
            # reuse match only a durable host copy counts, and a device-only node
            # truncates the match so it reloads from host or recomputes instead of
            # serving a stale ring. The device-only match must still report the
            # request's own freshly-computed nodes as resident, else the self-match
            # returns empty device indices.
            missing_state = bool(state_ride_attrs) and any(
                getattr(node, a, None) is None for a in state_ride_attrs
            )
            if (
                strict_bit_exact
                and not match_device_only
                and (cd.host_value is None or missing_state)
            ):
                state["len"] = 0
                return False
            state["len"] += len(node.key)
            return state["len"] >= sliding_window_size

        return validator

    def finalize_match_result_in_tree_core(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        ct = self.component_type
        n_swa = 0
        swa_host_hit = 0
        node = result.best_match_node
        root = self.tree_core.root_node
        # On the reuse path the device SWA ring is not durable cross-request truth,
        # so a node that is both device-resident and host-backed still counts as a
        # host hit -- otherwise swa_host_hit_length stays 0 and the load_back gate
        # never opens. Same host-backed predicate as build_hicache_transfers below.
        # Self-match trusts cd.value first, since a request's own fresh nodes
        # aren't host-backed yet.
        strict_reuse = self._strict_bit_exact and params.for_reuse
        while node is not root and n_swa < self.sliding_window_size:
            cd = node.component_data[ct]
            if strict_reuse and cd.host_value is not None:
                swa_host_hit += len(cd.host_value)
                n_swa += len(cd.host_value)
            elif cd.value is not None:
                n_swa += len(cd.value)
            elif cd.host_value is not None:
                # TODO(hzh): load_back may currently restore a full host-tombstone
                # segment whose length exceeds sliding_window_size. Once
                # load_back is constrained to fetch only one sliding window
                # worth of pages, cap swa_host_hit at sliding_window_size
                # here so the scheduler budget matches the actual device-pool
                # consumption.
                swa_host_hit += len(cd.host_value)
                n_swa += len(cd.host_value)
            else:
                break
            node = node.parent
        if swa_host_hit > 0:
            return result._replace(
                swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit)
            )
        return result

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> int:
        if params.prev_prefix_len >= total_prefix_len + prefix_len:
            return prefix_len

        is_tombstone = node.component_data[self.component_type].value is None
        if not is_tombstone:
            return prefix_len

        full_cd = node.component_data[BASE_COMPONENT_TYPE]
        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            node.component_data[self.component_type].lock_ref == 0
        ), f"tombstone {self.component_type} lock_ref should be 0, node {node.id}"
        assert (
            swa_evicted_seqlen % self.tree_core.page_size == 0
        ), f"{self.component_type}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        if swa_evicted_seqlen <= total_prefix_len:
            # Branch 1: entire value_slice is within SWA window — recover
            old_full = full_cd.value
            if full_cd.lock_ref > 0:
                cache_actions.append(
                    RecoverSWAWithLockedFull(node.id, old_full, value_slice)
                )
                return 0
            full_cd.value = value_slice.clone()
            cache_actions.append(FreeDeviceKV([old_full]))
            cache_actions.append(SWARebuild(node.id, value_slice))
            return 0
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # Branch 2: value_slice[start_idx:] is within SWA window — partial recover
            start_idx = swa_evicted_seqlen - total_prefix_len
            is_locked = full_cd.lock_ref > 0
            old_full = full_cd.value[start_idx:]
            _, action = self.tree_core._split_node(node.key, node, start_idx)
            if action is not None:
                cache_actions.append(action)
            new_full = value_slice[start_idx:]
            if is_locked:
                cache_actions.append(
                    RecoverSWAWithLockedFull(node.id, old_full, new_full)
                )
                return start_idx
            node.component_data[BASE_COMPONENT_TYPE].value = new_full.clone()
            cache_actions.append(FreeDeviceKV([old_full]))
            cache_actions.append(SWARebuild(node.id, new_full))
            return start_idx
        else:
            # Branch 3: entire value_slice is outside SWA window — not consumed
            return prefix_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        # _unevict_node_on_insert already wrote the request's fresh KV slice
        # into the base value. We just need to rebuild SWA from that slice for
        # the in-window portion. There is no old SWA slot to free here.
        ct = self.component_type
        if node.component_data[ct].value is not None:
            return
        assert (
            node.component_data[ct].lock_ref == 0
        ), f"tombstone {ct} lock_ref should be 0 on unevict, node {node.id}"
        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            swa_evicted_seqlen % self.tree_core.page_size == 0
        ), f"{ct}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        if swa_evicted_seqlen <= total_prefix_len:
            pass  # entire node is within the SWA window
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            start_idx = swa_evicted_seqlen - total_prefix_len
            _, action = self.tree_core._split_node(node.key, node, start_idx)
            if action is not None:
                cache_actions.append(action)
        else:
            return
        cache_actions.append(
            SWARebuild(
                node.id,
                node.component_data[BASE_COMPONENT_TYPE].value,
            )
        )

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        if not is_new_leaf:
            return

        node_start = result.prefix_len
        split_pos = params.swa_evicted_seqlen - node_start
        if split_pos >= len(node.key):
            # Entire leaf is outside the SWA window — left as a tombstone.
            return
        # Absolute end of the leaf. Every split below (straddle, interior stride,
        # window cap) keeps `node` as the trailing piece, so this boundary — the
        # key the capture staged its window under — is invariant.
        leaf_end = node_start + len(node.key)
        tombstone_parent = None
        if split_pos > 0:
            # Node straddles the boundary: split into an out-of-window parent
            # (tombstone) and an in-window child; `node` becomes the child.
            tombstone_parent, action = self.tree_core._split_node(
                node.key, node, split_pos
            )
            assert action is None, "new leaf cannot be write-through-pending"
        if tombstone_parent is not None:
            # The out-of-window tombstone span [node_start, swa_evicted_seqlen)
            # also carries prefill-captured windows at the interior stride page
            # boundaries. Claim them as host-only carriers so cross-request reuse
            # clamps to the nearest stride page instead of the coarse chunk end.
            # They hold no device SWA value, so this can run inline.
            self._bind_interior_captured_swa_hosts(
                tombstone_parent, node_start, params.swa_evicted_seqlen
            )
        swa_start = max(node_start, params.swa_evicted_seqlen)
        interior_carriers: list[UnifiedTreeNode] = []
        if self._strict_bit_exact and split_pos <= 0:
            # The capture stages a host window at every stride page boundary of
            # this leaf, but the bind action below only claims the leaf-end one;
            # the interior ones would be freed unused at cleanup_after_caching_req
            # and reuse would clamp to the coarse leaf end. Claim them as host
            # carriers. Inline rather than an action because it keys off tree
            # geometry only, and it must precede the window cap, whose split would
            # otherwise put the earlier stride boundaries outside `node`.
            interior_carriers = self._bind_interior_captured_swa_hosts(
                node, swa_start, leaf_end
            )
        # Cap the in-window leaf at one window for lock granularity, then rebuild SWA
        # onto the in-window node(s) at apply time; rebuild the older prefix first so
        # the in-window tail lands more-MRU. Every piece the splits above carved out
        # of the in-window span needs its own rebuild: the ring slots it covers are
        # reachable only through the node that owns them.
        capped_parent = self._maybe_split_leaf_for_swa_lock(node)
        for carrier in reversed(interior_carriers):
            cache_actions.append(
                SWARebuild(
                    carrier.id,
                    carrier.component_data[BASE_COMPONENT_TYPE].value,
                )
            )
        if capped_parent is not None:
            cache_actions.append(
                SWARebuild(
                    capped_parent.id,
                    capped_parent.component_data[BASE_COMPONENT_TYPE].value,
                )
            )
        cache_actions.append(
            SWARebuild(
                node.id,
                node.component_data[BASE_COMPONENT_TYPE].value,
            )
        )
        # Strict bit-exact: claim the prefill-captured host window ending at this
        # leaf's boundary. Deferred to an action ordered after the SWARebuild
        # above, because the bind only applies once the node carries its SWA
        # device value.
        if self._strict_bit_exact:
            cache_actions.append(BindCapturedSWAHost(node.id, leaf_end))

    def _maybe_split_leaf_for_swa_lock(
        self, leaf: UnifiedTreeNode
    ) -> Optional[UnifiedTreeNode]:
        """Cap a fresh in-window SWA leaf at one page-aligned window so locking it pins
        only one window of SWA pool, not the whole (long chunked-prefill) leaf; return
        the split-off parent (older window) or None. The SWA value is stamped later, so
        this runs on the tombstone leaf.
        """
        ct = self.component_type
        cd = leaf.component_data[ct]
        if leaf is self.tree_core.root_node or cd.lock_ref > 0:
            return None

        page_size = self.tree_core.page_size
        # Smallest page-aligned size that still covers the sliding window.
        tail_size = (self.sliding_window_size + page_size - 1) // page_size * page_size
        leaf_len = len(leaf.key)
        if leaf_len <= tail_size:
            return None
        split_at = leaf_len - tail_size
        if page_size > 1 and (split_at % page_size != 0 or leaf_len % page_size != 0):
            return None

        new_parent, action = self.tree_core._split_node(leaf.key, leaf, split_at)
        assert action is None, "fresh SWA leaf cannot be write-through-pending"
        return new_parent

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.component_data[self.component_type].lock_ref = child.component_data[
            self.component_type
        ].lock_ref
        new_parent.component_data[self.component_type].session_ref = (
            child.component_data[self.component_type].session_ref
        )
        assert new_parent.component_data[self.component_type].session_ids is None

        child_swa_value = child.component_data[self.component_type].value
        if child_swa_value is not None:
            split_len = len(new_parent.key)
            new_parent.component_data[self.component_type].value = child_swa_value[
                :split_len
            ].clone()
            child.component_data[self.component_type].value = child_swa_value[
                split_len:
            ].clone()
        else:
            new_parent.component_data[self.component_type].value = None

        child_swa_host_value = child.component_data[self.component_type].host_value
        if child_swa_host_value is not None:
            split_len = len(new_parent.key)
            full_span = split_len + len(child.key)
            host_lru = self.tree_core.host_lru_lists[self.component_type]
            # Only the strict ring-paged pool carries a slot geometry; the
            # best-effort SWA host pool has no slot_page_size, so gate on the flag
            # before reaching for it and still read it defensively.
            ring = (
                getattr(self._swa_kv_pool_host, "slot_page_size", None)
                if self._strict_bit_exact
                else None
            )
            if ring is not None and len(child_swa_host_value) == ring:
                # Strict host_value is always exactly one ring page (the window at
                # the child's end), so it belongs whole to the child even when the
                # node happens to span exactly `ring` tokens -- splitting it by key
                # length would hand the pool a half page and break the page
                # alignment that page_row = host_value[0] // ring depends on.
                new_parent.component_data[self.component_type].host_value = None
            elif len(child_swa_host_value) == full_span:
                # Common case: host_value spans the whole node; split by key len.
                new_parent.component_data[self.component_type].host_value = (
                    child_swa_host_value[:split_len].clone()
                )
                child.component_data[self.component_type].host_value = (
                    child_swa_host_value[split_len:].clone()
                )
            else:
                # host_value holds only the sliding window at the child's end
                # boundary, so it belongs entirely to the child. The parent's own
                # boundary window (if any) is stored separately, not here.
                new_parent.component_data[self.component_type].host_value = None
                # child keeps child_swa_host_value unchanged
            if (
                new_parent.component_data[self.component_type].value is None
                and new_parent.component_data[self.component_type].host_value
                is not None
            ):
                host_lru.insert_mru(new_parent)
            if child.component_data[
                self.component_type
            ].value is None and not host_lru.in_list(child):
                host_lru.insert_mru(child)

        # parent inherits the swa_uuid from child for swa lock ref
        new_parent.component_data[self.component_type].metadata["uuid"] = (
            child.component_data[self.component_type].metadata.get("uuid")
        )
        child.component_data[self.component_type].metadata.pop("uuid", None)

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        ct = self.component_type
        cd = node.component_data[ct]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            # Pass full indices to free_swa so slots with no SWA pair are
            # skipped. Freeing swa_value directly would double free those
            # entries since they all map to the same sentinel slot.
            device_frees[self.component_type].append(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            freed = len(cd.value)
            self.tree_core.component_evictable_size_[ct] -= freed
            cd.value = None
            # A captured page not yet promoted to host_value must not outlive its
            # device SWA; free it and let the node degrade to recompute.
            #
            # Interior stride carriers are exempt: they are out-of-window by
            # construction, so this tombstone always fires before the finish-time
            # coordinated BACKUP_HOST. Their pending page is a standalone host copy
            # whose lifetime tracks the Full component (the node survives here
            # because base is still resident), not the SWA device ring, so keep it
            # for the coordinated backup to promote. It is freed only at true node
            # removal (_remove_leaf_from_parent).
            pending = getattr(node, "_swa_pending_host", None)
            if pending is not None and not getattr(
                node, "_swa_interior_carrier", False
            ):
                if self._swa_kv_pool_host is not None:
                    self._swa_kv_pool_host.free(pending)
                node._swa_pending_host = None
                # Co-lifetime: the pending state tiles die with the SWA pending
                # window (never promoted -> node degrades to recompute).
                _free_state_bindings(self, node)

        # Host layer
        host_lru = self.tree_core.host_lru_lists[ct]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            host_frees[ct].append(cd.host_value)
            cd.host_value = None
            # Co-lifetime: the durable host state tiles die with the SWA host
            # window on host eviction (L1-only state scope; L3 state persistence
            # is a later phase).
            _free_state_bindings(self, node)
            if host_lru.in_list(node):
                host_lru.remove_node(node)

        # After device tombstone: if host_value remains, move into host LRU
        if (
            target is EvictLayer.DEVICE
            and cd.value is None
            and cd.host_value is not None
        ):
            if not host_lru.in_list(node):
                host_lru.insert_mru(node)

        # Stride model leak guard: an interior stride carrier holds its captured
        # window as a not-yet-promoted ``_swa_pending_host`` and has no device
        # SWA value, so the device branch above never frees it. If the node is
        # dropped before its coordinated BACKUP_HOST promotes the page to
        # host_value, free it here to avoid leaking the SWA host pool.
        pending = getattr(node, "_swa_pending_host", None)
        if (
            pending is not None
            and cd.value is None
            and cd.host_value is None
            and not getattr(node, "_swa_interior_carrier", False)
        ):
            if self._swa_kv_pool_host is not None:
                self._swa_kv_pool_host.free(pending)
            node._swa_pending_host = None
            _free_state_bindings(self, node)

        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 1

    def _evict_device_start(self, request_cnt: int) -> None:
        """Begin the device-eviction walk from this component's LRU cursor."""
        self._evict_device_request_cnt = request_cnt
        if self.tree_core.enable_session_radix_cache:
            lru = self.tree_core.lru_lists[self.component_type]
            lru.cursor_begin()
            self._evict_device_cursor = lru.cursor_next()
        else:
            self._evict_device_cursor = self.tree_core.lru_lists[
                self.component_type
            ].get_lru_no_lock()

    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        """Return the next device-leaf node for the driver to evict, or None.
        Internal nodes are tombstoned inline (no IO). If the previous node's
        eviction removed the cursor, the walk resumes from the partition
        sentinel with session refs on, else it restarts at the LRU tail."""
        ct = self.component_type
        lru = self.tree_core.lru_lists[ct]
        enabled = self.tree_core.enable_session_radix_cache
        if self._evict_device_cursor is not None and not lru.in_list(
            self._evict_device_cursor
        ):
            self._evict_device_cursor = (
                lru.cursor_next() if enabled else lru.get_lru_no_lock()
            )
        while (
            tracker[ct] < self._evict_device_request_cnt
            and self._evict_device_cursor is not None
            and lru.in_list(self._evict_device_cursor)
        ):
            x = self._evict_device_cursor
            assert x.component_data[ct].value is not None
            if x in self.tree_core.evictable_device_leaves and (
                not enabled or self._can_evict_leaf_atomically(x)
            ):
                self._evict_device_cursor = (
                    lru.cursor_next() if enabled else lru.get_prev_no_lock(x)
                )
                return x.id
            if not enabled:
                x_next = lru.get_prev_no_lock(x)
            self.tree_core._evict_component_and_detach_lru(
                x,
                self,
                target=EvictLayer.DEVICE,
                tracker=tracker,
                device_frees=device_frees,
                host_frees=host_frees,
            )
            self.tree_core._cascade_evict(
                x, self, tracker, device_frees=device_frees, host_frees=host_frees
            )
            self._evict_device_cursor = lru.cursor_next() if enabled else x_next
        return None

    def _evict_device_end(self) -> None:
        """Clear the device-eviction walk cursor state."""
        if self.tree_core.enable_session_radix_cache:
            self.tree_core.lru_lists[self.component_type].cursor_end()
        self._evict_device_cursor = None

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        root = self.tree_core.root_node
        sliding_window_size = self.sliding_window_size
        swa_lock_size = 0
        swa_uuid = None
        uuid_key = "host_uuid" if lock_host else "uuid"
        lru = (
            self.tree_core.host_lru_lists[ct]
            if lock_host
            else self.tree_core.lru_lists[ct]
        )

        # Tombstoned nodes (cd.value is None) have no SWA chunk to protect
        # skip them and keep walking up. This path is hit when HiCache
        # backs up a FULL present internal node whose SWA was already evicted.
        cur = node
        while cur != root and swa_lock_size < sliding_window_size:
            comp = cur.component_data[ct]
            value = comp.host_value if lock_host else comp.value
            if value is None:
                result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)
                cur = cur.parent
                continue

            ref = comp.host_lock_ref if lock_host else comp.lock_ref
            if ref == 0:
                if lock_host:
                    if lru.in_list(cur):
                        lru.remove_node(cur)
                else:
                    key_len = len(cur.key)
                    self.tree_core.component_evictable_size_[ct] -= key_len
                    self.tree_core.component_protected_size_[ct] += key_len
            if lock_host:
                comp.host_lock_ref = ref + 1
            else:
                comp.lock_ref = ref + 1
            swa_lock_size += len(value)
            if swa_lock_size >= sliding_window_size:
                if comp.metadata.get(uuid_key) is None:
                    comp.metadata[uuid_key] = next_component_uuid()
                swa_uuid = comp.metadata[uuid_key]
            cur = cur.parent

        if lock_host:
            result.swa_uuid_for_host_lock = swa_uuid
        else:
            result.swa_uuid_for_lock = swa_uuid
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        root = self.tree_core.root_node
        swa_uuid_for_lock = (
            (params.swa_uuid_for_host_lock if lock_host else params.swa_uuid_for_lock)
            if params
            else None
        )
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        dec_swa = True
        uuid_key = "host_uuid" if lock_host else "uuid"

        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.
        cur = node
        while cur != root and dec_swa:
            comp = cur.component_data[ct]
            if cur.id in skip_lock_node_ids:
                cur = cur.parent
                continue
            ref = comp.host_lock_ref if lock_host else comp.lock_ref
            if ref == 0:
                cur = cur.parent
                continue
            if ref == 1:
                if lock_host:
                    if comp.value is None and comp.host_value is not None:
                        host_lru = self.tree_core.host_lru_lists[ct]
                        if not host_lru.in_list(cur):
                            host_lru.insert_mru(cur)
                else:
                    key_len = len(comp.value)
                    self.tree_core.component_evictable_size_[ct] += key_len
                    self.tree_core.component_protected_size_[ct] -= key_len
            if lock_host:
                comp.host_lock_ref = ref - 1
            else:
                comp.lock_ref = ref - 1
            if swa_uuid_for_lock and comp.metadata.get(uuid_key) == swa_uuid_for_lock:
                dec_swa = False
            cur = cur.parent

    def release_window_lock(
        self,
        node: UnifiedTreeNode,
        swa_uuid_for_lock: Optional[int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Early-release the SWA lock along [node, swa_uuid_for_lock] while
        leaving Full and Mamba locks intact.

        Called when a request's decode position has advanced past the sliding
        window — the SWA portion of the tree lock is no longer needed but the
        Full lock must stay so the request's prefix is protected.

        Caller (UnifiedRadixCache.dec_swa_lock_only) must ensure this is
        invoked at most once per (node, swa_uuid_for_lock) pair.
        """
        ct = self.component_type
        root = self.tree_core.root_node

        cur = node
        while cur is not root:
            cd = cur.component_data[ct]
            # Acquire skips tombstoned nodes; release must skip them too. Same
            # for nodes with lock_ref == 0 — acquire never credited them.
            if cd.value is None or cd.lock_ref == 0:
                if swa_uuid_for_lock and cd.metadata.get("uuid") == swa_uuid_for_lock:
                    break
                cur = cur.parent
                continue

            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                key_len = len(cur.key)
                self.tree_core.component_protected_size_[ct] -= key_len
                self.tree_core.component_evictable_size_[ct] += key_len
                if self.tree_core._is_device_leaf(cur):
                    self.tree_core._evict_component_and_detach_lru(
                        cur,
                        self,
                        target=EvictLayer.DEVICE,
                        device_frees=device_frees,
                        host_frees=host_frees,
                    )

            if swa_uuid_for_lock and cd.metadata.get("uuid") == swa_uuid_for_lock:
                break
            cur = cur.parent

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        # Unfinished requests can already have an SWA-evicted prefix; preserve
        # that boundary so insertion creates a tombstone instead of live SWA KV.
        insert_params.swa_evicted_seqlen = req.kv.swa_evicted_seqlen
        self._capture_rid = req.req_pool_idx
        return None

    def free_out_of_window_slots(
        self, req: Req, pre_len: int, insert_params: InsertParams
    ) -> None:
        if self.sliding_window_size is not None:
            free_swa_out_of_window_slots(
                req,
                pre_len,
                sliding_window_size=self.sliding_window_size,
                page_size=self.cache.page_size,
                req_to_token_pool=self.cache.req_to_token_pool,
                token_to_kv_pool_allocator=self.cache.token_to_kv_pool_allocator,
                retain_floor=self.cache.swa_retain_floor(req),
            )
        insert_params.swa_evicted_seqlen = req.kv.swa_evicted_seqlen

    # ---- HiCache Hooks ----

    def prepare_prefetch(
        self,
        node_id: NodeId,
        *,
        prefetch_tokens: int = 0,
    ) -> PreparePrefetchResult:
        # unified_kv keeps SWA as a device-only ring -- nothing to prefetch into.
        if self._swa_kv_pool_host is None:
            return PreparePrefetchResult()
        if self._strict_bit_exact:
            # Trailing-window prefetch. Window granularity is the host SWA pool's
            # slot_page_size (the ring), not page_size: unified_kv packs one full
            # sliding window per page (ring == sliding_window, so window_pages == 1)
            # whereas a per-page ring would need ceil(window / ring) *contiguous*
            # pages. window_pages must be 1 because `_sync_trailing_keys` hands
            # back contiguous Full-page hashes, and only one maps cleanly onto a
            # ring-spaced carrier.
            ring = self._swa_kv_pool_host.slot_page_size
            window_pages = max(1, (self.sliding_window_size + ring - 1) // ring)
            # Not worth a partial window: need at least one ring of freshly
            # prefetched Full tokens behind the tail.
            if prefetch_tokens < ring:
                return PreparePrefetchResult()
            num_tokens = window_pages * ring
        else:
            sw_pages = (
                self.cache.sliding_window_size + self.cache.page_size - 1
            ) // self.cache.page_size
            if sw_pages == 0 or prefetch_tokens // self.cache.page_size < sw_pages:
                return PreparePrefetchResult()
            num_tokens = sw_pages * self.cache.page_size
        host_indices = self._swa_kv_pool_host.alloc(num_tokens)
        if host_indices is None:
            self.cache.evict_host(num_tokens, ComponentType.SWA)
            host_indices = self._swa_kv_pool_host.alloc(num_tokens)
        if host_indices is None:
            return PreparePrefetchResult(alloc_failed=True)
        return PreparePrefetchResult(host_indices=host_indices)

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        mamba_pool_idx: Optional[torch.Tensor] = None,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        # unified_kv keeps SWA as a device-only ring.
        if not self.tree_core.has_swa_host_pool and self.tree_core.enable_hicache:
            return None

        if phase == CacheTransferPhase.BACKUP_HOST:
            cd = node.component_data[ct]
            if cd.host_value is not None:
                # Already populated from a prior backup; do not re-copy.
                return None
            pending = getattr(node, "_swa_pending_host", None)
            if pending is not None:
                # Adopt the prefill-captured host page through the coordinated
                # backup, so SWA host_value is set together with Full host_value
                # and never before. device_indices is None -> write_backup skips
                # the redundant device->host copy. This also covers interior stride
                # carriers, which hold a captured window but no device SWA value;
                # the pending check must precede the ``cd.value is None`` guard
                # below so they are not skipped.
                return [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=pending,
                        device_indices=None,
                    )
                ]
            if cd.value is None:
                return None
            if self._strict_bit_exact:
                # Strict: SWA host pages are allocated only at prefill capture
                # time. With no captured page (host pool full / window missed),
                # emit no SWA host_value; the node falls back to recompute on
                # reuse. Never back up the device ring here -- it holds only
                # the latest window per slot (older windows byte-stale) and
                # allocating host at backup can exhaust the small SWA pool.
                return None
            # Best-effort: back up the device ring.
            # cd.value already holds SWA-pool indices (translated at insert time).
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=cd.value.to(torch.int64),
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            # `node` is best_match_node; the SWA validator guarantees every
            # ancestor within `sliding_window_size` has value or host_value.
            n_swa = 0
            backed_up: list[torch.Tensor] = []
            nodes: list = []
            cur = node
            while (
                cur is not self.tree_core.root_node and n_swa < self.sliding_window_size
            ):
                cd = cur.component_data[ct]
                assert cd.host_value is not None or cd.value is not None
                if self._strict_bit_exact and cd.host_value is not None:
                    # The device SWA ring is not durable cross-request truth even
                    # when `cd.value` is still set -- it may be a recycled slot
                    # from a prior request. Collect the host copy so
                    # _restore_device_value overwrites the stale slot on commit.
                    # Same host-backed predicate as finalize_match_result's
                    # for_reuse gate above.
                    backed_up.append(cd.host_value)
                    nodes.append(cur)
                    n_swa += len(cd.host_value)
                elif cd.value is not None:
                    # device exists (best-effort mode, or strict with no
                    # durable host copy), skip it
                    n_swa += len(cd.value)
                else:
                    # host only, collect it
                    backed_up.append(cd.host_value)
                    nodes.append(cur)
                    n_swa += len(cd.host_value)
                cur = cur.parent

            if not backed_up:
                return None

            backed_up.reverse()
            nodes.reverse()

            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.cat(backed_up),
                    device_indices=None,
                    nodes_to_load=[n.id for n in nodes],
                )
            ]

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            cd = node.component_data[ct]
            if cd.host_value is None or not node.hash_value:
                return None
            if self._strict_bit_exact:
                # Persist the captured window under the carrier node's own Full page
                # hash so the SWA window and its Full page share one L3 key-family
                # lifetime. A strict carrier holds exactly one window page (==
                # slot_page_size), stored as a single trailing page keyed by
                # hash_value[-1], mirroring the mamba sidecar. Windows that are not
                # a whole page are skipped, not truncated.
                ring = self._swa_kv_pool_host.slot_page_size
                if len(cd.host_value) != ring:
                    return None
                return [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=cd.host_value,
                        keys=[self._swa_l3_key(node)],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                ]
            num_pages = len(cd.host_value) // self.tree_core.page_size
            if num_pages == 0:
                return None
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=cd.host_value[-num_pages * self.tree_core.page_size :],
                    keys=node.hash_value[-num_pages:],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        if phase == CacheTransferPhase.PREFETCH:
            assert host_indices is not None
            # Strict windows are ring-paged; the placeholder key count must match
            # the page count `_sync_trailing_keys` will rewrite to Full hashes,
            # so each requested window page is keyed by the same Full hash its
            # carrier stored it under (BACKUP_STORAGE keys=[_swa_l3_key]) -- a
            # window is then fetched iff its Full page also hit.
            stride = (
                self._swa_kv_pool_host.slot_page_size
                if self._strict_bit_exact
                else self.tree_core.page_size
            )
            sw_pages = host_indices.numel() // stride
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=host_indices,
                    keys=["__placeholder__"] * sw_pages,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                cd = node.component_data[ct]
                if cd.host_value is None:
                    # Same bookkeeping the eager insert path used (host_value +
                    # evictable-leaf sets); host-LRU insert is deferred to the
                    # device tombstone (cd.value is still set here).
                    self._attach_swa_host_value(node, transfers[0].host_indices)
                if transfers[0].device_indices is None:
                    # Adopted the pre-staged capture page; ownership now held by
                    # host_value (same page) -> drop the pending ref.
                    node._swa_pending_host = None
                # Co-lifetime: promote the ridden c4/c4-indexer state tiles to
                # durable host values together with the SWA host_value (never
                # before), so the state and its window share one host lifetime.
                _promote_state_pending(self, node)
            # If the owning request finished while this host backup was still in
            # flight (host_value was None at cache_finished_req, so
            # evict_device_on_owner_release deferred the device free), drop the
            # now-recycled device SWA value now that the host copy is durable and
            # no holder remains. Otherwise the async write_through leaves the
            # device ring alive to be trusted on cross-request reuse.
            if getattr(node, "_swa_release_pending", False):
                cd = node.component_data[ct]
                if (
                    self._strict_bit_exact
                    and self._swa_kv_pool_host is not None
                    and cd.value is not None
                    and cd.host_value is not None
                    and cd.lock_ref == 0
                ):
                    node._swa_release_pending = False
                    device_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(
                        list
                    )
                    host_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(
                        list
                    )
                    self.tree_core._evict_component_and_detach_lru(
                        node,
                        self,
                        target=EvictLayer.DEVICE,
                        device_frees=device_frees,
                        host_frees=host_frees,
                    )
                    self.cache._free_values(device_frees, host_frees)
            return

        if phase == CacheTransferPhase.LOAD_BACK:
            assert transfers and transfers[0].device_indices is not None
            xfer = transfers[0]
            device_indices = xfer.device_indices

            full_chunks: list[torch.Tensor] = []
            swa_chunks: list[torch.Tensor] = []
            offset = 0
            for nid in xfer.nodes_to_load or []:
                n = self.tree_core.node_by_id(nid)
                cd_n = n.component_data[ct]
                n_tokens = len(cd_n.host_value)
                swa_chunk = device_indices[offset : offset + n_tokens].clone()
                self._restore_device_value(nid, swa_chunk)
                # host_value holds the sliding window [B-n_tokens, B). Map its full
                # indices to the restored SWA slots; out-of-window full tokens keep
                # sentinel 0 and are never read under the SWA mask. The window can
                # start before the node's own start when the node was split shorter
                # than the window (host_value still spans the whole window), so
                # gather the window's full indices across the node and its ancestors
                # in token order to keep full<->swa lengths matched. Unsplit nodes
                # already have >= n_tokens and touch no ancestor.
                window_full = self._gather_window_full_indices(n, n_tokens)
                # Diagnostic guard (S2/S3): the full<->swa mapping must feed
                # equal-length index tensors. If the restored SWA device
                # chunk is shorter than the window full indices (device
                # under-allocation, or a node whose host_value length changed
                # between build and commit), fail here with the exact sizes
                # instead of the opaque allocator assert (full==swa).
                if swa_chunk.numel() < window_full.numel():
                    _fv2 = n.component_data[BASE_COMPONENT_TYPE].value
                    raise AssertionError(
                        "SWA load_back index-length mismatch "
                        f"(swa_chunk={swa_chunk.numel()} < "
                        f"window_full={window_full.numel()}): "
                        f"n_tokens={n_tokens} offset={offset} "
                        f"dev_total={device_indices.numel()} "
                        f"host_total={len(xfer.host_indices)} "
                        f"own_full={None if _fv2 is None else len(_fv2)}"
                    )
                full_chunks.append(window_full)
                swa_chunks.append(swa_chunk[-window_full.numel() :])
                if _SWA_DBG_CHECKSUM and hasattr(self, "_dbg_verify_restore"):
                    self._dbg_verify_restore(cd_n)
                offset += n_tokens
            assert offset == len(xfer.host_indices)
            # rebuild the mapping for the loaded SWA chunk, defer to orchestrator level
            if full_chunks:
                cache_actions.append(RebuildFullToSWAMapping(full_chunks, swa_chunks))
            return

        if phase == CacheTransferPhase.PREFETCH:
            self._commit_prefetch(
                node,
                transfers,
                cache_actions=cache_actions,
                insert_result=insert_result,
                pool_storage_result=pool_storage_result,
            )
            return

    def _restore_device_value(self, node_id: NodeId, value: torch.Tensor) -> None:
        """Store a (re)assigned SWA device value on the node.

        Wraps the TreeCore store so the strict-mode deferred owner-release intent
        is cleared: the value is live for the current holder, so a stale
        ``_swa_release_pending`` from the node's prior life must not survive and
        drop it at the next BACKUP_HOST commit. The flag is SWA-specific, so it is
        cleared here rather than in the generic TreeCore setter.
        """
        node = self.tree_core.node_by_id(node_id)
        if getattr(node, "_swa_release_pending", False):
            node._swa_release_pending = False
        self.tree_core.set_component_device_value(node_id, self.component_type, value)

    # ---- Strict bit-exact SWA: capture / bind ----

    def _bind_captured_swa_host(self, node: UnifiedTreeNode, node_end: int) -> None:
        """Stash the prefill-captured host page as a pending ref on the node.

        We do not set host_value here: the SWA host_value must not exist before the
        node's Full host_value, so it is attached later by the coordinated BACKUP_HOST
        commit. Until then the page is held in node._swa_pending_host and freed on
        device eviction if the node is never backed up.

        The node ends at boundary ``node_end`` with captured window
        [node_end-win, node_end) keyed (rid, node_end). The earlier, out-of-window
        part of the node is never attended and is not stored. If the window was not
        captured (pool full / outside this chunk), leave it to the normal backup /
        recompute path.

        The caller passes the boundary rather than a start offset: a leaf can be
        split after the insert computes its geometry, which changes the value
        length but not the end boundary the capture keyed on.
        """
        if _SWA_DBG_CHECKSUM:
            logger.warning("[BIND-DBG] enter node_end=%s", node_end)
        hp = self._swa_kv_pool_host
        if hp is None:
            if _SWA_DBG_CHECKSUM:
                logger.warning("[BIND-DBG] early: no swa host pool")
            return
        staging = getattr(hp, "_capture_staging", None)
        rid = self._capture_rid
        if not staging or rid is None:
            if _SWA_DBG_CHECKSUM:
                logger.warning(
                    "[BIND-DBG] early: staging_empty=%s rid=%s", not staging, rid
                )
            return
        cd = node.component_data[self.component_type]
        if cd.value is None or cd.host_value is not None:
            if _SWA_DBG_CHECKSUM:
                logger.warning(
                    "[BIND-DBG] early: cd.value_none=%s host_value_set=%s",
                    cd.value is None,
                    cd.host_value is not None,
                )
            return
        win = hp.slot_page_size
        h = staging.pop((rid, int(node_end)), None)
        if h is None:
            if _SWA_DBG_CHECKSUM:
                logger.warning(
                    "[BIND-DBG] SWA staging MISS key=(%s,%s) staging_keys=%s",
                    rid,
                    int(node_end),
                    list(staging.keys())[:6],
                )
            # Window not captured -> fall back to normal backup / recompute.
            return
        host_value = h.to(torch.int64)
        if len(host_value) != win:
            hp.free(host_value)
            return
        # Defer attach to the coordinated BACKUP_HOST (co-lifetime with Full host).
        node._swa_pending_host = host_value
        # Claim the c4/c4-indexer overlap-state tiles at the same (rid, node_end)
        # when present, but never drop the SWA window when they are missing: the
        # window rides on its own, and create_match_validator keeps a state-less
        # boundary out of the strict reuse boundary, clamping to the nearest
        # window+state boundary instead. Dropping every state-less interior window
        # here collapsed partial-prefix reuse to zero. No-op when strict state
        # offload is unwired.
        _bind_state_rides(self, node, rid, int(node_end))
        if _SWA_DBG_CHECKSUM:
            crc_map = getattr(hp, "_capture_crc", None)
            if crc_map:
                keys = [k for k in crc_map if k[0] == rid and k[1] == int(node_end)]
                if keys:
                    cd.metadata["dbg_swa_crc"] = {k[2]: crc_map.pop(k) for k in keys}

    def _bind_interior_captured_swa_hosts(
        self, region_node: UnifiedTreeNode, region_start: int, region_end: int
    ) -> list[UnifiedTreeNode]:
        """Claim the prefill-captured windows at the interior stride page boundaries of
        a chunk's out-of-window tombstone span.

        _bind_captured_swa_host only claims the single window at the chunk end. The
        finer stride-gated windows the capture also offloaded (keyed (rid, B) for
        interior boundaries) would otherwise be dropped by cleanup_after_caching_req,
        clamping the reuse boundary to the coarse chunk end. Here we split region_node
        at each staged boundary B so a node ends at B, and stash the captured window
        as that node's _swa_pending_host.

        host_value is not set here: the window rides the same deferred path as the
        tail (_swa_pending_host -> BACKUP_HOST). On the out-of-window tombstone
        span the carriers have no device SWA value; the BACKUP_HOST transfer
        builder adopts the pending page for such device-less nodes.

        Returns the carriers in creation order (newest span first). On the
        in-window span the caller must stamp each one's slice of the SWA device
        value, otherwise the ring slots those carriers cover belong to no node
        and can never be freed.
        """
        carriers: list[UnifiedTreeNode] = []
        hp = self._swa_kv_pool_host
        if hp is None:
            return carriers
        staging = getattr(hp, "_capture_staging", None)
        rid = self._capture_rid
        if not staging or rid is None:
            return carriers
        win = hp.slot_page_size
        page = self.tree_core.page_size
        boundaries = sorted(
            int(b)
            for (r, b) in list(staging.keys())
            if r == rid and region_start < int(b) < region_end
        )
        if not boundaries:
            return carriers
        # Attach largest boundary first: each split keeps ``region_start`` as the
        # anchor, and the node object ending at B retains its stashed page across
        # subsequent splits (redistribute_on_node_split does not move the plain
        # ``_swa_pending_host`` attribute, which stays with the truncated child).
        cur = region_node
        for B in reversed(boundaries):
            split_len = B - region_start
            if split_len <= 0 or split_len >= len(cur.key):
                continue
            if page > 1 and split_len % page != 0:
                continue
            h = staging.get((rid, B))
            if h is None:
                continue
            host_value = h.to(torch.int64)
            if len(host_value) != win:
                continue
            staging.pop((rid, B), None)
            new_parent, action = self.tree_core._split_node(cur.key, cur, split_len)
            assert (
                action is None
            ), "interior SWA carrier cannot be write-through-pending"
            new_parent._swa_pending_host = host_value
            # Claim the interior carrier's state tiles at (rid, B) when present, but
            # keep the SWA window even on a miss: the match validator excludes a
            # state-less interior boundary from the strict reuse boundary. Dropping
            # the window here zeroed partial-prefix reuse, since interior state is
            # evicted or exhausted far more often than the chunk tail.
            _bind_state_rides(self, new_parent, rid, B)
            # Mark as an interior stride carrier: its captured page lifetime tracks
            # the Full component and is dropped only at true node removal, not the
            # SWA device ring, which is always recycled out-of-window before the
            # finish-time coordinated BACKUP_HOST. See evict_component and
            # unified_tree_core._remove_leaf_from_parent.
            new_parent._swa_interior_carrier = True
            carriers.append(new_parent)
            cur = new_parent
        return carriers

    def _gather_window_full_indices(
        self, node: UnifiedTreeNode, n_tokens: int
    ) -> torch.Tensor:
        """Collect the last n_tokens FULL indices ending at node boundary, in
        token order, walking into ancestors when the node own full value is
        shorter than the sliding window (post-split case). In the common case
        the node own full value already has >= n_tokens, so this returns
        full.value[-n_tokens:] without touching any ancestor."""
        parts = []
        need = n_tokens
        cur = node
        root = self.tree_core.root_node
        while need > 0 and cur is not None and cur is not root:
            fv = cur.component_data[BASE_COMPONENT_TYPE].value
            if fv is None or len(fv) == 0:
                break
            take = min(need, len(fv))
            parts.append(fv[len(fv) - take :])
            need -= take
            if need <= 0:
                break
            cur = getattr(cur, "parent", None)
        assert parts, "no FULL indices available to restore SWA window"
        return torch.cat(list(reversed(parts)))

    def _swa_l3_key(self, node) -> str:
        """L3 key for a carrier node's captured SWA window.

        Keyed by the carrier's own Full page hash (hash_value[-1]), so the window
        lives and dies with that Full page in the storage backend. Centralized
        here so a future namespace change touches one place.
        """
        return node.hash_value[-1]

    # ---- Strict bit-exact SWA: device-ring invalidation ----

    def evict_device_on_owner_release(self, node: UnifiedTreeNode) -> None:
        """Drop a node's device SWA ring value once its owning request has finished and
        no one else holds the SWA lock, so cross-request reuse restores the true
        window from host instead of trusting the device ring.

        The device SWA lives in a per-request ring (req_slot*ring + pos%ring) that the
        owner overwrites as it decodes and that is recycled when its req_pool_idx is
        reused, so its bytes are only valid for the owner's live window. Called from
        cache_finished_req after the owner released its lock: the ring slots still
        belong to the finished request (safe to free) and the host copy is now the
        source of truth.

        Requires: strict mode + SWA host pool wired; device value present; host_value
        committed (a pending-only page is left until its BACKUP_HOST commits); SWA
        lock_ref == 0.
        """
        if not self._strict_bit_exact or self._swa_kv_pool_host is None:
            return
        cd = node.component_data[self.component_type]
        if cd.value is None:
            return
        if cd.host_value is None or cd.lock_ref > 0:
            # Host copy not durable yet (async write_through still in flight) or
            # another request still holds the SWA lock, so the device ring value
            # cannot be freed right now. Once this owner is gone the ring slot is
            # recycled and its bytes go stale, so it must not be trusted for
            # cross-request reuse. Mark the node so the coordinated BACKUP_HOST
            # commit drops the device value as soon as the host copy is durable and
            # no holder remains.
            node._swa_release_pending = True
            return
        # Cache-level entry point (cache_finished_req), so the freed slots are
        # drained here rather than threaded back through an eviction result.
        # defaultdict matches the BaseEvictionResult contract evict_component
        # appends into.
        device_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
        host_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
        self.tree_core._evict_component_and_detach_lru(
            node,
            self,
            target=EvictLayer.DEVICE,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        self.cache._free_values(device_frees, host_frees)

    def free_pending_host_on_remove(self, node: UnifiedTreeNode) -> None:
        """Free an interior stride carrier's not-yet-promoted capture page at the single
        node-removal chokepoint.

        Pending's lifetime tracks the Full (base) component, so it is dropped only here
        (node truly leaves the tree), never on a mere SWA device tombstone. Called from
        unified_tree_core._remove_leaf_from_parent, which runs regardless of
        per-component eviction order, so nothing leaks.
        """
        pending = getattr(node, "_swa_pending_host", None)
        if pending is not None:
            if self._swa_kv_pool_host is not None:
                self._swa_kv_pool_host.free(pending)
            node._swa_pending_host = None
        # Node truly leaving the tree: release any ridden c4/c4-indexer state
        # tiles (pending or durable host) so the state host pools don't leak.
        _free_state_bindings(self, node)

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        # Release any capture staging this request owns that no node claimed
        # (interior / out-of-window windows), then drop the stashed rid. Key off
        # req.req_pool_idx rather than _capture_rid: on the retract/abort path
        # caching runs with is_insert=False, so prepare_for_caching_req never ran
        # and _capture_rid is unset, while decode capture stages (req_pool_idx, B)
        # across many steps -- those windows would leak here and could later
        # mis-bind to a new request once req_pool_idx is recycled. The two agree on
        # the is_insert=True path. Fall back to the stashed rid only if the slot is
        # already gone.
        rid = req.req_pool_idx if req.req_pool_idx is not None else self._capture_rid
        self._capture_rid = None
        if rid is None:
            return
        # Sweep the SWA pool and every state ride pool: each owns its own staging
        # dict, and most staged state tiles are never claimed (only page
        # boundaries that become node boundaries get bound), so skipping the ride
        # pools here exhausts their slack region and silently drops every boundary
        # out of strict reuse.
        pools = [self._swa_kv_pool_host]
        pools += [hp for hp, _p, _hv, _pend, _li in _state_rides(self)]
        for hp in pools:
            if hp is None:
                continue
            staging = getattr(hp, "_capture_staging", None)
            if staging:
                for k in [k for k in staging if k[0] == rid]:
                    hp.free(staging.pop(k))
            for crc_attr in ("_capture_crc", "_capture_state_crc"):
                crc_map = getattr(hp, crc_attr, None)
                if crc_map:
                    for k in [k for k in crc_map if k[0] == rid]:
                        crc_map.pop(k, None)

    # ---- Strict bit-exact SWA: deferred positional restore ----

    def restore_pending_swa_windows(self, req, req_pool_idx, io_backend):
        """Positional SWA restore for unified_kv, deferred from load_back.

        The SWA read is purely positional (slot*ring + pos%ring) and never consults
        full_to_swa_index_mapping, so the reused window bytes must physically sit at
        req_pool_idx*ring + pos%ring. req_pool_idx is only known now (after
        prepare_for_extend), so load_back stashed the host window pages on the req;
        copy them H->D into this request's ring block, one window page per layer.

        The host window page is exactly one ring block (host slot_page_size == ring),
        so the restore is a faithful whole-ring-block copy into [r*ring, (r+1)*ring):
        host row i lands at device row r*ring+i, so the pos%ring layout is preserved
        byte-for-byte. This holds for any ring = sliding_window + spec_extra and does
        NOT require page % ring == 0. The host pool is window-paged, so the copy is
        page-granular via the pool transfer_kv.
        """
        windows = getattr(req, "_swa_restore_windows", None)
        req._swa_restore_windows = None
        if not windows:
            return
        hp = self._swa_kv_pool_host
        if hp is None:
            return
        # ring size == host SWA pool page size (slot_page_size=swa_ring_size);
        # avoids depending on a pool handle the cache does not expose.
        ring = hp.slot_page_size
        # Only the trailing `ring` tokens fit the per-request ring block, and any
        # earlier window maps onto the same ring rows and would just be overwritten,
        # so restore the trailing window rather than concatenating all of them.
        # build_hicache_transfer reverses the leaf->root walk, so windows are
        # ordered root->leaf and the trailing one is last.
        host_idx = windows[-1][1].to(torch.int64)
        if host_idx.numel() != ring:
            # Not restorable: the request will read whatever the previous occupant
            # of this ring block left behind, so this is a correctness hole rather
            # than a missed optimization. Never silent -- rate-limited so a
            # pathological shape cannot flood the log.
            _n = getattr(hp, "_lb_restore_skip_dbg", 0) + 1
            hp._lb_restore_skip_dbg = _n
            if _n & (_n - 1) == 0:
                logger.error(
                    "[LB-RESTORE] skipped non-single-page window #%d: "
                    "host_tokens=%s ring=%s windows=%d",
                    _n,
                    int(host_idx.numel()),
                    ring,
                    len(windows),
                )
            return
        r = int(req_pool_idx)
        base = r * ring
        device_idx = torch.arange(
            base, base + ring, dtype=torch.int64, device=hp.gpu_device
        )
        host_idx = host_idx.to(hp.gpu_device)
        # The host page for this window was written by a capture D2H (prefill window
        # or decode source) enqueued non_blocking on the compute stream, followed by
        # a recorded completion event. Wait on it before reading the host page, so a
        # cross-stream or cross-batch reuse never restores a half-written window;
        # under scheduler overlap the producer and consumer streams can differ.
        # No-op when no capture has run on this pool yet.
        if hasattr(hp, "wait_capture_done"):
            hp.wait_capture_done()
        # Restore every layer in one fused transfer instead of a Python loop of
        # `layer_num` per-layer copies (61 launches -> 1). Same transfer primitive
        # and page indices, so the device landing is byte-for-byte identical; this
        # only removes launch overhead on the prefill reuse hot path.
        if hasattr(hp, "load_to_device_all_layer"):
            hp.load_to_device_all_layer(None, host_idx, device_idx, io_backend)
        else:
            for li in range(hp.layer_num):
                hp.load_to_device_per_layer(None, host_idx, device_idx, li, io_backend)
        # Ride the c4/c4-indexer overlap state back onto the device state ring for
        # this reused window; device_idx is the restored ring block, indexed by
        # position to find the boundary group. Wait on each state pool's own
        # capture-done event first, mirroring the SWA wait above. No-op when state
        # offload is unwired or the node carries no state host value.
        for _shp, _spools, _shv, _spend, _sli in _state_rides(self):
            if hasattr(_shp, "wait_capture_done"):
                _shp.wait_capture_done()
        for _wnode, _ in windows:
            _restore_state_windows(self, _wnode, device_idx)
        if _SWA_DBG_CHECKSUM:
            if hasattr(self, "_dbg_verify_restore"):
                for node, _ in windows:
                    self._dbg_verify_restore(node.component_data[self.component_type])
            self._dbg_verify_device_landing(hp, r, int(host_idx[0].item()) // ring)

    # ---- Strict bit-exact SWA: gated acceptance checks ----

    def _dbg_verify_device_landing(self, hp, r, host_page_row):
        """Gated acceptance check (SGLANG_SWA_DBG_CHECKSUM, default off): after the positional H2D, read the
        device ring page back and assert it byte-matches the host window page.
        Proves the copy landed at unified_kv row r*ring (device page row == r),
        not just that the host bytes are intact."""
        import torch as _torch

        _torch.cuda.synchronize()
        bad = 0
        for li in range(hp.layer_num):
            dev = hp.device_buffers[li][r].detach().to("cpu")
            host = hp.data_refs[li][host_page_row].detach().to("cpu")
            if dev.numel() != host.numel() or not bool((dev == host).all().item()):
                bad += 1
                if bad <= 3:
                    logger.warning(
                        "[LB-DEV] MISMATCH layer=%s r=%s host_page=%s "
                        "dev_bytes=%s host_bytes=%s",
                        li,
                        r,
                        host_page_row,
                        int(dev.numel()),
                        int(host.numel()),
                    )
        n = getattr(hp, "_dbg_dev_verified", 0) + 1
        hp._dbg_dev_verified = n
        if bad == 0 and (n <= 5 or n % 50 == 0):
            logger.warning(
                "[LB-DEV] device landing byte-exact: %d restores "
                "(layers=%d, page row r=%d)",
                n,
                hp.layer_num,
                r,
            )
        elif bad:
            logger.warning(
                "[LB-DEV] device landing FAILED: %d/%d layers mismatch (r=%d)",
                bad,
                hp.layer_num,
                r,
            )

    def _dbg_verify_restore(self, cd_n) -> None:
        """Gated acceptance check (SGLANG_SWA_DBG_CHECKSUM, default off): assert
        the bound host page still matches the checksum captured at prefill,
        proving the restore path served byte-exact windows. Immune to model
        non-determinism."""
        hp = self._swa_kv_pool_host
        crcs = (cd_n.metadata or {}).get("dbg_swa_crc")
        if hp is None or not crcs or cd_n.host_value is None:
            return
        slot_page = hp.slot_page_size
        page_row = int(cd_n.host_value[0].item()) // slot_page
        for layer, expected in crcs.items():
            b = hp.data_refs[layer][page_row].view(torch.uint8).reshape(-1)
            idx = torch.arange(b.numel(), device=b.device, dtype=torch.int64) + 1
            got = int((b.to(torch.int64) * idx).sum().item())
            assert got == expected, (
                f"[SWA-DBG] restore checksum mismatch layer={layer} "
                f"page_row={page_row} expected={expected} got={got}"
            )
        hp._dbg_restore_verified = getattr(hp, "_dbg_restore_verified", 0) + 1
        n = hp._dbg_restore_verified
        if n <= 5 or n % 50 == 0:
            logger.warning(
                "[SWA-DBG] restore verified bit-exact: %d windows (layers/window=%d)",
                n,
                len(crcs),
            )

    def take_positional_restore_windows(
        self, transfers: list[PoolTransfer]
    ) -> list[tuple[UnifiedTreeNode, torch.Tensor]]:
        """The (carrier node, host window) pairs of a LOAD_BACK spec, for the
        deferred positional restore.

        unified_kv computes the SWA device slot as ``req_pool_idx*ring +
        pos%ring``, which is only known after prepare_for_extend. So no device
        slot is allocated and no H->D transfer is issued at load_back time (the
        swa_attn_allocator is page_size-based and unrelated to the positional
        ring). The cache stashes these pairs on the Req instead, and
        ``restore_pending_swa_windows`` copies them positionally before the
        first forward. Kept here (not in the cache) so the node/host pairing
        stays with the component that built the transfer.
        """
        windows: list[tuple[UnifiedTreeNode, torch.Tensor]] = []
        for xfer in transfers:
            if xfer.name is not PoolName.SWA or not xfer.nodes_to_load:
                continue
            offset = 0
            for nid in xfer.nodes_to_load:
                n = self.tree_core.node_by_id(nid)
                n_tokens = len(n.component_data[self.component_type].host_value)
                windows.append((n, xfer.host_indices[offset : offset + n_tokens]))
                offset += n_tokens
        return windows

    def _release_swa_host(
        self,
        host_indices: torch.Tensor,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        if host_indices is not None and host_indices.numel() > 0:
            cache_actions.append(
                FreeComponentHostSlot([host_indices], component_type=ComponentType.SWA)
            )

    def _attach_swa_host_value(
        self, node: UnifiedTreeNode, host_indices: torch.Tensor
    ) -> None:
        """Write host_indices into node's SWA host_value and refresh tree state."""
        ct = self.component_type
        cd = node.component_data[ct]
        cd.host_value = host_indices.clone()
        host_lru = self.tree_core.host_lru_lists[ct]
        if cd.value is None and not host_lru.in_list(node):
            host_lru.insert_mru(node)
        self.tree_core._update_evictable_leaf_sets(node)
        if node.parent:
            self.tree_core._update_evictable_leaf_sets(node.parent)

    def _commit_prefetch(
        self,
        anchor,
        transfers: list[PoolTransfer],
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        """Fill the prefetched SWA window onto the leaf→anchor path.

        All-or-nothing over one full window: ``loaded_pages`` is the cross-rank
        MIN, so ``loaded_pages < window_pages`` drops the whole window (keeps the
        tree identical across TP ranks). Otherwise map the buffer to token range
        ``[loaded_start, total_len)`` and walk leaf→anchor, filling SWA
        tombstones and releasing slices that already have host_value.
        """
        if not transfers:
            return
        ct = self.component_type
        page_size = self.tree_core.page_size
        host_indices = transfers[0].host_indices
        # strict windows are ring-paged (one item == one window == ring tokens);
        # non-strict prefetch is page_size-paged. Use the right stride so
        # loaded_start / slice offsets line up with the carrier key length.
        stride = (
            self._swa_kv_pool_host.slot_page_size
            if self._strict_bit_exact
            else page_size
        )
        window_require_pages = (
            host_indices.numel() // stride if host_indices is not None else 0
        )
        loaded_pages = (
            pool_storage_result.extra_pool_hit_pages.get(PoolName.SWA, 0)
            if pool_storage_result
            else 0
        )
        if self._strict_bit_exact and host_indices is not None:
            # One window covers ``stride // page_size`` Full pages, all of which must
            # be present in L3. If Full was evicted under the window, drop the whole
            # buffer and recompute rather than attach a desynced window. Drop instead
            # of assert, so a benign eviction race cannot crash the server.
            stride_pages = max(1, stride // page_size)
            full_hit = pool_storage_result.kv_hit_pages if pool_storage_result else 0
            if loaded_pages * stride_pages > full_hit:
                self._release_swa_host(host_indices, cache_actions)
                return
            # The sidecar c4 / indexer state pools ride the SWA window key family,
            # so state_hit should equal loaded_pages. The file backend's
            # batch_exists MIN coupling already guarantees that, but other backends
            # (e.g. flexkv) or a partially-failing per-pool get can load SWA while a
            # coupled state page is missing, which would restore a desynced window.
            # Drop the whole window (recompute) if a registered state pool loaded
            # fewer pages than SWA. Only enforced when the pool key is present.
            extra_hit = (
                pool_storage_result.extra_pool_hit_pages if pool_storage_result else {}
            )
            for state_pool in (
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            ):
                if state_pool in extra_hit and extra_hit[state_pool] < loaded_pages:
                    self._release_swa_host(host_indices, cache_actions)
                    return
        target = (
            self.tree_core.node_by_id(insert_result.inserted_host_node)
            if insert_result is not None
            and insert_result.inserted_host_node is not None
            else None
        )
        if (
            target is None
            or window_require_pages == 0
            or loaded_pages < window_require_pages
        ):
            self._release_swa_host(host_indices, cache_actions)
            return

        if self._strict_bit_exact:
            # BACKUP_HOST / BACKUP_STORAGE attach one whole ``ring`` page to the
            # carrier node and PREFETCH forces ``window_pages == 1``, so the
            # prefetched buffer is exactly one SWA ring page (one window) and
            # attaches whole to the window's carrier -- the inserted host node,
            # keyed by its trailing Full page hash, the same node offload stored it
            # under. Never carve token sub-ranges out of a page_size radix node via
            # ``_split_node``: ``child_key(page_size)`` assumes >= page_size logical
            # units, and a ring (e.g. 131) smaller than page_size (256) has none, so
            # it raises IndexError. ``restore_pending_swa_windows`` copies this page
            # positionally into the request ring block later.
            cd_t = target.component_data[ct]
            if window_require_pages == 1 and cd_t.host_value is None:
                self._attach_swa_host_value(target, host_indices)
                _attach_state_durable_row(
                    self, target, host_indices, insert_result.total_len
                )
            else:
                # Already present, or an unexpected multi-window buffer: fail-safe
                # (recompute) -- never crash, never attach a desynced window.
                self._release_swa_host(host_indices, cache_actions)
            return

        # Buffer covers token range [loaded_start, total_len).
        loaded_start = insert_result.total_len - window_require_pages * stride

        # Walk leaf → anchor; ``pos`` is the right edge of ``cur`` in tokens.
        pos, cur = insert_result.total_len, target
        while cur is not anchor and pos > loaded_start:
            node_start = pos - len(cur.key)
            # Intersection of cur's range and the buffer.
            fill_start = max(node_start, loaded_start)
            fill_len = pos - fill_start
            buf_off = fill_start - loaded_start
            slice_ = host_indices[buf_off : buf_off + fill_len]

            cd = cur.component_data[ct]
            if cd.host_value is None and fill_len > 0:
                # Tombstone: split off the in-buffer tail if needed, then fill.
                if fill_start > node_start:
                    _, action = self.tree_core._split_node(
                        cur.key, cur, fill_start - node_start
                    )
                    if action is not None:
                        cache_actions.append(action)
                self._attach_swa_host_value(cur, slice_)
                # Independent-pool sidecar: the c4/indexer state rode this
                # window's coupled key family into the SAME durable row; point the
                # carrier at it so the reuse restores state (bit-exact boundary).
                _attach_state_durable_row(self, cur, slice_, pos)
            else:
                # Already has SWA (or empty overlap): drop this slice.
                self._release_swa_host(slice_, cache_actions)

            pos = node_start
            cur = cur.parent

        # Buffer prefix that fell outside the anchor→leaf path.
        if pos > loaded_start:
            self._release_swa_host(host_indices[: pos - loaded_start], cache_actions)

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Evict SWA host resources.
        Internal nodes: private tombstone (free SWA host only).
        Host leaves: atomic eviction via _evict_host_leaf."""
        ct = self.component_type
        if self._strict_bit_exact:
            # Bit-exact: free SWA host space only by evicting whole host leaves
            # (atomic Full+SWA), never by tombstoning an internal node's SWA
            # alone. This keeps the invariant "Full-host copy => SWA-host copy",
            # so any Full-host hit can restore its true sliding window instead
            # of reprefilling the tail. Sizing then only affects hit rate.
            full_tokens, leaves = self.tree_core.drive_host_leaf_eviction(
                num_tokens, ct, tracker, device_frees, host_frees
            )
            # Telemetry lives cache-side; the TreeCore stays free of cache policy.
            self.cache._note_binding_full_coevict(full_tokens, leaves)
            return
        host_lru = self.tree_core.host_lru_lists[ct]
        enabled = self.tree_core.enable_session_radix_cache
        if enabled:
            host_lru.cursor_begin()
            x = host_lru.cursor_next(host_lock=True)
        else:
            x = host_lru.get_lru_no_host_lock()
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):
            if not enabled:
                x_next = host_lru.get_prev_no_host_lock(x)
            cd = x.component_data[ct]
            if x in self.tree_core.evictable_host_leaves and (
                not enabled or self._can_evict_leaf_atomically(x)
            ):
                self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)
            else:
                assert cd.host_value is not None
                self.tree_core._evict_component_and_detach_lru(
                    x,
                    self,
                    target=EvictLayer.HOST,
                    tracker=tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                )
                self.tree_core._cascade_evict(
                    x,
                    self,
                    tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                    target=EvictLayer.HOST,
                )
            if enabled:
                x = host_lru.cursor_next(host_lock=True)
            else:
                x = x_next
        if enabled:
            host_lru.cursor_end()

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:
        if self._swa_kv_pool_host is None:
            return
        for host_value in host_values:
            self._swa_kv_pool_host.free(host_value)

    def apply_component_action(self, action: ComponentAction) -> None:
        alloc = self.cache.token_to_kv_pool_allocator
        if isinstance(action, FreeComponentDeviceSlot):
            for indices in action.indices:
                alloc.free_swa(indices)
            return
        if isinstance(action, FreeComponentHostSlot):
            for host_indices in action.host_indices:
                if host_indices is not None and host_indices.numel() > 0:
                    self.cache.cache_controller.append_host_mem_release(
                        extra_pools=[
                            PoolTransfer(name=PoolName.SWA, host_indices=host_indices)
                        ]
                    )
            return
        if isinstance(action, RebuildFullToSWAMapping):
            assert len(action.full_indices) == len(action.swa_indices)
            for full, swa in zip(action.full_indices, action.swa_indices):
                alloc.set_full_to_swa_mapping(full, swa)
            return
        if isinstance(action, RecoverSWAWithLockedFull):
            # Keep the locked full; remap it onto the incoming full's SWA translation,
            # freeing only the incoming full, then store the swa on the node.
            swa_value = self._translate_full_to_swa(action.incoming_full)
            alloc.set_full_to_swa_mapping(action.kept_full, swa_value)
            alloc.full_to_swa_index_mapping[action.incoming_full.to(torch.int64)] = 0
            alloc.full_attn_allocator.free(action.incoming_full)
            self._restore_device_value(action.node_id, swa_value)
            return
        if isinstance(action, SWARebuild):
            # Translate the node's source full value to SWA and store it on the node.
            swa_value = self._translate_full_to_swa(action.source_value)
            self._restore_device_value(action.node_id, swa_value)
            return
        if isinstance(action, BindCapturedSWAHost):
            self._bind_captured_swa_host(
                self.tree_core.node_by_id(action.node_id), action.node_end
            )
            return
        raise AssertionError(
            f"SWAComponent: unhandled ComponentAction {type(action).__name__}"
        )
