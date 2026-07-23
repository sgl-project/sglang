from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

logger = logging.getLogger(__name__)
_SWA_DBG_CHECKSUM = os.environ.get("SGLANG_SWA_DBG_CHECKSUM") == "1"
# Strict reuse state gate: a reuse boundary must have its c4/indexer overlap
# state durably backed, else restore can't rebuild it bit-exact (§12 dirty
# read). Intrinsic to strict bit-exact mode (no opt-out); naturally no-ops when
# state riding is unwired (_state_rides() is then empty).


# ----------------------------------------------------------------------
# c4 / c4-indexer overlap compress-state riding (Phase C, I8').
#
# In strict mode a captured SWA window is only bit-exact on reuse if the c4
# overlap state at the reuse page boundary [B-ratio, B) is ALSO restored: c4
# compression reads ``pre_kv_state`` (the prior group's raw KV/score) at each
# boundary, and the small device state ring only holds the latest few groups, so
# a reusing request would otherwise read a prior request's stale slot -- the
# dirty read the non-strict path masks via tail reprefill but the strict path
# cannot. So each SWA carrier atomically co-owns the state tiles (attn + indexer)
# captured at the same (rid, B); they ride the SWA window's full lifetime:
# bind -> BACKUP_HOST promote -> LOAD_BACK restore -> free.
#
# These are module-level functions (not methods) taking the SWA component as
# first arg, so they can be unit-tested against duck-typed fakes without a full
# component instance. Each ride tuple is
#   (host_pool, device_state_pools, node_host_value_attr, node_pending_attr, li_map)
# ----------------------------------------------------------------------

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


def _attach_state_durable_row(component, node, swa_slice) -> None:
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
    swa_row = int(swa_slice[0].item()) // hp0.slot_page_size
    for hp, _pools, host_value_attr, _pending, _li in _state_rides(component):
        # Only pools with a reserved durable region are addressed by SWA row.
        if not int(getattr(hp, "_durable_reserve_slots", 0) or 0):
            continue
        setattr(node, host_value_attr, _state_durable_indices(hp, swa_row))


def _free_state_bindings(component, node) -> None:
    """Release both pending and durable state tiles back to their host pools (SWA
    carrier dropped without a durable host backup, or node removed)."""
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


def _restore_state_windows(component, node, swa_chunk: torch.Tensor) -> None:
    """Restore the c4 / c4-indexer overlap state for the reused window onto the
    device state ring, so the reusing request's boundary read is bit-exact.

    swa_chunk are the SWA slots just restored for this window; its trailing ratio
    slots are the boundary group [B-ratio, B). Each slot's device state row is
    translate_from_swa_loc_to_state_loc(slot) and its host tile offset is
    slot % ring_size (the device state ring is indexed by slot % ring_size, a pure
    function of the slot),
    so the captured window lands on the exact rows the compressor will read,
    regardless of the reusing request's slot base.
    """
    rides = _state_rides(component)
    if not rides:
        return
    for hp, pools, host_value_attr, _pending_attr, li_map in rides:
        host_value = getattr(node, host_value_attr, None)
        if host_value is None:
            continue
        ring = hp.slot_page_size
        slot_bytes = hp.item_bytes // ring
        page_row = int(host_value[0].item()) // ring
        for layer_id, li in li_map.items():
            sp = pools[layer_id] if layer_id < len(pools) else None
            if sp is None:
                continue
            ratio = sp.ratio
            win = swa_chunk[-ratio:]
            state_locs = sp.translate_from_swa_loc_to_state_loc(win)
            off0 = 0  # pack at tile start; must match capture off0=0
            dev = sp.kv_score_buffer.kv_score
            host_tile = hp.data_refs[li][page_row]
            flat = host_tile[off0 * slot_bytes : (off0 + ratio) * slot_bytes]
            window = flat.view(dev.dtype).reshape(ratio, -1).to(device=dev.device)
            dev[state_locs] = window
            if _SWA_DBG_CHECKSUM:
                _dbg_verify_state_restore(
                    node,
                    host_value_attr,
                    hp,
                    li,
                    page_row,
                    off0,
                    flat,
                    dev,
                    state_locs,
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


from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
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
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    TreeComponent,
    next_component_uuid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
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
            cache.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        ), f"SWAComponent requires SWATokenToKVPoolAllocator, got {type(cache.token_to_kv_pool_allocator)}"
        super().__init__(cache, params)
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
                self.cache.lru_lists[
                    self.component_type
                ].reset_node_and_window_ancestors_mru(
                    node,
                    root_node,
                    self.sliding_window_size + self.cache.page_size,
                    self.node_has_component_data,
                )
            case _:
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")

    def _restore_device_value(self, node: UnifiedTreeNode, value: torch.Tensor) -> None:
        ct = self.component_type
        node.component_data[ct].value = value
        # A freshly (re)assigned device SWA value is live for the current
        # holder; drop any stale deferred owner-release intent from a prior life.
        if getattr(node, "_swa_release_pending", False):
            node._swa_release_pending = False
        host_lru = self.cache.host_lru_lists[ct]
        if host_lru.in_list(node):
            host_lru.remove_node(node)
        self.cache.lru_lists[ct].insert_mru(node)
        self.cache.component_evictable_size_[ct] += len(value)

    def _restore_device_value_with_locked_full(
        self,
        node: UnifiedTreeNode,
        full_value: torch.Tensor,
        incoming_full_value: torch.Tensor,
    ) -> None:
        allocator = self.cache.token_to_kv_pool_allocator
        swa_value = self._translate_full_to_swa(incoming_full_value)
        allocator.set_full_to_swa_mapping(full_value, swa_value)
        allocator.full_to_swa_index_mapping[incoming_full_value.to(torch.int64)] = 0
        allocator.full_attn_allocator.free(incoming_full_value)
        self._restore_device_value(node, swa_value)

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        sliding_window_size = self.sliding_window_size
        ct = self.component_type
        strict_bit_exact = self._strict_bit_exact
        # stride model (I2-inv): sparse-SWA reuse correctness relies on THIS
        # runtime clamp -- the reuse validator rejects pages without a durable
        # SWA host copy so the boundary clamps to the nearest page that has a
        # window -- NOT on an insert-time SWA>=Full superset guard (former S5,
        # retired). Under stride>1 non-stride pages legitimately have a Full-host
        # copy but no SWA window; that is a normal, expected non-reuse boundary.
        #
        # Strict state-ride gate (regression fix, step 2): a reuse boundary is
        # only bit-exact if its c4/c4-indexer overlap state also survives -- else
        # restore cannot rebuild the boundary and reuse would read a dirty state
        # ring (§12). So on the REUSE match a node whose durable state host value
        # is missing is treated like a missing SWA window: it resets the running
        # window length, letting the accumulate-reset logic clamp the boundary
        # back to the nearest page that has BOTH a window and its state -- instead
        # of dropping such windows at bind time (which zeroed partial reuse).
        # Empty when state riding is unwired -> no gating (unchanged behavior).
        state_ride_attrs = tuple(
            hv_attr for (_hp, _pools, hv_attr, _pend, _li) in _state_rides(self)
        )
        state = {"len": float("inf")}

        # unified_kv never caches the SWA ring (per-request, not content-stable),
        # so SWA bookkeeping must not gate the match here.
        swa_device_only_hicache = (
            self._swa_kv_pool_host is None and self.cache.cache_controller is not None
        )

        def validator(node: UnifiedTreeNode) -> bool:
            cd = node.component_data[ct]
            # HiCache: a host-only tombstone is a valid match boundary too
            # — load_back will restore SWA from host before use.
            if cd.value is None and (match_device_only or cd.host_value is None):
                state["len"] = 0
                if swa_device_only_hicache and (node.backuped or not node.evicted):
                    return True
                # Bit-exact device-anchor path: the SWA device ring only keeps
                # the last `sliding_window_size` tokens, so an out-of-window
                # node's SWA slot (cd.value) is recycled to None even though its
                # SWA truth is durably host-backed and its FULL KV is still
                # device resident. On the device-anchor lookup (match_device_only)
                # such a node must NOT truncate the FULL device anchor:
                # cache_unfinished_req's self-match needs device_indices to cover
                # the whole FULL-resident prefix (else cache_protected_len can
                # exceed len(new_indices)), and the SWA window is restored
                # positionally from host on reuse. The FULL validator still gates
                # FULL device residency, and reuse over-reach is reclamped to the
                # host-gated boundary via for_reuse, so this is safe both ways.
                if match_device_only and cd.host_value is not None:
                    return True
                return False
            # I2-prime: strict bit-exact never trusts the per-request device SWA
            # ring as a cross-request truth source for the REUSE boundary (the
            # device ring is recycled when the owner's req_pool_idx is reused).
            # Only a durable host copy counts for the device-or-host reuse match;
            # a node with device value but no host copy truncates the reuse match
            # so it restores from host (LOAD_BACK) or recomputes (I6) instead of
            # serving a stale device ring. This is scoped to the reuse match
            # (``not match_device_only``): the device-only match must still report
            # a request's own freshly-computed, not-yet-backed-up nodes as device
            # resident, else cache_unfinished_req's self-match returns empty
            # device indices (new_prefix_len > len(new_indices)). Stale device
            # residency across requests is instead closed by the deferred
            # owner-release tombstone, which nulls the device value once the host
            # copy is durable.
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

    def finalize_match_result(
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
        root = self.cache.root_node
        # Mine 2 (warm reuse): on the reuse path in strict mode, the per-request
        # device SWA ring is not a durable cross-request truth (I2'), so a node
        # that is BOTH device-resident and host-backed must still be counted as
        # a host hit -- otherwise swa_host_hit_length stays 0 and the load_back
        # gate never opens for it. This uses the SAME host-backed predicate as
        # build_hicache_transfers(LOAD_BACK) below. Self-match (for_reuse=False)
        # keeps the OLD behavior: cd.value is trusted first, since the request's
        # own freshly-computed nodes aren't host-backed yet and
        # cache_unfinished_req relies on this not falsely opening the gate.
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
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{self.component_type}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        if swa_evicted_seqlen <= total_prefix_len:
            # Branch 1: entire value_slice is within SWA window — recover
            if full_cd.lock_ref > 0:
                self._restore_device_value_with_locked_full(
                    node, full_cd.value, value_slice
                )
                return 0
            self.cache.token_to_kv_pool_allocator.free(full_cd.value)
            full_cd.value = value_slice.clone()
            swa_value = self._translate_full_to_swa(full_cd.value)
            self._restore_device_value(node, swa_value)
            return 0
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # Branch 2: value_slice[start_idx:] is within SWA window — partial recover
            start_idx = swa_evicted_seqlen - total_prefix_len
            if full_cd.lock_ref > 0:
                self.cache._split_node(node.key, node, start_idx)
                full_cd = node.component_data[BASE_COMPONENT_TYPE]
                self._restore_device_value_with_locked_full(
                    node, full_cd.value, value_slice[start_idx:]
                )
                return start_idx
            self.cache.token_to_kv_pool_allocator.free(full_cd.value[start_idx:])
            self.cache._split_node(node.key, node, start_idx)
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice[
                start_idx:
            ].clone()
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)
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
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{ct}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        full_value = node.component_data[BASE_COMPONENT_TYPE].value
        if swa_evicted_seqlen <= total_prefix_len:
            swa_value = self._translate_full_to_swa(full_value)
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            start_idx = swa_evicted_seqlen - total_prefix_len
            self.cache._split_node(node.key, node, start_idx)
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            swa_value = self._translate_full_to_swa(full_value)
        else:
            return
        self._restore_device_value(node, swa_value)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        if not is_new_leaf:
            return

        node_start = result.prefix_len
        split_pos = params.swa_evicted_seqlen - node_start

        if split_pos <= 0:
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)
        elif split_pos < len(node.key):
            # Node straddles the SWA eviction boundary
            # Split into parent (tombstone, no SWA) and child (with SWA)
            # After _split_node, `node` becomes the child
            tombstone_parent = self.cache._split_node(node.key, node, split_pos)
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)
            # Stride model: the out-of-window tombstone span
            # [node_start, swa_evicted_seqlen) also carries prefill-captured
            # windows at the interior stride page boundaries. Claim them as
            # host-only carrier nodes so cross-request reuse clamps to the
            # nearest stride page instead of the coarse chunk end.
            self._bind_interior_captured_swa_hosts(
                tombstone_parent, node_start, params.swa_evicted_seqlen
            )
        else:
            # Entire leaf is outside the SWA window — left as a tombstone.
            return

        # Bind the prefill-captured host window to this SWA node. Both branches
        # above leave `node` covering [swa_start, swa_start + len(value)) with
        # swa_start == max(node_start, swa_evicted_seqlen).
        swa_start = max(node_start, params.swa_evicted_seqlen)
        self._bind_captured_swa_host(node, swa_start)
        if split_pos <= 0:
            # Stride reuse (in-window leaf): the capture (``capture_swa_windows``)
            # stages a host window at EVERY stride page boundary of this leaf, but
            # only the leaf-end window is claimed above; the interior ones are
            # otherwise freed unused at ``cleanup_after_caching_req`` and
            # cross-request reuse clamps to the coarse chunk/leaf end. Claim them
            # here as host carriers: split at each staged stride boundary and stash
            # the window as ``_swa_pending_host`` (co-lifetime -> host_value at
            # BACKUP_HOST). The device SWA value is re-sliced by
            # ``redistribute_on_node_split`` (no ring-slot alloc/accounting change),
            # so this is purely additive host state that lets the reuse validator
            # extend to stride granularity. ``node`` (the suffix) keeps the
            # leaf-end pending window across the splits.
            node_end = swa_start + len(node.component_data[self.component_type].value)
            self._bind_interior_captured_swa_hosts(node, swa_start, node_end)
        self._maybe_split_leaf_for_swa_lock(node)

    def _maybe_split_leaf_for_swa_lock(self, leaf: UnifiedTreeNode) -> None:
        """Cap a fresh SWA leaf at one page-aligned window so locking it pins
        only one window of SWA pool, not the whole (long chunked-prefill) leaf.
        """
        ct = self.component_type
        cd = leaf.component_data[ct]
        if leaf is self.cache.root_node or cd.value is None or cd.lock_ref > 0:
            return

        page_size = self.cache.page_size
        # Smallest page-aligned size that still covers the sliding window.
        tail_size = (self.sliding_window_size + page_size - 1) // page_size * page_size
        leaf_len = len(leaf.key)
        if leaf_len <= tail_size:
            return
        split_at = leaf_len - tail_size
        if page_size > 1 and (split_at % page_size != 0 or leaf_len % page_size != 0):
            return

        self.cache._split_node(leaf.key, leaf, split_at)

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.component_data[self.component_type].lock_ref = child.component_data[
            self.component_type
        ].lock_ref

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
            host_lru = self.cache.host_lru_lists[self.component_type]
            if len(child_swa_host_value) == full_span:
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

    def free_pending_host_on_remove(self, node: UnifiedTreeNode) -> None:
        """Free an interior stride carrier's not-yet-promoted capture page at the single
        node-removal chokepoint.

        Pending's lifetime tracks the Full (base) component, so it is dropped only here
        (node truly leaves the tree), never on a mere SWA device tombstone. Called from
        unified_radix_cache._remove_leaf_from_parent, which runs regardless of
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

    def evict_component(
        self,
        node: UnifiedTreeNode,
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
            self.cache.token_to_kv_pool_allocator.free_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            freed = len(cd.value)
            self.cache.component_evictable_size_[ct] -= freed
            cd.value = None
            # Co-lifetime: a captured page not yet promoted to host_value must
            # not outlive its device SWA; free it (node degrades to recompute).
            #
            # R1 exemption: interior stride carriers are out-of-window by
            # construction, so this device SWA tombstone (SWA pool pressure)
            # always fires before the finish-time coordinated BACKUP_HOST. Their
            # pending page is a standalone bit-exact host copy whose lifetime
            # tracks the Full (base) component (the node survives here because
            # base is still resident), not the SWA device ring. Keep it so the
            # coordinated base backup can still promote it (I3 preserved). It is
            # freed only at true node removal (_remove_leaf_from_parent).
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
        host_lru = self.cache.host_lru_lists[ct]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            if self._swa_kv_pool_host is not None:
                self._swa_kv_pool_host.free(cd.host_value)
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
            # Host copy not durable yet (async write_through backup still in
            # flight) or another request still holds the SWA lock, so we cannot
            # free the device ring value right now. But once this owner is gone
            # the per-request ring slot is recycled and its bytes become stale,
            # so the value MUST NOT be trusted for cross-request reuse. Defer:
            # mark the node so the coordinated BACKUP_HOST commit drops the
            # device value the instant the host copy becomes durable and no
            # holder remains. Without this the device ring is never invalidated
            # and reuse would keep a stale device slot alive (I1/I2 violation).
            node._swa_release_pending = True
            return
        self.cache._evict_component_and_detach_lru(node, self, target=EvictLayer.DEVICE)

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 1

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.swa_num_tokens
        ct = self.component_type
        lru = self.cache.lru_lists[ct]
        x = lru.get_lru_no_lock()
        while tracker[ct] < request and x is not None and lru.in_list(x):
            assert x.component_data[ct].value is not None
            if x in self.cache.evictable_device_leaves:
                # D-leaf: atomic eviction of all components
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_device_leaf(x, tracker)
                if not lru.in_list(x_next):
                    x_next = lru.get_lru_no_lock()
                x = x_next
            else:
                # Internal: tombstone SWA + cascade
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.DEVICE, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        root = self.cache.root_node
        sliding_window_size = self.sliding_window_size
        swa_lock_size = 0
        swa_uuid = None
        uuid_key = "host_uuid" if lock_host else "uuid"
        lru = self.cache.host_lru_lists[ct] if lock_host else self.cache.lru_lists[ct]

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
                    self.cache.component_evictable_size_[ct] -= key_len
                    self.cache.component_protected_size_[ct] += key_len
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
        root = self.cache.root_node
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
                        host_lru = self.cache.host_lru_lists[ct]
                        if not host_lru.in_list(cur):
                            host_lru.insert_mru(cur)
                else:
                    key_len = len(comp.value)
                    self.cache.component_evictable_size_[ct] += key_len
                    self.cache.component_protected_size_[ct] -= key_len
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
        swa_uuid_for_lock: Optional[int] = None,
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
        root = self.cache.root_node

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
                self.cache.component_protected_size_[ct] -= key_len
                self.cache.component_evictable_size_[ct] += key_len
                if self.cache._is_device_leaf(cur):
                    self.cache._evict_component_and_detach_lru(
                        cur, self, target=EvictLayer.DEVICE
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
            )
        insert_params.swa_evicted_seqlen = req.kv.swa_evicted_seqlen

    # ---- HiCache Hooks ----

    def _bind_captured_swa_host(self, node: UnifiedTreeNode, swa_start: int) -> None:
        """Stash the prefill-captured host page as a pending ref on the node.

        We do not set host_value here: the SWA host_value must not exist before the
        node's Full host_value, so it is attached later by the coordinated BACKUP_HOST
        commit. Until then the page is held in node._swa_pending_host and freed on
        device eviction if the node is never backed up.

        The node ends at boundary B with captured window [B-win, B) keyed (rid, B).
        If it was not captured (pool full / outside this chunk), leave it to the
        normal backup / recompute path.
        """
        if _SWA_DBG_CHECKSUM:
            logger.warning("[BIND-DBG] enter swa_start=%s", swa_start)
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
        # The node ends at page boundary B = swa_start + len(value); its host
        # copy is the single captured window keyed (rid, B). The earlier,
        # out-of-window part of the node is never attended and is not stored.
        node_end = swa_start + len(cd.value)
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
        # Decoupled co-lifetime (regression fix): claim the c4/c4-indexer overlap
        # -state tiles at the same (rid, node_end) IF present, but NEVER drop the
        # SWA window when they are missing. The window rides on its own; a
        # boundary that lacks its state is instead excluded from the strict REUSE
        # boundary by create_match_validator (graceful clamp to the nearest
        # window+state boundary). This avoids both (a) the hard clamp-to-0 that
        # dropping every state-less interior window caused (partial-prefix reuse
        # regression) and (b) the §12 dirty read (a state-less boundary is simply
        # not crossed on reuse). No-op when strict state offload is unwired.
        _bind_state_rides(self, node, rid, int(node_end))
        if _SWA_DBG_CHECKSUM:
            crc_map = getattr(hp, "_capture_crc", None)
            if crc_map:
                keys = [k for k in crc_map if k[0] == rid and k[1] == int(node_end)]
                if keys:
                    cd.metadata["dbg_swa_crc"] = {k[2]: crc_map.pop(k) for k in keys}

    def _bind_interior_captured_swa_hosts(
        self, region_node: UnifiedTreeNode, region_start: int, region_end: int
    ) -> None:
        """Claim the prefill-captured windows at the interior stride page boundaries of
        a chunk's out-of-window tombstone span.

        _bind_captured_swa_host only claims the single window at the chunk end. The
        finer stride-gated windows the capture also offloaded (keyed (rid, B) for
        interior boundaries) would otherwise be dropped by cleanup_after_caching_req,
        clamping the reuse boundary to the coarse chunk end. Here we split region_node
        at each staged boundary B so a node ends at B, and stash the captured window
        as that node's _swa_pending_host.

        host_value is not set here: the window rides the same deferred path as the
        tail (_swa_pending_host -> BACKUP_HOST). These interior carriers have no
        device SWA value (they are outside the sliding window); the BACKUP_HOST
        transfer builder adopts the pending page for such device-less nodes.
        """
        hp = self._swa_kv_pool_host
        if hp is None:
            return
        staging = getattr(hp, "_capture_staging", None)
        rid = self._capture_rid
        if not staging or rid is None:
            return
        win = hp.slot_page_size
        page = self.cache.page_size
        boundaries = sorted(
            int(b)
            for (r, b) in list(staging.keys())
            if r == rid and region_start < int(b) < region_end
        )
        if not boundaries:
            return
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
            new_parent = self.cache._split_node(cur.key, cur, split_len)
            new_parent._swa_pending_host = host_value
            # Decoupled co-lifetime (regression fix): claim the interior carrier's
            # state tiles at (rid, B) IF present, but keep the SWA window even on a
            # miss. A state-less interior boundary is excluded from the strict
            # reuse boundary by the match validator (graceful clamp), instead of
            # dropping the window here -- which zeroed partial-prefix reuse because
            # interior state gets evicted/exhausted far more than the chunk tail.
            _bind_state_rides(self, new_parent, rid, B)
            # R1: mark interior stride carrier. Its captured page lifetime tracks
            # the Full (base) component (dropped only at true node removal), NOT
            # the SWA device ring (which is always recycled out-of-window before
            # the finish-time coordinated BACKUP_HOST). See evict_component and
            # unified_radix_cache._remove_leaf_from_parent.
            new_parent._swa_interior_carrier = True
            cur = new_parent

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        # Release any capture staging owned by this request that no node claimed
        # (interior / out-of-window windows), then drop the stashed rid.
        # Key off the request's own req_pool_idx (not the stashed _capture_rid):
        # on the retract/abort path caching runs with is_insert=False, so
        # prepare_for_caching_req -- and thus _capture_rid -- never ran. Decode
        # capture (Task B0) stages (req_pool_idx, B) across many steps, so relying
        # on _capture_rid would leak those windows here and, once req_pool_idx is
        # recycled, risk mis-binding a stale window to a new request. For the
        # is_insert=True path req.req_pool_idx == _capture_rid, so behavior is
        # unchanged. Fall back to the stashed rid only if the slot is already gone.
        hp = self._swa_kv_pool_host
        rid = req.req_pool_idx if req.req_pool_idx is not None else self._capture_rid
        self._capture_rid = None
        if hp is None or rid is None:
            return
        staging = getattr(hp, "_capture_staging", None)
        if not staging:
            return
        leftover = [k for k in staging if k[0] == rid]
        for k in leftover:
            hp.free(staging.pop(k))
        if _SWA_DBG_CHECKSUM:
            crc_map = getattr(hp, "_capture_crc", None)
            if crc_map:
                for k in [k for k in crc_map if k[0] == rid]:
                    crc_map.pop(k, None)

    def _swa_l3_key(self, node) -> str:
        """L3 key for a carrier node's captured SWA window.

        I4' couples SWA-L3 to Full-L3: the window is keyed by the carrier's own
        Full page hash (hash_value[-1]), so it lives and dies with that Full
        page in the storage backend. Centralized here so a future namespace
        change touches one place.
        """
        return node.hash_value[-1]

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        req: Optional[Req] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        # unified_kv keeps SWA as a device-only ring.
        if self._swa_kv_pool_host is None and self.cache.cache_controller is not None:
            return None

        if phase == CacheTransferPhase.BACKUP_HOST:
            cd = node.component_data[ct]
            if cd.host_value is not None:
                # Already populated from a prior backup; do not re-copy.
                return None
            pending = getattr(node, "_swa_pending_host", None)
            if pending is not None:
                # Co-lifetime: adopt the prefill-captured host page (already on
                # host) through the coordinated backup, so SWA host_value is set
                # together with Full host_value (never before). device_indices is
                # None -> write_backup skips the (redundant) device->host copy.
                # This also covers interior stride carrier nodes, which have no
                # device SWA value (out of the sliding window) but do hold a
                # captured window; the pending check must precede the
                # ``cd.value is None`` guard below so they are not skipped.
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
                # reuse (I6). Never back up the device ring here -- it holds only
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
            while cur is not self.cache.root_node and n_swa < self.sliding_window_size:
                cd = cur.component_data[ct]
                assert cd.host_value is not None or cd.value is not None
                if self._strict_bit_exact and cd.host_value is not None:
                    # Mine 2 (warm reuse): the per-request device SWA ring is
                    # not a durable cross-request truth in strict mode, even
                    # when `cd.value` is still set (stale, recycled slot from
                    # a prior request). Collect the host copy so it is loaded
                    # and commit_hicache_transfer's _restore_device_value
                    # overwrites the stale slot with host truth. Same
                    # host-backed predicate as finalize_match_result's
                    # for_reuse=True gate above.
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

            if self._unified_positional_swa and req is not None:
                # unified_kv: the SWA device slot is req_pool_idx*ring + pos%ring,
                # computable only after prepare_for_extend assigns req_pool_idx.
                # Do NOT allocate a device slot / issue an H->D transfer here
                # (the swa_attn_allocator is page_size=256 and unrelated to the
                # positional ring -> alloc(win) returns empty -> crash). Stash the
                # window pages (host_value, in token order) on the req; the
                # scheduler restores them positionally before the first forward.
                req._swa_restore_windows = list(zip(nodes, backed_up))
                return None

            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.cat(backed_up),
                    device_indices=None,
                    nodes_to_load=nodes,
                )
            ]

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            cd = node.component_data[ct]
            if cd.host_value is None or not node.hash_value:
                return None
            if self._strict_bit_exact:
                # I4' (SWA-L3 => Full-L3): persist the captured window under the
                # carrier node's own Full page hash so the SWA window and its
                # Full page share one L3 operation/key-family lifetime. A strict
                # carrier holds exactly one window page (== slot_page_size); it
                # is stored as a single trailing page keyed by hash_value[-1],
                # mirroring the mamba sidecar pattern. Windows that are not a
                # whole page (partial / mis-sized) are skipped, not truncated.
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
            num_pages = len(cd.host_value) // self.cache.page_size
            if num_pages == 0:
                return None
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=cd.host_value[-num_pages * self.cache.page_size :],
                    keys=node.hash_value[-num_pages:],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        if phase == CacheTransferPhase.PREFETCH:
            if self._strict_bit_exact:
                # I4' trailing-window prefetch. Request the sliding window that
                # ends at the prefetch tail as `window_pages` placeholder keys;
                # the controller (`_sync_trailing_keys`) rewrites them to the
                # *actual* trailing Full page hashes of the storage hit range.
                # Each requested window page is thus keyed by the same Full hash
                # its carrier stored it under (BACKUP_STORAGE keys=[hash_value
                # [-1]] == _swa_l3_key), so a window is fetched iff its Full page
                # also hit -- SWA can never outlive its Full page (re-checked by
                # the coupling guard in _commit_prefetch). Using `node.hash_value`
                # here would be wrong twice over: it is the *matched prefix*
                # (empty after a cache flush, the exact case L3 must serve), and
                # `_sync_trailing_keys` overwrites the keys regardless.
                #
                # Window granularity is the host SWA pool slot_page_size (ring):
                # unified_kv packs one full sliding window per page (ring ==
                # sliding_window -> window_pages == 1), a per-page ring needs
                # ceil(window / ring) *contiguous* pages. The ring stride (not
                # page_size) is what forces window_pages == 1 under unified_kv,
                # which is required because `_sync_trailing_keys` hands back
                # contiguous Full-page hashes -- only 1 maps cleanly onto a
                # ring-spaced carrier; N contiguous KV hashes would not.
                ring = self._swa_kv_pool_host.slot_page_size
                window_pages = max(1, (self.sliding_window_size + ring - 1) // ring)
                # Not worth a partial window: need at least one ring of freshly
                # prefetched Full tokens behind the tail.
                if prefetch_tokens < ring:
                    return None
                num_tokens = window_pages * ring
                host_indices = self._swa_kv_pool_host.alloc(num_tokens)
                if host_indices is None:
                    self.cache.evict_host(num_tokens, ComponentType.SWA)
                    host_indices = self._swa_kv_pool_host.alloc(num_tokens)
                if host_indices is None:
                    return []
                return [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=host_indices,
                        keys=["__placeholder__"] * window_pages,
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                ]
            # non-strict: require a full contiguous sliding window (unchanged).
            sw_pages = (
                self.sliding_window_size + self.cache.page_size - 1
            ) // self.cache.page_size
            if sw_pages == 0 or prefetch_tokens // self.cache.page_size < sw_pages:
                return None
            num_tokens = sw_pages * self.cache.page_size
            host_indices = self._swa_kv_pool_host.alloc(num_tokens)
            if host_indices is None:
                self.cache.evict_host(num_tokens, ComponentType.SWA)
                host_indices = self._swa_kv_pool_host.alloc(num_tokens)
            if host_indices is None:
                return []
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
            # Deferred owner-release tombstone: if the owning request finished
            # while this host backup was still in flight (host_value was None at
            # cache_finished_req, so evict_device_on_owner_release deferred the
            # device free), drop the now-recycled per-request device SWA value
            # now that the host copy is durable and no holder remains. This
            # closes the async write_through race where the device ring would
            # otherwise stay alive and be trusted on cross-request reuse.
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
                    self.cache._evict_component_and_detach_lru(
                        node, self, target=EvictLayer.DEVICE
                    )
            return

        if phase == CacheTransferPhase.LOAD_BACK:
            assert transfers and transfers[0].device_indices is not None
            xfer = transfers[0]
            device_indices = xfer.device_indices
            allocator = self.cache.token_to_kv_pool_allocator

            offset = 0
            for n in xfer.nodes_to_load or []:
                cd_n = n.component_data[ct]
                n_tokens = len(cd_n.host_value)
                swa_chunk = device_indices[offset : offset + n_tokens].clone()
                self._restore_device_value(n, swa_chunk)
                # host_value holds the sliding window [B-n_tokens, B). Map its
                # full indices to the restored SWA slots (out-of-window full
                # tokens keep sentinel 0, never read under the SWA mask). The
                # window may extend before this node own start when the node was
                # split shorter than the window (its host_value still spans the
                # whole window); gather the window full indices across the node
                # and its ancestors, in token order, so full<->swa lengths
                # match. In the common (unsplit) case the node own full value
                # already has >= n_tokens and no ancestor is touched.
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
                allocator.set_full_to_swa_mapping(
                    window_full, swa_chunk[-window_full.numel() :]
                )
                if _SWA_DBG_CHECKSUM and hasattr(self, "_dbg_verify_restore"):
                    self._dbg_verify_restore(cd_n)
                offset += n_tokens
            assert offset == len(xfer.host_indices)
            return

        if phase == CacheTransferPhase.PREFETCH:
            self._commit_prefetch(
                node,
                transfers,
                insert_result=insert_result,
                pool_storage_result=pool_storage_result,
            )
            return

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
        host_idx = torch.cat([hv.to(torch.int64) for _, hv in windows])
        # Only the trailing `ring` tokens fit the per-request ring block; a
        # page-aligned reuse window is exactly one host page (== ring tokens).
        if host_idx.numel() != ring:
            if _SWA_DBG_CHECKSUM:
                logger.warning(
                    "[LB-RESTORE] skip non-single-page window: host_tokens=%s ring=%s",
                    int(host_idx.numel()),
                    ring,
                )
            return
        r = int(req_pool_idx)
        base = r * ring
        device_idx = torch.arange(
            base, base + ring, dtype=torch.int64, device=hp.gpu_device
        )
        host_idx = host_idx.to(hp.gpu_device)
        # H2 gate: the host page for this window was written by a capture D2H
        # (prefill window capture or decode-source capture), enqueued
        # non_blocking on the forward/compute stream and followed by a recorded
        # capture-completion event. Make this restore H2D wait on it before
        # reading the host page so a cross-stream / cross-batch reuse never
        # restores a half-written window. This no longer relies on the consumer
        # happening to share the producer's stream (the previous decode "ordered
        # by construction" assumption, which silently broke under the scheduler
        # overlap when producer and consumer could differ). No-op when no capture
        # has run on this pool yet.
        if hasattr(hp, "wait_capture_done"):
            hp.wait_capture_done()
        # B1.3: restore every layer in one fused transfer instead of a Python
        # loop of `layer_num` per-layer copies (61 launches -> 1). Byte-for-byte
        # identical device landing (same transfer primitive + page indices); it
        # only removes launch/Python overhead on the prefill reuse hot path.
        if hasattr(hp, "load_to_device_all_layer"):
            hp.load_to_device_all_layer(None, host_idx, device_idx, io_backend)
        else:
            for li in range(hp.layer_num):
                hp.load_to_device_per_layer(None, host_idx, device_idx, li, io_backend)
        # Phase C: ride the c4/c4-indexer overlap state back onto the device
        # state ring for this reused window (device_idx is the restored ring
        # block; its trailing `ratio` slots are the boundary group). Gate each
        # state pool's own capture-done event first (hazard H2, cross-stream H2D
        # safety), mirroring the SWA wait above. No-op when state offload is
        # unwired or the node carries no state host value.
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
        root = getattr(self.cache, "root_node", None)
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

    def _release_swa_host(self, host_indices: torch.Tensor) -> None:
        if host_indices is not None and host_indices.numel() > 0:
            self.cache.cache_controller.append_host_mem_release(
                extra_pools=[PoolTransfer(name=PoolName.SWA, host_indices=host_indices)]
            )

    def _attach_swa_host_value(
        self, node: UnifiedTreeNode, host_indices: torch.Tensor
    ) -> None:
        """Write host_indices into node's SWA host_value and refresh tree state."""
        ct = self.component_type
        cd = node.component_data[ct]
        cd.host_value = host_indices.clone()
        host_lru = self.cache.host_lru_lists[ct]
        if cd.value is None and not host_lru.in_list(node):
            host_lru.insert_mru(node)
        self.cache._update_evictable_leaf_sets(node)
        if node.parent:
            self.cache._update_evictable_leaf_sets(node.parent)

    def _commit_prefetch(
        self,
        anchor,
        transfers: list[PoolTransfer],
        *,
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
        page_size = self.cache.page_size
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
            # I4' coupling guard: one window covers ``stride // page_size`` Full
            # pages, all of which must be present in L3 (kv_hit_pages). If Full
            # was evicted under the window (SWA outlived Full), drop the whole
            # buffer and fall back to recompute -- never attach a desynced
            # window. Fail-safe (drop), not assert, so a benign eviction race
            # cannot crash the server.
            stride_pages = max(1, stride // page_size)
            full_hit = pool_storage_result.kv_hit_pages if pool_storage_result else 0
            if loaded_pages * stride_pages > full_hit:
                self._release_swa_host(host_indices)
                return
            # Co-lifetime STATE coupling guard (defense-in-depth): the sidecar
            # c4 / indexer state pools ride the SWA window key family, so under
            # co-lifetime state_hit == loaded_pages. The file backend's
            # batch_exists MIN coupling already enforces this (no-op here), but
            # other backends (e.g. flexkv) or a partially-failing per-pool get can
            # load SWA while a coupled state page is missing -- attaching state
            # would then restore a desynced (dirty) window. Drop the whole window
            # (recompute) if a registered state pool loaded fewer pages than SWA;
            # only enforced when the pool key is present.
            extra_hit = (
                pool_storage_result.extra_pool_hit_pages if pool_storage_result else {}
            )
            for state_pool in (
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            ):
                if state_pool in extra_hit and extra_hit[state_pool] < loaded_pages:
                    self._release_swa_host(host_indices)
                    return
        target = insert_result.inserted_host_node if insert_result else None
        if (
            target is None
            or window_require_pages == 0
            or loaded_pages < window_require_pages
        ):
            self._release_swa_host(host_indices)
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
                    self.cache._split_node(cur.key, cur, fill_start - node_start)
                self._attach_swa_host_value(cur, slice_)
                # Independent-pool sidecar: the c4/indexer state rode this
                # window's coupled key family into the SAME durable row; point the
                # carrier at it so the reuse restores state (bit-exact boundary).
                _attach_state_durable_row(self, cur, slice_)
            else:
                # Already has SWA (or empty overlap): drop this slice.
                self._release_swa_host(slice_)

            pos = node_start
            cur = cur.parent

        # Buffer prefix that fell outside the anchor→leaf path.
        if pos > loaded_start:
            self._release_swa_host(host_indices[: pos - loaded_start])

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
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
            self.cache.drive_host_leaf_eviction(num_tokens, ct, tracker)
            return
        host_lru = self.cache.host_lru_lists[ct]
        x = host_lru.get_lru_no_host_lock()
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):
            x_next = host_lru.get_prev_no_host_lock(x)
            cd = x.component_data[ct]
            if x in self.cache.evictable_host_leaves:
                self.cache._evict_host_leaf(x, tracker)
            else:
                assert cd.host_value is not None
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.HOST, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker, target=EvictLayer.HOST)
            x = x_next
