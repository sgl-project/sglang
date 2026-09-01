from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

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
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    FreeComponentHostSlot,
    FreeDeviceKVFullOnly,
    RebuildFullToSWAMapping,
    RecoverSWAWithLockedFull,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
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
        self.full_window_pages = (
            self.sliding_window_size + params.page_size - 1
        ) // params.page_size
        # HiCache state: set to host SWA pool when HiCache enabled
        self._swa_kv_pool_host = None

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

    def _unified_allocator(self):
        """The unified SWA composite, or None when running on the static pool."""
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedSWATokenToKVPoolAllocator,
        )

        allocator = self.cache.token_to_kv_pool_allocator
        if isinstance(allocator, UnifiedSWATokenToKVPoolAllocator):
            return allocator
        return None

    def _page_pairs(
        self, full_value: torch.Tensor, incoming_full_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Page ids of two token ranges that address the SAME logical tokens.

        Dedupes by FIRST OCCURRENCE with one shared mask rather than
        `torch.unique`: unique sorts by id value, and allocation hands out
        virtual ids in no particular order, so sorting would pair page k of one
        range with an unrelated page of the other. One mask keeps the pairing
        positional, hence logical.
        """
        page_size = self.tree_core.page_size
        kept = full_value.detach().to(torch.int64) // page_size
        incoming = incoming_full_value.detach().to(torch.int64) // page_size
        assert kept.numel() == incoming.numel(), (
            f"locked-full recovery needs a 1:1 token correspondence, got "
            f"{kept.numel()} kept vs {incoming.numel()} incoming"
        )
        starts = torch.ones_like(kept, dtype=torch.bool)
        starts[1:] = kept[1:] != kept[:-1]
        incoming_starts = torch.ones_like(incoming, dtype=torch.bool)
        incoming_starts[1:] = incoming[1:] != incoming[:-1]
        assert torch.equal(starts, incoming_starts), (
            "the two ranges break into pages at different offsets, so no "
            "page-granular ownership transfer expresses the token mapping"
        )
        return kept[starts], incoming[starts]

    def _transfer_swa_pages(
        self,
        allocator,
        full_value: torch.Tensor,
        incoming_full_value: torch.Tensor,
    ) -> None:
        """Move swa page OWNERSHIP from the incoming ids onto the node's ids.

        The static recipe re-points the node's locked full ids at the incoming
        swa pages through `full_to_swa_index_mapping`. Under the unified pool
        the swa sub-pool's v2p IS that mapping, so the same move is a rebind:
        give the node's virtual pages the incoming pages' physical pages, then
        tombstone the incoming ones. No page is allocated or freed, so no
        capacity changes — only ownership does.
        """
        swa = allocator.swa_attn_allocator
        kept_pages, incoming_pages = self._page_pairs(full_value, incoming_full_value)
        physical = swa.virtual_to_physical[incoming_pages]
        # `> 0` strict: -1 = tombstoned, 0 = the padding sink. The incoming ids
        # were just allocated by the in-flight request, so every page must be
        # live; a violation means we would hand the node the sink and serve
        # zeros, which is worth a hard failure rather than silent corruption.
        assert bool(
            (physical > 0).all()
        ), f"incoming swa pages must all be live, got {physical.tolist()}"
        swa.bind(kept_pages, physical)
        swa.virtual_to_physical.index_fill_(0, incoming_pages, -1)
        swa.clear_inverse_history()

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
        while node is not root and n_swa < self.sliding_window_size:
            cd = node.component_data[ct]
            if cd.value is not None:
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
        result: InsertResult,
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
            result.record_adopted_range(
                self.component_type,
                total_prefix_len,
                total_prefix_len + prefix_len,
            )
            old_full = full_cd.value
            if full_cd.lock_ref > 0:
                cache_actions.append(
                    RecoverSWAWithLockedFull(node.id, old_full, value_slice)
                )
                return 0
            result.record_adopted_range(
                BASE_COMPONENT_TYPE,
                total_prefix_len,
                total_prefix_len + prefix_len,
            )
            full_cd.value = value_slice.clone()
            cache_actions.append(FreeDeviceKVFullOnly([old_full]))
            cache_actions.append(SWARebuild(node.id, value_slice))
            return 0
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # Branch 2: value_slice[start_idx:] is within SWA window — partial recover
            start_idx = swa_evicted_seqlen - total_prefix_len
            result.record_adopted_range(
                self.component_type,
                swa_evicted_seqlen,
                total_prefix_len + prefix_len,
            )
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
            result.record_adopted_range(
                BASE_COMPONENT_TYPE,
                swa_evicted_seqlen,
                total_prefix_len + prefix_len,
            )
            node.component_data[BASE_COMPONENT_TYPE].value = new_full.clone()
            cache_actions.append(FreeDeviceKVFullOnly([old_full]))
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
        result: InsertResult,
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
        result.record_adopted_range(
            self.component_type,
            max(total_prefix_len, swa_evicted_seqlen),
            total_prefix_len + prefix_len,
        )
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
        node_end = node_start + len(node.key)
        split_pos = params.swa_evicted_seqlen - node_start
        if split_pos >= len(node.key):
            # Entire leaf is outside the SWA window — left as a tombstone.
            return
        result.record_adopted_range(
            self.component_type,
            max(node_start, params.swa_evicted_seqlen),
            node_end,
        )
        if split_pos > 0:
            # Node straddles the boundary: split into an out-of-window parent
            # (tombstone) and an in-window child; `node` becomes the child.
            _, action = self.tree_core._split_node(node.key, node, split_pos)
            assert action is None, "new leaf cannot be write-through-pending"
        # Cap the in-window leaf at one window for lock granularity, then rebuild SWA
        # onto the in-window node(s) at apply time; rebuild the older prefix first so
        # the in-window tail lands more-MRU.
        capped_parent = self._maybe_split_leaf_for_swa_lock(node)
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
            new_parent.component_data[self.component_type].host_value = (
                child_swa_host_value[:split_len].clone()
            )
            child.component_data[self.component_type].host_value = child_swa_host_value[
                split_len:
            ].clone()
            host_lru = self.tree_core.host_lru_lists[self.component_type]
            if new_parent.component_data[self.component_type].value is None:
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

        # Host layer
        host_lru = self.tree_core.host_lru_lists[ct]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            host_frees[ct].append(cd.host_value)
            cd.host_value = None
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
        """Advance one device-eviction step and return a leaf, if selected.

        An internal tombstone is one complete step so the caller can apply its
        pending frees and recheck allocator capacity before the next mutation.
        If the previous node's eviction removed the cursor, the walk resumes
        from the partition sentinel with session refs on, else it restarts at
        the LRU tail.
        """
        ct = self.component_type
        lru = self.tree_core.lru_lists[ct]
        enabled = self.tree_core.enable_session_radix_cache
        if self._evict_device_cursor is not None and not lru.in_list(
            self._evict_device_cursor
        ):
            self._evict_device_cursor = (
                lru.cursor_next() if enabled else lru.get_lru_no_lock()
            )
        if (
            tracker[ct] >= self._evict_device_request_cnt
            or self._evict_device_cursor is None
            or not lru.in_list(self._evict_device_cursor)
        ):
            return None

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
        sw_pages = self.full_window_pages
        if sw_pages == 0:
            return PreparePrefetchResult()
        prefetch_pages = prefetch_tokens // self.cache.page_size
        if prefetch_pages >= sw_pages:
            num_pages = sw_pages
        elif prefetch_pages <= 0:
            return PreparePrefetchResult()
        elif (
            self.tree_core.is_root(node_id)
            or self.cache.host_memory_mode == "buffer_only"
        ):
            # Sub-window fetch: at root the sequence IS its window; mid-tree
            # (buffer mode) the window head is the device prefix's own ring
            # state, so only the suffix needs fetching.
            num_pages = prefetch_pages
        else:
            # Cache-mode graft: a mid-tree window head is not
            # device-guaranteed, require a full window.
            return PreparePrefetchResult()
        num_tokens = num_pages * self.cache.page_size
        host_indices = self.cache.host_pool_group.alloc(
            num_tokens,
            pool=PoolName.SWA,
            reclaim=lambda size: self.cache.evict_host(size, ComponentType.SWA),
        )
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
            if cd.value is None:
                return None
            # cd.value already holds SWA-pool indices (translated at insert time).
            # Host pool indexing wants int64.
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
                if cd.value is not None:
                    # device exists, skip it
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
            # Keys are unknowable at build time; placeholders carry the
            # count, _sync_trailing_keys fills the real trailing hashes.
            num_pages = host_indices.numel() // self.tree_core.page_size
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=host_indices,
                    keys=["__placeholder__"] * num_pages,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        return None

    def build_external_linker_transfer(
        self,
        phase: LinkerTransferPhase,
        node: Optional[UnifiedTreeNode],
        keys: Optional[Sequence[str]],
    ) -> Optional[PoolTransfer]:
        page = self.cache.page_size
        window_pages = (self.sliding_window_size + page - 1) // page

        if phase == LinkerTransferPhase.OFFLOAD:
            if node is None or not node.hash_value:
                return None
            value = node.component_data[self.component_type].value
            if value is None or len(value) < page:
                return None

            num_pages = len(value) // page
            return PoolTransfer(
                name=PoolName.SWA,
                device_indices=value[-num_pages * page :].to(torch.int64),
                keys=node.hash_value[-num_pages:],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )

        if not keys:
            return None

        # `keys` already start at the first device-uncached page, so the trailing
        # window is simply their tail.
        tail_keys = list(keys[max(0, len(keys) - window_pages) :])
        if not tail_keys:
            return None

        transfer = PoolTransfer(
            name=PoolName.SWA,
            keys=tail_keys,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        if phase == LinkerTransferPhase.LOAD:
            num_tokens = len(tail_keys) * page
            allocator = self.cache.token_to_kv_pool_allocator.swa_attn_allocator
            shortfall = max(0, num_tokens - allocator.available_size())
            if shortfall:
                self.cache.evict(EvictParams(swa_num_tokens=shortfall))
            transfer.device_indices = allocator.alloc(num_tokens)
            if transfer.device_indices is None:
                return None
            transfer.device_indices = transfer.device_indices.to(torch.int64)
        return transfer

    def update_external_linker_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        full_transfer: PoolTransfer,
        transfer: PoolTransfer,
        prefix_len: int,
        *,
        insert_result: Optional[InsertResult] = None,
        canonical_full: Optional[torch.Tensor] = None,
    ) -> Optional[PoolTransfer]:
        if phase == ExternalLinkerLoadPhase.ABORT:
            self.cache.token_to_kv_pool_allocator.swa_attn_allocator.free(
                transfer.device_indices
            )
            return None

        allocator = self.cache.token_to_kv_pool_allocator
        if phase == ExternalLinkerLoadPhase.PREPARE:
            swa_len = len(transfer.device_indices)
            allocator.set_full_to_swa_mapping(
                full_transfer.device_indices[-swa_len:], transfer.device_indices
            )
            page = self.cache.page_size
            window = ((self.sliding_window_size + page - 1) // page) * page
            boundary = max(0, prefix_len - window)
            if req.kv is None:
                from sglang.srt.managers.schedule_batch import ReqKvInfo

                req.kv = ReqKvInfo(
                    kv_allocated_len=prefix_len,
                    swa_evicted_seqlen=boundary,
                )
            else:
                req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, boundary)
            return transfer

        assert phase == ExternalLinkerLoadPhase.COMMIT
        assert insert_result is not None and canonical_full is not None
        assert len(canonical_full) == len(transfer.device_indices)
        allocator.set_full_to_swa_mapping(canonical_full, transfer.device_indices)
        return transfer

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
                    cd.host_value = transfers[0].host_indices.clone()
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
                cd_full_n = n.component_data[BASE_COMPONENT_TYPE]
                n_tokens = len(cd_n.host_value)
                swa_chunk = device_indices[offset : offset + n_tokens].clone()
                self.tree_core.set_component_device_value(
                    n.id, self.component_type, swa_chunk
                )
                assert cd_full_n.value is not None and len(cd_full_n.value) == n_tokens
                full_chunks.append(cd_full_n.value)
                swa_chunks.append(swa_chunk)
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
        window_require_pages = (
            host_indices.numel() // page_size if host_indices is not None else 0
        )
        loaded_pages = (
            pool_storage_result.extra_pool_hit_pages.get(PoolName.SWA, 0)
            if pool_storage_result
            else 0
        )
        target = (
            self.tree_core.node_by_id(insert_result.inserted_host_node)
            if insert_result is not None
            and insert_result.inserted_host_node is not None
            else None
        )
        if anchor is not self.tree_core.root_node:
            # Cache-mode graft commit only (buffer fills never reach here):
            # a hit-shrunk window mid-tree is missing its head — drop it.
            # Root anchors are complete windows of their own.
            if window_require_pages < self.full_window_pages:
                self._release_swa_host(host_indices, cache_actions)
                return
        if (
            target is None
            or window_require_pages == 0
            or loaded_pages < window_require_pages
        ):
            self._release_swa_host(host_indices, cache_actions)
            return

        # Buffer covers token range [loaded_start, total_len).
        loaded_start = insert_result.total_len - window_require_pages * page_size

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
            self.cache.host_pool_group.free(host_value, pool=PoolName.SWA)

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
            # Keep the locked full; hand the node the INCOMING ids' swa pages,
            # freeing only the incoming full, then store the swa on the node.
            unified = self._unified_allocator()
            if unified is not None:
                # No `full_to_swa_index_mapping` here: the swa sub-pool's v2p IS
                # the mapping. Rebind page ownership, then free through the
                # composite -- its `swa_v2p_pages > 0` filter skips the
                # just-tombstoned swa side, releasing only the full one.
                self._transfer_swa_pages(
                    unified, action.kept_full, action.incoming_full
                )
                unified.free(action.incoming_full)
                self.tree_core.set_component_device_value(
                    action.node_id,
                    self.component_type,
                    self._translate_full_to_swa(action.kept_full),
                )
                return
            swa_value = self._translate_full_to_swa(action.incoming_full)
            alloc.set_full_to_swa_mapping(action.kept_full, swa_value)
            alloc.clear_full_to_swa_mapping(action.incoming_full)
            alloc.free_full(action.incoming_full)
            self.tree_core.set_component_device_value(
                action.node_id, self.component_type, swa_value
            )
            return
        if isinstance(action, SWARebuild):
            # Translate the node's source full value to SWA and store it on the node.
            swa_value = self._translate_full_to_swa(action.source_value)
            self.tree_core.set_component_device_value(
                action.node_id, self.component_type, swa_value
            )
            return
        raise AssertionError(
            f"SWAComponent: unhandled ComponentAction {type(action).__name__}"
        )
