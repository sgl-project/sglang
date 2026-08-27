"""C128 sidecar ownership for the DSV4 NPU Unified Radix Cache.

The component deliberately exposes only complete physical C128 pages to the
radix tree; partial tail pages remain request-owned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    PrepareLoadBackResult,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedTreeNode,
    )

class C128SidecarComponent(TreeComponent):
    component_type = ComponentType.C128

    # Bound by _apply_stack_result (hybrid_pool_assembler) on the NPU path:
    # C128 is an independent-index pool whose host values live in this pool.
    _c128_kv_pool_host = None

    @property
    def allocator(self):
        return self.cache.token_to_kv_pool_allocator

    def _adjust_session_path(
        self,
        leaf: UnifiedTreeNode,
        stop: UnifiedTreeNode,
        delta: int,
    ) -> None:
        """Adjust session protection for every C128 boundary on a radix path."""
        node = leaf
        while node is not stop and node is not self.tree_core.root_node:
            cd = node.component_data[self.component_type]
            if delta < 0:
                assert cd.session_ref > 0
            prev_ref = cd.session_ref
            cd.session_ref += delta
            if (prev_ref == 0) != (cd.session_ref == 0):
                self._refresh_session_partition(node)
            node = node.parent

    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        self._adjust_session_path(leaf, self.tree_core.root_node, -1)

    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:
        stop = old_ancestor or self.tree_core.root_node
        self._adjust_session_path(leaf, stop, 1)

    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:
        stop = fallback or self.tree_core.root_node
        self._adjust_session_path(leaf, stop, -1)

    def _attach(self, node: UnifiedTreeNode, pages: torch.Tensor) -> None:
        if pages.numel() == 0:
            return
        ct = self.component_type
        cd = node.component_data[ct]
        assert cd.value is None
        value = pages.clone()
        self.tree_core.set_component_device_value(node.id, ct, value)
        self.allocator.retain_c128_pages(value)

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        # Pages attach only to full-group endpoints. A host-backed endpoint is a
        # valid match unless match_device_only requires device residency.
        def _valid(node: UnifiedTreeNode) -> bool:
            cd = node.component_data[self.component_type]
            if match_device_only:
                return cd.value is not None
            return cd.value is not None or cd.host_value is not None

        return _valid

    def _collect_device_pages(self, node_id: int) -> torch.Tensor:
        chunks = []
        node = self.tree_core.node_by_id(node_id)
        root = self.tree_core.root_node
        while node is not root:
            value = node.component_data[self.component_type].value
            if value is not None:
                chunks.append(value)
            node = node.parent
        chunks.reverse()
        return (
            torch.cat(chunks)
            if chunks
            else self.allocator.c128_attn_allocator.free_pages.new_empty((0,))
        )

    def finalize_match_result_in_cache(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:
        req = params.req
        if req is None:
            return result

        pages = self._collect_device_pages(result.best_match_node)
        self.cache.req_to_token_pool.set_c128_prefix_pages(req, pages)
        return result

    def finalize_load_back(
        self, req: Optional[Req], prep: PrepareLoadBackResult, success: bool
    ) -> None:
        if not success or req is None:
            return

        # match_prefix runs before load-back, so a host-only C128 endpoint is
        # absent from the request-local page table built by the match finalizer.
        # Refresh it after commit_load_back attaches the restored page to the tree.
        pages = self._collect_device_pages(req.best_match_node)
        self.cache.req_to_token_pool.set_c128_prefix_pages(req, pages)

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        pages = params.c128_value
        assert pages is not None
        group_tokens = 128 * self.allocator.c128_attn_allocator.page_size
        start = total_prefix_len // group_tokens
        end = (total_prefix_len + prefix_len) // group_tokens
        self._attach(node, pages[start:end])

    @staticmethod
    def _node_depth(node: UnifiedTreeNode) -> int:
        depth = 0
        while node.parent is not None:
            depth += len(node.key)
            node = node.parent
        return depth

    @staticmethod
    def _split_pending_swa_rebuild(
        new_parent: UnifiedTreeNode,
        child: UnifiedTreeNode,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """Keep a deferred SWA rebuild aligned when C128 splits its source node."""
        for i, pending in enumerate(cache_actions):
            if isinstance(pending, SWARebuild) and pending.node_id == child.id:
                cache_actions[i : i + 1] = [
                    SWARebuild(
                        new_parent.id,
                        new_parent.component_data[BASE_COMPONENT_TYPE].value,
                    ),
                    SWARebuild(
                        child.id,
                        child.component_data[BASE_COMPONENT_TYPE].value,
                    ),
                ]
                return

    def _ensure_boundary_node(
        self,
        tail: UnifiedTreeNode,
        boundary: int,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> UnifiedTreeNode:
        node = tail
        while node.parent is not None and self._node_depth(node.parent) >= boundary:
            node = node.parent

        node_end = self._node_depth(node)
        if node_end == boundary:
            return node

        node_start = node_end - len(node.key)
        assert node_start < boundary < node_end
        new_parent, action = self.tree_core._split_node(
            node.key, node, boundary - node_start
        )
        if action is not None:
            cache_actions.append(action)
        self._split_pending_swa_rebuild(new_parent, node, cache_actions)
        return new_parent

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
        assert params.key is not None
        assert params.c128_value is not None

        # Full/SWA may initially represent the new suffix as one long leaf.
        # Materialize every complete C128 group boundary so a later branch can
        # always match the nearest full C128-page prefix instead of falling back to
        # the previous, potentially much shorter, Radix node.
        group_tokens = 128 * self.allocator.c128_attn_allocator.page_size
        first_boundary = (result.prefix_len // group_tokens + 1) * group_tokens
        for boundary in range(first_boundary, len(params.key) + 1, group_tokens):
            boundary_node = self._ensure_boundary_node(node, boundary, cache_actions)
            page_index = boundary // group_tokens - 1
            self._attach(boundary_node, params.c128_value[page_index : page_index + 1])

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ) -> None:
        # Every stored value belongs to the old child's end boundary. Splitting
        # inside that group leaves the page on the child.
        ct = self.component_type
        new_parent.component_data[ct].session_ref = child.component_data[ct].session_ref
        assert new_parent.component_data[ct].session_ids is None

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        if EvictLayer.DEVICE in target and cd.value is not None:
            # Device pages use retain/release_c128_pages refcounts.
            # _drain_device_frees converts these IDs into release actions.
            device_frees[self.component_type].append(cd.value)
            self.tree_core.component_evictable_size_[self.component_type] -= len(
                cd.value
            )
            cd.value = None
            # A device tombstone with a host copy makes the node host-only.
            # Promote it so every host-only node remains in host_lru.
            if cd.host_value is not None:
                host_lru = self.tree_core.host_lru_lists[self.component_type]
                if not host_lru.in_list(node):
                    host_lru.insert_mru(node)
        if EvictLayer.HOST in target and cd.host_value is not None:
            # Host values have no refcount; free_host_values returns them directly.
            host_frees[self.component_type].append(cd.host_value)
            cd.host_value = None
        # C128 pages are auxiliary to Full tokens and must not inflate the public
        # token-eviction count; the host return also stays 0 so a FULL host-leaf
        # eviction's tracker only counts FULL tokens (C128 is a required payload).
        return 0, 0

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> int:
        logical_len = token_ids_len
        if self.tree_core.is_eagle and logical_len > 0:
            logical_len -= 1
        group_tokens = 128 * self.allocator.c128_attn_allocator.page_size
        cache_len = logical_len // group_tokens * group_tokens
        num_pages = cache_len // group_tokens
        insert_params.c128_value = self.cache.req_to_token_pool.req_to_c128_sidecar[
            int(req.kv.req_pool_idx), :num_pages
        ].clone()
        return cache_len + 1 if self.tree_core.is_eagle and cache_len > 0 else cache_len

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, FreeComponentDeviceSlot):
            for page_ids in action.indices:
                self.allocator.release_c128_pages(page_ids)
            return
        raise AssertionError(
            f"C128SidecarComponent: unhandled action {type(action).__name__}"
        )

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def _evict_device_start(self, request_cnt):
        pass

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self) -> None:
        pass

    def acquire_component_lock(self, node, result, lock_host=False):
        # Device path is a no-op: C128 device pages are owned via refcount, not
        # the FULL path-lock. Host path mirrors FULL's single-node host lock.
        if lock_host:
            cd = node.component_data[self.component_type]
            # write_back mode: the anchor may be device-only (no host_value);
            # pin it anyway.
            if cd.host_value is None and not self.tree_core.is_write_back:
                return result
            cd.host_lock_ref += 1
            self.tree_core._update_evictable_leaf_sets(node)
        return result

    def release_component_lock(self, node, params, lock_host=False) -> None:
        if lock_host:
            cd = node.component_data[self.component_type]
            if cd.host_lock_ref == 0:
                return
            # Mirror of `acquire`. write_back uses a pure counter.
            if cd.host_value is None and not self.tree_core.is_write_back:
                return
            cd.host_lock_ref -= 1
            self.tree_core._update_evictable_leaf_sets(node)

    def free_host_values(self, host_values) -> None:
        if self._c128_kv_pool_host is None:
            return
        for host_value in host_values:
            self._c128_kv_pool_host.free(host_value)

    # ---- HiCache Hooks ----

    @staticmethod
    def _expand_page_indices(page_ids: torch.Tensor, page_size: int) -> torch.Tensor:
        """Expand each page ID to ``page_id * page_size + arange(page_size)``."""
        page_ids = page_ids.view(-1)
        if page_ids.numel() == 0:
            return page_ids.new_empty((0,), dtype=torch.int64)
        return (
            page_ids[:, None] * page_size
            + torch.arange(page_size, device=page_ids.device)
        ).flatten()

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
        page_size = self.allocator.c128_attn_allocator.page_size

        if phase == CacheTransferPhase.BACKUP_HOST:
            # Back up the C128 pages attached to this node (its group endpoints).
            # The transfer is independent (indices_from_pool=None): the controller
            # allocates C128 host slots of len(device_indices) = groups * P.
            page_ids = node.component_data[ct].value
            if page_ids is None or page_ids.numel() == 0:
                return None
            return [
                PoolTransfer(
                    name=PoolName.DEEPSEEK_V4_C128,
                    indices_from_pool=None,
                    device_indices=self._expand_page_indices(page_ids, page_size),
                    nodes_to_load=[node.id],
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            # Collect host values from complete-group endpoints on the evicted path.
            # For example, G groups produce G * page_size host and device indices.
            backed_up: list[torch.Tensor] = []
            nodes: list[UnifiedTreeNode] = []
            cur = node
            while cur is not self.tree_core.root_node and cur.evicted:
                cd = cur.component_data[ct]
                if cd.host_value is not None:
                    backed_up.append(cd.host_value)
                    nodes.append(cur)
                cur = cur.parent
            if not backed_up:
                return None
            backed_up.reverse()
            nodes.reverse()
            return [
                PoolTransfer(
                    name=PoolName.DEEPSEEK_V4_C128,
                    indices_from_pool=None,
                    host_indices=torch.cat(backed_up),
                    device_indices=None,
                    nodes_to_load=[n.id for n in nodes],
                )
            ]

        # BACKUP_STORAGE / PREFETCH are deferred to Plan-3.
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
        page_size = self.allocator.c128_attn_allocator.page_size

        if phase == CacheTransferPhase.BACKUP_HOST:
            # Publish the controller-allocated host slots as this node's C128
            # host residency (L2). The device value stays until demote/evict.
            if transfers and transfers[0].host_indices is not None:
                node.component_data[ct].host_value = transfers[0].host_indices.clone()
                # An evict-then-backup race can make this node host-only.
                # Insert it now so every host-only node remains in host_lru.
                if node.component_data[ct].value is None:
                    host_lru = self.tree_core.host_lru_lists[ct]
                    if not host_lru.in_list(node):
                        host_lru.insert_mru(node)
            return

        if phase == CacheTransferPhase.LOAD_BACK:
            if not transfers or transfers[0].device_indices is None:
                return
            xfer = transfers[0]
            device_indices = xfer.device_indices
            offset = 0
            for nid in xfer.nodes_to_load or []:
                n = self.tree_core.node_by_id(nid)
                cd = n.component_data[ct]
                n_len = len(cd.host_value)
                # Each page occupies P consecutive expanded slots; ``// P`` yields
                # P copies of the page id, so unique() recovers the distinct page
                # ids (the allocator's retain_c128_pages does NOT dedup).
                page_ids = torch.unique(
                    device_indices[offset : offset + n_len] // page_size
                )
                # Once retained, tree eviction releases these pages.
                # The controller frees them only on rollback before commit.
                self.allocator.retain_c128_pages(page_ids)
                self.tree_core.set_component_device_value(nid, ct, page_ids.clone())
                offset += n_len
            return
