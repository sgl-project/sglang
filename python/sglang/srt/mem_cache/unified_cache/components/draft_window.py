"""DFLASH draft KV kept as the windows of the Mamba checkpoints that use it.

A draft resuming from a checkpoint at ``p`` needs its KV for ``[p - W', p)``
(``W'``: the window rounded up to a page). The draft pool is the SWA side of the
target allocator. ``cover_ref`` / ``host_cover_ref`` count the checkpoints on
each tier whose window contains a node. Draft KV no window covers is freed, and
evicting a checkpoint releases its window.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams, InsertParams
from sglang.srt.mem_cache.unified_cache.cache_action import (
    RecoverSWAWithLockedFull,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    InternalStateBackup,
)
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import InsertResult
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hicache_storage import (
        PoolTransfer,
        PoolTransferResult,
    )
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class _Tier(NamedTuple):
    """Where a tier keeps a node's value, lock and window count."""

    layer: EvictLayer
    value: str
    lock_ref: str
    cover_ref: str


_DEVICE = _Tier(EvictLayer.DEVICE, "value", "lock_ref", "cover_ref")
_HOST = _Tier(EvictLayer.HOST, "host_value", "host_lock_ref", "host_cover_ref")

# Fruitless host checkpoint evictions in a row before a host draft allocation gives up.
_MAX_FRUITLESS_HOST_BUNDLES = 8


class DraftWindowComponent(SWAComponent):
    """Draft KV kept as the windows of Mamba checkpoints on each tier."""

    device_eviction_proxy = ComponentType.MAMBA

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        super().__init__(cache, params)
        self.window_tokens = self.full_window_pages * params.page_size
        # Per tier, unlocked nodes with uncovered draft KV, oldest first.
        self._uncovered: dict[EvictLayer, dict[NodeId, UnifiedTreeNode]] = {
            EvictLayer.DEVICE: {},
            EvictLayer.HOST: {},
        }
        self._uncovered_root: Optional[UnifiedTreeNode] = None

    @staticmethod
    def cover_ref(node: UnifiedTreeNode, tier: _Tier = _DEVICE) -> int:
        return node.component_data[ComponentType.SWA].metadata.get(tier.cover_ref, 0)

    @staticmethod
    def host_cover_ref(node: UnifiedTreeNode) -> int:
        return DraftWindowComponent.cover_ref(node, _HOST)

    def _window(self, checkpoint: UnifiedTreeNode) -> list[UnifiedTreeNode]:
        """Nodes of the window ending at ``checkpoint``, deepest first."""
        nodes = []
        covered = 0
        node = checkpoint
        root = self.tree_core.root_node
        while node is not root and covered < self.window_tokens:
            nodes.append(node)
            covered += len(node.key)
            node = node.parent
        assert covered <= self.window_tokens or node is root, (
            f"draft window of node {checkpoint.id} does not start on a node "
            f"boundary ({covered} > {self.window_tokens} tokens)"
        )
        return nodes

    def acquire_window(
        self, checkpoint: UnifiedTreeNode, tier: _Tier = _DEVICE
    ) -> None:
        """Count a new checkpoint's window on every node it covers."""
        uncovered = self._uncovered_queue(tier)
        for node in self._window(checkpoint):
            metadata = node.component_data[self.component_type].metadata
            metadata[tier.cover_ref] = metadata.get(tier.cover_ref, 0) + 1
            uncovered.pop(node.id, None)

    def release_window(
        self,
        checkpoint: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        tier: _Tier = _DEVICE,
    ) -> None:
        """Drop a checkpoint's window, freeing draft KV no other window covers."""
        for node in self._window(checkpoint):
            cd = node.component_data[self.component_type]
            ref = cd.metadata.get(tier.cover_ref, 0)
            assert ref > 0, (
                f"draft window release hit {tier.cover_ref}=0 on node {node.id}"
            )
            cd.metadata[tier.cover_ref] = ref - 1
            if (
                ref == 1
                and getattr(cd, tier.value) is not None
                and getattr(cd, tier.lock_ref) == 0
            ):
                self._free(node, tier, device_frees, host_frees)

    def _free(
        self,
        node: UnifiedTreeNode,
        tier: _Tier,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        tracker: Optional[dict[ComponentType, int]] = None,
    ) -> None:
        self.tree_core._evict_component_and_detach_lru(
            node,
            self,
            device_frees,
            host_frees,
            target=tier.layer,
            tracker=tracker,
        )
        self.tree_core._update_evictable_leaf_sets(node)

    def _uncovered_queue(self, tier: _Tier) -> dict[NodeId, UnifiedTreeNode]:
        root = self.tree_core.root_node
        # A cache reset replaces the root, so drop the stale queues.
        if self._uncovered_root is not root:
            for uncovered in self._uncovered.values():
                uncovered.clear()
            self._uncovered_root = root
        return self._uncovered[tier.layer]

    def _queue_if_uncovered(self, node: UnifiedTreeNode, tier: _Tier) -> None:
        cd = node.component_data[self.component_type]
        if (
            getattr(cd, tier.value) is not None
            and getattr(cd, tier.lock_ref) == 0
            and cd.metadata.get(tier.cover_ref, 0) == 0
        ):
            self._uncovered_queue(tier)[node.id] = node

    def _pop_reclaimable(self, tier: _Tier) -> Optional[UnifiedTreeNode]:
        uncovered = self._uncovered_queue(tier)
        while uncovered:
            node_id = next(iter(uncovered))
            node = uncovered.pop(node_id)
            cd = node.component_data[self.component_type]
            if (
                self.tree_core._node_arena.get(node_id) is node
                and getattr(cd, tier.value) is not None
                and getattr(cd, tier.lock_ref) == 0
                and cd.metadata.get(tier.cover_ref, 0) == 0
            ):
                return node
        return None

    def _collect_unbacked_swa_nodes(
        self, node: UnifiedTreeNode
    ) -> list[UnifiedTreeNode]:
        # Only a checkpoint's window is worth a host copy.
        mamba = node.component_data[ComponentType.MAMBA]
        if mamba.value is None and mamba.host_value is None:
            return []
        return super()._collect_unbacked_swa_nodes(node)

    def _split_at_window_start(
        self,
        *,
        node: UnifiedTreeNode,
        start: int,
        end: int,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """Split ``[start, end)`` where the inserted checkpoint's window starts."""
        window_start = len(params.key) - self.window_tokens
        if not start < window_start < end:
            return
        for fragment, fragment_start, fragment_end in self._segments(
            node=node, start=start, end=end
        ):
            if fragment_start < window_start < fragment_end:
                _, action = self.tree_core._split_node(
                    fragment.key, fragment, window_start - fragment_start
                )
                if action is not None:
                    cache_actions.append(action)
                return

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
        self._split_at_window_start(
            node=node,
            start=total_prefix_len,
            end=total_prefix_len + prefix_len,
            params=params,
            cache_actions=cache_actions,
        )
        return super().update_component_on_insert_overlap(
            node,
            prefix_len,
            total_prefix_len,
            value_slice,
            params,
            result,
            cache_actions,
        )

    def _populate_range(
        self, *, node, start, end, params, result, cap_leaf, cache_actions
    ):
        # The window-start split already caps the leaf at one window.
        self._split_at_window_start(
            node=node, start=start, end=end, params=params, cache_actions=cache_actions
        )
        super()._populate_range(
            node=node,
            start=start,
            end=end,
            params=params,
            result=result,
            cap_leaf=False,
            cache_actions=cache_actions,
        )

    def apply_component_action(self, action: ComponentAction) -> None:
        super().apply_component_action(action)
        # Draft KV stamped outside every window is reclaimable once unlocked.
        if isinstance(action, (SWARebuild, RecoverSWAWithLockedFull)):
            self._queue_if_uncovered(self.tree_core.node_by_id(action.node_id), _DEVICE)

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
        super().commit_hicache_transfer(
            node,
            phase,
            transfers,
            cache_actions=cache_actions,
            insert_result=insert_result,
            pool_storage_result=pool_storage_result,
        )
        # A checkpoint committed by the same transfer dequeues its window after this.
        tier = {
            CacheTransferPhase.BACKUP_HOST: _HOST,
            CacheTransferPhase.LOAD_BACK: _DEVICE,
        }.get(phase)
        if tier is None or not transfers:
            return
        for node_id in transfers[0].nodes_to_load or [node.id]:
            self._queue_if_uncovered(self.tree_core.node_by_id(node_id), tier)

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        super().redistribute_on_node_split(new_parent, child)
        parent_metadata = new_parent.component_data[self.component_type].metadata
        for tier in (_DEVICE, _HOST):
            ref = self.cover_ref(child, tier)
            if ref:
                parent_metadata[tier.cover_ref] = ref
            if child.id in self._uncovered_queue(tier):
                self._queue_if_uncovered(new_parent, tier)

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams,
        lock_host: bool = False,
    ) -> None:
        super().release_component_lock(node, params, lock_host)
        tier = _HOST if lock_host else _DEVICE
        # The segment the lock covered, as acquire_component_lock measured it.
        covered = 0
        cur = node
        root = self.tree_core.root_node
        while cur is not root and covered < self.sliding_window_size:
            self._queue_if_uncovered(cur, tier)
            covered += len(cur.key)
            cur = cur.parent

    def release_window_lock(
        self,
        node: UnifiedTreeNode,
        swa_uuid_for_lock: Optional[int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Unlike plain SWA, a covered leaf keeps its draft KV."""
        ct = self.component_type
        root = self.tree_core.root_node
        cur = node
        while cur is not root:
            cd = cur.component_data[ct]
            assert cd.lock_ref > 0, (
                f"draft window release hit lock_ref=0 on node {cur.id}"
            )
            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                self.tree_core._update_evictable_leaf_sets(cur)
                if cd.value is not None:
                    value_len = len(cd.value)
                    self.tree_core.component_protected_size_[ct] -= value_len
                    self.tree_core.component_evictable_size_[ct] += value_len
                    if self.cover_ref(cur) == 0:
                        self._free(cur, _DEVICE, device_frees, host_frees)
            if swa_uuid_for_lock and cd.metadata.get("uuid") == swa_uuid_for_lock:
                break
            cur = cur.parent

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        for layer, uncovered in self._uncovered.items():
            if layer in target:
                uncovered.pop(node.id, None)
        return super().evict_component(node, device_frees, host_frees, target)

    def _evict_device_start(self, request_cnt: int) -> None:
        self._evict_device_request_cnt = request_cnt

    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> NodeId | InternalStateBackup | None:
        """Reclaim one uncovered node. Covered windows go with their checkpoints."""
        if tracker[self.component_type] < self._evict_device_request_cnt:
            node = self._pop_reclaimable(_DEVICE)
            if node is not None:
                self._free(node, _DEVICE, device_frees, host_frees, tracker)
        return None

    def _evict_device_end(self) -> None:
        pass

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Free uncovered host draft KV, then whole host bundles in Mamba LRU order."""
        ct = self.component_type
        while tracker[ct] < num_tokens:
            node = self._pop_reclaimable(_HOST)
            if node is None:
                break
            self._free(node, _HOST, device_frees, host_frees, tracker)
        mamba = self.tree_core.components_by_type[ComponentType.MAMBA]
        fruitless = 0
        while tracker[ct] < num_tokens and fruitless < _MAX_FRUITLESS_HOST_BUNDLES:
            freed_before = len(host_frees[ct])
            bundle_tracker: dict[ComponentType, int] = defaultdict(int)
            mamba.drive_host_eviction(1, bundle_tracker, device_frees, host_frees)
            if not any(bundle_tracker.values()):
                break
            for component, count in bundle_tracker.items():
                if component != ct:
                    tracker[component] += count
            freed = sum(len(v) for v in host_frees[ct][freed_before:])
            tracker[ct] += freed
            # A window shared with other checkpoints frees nothing.
            fruitless = 0 if freed else fruitless + 1

    def validate_component_state(
        self,
        reachable_nodes: set[UnifiedTreeNode],
        report_error: Callable[[str], None],
    ) -> None:
        ct = self.component_type
        root = self.tree_core.root_node
        nodes = [node for node in reachable_nodes if node is not root]
        for tier in (_DEVICE, _HOST):
            expected: dict[NodeId, int] = defaultdict(int)
            for node in nodes:
                if (
                    getattr(node.component_data[ComponentType.MAMBA], tier.value)
                    is None
                ):
                    continue
                for covered in self._window(node):
                    expected[covered.id] += 1
            uncovered = self._uncovered_queue(tier)
            for node in nodes:
                ref = self.cover_ref(node, tier)
                if ref != expected.get(node.id, 0):
                    report_error(
                        f"{ct} node {node.id} {tier.cover_ref}={ref}, "
                        f"expected {expected.get(node.id, 0)}"
                    )
                cd = node.component_data[ct]
                if (
                    getattr(cd, tier.value) is not None
                    and getattr(cd, tier.lock_ref) == 0
                    and ref == 0
                    and uncovered.get(node.id) is not node
                ):
                    report_error(
                        f"{ct} node {node.id} holds uncovered {tier.layer.name} "
                        "draft KV unqueued"
                    )


class DraftWindowMambaComponent(MambaComponent):
    """Mamba component whose checkpoints own draft windows on each tier."""

    def _draft_window(self) -> DraftWindowComponent:
        return self.tree_core.components_by_type[ComponentType.SWA]

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        cd = node.component_data[self.component_type]
        had_checkpoint = cd.value is not None
        super().commit_insert_component_data(
            node, is_new_leaf, params, result, cache_actions
        )
        if not had_checkpoint and cd.value is not None:
            self._draft_window().acquire_window(node, _DEVICE)

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
        # Storage is gated off (dflash_draft_kv_bundling_blocker).
        assert phase != CacheTransferPhase.PREFETCH, (
            "draft window bundles do not support storage prefetch"
        )
        cd = node.component_data[self.component_type]
        held = {tier: getattr(cd, tier.value) is not None for tier in (_DEVICE, _HOST)}
        super().commit_hicache_transfer(
            node,
            phase,
            transfers,
            cache_actions=cache_actions,
            insert_result=insert_result,
            pool_storage_result=pool_storage_result,
        )
        for tier, had_checkpoint in held.items():
            if not had_checkpoint and getattr(cd, tier.value) is not None:
                self._draft_window().acquire_window(node, tier)

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        releasing = [
            tier
            for tier in (_DEVICE, _HOST)
            if tier.layer in target and getattr(cd, tier.value) is not None
        ]
        freed = super().evict_component(node, device_frees, host_frees, target)
        for tier in releasing:
            self._draft_window().release_window(node, device_frees, host_frees, tier)
        return freed
