from __future__ import annotations

import heapq
import time
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import TreeNode

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


class BoundaryHiRadixCache(HiRadixCache):
    """Experimental HiRadixCache variant for the L1/L2 boundary invariant."""

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        super().__init__(params=params, server_args=server_args)
        self.cache_controller.write_policy = "write_through"
        self.write_through_threshold = 1

    @staticmethod
    def is_d_only(node: TreeNode) -> bool:
        return node.value is not None and node.host_value is None

    @staticmethod
    def is_dh(node: TreeNode) -> bool:
        return node.value is not None and node.host_value is not None

    @staticmethod
    def is_h_only(node: TreeNode) -> bool:
        return node.value is None and node.host_value is not None

    def _collect_boundary_nodes(self) -> list[TreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def _has_h_descendant(self, node: TreeNode) -> bool:
        stack = list(node.children.values())
        while stack:
            child = stack.pop()
            if self.is_h_only(child):
                return True
            stack.extend(child.children.values())
        return False

    def _has_host_descendant(self, node: TreeNode) -> bool:
        stack = list(node.children.values())
        while stack:
            child = stack.pop()
            if child.host_value is not None:
                return True
            stack.extend(child.children.values())
        return False

    def _is_duplicate_host_candidate(self, node: TreeNode) -> bool:
        return (
            node is not self.root_node
            and self.is_dh(node)
            and node.host_ref_counter == 0
            and node.write_through_pending_id is None
            and not self._has_h_descendant(node)
        )

    def _is_h_only_leaf_candidate(self, node: TreeNode) -> bool:
        return (
            node is not self.root_node
            and self.is_h_only(node)
            and node.host_ref_counter == 0
            and node.write_through_pending_id is None
            and not self._has_host_descendant(node)
        )

    def _depth(self, node: TreeNode) -> int:
        depth = 0
        while node is not self.root_node and node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def _ensure_d_ancestors_are_boundaries(
        self, node: TreeNode, *, write_back: bool
    ) -> bool:
        ancestors = []
        current = node.parent
        while current is not None and current is not self.root_node:
            if self.is_d_only(current):
                ancestors.append(current)
            current = current.parent

        for ancestor in reversed(ancestors):
            if self.write_backup(ancestor, write_back=write_back) <= 0:
                return False
        return True

    def write_backup(self, node: TreeNode, write_back=False) -> int:
        if node.value is None:
            return 0

        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                **self._get_extra_pools(),
            )
        if host_indices is None:
            return 0

        node.host_value = host_indices.clone()
        assert len(node.host_value) > 0
        self._track_write_through_node(node, len(node.key))
        if not write_back:
            self.inc_lock_ref(node)
        return len(host_indices)

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1
        if not node.backuped and node.hit_count >= self.write_through_threshold:
            if self._ensure_d_ancestors_are_boundaries(node, write_back=False):
                self.write_backup(node)

    def _ensure_host_copy(self, node: TreeNode, *, wait: bool = False) -> int:
        if node.host_value is not None:
            return len(node.host_value)
        if node.value is None:
            return 0

        if hasattr(self.cache_controller, "write"):
            written = self.write_backup(node, write_back=True)
            if wait and written > 0:
                self.writing_check(write_back=True)
            return written

        # Unit-test simulation path: no real host pool/controller is present.
        node.host_value = node.value.clone()
        return len(node.host_value)

    def _evict_boundary_l1_node(self, node: TreeNode) -> int:
        if node is self.root_node or node.value is None or node.lock_ref > 0:
            return 0

        parent = node.parent
        if node.host_value is None:
            if parent is not None and self.is_d_only(parent):
                if self._ensure_host_copy(parent, wait=True) <= 0:
                    return 0
                self._update_host_leaf_status(parent)
            if self._ensure_host_copy(node, wait=True) <= 0:
                return 0

        self._record_remove_event(node, medium=StorageMedium.GPU)
        num_evicted = self.cache_controller.evict_device(node.value)
        if num_evicted <= 0:
            return 0

        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        if parent is not None:
            self._update_leaf_status(parent)
            self._update_host_leaf_status(parent)
        return num_evicted

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node)
            for node in self.evictable_leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < params.num_tokens and eviction_heap:
            _priority, node = heapq.heappop(eviction_heap)
            if node.lock_ref > 0 or node.value is None:
                continue

            num_evicted += self._evict_boundary_l1_node(node)

            parent = node.parent
            if parent is None:
                continue
            for child in parent.children.values():
                if child.value is not None:
                    break
            else:
                if parent is not self.root_node and parent.value is not None:
                    heapq.heappush(
                        eviction_heap,
                        (self.eviction_strategy.get_priority(parent), parent),
                    )

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _update_host_leaf_status(self, node: TreeNode):
        if (
            node.host_value is None
            or node.host_ref_counter > 0
            or node is self.root_node
        ):
            self.evictable_host_leaves.discard(node)
            return

        if self._is_duplicate_host_candidate(node) or self._is_h_only_leaf_candidate(
            node
        ):
            self.evictable_host_leaves.add(node)
        else:
            self.evictable_host_leaves.discard(node)

    def _refresh_host_status_around(self, node: TreeNode | None):
        while node is not None:
            self._update_host_leaf_status(node)
            node = node.parent

    def evict_host(self, num_tokens: int) -> int:
        num_evicted = 0
        while num_evicted < num_tokens:
            candidate = self._pick_l2_candidate(duplicates=True)
            if candidate is not None:
                num_evicted += self._drop_duplicate_host_copy(candidate)
                continue

            candidate = self._pick_l2_candidate(duplicates=False)
            if candidate is None:
                break
            num_evicted += self._delete_h_only_leaf(candidate)

        return num_evicted

    def _pick_l2_candidate(self, *, duplicates: bool) -> TreeNode | None:
        candidates = []
        for node in self._collect_boundary_nodes():
            if duplicates:
                if self._is_duplicate_host_candidate(node):
                    candidates.append(
                        (
                            self._depth(node),
                            self.eviction_strategy.get_priority(node),
                            node,
                        )
                    )
            elif self._is_h_only_leaf_candidate(node):
                candidates.append((self.eviction_strategy.get_priority(node), node))

        if not candidates:
            return None
        heapq.heapify(candidates)
        return heapq.heappop(candidates)[-1]

    def _drop_duplicate_host_copy(self, node: TreeNode) -> int:
        self._record_remove_event(node, medium=StorageMedium.CPU)
        num_evicted = self.cache_controller.evict_host(node.host_value)
        node.host_value = None
        self.evictable_host_leaves.discard(node)
        self._refresh_host_status_around(node.parent)
        return num_evicted

    def _delete_h_only_leaf(self, node: TreeNode) -> int:
        assert self.is_h_only(node), f"node {node.id} is not H-only"
        assert not self._has_host_descendant(node), f"node {node.id} is not an H leaf"

        self._record_remove_event(node, medium=StorageMedium.CPU)
        num_evicted = self.cache_controller.evict_host(node.host_value)
        parent = node.parent
        key = node.key.child_key(self.page_size)
        removed = parent.children.pop(key, None)
        assert removed is node, f"parent does not have child key, {key}"
        self.evictable_host_leaves.discard(node)
        node.parent = None
        if parent is not None:
            self._update_leaf_status(parent)
            self._refresh_host_status_around(parent)
        return num_evicted
