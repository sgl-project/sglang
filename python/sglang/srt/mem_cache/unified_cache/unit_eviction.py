"""Evicting a hybrid model's cached state together with the KV only it serves
(``--enable-mamba-unit-eviction``).

A request resumes a cached prefix only at a node holding a Mamba state, so
full-attention KV with no state at or below it serves no request. When a node
loses its state and none remains at or below it, its subtree and its
stateless ancestors, up to the nearest one whose subtree still holds a state,
are deleted; locked nodes, session references and nodes with in-flight copies
keep their KV and that of their ancestors.

Every node counts the nodes at or below it that hold a state on either tier,
and its children whose subtrees hold one. A state appearing or going updates
the counts on its path to the root, so asking whether a subtree still serves a
request takes no walk.

Evictable leaves that no state serves wait in a queue, least recent first,
filled as the tree makes leaves evictable and checked when taken. Eviction
takes from it before anything else, so it never sorts the tree's leaves for
them. Among themselves they go by recency alone, whatever
--radix-eviction-policy ranks the other KV by and whether a session refers to
them: no request can resume from them.
"""

from __future__ import annotations

import heapq
import itertools
from typing import TYPE_CHECKING, Optional

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.kv.plan import TREE_STATE
from sglang.srt.mem_cache.unified_cache.component_type import BASE_COMPONENT_TYPE

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        UnifiedTreeCore,
        UnifiedTreeNode,
    )

# Queue entries kept beyond twice the evictable leaves before a rebuild.
_QUEUE_SLACK = 64


class StateUnits:
    """The subtrees that still hold a state, the release of the KV that none
    serves, and the queue of evictable leaves whose KV none serves."""

    def __init__(self, core: UnifiedTreeCore):
        self.core = core
        self.state_component = core.slot_id(TREE_STATE)
        # (last access, seq, node), and each queued node's live (last access, seq).
        self._dead: list = []
        self._dead_entry: dict[UnifiedTreeNode, tuple[float, int]] = {}
        self._seq = itertools.count()

    def reset(self) -> None:
        self._dead.clear()
        self._dead_entry.clear()

    # ---- the counts ----

    def holds(self, node: UnifiedTreeNode) -> bool:
        """Whether ``node`` holds a state on either tier."""
        cd = node.component_data[self.state_component]
        return cd.value is not None or cd.host_value is not None

    @staticmethod
    def subtree_holds(node: UnifiedTreeNode) -> bool:
        """Whether a state at or below ``node`` remains: a match ends at a
        state, so KV with none at or below it serves no request."""
        return node._unit_states_below > 0

    @staticmethod
    def branches(node: UnifiedTreeNode) -> int:
        """The children of ``node`` whose subtrees hold a state."""
        return node._unit_state_branches

    def state_changed(self, node: UnifiedTreeNode) -> None:
        """Count ``node``'s state appearing or going, on either tier."""
        holds = self.holds(node)
        if holds != node._unit_holds_state:
            node._unit_holds_state = holds
            self._add(node, 1 if holds else -1)

    @staticmethod
    def split(new_parent: UnifiedTreeNode, child: UnifiedTreeNode) -> None:
        """``new_parent`` took the front of ``child``'s key; the state, if
        any, stays with ``child``."""
        new_parent._unit_states_below = child._unit_states_below
        new_parent._unit_state_branches = int(child._unit_states_below > 0)

    @staticmethod
    def _add(node: UnifiedTreeNode, delta: int) -> None:
        cur = node
        while cur is not None:
            before = cur._unit_states_below
            cur._unit_states_below = before + delta
            parent = cur.parent
            if parent is not None and (before > 0) != (cur._unit_states_below > 0):
                parent._unit_state_branches += 1 if before == 0 else -1
            cur = parent

    def check(self, report) -> None:
        """Report each node whose counts differ from a recount, and each
        evictable leaf that no state serves and that waits in no queue."""
        order = [self.core.root_node]
        for cur in order:
            order.extend(cur.children.values())
        below = {}
        for node in reversed(order):
            holds = self.holds(node)
            counts = [below[child] for child in node.children.values()]
            below[node] = int(holds) + sum(counts)
            expected = (holds, below[node], sum(1 for n in counts if n))
            found = (
                node._unit_holds_state,
                node._unit_states_below,
                node._unit_state_branches,
            )
            if found != expected:
                report(f"[Units] node {node.id} counts {found} != recount {expected}")
        for node in self.core.evictable_device_leaves:
            if not self.subtree_holds(node) and node not in self._dead_entry:
                report(f"[Units] node {node.id}: KV no state serves is not queued")

    # ---- the queue of KV that no state serves ----

    def evictable_leaf(self, node: UnifiedTreeNode) -> None:
        """The tree made ``node`` an evictable device leaf: queue it if no
        state at or below it serves its KV."""
        if self.subtree_holds(node):
            return
        accessed = node.last_access_time
        entry = self._dead_entry.get(node)
        if entry is not None and entry[0] == accessed:
            return
        seq = next(self._seq)
        self._dead_entry[node] = (accessed, seq)
        heapq.heappush(self._dead, (accessed, seq, node))
        if len(self._dead) > 2 * len(self.core.evictable_device_leaves) + _QUEUE_SLACK:
            self._requeue()

    def _requeue(self) -> None:
        self._dead_entry = {}
        self._dead = []
        for node in self.core.evictable_device_leaves:
            if not self.subtree_holds(node):
                seq = next(self._seq)
                self._dead_entry[node] = (node.last_access_time, seq)
                self._dead.append((node.last_access_time, seq, node))
        heapq.heapify(self._dead)

    def next_dead_leaf(self) -> Optional[UnifiedTreeNode]:
        """The least recent evictable leaf that no state serves."""
        heap, entries = self._dead, self._dead_entry
        leaves = self.core.evictable_device_leaves
        while heap:
            accessed, seq, node = heap[0]
            entry = entries.get(node)
            if entry is None or entry[1] != seq:
                heapq.heappop(heap)
            elif node not in leaves or self.subtree_holds(node):
                heapq.heappop(heap)
                del entries[node]
            elif node.last_access_time != accessed:
                # Used since it was queued: its turn moved.
                seq = next(self._seq)
                entries[node] = (node.last_access_time, seq)
                heapq.heapreplace(heap, (node.last_access_time, seq, node))
            else:
                heapq.heappop(heap)
                del entries[node]
                return node
        return None

    # ---- the release ----

    @staticmethod
    def pinned(node: UnifiedTreeNode) -> bool:
        """A request, a session or an in-flight copy still uses the node."""
        return (
            node.write_back_queued
            or node.write_through_pending_id is not None
            or node.load_back_pending_id is not None
            or any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0 or cd.session_ref > 0
                for cd in node.component_data
            )
        )

    def release(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """``node`` lost its state. Delete the KV no state needs any more: the
        subtree of ``node`` when it holds no state, and the ancestors up to the
        nearest one whose subtree still holds a state. A pinned node keeps its
        KV and that of its ancestors."""
        core = self.core
        if node is core.root_node or self.subtree_holds(node):
            return
        top = node
        while top.parent is not core.root_node and not self.subtree_holds(top.parent):
            top = top.parent
        # Breadth-first, so the reversed order deletes children before parents.
        order = [top]
        for cur in order:
            order.extend(cur.children.values())
        survivor = top.parent
        for cur in reversed(order):
            if cur.children or self.holds(cur) or self.pinned(cur):
                core._update_evictable_leaf_sets(cur)
                continue
            self.delete_dead(
                cur, tracker, device_frees, host_frees, cache_actions=cache_actions
            )
        core._update_evictable_leaf_sets(survivor)

    def delete_dead(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """Free a childless node's layers on both tiers and delete it."""
        core = self.core
        if not node.evicted and node.backuped:
            core.kv_events.record_remove(node, medium=StorageMedium.CPU)
        core._release_all_component_layers(
            node,
            StorageMedium.CPU if node.evicted else StorageMedium.GPU,
            tracker,
            device_frees,
            host_frees,
            cache_actions=cache_actions,
        )
        # Full defers its tombstone to the caller (see _cascade_evict); a walk
        # holding this node must see it evicted.
        node.component_data[BASE_COMPONENT_TYPE].value = None
        core._remove_leaf_from_parent(node)
