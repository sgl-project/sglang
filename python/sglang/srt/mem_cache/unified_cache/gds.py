"""GreedyDual-Size eviction of a hybrid model's cached states
(``--radix-eviction-policy gds``).

A unit is a node holding a Mamba state on device, together with the
full-attention KV only it can serve: when no other state lies below it, the
node's own KV and that of its stateless single-child ancestors, up to the
nearest one holding a state. Each unit is scored ``stamp + gain / size``:

- ``gain``: the seconds a hit on it saves, the prefill of
  ``[fallback, depth)`` under a two-coefficient cost model, ``fallback`` being
  the depth of the nearest ancestor holding a state (or 0);
- ``size``: what evicting it frees, its state slot plus its exclusive KV, in
  KV tokens (a state slot counts as the KV tokens of the same bytes);
- ``stamp``: the clock ``L`` when the unit was last cached or hit, ``L`` being
  the score of the last unit evicted.

The lowest score goes first. With ``leaf_last``, a unit with exactly one
child branch holding a state is interior, every interior unit goes before any
leaf, and ``L`` rises only when a leaf goes. A branch point (two child
branches holding states) ranks as a leaf. Which subtrees hold a state comes
from the tree's state counts (``StateUnits``), so scoring walks no subtree.

Scores are recomputed when a unit is taken from the heap; a unit whose score
rose is pushed back. Events that lower a score push the affected units at
once: a deeper unit cached under one, a unit cached above one, a branch below
one going, a sibling branch going and leaving their parent with one child
and no state, or the path cap (--mamba-max-states-per-path) taking the state
above one, whose tokens it then gains and whose KV it may hold. Entries a
push or an eviction made stale stay in the heap until it holds twice as many
entries as units, when it is rebuilt from the live ones.

KV that no state serves goes before any unit, least recent first, from the
queue unit eviction keeps (``StateUnits``), so an eviction ranks no leaves.
"""

from __future__ import annotations

import heapq
import itertools
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.kv.plan import TREE_STATE
from sglang.srt.mem_cache.unified_cache.component_type import BASE_COMPONENT_TYPE

if TYPE_CHECKING:
    from sglang.srt.mem_cache.evict_policy import GreedyDualSizeStrategy
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        UnifiedTreeCore,
        UnifiedTreeNode,
    )

INTERIOR, LEAF = 1, 2
# Units collected below a node whose exclusive KV may have grown.
_NEIGHBOR_SCAN_LIMIT = 64
# Heap entries kept beyond twice the units before a rebuild.
_HEAP_SLACK = 64


class GreedyDualSizeUnits:
    """The tree's state units in GreedyDual-Size order."""

    def __init__(self, core: UnifiedTreeCore, strategy: GreedyDualSizeStrategy):
        self.core = core
        self.state_component = core.slot_id(TREE_STATE)
        self.per_token, self.per_token_per_100k = strategy.prefill_cost
        self.leaf_last = strategy.leaf_last
        # A state slot's size in KV tokens; the cache sets it from its pools.
        self.state_tokens: Optional[float] = None
        self.clock = 0.0
        self.num_evicted = 0
        self._heap: list = []
        self._seq = itertools.count()
        self._version: dict[UnifiedTreeNode, int] = {}
        self._stamp: dict[UnifiedTreeNode, float] = {}
        self._depth: dict[UnifiedTreeNode, int] = {}
        self._parked: list[UnifiedTreeNode] = []

    # ---- events from the tree ----

    def added(self, node: UnifiedTreeNode) -> None:
        self.used(node)
        above = self._nearest_unit_above(node)
        if above is not None:
            # It has a deeper unit now: it may be interior.
            self.push(above)
        # The units below now gain and hold KV only from here down.
        self.push_units_below(node)

    def used(self, node: UnifiedTreeNode) -> None:
        self._stamp[node] = self.clock
        self.push(node)

    def removed(self, node: UnifiedTreeNode) -> None:
        """The node's device state went; its heap entries are stale now."""
        self._version.pop(node, None)
        self._stamp.pop(node, None)

    def forget(self, node: UnifiedTreeNode) -> None:
        """The node left the tree."""
        self.removed(node)
        self._depth.pop(node, None)

    def reset(self) -> None:
        self.clock = 0.0
        self._heap.clear()
        self._version.clear()
        self._stamp.clear()
        self._depth.clear()
        self._parked.clear()

    # ---- scoring ----

    def _is_unit(self, node: UnifiedTreeNode) -> bool:
        return node.component_data[self.state_component].value is not None

    def depth(self, node: UnifiedTreeNode) -> int:
        """Tokens from the root to the end of ``node``."""
        chain = []
        cur = node
        while cur is not self.core.root_node and cur not in self._depth:
            chain.append(cur)
            cur = cur.parent
        depth = 0 if cur is self.core.root_node else self._depth[cur]
        for cur in reversed(chain):
            depth += len(cur.key)
            self._depth[cur] = depth
        return depth

    def gain(self, node: UnifiedTreeNode) -> float:
        b, n = 0, self.depth(node)
        holds = self.core.state_units.holds
        cur = node.parent
        while cur is not None and cur is not self.core.root_node:
            if holds(cur):
                b = self.depth(cur)
                break
            cur = cur.parent
        return (n - b) * self.per_token + self.per_token_per_100k * (
            n * n - b * b
        ) / 2e5

    def shape(self, node: UnifiedTreeNode) -> tuple[int, int]:
        """(child branches holding a state, KV tokens only this unit serves)."""
        units = self.core.state_units
        branches = units.branches(node)
        if branches:
            return branches, 0
        exclusive = self._device_tokens(node)
        cur = node.parent
        while (
            cur is not self.core.root_node
            and not units.holds(cur)
            and len(cur.children) == 1
        ):
            exclusive += self._device_tokens(cur)
            cur = cur.parent
        return 0, exclusive

    @staticmethod
    def _device_tokens(node: UnifiedTreeNode) -> int:
        value = node.component_data[BASE_COMPONENT_TYPE].value
        return 0 if value is None else len(value)

    def score(self, node: UnifiedTreeNode) -> tuple:
        branches, exclusive = self.shape(node)
        score = self._stamp.get(node, self.clock) + self.gain(node) / (
            self.state_tokens + exclusive
        )
        if not self.leaf_last:
            return (score,)
        return (INTERIOR if branches == 1 else LEAF, score)

    def frees_kv(self, node: UnifiedTreeNode) -> bool:
        branches, exclusive = self.shape(node)
        return branches == 0 and exclusive > 0

    # ---- the heap ----

    def push(self, node: UnifiedTreeNode) -> None:
        if not self._is_unit(node):
            return
        version = next(self._seq)
        self._version[node] = version
        heapq.heappush(self._heap, (self.score(node), version, node))
        if len(self._heap) > 2 * len(self._version) + _HEAP_SLACK:
            self._heap = [
                entry for entry in self._heap if self._version.get(entry[2]) == entry[1]
            ]
            heapq.heapify(self._heap)

    def _nearest_unit_above(self, node: UnifiedTreeNode) -> Optional[UnifiedTreeNode]:
        cur = node.parent
        while cur is not None and cur is not self.core.root_node:
            if self._is_unit(cur):
                return cur
            cur = cur.parent
        return None

    def after_eviction(self, node: UnifiedTreeNode) -> None:
        """An eviction took a branch or the state below ``node``, the nearest
        node left. Its own unit lost a branch; when it is left with one child
        and no state, the units under that child may own its KV now too."""
        self.push(node)
        if len(node.children) == 1 and not self.core.state_units.holds(node):
            self.push_units_below(node)

    def push_units_below(self, node: UnifiedTreeNode) -> None:
        """Push the nearest units below ``node``: their gain or their
        exclusive KV may have shrunk or grown."""
        # Depth first, newest child first, without listing a wide node's children.
        stack = [reversed(node.children.values())]
        seen = 0
        while stack and seen < _NEIGHBOR_SCAN_LIMIT:
            cur = next(stack[-1], None)
            if cur is None:
                stack.pop()
                continue
            seen += 1
            if self._is_unit(cur):
                self.push(cur)
            else:
                stack.append(reversed(cur.children.values()))

    def next_victim(self, *, kv_only: bool, evictable) -> Optional[UnifiedTreeNode]:
        """Take the unit with the lowest current score that ``evictable``
        accepts; with ``kv_only``, only a unit that frees KV. Units passed over
        wait for ``end_walk``."""
        heap = self._heap
        while heap:
            score, version, node = heap[0]
            if self._version.get(node) != version or not self._is_unit(node):
                heapq.heappop(heap)
                continue
            current = self.score(node)
            if current > score:
                version = next(self._seq)
                self._version[node] = version
                heapq.heapreplace(heap, (current, version, node))
                continue
            heapq.heappop(heap)
            if not evictable(node) or (kv_only and not self.frees_kv(node)):
                self._parked.append(node)
                continue
            if not self.leaf_last or current[0] == LEAF:
                self.clock = max(self.clock, current[-1])
            self._version.pop(node, None)
            self.num_evicted += 1
            return node
        return None

    def end_walk(self) -> None:
        parked, self._parked = self._parked, []
        for node in parked:
            self.push(node)

    def check(self, report) -> None:
        """Between walks, report each unit whose live heap entry is missing
        or above its score now: it would go later than its score says."""
        live = {
            node: score
            for score, version, node in self._heap
            if self._version.get(node) == version
        }
        order = [self.core.root_node]
        for cur in order:
            order.extend(cur.children.values())
            if cur is self.core.root_node or not self._is_unit(cur):
                continue
            entry, score = live.get(cur), self.score(cur)
            if entry is None or entry > score:
                report(f"[GDS] node {cur.id}: heap entry {entry} above score {score}")
