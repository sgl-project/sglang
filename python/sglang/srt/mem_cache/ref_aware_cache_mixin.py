"""Mixin providing ref-aware tiered eviction logic.

This mixin extracts the reusable tier-tracking and ref-management logic from
RefAwareHiRadixCache so it can be shared between different cache backends
(HiRadixCache, plain RadixCache, etc.) without duplicating code.

Usage:
    class MyConcretCache(RefAwareCacheMixin, SomeBaseCache):
        def __init__(self, params, server_args):
            super().__init__(params=params, server_args=server_args)
            self._init_ref_aware_state(server_args)
"""

from __future__ import annotations

import heapq
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    TreeNode,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RefInfo:
    is_high: bool
    nodes: Set[TreeNode] = field(default_factory=set)
    cached_tokens: int = 0


# ---------------------------------------------------------------------------
# Eviction tier constants
# ---------------------------------------------------------------------------

TIER_UNUSED = 0  # high_ref == 0, low_ref == 0
TIER_LOW_REF = 1  # high_ref == 0, low_ref > 0
TIER_HIGH_REF = 2  # high_ref > 0


def _classify_node_tier(node: TreeNode) -> int:
    if node.high_ref > 0:
        return TIER_HIGH_REF
    if node.low_ref > 0:
        return TIER_LOW_REF
    return TIER_UNUSED


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------


class RefAwareCacheMixin:
    """Mixin that adds ref-aware tiered eviction to any radix-style cache.

    Concrete subclasses must call ``_init_ref_aware_state(server_args)`` from
    their own ``__init__`` and ``_reset_ref_aware_state()`` from ``reset()``.
    """

    # ------------------------------------------------------------------
    # Initialization helpers (NOT __init__)
    # ------------------------------------------------------------------

    def _init_ref_aware_state(self, server_args: ServerArgs):
        """Initialize all ref-aware tier tracking state.

        Must be called from the concrete class's ``__init__`` after
        ``super().__init__(...)`` has been invoked.
        """
        self.high_priority_threshold = getattr(
            server_args, "high_priority_threshold", 1
        )
        self._enable_priority_scheduling = getattr(
            server_args, "enable_priority_scheduling", False
        )
        self.unused_evictable_leaves: set = set()
        self.low_ref_evictable_leaves: set = set()
        self.high_ref_evictable_leaves: set = set()
        self.unused_evictable_size_: int = 0
        self.low_ref_evictable_size_: int = 0
        self.high_ref_evictable_size_: int = 0
        self.session_id_to_ref_info: Dict[str, RefInfo] = {}
        self._evict_scope_stack: list[tuple[bool, bool]] = []

    def _reset_ref_aware_state(self):
        """Clear all ref-aware tier tracking state.

        Must be called from the concrete class's ``reset()`` method.
        """
        self.unused_evictable_leaves.clear()
        self.low_ref_evictable_leaves.clear()
        self.high_ref_evictable_leaves.clear()
        self.unused_evictable_size_ = 0
        self.low_ref_evictable_size_ = 0
        self.high_ref_evictable_size_ = 0
        self.session_id_to_ref_info.clear()
        self._evict_scope_stack.clear()

    # ------------------------------------------------------------------
    # Priority classification
    # ------------------------------------------------------------------

    def is_high_priority(self, priority: int) -> bool:
        if not self._enable_priority_scheduling:
            return True
        return priority >= self.high_priority_threshold

    # ------------------------------------------------------------------
    # Tier bookkeeping
    # ------------------------------------------------------------------

    def _move_node_tier(self, node: TreeNode, old_tier: int, new_tier: int):
        assert (
            not node.evicted and node.lock_ref == 0
        ), "_move_node_tier called for evicted or lock-held node"
        node_size = len(node.key)
        old_set = self._tier_leaf_set(old_tier)
        new_set = self._tier_leaf_set(new_tier)
        if node in old_set:
            old_set.discard(node)
            # Only re-add if node is still a valid evictable leaf
            is_leaf = all(c.evicted for c in node.children.values())
            if is_leaf:
                new_set.add(node)
        self._add_tier_size(old_tier, -node_size)
        self._add_tier_size(new_tier, node_size)

    def _tier_leaf_set(self, tier: int) -> set:
        if tier == TIER_UNUSED:
            return self.unused_evictable_leaves
        elif tier == TIER_LOW_REF:
            return self.low_ref_evictable_leaves
        else:
            return self.high_ref_evictable_leaves

    def _add_tier_size(self, tier: int, delta: int):
        if tier == TIER_UNUSED:
            self.unused_evictable_size_ += delta
        elif tier == TIER_LOW_REF:
            self.low_ref_evictable_size_ += delta
        else:
            self.high_ref_evictable_size_ += delta

    def _account_new_evictable_node(self, node: TreeNode):
        if node in (None, self.root_node) or node.evicted or node.lock_ref > 0:
            return
        self._add_tier_size(_classify_node_tier(node), len(node.key))

    # ------------------------------------------------------------------
    # Leaf status tracking
    # ------------------------------------------------------------------

    def _update_leaf_status(self, node: TreeNode):
        super()._update_leaf_status(node)
        self._update_ref_aware_leaf_status(node)

    def _update_ref_aware_leaf_status(self, node: TreeNode):
        self.unused_evictable_leaves.discard(node)
        self.low_ref_evictable_leaves.discard(node)
        self.high_ref_evictable_leaves.discard(node)

        if node.evicted or node.lock_ref > 0:
            return

        for child in node.children.values():
            if not child.evicted:
                return

        tier = _classify_node_tier(node)
        self._tier_leaf_set(tier).add(node)

    # ------------------------------------------------------------------
    # Lock ref hooks
    # ------------------------------------------------------------------

    def _on_lock_ref_node(self, node: TreeNode):
        pass

    # ------------------------------------------------------------------
    # inc_lock_ref / dec_lock_ref — full reimplementation with tier accounting
    # ------------------------------------------------------------------

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
                if not node.evicted:
                    tier = _classify_node_tier(node)
                    tier_set = self._tier_leaf_set(tier)
                    if node in tier_set:
                        tier_set.discard(node)
                    self._add_tier_size(tier, -len(node.key))
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._on_lock_ref_node(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(self, node: TreeNode, params=None) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
                if not node.evicted:
                    tier = _classify_node_tier(node)
                    self._add_tier_size(tier, len(node.key))
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._on_lock_ref_node(node)
            if node.parent is None:
                assert node is self.root_node
            node = node.parent
        return DecLockRefResult(delta=delta)

    # ------------------------------------------------------------------
    # _delete_leaf — tier cleanup + tracked_session_ids cleanup
    # ------------------------------------------------------------------

    def _delete_leaf(self, node):
        tier = _classify_node_tier(node)
        self._tier_leaf_set(tier).discard(node)
        self._add_tier_size(tier, -len(node.key))
        for session_id in node.tracked_session_ids:
            ref_info = self.session_id_to_ref_info.get(session_id)
            if ref_info is not None:
                ref_info.nodes.discard(node)
        node.tracked_session_ids.clear()
        super()._delete_leaf(node)

    # ------------------------------------------------------------------
    # Tiered eviction size queries
    # ------------------------------------------------------------------

    def evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        total = self.unused_evictable_size_
        if allow_low:
            total += self.low_ref_evictable_size_
        if allow_high:
            total += self.high_ref_evictable_size_
        return total

    def safe_evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        """Return safely evictable size by tier.

        Default implementation returns the same as evictable_size_by_tier.
        Override in HiCache variants where host-backed nodes change the
        calculation.
        """
        return self.evictable_size_by_tier(allow_low=allow_low, allow_high=allow_high)

    @contextmanager
    def scoped_evict(self, allow_low: bool = True, allow_high: bool = False):
        self._evict_scope_stack.append((allow_low, allow_high))
        try:
            yield
        finally:
            self._evict_scope_stack.pop()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        protected_size = self.protected_size()
        pool_size = getattr(self.token_to_kv_pool_allocator, "size", None)
        tier_sum = (
            self.unused_evictable_size_
            + self.low_ref_evictable_size_
            + self.high_ref_evictable_size_
        )
        leaked = (
            pool_size - (available_size + evictable_size + protected_size)
            if pool_size is not None
            else None
        )
        return (
            f"Available tokens: {available_size + evictable_size} "
            f"({available_size=} + {evictable_size=}, "
            f"unused_evictable_size={self.unused_evictable_size_}, "
            f"low_ref_evictable_size={self.low_ref_evictable_size_}, "
            f"high_ref_evictable_size={self.high_ref_evictable_size_}, "
            f"{protected_size=}, {pool_size=}, {tier_sum=}, {leaked=})\n"
        )

    # ------------------------------------------------------------------
    # Tier priority & shared eviction heap
    # ------------------------------------------------------------------

    def _get_tier_priority(self, node: TreeNode, target_tier: int):
        """Compute eviction priority for a node within its tier.

        Primary key: high_ref count (more -> evict later).
        Secondary key: low_ref count (more -> evict later).
        Tertiary key: time-based tiebreaker matching the tier's semantics.
        """
        if target_tier == TIER_HIGH_REF:
            return (node.high_ref, node.low_ref, -node.last_access_time)
        return (node.high_ref, node.low_ref, self.eviction_strategy.get_priority(node))

    def _evict_from_tier_heap(
        self,
        num_tokens: int,
        leaf_set: set,
        target_tier: int,
        evict_one_fn,
    ) -> int:
        """
        Shared heap-based eviction framework.
        """
        leaves = list(leaf_set)
        eviction_heap = [
            (self._get_tier_priority(node, target_tier), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and eviction_heap:
            _priority, x = heapq.heappop(eviction_heap)
            if x.lock_ref > 0:
                continue
            if _classify_node_tier(x) != target_tier:
                continue

            num_evicted += evict_one_fn(x)

            for child in x.parent.children.values():
                if not child.evicted:
                    break
            else:
                if x.parent.lock_ref == 0 and x.parent != self.root_node:
                    if _classify_node_tier(x.parent) == target_tier:
                        new_priority = self._get_tier_priority(x.parent, target_tier)
                        heapq.heappush(eviction_heap, (new_priority, x.parent))

        return num_evicted

    # ------------------------------------------------------------------
    # Explicit ref management for multi-turn requests
    # ------------------------------------------------------------------

    def session_id_for_req(self, req: Req) -> Optional[str]:
        session_id = getattr(req, "session_id", None)
        if session_id is None:
            session_id = getattr(getattr(req, "session", None), "session_id", None)
        return session_id

    def register_ref(self, req: Req):
        session_id = self.session_id_for_req(req)
        if session_id is None:
            return
        is_high = self.is_high_priority(getattr(req, "priority", 0) or 0)

        if session_id not in self.session_id_to_ref_info:
            self.session_id_to_ref_info[session_id] = RefInfo(is_high=is_high)

        ref_info = self.session_id_to_ref_info[session_id]

        last_node = getattr(req, "last_node", None)
        if last_node not in (None, self.root_node):
            new_nodes = self._collect_untracked_nodes_from_last_node(
                last_node, ref_info.nodes
            )
        else:
            token_ids = (req.origin_input_ids + req.output_ids)[: req.kv_committed_len]
            if not token_ids:
                return

            radix_key = RadixKey(
                token_ids, getattr(req, "extra_key", None)
            ).page_aligned(self.page_size)
            if len(radix_key) == 0:
                return

            nodes_on_path = self._collect_nodes_on_path(radix_key)
            new_nodes = [node for node in nodes_on_path if node not in ref_info.nodes]

        for node in new_nodes:
            self._inc_priority_ref_single(node, is_high)
            ref_info.nodes.add(node)
            node.tracked_session_ids.add(session_id)

        ref_info.cached_tokens = sum(len(n.key) for n in ref_info.nodes)

    def _collect_nodes_on_path(self, key: RadixKey):
        node = self.root_node
        nodes = []

        while len(key) > 0:
            ck = key.child_key(self.page_size)
            if ck not in node.children:
                break
            child = node.children[ck]
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len <= 0:
                break
            nodes.append(child)
            if prefix_len < len(child.key):
                break
            node = child
            key = key[prefix_len:]
        return nodes

    def _collect_untracked_nodes_from_last_node(
        self, node: Optional[TreeNode], tracked_nodes: Set[TreeNode]
    ) -> list[TreeNode]:
        nodes = []
        while node not in (None, self.root_node):
            if node in tracked_nodes:
                break
            nodes.append(node)
            node = node.parent
        return nodes

    def _inc_priority_ref_single(self, node: TreeNode, is_high: bool):
        old_tier = _classify_node_tier(node)
        if is_high:
            node.high_ref += 1
        else:
            node.low_ref += 1
        new_tier = _classify_node_tier(node)
        if not node.evicted and node.lock_ref == 0 and old_tier != new_tier:
            self._move_node_tier(node, old_tier, new_tier)

    def _dec_priority_ref_single(self, node: TreeNode, is_high: bool):
        old_tier = _classify_node_tier(node)
        if is_high:
            node.high_ref = max(0, node.high_ref - 1)
        else:
            node.low_ref = max(0, node.low_ref - 1)
        new_tier = _classify_node_tier(node)
        if not node.evicted and node.lock_ref == 0 and old_tier != new_tier:
            self._move_node_tier(node, old_tier, new_tier)

    def release_ref(self, session_id: str) -> Tuple[bool, str]:
        if session_id is None:
            return False, "session_id is None"
        ref_info = self.session_id_to_ref_info.pop(session_id, None)
        if ref_info is None:
            return True, f"session_id {session_id} not tracked"

        for node in ref_info.nodes:
            self._dec_priority_ref_single(node, ref_info.is_high)
            node.tracked_session_ids.discard(session_id)

        return True, f"released {len(ref_info.nodes)} nodes for session_id {session_id}"

    def update_ref(self, session_id: str, new_priority: int) -> Tuple[bool, str]:
        if session_id is None:
            return False, "session_id is None"
        ref_info = self.session_id_to_ref_info.get(session_id)
        if ref_info is None:
            return False, f"session_id {session_id} not found in ref tracking"

        new_is_high = self.is_high_priority(new_priority)

        if new_is_high == ref_info.is_high:
            return True, "priority class unchanged"

        for node in ref_info.nodes:
            self._dec_priority_ref_single(node, ref_info.is_high)
            self._inc_priority_ref_single(node, new_is_high)
        ref_info.is_high = new_is_high
        return True, f"updated {len(ref_info.nodes)} nodes for session_id {session_id}"

    # ------------------------------------------------------------------
    # Split node override to propagate high_ref / low_ref
    # ------------------------------------------------------------------

    def _split_node(self, key, child, split_len):
        new_node = super()._split_node(key, child, split_len)
        new_node.high_ref = child.high_ref
        new_node.low_ref = child.low_ref
        new_node.tracked_session_ids = set(child.tracked_session_ids)
        # Update session_id_to_ref_info: add new_node to each tracking session_id's node set
        for session_id in new_node.tracked_session_ids:
            ref_info = self.session_id_to_ref_info.get(session_id)
            if ref_info is not None:
                ref_info.nodes.add(new_node)
        self._update_ref_aware_leaf_status(new_node)
        self._update_ref_aware_leaf_status(child)
        return new_node
