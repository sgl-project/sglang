"""
Cache components (Full/Mamba/SWA/etc) implementation.
"""

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from .lru_list import LRUList
from .tree_node import ComponentData, TreeNode, get_last_access_time

if TYPE_CHECKING:
    from .hybrid_radix_tree import HybridRadixTree

logger = logging.getLogger(__name__)


class CacheComponent(ABC):
    """Base class for cache components (Full/Mamba/SWA).

    Primary vs Secondary:
        - Primary: Cannot be tombstone, always has value (e.g., Full attention KV)
        - Secondary: Can be tombstone, optional value (e.g., Mamba state, SWA)

    Required methods:
        - get_priority(is_leaf): Eviction priority
        - get_lock_range(node, root): Nodes to lock

    Optional overrides:
        - Tree operation hooks: check_prefix_match_constraints, on_insert_split_node, on_insert_leaf
        - Request hooks: prepare_component_value, free_value
        - Post-cache hooks: after_request_cached
    """

    def __init__(self, name: str, is_primary: bool):
        self.name = name
        self.is_primary = is_primary
        self.allocator = None
        self.lru_list = LRUList(name)
        self.tree: Optional["HybridRadixTree"] = None
        self.evictable_size = 0
        self.protected_size = 0

    def setup(self, tree: "HybridRadixTree", allocator):
        """Setup tree and allocator (called during registration)."""
        self.tree = tree
        self.allocator = allocator

    # ========== Required Methods ==========

    @abstractmethod
    def get_priority(self, is_leaf: bool) -> int:
        """Eviction priority (higher = evict later)."""
        pass

    @abstractmethod
    def get_lock_range(self, node: TreeNode, root: TreeNode) -> List[TreeNode]:
        """Return nodes to lock."""
        pass

    # ========== Match Hooks ==========

    def update_prefix_match_context(self, node: TreeNode, context: dict, init: bool = False) -> None:
        """Update component-specific match context during traversal."""
        pass

    def check_prefix_match_constraints(self, node: TreeNode, context: dict, child: Optional[TreeNode] = None) -> bool:
        """Check if node satisfies component constraints given current context.
        
        Args:
            node: Current node to check
            child: Next child node (None for final check)
            context: Match context accumulated during traversal
        
        Returns:
            bool: True = should accept node as best_match, False = reject
        """
        data = self.get_component_data(node)
        return data is not None and not data.is_tombstone()

    def on_match_complete(self, matched_node: TreeNode, req):
        """Called after match_prefix completes (before returning result).
        
        Use for post-match operations like:
        - Mamba: COW (copy-on-write) of matched state
        - Resource preparation for upcoming cache operation
        """
        pass

    # ========== Insert Hooks ==========

    def on_insert_split_node(
        self, old_node: TreeNode, new_node: TreeNode, split_len: int
    ):
        """Called when node is split (handle value split, lock_ref, LRU)."""
        pass

    def on_insert_leaf(self, node: TreeNode, value: torch.Tensor):
        """Called when a new leaf node is inserted."""
        existing_data = self.get_component_data(node)

        if existing_data and not existing_data.is_tombstone():
            if self.lru_list.in_list(node):
                self.lru_list.reset_node_mru(node)
            return

        data = self.get_component_data(node, create=True)
        data.value = value
        self.evictable_size += len(value)

        if not self.lru_list.in_list(node):
            self.lru_list.insert_mru(node)
        else:
            self.lru_list.reset_node_mru(node)

    # ========== Eviction Logic ==========

    def evict_cascade(self, num_tokens: int) -> int:
        """Evict tokens using priority-based cascading.

        Priority rules:
        - Primary component (full): always priority 100
        - Secondary components (mamba): priority 100 for leaves, 1 for non-leaves
        - Evicting a component triggers cascading eviction for all components with equal or lower priority
        """
        if num_tokens <= 0:
            return 0

        # Get initial LRU node based on component type
        if self.is_primary:
            node = self.lru_list.get_leaf_lru_no_lock(self.is_locked)
        else:
            node = self.lru_list.get_lru_no_lock(self.is_locked)

        evicted = 0
        while evicted < num_tokens and self.lru_list.in_list(node):
            is_leaf = len(node.children) == 0

            # Get next node BEFORE evicting current node
            if is_leaf:
                if self.is_primary:
                    next_node = self.lru_list.get_prev_leaf_no_lock(
                        node, self.is_locked
                    )
                else:
                    next_node = self.lru_list.get_prev_no_lock(node, self.is_locked)
            else:
                next_node = self.lru_list.get_prev_no_lock(node, self.is_locked)

            # Evict current node with cascading
            trigger_priority = self.get_priority(is_leaf)
            size_before = self.evictable_size

            for comp in self.tree.components.values():
                if comp.get_priority(is_leaf) <= trigger_priority:
                    comp.free_value(node, update_accounting=True)

            if is_leaf:
                self.tree._delete_node_cascade(node)
                # If parent became a leaf after deletion, restart from LRU leaf
                if self.is_primary and node.parent and len(node.parent.children) == 0:
                    next_node = self.lru_list.get_leaf_lru_no_lock(self.is_locked)

            size_after = self.evictable_size
            evicted += size_before - size_after
            node = next_node

        return evicted

    # ========== Cache Hooks ==========

    def prepare_component_value(self, req, is_finished: bool):
        """Prepare component value from request to cache.

        For primary: KV indices
        For secondary: component-specific state
        """
        return None

    def after_request_cached(self, req, is_finished: bool, existed: bool):
        """Called after request has been cached into tree.

        Returns buffer_to_keep for special cleanup (e.g., Mamba ping-pong buffer).
        """
        return None

    def free_value(self, value_or_node, update_accounting: bool = True) -> int:
        """Free component value.

        Args:
            value_or_node: TreeNode (if update_accounting=True) or raw value tensor (if False)
            update_accounting: Whether to update evictable_size and lru_list
                              True: value is in tree, need to update statistics
                              False: value was never cached, just free resource

        Returns:
            Number of tokens freed (0 if update_accounting=False)
        """
        if update_accounting:
            # Value is in tree node, need to update statistics
            node = value_or_node
            data = self.get_component_data(node)
            if data is None or data.is_tombstone():
                return 0

            value = data.value
            freed = len(value)
            self.allocator.free(value)
            data.value = None
            self.evictable_size -= freed

            if len(node.children) > 0 and self.lru_list.in_list(node):
                self.lru_list.remove_node(node)

            return freed
        else:
            # Value was never cached, just free resource
            value = value_or_node
            if value is not None:
                self.allocator.free(value)
            return 0

    # ============ Internal methods ============

    def get_component_data(
        self, node: TreeNode, create: bool = False
    ) -> Optional[ComponentData]:
        """Get component data from node, optionally create if not exists."""
        if self.name not in node.component_data:
            if create:
                node.component_data[self.name] = ComponentData()
            else:
                return None
        return node.component_data[self.name]

    def is_locked(self, node: TreeNode) -> bool:
        """Check if node is locked."""
        data = self.get_component_data(node)
        return data is not None and data.lock_ref > 0

    def sanity_check_lru(self, tree):
        """Sanity check LRU list consistency."""
        pass


# ============ Concrete component implementations ============


class FullComponent(CacheComponent):
    """Full Attention KV Cache Component (Primary).

    Characteristics:
        - Always MAX priority (100) for both leaf and non-leaf nodes
        - Cannot be tombstone
        - Chain lock: locks node-to-root path
        - Values are split during node splitting
    """

    def __init__(self):
        super().__init__("full", is_primary=True)

    def get_priority(self, is_leaf: bool) -> int:
        """Always MAX priority."""
        return 100

    def get_lock_range(self, node: TreeNode, root: TreeNode) -> List[TreeNode]:
        """Chain lock: lock from node to root."""
        nodes = []
        while node != root:
            nodes.append(node)
            node = node.parent
        return nodes

    def on_insert_split_node(
        self, old_node: TreeNode, new_node: TreeNode, split_len: int
    ):
        """Split value between parent and child nodes."""
        old_data = self.get_component_data(old_node)
        if not old_data or old_data.is_tombstone():
            return

        new_data = self.get_component_data(new_node, create=True)
        new_data.lock_ref = old_data.lock_ref
        new_data.value = old_data.value[:split_len]
        old_data.value = old_data.value[split_len:]

        old_node.last_access_time = get_last_access_time()
        self.lru_list.remove_node(old_node)
        self.lru_list.insert_mru(new_node)
        self.lru_list.insert_mru(old_node)

    def prepare_component_value(self, req, is_finished: bool):
        """Return KV indices from request."""
        kv_indices = self.tree.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req._cache_len
        ]
        if self.tree.page_size != 1:
            page_aligned_len = (
                len(kv_indices) // self.tree.page_size * self.tree.page_size
            )
            return kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
        else:
            return kv_indices.to(dtype=torch.int64, copy=True)

    def sanity_check_lru(self, tree):
        """Sanity check full LRU list consistency."""
        import heapq

        # Collect all nodes (excluding root)
        nodes = []

        def _dfs(node):
            for child in node.children.values():
                nodes.append(child)
                _dfs(child)

        _dfs(tree.root)

        total_nodes = len(nodes)
        total_lru = len(self.lru_list.cache)

        heapq.heapify(nodes)
        assert (
            len(nodes) == total_lru
        ), f"Full: len(nodes)={len(nodes)} != total_lru={total_lru}"

        x_lru = self.lru_list._get_lru()
        while len(nodes):
            x = heapq.heappop(nodes)
            assert (
                x_lru is not None and x_lru.id in self.lru_list.cache
            ), f"Full LRU list error: x_lru is None or not in cache"
            assert (
                x == x_lru
            ), f"Full LRU list mismatch: x.id={x.id} != x_lru.id={x_lru.id}"

            data = self.get_component_data(x_lru)
            assert (
                data.lock_ref == 0
            ), f"Full: node should not be locked when idle, lock_ref={data.lock_ref}, id={x_lru.id}"
            x_lru = getattr(x_lru, self.lru_list.prv)

        lru_evictable_size = self.lru_list.sanity_check_evictable_size(self)
        assert (
            self.evictable_size == lru_evictable_size
        ), f"Full: evictable_size={self.evictable_size} != lru_evictable_size={lru_evictable_size}"


class MambaComponent(CacheComponent):
    """Mamba State Cache Component (Secondary).

    Characteristics:
        - Variable priority: MAX (100) for leaf, LOW (1) for non-leaf
        - Can be tombstone (parent nodes after splitting)
        - Node-local lock: locks only the node itself
        - Values cannot be split (state is indivisible)
    """

    def __init__(self):
        super().__init__("mamba", is_primary=False)

    def get_priority(self, is_leaf: bool) -> int:
        """Leaf: MAX priority, Non-leaf: LOW priority."""
        return 100 if is_leaf else 1

    def get_lock_range(self, node: TreeNode, root: TreeNode) -> List[TreeNode]:
        """Node-local lock: lock only if node has mamba value."""
        data = self.get_component_data(node)
        if data and not data.is_tombstone():
            return [node]
        return []

    def on_match_complete(self, matched_node: TreeNode, req):
        """Copy mamba state from tree to request space (COW)."""
        node_data = self.get_component_data(matched_node)
        if node_data is None or node_data.is_tombstone():
            return

        mamba_pool = self.tree.req_to_token_pool.mamba_pool
        src_index = node_data.value

        if req.mamba_pool_idx is None:
            dst_index = mamba_pool.alloc(1)

            if dst_index is None:
                self.tree.inc_lock_ref(matched_node)
                self.evict_cascade(1)
                dst_index = mamba_pool.alloc(1)
                self.tree.dec_lock_ref(matched_node)
                assert dst_index is not None, "Failed to allocate mamba cache"

            mamba_pool.copy_from(src_index, dst_index)
            req.mamba_pool_idx = dst_index[0]
        else:
            dst_index = req.mamba_pool_idx.unsqueeze(0)
            mamba_pool.copy_from(src_index, dst_index)

    def on_insert_split_node(
        self, old_node: TreeNode, new_node: TreeNode, split_len: int
    ):
        """Mamba state cannot be split."""
        old_data = self.get_component_data(old_node)
        if not old_data:
            return

        new_data = self.get_component_data(new_node, create=True)
        new_data.value = None
        new_data.lock_ref = 0

        old_node.last_access_time = get_last_access_time()

        if not old_data.is_tombstone():
            self.lru_list.remove_node(old_node)
            self.lru_list.insert_mru(old_node)

    def prepare_component_value(self, req, is_finished: bool):
        """Prepare mamba state from request."""
        enable_extra_buffer = self.tree.enable_mamba_extra_buffer

        # Use ping-pong buffer if enabled
        if enable_extra_buffer:
            buffer_idx = self.tree.req_to_token_pool.get_mamba_ping_pong_other_idx(
                req.mamba_next_track_idx
            )
            mamba_state = req.mamba_ping_pong_track_buffer[buffer_idx]
            return mamba_state.unsqueeze(-1).clone()

        # For finished request, reuse existing pool idx if available
        if is_finished and req.mamba_pool_idx is not None:
            return req.mamba_pool_idx.unsqueeze(-1).clone()

        # Otherwise fork from request's mamba state
        return self._fork_mamba_state(req)

    def _fork_mamba_state(self, req):
        """Fork mamba state from request, evict if pool is full."""
        mamba_pool = self.tree.req_to_token_pool.mamba_pool
        mamba_state = self.tree.req_to_token_pool.get_mamba_indices(
            req.req_pool_idx
        ).unsqueeze(-1)

        mamba_state_forked = mamba_pool.fork_from(mamba_state)
        if mamba_state_forked is None:
            self.evict_cascade(1)
            mamba_state_forked = mamba_pool.fork_from(mamba_state)
            assert mamba_state_forked is not None, "Failed to allocate mamba cache"

        return mamba_state_forked

    def after_request_cached(self, req, is_finished: bool, existed: bool):
        """Return ping-pong buffer to keep if needed."""
        enable_extra_buffer = self.tree.enable_mamba_extra_buffer

        if is_finished and enable_extra_buffer and not existed:
            return self.tree.req_to_token_pool.get_mamba_ping_pong_other_idx(
                req.mamba_next_track_idx
            )
        return None

    def sanity_check_lru(self, tree):
        """Sanity check mamba LRU list consistency."""
        import heapq

        # Collect all non-tombstone nodes (nodes with mamba value)
        nodes = []

        def _dfs(node):
            data = self.get_component_data(node)
            if data and not data.is_tombstone():
                nodes.append(node)
            for child in node.children.values():
                _dfs(child)

        _dfs(tree.root)

        total_nodes = len(nodes)
        total_lru = len(self.lru_list.cache)

        heapq.heapify(nodes)
        assert (
            len(nodes) == total_lru
        ), f"Mamba: len(nodes)={len(nodes)} != total_lru={total_lru}"

        x_lru = self.lru_list._get_lru()
        while len(nodes):
            x = heapq.heappop(nodes)
            assert (
                x_lru is not None and x_lru.id in self.lru_list.cache
            ), f"Mamba LRU list error: x_lru is None or not in cache"
            assert (
                x == x_lru
            ), f"Mamba LRU list mismatch: x.id={x.id} != x_lru.id={x_lru.id}"

            data = self.get_component_data(x_lru)
            assert (
                data.lock_ref == 0
            ), f"Mamba: node should not be locked when idle, lock_ref={data.lock_ref}, id={x_lru.id}"
            x_lru = getattr(x_lru, self.lru_list.prv)

        lru_evictable_size = self.lru_list.sanity_check_evictable_size(self)
        assert (
            self.evictable_size == lru_evictable_size
        ), f"Mamba: evictable_size={self.evictable_size} != lru_evictable_size={lru_evictable_size}"


class SWAComponent(CacheComponent):
    """SWA (Sliding Window Attention) Cache Component (Secondary).

    Characteristics:
        - Variable priority: MAX (100) for leaf, LOW (1) for non-leaf (same as Mamba)
        - Can be tombstone for non-leaf nodes
        - Lock range: sliding window size from last node toward root
        - Uses swa_uuid to mark lock boundary
    """

    def __init__(self, sliding_window_size: int):
        super().__init__("swa", is_primary=False)
        self.sliding_window_size = sliding_window_size

    def get_priority(self, is_leaf: bool) -> int:
        """Leaf nodes have MAX priority, non-leaf have LOW priority."""
        return 100 if is_leaf else 1

    def get_lock_range(self, node: TreeNode, root: TreeNode) -> List[TreeNode]:
        """Return nodes within sliding window range from node toward root.

        Unlike Mamba (only current node), SWA locks sliding_window_size tokens
        from the last node toward root.
        """
        nodes = []
        locked_size = 0

        while node != root:
            nodes.append(node)
            data = self.get_component_data(node)

            # Accumulate locked size (only count non-tombstone nodes)
            if data and not data.is_tombstone():
                locked_size += len(data.value)

            # Stop when reaching sliding window size
            if locked_size >= self.sliding_window_size:
                break

            node = node.parent

        return nodes
    
    def update_prefix_match_context(self, node: TreeNode, context: dict, init: bool = False) -> bool:
        """Update SWA match context during traversal.
        
        Tracks distance since last tombstone.
        """
        if init:
            # For path connected to root without tombstone
            context['swa_match_len_since_tombstone'] = float('inf')
            return True
        
        data = self.get_component_data(node)
        match_len = context.get('swa_match_len_since_tombstone', float('inf'))
        
        # Update distance based on whether node is tombstone
        if data and data.is_tombstone():
            # Reached tombstone: reset distance
            context['swa_match_len_since_tombstone'] = 0
        elif data and not data.is_tombstone():
            # Not tombstone: accumulate distance
            context['swa_match_len_since_tombstone'] = match_len + len(data.value)
        
        return True

    def check_prefix_match_constraints(self, node: TreeNode, context: dict, child: Optional[TreeNode] = None) -> bool:
        """Check if node satisfies SWA constraints given current context.

        Logic:
            - During traversal (child != None): 
              Accept if child is tombstone AND accumulated distance >= sliding_window_size
            - Final check (child == None): 
              Accept if accumulated distance >= sliding_window_size
        """
        match_len = context.get('swa_match_len_since_tombstone', float('inf'))
        
        if child is None:
            return match_len >= self.sliding_window_size
        else:
            child_data = self.get_component_data(child)
            child_is_tombstone = child_data is not None and child_data.is_tombstone()
            return child_is_tombstone and match_len >= self.sliding_window_size

    def on_insert_split_node(
        self, old_node: TreeNode, new_node: TreeNode, split_len: int
    ):
        """SWA value cannot be split (same as Mamba)."""
        pass

    def on_insert_leaf(self, node: TreeNode, value: torch.Tensor):
        """Insert SWA state into new leaf node."""
        pass

    def prepare_component_value(self, req, is_finished: bool):
        """Prepare SWA value from request."""
        pass

    def free_value(self, value_or_node, update_accounting: bool = True) -> int:
        """Free SWA value."""
        pass

    def sanity_check_lru(self, tree):
        """Sanity check SWA LRU list consistency."""
        pass