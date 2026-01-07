"""
LRU List implementation for cache eviction.
"""

from typing import Optional

from .tree_node import TreeNode


class LRUList:
    """Doubly-linked list for LRU cache management."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        # Attribute names for prev/next pointers
        self.prv = f"{component_name}_prev"
        self.nxt = f"{component_name}_next"

        # Dummy head and tail
        self.head = TreeNode()
        self.tail = TreeNode()
        setattr(self.head, self.nxt, self.tail)
        setattr(self.tail, self.prv, self.head)

        self.cache = {}  # node_id -> node

    def _add_node(self, node: TreeNode):
        """Add node right after head (most recently used)."""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node: TreeNode, new_node: TreeNode):
        """Add new_node right after old_node."""
        setattr(new_node, self.prv, old_node)
        setattr(new_node, self.nxt, getattr(old_node, self.nxt))
        setattr(getattr(old_node, self.nxt), self.prv, new_node)
        setattr(old_node, self.nxt, new_node)

    def _remove_node(self, node: TreeNode):
        """Remove node from linked list."""
        setattr(getattr(node, self.prv), self.nxt, getattr(node, self.nxt))
        setattr(getattr(node, self.nxt), self.prv, getattr(node, self.prv))

    def _get_lru(self) -> Optional[TreeNode]:
        """Get the least recently used node."""
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def insert_mru(self, node: TreeNode):
        """Insert a new node as most recently used."""
        assert node.id not in self.cache, f"Node {node.id} already in LRU list"
        self.cache[node.id] = node
        self._add_node(node)

    def reset_node_mru(self, node: TreeNode):
        """Move existing node to most recently used position."""
        assert node.id in self.cache, f"Node {node.id} not in LRU list"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(
        self, node: TreeNode, root: TreeNode, component=None
    ):
        """Move node and parents to MRU position, with child more recent than parent."""
        prev_node = self.head
        while node != root:
            if node.id in self.cache:
                if component:
                    data = node.component_data.get(component.name)
                    if not data or data.is_tombstone():
                        node = node.parent
                        continue

                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def remove_node(self, node: TreeNode):
        """Remove node from LRU list."""
        assert node.id in self.cache, f"Node {node.id} not in LRU list"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self, lock_checker) -> Optional[TreeNode]:
        """Get LRU node that is not locked."""
        x = getattr(self.tail, self.prv)
        while x != self.head:
            if not lock_checker(x):
                return x
            x = getattr(x, self.prv)
        return None

    def get_leaf_lru_no_lock(self, lock_checker) -> Optional[TreeNode]:
        """Get LRU leaf node that is not locked."""
        x = getattr(self.tail, self.prv)
        while x != self.head:
            if not lock_checker(x) and len(x.children) == 0:
                return x
            x = getattr(x, self.prv)
        return None

    def get_prev_no_lock(self, node: TreeNode, lock_checker) -> Optional[TreeNode]:
        """Get the previous (more recently used) node that is not locked."""
        assert node.id in self.cache, f"Node {node.id} not in LRU list"
        x = getattr(node, self.prv)
        while x != self.head and lock_checker(x):
            x = getattr(x, self.prv)
        return x if x != self.head else None

    def get_prev_leaf_no_lock(self, node: TreeNode, lock_checker) -> Optional[TreeNode]:
        """Get the previous (more recently used) leaf node that is not locked."""
        assert node.id in self.cache, f"Node {node.id} not in LRU list"
        x = getattr(node, self.prv)
        while x != self.head and (lock_checker(x) or len(x.children) > 0):
            x = getattr(x, self.prv)
        return x if x != self.head else None

    def in_list(self, node: Optional[TreeNode]) -> bool:
        """Check if node is in the LRU list."""
        return node is not None and node.id in self.cache

    def sanity_check_evictable_size(self, component) -> int:
        """Calculate evictable size by iterating LRU list (expensive, for debug)."""
        node = self.get_lru_no_lock(component.is_locked)
        evictable_size = 0
        while self.in_list(node):
            data = component.get_component_data(node)
            if data and not data.is_tombstone():
                evictable_size += len(data.value)
            node = self.get_prev_no_lock(node, component.is_locked)
        return evictable_size
