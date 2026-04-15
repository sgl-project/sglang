"""Pure-Python radix trie for MLX prefix caching.

Maps token-ID sequences to KV-pool slot IDs with LRU eviction.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TrieNode:
    """A node in the radix trie."""

    __slots__ = (
        "children",
        "parent",
        "key",
        "value",
        "lock_ref",
        "last_access_time",
    )

    def __init__(self):
        self.children: dict[int, TrieNode] = {}
        self.parent: Optional[TrieNode] = None
        self.key: list[int] = []  # token IDs on this edge
        self.value: list[int] = []  # pool slot IDs (1:1 with key)
        self.lock_ref: int = 0  # >0 means in-use, not evictable
        self.last_access_time: float = time.monotonic()


class MatchResult:
    """Result of a prefix match."""

    __slots__ = ("slot_ids", "prefix_len", "last_node")

    def __init__(
        self,
        slot_ids: list[int],
        prefix_len: int,
        last_node: TrieNode,
    ):
        self.slot_ids = slot_ids
        self.prefix_len = prefix_len
        self.last_node = last_node


class MlxRadixTrie:
    """Radix trie mapping token-ID prefixes to KV-pool slot IDs."""

    def __init__(self, pool_capacity: int):
        self.pool_capacity = pool_capacity
        self.root = TrieNode()
        self.root.lock_ref = 1  # root is never evicted
        self._evictable_size = 0

    def match_prefix(self, token_ids: list[int]) -> MatchResult:
        """Find the longest cached prefix of *token_ids*."""
        if not token_ids:
            return MatchResult([], 0, self.root)

        matched_slots: list[int] = []
        node = self.root
        pos = 0
        now = time.monotonic()
        n_tokens = len(token_ids)

        while pos < n_tokens:
            first_token = token_ids[pos]
            child = node.children.get(first_token)
            if child is None:
                break

            # Match as many tokens of the child's key as possible
            key = child.key
            key_len = len(key)
            match_len = 0
            remaining = n_tokens - pos
            limit = min(key_len, remaining)
            while match_len < limit:
                if token_ids[pos + match_len] != key[match_len]:
                    break
                match_len += 1

            if match_len == key_len:
                matched_slots.extend(child.value)
                pos += key_len
                child.last_access_time = now
                node = child
            else:
                if match_len > 0:
                    self._split_node(child, match_len)
                    new_child = node.children[first_token]
                    matched_slots.extend(new_child.value)
                    pos += match_len
                    new_child.last_access_time = now
                    node = new_child
                break

        return MatchResult(matched_slots, pos, node)

    def insert(self, token_ids: list[int], slot_ids: list[int]) -> int:
        """Insert token→slot mapping; returns already-cached prefix length."""
        assert len(token_ids) == len(slot_ids)
        if not token_ids:
            return 0
        now = time.monotonic()
        return self._insert_helper(self.root, token_ids, slot_ids, 0, now)

    def evict(self, num_slots: int) -> list[int]:
        """Evict LRU leaves until *num_slots* are freed; returns freed IDs."""
        freed: list[int] = []
        while len(freed) < num_slots and self._evictable_size > 0:
            leaf = self._find_lru_leaf()
            if leaf is None:
                break
            freed.extend(leaf.value)
            self._evictable_size -= len(leaf.value)
            self._remove_leaf(leaf)
        return freed

    @property
    def evictable_size(self) -> int:
        return self._evictable_size

    def reset(self) -> list[int]:
        """Clear the entire trie.  Returns all slot IDs that were stored."""
        all_slots = self._collect_all_slots(self.root)
        self.root = TrieNode()
        self.root.lock_ref = 1
        self._evictable_size = 0
        return all_slots

    def inc_ref(self, node: TrieNode) -> None:
        """Increment lock reference on *node* (mark in-use)."""
        delta = 1 if node.lock_ref == 0 else 0
        node.lock_ref += 1
        if delta and node is not self.root and node.value:
            self._evictable_size -= len(node.value)

    def dec_ref(self, node: TrieNode) -> None:
        """Decrement lock reference on *node*."""
        if node.lock_ref <= 0:
            return
        node.lock_ref -= 1
        if node.lock_ref == 0 and node is not self.root and node.value:
            self._evictable_size += len(node.value)

    def _split_node(self, node: TrieNode, split_pos: int) -> None:
        """Split *node* into prefix[:split_pos] → suffix[split_pos:]."""
        parent = node.parent
        assert parent is not None

        new_node = TrieNode()
        new_node.key = node.key[:split_pos]
        new_node.value = node.value[:split_pos]
        new_node.parent = parent
        new_node.lock_ref = node.lock_ref
        new_node.last_access_time = node.last_access_time

        node.key = node.key[split_pos:]
        node.value = node.value[split_pos:]
        node.parent = new_node

        first_token_old = new_node.key[0]
        parent.children[first_token_old] = new_node
        new_node.children[node.key[0]] = node

    def _insert_helper(
        self,
        node: TrieNode,
        token_ids: list[int],
        slot_ids: list[int],
        pos: int,
        now: float,
    ) -> int:
        """Recursive insert; returns length of already-existing prefix."""
        if pos >= len(token_ids):
            return pos

        first_token = token_ids[pos]
        child = node.children.get(first_token)

        if child is None:
            new_leaf = TrieNode()
            new_leaf.key = token_ids[pos:]
            new_leaf.value = slot_ids[pos:]
            new_leaf.parent = node
            new_leaf.last_access_time = now
            node.children[first_token] = new_leaf
            self._evictable_size += len(new_leaf.value)
            return pos
        key = child.key
        key_len = len(key)
        remaining = len(token_ids) - pos
        match_len = 0
        while match_len < key_len and match_len < remaining:
            if token_ids[pos + match_len] != key[match_len]:
                break
            match_len += 1

        if match_len < key_len:
            self._split_node(child, match_len)
            new_parent = node.children[first_token]
            prefix_len = pos + match_len

            rest_tokens = token_ids[prefix_len:]
            rest_slots = slot_ids[prefix_len:]
            if rest_tokens:
                new_leaf = TrieNode()
                new_leaf.key = rest_tokens
                new_leaf.value = rest_slots
                new_leaf.parent = new_parent
                new_leaf.last_access_time = now
                new_parent.children[rest_tokens[0]] = new_leaf
                self._evictable_size += len(new_leaf.value)
            return prefix_len
        else:
            child.last_access_time = now
            return self._insert_helper(child, token_ids, slot_ids, pos + key_len, now)

    def _find_lru_leaf(self) -> Optional[TrieNode]:
        """Find the least-recently-used evictable leaf."""
        best: Optional[TrieNode] = None
        best_time = float("inf")
        stack = list(self.root.children.values())
        while stack:
            n = stack.pop()
            if not n.children and n.lock_ref == 0:
                if n.last_access_time < best_time:
                    best_time = n.last_access_time
                    best = n
            else:
                for c in n.children.values():
                    stack.append(c)
        return best

    def _remove_leaf(self, leaf: TrieNode) -> None:
        """Remove a leaf and merge its parent if only one child remains."""
        parent = leaf.parent
        if parent is None:
            return
        to_remove = None
        for token, child in parent.children.items():
            if child is leaf:
                to_remove = token
                break
        if to_remove is not None:
            del parent.children[to_remove]

        # Merge parent with its only remaining child (if not root)
        if (
            parent is not self.root
            and len(parent.children) == 1
            and parent.lock_ref == 0
        ):
            only_child = next(iter(parent.children.values()))
            parent.key = parent.key + only_child.key
            parent.value = parent.value + only_child.value
            parent.children = only_child.children
            parent.last_access_time = only_child.last_access_time
            parent.lock_ref = only_child.lock_ref
            for grandchild in parent.children.values():
                grandchild.parent = parent

    def _collect_all_slots(self, node: TrieNode) -> list[int]:
        """Collect all slot IDs stored in the subtree rooted at *node*."""
        slots: list[int] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n is not self.root:
                slots.extend(n.value)
            for c in n.children.values():
                stack.append(c)
        return slots
