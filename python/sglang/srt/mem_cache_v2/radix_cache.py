import heapq
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.srt.mem_cache_v2.cache_index import CacheIndex, MatchResult


@dataclass
class TreeNode:
    key: tuple[int, ...]  # token sequence
    value: torch.Tensor  # device token indices
    parent: "TreeNode | None"
    children: dict[Any, "TreeNode"] = field(default_factory=dict)
    is_ready: bool = True  # the node is ready to be used for computation
    last_access_time: float = field(default_factory=time.monotonic)
    ref_count: int = 0

    def __lt__(self, other: "TreeNode"):
        # monkey patch this for various eviction strategies like LRU, LFU, etc.
        return self.last_access_time < other.last_access_time


# For lora-support, aka. multiple-namespace, we can have multiple tree instance and distinguish them at root node.
class RadixCache(CacheIndex):
    def __init__(self, page_size: int):
        self.root = TreeNode(
            key=tuple(), value=torch.empty(0, dtype=torch.int32), parent=None
        )
        self.page_size = page_size
        self._evictable_size = 0
        self._protected_size = 0

    def _get_child_key(self, key: tuple[int, ...]):
        return key[: self.page_size]

    def _common_len(self, a: tuple[int, ...], b: tuple[int, ...]):
        num_pages = min(len(a), len(b)) // self.page_size
        for i in range(num_pages):
            start = i * self.page_size
            if a[start : start + self.page_size] != b[start : start + self.page_size]:
                return start
        return num_pages * self.page_size

    def match_prefix(self, key: tuple[int, ...]):
        matched_nodes: list[TreeNode] = []

        node = self.root
        child_key = self._get_child_key(key)
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self._common_len(key, child.key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child, prefix_len)
                matched_nodes.append(new_node)
                node = new_node
                break
            else:
                matched_nodes.append(child)
                node = child
                key = key[prefix_len:]
                child_key = self._get_child_key(key)

        if matched_nodes:
            values = torch.cat([node.value for node in matched_nodes])
        else:
            values = torch.empty(0, dtype=torch.int32)

        return MatchResult(
            matched_indices=values,
            allocation_key=node,
        )

    def _split_node(self, node: TreeNode, split_len: int):
        new_node = TreeNode(
            key=node.key[:split_len],
            value=node.value[:split_len],
            parent=node.parent,
            children={self._get_child_key(node.key[split_len:]): node},
        )
        node.parent = new_node
        node.key = node.key[split_len:]
        node.value = node.value[split_len:]
        assert new_node.parent
        new_node.parent.children[self._get_child_key(new_node.key)] = new_node
        return new_node

    def insert(
        self, key: tuple[int, ...], value: torch.Tensor, allocation_key: TreeNode
    ):
        assert len(value) == len(key), "Value and key must have the same length"

        match_result = self.match_prefix(key)
        last_node: TreeNode = match_result.allocation_key
        matched_indices = match_result.matched_indices
        matched_len = len(matched_indices)

        # lock the nodes between allocation_key and last_node
        current = last_node
        while current != allocation_key:
            assert current, "New node is above the allocation key"
            current.ref_count += 1
            current = current.parent

        if len(value) > matched_len:
            new_node = TreeNode(
                key=key[matched_len:],
                value=value[matched_len:],
                parent=last_node,
                ref_count=1,
            )
            last_node.children[self._get_child_key(key[matched_len:])] = new_node
            last_node = new_node

        return last_node, matched_indices

    def _collect_leaves(self) -> list[TreeNode]:
        def is_leaf(node: TreeNode) -> bool:
            return len(node.children) == 0 and len(node.value) > 0

        leaves: list[TreeNode] = []
        stack: list[TreeNode] = [self.root]
        while stack:
            node = stack.pop()
            if is_leaf(node):
                leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def evict(self, num_tokens: int):
        evicted: list[TreeNode] = []
        num_evicted = 0
        candidates = [n for n in self._collect_leaves() if n.ref_count == 0]
        heapq.heapify(candidates)

        while num_evicted < num_tokens and candidates:
            node = heapq.heappop(candidates)
            evicted.append(node)
            num_evicted += len(node.value)

            assert node.parent, "Trying to evict root node"
            parent = node.parent
            parent.children.pop(self._get_child_key(node.key))
            if (
                len(parent.children) == 0
                and parent != self.root
                and parent.ref_count == 0
            ):
                heapq.heappush(candidates, parent)

        if evicted:
            return torch.cat([node.value for node in evicted])
        return torch.empty(0, dtype=torch.int32)

    def allocate(self, allocation_key: TreeNode):
        not_ready = []

        current = allocation_key
        while current:
            if current.ref_count == 0:
                self._evictable_size -= len(current.value)
                self._protected_size += len(current.value)
            if not current.is_ready:
                not_ready.append(current)
            current.ref_count += 1
            current = current.parent

        # return the not ready indices
        if not_ready:
            return torch.cat([node.value for node in not_ready])
        return torch.empty(0, dtype=torch.int32)

    def free(self, allocation_key: TreeNode):
        num_unlocked = 0
        current = allocation_key
        while current:
            current.ref_count -= 1
            if current.ref_count == 0:
                self._evictable_size += len(current.value)
                self._protected_size -= len(current.value)
            num_unlocked += len(current.value)
            current = current.parent
        return num_unlocked
