from __future__ import annotations

import threading

from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.utils import (
    block_hash_aliases,
    compute_node_hash_values,
    hash_str_to_int64,
)


class HiCacheHostBlockIndex:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.lock = threading.RLock()
        self.block_index: dict[int, tuple[TreeNode, int, str]] = {}

    def clear(self) -> None:
        with self.lock:
            self.block_index.clear()

    def index_node(self, node: TreeNode) -> None:
        if node.host_value is None:
            self.drop_node(node)
            return
        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)

        num_pages = min(len(node.hash_value), len(node.host_value) // self.page_size)
        with self.lock:
            self.drop_node(node, locked=True)
            for page_idx in range(num_pages):
                hash_value = node.hash_value[page_idx]
                block_hash = hash_str_to_int64(hash_value)
                for alias in block_hash_aliases(block_hash):
                    self.block_index[alias] = (node, page_idx, hash_value)

    def drop_node(self, node: TreeNode, *, locked: bool = False) -> None:
        def drop() -> None:
            stale = [
                block_hash
                for block_hash, entry in self.block_index.items()
                if entry[0] is node
            ]
            for block_hash in stale:
                self.block_index.pop(block_hash, None)

        if locked:
            drop()
        else:
            with self.lock:
                drop()

    def lookup(
        self, wanted_hashes: set[int], *, protect: bool = False
    ) -> (
        dict[int, tuple[TreeNode, int, str]]
        | tuple[dict[int, tuple[TreeNode, int, str]], list[TreeNode]]
    ):
        matches: dict[int, tuple[TreeNode, int, str]] = {}
        protected_nodes: list[TreeNode] = []
        protected_ids: set[int] = set()
        wanted_aliases: set[int] = set()
        for block_hash in wanted_hashes:
            wanted_aliases.update(block_hash_aliases(block_hash))

        stale_aliases = []
        with self.lock:
            for alias in wanted_aliases:
                entry = self.block_index.get(alias)
                if entry is None:
                    continue
                node, page_idx, hash_value = entry
                valid = (
                    node.host_value is not None
                    and node.hash_value is not None
                    and 0 <= page_idx < len(node.hash_value)
                    and page_idx * self.page_size < len(node.host_value)
                    and node.hash_value[page_idx] == hash_value
                )
                if valid:
                    matches[alias] = entry
                    if protect and node.id not in protected_ids:
                        node.protect_host()
                        protected_nodes.append(node)
                        protected_ids.add(node.id)
                else:
                    stale_aliases.append(alias)

            for alias in stale_aliases:
                self.block_index.pop(alias, None)
        if protect:
            return matches, protected_nodes
        return matches
