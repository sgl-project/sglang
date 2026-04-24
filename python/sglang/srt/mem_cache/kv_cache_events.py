from __future__ import annotations

from typing import Any, List, Optional

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64


def compute_node_hash_values(node: Any, page_size: int) -> List[str]:
    hash_values = []

    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class KVCacheEventMixin:
    def _init_kv_cache_events(self, enable_kv_cache_events: bool) -> None:
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []

    def _ensure_hash_value(self, node: Optional[Any]) -> None:
        if node is None or node.hash_value is not None:
            return
        self._ensure_hash_value(node.parent)
        node.hash_value = compute_node_hash_values(node, self.page_size)

    def _record_store_event(self, node: Any) -> None:
        if not self.enable_kv_cache_events:
            return

        self._ensure_hash_value(node.parent)
        self._ensure_hash_value(node)

        parent_block_hash = None
        if node.parent is not None and node.parent != self.root_node:
            if node.parent.hash_value is not None and len(node.parent.hash_value) > 0:
                parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

        page_index = 0
        for start in range(0, len(node.key), self.page_size):
            page_tokens = node.key.token_ids[start : start + self.page_size]
            if not page_tokens:
                continue

            block_hash = hash_str_to_int64(node.hash_value[page_index])
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=page_tokens,
                    block_size=len(page_tokens),
                    lora_id=None,
                )
            )

            parent_block_hash = block_hash
            page_index += 1

    def _record_remove_event(self, node: Any) -> None:
        if not self.enable_kv_cache_events:
            return

        self._ensure_hash_value(node)

        page_index = 0
        for start in range(0, len(node.key), self.page_size):
            page_tokens = node.key.token_ids[start : start + self.page_size]
            if not page_tokens:
                continue

            block_hash = hash_str_to_int64(node.hash_value[page_index])
            self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))
            page_index += 1

    def _record_all_cleared_event(self) -> None:
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
