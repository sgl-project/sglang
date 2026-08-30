# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KV cache placement event recording.

Produces the ``BlockStored`` / ``BlockRemoved`` / ``AllBlocksCleared`` events
consumed by KV-aware routers (e.g. dynamo). A cache holds one recorder and calls
it; the recorder owns the queue and needs nothing back from its owner.
"""

from typing import Any, Optional

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithMetadata,
    StorageMedium,
)
from sglang.srt.mem_cache.utils import (
    compute_node_event_hash_values,
    compute_node_hash_values,
    hash_str_to_int64,
)


class KVCacheEventRecorder:
    """Collects KV placement events for one cache.

    ``enabled=False`` makes every ``record_*`` call a no-op and ``take`` return an
    empty list, so callers never have to guard.
    """

    def __init__(self, *, enabled: bool, page_size: int):
        self.enabled = enabled
        self.page_size = page_size
        self._queue: list = []

    def enqueue(self, event) -> None:
        """Append an event, coalescing it with a compatible queue tail.

        KV event batches already support multiple block hashes.  Combining them
        here avoids emitting one event per page while preserving ordering and
        the parent-linked store chains consumers use to rebuild the cache tree.
        """
        if self._queue:
            tail = self._queue[-1]

            if isinstance(tail, BlockRemoved) and isinstance(event, BlockRemoved):
                if tail.medium == event.medium:
                    tail.block_hashes.extend(event.block_hashes)
                    return

            elif isinstance(tail, BlockStored) and isinstance(event, BlockStored):
                tail_metadata = (
                    tail.metadata if isinstance(tail, BlockStoredWithMetadata) else None
                )
                event_metadata = (
                    event.metadata
                    if isinstance(event, BlockStoredWithMetadata)
                    else None
                )
                if (
                    tail.medium == event.medium
                    and tail.lora_id == event.lora_id
                    and tail.block_size == event.block_size
                    and tail_metadata == event_metadata
                    and tail.block_hashes
                    and event.parent_block_hash == tail.block_hashes[-1]
                ):
                    tail.block_hashes.extend(event.block_hashes)
                    tail.token_ids.extend(event.token_ids)
                    return

        self._queue.append(event)

    def _node_event_hash_values(self, node: Any) -> list:
        """Hash values to publish for ``node``, computing them if not yet set."""
        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)
        if node.key.cache_salt is not None:
            return compute_node_event_hash_values(node, self.page_size)
        return node.hash_value

    def _parent_block_hash(self, node: Any) -> Optional[int]:
        """The hash the first page of ``node`` links back to.

        ``None`` when the parent is the tree root: a root carries an empty
        ``hash_value`` and no event hash, so it contributes no link. Every other
        node on the path has a parent, which is what distinguishes the two.
        """
        parent = node.parent
        if parent is None or parent.parent is None:
            return None
        if node.key.cache_salt is not None:
            parent_hash_values = parent.event_hash_value
            assert parent_hash_values is not None
        else:
            parent_hash_values = parent.hash_value
        if not parent_hash_values:
            return None
        return hash_str_to_int64(parent_hash_values[-1])

    def record_store(self, node: Any, medium=None) -> None:
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        event_hash_values = self._node_event_hash_values(node)
        parent_block_hash = self._parent_block_hash(node)

        page_index = 0
        logical_len = len(node.key)
        is_bigram = node.key.is_bigram
        raw = node.key.token_ids
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue
            # Preserve historical event payload: bigram pages expose tuples.
            if is_bigram:
                page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
            else:
                page_tokens = list(raw[start:end])

            block_hash = hash_str_to_int64(event_hash_values[page_index])

            event_args = {
                "block_hashes": [block_hash],
                "parent_block_hash": parent_block_hash,
                "token_ids": page_tokens,
                "block_size": len(page_tokens),
                "lora_id": None,
                "medium": medium,
            }
            if node.key.cache_salt is None:
                event = BlockStored(**event_args)
            else:
                event = BlockStoredWithMetadata(
                    **event_args,
                    metadata=BlockStoredMetadata(cache_salt=node.key.cache_salt),
                )
            self.enqueue(event)

            parent_block_hash = block_hash
            page_index += 1

    def record_remove(self, node: Any, medium=None) -> None:
        # One BlockRemoved per radix node.
        # ``medium`` defaults to StorageMedium.GPU but callers may override for
        # lower-tier removals (e.g. StorageMedium.CPU when evicting from host).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        # Hash values must match what was stored.
        event_hash_values = self._node_event_hash_values(node)

        block_hashes = []
        logical_len = len(node.key)
        page_index = 0
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue

            block_hashes.append(hash_str_to_int64(event_hash_values[page_index]))
            page_index += 1

        if block_hashes:
            self.enqueue(BlockRemoved(block_hashes=block_hashes, medium=medium))

    def record_all_cleared(self) -> None:
        if not self.enabled:
            return
        self.enqueue(AllBlocksCleared())

    def take(self) -> list:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enabled:
            return []
        events = self._queue
        self._queue = []
        return events
