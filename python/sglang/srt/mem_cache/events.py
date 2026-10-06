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

import weakref
from collections import Counter
from typing import Any, Optional

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.mem_cache.utils import (
    compute_node_event_hash_values,
    compute_node_hash_values,
    hash_str_to_int64,
    kv_event_lora_seed,
    namespace_event_block_hash,
)


class KvEventLoraNames:
    """LoRA adapter name per cache namespace (``extra_key``), for KV events.

    A name is kept while a request of its namespace is alive or a block
    published under it is not yet removed, so removals hash like their stores.
    """

    def __init__(self):
        self._names: dict[str, str] = {}
        self._requests: dict[str, weakref.WeakSet] = {}
        self._published_blocks: Counter = Counter()
        # Namespaces that may have lost their last request or block.
        self._prune_candidates: set[str] = set()
        self._prune_at = 64

    def register(self, req: Any, lora_name: Optional[str]) -> None:
        """Name ``req``'s namespace; ``req`` needs ``extra_key`` and ``lora_id``."""
        if req.lora_id is None or not lora_name:
            return
        self._names[req.extra_key] = lora_name
        self._requests.setdefault(req.extra_key, weakref.WeakSet()).add(req)
        self._prune_candidates.add(req.extra_key)
        # Caches that never take events (non-publishing ranks, no radix cache)
        # rely on this amortized prune.
        if len(self._prune_candidates) >= self._prune_at:
            self.prune()
            self._prune_at = 2 * len(self._prune_candidates) + 64

    def get(self, extra_key: Optional[str]) -> Optional[str]:
        return self._names.get(extra_key)

    def count_published(self, extra_key: Optional[str], num_blocks: int) -> None:
        if extra_key in self._names:
            self._published_blocks[extra_key] += num_blocks
            if self._published_blocks[extra_key] <= 0:
                self._prune_candidates.add(extra_key)

    def clear_published(self) -> None:
        self._published_blocks.clear()
        self._prune_candidates.update(self._names)

    def prune(self) -> None:
        """Forget namespaces with no live request and no published block."""
        for key in list(self._prune_candidates):
            if self._requests[key]:
                continue
            self._prune_candidates.discard(key)
            if self._published_blocks[key] <= 0:
                del self._names[key], self._requests[key]
                self._published_blocks.pop(key, None)


class KVCacheEventRecorder:
    """Collects KV placement events for one cache.

    ``enabled=False`` makes every ``record_*`` call a no-op and ``take`` return an
    empty list, so callers never have to guard.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        page_size: int,
        lora_names: Optional[KvEventLoraNames] = None,
    ):
        self.enabled = enabled
        self.page_size = page_size
        self.lora_names = lora_names if lora_names is not None else KvEventLoraNames()
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
                if (
                    tail.medium == event.medium
                    and tail.lora_id == event.lora_id
                    and tail.block_size == event.block_size
                    and tail.cache_salt == event.cache_salt
                    and tail.session_id == event.session_id
                    and tail.lora_name == event.lora_name
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
        if node.key.extra_key is None and node.key.cache_salt is None:
            return node.hash_value
        return compute_node_event_hash_values(node, self.page_size)

    def _parent_block_hash(self, node: Any, seed: Optional[bytes]) -> Optional[int]:
        """The hash the first page of ``node`` links back to.

        ``None`` when the parent is the tree root: a root carries an empty
        ``hash_value`` and no event hash, so it contributes no link. Every other
        node on the path has a parent, which is what distinguishes the two.
        """
        parent = node.parent
        if parent is None or parent.parent is None:
            return None
        if node.key.extra_key is not None or node.key.cache_salt is not None:
            parent_hash_values = parent.event_hash_value
            assert parent_hash_values is not None
        else:
            parent_hash_values = parent.hash_value
        if not parent_hash_values:
            return None
        return namespace_event_block_hash(
            hash_str_to_int64(parent_hash_values[-1]), seed
        )

    def record_store(
        self, node: Any, medium=None, *, session_id: Optional[str] = None
    ) -> None:
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        lora_name = self.lora_names.get(node.key.extra_key)
        seed = kv_event_lora_seed(lora_name)
        event_hash_values = self._node_event_hash_values(node)
        parent_block_hash = self._parent_block_hash(node, seed)

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

            block_hash = namespace_event_block_hash(
                hash_str_to_int64(event_hash_values[page_index]), seed
            )

            self.enqueue(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=page_tokens,
                    block_size=len(page_tokens),
                    lora_id=None,
                    medium=medium,
                    cache_salt=node.key.cache_salt,
                    session_id=session_id,
                    lora_name=lora_name,
                )
            )

            parent_block_hash = block_hash
            page_index += 1
        self.lora_names.count_published(node.key.extra_key, page_index)

    def record_remove(self, node: Any, medium=None) -> None:
        # One BlockRemoved per radix node.
        # ``medium`` defaults to StorageMedium.GPU but callers may override for
        # lower-tier removals (e.g. StorageMedium.CPU when evicting from host).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        # Hash values must match what was stored.
        seed = kv_event_lora_seed(self.lora_names.get(node.key.extra_key))
        event_hash_values = self._node_event_hash_values(node)

        block_hashes = []
        logical_len = len(node.key)
        page_index = 0
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue

            block_hashes.append(
                namespace_event_block_hash(
                    hash_str_to_int64(event_hash_values[page_index]), seed
                )
            )
            page_index += 1

        if block_hashes:
            self.enqueue(BlockRemoved(block_hashes=block_hashes, medium=medium))
            self.lora_names.count_published(node.key.extra_key, -len(block_hashes))

    def record_all_cleared(self) -> None:
        if not self.enabled:
            return
        self.enqueue(AllBlocksCleared())
        self.lora_names.clear_published()

    def take(self) -> list:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        self.lora_names.prune()
        if not self.enabled:
            return []
        events = self._queue
        self._queue = []
        return events
