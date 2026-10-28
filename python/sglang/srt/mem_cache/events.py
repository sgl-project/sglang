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

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Optional

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
    kv_event_namespace_seed,
    namespaced_block_hash,
)
from sglang.srt.mem_cache.utils import (
    compute_node_event_hash_values,
    compute_node_hash_values,
    hash_str_to_int64,
)

if TYPE_CHECKING:
    from sglang.srt.lora.lora_registry import LoRARef

logger = logging.getLogger(__name__)

# Length of LoRARef.lora_id, a uuid4 or uuid5 hex string. Req adds the id to the
# end of extra_key, so the last 32 characters of extra_key are the id.
_LORA_ID_LEN = 32


class LoRANameTable:
    """Find the LoRA adapter name from the lora_id at the end of extra_key.

    Req adds lora_id to the end of extra_key, so a tree node has only the id.
    The table keeps all entries. A lora_id is not used again, and nodes of an
    unloaded adapter can publish stores until eviction.
    """

    def __init__(self):
        self._names: dict[str, str] = {}

    @classmethod
    def from_lora_refs(cls, lora_refs: Optional[Iterable[LoRARef]]) -> LoRANameTable:
        table = cls()
        for lora_ref in lora_refs or ():
            table.register(lora_id=lora_ref.lora_id, lora_name=lora_ref.lora_name)
        return table

    def register(self, *, lora_id: str, lora_name: str) -> None:
        self._names[lora_id] = lora_name

    def get(self, extra_key: Optional[str]) -> Optional[str]:
        return self.resolve(extra_key)

    def resolve(self, extra_key: Optional[str]) -> Optional[str]:
        if not extra_key:
            return None
        return self._names.get(extra_key[-_LORA_ID_LEN:])


class KvEventLoraNames(LoRANameTable):
    """Public names of selected LoRA namespaces, kept for the publisher lifetime.

    Write-back can publish a host store after the last GPU removal and after
    its request dies, so request or published-block counts cannot bound this.
    """

    def __init__(self):
        # Adapter-id registrations keep the existing Dynamo load-time lookup.
        self._adapter_names: dict[str, str] = {}
        self._names: dict[str, str] = {}

    def register(
        self,
        req: Any = None,
        lora_name: Optional[str] = None,
        *,
        lora_id: Optional[str] = None,
    ) -> None:
        """Name ``req``'s namespace; ``req`` needs ``extra_key`` and ``lora_id``."""
        if lora_id is not None:
            self._adapter_names[lora_id] = lora_name
            return
        if req.lora_id is None:
            return
        # Older IPC senders omit the optional name; admission still resolves an ID.
        lora_name = lora_name or self._adapter_names.get(req.lora_id)
        if not lora_name:
            return
        self._names[req.extra_key] = lora_name

    def get(self, extra_key: Optional[str]) -> Optional[str]:
        return self._names.get(extra_key)

    def resolve(self, extra_key: Optional[str]) -> Optional[str]:
        if not extra_key:
            return None
        return self._names.get(extra_key) or self._adapter_names.get(
            extra_key[-_LORA_ID_LEN:]
        )


# schedule_batch.MM_PAD_SHIFT_VALUE; schedule_batch imports this module
# (through base_prefix_cache), so it cannot be imported here.
_MM_PAD_SHIFT_VALUE = 1_000_000


class MmRoutingLabels:
    """Pad value -> caller routing label, first label wins, LRU-bounded.

    Pad values derive from SGLang's own item identity, so every token of an
    item maps to the label of the first labeled request that brought it in.
    Entries can expire while blocks stay cached; their later events then carry
    no span and the router treats those tokens as unmapped (fail closed).

    Known limitation: once an entry is evicted, a later request can register
    the same pad value under a different label. Dynamo's id map already keeps
    the first label for any block it has indexed, so a relabeled re-store is
    counted as a conflict and dropped; the original routing continues to work.
    The only gap is an *unindexed* block that is re-stored after eviction:
    it is indexed under the new label. Both labels identify the same image, so
    routing to either worker yields a cache hit.

    Only the "dynamo" KV event format reports labels, so registration is a
    no-op until an enabled recorder using that format turns it on.
    Tensor-parallel ranks that do not publish events, and the default format,
    never fill it.

    Memory: the default capacity of 2**20 entries uses roughly 100–150 MiB RSS
    in CPython (key int objects + OrderedDict overhead + label strings).  With
    Dynamo's 16-character hex labels the measured cost is ~104 MiB; with the
    128-byte maximum it is ~180 MiB.
    """

    def __init__(self, capacity: int = 1 << 20):
        self._labels: OrderedDict[int, str] = OrderedDict()
        self._capacity = capacity
        self.enabled = False

    def register(self, pad_values, label: str) -> None:
        for value in pad_values:
            if value in self._labels:
                self._labels.move_to_end(value)
                continue
            self._labels[value] = label
            if len(self._labels) > self._capacity:
                self._labels.popitem(last=False)

    def get(self, value: int) -> Optional[str]:
        return self._labels.get(value)

    def __bool__(self) -> bool:
        return bool(self._labels)


MM_ROUTING_LABELS = MmRoutingLabels()


def enable_mm_routing_labels() -> None:
    """Record routing labels: this process records dynamo-format KV events."""
    MM_ROUTING_LABELS.enabled = True


def attach_mm_routing_labels(mm_items, labels_by_modality) -> None:
    """Attach caller routing labels to processed items, per media type.

    Each type's inputs fill that type's placeholders in order, so the k-th
    image item is the k-th image input however images and videos interleave
    in the prompt. A type whose item count differs from its labels, such as
    an unsplit bundle, stays unlabeled and its tokens are reported without
    spans.
    """
    if not labels_by_modality or not mm_items:
        return
    from sglang.srt.managers.schedule_batch import MultimodalDataItem

    items_by_modality = {}
    for item in mm_items:
        if isinstance(item, MultimodalDataItem):
            items_by_modality.setdefault(item.modality.name.lower(), []).append(item)
    for modality, labels in labels_by_modality.items():
        items = items_by_modality.get(modality, [])
        if len(items) != len(labels):
            logger.warning(
                "mm_routing_labels has %d %s labels for %d %s items; "
                "leaving those items unlabeled.",
                len(labels),
                modality,
                len(items),
                modality,
            )
            continue
        for item, label in zip(items, labels):
            item.routing_label = label or None


def register_mm_routing_labels(mm_items) -> None:
    if not MM_ROUTING_LABELS.enabled:
        return
    for item in mm_items or ():
        label = item.routing_label
        if label:
            MM_ROUTING_LABELS.register(item.padding_values(), label)


def mm_routing_spans(token_ids, block_size: int, num_blocks: int):
    """Per-block [start, length, label] runs of labeled multimodal tokens."""
    if not MM_ROUTING_LABELS or block_size <= 0 or not token_ids:
        return None
    if isinstance(token_ids[0], tuple):
        return None  # bigram (Eagle) keys are not labeled
    # Text holds no value at or above the pad shift, and max() runs in C, so
    # text skips the per-token loop: whole events here, blocks of mixed
    # events below.
    if max(token_ids) < _MM_PAD_SHIFT_VALUE:
        return None
    per_block = []
    any_span = False
    for block_index in range(num_blocks):
        block = token_ids[block_index * block_size : (block_index + 1) * block_size]
        spans = []
        per_block.append(spans)
        if not block or max(block) < _MM_PAD_SHIFT_VALUE:
            continue
        run_start, run_label = 0, None
        for position, token in enumerate(block):
            label = (
                MM_ROUTING_LABELS.get(token) if token >= _MM_PAD_SHIFT_VALUE else None
            )
            if label != run_label:
                if run_label is not None:
                    spans.append((run_start, position - run_start, run_label))
                run_start, run_label = position, label
        if run_label is not None:
            spans.append((run_start, len(block) - run_start, run_label))
        any_span = any_span or bool(spans)
    return per_block if any_span else None


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
        dynamo_format: bool = False,
        lora_names: Optional[KvEventLoraNames] = None,
    ):
        self.enabled = enabled
        if enabled and dynamo_format:
            enable_mm_routing_labels()
        self.page_size = page_size
        self.dynamo_format = dynamo_format
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
                    if tail.mm_routing_spans or event.mm_routing_spans:
                        # Extend in place: a chain coalesces one page at a time,
                        # so rebuilding the list would be quadratic in its length.
                        if not tail.mm_routing_spans:
                            tail.mm_routing_spans = [[] for _ in tail.block_hashes]
                        tail.mm_routing_spans.extend(
                            event.mm_routing_spans or [[] for _ in event.block_hashes]
                        )
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

    def _parent_block_hash(self, node: Any) -> Optional[int]:
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
        return hash_str_to_int64(parent_hash_values[-1])

    def _namespace_seed(self, node: Any) -> Optional[bytes]:
        if not self.dynamo_format:
            return kv_event_namespace_seed(
                extra_key=(
                    node.key.extra_key
                    if self.lora_names.get(node.key.extra_key) is not None
                    else None
                ),
                cache_salt=None,
            )
        return kv_event_namespace_seed(
            extra_key=node.key.extra_key, cache_salt=node.key.cache_salt
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

        event_hash_values = self._node_event_hash_values(node)
        namespace_seed = self._namespace_seed(node)
        parent_block_hash = self._parent_block_hash(node)
        if parent_block_hash is not None:
            parent_block_hash = namespaced_block_hash(
                parent_block_hash, namespace_seed=namespace_seed
            )
        lora_name = (
            self.lora_names.resolve(node.key.extra_key)
            if self.dynamo_format
            else self.lora_names.get(node.key.extra_key)
        )

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

            block_hash = namespaced_block_hash(
                hash_str_to_int64(event_hash_values[page_index]),
                namespace_seed=namespace_seed,
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
                    mm_routing_spans=(
                        mm_routing_spans(page_tokens, len(page_tokens), 1)
                        if self.dynamo_format and not is_bigram
                        else None
                    ),
                )
            )

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
        namespace_seed = self._namespace_seed(node)

        block_hashes = []
        logical_len = len(node.key)
        page_index = 0
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue

            block_hashes.append(
                namespaced_block_hash(
                    hash_str_to_int64(event_hash_values[page_index]),
                    namespace_seed=namespace_seed,
                )
            )
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
