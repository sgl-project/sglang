from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import logging
import sys
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # is bigram key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification.

    Args:
        node: The TreeNode to compute hash values for
        page_size: The page size for chunking tokens

    Returns:
        List of SHA256 hex strings, one per page
    """
    hash_values = []

    # Get parent's last hash value if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # Check if parent is root by checking if it has empty key
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # Use SHA256-based chaining via get_hash_str
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority'."
            )
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        if self.is_eagle and not key.is_bigram:
            key.token_ids = convert_to_bigram_key(key.token_ids)
            if value is not None:
                value = value[: len(key)]

        return key, value

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            key (RadixKey): The lookup key containing a list of token ids and an
                optional ``extra_key`` namespace tag. If ``page_size > 1`` the
                length is internally truncated to a multiple of ``page_size``
                before matching. Passing an empty key returns an empty result
                with the root as the last node.
            **kwargs: Reserved for future extensions (ignored currently).

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, key: RadixKey, value=None, chunked=False, priority: int = 0):
        if self.disable:
            return 0

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        return self._insert_helper(self.root_node, key, value, priority)

    def _page_align_keys(self, key: list) -> list:
        if self.page_size == 1:
            return key
        page_aligned_len = len(key) // self.page_size * self.page_size
        return key[:page_aligned_len]

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(req.fill_ids) if self.is_eagle else req.fill_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            new_prefix_len = self.insert(radix_key, values, priority=priority)
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(req.fill_ids) if self.is_eagle else req.fill_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            radix_key,
            values,
            chunked=chunked,
            priority=getattr(req, "priority", 0) or 0,
        )

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(radix_key)
        (new_indices, new_last_node) = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int):
        if self.disable:
            return

        start_time = time.perf_counter()
        leaves = self._collect_leaves()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        self._record_remove_event(child)
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _insert_helper(self, node: TreeNode, key: RadixKey, value, priority: int = 0):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                node = new_node
            else:
                node.priority = max(node.priority, priority)

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = list(self.root_node.children.values())

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                if cur_node.lock_ref == 0:
                    ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
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

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

                page_index += 1

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 4, 5], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    tree.insert(RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    tree.pretty_print()

    print(tree.match_prefix(RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None)))
