from __future__ import annotations

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
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union

import torch

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:

    def __init__(self, token_ids: List[int], extra_key: Optional[str] = None):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key

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

    def __init__(self, id: Optional[int] = None):
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


def _convert_to_bigram_key(tokens: List[int]) -> List[Tuple[int, int]]:
    # EAGLE uses bigram keys in the radix tree since draft sequence is the one-token-shifted version of target
    # [1, 2, 3, 4] -> [(1,2), (2,3), (3,4)]
    if len(tokens) < 2:
        return []
    if isinstance(tokens[0], tuple):
        return tokens
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        enable_metrics: bool = False,
        enable_kv_cache_events: bool = False,
        eviction_policy: str = "lru",
        is_eagle: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []
        self.is_eagle = is_eagle

        if enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=page_size)

        if is_eagle:
            self.key_convert_fn = _convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key

        if eviction_policy.lower() == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif eviction_policy.lower() == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif eviction_policy.lower() == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif eviction_policy.lower() == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif eviction_policy.lower() == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo'."
            )
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

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
        key.token_ids = self.key_convert_fn(key.token_ids)

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

    def insert(self, key: RadixKey, value=None, chunked=False):
        if self.disable:
            return 0

        key.token_ids = self.key_convert_fn(key.token_ids)

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        if self.is_eagle:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        committed_kv_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :committed_kv_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:committed_kv_len]
        # For EAGLE radix cache, we will convert the key to bigram key, e.g. [1,2,3,4] -> [(1,2), (2,3), (3,4)], the length will -1. ((len([(1,2), (2,3), (3,4)]) = len([1,2,3,4]) - 1))
        # So for the corresponding kv length should also -1. Then we get the actual_kv_len, and use it to do later calculation and slicing.
        actual_kv_len = committed_kv_len - 1 if self.is_eagle else committed_kv_len
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :committed_kv_len
        ]

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        page_aligned_token_len = (
            page_aligned_len + 1 if self.is_eagle else page_aligned_len
        )

        old_prefix_len = len(req.prefix_indices)
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes one ref in memory pool
        if is_insert:
            new_prefix_len = self.insert(
                RadixKey(token_ids[:page_aligned_token_len], req.extra_key),
                page_aligned_kv_indices,
            )
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[old_prefix_len:new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[old_prefix_len:page_aligned_len]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        all_token_len = len(token_ids)
        # For EAGLE radix cache, we will convert the key to bigram key, e.g. [1,2,3,4] -> [(1,2), (2,3), (3,4)], the length will -1. ((len([(1,2), (2,3), (3,4)]) = len([1,2,3,4]) - 1))
        # So for the corresponding kv length should also -1. Then we get the actual_kv_len, and use it to do later calculation and slicing.
        actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :all_token_len
        ]

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        # For EAGLE, the page_aligned_len is for the bigram key, the normal key len should +1
        page_aligned_token_len = (
            page_aligned_len + 1 if self.is_eagle else page_aligned_len
        )
        page_aligned_token_ids = token_ids[:page_aligned_token_len]

        old_prefix_len = len(req.prefix_indices)
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            RadixKey(page_aligned_token_ids, req.extra_key),
            page_aligned_kv_indices,
            chunked=chunked,
        )
        self.token_to_kv_pool_allocator.free(kv_indices[old_prefix_len:new_prefix_len])

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node, _, _ = self.match_prefix(
            RadixKey(token_ids=page_aligned_token_ids, extra_key=req.extra_key)
        )
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),
            new_indices[old_prefix_len:],
        )

        # The last_matched_prefix_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `last_matched_prefix_len` field to make sure the partial part can be freed correctly.
        req.last_matched_prefix_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if self.page_size != 1:
            # Handle partial page, the partial part should be freed in the next cache_unfinished_req and final cache_finished_req.
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            if self.is_eagle:
                # Attach the kv index of the last token for EAGLE, it can be used in chunked prefill
                req.prefix_indices = torch.cat(
                    [new_indices, kv_indices[actual_kv_len:]]
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
        self._record_remove_event(child)
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._record_store_event(new_node)
        self._record_store_event(child)

        return new_node

    def _insert_helper(self, node: TreeNode, key: RadixKey, value):
        access_time = time.monotonic()
        node.last_access_time = access_time
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
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
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
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
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
            # First chunk links to the last page of the parent node (if any).
            if node.parent is None or node != self.root_node:
                parent_block_hash = None
            else:
                last_page_start = (
                    (len(node.parent.key) - 1) // self.page_size
                ) * self.page_size
                parent_parent_tokens = node.parent.key.token_ids[last_page_start:]
                parent_block_hash = hash(tuple(parent_parent_tokens))

            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash(tuple(page_tokens))

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                    )
                )

                # Chain next chunk to this one.
                parent_block_hash = block_hash

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue
                block_hash = hash(tuple(page_tokens))
                self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

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
    tree = RadixCache(None, None, page_size=1, disable=False)

    # Example token id sequences (as lists of ints)
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 4, 5], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    tree.insert(RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    tree.pretty_print()

    print(tree.match_prefix(RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None)))
