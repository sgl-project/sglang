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
The radix tree data structure for managing the hybrid (full and SWA) KV cache.
"""

import heapq
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: List[int] = None
        self.value: Optional[torch.Tensor] = None
        # swa_tombstone is used to indicate the kv indices have been freed for swa layers
        self.swa_tombstone = False
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i


class SWARadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: SWATokenToKVPoolAllocator,
        sliding_window_size: int,
        page_size: int,
        disable: bool = False,
    ):
        assert isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])

        self.sliding_window_size = sliding_window_size
        self.reset()

    ##### Public API #####

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.full_evictable_size_ = 0
        self.swa_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.swa_protected_size_ = 0

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, key: List, value=None, prev_prefix_len: int = 0) -> int:
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, prev_prefix_len)

    def cache_finished_req(self, req: Req) -> None:
        """Cache request when it finishes."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx,
                : len(req.origin_input_ids) + len(req.output_ids) - 1,
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()

        # Radix Cache takes one ref in memory pool
        # insert the token_ids and kv_indices into the radix tree
        # Note: the insert function already frees the overlapped kv_indices
        new_prefix_len = self.insert(
            token_ids[:page_aligned_len],
            page_aligned_kv_indices,
            len(req.prefix_indices),
        )

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req) -> None:
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()
        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Radix Cache takes one ref in memory pool
        # Note: the insert function already frees the overlapped kv_indices
        new_prefix_len = self.insert(
            page_aligned_token_ids, page_aligned_kv_indices, len(req.prefix_indices)
        )

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node, _, _ = self.match_prefix(page_aligned_token_ids)
        assert len(req.prefix_indices) <= len(
            new_indices
        ), f"{req.prefix_indices=}, {new_indices=}"
        assert new_prefix_len <= len(new_indices), f"{new_prefix_len=}, {new_indices=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if self.page_size != 1:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_swa_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #swa_tokens: {total_swa_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def evict(self, full_num_tokens: int, swa_num_tokens: int = 0) -> None:
        if self.disable:
            return

        full_num_evicted = 0
        swa_num_evicted = 0
        if full_num_tokens > 0:
            nodes = self._collect_leaves()
            # heapify based on last_access_time
            heapq.heapify(nodes)

            while full_num_evicted < full_num_tokens and len(nodes):
                x = heapq.heappop(nodes)

                # root node is not evictable
                if x == self.root_node:
                    break
                # if locked, means node is in use, skip
                if x.lock_ref > 0:
                    continue

                # evict leaf node, evict full and swa tokens
                self.token_to_kv_pool_allocator.free(x.value)
                full_num_evicted += len(x.value)
                swa_num_evicted += len(x.value)
                self._delete_leaf(x)

                # Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
                x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
                full_num_evicted += leaf_full_num_evicted

                # if parent has no more children, it is a leaf, so add to heap
                if len(x.parent.children) == 0:
                    heapq.heappush(nodes, x.parent)

        if swa_num_evicted < swa_num_tokens:
            nodes = self._collect_nontombstone_nodes()
            # TODO(hm): optimize by maintaining a linked list sorted by last_access_time
            heapq.heapify(nodes)

            # evict lru leaf nodes until swa_num_tokens is reached
            while swa_num_evicted < swa_num_tokens and len(nodes):
                x = heapq.heappop(nodes)
                assert not x.swa_tombstone, f"duplicate swa tombstone node, {x.key=}"

                # root node is not evictable
                if x == self.root_node:
                    continue
                # if locked, means node is in use, skip
                if x.lock_ref > 0:
                    continue

                if len(x.children) > 0:
                    # an internal node, tombstone it to evict swa tokens
                    self.token_to_kv_pool_allocator.free_swa(x.value)
                    swa_num_evicted += len(x.value)
                    self._tombstone_internal_node(x)
                else:
                    # a leaf node, evict it to evict full and swa tokens
                    self.token_to_kv_pool_allocator.free(x.value)
                    full_num_evicted += len(x.value)
                    swa_num_evicted += len(x.value)
                    self._delete_leaf(x)

                    # Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
                    self._iteratively_delete_tombstone_leaf(x)

    def inc_lock_ref(self, node: TreeNode) -> Tuple[int, int]:
        if self.disable:
            return 0, 0

        full_delta = 0
        swa_delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
                full_delta -= len(node.value)
                if not node.swa_tombstone:
                    self.swa_evictable_size_ -= len(node.value)
                    self.swa_protected_size_ += len(node.value)
                    swa_delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return full_delta, swa_delta

    def dec_lock_ref(self, node: TreeNode) -> Tuple[int, int]:
        if self.disable:
            return 0, 0

        full_delta = 0
        swa_delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
                full_delta += len(node.value)
                if not node.swa_tombstone:
                    self.swa_evictable_size_ += len(node.value)
                    self.swa_protected_size_ -= len(node.value)
                    swa_delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return full_delta, swa_delta

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and swa_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def swa_evictable_size(self) -> int:
        return self.swa_evictable_size_

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and swa_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def swa_protected_size(self) -> int:
        # protected size refers to the size of the swa cache that is locked
        return self.swa_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, key: List) -> Tuple[List[torch.Tensor], TreeNode]:
        """
        SWA prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without swa tombstone,
        or 2. the number of matching tokens from the matched node to the last swa tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = self.get_child_key_fn(key)

        value = []
        # for path connected to root without tombstone, always match, so set to inf
        match_len_since_tombstone = float("inf")
        best_value_len = 0
        best_last_node = node
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]

            # update best_value_len and best_last_node if needed
            if (
                child.swa_tombstone
                and match_len_since_tombstone >= self.sliding_window_size
            ):
                best_value_len = len(value)
                best_last_node = node
                match_len_since_tombstone = 0

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                if not new_node.swa_tombstone:
                    match_len_since_tombstone += len(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                if not child.swa_tombstone:
                    match_len_since_tombstone += len(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if match_len_since_tombstone >= self.sliding_window_size:
            best_value_len = len(value)
            best_last_node = node

        # update time for matched nodes, and make nodes closer to root have earlier access time
        cur_time = time.monotonic()
        while node:
            node.last_access_time = cur_time
            cur_time -= 0.0001
            node = node.parent

        return value[:best_value_len], best_last_node

    def _split_node(self, key: List[int], child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.swa_tombstone = child.swa_tombstone
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.last_access_time = (
            time.monotonic()
        )  # child time should be later than parent's time for swa tombstone
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _insert_helper(
        self, node: TreeNode, key: List, value, update_kv_after_len: int
    ) -> int:
        # Update the last access time from root to leaf, so that
        # swa will tombstone the node closer to root first
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            # if tombstone after update_kv_after_len, update node.value to be the input value.
            # This is needed because it is possible that the last sliding window size tokens
            # contains tombstone. If this is the case and we don't update the kv value, then
            # the prefill prefix matching will stuck.
            if update_kv_after_len < total_prefix_length + prefix_len:
                first_diff_idx = max(0, update_kv_after_len - total_prefix_length)
                if node.swa_tombstone:
                    self.token_to_kv_pool_allocator.free(node.value[first_diff_idx:])
                    node.value = value[:prefix_len]
                    node.swa_tombstone = False
                    if node.lock_ref == 0:
                        self.swa_evictable_size_ += len(node.value)
                    else:
                        self.swa_protected_size_ += len(node.value)
                else:
                    self.token_to_kv_pool_allocator.free(
                        value[first_diff_idx:prefix_len]
                    )

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.full_evictable_size_ += len(value)
            self.swa_evictable_size_ += len(value)
        return total_prefix_length

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.swa_tombstone and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.lock_ref > 0:
                break
            # delete tombstone node evicts full tokens
            self.token_to_kv_pool_allocator.free(node.parent.value)
            full_num_evicted += len(node.parent.value)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
                f"swa_tombstone={current_node.swa_tombstone}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            not node.swa_tombstone
        ), f"Invariant violated: leaf node is a tombstone, {node.key=}"
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.full_evictable_size_ -= len(node.key)
        self.swa_evictable_size_ -= len(node.key)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.key=}"
        node.swa_tombstone = True
        self.swa_evictable_size_ -= len(node.key)

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.swa_tombstone
        ), f"Deleting a unexpected non-tombstone leaf node, {node.key=}"
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.full_evictable_size_ -= len(node.key)

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_swa_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if not current_node.swa_tombstone:
                total_swa_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_swa_size

    def _collect_leaves(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if not cur_node.swa_tombstone:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list
