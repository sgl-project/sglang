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
from numpy import float64

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
)
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.utils import convert_to_bigram_key

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

import logging

logger = logging.getLogger(__name__)


class TreeNode:

    counter = 0
    swa_uuid_counter = 1
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        # swa_tombstone is used to indicate the kv indices have been freed for swa layers
        self.swa_tombstone = False
        # invariant: for any node, if swa_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, swa_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= swa_lock_ref.
        self.full_lock_ref = 0
        self.swa_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        # store the host indices of KV cache
        self.host_value = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.swa_prev = None
        self.swa_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
        self.swa_uuid = None

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def gen_swa_uuid() -> int:
    TreeNode.swa_uuid_counter += 1
    return TreeNode.swa_uuid_counter


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, is_swa_list: bool = False):
        self.is_swa_list = is_swa_list
        if self.is_swa_list:
            self.prv = "swa_prev"
            self.nxt = "swa_next"
            self.lock_ref = "swa_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Resetting swa tombstone node in swa lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            # for swa lru list, only reset non-tombstone nodes
            if not self.is_swa_list or not node.swa_tombstone:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Inserting swa tombstone node in swa lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Removing swa tombstone node from swa lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += len(node.value)
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: "SWARadixCache"):
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.
        """
        try:
            if self.is_swa_list:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru_plus_1 = len(self.cache) + 1
            # heapify based on last_access_time
            heapq.heapify(nodes)
            # the root node is not in the lru list
            assert (
                len(nodes) == len(self.cache) + 1
            ), f"len(nodes): {len(nodes)} != len(self.cache) + 1: {len(self.cache) + 1}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.is_swa_list=}, x: {x.id=} != x_lru: {x_lru.id=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"
                assert (
                    x_lru.swa_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.swa_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.is_swa_list:
                evictable_size = tree_cache.swa_evictable_size()
                lru_list_evictable_size = tree_cache.swa_lru_list_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = tree_cache.full_lru_list_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.is_swa_list=}, total nodes: {total_nodes}, total lru plus 1: {total_lru_plus_1}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            msg = f"SWA Radix tree sanity check failed, ping @hanming-lu: {e}"
            logger.error(msg)
            raise Exception(msg)


class SWARadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams, sliding_window_size: int):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.is_eagle = params.is_eagle

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

        if self.is_eagle:
            self.key_convert_fn = convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key

        self.init_metrics_collector()

        self.sliding_window_size = sliding_window_size
        self.reset()

    ##### Public API #####

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.full_lock_ref = 1
        self.root_node.swa_lock_ref = 1
        self.full_evictable_size_ = 0
        self.swa_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.swa_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(is_swa_list=False)
        self.swa_lru_list = LRUList(is_swa_list=True)

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            key: A RadixKey contains token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        key.token_ids = self.key_convert_fn(key.token_ids)

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

    def insert(self, key: RadixKey, value=None, prev_prefix_len: int = 0) -> int:
        if self.disable:
            return 0

        key.token_ids = self.key_convert_fn(key.token_ids)

        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)

        if self.is_eagle:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        return self._insert_helper(self.root_node, key, value, prev_prefix_len)

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        """Cache request when it finishes."""
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        # For EAGLE radix cache, we will convert the key to bigram key, e.g. [1,2,3,4] -> [(1,2), (2,3), (3,4)], the length will -1. ((len([(1,2), (2,3), (3,4)]) = len([1,2,3,4]) - 1))
        # So for the corresponding kv length should also -1. Then we get the actual_kv_len, and use it to do later calculation and slicing.
        actual_kv_len = kv_committed_len - 1 if self.is_eagle else kv_committed_len
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
            if self.is_eagle:
                self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])

        page_aligned_token_len = (
            page_aligned_len + 1 if self.is_eagle else page_aligned_len
        )

        old_prefix_len = len(req.prefix_indices)
        if self.is_eagle and old_prefix_len > req.cache_protected_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes one ref in memory pool
        # insert the token_ids and kv_indices into the radix tree
        # Note: the insert function already frees the overlapped kv_indices
        if is_insert:
            new_prefix_len = self.insert(
                RadixKey(token_ids[:page_aligned_token_len], req.extra_key),
                page_aligned_kv_indices,
                old_prefix_len,
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[old_prefix_len:page_aligned_len]
            )

        # free the unaligned tail
        if not self.is_eagle:
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node, req.swa_uuid_for_lock)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices
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
        if self.is_eagle and old_prefix_len > req.cache_protected_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes one ref in memory pool
        # Note: the insert function already frees the overlapped kv_indices
        new_prefix_len = self.insert(
            RadixKey(page_aligned_token_ids, req.extra_key),
            page_aligned_kv_indices,
            old_prefix_len,
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(
            RadixKey(page_aligned_token_ids, req.extra_key)
        )
        (new_indices, new_last_node) = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        assert old_prefix_len <= len(
            new_indices
        ), f"{req.prefix_indices=}, {new_indices=}"
        assert new_prefix_len <= len(new_indices), f"{new_prefix_len=}, {new_indices=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),
            new_indices[old_prefix_len:],
        )

        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node, req.swa_uuid_for_lock)
        swa_uuid_for_lock = self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if self.page_size != 1:
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
        req.swa_uuid_for_lock = swa_uuid_for_lock

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_swa_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #swa_tokens: {total_swa_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def evict(self, full_num_tokens: int, swa_num_tokens: int = 0) -> None:
        if self.disable:
            return
        start_time = time.perf_counter()
        full_num_evicted = 0
        swa_num_evicted = 0
        if full_num_tokens > 0:
            # get the least recently used leaf node that is not locked
            x = self.full_lru_list.get_leaf_lru_no_lock()

            while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
                assert (
                    x != self.root_node
                ), f"root node should not exist in full lru list, {x.id=}"
                assert x.full_lock_ref == 0, f"node is in use, {x.id=}"

                # 1. free node kv indices, evict full and swa tokens
                self.token_to_kv_pool_allocator.free(x.value)
                full_num_evicted += len(x.value)
                swa_num_evicted += len(x.value)

                # 2. get the next leaf, update the lru lists
                x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
                self.full_lru_list.remove_node(x)
                self.swa_lru_list.remove_node(x)

                # 3. delete the leaf node
                self._delete_leaf(x)

                # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
                x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
                full_num_evicted += leaf_full_num_evicted

                # 5. if parent has no more children, it is a leaf. It is possible that this node is lru, so
                # we need to get the first leaf node in the lru list
                if len(x.parent.children) == 0:
                    x_next = self.full_lru_list.get_leaf_lru_no_lock()

                x = x_next

        if swa_num_evicted < swa_num_tokens:
            # get the least recently used node that is not locked, doesn't have to be a leaf
            x = self.swa_lru_list.get_lru_no_lock()

            # evict lru leaf nodes until swa_num_tokens is reached
            while swa_num_evicted < swa_num_tokens and (self.swa_lru_list.in_list(x)):
                assert not x.swa_tombstone, f"duplicate swa tombstone node, {x.id=}"
                assert x != self.root_node, f"root node is not evictable, {x.id=}"
                assert x.swa_lock_ref == 0, f"node is in use by swa kv indices, {x.id=}"

                if len(x.children) > 0:
                    # 1. an internal node, free swa tokens.
                    self.token_to_kv_pool_allocator.free_swa(x.value)
                    swa_num_evicted += len(x.value)

                    # 2. get the next node, update the lru lists
                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    self.swa_lru_list.remove_node(x)

                    # 3. tombstone the node
                    self._tombstone_internal_node(x)
                else:
                    assert (
                        x.full_lock_ref == 0
                    ), f"leaf node with full lock must also have swa lock, {x.id=}"
                    # 1. a leaf node, free full and swa tokens
                    self.token_to_kv_pool_allocator.free(x.value)
                    full_num_evicted += len(x.value)
                    swa_num_evicted += len(x.value)

                    # 2. get the next node, update the lru lists
                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    self.full_lru_list.remove_node(x)
                    self.swa_lru_list.remove_node(x)

                    # 3. delete the leaf node
                    self._delete_leaf(x)

                    # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
                    self._iteratively_delete_tombstone_leaf(x)

                x = x_next

        self.update_eviction_metrics(full_num_evicted + swa_num_evicted, start_time)

    def inc_lock_ref(self, node: TreeNode) -> Optional[int]:
        """
        Increment the lock reference count for the node. Returns the swa_uuid_for_lock, which needs
        to be passed to dec_lock_ref.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the swa_lock_ref for nodes between the [last node, swa_uuid_for_lock], inclusive.
        """
        if self.disable:
            return None

        swa_lock_size = 0
        swa_uuid_for_lock = None
        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1

            # lock swa if we have not reached the sliding window size.
            # When we reach the sliding window size, we will set the swa_uuid_for_lock.
            # caller needs to pass the swa_uuid_for_lock to dec_lock_ref
            if swa_lock_size < self.sliding_window_size:
                assert (
                    not node.swa_tombstone
                ), f"inc_lock_swa on swa_tombstone node, {node.id=}"
                if node.swa_lock_ref == 0:
                    self.swa_evictable_size_ -= len(node.value)
                    self.swa_protected_size_ += len(node.value)
                node.swa_lock_ref += 1
                swa_lock_size += len(node.value)
                if swa_lock_size >= self.sliding_window_size:
                    if node.swa_uuid is None:
                        node.swa_uuid = gen_swa_uuid()
                    swa_uuid_for_lock = node.swa_uuid
            node = node.parent
        return swa_uuid_for_lock

    def dec_lock_ref(self, node: TreeNode, swa_uuid_for_lock: Optional[int] = None):
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the swa_lock_ref for nodes between the [last node, swa_uuid_for_lock], inclusive.
        If swa_uuid_for_lock is None, it unlocks to the root, exclusive.
        """
        if self.disable:
            return

        dec_lock_swa = True
        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1

            if dec_lock_swa:
                assert (
                    not node.swa_tombstone
                ), f"dec_lock_ref on swa_tombstone node, {node.id=}"
                assert (
                    node.swa_lock_ref > 0
                ), f"dec_lock_ref on node with {node.swa_lock_ref=}, {node.id=}"

                if node.swa_lock_ref == 1:
                    self.swa_evictable_size_ += len(node.value)
                    self.swa_protected_size_ -= len(node.value)
                node.swa_lock_ref -= 1
                if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:
                    dec_lock_swa = False

            node = node.parent

    def sanity_check(self):
        self.full_lru_list.sanity_check(self)
        self.swa_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and swa_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def swa_evictable_size(self) -> int:
        return self.swa_evictable_size_

    # Note: this is expensive, only use for debug
    def full_lru_list_evictable_size(self) -> int:
        return self.full_lru_list.sanity_check_evictable_size()

    # Note: this is expensive, only use for debug
    def swa_lru_list_evictable_size(self) -> int:
        return self.swa_lru_list.sanity_check_evictable_size()

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

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode]:
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

        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows swa to evict nodes closer to root first
        node_update = best_last_node
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        self.swa_lru_list.reset_node_and_parents_mru(node_update, self.root_node)

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        return value[:best_value_len], best_last_node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.swa_tombstone = child.swa_tombstone
        new_node.full_lock_ref = child.full_lock_ref
        new_node.swa_lock_ref = child.swa_lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        # parent inherits the swa_uuid from child for swa lock ref
        new_node.swa_uuid = child.swa_uuid
        child.swa_uuid = None
        # child time should be later than parent's time for swa tombstone
        child.last_access_time = get_last_access_time()

        # remove the child from the lru lists because it is being split
        self.full_lru_list.remove_node(child)
        if not new_node.swa_tombstone:
            self.swa_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if not new_node.swa_tombstone:
            self.swa_lru_list.insert_mru(new_node)
            self.swa_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self, node: TreeNode, key: RadixKey, value, update_kv_after_len: int
    ) -> int:
        # Update the last access time from root to leaf, so that
        # swa will tombstone the node closer to root first
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            if not node.swa_tombstone:
                self.swa_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if not node.swa_tombstone:
                self.swa_lru_list.reset_node_mru(node)
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
                    assert (
                        node.swa_lock_ref == 0
                    ), f"tombstone swa_lock_ref should always be 0, {node.full_lock_ref=}, {node.swa_lock_ref=}, {node.id=}"
                    self.token_to_kv_pool_allocator.free(node.value[first_diff_idx:])
                    node.value = value[:prefix_len]
                    node.swa_tombstone = False

                    # insert the node into the lru lists
                    self.swa_lru_list.insert_mru(node)

                    self.swa_evictable_size_ += len(node.value)
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
            self.full_lru_list.insert_mru(new_node)
            self.swa_lru_list.insert_mru(new_node)
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
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.swa_lock_ref == 0
            ), f"tombstone swa_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.swa_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens
            self.token_to_kv_pool_allocator.free(node.parent.value)
            full_num_evicted += len(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            not node.swa_tombstone
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"
        self.full_evictable_size_ -= len(node.key)
        self.swa_evictable_size_ -= len(node.key)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        node.swa_tombstone = True
        self.swa_evictable_size_ -= len(node.key)

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.swa_tombstone
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

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

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                current_node.id,
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"sr={current_node.swa_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"sll={self.swa_lru_list.in_list(current_node)}",
                f"ts={current_node.swa_tombstone}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

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
