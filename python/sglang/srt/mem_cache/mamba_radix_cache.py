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
The radix tree data structure for managing the hybrid (full and Mamba) KV cache.
"""

import os
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import split_node_hash_value
from sglang.srt.runtime_context import get_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

import logging

from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# Debug-only invariant checks in the Mamba slot-donation path call tensor.item(),
# which forces a per-request cudaStreamSynchronize on the scheduler thread. Under
# load this can serialize/stall the scheduler. Gate them off by default; set
# SGLANG_MAMBA_DEBUG_ASSERTS=1 to re-enable for debugging.
_MAMBA_DEBUG_ASSERTS = os.environ.get("SGLANG_MAMBA_DEBUG_ASSERTS", "0") == "1"


class TreeNode:

    counter = 0
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None
        self.mamba_host_value: Optional[torch.Tensor] = None
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # for full_lock, once it is locked, its parent must be locked as well
        # for mamba_lock, it only need lock node itself
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        # `last_access_time` tracks the full LRU (whole matched path is reused as prefix);
        # `mamba_last_access_time` tracks the mamba LRU, which only touches the single state
        # actually consumed per access, so the two orders diverge and need separate stamps.
        self.last_access_time = get_last_access_time()
        self.mamba_last_access_time = self.last_access_time

        self.hit_count = 0
        self.host_ref_counter = 0
        self.host_mamba_ref_counter = 0
        # store the host indices of KV cache
        self.host_value = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.mamba_prev = None
        self.mamba_next = None
        self.host_mamba_prev = None
        self.host_mamba_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def mamba_evicted(self):
        return self.mamba_value is None

    @property
    def backuped(self):
        return self.host_value is not None

    @property
    def mamba_backuped(self):
        return self.mamba_host_value is not None

    def protect_host(self):
        """Protect the host KV value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host KV value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def protect_host_mamba(self):
        """Protect the host mamba value from eviction."""
        self.host_mamba_ref_counter += 1

    def release_host_mamba(self):
        """Release the host mamba value, allowing it to be evicted."""
        if self.host_mamba_ref_counter > 0:
            self.host_mamba_ref_counter -= 1
        else:
            raise RuntimeError("Host mamba reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []
        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, mamba: bool = False):
        self.mamba = mamba
        if self.mamba:
            self.prv = "mamba_prev"
            self.nxt = "mamba_next"
            self.lock_ref = "mamba_lock_ref"
            self.time_attr = "mamba_last_access_time"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
            self.time_attr = "last_access_time"
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
        # Clear self pointers to break reference cycles among evicted nodes.
        setattr(node, self.prv, None)
        setattr(node, self.nxt, None)

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
            not self.mamba or node.mamba_value is not None
        ), f"Resetting mamba tombstone node in mamba lru list: {node.id=}"
        if self.mamba:
            node.mamba_last_access_time = get_last_access_time()
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            if not self.mamba or node.mamba_value is not None:
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
            not self.mamba or node.mamba_value is not None
        ), f"Inserting mamba tombstone node in mamba lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        if self.mamba:
            node.mamba_last_access_time = get_last_access_time()
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Removing mamba tombstone node from mamba lru list: {node.id=}"
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

    def pretty_print(self, tree_cache: Optional[MambaRadixCache] = None):
        """
        Pretty print the lru list
        """
        msg = f"{self.mamba=} LRU list: "
        x_lru = self._get_lru()
        while x_lru is not None and x_lru.id in self.cache:
            msg += f"[{x_lru.id}] {getattr(x_lru, self.time_attr):f} -> "
            x_lru = getattr(x_lru, self.prv)
        print(msg)

        if not tree_cache:
            return
        msg = f"{self.mamba=} Nodes (sorted by {self.time_attr}): "
        if self.mamba:
            nodes = tree_cache._collect_nontombstone_nodes()
        else:
            nodes = tree_cache._collect_all_nodes()
        nodes.sort(key=lambda n: getattr(n, self.time_attr))
        for x in nodes:
            msg += f"[{x.id}] {getattr(x, self.time_attr):f} -> "
        print(msg)

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += (
                len(node.value) if not self.mamba else len(node.mamba_value)
            )
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: MambaRadixCache):
        """
        Check the lru list is valid by rebuilding it from the tree, sorting by this list's
        access-time stamp, and checking the order matches the linked list.
        """
        try:
            if self.mamba:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru = len(self.cache)
            # rebuild expected order from this list's own access-time stamp (full and mamba
            # lists have independent recency, so they use different stamps)
            nodes.sort(key=lambda n: getattr(n, self.time_attr))
            # the root node is not in the lru list
            assert len(nodes) == (
                total_lru + (0 if self.mamba else 1)
            ), f"len(nodes): {len(nodes)}, total_lru: {total_lru}"

            x_lru = self._get_lru()
            for x in nodes:
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x_lru is not None and x_lru.id in self.cache
                ), f"Incorrect LRU list, x_lru is None or not in cache: {x_lru=}, {x.id=}"

                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.mamba=}, x: {x.id=} != x_lru: {x_lru.id=}, {getattr(x, self.time_attr)=}, {getattr(x_lru, self.time_attr)=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.id=}"
                assert (
                    x_lru.mamba_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.mamba_lock_ref=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.mamba:
                evictable_size = tree_cache.mamba_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.mamba=}, total nodes: {total_nodes}, total lru: {total_lru}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            if get_parallel().tp_rank == 0:
                msg = f"Mamba Radix tree sanity check failed, ping @yizhang2077: {e}"
                logger.error(msg)
                tree_cache.pretty_print()
                tree_cache.full_lru_list.pretty_print(tree_cache)
                tree_cache.mamba_lru_list.pretty_print(tree_cache)
                raise Exception(msg)


class MambaRadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert (
            isinstance(params.token_to_kv_pool_allocator, TokenToKVPoolAllocator)
            or isinstance(
                params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator
            )
            or isinstance(
                params.token_to_kv_pool_allocator, UnifiedMambaTokenToKVPoolAllocator
            )
        )
        self.req_to_token_pool: HybridReqToTokenPool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.mamba_cache_chunk_size = get_server_args().mamba_cache_chunk_size

        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        self.enable_mamba_extra_buffer_lazy = params.enable_mamba_extra_buffer_lazy
        self.kv_event_queue = []

        if not self.enable_mamba_extra_buffer:
            assert (
                self.page_size == 1
            ), f"Page size must be 1 for MambaRadixCache v1, got {self.page_size}"

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        self.reset()

    ##### Public API #####

    def supports_mamba(self) -> bool:
        return True

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.value = []
        self.root_node.hash_value = []
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self.full_evictable_size_ = 0
        self.mamba_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.mamba_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(mamba=False)
        self.mamba_lru_list = LRUList(mamba=True)
        self._record_all_cleared_event()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            params: MatchPrefixParams containing key and optional Mamba-specific parameters.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                best_match_node=self.root_node,
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        mamba_value = params.mamba_value
        prev_prefix_len = params.prev_prefix_len

        if value is None:
            value = torch.tensor([x for x in key.raw_token_ids()], dtype=torch.int64)
        prefix_len, mamba_exist = self._insert_helper(
            self.root_node, key, value, mamba_value, params.chunked, prev_prefix_len
        )
        return InsertResult(prefix_len=prefix_len, mamba_exist=mamba_exist)

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> None:
        """Cache request when it finishes."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_mamba_cache(req)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len_to_handle
        ]

        if is_insert:
            if self.enable_mamba_extra_buffer:
                cache_len = req.mamba_last_track_seqlen
            else:
                cache_len = len(token_ids)
                # ReplaySSM (no_buffer): `temporal[slot]` lags the live state by
                # the slot's unflushed ring depth (`write_pos`), so cap the
                # donate to the last flush boundary (where temporal is current)
                # and reset the cursor, keeping the donated checkpoint consistent
                # with its key length. page_size is asserted == 1, so no realign.
                write_pos_buf = self.req_to_token_pool.mamba_pool.replayssm_write_pos
                if write_pos_buf is not None:
                    cache_len -= int(write_pos_buf[req.mamba_pool_idx].item())
                    write_pos_buf[req.mamba_pool_idx] = 0
            if cache_len is None:
                cache_len = 0
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

            if self.page_size != 1:
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size
                page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                    dtype=torch.int64, copy=True
                )
            else:
                page_aligned_len = len(kv_indices)
                page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

            assert (
                cache_len == page_aligned_len
            ), f"It is required {cache_len=}, {page_aligned_len=}, {kv_len_to_handle=}, {len(req.origin_input_ids)=}, {len(req.output_ids)=} ping @yizhang2077 if you see this"

            # Radix Cache takes one ref in memory pool
            # insert the token_ids and kv_indices into the radix tree
            if self.enable_mamba_extra_buffer:
                mamba_ping_pong_track_buffer_to_keep = (
                    self.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
                )
                src_active = req.mamba_ping_pong_track_buffer[
                    mamba_ping_pong_track_buffer_to_keep
                ].unsqueeze(-1)
                if _MAMBA_DEBUG_ASSERTS:
                    # .item() forces a cudaStreamSynchronize; only pay it when debugging.
                    assert src_active.item() != -1, (
                        f"Cached mamba slot is -1: keep_idx={mamba_ping_pong_track_buffer_to_keep}, "
                        f"buf={req.mamba_ping_pong_track_buffer.tolist()}, "
                        f"next_track_idx={req.mamba_next_track_idx}, "
                        f"last_track_seqlen={req.mamba_last_track_seqlen}, "
                        f"rid={req.rid}"
                    )
                if self.int8_ckpt_pool is not None:
                    mamba_value = self._commit_int8_checkpoint(src_active)
                    # quantized -> no ping-pong slot needs keeping
                    mamba_ping_pong_track_buffer_to_keep = None
                else:
                    mamba_value = src_active.clone()
            else:
                if self.int8_ckpt_pool is not None:
                    mamba_value = self._commit_int8_checkpoint(
                        req.mamba_pool_idx.unsqueeze(-1)
                    )
                else:
                    mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
                mamba_ping_pong_track_buffer_to_keep = None

            result = self.insert(
                InsertParams(
                    key=RadixKey(token_ids[:page_aligned_len], req.extra_key),
                    value=page_aligned_kv_indices,
                    mamba_value=mamba_value,
                    prev_prefix_len=req.cache_protected_len,
                )
            )
            mamba_exist = result.mamba_exist
            if mamba_exist and self.int8_ckpt_pool is not None:
                # state already cached -> the int8 slot we just allocated is a duplicate
                self.int8_ckpt_pool.free(mamba_value)
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])
            mamba_exist = True

        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None

        # With int8 checkpoints the radix owns an int8 slot (not the request's active
        # slot), so the active mamba slot must always be returned to the active pool.
        free_mamba_cache = (
            True
            if (self.enable_mamba_extra_buffer or self.int8_ckpt_pool is not None)
            else mamba_exist
        )

        if free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
            )

        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""

        def _skip_cache_unfinished_req(req: Req) -> None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.extend_range.end
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        token_ids = req.get_fill_ids()
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )
        if self.disable or cache_len is None:
            return _skip_cache_unfinished_req(req)

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        # kv_indices is the kv indices to be cached
        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        assert page_aligned_len == len(
            kv_indices
        ), f"page_aligned_len != len(kv_indices), {page_aligned_len=}, {len(kv_indices)=}, {cache_len=}, {self.page_size=}, {self.mamba_cache_chunk_size=}"

        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Donate the mamba index to the radix cache instead of copying.
        # This avoids a data copy that would race with the forward stream.
        if self.int8_ckpt_pool is not None:
            # int8 path: quantize the to-be-cached active state into an int8 slot
            # (strategy-agnostic donate hook).
            if self.enable_mamba_extra_buffer:
                new_slot = self._alloc_mamba_slot()
                src_active = self.req_to_token_pool.donate_mamba_ping_pong_slot(
                    req, new_slot
                )
                mamba_value_donated = self._commit_int8_checkpoint(src_active)
                self.req_to_token_pool.mamba_allocator.free(src_active)
            else:
                mamba_value_donated = self._commit_int8_checkpoint(
                    req.mamba_pool_idx.view(-1)
                )
        elif self.enable_mamba_extra_buffer:
            new_slot = self._alloc_mamba_slot()
            mamba_value_donated = self.req_to_token_pool.donate_mamba_ping_pong_slot(
                req, new_slot
            )
        else:
            mamba_value_donated = self._alloc_mamba_slot()
            # mamba_pool is a pure PHYSICAL store; translate both slot ids
            # virtual->physical (identity for the non-unified memory pool) before the copy.
            translate = self.req_to_token_pool.translate_mamba_indices
            self.req_to_token_pool.mamba_pool.copy_from(
                translate(req.mamba_pool_idx.unsqueeze(0)),
                translate(mamba_value_donated),
            )

        result = self.insert(
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key),
                value=page_aligned_kv_indices,
                mamba_value=mamba_value_donated,
                prev_prefix_len=req.cache_protected_len,
                chunked=chunked,
            )
        )
        new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
        if mamba_exist:
            self._free_mamba_value(mamba_value_donated)

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        if not mamba_exist:
            assert torch.equal(new_last_node.mamba_value, mamba_value_donated)

        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {len(page_aligned_token_ids)=}, {mamba_exist=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # NOTE: this is needed for both page_size == 1 and page_size > 1
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_mamba_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #mamba_num: {total_mamba_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def _evict_leaf_node(
        self, x: TreeNode, is_evict_mamba: bool
    ) -> Tuple[int, int, TreeNode, TreeNode]:
        assert (
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"

        assert x.mamba_value is not None, f"leaf node mamba value is not None, {x.id=}"
        # 1. a leaf node, free full tokens and mamba
        self._record_remove_event(x)
        self.token_to_kv_pool_allocator.free(x.value)
        full_num_evicted = len(x.value)
        self._free_mamba_value(x.mamba_value)
        mamba_num_evicted = len(x.mamba_value)

        # 2. get the next node, update the lru lists
        if is_evict_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
        self.full_lru_list.remove_node(x)
        self.mamba_lru_list.remove_node(x)

        # 3. delete the leaf node
        self._delete_leaf(x)

        # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_evicted = 0
        mamba_num_evicted = 0

        if params.num_tokens > 0:
            full_num_evicted = self.evict_full(params.num_tokens)
        if params.mamba_num > 0:
            mamba_num_evicted = self.evict_mamba(params.mamba_num)

        return EvictResult(
            num_tokens_evicted=full_num_evicted, mamba_num_evicted=mamba_num_evicted
        )

    def evict_mamba(self, mamba_num: int) -> int:
        """Evict mamba states. Returns the number of mamba states evicted."""
        if self.disable or mamba_num <= 0:
            return 0
        # get the least recently used node that is not locked, doesn't have to be a leaf
        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        # evict lru leaf nodes until mamba_num_tokens is reached
        while mamba_num_evicted < mamba_num and (self.mamba_lru_list.in_list(x)):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert (
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}"

            if len(x.children) > 0:
                # 1. an internal node, free mamba tokens.
                self._free_mamba_value(x.mamba_value)
                mamba_num_evicted += len(x.mamba_value)

                # 2. get the next node, update the lru lists
                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                self.mamba_lru_list.remove_node(x)

                # 3. tombstone the node
                self._tombstone_internal_node(x)
            else:
                _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
                mamba_num_evicted += mamba_evicted_delta

            x = x_next

        return mamba_num_evicted

    def evict_full(self, full_num_tokens: int) -> int:
        """Evict full KV cache. Returns the number of tokens evicted."""
        if self.disable or full_num_tokens <= 0:
            return 0

        full_num_evicted = 0
        # get the least recently used leaf node that is not locked
        x = self.full_lru_list.get_leaf_lru_no_lock()

        while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
            assert (
                x != self.root_node
            ), f"root node should not exist in full lru list, {x.id=}"
            full_num_evicted_delta, _, x, x_next = self._evict_leaf_node(x, False)
            full_num_evicted += full_num_evicted_delta

            # if parent has no more children, it is a leaf. It is possible that this node is lru, so
            # we need to get the first leaf node in the lru list
            if len(x.parent.children) == 0:
                x_next = self.full_lru_list.get_leaf_lru_no_lock()

            x = x_next

        return full_num_evicted

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return IncLockRefResult()

        # protect mamba value in current node if it exists
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1

        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1
            node = node.parent
        return IncLockRefResult()

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return DecLockRefResult()

        if node.mamba_value is not None:
            assert (
                node.mamba_lock_ref > 0
            ), f"dec_lock_ref on node with {node.mamba_lock_ref=}, {node.id=}"
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
            node.mamba_lock_ref -= 1

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1
            node = node.parent

        return DecLockRefResult()

    def sanity_check(self):
        if self.disable:
            return
        self.full_lru_list.sanity_check(self)
        self.mamba_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable_size_

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and mamba_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        # protected size refers to the size of the mamba cache that is locked
        return self.mamba_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def all_mamba_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            if node.mamba_value is not None:
                values.append(node.mamba_value)
            for _, child in node.children.items():
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = self.full_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
        )

    ##### Internal Helper Functions #####

    def _alloc_mamba_slot(self) -> torch.Tensor:
        """Allocate one mamba pool slot, evicting if necessary."""
        slot = self.req_to_token_pool.mamba_allocator.alloc(1)
        if slot is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.req_to_token_pool.mamba_allocator.alloc(1)
            assert slot is not None, "Can not alloc mamba cache"
        return slot

    @property
    def int8_ckpt_pool(self):
        """The int8 checkpoint pool, or None when --enable-int8-mamba-checkpoint is off.
        When enabled, radix-cached mamba states live HERE (int8), not in the active
        bf16 pool -> ~2x cached-prefix capacity at fixed memory."""
        return getattr(self.req_to_token_pool, "mamba_ckpt_pool", None)

    def _alloc_int8_ckpt_slot(self) -> torch.Tensor:
        """Allocate one int8 checkpoint slot, evicting cached states if the pool is full."""
        slot = self.int8_ckpt_pool.alloc(1)
        if slot is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.int8_ckpt_pool.alloc(1)
            assert slot is not None, "Can not alloc int8 mamba checkpoint slot"
        return slot

    def _commit_int8_checkpoint(self, active_slots: torch.Tensor) -> torch.Tensor:
        """Quantize the active-pool state at ``active_slots`` into a fresh int8
        checkpoint slot and return that slot. Strategy-agnostic donate hook: both
        no_buffer (copy_from) and extra_buffer (ping-pong) converge here. The caller
        frees ``active_slots`` separately."""
        ckpt_slot = self._alloc_int8_ckpt_slot()
        self.int8_ckpt_pool.store_from_active(
            self.req_to_token_pool.mamba_pool, active_slots, ckpt_slot
        )
        return ckpt_slot

    def _free_mamba_value(self, mamba_value: torch.Tensor) -> None:
        """Free a node's mamba_value to the right allocator (int8 ckpt pool or the
        active mamba allocator)."""
        if self.int8_ckpt_pool is not None:
            self.int8_ckpt_pool.free(mamba_value)
        else:
            self.req_to_token_pool.mamba_allocator.free(mamba_value)

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        Mamba prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without mamba tombstone,
        or 2. the number of matching tokens from the matched node to the last mamba tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = key.child_key(self.page_size)

        value: List[torch.Tensor] = []
        best_value_len = 0
        best_last_node = node
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # update best_value_len and best_last_node if needed
            if node.mamba_value is not None:
                best_value_len = len(value)
                best_last_node = node

            prefix_len = child.key.match(key, page_size=self.page_size)
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
                    child_key = key.child_key(self.page_size)
        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if node.mamba_value is not None:
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:
        """Preprocess the key before matching."""
        key = params.key

        if self.disable or len(key) == 0:
            return None

        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the matched result."""
        cow_mamba = params.cow_mamba
        req = params.req

        # Full KV of the whole matched path is reused as prefix, so refresh the entire
        # chain (nodes closer to root end up least recently used, evicted first).
        node_update = last_node
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        # Mamba only consumes last_node's state (cf. inc_lock_ref, which locks just this
        # node's mamba_value). Refreshing ancestors would keep a whole session's states
        # adjacent in the mamba LRU and evict cold sessions wholesale; touch only the used
        # state so older leaves survive.
        if last_node is not self.root_node and last_node.mamba_value is not None:
            self.mamba_lru_list.reset_node_mru(last_node)

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        # Calculate the branching point. It is defined as the last aligned position that
        # does not have a mamba value.
        if len(value) > best_value_len:
            chunk_aligned_seqlen = (
                sum(len(v) for v in value) // self.mamba_cache_chunk_size
            ) * self.mamba_cache_chunk_size
            mamba_branching_seqlen = (
                chunk_aligned_seqlen if chunk_aligned_seqlen > 0 else None
            )
        else:
            mamba_branching_seqlen = None

        # Defer COW to forward stream: record source index, allocate destination
        if cow_mamba and last_node.mamba_value is not None:
            if req.mamba_pool_idx is None:
                dst_index = self.req_to_token_pool.mamba_allocator.alloc(1)
                if dst_index is None:
                    self.inc_lock_ref(last_node)
                    self.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.req_to_token_pool.mamba_allocator.alloc(1)
                    self.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                req.mamba_pool_idx = dst_index[0]
            req.mamba_cow_src_index = last_node.mamba_value
            req.mamba_needs_clear = False

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.mamba_value = None  # mamba cache can not be split
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = 0
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()

        # child time should be later than the new parent's time in the full LRU
        child.last_access_time = get_last_access_time()

        # A split does not change the set of live mamba states (child keeps its value,
        # new_node is a mamba tombstone), so the mamba LRU is left untouched — only the
        # full LRU reorders around the new intermediate node.
        self.full_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        # insert the new node and child into the full lru list, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        mamba_value,
        chunked: bool = False,
        prev_prefix_len: int = 0,
    ) -> Tuple[int, bool]:
        # Refresh the full LRU from root to leaf (the whole path is reused as prefix).
        # The mamba states of these existing nodes were not recomputed this insert, so
        # the mamba LRU is left untouched here; only genuinely new mamba states (the new
        # leaf / a revived tombstone below) are inserted.
        assert mamba_value is not None, "Mamba value should not be None here."
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0, True

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prev_prefix_len < total_prefix_length + prefix_len:
                start = max(0, prev_prefix_len - total_prefix_length)
                self.token_to_kv_pool_allocator.free(value[start:prefix_len])

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        mamba_value_exist = False
        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.mamba_value = mamba_value
            self.full_lru_list.insert_mru(new_node)
            self.mamba_lru_list.insert_mru(new_node)
            node.children[child_key] = new_node
            self.full_evictable_size_ += len(value)
            self.mamba_evictable_size_ += len(mamba_value)
            self._record_store_event(new_node)
        elif node.mamba_value is None:  # add for mamba tombstone
            node.mamba_value = mamba_value
            self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.insert_mru(node)
            self.mamba_evictable_size_ += len(mamba_value)
            node.last_access_time = get_last_access_time()
        else:  # mamba value already exists
            mamba_value_exist = True
            self.full_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()

        return total_prefix_length, mamba_value_exist

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.mamba_value is None and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.mamba_lock_ref == 0
            ), f"tombstone mamba_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.mamba_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens
            self._record_remove_event(node.parent)
            self.token_to_kv_pool_allocator.free(node.parent.value)
            full_num_evicted += len(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is not None
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)
        self.mamba_evictable_size_ -= len(node.mamba_value)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        self.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_value = None

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is None
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if cur_node.mamba_value is not None:
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
                f"[{current_node.id}]",
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"mr={current_node.mamba_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"mll={self.mamba_lru_list.in_list(current_node)}",
                f"mv={current_node.mamba_value}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_mamba_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if current_node.mamba_value is not None:
                total_mamba_size += len(current_node.mamba_value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_mamba_size
