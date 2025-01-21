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
from typing import TYPE_CHECKING, Callable, List, Optional
import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        disable: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.disable = disable
        self.root_node = None
        self.evictable_size_ = 0
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0

    def _match_prefix_helper(self, key: List):
        current_node = self.root_node
        current_node.last_access_time = time.time()
        # key: the rest elements of key those are not compared or not part of prefix

        prefix_len = 0
        total_length = 0
        current_key = key
        while len(current_key) > 0:
            prefix_len = _key_match(current_node.key, current_key)
            total_length += prefix_len
            if prefix_len == len(current_node.key) and \
                    len(current_key) > prefix_len and \
                    current_key[prefix_len] in current_node.children.keys():
                current_node = current_node.children[current_key[prefix_len]]
                current_node.last_access_time = time.time()
                current_key = current_key[prefix_len:]
            else:
                break

        matched_value = key[:total_length]
        return matched_value, current_node, prefix_len
    
    def match_prefix(self, key: List, **kwargs):
        if self.disable:
            return torch.tensor([], dtype=torch.int32), self.root_node
        val, last_node, _ = self._match_prefix_helper(key)
        return val, last_node

    def insert(self, key: List, value=None):
        if self.disable:
            return 0
        if value is None:
            value = key

        assert hasattr(value, "__getitem__")
        assert hasattr(value, "__len__")
        assert len(value) == len(key)

        current_node = self.root_node
        current_node.last_access_time = time.time()
        match_value, current_node, last_prefix_len = self._match_prefix_helper(key)
        total_match_length = len(match_value)
        if last_prefix_len != len(current_node.key) or total_match_length != len(key):
            self._insert_node(key[total_match_length - last_prefix_len:], value[total_match_length - last_prefix_len:], current_node, last_prefix_len)
        return total_match_length

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it finishes."""
        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.disable:
            self.token_to_kv_pool.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices): new_prefix_len])

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if token_ids is None:
            token_ids = req.fill_ids

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(token_ids)
        assert len(new_indices) == len(token_ids)
        self.req_to_token_pool.req_to_token[
            req.req_pool_idx, len(req.prefix_indices): len(new_indices)
        ] = new_indices[len(req.prefix_indices): ]

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens: int, evict_callback: Callable):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            evict_callback(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    @staticmethod
    def _insert_node(key, value, child: TreeNode, split_len: int):
        if split_len < len(child.value):
            append_node = TreeNode()
            append_node.key = child.key[split_len:]
            append_node.value = child.value[split_len:]
            append_node.parent = child
            append_node.lock_ref = child.lock_ref
            append_node.children = child.children
            child.value = child.value[:split_len]
            child.key = child.key[:split_len]
            child.children = {append_node.key[0]: append_node}
        new_node = child

        if split_len < len(key):
            append_node = TreeNode()
            append_node.key = key[split_len:]
            append_node.value = value[split_len:]
            append_node.parent = new_node
            child_dict = {key[split_len]: append_node}
            new_node.children.update(child_dict)
        return new_node

    @staticmethod
    def _print_helper(node: TreeNode, indent: int):
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(" " * current_indent, len(current_node.key), current_node.key[:10], f"r={current_node.lock_ref}")
            for _, child in current_node.children.items():
                stack.append((child, current_indent + 2))

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    @staticmethod
    def _total_size_helper(node: TreeNode):
        total_size = 0
        stack = [node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            stack.extend(current_node.children.values())
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list


if __name__ == "__main__":
    tree = RadixCache(None, None, False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    tree.insert("Hello_world! Happy")
    tree.insert("I love you!")
    tree.pretty_print()

    print(tree.match_prefix("I love her!")[1].key)

    tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))
    #
    def evict_callback(x):
       print("evict", x)
       return len(x)

    tree.evict(5, evict_callback)
    tree.evict(10, evict_callback)
    tree.pretty_print()
