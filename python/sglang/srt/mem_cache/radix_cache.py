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
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    BaseTokenToKVPool,
    MLATokenToKVPoolHost,
    ReqToTokenPool,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

# import logging

# logger = logging.getLogger(__name__)


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()

        self.hit_count = 0
        # indicating the node is written to host
        # self.writing = False
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
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: List[int], **kwargs) -> Tuple[torch.Tensor, int]:
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
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node)
        if value:
            value = torch.concat(value)
        else:
            value = torch.tensor([], dtype=torch.int32)
        return value, last_node[0]

    def insert(self, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it finishes."""
        if self.disable:
            if token_ids is None:
                token_ids_len = len(req.origin_input_ids) + len(req.output_ids) - 1
            else:
                token_ids_len = len(token_ids)

            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :token_ids_len
            ]
            self.token_to_kv_pool.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

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
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

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
                self.protected_size_ += len(node.value)
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
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
        self, node: TreeNode, key: List, value, last_node: TreeNode
    ):
        node.last_access_time = time.time()
        if len(key) == 0:
            return

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node
            else:
                value.append(child.value)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node)

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[0]] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            new_node = self._split_node(child.key, child, prefix_len)
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:]
            )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)
        return 0

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key[:10], f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self, node: TreeNode):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

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


class HiRadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
    ):
        self.token_to_kv_pool_host = MLATokenToKVPoolHost(token_to_kv_pool)
        self.cache_controller = HiCacheController(
            token_to_kv_pool, self.token_to_kv_pool_host
        )

        self.id_to_node = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = 1
        self.load_back_threshold = 10
        super().__init__(req_to_token_pool, token_to_kv_pool, disable=False)

    def reset(self):
        TreeNode.counter = 0
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def inc_hit_count(self, node: TreeNode):
        if self.cache_controller.write_policy != "write_through_selective":
            return
        node.hit_count += 1
        if node.host_value is None and node.hit_count > self.write_through_threshold:
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                priority=-self.get_height(node),
                node_id=node.id,
            )
            if host_indices is None:
                self.evict_host(len(node.value))
                host_indices = self.cache_controller.write(
                    device_indices=node.value,
                    priority=-self.get_height(node),
                    node_id=node.id,
                )
            if host_indices is not None:
                node.host_value = host_indices
                self.id_to_node[node.id] = node
                self.inc_lock_ref(node)
            else:
                # todo, check broken chains
                pass

    def evictable_size(self):
        if (
            self.cache_controller.write_policy == "write_through"
            or self.cache_controller.write_policy == "write_through_selective"
        ):
            while not self.cache_controller.ack_write_queue.empty():
                try:
                    ack_id = self.cache_controller.ack_write_queue.get_nowait()
                    self.dec_lock_ref(self.id_to_node[ack_id])
                    # clear the reference
                    del self.id_to_node[ack_id]
                except Exception:
                    break
        self.loading_check()
        return self.evictable_size_

    def evict(self, num_tokens: int, evict_callback=None):
        leaves = self._collect_leaves_device()
        heapq.heapify(leaves)

        num_evicted = 0
        pending_nodes = []
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x.lock_ref > 0:
                continue

            if x.host_value is None:
                if self.cache_controller.write_policy == "write_back":
                    num_evicted += self._evict_write_back(x)
                    if x.host_value is not None:
                        pending_nodes.append(x)
                elif self.cache_controller.write_policy == "write_through_selective":
                    num_evicted += self._evict_write_through_selective(x)
                else:
                    raise NotImplementedError
            else:
                # assert self.token_to_kv_pool_host.is_synced(x.host_value), (
                #     x.host_value,
                #     self.token_to_kv_pool_host.get_state(x.host_value),
                # )
                num_evicted += self._evict_write_through(x)

            for child in x.parent.children.values():
                if child in pending_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                heapq.heappush(leaves, x.parent)

        # blocking for write back completion
        while len(pending_nodes) > 0:
            for x in pending_nodes:
                if self.token_to_kv_pool_host.is_synced(x.host_value):
                    num_evicted = self.cache_controller.evict_device(
                        x.value, x.host_value
                    )
                    self.evictable_size_ -= num_evicted
                    x.value = None
                    pending_nodes.remove(x)
                    break
            else:
                time.sleep(0.1)

    def _evict_write_back(self, node: TreeNode):
        host_indices = self.cache_controller.write(node.value)
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(node.value)
        if host_indices is None:
            raise RuntimeError("No sufficient host memory available")
        else:
            node.host_value = host_indices
            return len(node.host_value)

    def _evict_write_through(self, node: TreeNode):
        # evict a node already written to host
        num_evicted = self.cache_controller.evict_device(node.value, node.host_value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        return num_evicted

    def _evict_write_through_selective(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break
            if not x.evicted:
                continue
            assert x.lock_ref == 0 and x.host_value is not None

            assert self.cache_controller.evict_host(x.host_value) > 0
            for k, v in x.parent.children.items():
                if v == x:
                    break
            del x.parent.children[k]

            if len(x.parent.children) == 0 and x.parent.evicted:
                heapq.heappush(leaves, x.parent)

    def load_back(self, node: TreeNode) -> Optional[torch.Tensor]:
        # todo: more loading policies

        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            last_node = node

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold:
            # skip loading back if the total size is too small
            return None

        self.inc_lock_ref(last_node)
        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(last_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.id_to_node[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)
        last_hit_node.loading = True

        return device_indices

    def loading_complete(self, node: TreeNode):
        self.loading_check()
        return node.loading

    def loading_check(self):
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack_id = self.cache_controller.ack_load_queue.get_nowait()
                node = self.id_to_node[ack_id]
                node.loading = False
                self.dec_lock_ref(node)
                del self.id_to_node[ack_id]
            except Exception:
                break

    def match_prefix(self, key: List, load_cache: bool = True, **kwargs):
        value, last_node = super().match_prefix(key, **kwargs)

        if last_node.evicted:
            if load_cache:
                loading_values = self.load_back(last_node)
                if loading_values is not None:
                    if len(value) == 0:
                        value = loading_values
                    else:
                        value = torch.cat([value, loading_values])

            while last_node.evicted:
                last_node = last_node.parent

        return value, last_node

    def _match_prefix_helper(
        self, node: TreeNode, key: List, value, last_node: TreeNode
    ):
        node.last_access_time = time.time()
        if len(key) == 0:
            return

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                self.inc_hit_count(new_node)
                if not new_node.evicted:
                    value.append(new_node.value)
                last_node[0] = new_node
            else:
                self.inc_hit_count(child)
                if not child.evicted:
                    value.append(child.value)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node)

    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.host_value is not None:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key[0]] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            if prefix_len == len(child.key):
                if child.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    child.value = value[:prefix_len]
                    self.token_to_kv_pool_host.update_synced(child.host_value)
                    self.evictable_size_ += len(value[:prefix_len])
                    return self._insert_helper(
                        child, key[prefix_len:], value[prefix_len:]
                    )
                else:
                    self.inc_hit_count(child)
                    return prefix_len + self._insert_helper(
                        child, key[prefix_len:], value[prefix_len:]
                    )

            # partial match, split the node
            new_node = self._split_node(child.key, child, prefix_len)
            if new_node.evicted:
                new_node.value = value[:prefix_len]
                self.token_to_kv_pool_host.update_synced(new_node.host_value)
                self.evictable_size_ += len(new_node.value)
                return self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:]
                )
            else:
                self.inc_hit_count(new_node)
                return prefix_len + self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:]
                )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)

            # todo: deduplication
            if self.cache_controller.write_policy == "write_through":
                new_node.host_value = self.cache_controller.write(
                    device_indices=value,
                    priority=-self.get_height(new_node),
                    node_id=new_node.id,
                )
                if new_node.host_value is None:
                    self.evict_host(len(value))
                    new_node.host_value = self.cache_controller.write(
                        device_indices=value,
                        priority=-self.get_height(new_node),
                        node_id=new_node.id,
                    )
                if new_node.host_value is None:
                    # todo, change raising error to a longer waiting
                    raise RuntimeError(
                        "No sufficient host memory available for write through"
                    )
                else:
                    # protect the nodes pending for write through
                    self.id_to_node[new_node.id] = new_node
                    self.inc_lock_ref(new_node)
        return 0

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list


if __name__ == "__main__":
    tree = RadixCache(None, None, False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
