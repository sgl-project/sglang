import heapq
import logging
import time
from typing import List, Optional

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.memory_pool import (
    BaseTokenToKVPool,
    MLATokenToKVPoolHost,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode, _key_match

logger = logging.getLogger(__name__)


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

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = 1
        self.load_back_threshold = 10
        super().__init__(req_to_token_pool, token_to_kv_pool, disable=False)

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def write_backup(self, node: TreeNode):
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
            self.ongoing_write_through[node.id] = node
            self.inc_lock_ref(node)
        else:
            return None

        return len(host_indices)

    def inc_hit_count(self, node: TreeNode):
        if self.cache_controller.write_policy != "write_through_selective":
            return
        node.hit_count += 1
        if node.host_value is None and node.hit_count > self.write_through_threshold:
            self.write_backup(node)
            node.hit_count = 0

    def writing_check(self):
        while not self.cache_controller.ack_write_queue.empty():
            try:
                ack_id = self.cache_controller.ack_write_queue.get_nowait()
                self.dec_lock_ref(self.ongoing_write_through[ack_id])
                # clear the reference
                del self.ongoing_write_through[ack_id]
            except Exception:
                break

    def loading_check(self):
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack_id = self.cache_controller.ack_load_queue.get_nowait()
                start_node, end_node = self.ongoing_load_back[ack_id]
                self.dec_lock_ref(end_node)
                while end_node != start_node:
                    assert end_node.loading
                    end_node.loading = False
                    end_node = end_node.parent
                # clear the reference
                del self.ongoing_load_back[ack_id]
            except Exception:
                break

    def evictable_size(self):
        self.writing_check()
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
                    num_evicted += self.write_backup(x)
                elif self.cache_controller.write_policy == "write_through_selective":
                    num_evicted += self._evict_write_through_selective(x)
                else:
                    assert (
                        self.cache_controller.write_policy != "write_through"
                    ), "write_through should be inclusive"
                    raise NotImplementedError
            else:
                num_evicted += self._evict_write_through(x)

            for child in x.parent.children.values():
                if child in pending_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                heapq.heappush(leaves, x.parent)

        if self.cache_controller.write_policy == "write_back":
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                self.writing_check()
                time.sleep(0.1)

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
            # only evict the host value of evicted nodes
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

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
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
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (ancester_node, last_hit_node)
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
            node.loading = True
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices

    def loading_complete(self, node: TreeNode):
        self.loading_check()
        return node.loading == False

    def init_load_back(
        self,
        last_node: TreeNode,
        prefix_indices: torch.Tensor,
        mem_quota: Optional[int] = None,
    ):
        assert (
            len(prefix_indices) == 0 or prefix_indices.is_cuda
        ), "indices of device kV caches should be on GPU"
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                prefix_indices = (
                    loading_values
                    if len(prefix_indices) == 0
                    else torch.cat([prefix_indices, loading_values])
                )
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )

            while last_node.evicted:
                last_node = last_node.parent

        return last_node, prefix_indices

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
        new_node.loading = child.loading

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

            if self.cache_controller.write_policy == "write_through":
                self.write_backup(new_node)
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
