import heapq
import logging
import threading
import time
import hashlib
from typing import List, Optional

import torch
import numpy as np

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MHATokenToKVPoolHost,
    MLATokenToKVPool,
    MLATokenToKVPoolHost,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

from sglang.srt.mem_cache.mooncake_store import MooncakeStore

logger = logging.getLogger(__name__)



def token_ids_to_key(
    token_ids: List,
    local_rank: int
):
    token_ids_bytes = np.array(token_ids).tobytes()
    hash_object = hashlib.blake2b(token_ids_bytes)
    hash_hex = hash_object.hexdigest()
    return f"{int(hash_hex[:16], 16)}_{local_rank}"


def page_token_ids_to_key(
    prefix_page_ids: List,
    current_page_ids: List,
    local_rank: int
):
    prefix_str = ""
    if len(prefix_page_ids) > 0:
        prefix_page_ids_bytes = np.array(prefix_page_ids).tobytes()
        prefix_hash_object = hashlib.blake2b(prefix_page_ids_bytes)
        prefix_hash_hex = prefix_hash_object.hexdigest()
        prefix_str = f"{int(prefix_hash_hex[:16], 16)}"

    current_token_ids_bytes = np.array(current_page_ids).tobytes()
    current_hash_object = hashlib.blake2b(current_token_ids_bytes)
    current_hash_hex = current_hash_object.hexdigest()

    return f"{prefix_str}_{int(current_hash_hex[:16], 16)}_{local_rank}"

def get_node_l3_keys(
    token_ids: List,
    token_len: int,
    local_rank: int = 0,
    page_size: int = 1,
):
    l3_keys = []
    total_token_len = len(token_ids)
    if page_size == 1:
        # 每个token的key构建需要加上完整的前缀
        for i in range(total_token_len - token_len, total_token_len):
            l3_keys.append(token_ids_to_key(token_ids[:i + 1], local_rank))
    else:
        total_block_len = len(token_ids) // page_size
        num_key = token_len // page_size
        for i in range(total_block_len - num_key, total_block_len):
            prefix_block_token_ids = token_ids[: i * page_size]
            current_block_token_ids = token_ids[i * page_size : (i + 1) * page_size]
            l3_keys.append(page_token_ids_to_key(prefix_block_token_ids, current_block_token_ids, local_rank))

    return l3_keys

class HiRadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
        enable_mooncake_store_l3_cache: bool
    ):
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache, hicache_ratio, hicache_size, page_size
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache, hicache_ratio, hicache_size, page_size
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.mooncake_l3_kv_pool = None
        self.mooncake_l3_load_cache_event = None
        self.enable_mooncake_store_l3_cache = enable_mooncake_store_l3_cache
        if enable_mooncake_store_l3_cache:
            # TODO(huangtingwei9988):L3 cache only support write_through_selective and write_through write policy
            assert hicache_write_policy in ["write_through_selective", "write_through"]
            self.mooncake_l3_kv_pool = MooncakeStore()
            self.mooncake_l3_load_cache_event = threading.Event()

        self.tp_group = tp_cache_group
        self.page_size = page_size

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            page_size,
            enable_mooncake_store_l3_cache,
            load_cache_event=self.load_cache_event,
            write_policy=hicache_write_policy,
            mooncake_l3_kv_pool=self.mooncake_l3_kv_pool,
            mooncake_l3_load_cache_event=self.mooncake_l3_load_cache_event
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 3
        )
        self.load_back_threshold = 10
        super().__init__(
            req_to_token_pool, token_to_kv_pool_allocator, page_size, disable=False
        )

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

    def write_backup(self, node: TreeNode, write_back=False, token_ids: Optional[List]=None):
        l3_keys = []
        if self.enable_mooncake_store_l3_cache:
            l3_keys = get_node_l3_keys(token_ids, len(node.value),
                                       torch.cuda.current_device(), self.page_size)

        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            l3_keys=l3_keys if self.enable_mooncake_store_l3_cache else None
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                l3_keys=l3_keys if self.enable_mooncake_store_l3_cache else None
            )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        if len(l3_keys) > 0:
            node.l3_keys = l3_keys

        return len(host_indices)

    def inc_hit_count(self, node: TreeNode, token_ids: Optional[List]=None):
        if node.backuped or self.cache_controller.write_policy == "write_back":
            return
        node.hit_count += 1
        if node.hit_count >= self.write_through_threshold:
            self.write_backup(node, token_ids=token_ids)
            node.hit_count = 0

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                ack_id = self.cache_controller.ack_write_queue.get()
                del self.ongoing_write_through[ack_id]
            return
        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue.qsize(), dtype=torch.int
        )
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        if self.enable_mooncake_store_l3_cache:
            # l2 and l3 are offloaded at the same time, so ack_id needs to be received twice
            ack_id_count = {}
            for _ in range(queue_size.item()):
                ack_id = self.cache_controller.ack_write_queue.get()
                if ack_id in ack_id_count.keys():
                    self.dec_lock_ref(self.ongoing_write_through[ack_id])
                    del self.ongoing_write_through[ack_id]
                    ack_id_count[ack_id] += 1
                    if ack_id_count[ack_id] > 2:
                        raise ValueError("ack_id error in hiRadixCache write check")
                else:
                    ack_id_count[ack_id] = 1
        else:
            for _ in range(queue_size.item()):
                ack_id = self.cache_controller.ack_write_queue.get()
                self.dec_lock_ref(self.ongoing_write_through[ack_id])
                del self.ongoing_write_through[ack_id]

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
        return self.evictable_size_

    def evict(self, num_tokens: int):
        leaves = self._collect_leaves_device()
        heapq.heapify(leaves)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                heapq.heappush(leaves, x.parent)

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

    def _evict_backuped(self, node: TreeNode):
        # evict a node already written to host
        num_evicted = self.cache_controller.evict_device(node.value, node.host_value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
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

            num_evicted += self.cache_controller.evict_host(x.host_value)

            for k, v in x.parent.children.items():
                if v == x:
                    break
            if len(x.parent.children[k].l3_keys()) == 0:
                del x.parent.children[k]

            if len(x.parent.children) == 0 and x.parent.evicted:
                heapq.heappush(leaves, x.parent)

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies

        last_hit_node = node
        l2_nodes_to_load = []
        l3_nodes_to_load = []

        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            if node.l2_backuped:
                l2_nodes_to_load.insert(0, node)
            if self.enable_mooncake_store_l3_cache:
                if not node.l2_backuped and node.l3_backuped:
                    l3_nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = []
        if len(l2_nodes_to_load) > 0:
            host_indices = torch.cat([n.host_value for n in l2_nodes_to_load])
        l3_keys = [key for n in l3_nodes_to_load for key in n.l3_keys]

        total_load_back_size = len(host_indices) + len(l3_keys) * self.page_size

        if total_load_back_size < self.load_back_threshold or (
            total_load_back_size > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id, l3_keys=l3_keys
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id, l3_keys=l3_keys
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (ancester_node, last_hit_node)
        offset = 0
        for node in l2_nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
            node.loading = True
        for node in l3_nodes_to_load:
            node.value = device_indices[offset: offset + len(node.l3_keys) * self.page_size]
            offset += len(node.l3_keys) * self.page_size
            node.loading = True
        self.evictable_size_ += total_load_back_size
        self.inc_lock_ref(last_hit_node)

        return device_indices

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

    def ready_to_load_cache(self):
        self.load_cache_event.set()
        if self.mooncake_l3_load_cache_event:
            self.mooncake_l3_load_cache_event.set()

    def match_prefix(self, key: List[int], include_evicted=False, **kwargs):
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.disable or len(key) == 0:
            if include_evicted:
                return empty_value, self.root_node, self.root_node
            else:
                return empty_value, self.root_node

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        last_node_global = last_node
        while last_node.evicted:
            last_node = last_node.parent

        if include_evicted:
            return value, last_node, last_node_global
        else:
            return value, last_node

    def _match_prefix_helper(self, node: TreeNode, key: List):
        total_key = key
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            total_prefix_length += prefix_len
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                self.inc_hit_count(new_node, token_ids=total_key[:total_prefix_length])
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                key = key[prefix_len:]
                break
            else:
                self.inc_hit_count(child, token_ids=total_key[:total_prefix_length])
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        if self.enable_mooncake_store_l3_cache:
            # try to get the cross instance shared kv cache
            if len(key) and (not node.evicted or node.backuped):
                l3_keys = get_node_l3_keys(total_key, len(key),
                                           torch.cuda.current_device(), self.page_size)
                l3_exist_keys = []
                for item in l3_keys:
                    key_ = f"{item}_{0}"
                    if self.mooncake_l3_kv_pool.is_exist(key_):
                        l3_exist_keys.append(item)
                    else:
                        break

                if len(l3_exist_keys) > 0:
                    child_key = self.get_child_key_fn(key[:len(l3_exist_keys) * self.page_size])
                    new_node = TreeNode()
                    new_node.parent = node
                    new_node.key = key[:len(l3_exist_keys) * self.page_size]
                    node.children[child_key] = new_node
                    new_node.l3_keys = l3_exist_keys
                    node = new_node

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
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
        if child.l2_backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        if child.l3_backuped:
            new_node.l3_keys = child.l3_keys[:split_len // self.page_size]
            child.l3_keys = child.l3_keys[split_len // self.page_size:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        total_key = key
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len]
                    self.token_to_kv_pool_host.update_synced(node.host_value)
                    self.evictable_size_ += len(node.value)
                else:
                    total_prefix_length += prefix_len
                    self.inc_hit_count(node, token_ids=total_key[:total_prefix_length])
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                if new_node.evicted:
                    new_node.value = value[:prefix_len]
                    self.token_to_kv_pool_host.update_synced(new_node.host_value)
                    self.evictable_size_ += len(new_node.value)
                else:
                    total_prefix_length += prefix_len
                    self.inc_hit_count(new_node, token_ids=total_key[:total_prefix_length])
                node = new_node

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
            self.evictable_size_ += len(value)

            if self.cache_controller.write_policy != "write_back":
                self.inc_hit_count(new_node, token_ids=total_key)
        return total_prefix_length

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
