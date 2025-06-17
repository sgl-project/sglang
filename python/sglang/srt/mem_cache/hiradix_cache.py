import heapq
import logging
import threading
import time
from typing import List, Optional

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    HiCacheControllerDisk,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MHATokenToKVPoolDisk,
    MHATokenToKVPoolHost,
    MLATokenToKVPool,
    MLATokenToKVPoolDisk,
    MLATokenToKVPoolHost,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

logger = logging.getLogger(__name__)


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
        use_disk: bool,
        disk_path: float,
        disk_ratio: float,
        disk_size: int,
        disk_rank: int = 0,
    ):
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.hicache_write_policy = hicache_write_policy
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

        self.tp_group = tp_cache_group

        self.load_cache_event = threading.Event()

        self.use_disk = use_disk
        logger.info(f"HiRadixCache {use_disk=}")

        self._init_disk_pool(disk_path, disk_ratio, disk_size, disk_rank)
        self._init_cache_controller()

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

    def _init_disk_pool(
        self, disk_path: float, disk_ratio: float, disk_size: int, disk_rank: int
    ) -> None:
        return

    def _init_cache_controller(self) -> None:
        self.cache_controller = HiCacheController(
            self.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            load_cache_event=self.load_cache_event,
            write_policy=self.hicache_write_policy,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()

        self.ongoing_write_through.clear()
        self.ongoing_load_back.clear()
        self.token_to_kv_pool_host.clear()
        self._reset_disk()
        super().reset()

    def _reset_disk(self) -> None:
        return

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def write_backup(self, node: TreeNode, write_back=False, write_disk: bool = True):
        device_indices = node.value.clone()
        device_indices = device_indices.cpu()
        host_indices = self.cache_controller.write(
            device_indices=device_indices,
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=device_indices,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        return self._write_backup_disk(
            node, host_indices, device_indices, write_back, write_disk
        )

    def _write_backup_disk(
        self,
        node: TreeNode,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        write_back: bool,
        write_disk: bool,
    ):
        return len(host_indices)

    def inc_hit_count(self, node: TreeNode):
        if node.backuped or self.cache_controller.write_policy == "write_back":
            return
        if self._backuped_disk(node):
            return
        node.hit_count += 1
        if node.hit_count >= self.write_through_threshold:
            self.write_backup(node)
            node.hit_count = 0

    def writing_check_helper(self, write_back=False):
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
        for _ in range(queue_size.item()):
            ack_id = self.cache_controller.ack_write_queue.get()
            self.dec_lock_ref(self.ongoing_write_through[ack_id])
            del self.ongoing_write_through[ack_id]

    def loading_check_helper(self):
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

    def writing_check(self, write_back=False, block=False):
        if not block:
            self.writing_check_helper(write_back)
            return

        while len(self.ongoing_write_through) > 0:
            self.writing_check_helper(write_back)
            time.sleep(0.1)

    def loading_check(self, block=False):
        if not block:
            self.loading_check_helper()
            return

        while len(self.ongoing_load_back) > 0:
            self.loading_check_helper()
            time.sleep(0.1)

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

            if not x.backuped and not self._backuped_disk(x):
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
        leaves = self._collect_leaves_host()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            if x.lock_ref > 0:
                continue

                self._evict_host_helper(leaves, x)

    def _collect_leaves_host(self) -> List[TreeNode]:
        return self._collect_leaves()

    def _evict_host_helper(self, leaves: List[TreeNode], x: TreeNode):
        num_evicted += self.cache_controller.evict_host(x.host_value)

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

        origin_node = node
        nodes_to_load, nodes_to_load_from_disk, ancester_node, last_hit_node = (
            self._load_back_prepare(node)
        )

        device_indices = None
        node = origin_node

        if len(nodes_to_load) != 0 or len(nodes_to_load_from_disk) != 0:
            # protect the ancestor nodes from eviction
            delta = self.inc_lock_ref(ancester_node)
        else:
            return device_indices

        # 1. host2device
        if len(nodes_to_load) != 0:
            device_indices = self.load_back_host2device(
                ancester_node, last_hit_node, nodes_to_load, mem_quota, delta
            )
            if device_indices is None:
                self.dec_lock_ref(ancester_node)
                return device_indices

        if not self.use_disk or len(nodes_to_load_from_disk) == 0:
            self.dec_lock_ref(ancester_node)
            return device_indices

        # 2. disk2xxx
        device_indices_from_disk = self._load_back_disk2xxx(
            last_hit_node, node, nodes_to_load_from_disk
        )

        if device_indices is None:
            device_indices = device_indices_from_disk
        elif device_indices_from_disk is None:
            pass
        else:
            device_indices = torch.cat([device_indices, device_indices_from_disk])

        self.dec_lock_ref(ancester_node)
        return device_indices

    def _load_back_prepare(
        self,
        node: TreeNode,
    ):
        last_hit_node = node
        last_hit_recorded = False  # indicates if the last host node has been recorded
        nodes_to_load = []
        nodes_to_load_from_disk = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node
            if not last_hit_recorded:
                last_hit_node = ancester_node

        return nodes_to_load, nodes_to_load_from_disk, ancester_node, last_hit_node

    def load_back_host2device(
        self,
        ancester_node: TreeNode,
        last_hit_node: TreeNode,
        nodes_to_load: List[TreeNode],
        mem_quota: Optional[int],
        delta: int,
    ):
        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
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

    def _load_back_disk2xxx(
        self,
        last_hit_node: TreeNode,
        node: TreeNode,
        nodes_to_load_from_disk: List[TreeNode],
    ):
        return None

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
        self._ready_to_load_cache_disk()

    def _ready_to_load_cache_disk(self):
        return

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
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                self.inc_hit_count(new_node)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                self.inc_hit_count(child)
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

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
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]

        self._split_node_disk(new_node, child, split_len)

        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _split_node_disk(self, new_node: TreeNode, child: TreeNode, split_len: int):
        return

    def _insert_helper(self, node: TreeNode, key: List, value):
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
                    self._insert_helper_disk(node)
                    self.token_to_kv_pool_host.update_synced(node.host_value)
                    self.evictable_size_ += len(node.value)
                else:
                    self.inc_hit_count(node)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                if new_node.evicted:
                    new_node.value = value[:prefix_len]
                    self._insert_helper_disk(new_node)
                    self.token_to_kv_pool_host.update_synced(new_node.host_value)
                    self.evictable_size_ += len(new_node.value)
                else:
                    self.inc_hit_count(new_node)
                    total_prefix_length += prefix_len
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
                self.inc_hit_count(new_node)
        return total_prefix_length

    def _insert_helper_disk(self, node: TreeNode):
        return

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

    def _backuped_disk(self, x: TreeNode):
        return False


class HiRadixCacheDisk(HiRadixCache):
    def _init_disk_pool(
        self, disk_path: float, disk_ratio: float, disk_size: int, disk_rank: int
    ):
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.memory_pool_disk = MHATokenToKVPoolDisk(
                self.kv_cache,
                disk_path,
                disk_ratio,
                disk_size,
                self.page_size,
                disk_rank,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.memory_pool_disk = MLATokenToKVPoolDisk(
                self.kv_cache,
                disk_path,
                disk_ratio,
                disk_size,
                self.page_size,
                disk_rank,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.ongoing_write_through_device2disk = {}
        self.ongoing_load_back_disk2host = {}
        self.ongoing_load_back_disk2device = {}
        self.load_back_threshold_disk2host = 10
        self.load_back_threshold_disk2device = 10
        self.load_cache_event_disk2device = threading.Event()
        self.load_cache_event_disk2host = threading.Event()

    def _init_cache_controller(self):
        self.cache_controller = HiCacheControllerDisk(
            self.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.memory_pool_disk,
            self.page_size,
            load_cache_event=self.load_cache_event,
            load_cache_event_disk2device=self.load_cache_event_disk2device,
            load_cache_event_disk2host=self.load_cache_event_disk2host,
            write_policy=self.hicache_write_policy,
        )

    def _reset_disk(self):
        self.ongoing_write_through_device2disk.clear()
        self.ongoing_load_back_disk2host.clear()
        self.ongoing_load_back_disk2device.clear()
        self.memory_pool_disk.clear()

    def _write_backup_disk(
        self,
        node: TreeNode,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        write_back: bool,
        write_disk: bool,
    ):
        if not write_disk:
            return len(host_indices)

        # write to disk
        disk_indices = self.cache_controller.write_device2disk(
            device_indices=device_indices,
            node_id=node.id,
        )
        if disk_indices is None:
            self.evict_disk(len(node.value))
            disk_indices = self.cache_controller.write_device2disk(
                device_indices=device_indices,
                node_id=node.id,
            )
        if disk_indices is not None:
            node.disk_value = disk_indices
            self.ongoing_write_through_device2disk[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        return len(disk_indices)

    def _load_back_prepare(
        self,
        node: TreeNode,
    ):
        last_hit_node = node
        nodes_to_load = []
        last_hit_recorded = False  # indicates if the last host node has been recorded
        nodes_to_load_from_disk = []
        while node.evicted:
            # for disk
            if self.use_disk:
                if node.backuped_disk and not node.backuped:
                    assert not last_hit_recorded
                    nodes_to_load_from_disk.insert(0, node)
                    node = node.parent
                    continue
                elif not last_hit_recorded:
                    last_hit_recorded = True
                    last_hit_node = node  # record the last host node
            # for host
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node
            if not last_hit_recorded:
                last_hit_node = ancester_node

        return nodes_to_load, nodes_to_load_from_disk, ancester_node, last_hit_node

    def _load_back_disk2xxx(
        self,
        last_hit_node: TreeNode,
        node: TreeNode,
        nodes_to_load_from_disk: List[TreeNode],
    ):
        # disk2xxx
        disk_indices = torch.cat([n.disk_value for n in nodes_to_load_from_disk])

        device_indices_from_disk = None

        # 2. disk2device
        if len(disk_indices) < self.load_back_threshold_disk2device:
            return device_indices_from_disk

        device_indices_from_disk = self.cache_controller.load_disk2device(
            disk_indices=disk_indices, node_id=node.id
        )
        if device_indices_from_disk is None:
            self.evict(len(disk_indices))
            device_indices_from_disk = self.cache_controller.load_disk2device(
                disk_indices=disk_indices, node_id=node.id
            )

        if device_indices_from_disk is None:
            # no sufficient device memory to load back KV caches
            return device_indices_from_disk

        self.ongoing_load_back_disk2device[node.id] = (last_hit_node, node)
        offset = 0
        for node in nodes_to_load_from_disk:
            node.value = device_indices_from_disk[
                offset : offset + len(node.disk_value)
            ]
            offset += len(node.disk_value)
            node.loading_disk2device = True
        self.evictable_size_ += len(device_indices_from_disk)
        self.inc_lock_ref(node)

        # 3. disk2host
        if len(disk_indices) < self.load_back_threshold_disk2host:
            return device_indices_from_disk

        host_indices = self.cache_controller.load_disk2host(
            disk_indices=disk_indices, node_id=node.id
        )
        if host_indices is None:
            self.evict_host(len(disk_indices))
            host_indices = self.cache_controller.load_disk2host(
                disk_indices=disk_indices, node_id=node.id
            )

        if host_indices is None:
            # no sufficient host memory to load back KV caches
            return device_indices_from_disk

        self.ongoing_load_back_disk2host[node.id] = (last_hit_node, node)
        offset = 0
        for node in nodes_to_load_from_disk:
            node.host_value = host_indices[offset : offset + len(node.disk_value)]
            offset += len(node.disk_value)
            node.loading_disk2host = True
        self.inc_lock_ref(node)

        return device_indices_from_disk

    def _ready_to_load_cache_disk(self):
        self.load_cache_event_disk2device.set()
        self.load_cache_event_disk2host.set()

    def _writing_check_disk_helper(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                ack_id = self.cache_controller.ack_write_queue.get()
                del self.ongoing_write_through[ack_id]
            return

        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue_device2disk.qsize(), dtype=torch.int
        )
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        for _ in range(queue_size.item()):
            ack_id = self.cache_controller.ack_write_queue_device2disk.get()
            self.dec_lock_ref(self.ongoing_write_through_device2disk[ack_id])
            del self.ongoing_write_through_device2disk[ack_id]

    def _loading_check_disk_helper(self):
        def _loading_check_helper(ack_load_queue, ongoing_load_back, loading_check):
            while not ack_load_queue.empty():
                try:
                    ack_id = ack_load_queue.get_nowait()
                    start_node, end_node = ongoing_load_back[ack_id]
                    self.dec_lock_ref(end_node)
                    while end_node != start_node:
                        if loading_check == 0:
                            assert end_node.loading_disk2host
                            end_node.loading_disk2host = False
                        elif loading_check == 1:
                            assert end_node.loading_disk2device
                            end_node.loading_disk2device = False
                        end_node = end_node.parent
                    # clear the reference
                    del ongoing_load_back[ack_id]
                except Exception:
                    break

        # check disk2host
        _loading_check_helper(
            self.cache_controller.ack_load_queue_disk2host,
            self.ongoing_load_back_disk2host,
            0,
        )
        # check disk2device
        _loading_check_helper(
            self.cache_controller.ack_load_queue_disk2device,
            self.ongoing_load_back_disk2device,
            1,
        )

    def writing_check(self, write_back=False, block=False):
        if not block:
            self.writing_check_helper(write_back)
            self._writing_check_disk_helper(write_back)
            return

        while (
            len(self.ongoing_write_through)
            + (len(self.ongoing_write_through_device2disk) if self.use_disk else 0)
        ) > 0:
            self.writing_check_helper(write_back)
            self._writing_check_disk_helper(write_back)
            time.sleep(0.1)

    def loading_check(self, block=False):
        if not block:
            self.loading_check_helper()
            self._loading_check_disk_helper()
            return

        while (
            len(self.ongoing_load_back)
            + (
                len(self.ongoing_load_back_disk2device)
                + len(self.ongoing_load_back_disk2host)
            )
            if self.use_disk
            else 0
        ) > 0:
            self.loading_check_helper()
            self._loading_check_disk_helper()
            time.sleep(0.1)

    def _split_node_disk(self, new_node: TreeNode, child: TreeNode, split_len: int):
        new_node.loading_disk2host = child.loading_disk2host
        new_node.loading_disk2device = child.loading_disk2device
        if child.backuped_disk:
            new_node.disk_value = child.disk_value[:split_len]
            child.disk_value = child.disk_value[split_len:]

    def _insert_helper_disk(self, node: TreeNode):
        if node.host_value is None:
            assert node.disk_value is not None
            self.write_backup(node, write_disk=False)

    def _evict_host_helper(self, leaves: List[TreeNode], x: TreeNode):
        assert x.backuped and x.backuped_disk, "the node's value should be on disk"

        self._evict_host_backuped_disk(x)

        if self._is_host_leaf(x.parent):
            heapq.heappush(leaves, x.parent)

    def _evict_host_backuped_disk(self, node: TreeNode):
        # evict a node's host that already written to disk
        num_evicted = self.cache_controller.evict_host(node.host_value)
        assert num_evicted > 0
        node.host_value = None
        return num_evicted

    def evict_disk(self, num_tokens: int):
        leaves = self._collect_leaves(self._is_disk_leaf)
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break

            assert (
                x.evicted and not x.backuped and x.backuped_disk
            ), f"The node value should only be on disk. {x.evicted=} {x.backuped=} {x.backuped_disk=}"

            num_evicted += self.cache_controller.evict_disk(x.disk_value)

            for k, v in x.parent.children.items():
                if v == x:
                    break
            del x.parent.children[k]

            if self._is_disk_leaf(x.parent):
                heapq.heappush(leaves, x.parent)

    def _collect_leaves_host(self) -> List[TreeNode]:
        return self._collect_leaves(self._is_host_leaf)

    def _collect_leaves(self, is_leaf_fn) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf_fn(cur_node):
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())
        return ret_list

    def _is_host_leaf(self, node: TreeNode):
        """
        If the node's value is only on host, and all its children's value are on disk
        """
        if any(
            [
                not node.evicted,  # the node's value is on device
                not node.backuped,  # the node's value is not on host
                not node.backuped_disk,  # the node's value is not on disk
            ]
        ):
            return False
        if node == self.root_node:
            return False
        if len(node.children) == 0:
            return True
        for child in node.children.values():
            assert child.evicted, "child should be evicted"
            if not child.backuped_disk or child.backuped:
                return False
        return True

    def _is_disk_leaf(self, node: TreeNode):
        """
        If the node's value is only on disk and it has no children
        """
        if (
            node.evicted
            and not node.backuped
            and node.backuped_disk
            and len(node.children) == 0
        ):
            return True
        return False

    def _backuped_disk(self, x: TreeNode):
        return self.use_disk and x.backuped_disk
