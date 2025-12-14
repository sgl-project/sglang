import heapq
import logging
import time
from typing import Any, List, Optional, Tuple

from torch import Tensor

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.cache_controller_direct import (
    HiCacheControllerDirect,
    get_hash_list,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.metrics.collector import StorageMetricsCollector
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class HiRadixCacheDirect(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_device_allocator = params.token_to_kv_pool_allocator
        self.enable_storage = True
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.hicache_storage_pass_prefix_keys = False
        self.cache_controller = HiCacheControllerDirect(
            params.token_to_kv_pool_allocator,
            self.page_size,
            params.tp_cache_group,
            storage_backend=server_args.hicache_storage_backend,
            device_id=params.gpu_id,
        )
        if self.enable_storage_metrics:
            labels = {
                "storage_backend": server_args.hicache_storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
            }
            self.storage_metrics_collector = StorageMetricsCollector(labels=labels)

        # # record the node segments with ongoing load back
        self.ongoing_load_back = {}

        super().__init__(params=params)

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        super().reset()

    def clear_storage_backend(self) -> bool:
        try:
            # Check if the storage backend has a clear method (for nixl backends)
            if hasattr(self.cache_controller.storage_backend, "clear"):
                self.cache_controller.storage_backend.clear()
                return True
            else:
                logger.warning(
                    "hierarchical cache memcache store does not support clear operation."
                )
                return False
        except Exception as e:
            logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
            return False

    def evict(self, num_tokens: int):
        leaves = self._collect_leaves_device()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                num_evicted += self._evict_regular(x)

            for child in x.parent.children.values():
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def load_back(
        self,
        rid: str,
        last_hit_node: TreeNode,
        new_input_tokens: List[int],
        extra_key: Optional[str] = None,
        priority: int = 0,
    ) -> tuple[Tensor | None, TreeNode]:
        start_time = time.perf_counter()

        # protect the last_hit_node from eviction
        self.inc_lock_ref(last_hit_node)

        # alloc device memory
        new_input_len = len(new_input_tokens)
        device_indices = self.mem_pool_device_allocator.alloc(new_input_len)
        if device_indices is None:
            self.evict(new_input_len)
            device_indices = self.mem_pool_device_allocator.alloc(new_input_len)
            if device_indices is None:
                self.dec_lock_ref(last_hit_node)
                return None, last_hit_node

        # start to load kvcache from l3 into device hbm
        cached_device_indices, free_device_indices = self.cache_controller.load(
            rid=rid,
            new_input_tokens=new_input_tokens,
            device_indices=device_indices,
            last_hash=last_hit_node.get_last_hash_value(),
        )

        if free_device_indices is not None and cached_device_indices is not None:
            assert (len(cached_device_indices) + len(free_device_indices)) == len(
                device_indices
            )

        if free_device_indices is not None:
            self.mem_pool_device_allocator.free(free_device_indices)

        self.dec_lock_ref(last_hit_node)
        if cached_device_indices is None:
            return None, last_hit_node

        cached_token_len = len(cached_device_indices)
        inserted_len, new_node = self._insert(
            RadixKey(new_input_tokens[:cached_token_len], extra_key),
            cached_device_indices,
            last_hit_node,
            priority=priority,
        )

        if new_node is None:
            logger.error(f"====> failed to _insert: {len(cached_device_indices)=}")
            self.mem_pool_device_allocator.free(cached_device_indices)
            return None, last_hit_node

        if inserted_len != cached_token_len:
            self.mem_pool_device_allocator.free(cached_device_indices[inserted_len:])
            cached_device_indices = cached_device_indices[:inserted_len]
            cached_token_len = inserted_len

        self.ongoing_load_back[new_node.id] = new_node
        self.inc_lock_ref(new_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(cached_token_len)

        return cached_device_indices, new_node

    def init_load_back(
        self,
        last_host_node: Any,
        host_hit_length: int,
        req: Req,
    ) -> Tuple[Tensor | None, Any]:
        matched_len = len(req.prefix_indices)
        new_input_tokens = req.fill_ids[matched_len:]
        new_input_len = len(new_input_tokens)
        if new_input_len <= self.page_size:
            return None, req.last_node

        remainder = new_input_len % self.page_size
        if remainder == 0:
            # to avoid input tokens length = 0 while hit the entire input tokens
            new_input_tokens = new_input_tokens[: (new_input_len - self.page_size)]
        else:
            new_input_tokens = new_input_tokens[: (new_input_len - remainder)]

        last_node = req.last_node
        if not last_node.evicted:
            priority = getattr(req, "priority", 0) or 0
            loading_values, last_node = self.load_back(
                req.rid, last_node, new_input_tokens, req.extra_key, priority
            )
            if loading_values is not None:
                logger.debug(
                    f"init_load_back finished, "
                    f"{len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node
        else:
            # should not enter this branch
            logger.error(f"should not enter branch")
            while last_node.evicted:
                last_node = last_node.parent

        return None, last_node

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        """
        return -1

    def check_hicache_events(self):
        self.loading_check()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def loading_check(self):
        for node in self.ongoing_load_back.values():
            self.dec_lock_ref(node)
        self.ongoing_load_back.clear()

    def check_prefetch_progress(self, req_id: str) -> bool:
        return True

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
    ):
        pass

    def _insert(
        self,
        key: RadixKey,
        value: Tensor,
        parent,
        chunked=False,
        priority: int = 0,
    ):
        key, value = self.maybe_bigram_convert(key, value)
        if len(key) == 0:
            return len(value), None

        if self.is_eagle and value is not None:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        child_key = self.get_child_key_fn(key)

        new_node = TreeNode(priority=priority)
        new_node.parent = parent
        new_node.key = key
        new_node.value = value
        parent.children[child_key] = new_node
        self.evictable_size_ += len(value)

        if self.enable_storage:
            last_hash = parent.get_last_hash_value()
            assert (parent == self.root_node) or (
                last_hash is not None
            ), "Parent node must have a hash value with storage enabled"
            new_node.hash_value = get_hash_list(
                key.token_ids, last_hash, self.page_size
            )

        self._inc_hit_count(new_node, chunked)
        return len(value), new_node

    def insert(
        self,
        key: RadixKey,
        value=None,
        chunked=False,
        priority: int = 0,
    ) -> int:
        key, value = self.maybe_bigram_convert(key, value)
        if len(key) == 0:
            return 0

        if self.is_eagle and value is not None:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        origin_req_tokens = key.token_ids[:]
        origin_values = value.clone()

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0
        hash_keys = []
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                self._inc_hit_count(node, chunked)
                total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                total_prefix_length += prefix_len
                node = new_node

            if self.enable_storage:
                hash_keys.extend(node.hash_value)

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

            if self.enable_storage:
                last_hash = node.get_last_hash_value()
                assert (node == self.root_node) or (
                    last_hash is not None
                ), "Parent node must have a hash value with storage enabled"
                new_node.hash_value = get_hash_list(
                    key.token_ids, last_hash, self.page_size
                )
                hash_keys.extend(new_node.hash_value)

            self._inc_hit_count(new_node, chunked)

        if self.enable_storage:
            self.write_storage(origin_req_tokens, origin_values, hash_keys)

        return total_prefix_length

    def write_storage(
        self,
        origin_req_tokens,
        device_indices,
        hash_keys: List[str],
    ):
        token_num = len(origin_req_tokens)
        if token_num == 0:
            return
        assert token_num == len(device_indices)
        assert token_num == len(hash_keys) * self.page_size
        succ_num_tokens = self.cache_controller.write(hash_keys, device_indices)

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_backuped_tokens(succ_num_tokens)

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        if chunked:
            return
        node.hit_count += 1

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]

        if child.hash_value:
            new_node.hash_value = child.hash_value[: split_len // self.page_size]
            child.hash_value = child.hash_value[split_len // self.page_size :]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

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

    def release_aborted_request(self, rid: str):
        return
