from __future__ import annotations

import atexit
import heapq
import json
import logging
import os
import threading
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.mamba_radix_cache import (
    LRUList,
    MambaRadixCache,
    TreeNode,
    get_last_access_time,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    MambaPoolHost,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    PoolEntry,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import StorageMetricsCollector

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class HostLRUList(LRUList):
    def __init__(self):
        super().__init__(mamba=True)
        self.prv = "host_mamba_prev"
        self.nxt = "host_mamba_next"
        setattr(self.head, self.nxt, self.tail)
        setattr(self.tail, self.prv, self.head)

    def reset_node_mru(self, node):
        assert node.id in self.cache, f"Resetting node {node.id=} not in host mamba lru"
        assert (
            node.mamba_host_value is not None
        ), f"Resetting host mamba tombstone node in lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def insert_mru(self, node):
        assert (
            node.mamba_host_value is not None
        ), f"Inserting host mamba tombstone node in lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in host mamba lru list"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        assert node.id in self.cache, f"Removing node {node.id=} not in host mamba lru"
        assert (
            node.mamba_host_value is not None
        ), f"Removing host mamba tombstone node from lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)


class HiMambaRadixCache(MambaRadixCache):
    """Hierarchical cache for hybrid Mamba models."""

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics
        if server_args.hicache_io_backend == "direct":
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, "
                    "switching to page first direct layout"
                )

        self.page_size = params.page_size
        self.hybrid_kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(self.hybrid_kv_cache, HybridLinearKVPool):
            raise ValueError(
                "HiMambaRadixCache requires HybridLinearKVPool for hybrid SSM models."
            )
        if not isinstance(params.req_to_token_pool, HybridReqToTokenPool):
            raise ValueError(
                "HiMambaRadixCache requires HybridReqToTokenPool for hybrid SSM models."
            )

        self.kvcache = self.hybrid_kv_cache.full_kv_pool
        kv_host_pool_cls = (
            MLATokenToKVPoolHost
            if self.hybrid_kv_cache.use_mla
            else MHATokenToKVPoolHost
        )
        self.full_kv_pool_host = kv_host_pool_cls(
            self.kvcache,
            server_args.hicache_ratio,
            server_args.hicache_size,
            params.page_size,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        self.mamba_pool_host = MambaPoolHost(
            params.req_to_token_pool.mamba_pool,
            server_args.hicache_ratio,
            server_args.hicache_size,
            allocator_type=server_args.hicache_storage_backend,
            layout=server_args.hicache_mem_layout,
        )

        full_layer_ids = sorted(
            self.hybrid_kv_cache.full_attention_layer_id_mapping.keys()
        )
        mamba_layer_ids = sorted(params.req_to_token_pool.mamba_map.keys())
        self.transfer_layer_num = len(set(full_layer_ids) | set(mamba_layer_ids))

        full_layer_mapping = dict(self.hybrid_kv_cache.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        transfer_layer_num = self.transfer_layer_num

        def kv_layer_mapper(layer_id: int) -> Optional[int]:
            if not 0 <= layer_id < transfer_layer_num:
                return None
            return full_layer_mapping.get(layer_id)

        def mamba_layer_mapper(layer_id: int) -> Optional[int]:
            if not 0 <= layer_id < transfer_layer_num:
                return None
            return mamba_layer_mapping.get(layer_id)

        self.host_pool_group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=self.full_kv_pool_host,
                    device_pool=self.kvcache,
                    layer_mapper=kv_layer_mapper,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.MAMBA,
                    host_pool=self.mamba_pool_host,
                    device_pool=params.req_to_token_pool.mamba_pool,
                    layer_mapper=mamba_layer_mapper,
                    host_evict_fn=self.evict_mamba_host,
                    device_evict_fn=self.evict_mamba,
                ),
            ]
        )

        self.tp_group = params.tp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )

        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        self.cache_controller = HybridCacheController(
            params.token_to_kv_pool_allocator,
            self.host_pool_group,
            params.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            transfer_layer_num=self.transfer_layer_num,
        )
        params.req_to_token_pool.register_layer_transfer_counter(
            self.cache_controller.layer_done_counter
        )
        self.hybrid_kv_cache.register_layer_transfer_counter(
            self.cache_controller.layer_done_counter
        )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # track per-request tokens loaded from storage (L3 hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}

        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        self.evictable_full_device_leaves: set[TreeNode] = set()
        self.evictable_full_host_leaves: set[TreeNode] = set()
        self.mamba_host_lru_list = HostLRUList()

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        super().__init__(params=params)

    def reset(self) -> None:
        TreeNode.counter = 0
        self._flush_pending_storage_backups_before_reset()
        self.cache_controller.reset()
        self.full_kv_pool_host.clear()
        self.mamba_pool_host.clear()
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_full_device_leaves.clear()
        self.evictable_full_host_leaves.clear()
        self.mamba_host_lru_list = HostLRUList()
        logger.info(
            "HiMambaRadixCache reset completed: host_kv_available=%s host_mamba_available=%s",
            self.full_kv_pool_host.available_size(),
            self.mamba_pool_host.available_size(),
        )
        super().reset()

    def write_backup(self, node: TreeNode, write_back=False):
        # If mamba host slot already exists, refresh its LRU position.
        if node.mamba_value is not None and node.mamba_host_value is not None:
            if self.mamba_host_lru_list.in_list(node):
                self.mamba_host_lru_list.reset_node_mru(node)

        extra_pools = self.mamba_backup_transfers(node)
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            extra_pools=extra_pools,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                extra_pools=extra_pools,
            )
        if host_indices is not None:
            node.host_value = host_indices
            if extra_pools is not None:
                self.mamba_backup_commit(node, extra_pools)
            assert len(node.host_value) > 0
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None, req=None
    ) -> Optional[torch.Tensor]:
        """Load full KV back from host."""
        last_hit_node = node
        nodes_to_load = []

        while node.evicted:
            assert node.backuped, f"No backup on evicted node {node.id}"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancestor_node = node

        mamba_restore_nodes = []
        if last_hit_node.mamba_backuped and last_hit_node.mamba_evicted:
            mamba_restore_nodes.append(last_hit_node)

        result = self.inc_lock_ref(ancestor_node)
        delta = result.delta

        if nodes_to_load:
            full_host_indices = torch.cat([n.host_value for n in nodes_to_load])
        else:
            full_host_indices = torch.empty((0,), dtype=torch.int64, device="cpu")

        if (
            len(full_host_indices) > 0
            and (
                (len(full_host_indices) < self.load_back_threshold)
                or (
                    len(full_host_indices) > mem_quota + delta
                    if mem_quota is not None
                    else False
                )
            )
            and len(mamba_restore_nodes) == 0
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancestor_node)
            return None

        logger.debug(
            f"Init load back from cpu -> gpu, kv hit length: {len(full_host_indices)}, mamba host hit length: {len(mamba_restore_nodes)}"
        )
        mamba_pools = self.mamba_restore_transfers(
            last_hit_node, mamba_restore_nodes, req
        )
        full_device_indices = self.cache_controller.load(
            host_indices=full_host_indices,
            node_id=last_hit_node.id,
            extra_pools=mamba_pools,
        )
        if full_device_indices is None:
            if len(full_host_indices) > 0:
                self.evict(EvictParams(num_tokens=len(full_host_indices)))

            mamba_pools = self.mamba_restore_transfers(
                last_hit_node, mamba_restore_nodes, req
            )
            full_device_indices = self.cache_controller.load(
                host_indices=full_host_indices,
                node_id=last_hit_node.id,
                extra_pools=mamba_pools,
            )
        self.dec_lock_ref(ancestor_node)
        if full_device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.mamba_restore_commit(mamba_restore_nodes, mamba_pools)

        offset = 0
        for n in nodes_to_load:
            n_len = len(n.host_value)
            n.value = full_device_indices[offset : offset + n_len].clone()
            offset += n_len

            self.full_lru_list.insert_mru(n)
            self.full_evictable_size_ += n_len
            self._update_leaf_status(n)

        for n in mamba_restore_nodes:
            if self.mamba_lru_list.in_list(n):
                self.mamba_lru_list.reset_node_mru(n)
            else:
                self.mamba_lru_list.insert_mru(n)
                self.mamba_evictable_size_ += len(n.mamba_value)

        self._update_leaf_status(ancestor_node)

        self.inc_lock_ref(last_hit_node)
        self.ongoing_load_back[last_hit_node.id] = last_hit_node

        return full_device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        req = params.req
        if last_node.evicted or (last_node.mamba_evicted and last_node.mamba_backuped):
            loading_values = self.load_back(last_node, mem_quota, req=req)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node is not self.root_node and (
                last_node.evicted or last_node.mamba_evicted
            ):
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped and node.hit_count >= self.write_through_threshold:
            # write to host if the node is not backuped
            self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        backuped_node = self.ongoing_write_through.pop(ack_id)
                        if self.enable_storage:
                            self.write_backup_storage(backuped_node)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1

        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())

        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                self.dec_lock_ref(backuped_node)
                if self.enable_storage:
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        del self.cache_controller.ack_load_queue[:finish_count]

    def ready_to_load_host_cache(self) -> int:
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()

        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def _protect_host_node(self, node: TreeNode):
        node.protect_host()
        self.evictable_full_host_leaves.discard(node)

    def _release_host_node(self, node: TreeNode):
        node.release_host()
        if node.host_ref_counter == 0:
            self._update_full_host_leaf_status(node)

    def _discard_from_leaf_sets(self, node: TreeNode):
        self.evictable_full_device_leaves.discard(node)
        self.evictable_full_host_leaves.discard(node)

    def _update_leaf_status(self, node: TreeNode):
        self._update_full_device_leaf_status(node)
        self._update_full_host_leaf_status(node)

    def _update_full_device_leaf_status(self, node: TreeNode):
        if node == self.root_node or node.evicted or node.full_lock_ref > 0:
            self.evictable_full_device_leaves.discard(node)
            return
        for child in node.children.values():
            if not child.evicted:
                self.evictable_full_device_leaves.discard(node)
                return
        self.evictable_full_device_leaves.add(node)

    def _update_full_host_leaf_status(self, node: TreeNode):
        if (
            not node.evicted
            or not node.backuped
            or node == self.root_node
            or node.host_ref_counter > 0
        ):
            self.evictable_full_host_leaves.discard(node)
            return
        for child in node.children.values():
            if child.evicted and child.backuped:
                self.evictable_full_host_leaves.discard(node)
                return
        self.evictable_full_host_leaves.add(node)

    def _free_device_mamba(self, node: TreeNode) -> int:
        if node.mamba_value is None:
            return 0
        mamba_num = len(node.mamba_value)
        self.req_to_token_pool.mamba_pool.free(node.mamba_value)
        if node.mamba_lock_ref > 0:
            self.mamba_protected_size_ -= mamba_num
            node.mamba_lock_ref = 0
        else:
            self.mamba_evictable_size_ -= mamba_num
        if self.mamba_lru_list.in_list(node):
            self.mamba_lru_list.remove_node(node)
        node.mamba_value = None
        return mamba_num

    def _evict_to_host(self, node: TreeNode) -> Tuple[int, int]:
        # GPU -> CPU demotion: node stays in tree as evicted+backuped
        assert not node.evicted, f"already evicted, {node.id=}"
        assert node.backuped, f"not backuped, {node.id=}"

        num_full = len(node.value)

        self.cache_controller.evict_device(node.value)
        self.full_evictable_size_ -= num_full
        if self.full_lru_list.in_list(node):
            self.full_lru_list.remove_node(node)

        mamba_num = self._free_device_mamba(node)

        node.value = None
        self._update_leaf_status(node)
        self._update_full_device_leaf_status(node.parent)
        return num_full, mamba_num

    def _evict_regular(self, node: TreeNode) -> Tuple[int, int]:
        # evict a non-backuped device leaf — free GPU KV + mamba, delete from tree
        assert not node.evicted, f"already evicted, {node.id=}"
        assert not node.backuped, f"backuped node, {node.id=}"
        assert len(node.children) == 0, f"non-leaf, {node.id=}"

        full_num_evicted = len(node.value)

        self.cache_controller.evict_device(node.value)
        self.full_evictable_size_ -= full_num_evicted
        if self.full_lru_list.in_list(node):
            self.full_lru_list.remove_node(node)

        mamba_num_evicted = self._free_device_mamba(node)

        if node.mamba_host_value is not None:
            if self.mamba_host_lru_list.in_list(node):
                self.mamba_host_lru_list.remove_node(node)
            self.mamba_pool_host.free(node.mamba_host_value)
            node.mamba_host_value = None

        node.value = None
        self._discard_from_leaf_sets(node)

        parent = node.parent
        key = self.get_child_key_fn(node.key)
        v = parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self._update_leaf_status(parent)
        _, cascade_full_num_evicted, cascade_mamba_num_evicted = (
            self._iteratively_delete_tombstone_leaf(node)
        )
        return (
            full_num_evicted + cascade_full_num_evicted,
            mamba_num_evicted + cascade_mamba_num_evicted,
        )

    def _evict_host_leaf(self, node: TreeNode) -> int:
        # evict a host-resident leaf: free host KV + mamba, delete from tree, cascade
        assert node.evicted, f"not evicted, {node.id=}"
        assert node.backuped, f"not backuped, {node.id=}"
        assert node.mamba_value is None, f"has device mamba, {node.id=}"
        assert (
            node.host_ref_counter == 0
        ), f"in use, {node.id=} {node.host_ref_counter=}"

        full_num_evicted = self.cache_controller.evict_host(node.host_value)
        node.host_value = None

        if node.mamba_host_value is not None:
            if self.mamba_host_lru_list.in_list(node):
                self.mamba_host_lru_list.remove_node(node)
            self.mamba_pool_host.free(node.mamba_host_value)
            node.mamba_host_value = None

        self._discard_from_leaf_sets(node)
        parent = node.parent
        key = self.get_child_key_fn(node.key)
        v = parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self._update_leaf_status(parent)
        _, cascade_full_num_evicted, _ = self._iteratively_delete_tombstone_leaf(node)

        return full_num_evicted + cascade_full_num_evicted

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert node.mamba_value is None, f"has mamba value, {node.id=}"
        assert node.mamba_host_value is None, f"has mamba host value, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        parent = node.parent
        key = self.get_child_key_fn(node.key)
        v = parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self._discard_from_leaf_sets(node)

        if node.backuped and node.host_ref_counter == 0:
            self.cache_controller.evict_host(node.host_value)
            node.host_value = None

        self._update_leaf_status(parent)

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int, int]:
        full_num_evicted = 0
        mamba_num_evicted = 0

        while len(node.parent.children) == 0:
            if node.parent == self.root_node:
                break
            if node.parent.mamba_value is not None:
                break
            if node.parent.mamba_host_value is not None:
                break
            if node.parent.full_lock_ref > 0 or node.parent.mamba_lock_ref > 0:
                break

            parent = node.parent

            if not parent.evicted:
                full_num_evicted += len(parent.value)
                self.full_evictable_size_ -= len(parent.value)
                self.cache_controller.evict_device(parent.value)
                if self.full_lru_list.in_list(parent):
                    self.full_lru_list.remove_node(parent)

            self._discard_from_leaf_sets(parent)
            self._delete_tombstone_leaf(parent)
            node = parent

        return node, full_num_evicted, mamba_num_evicted

    def _evict_device_leaf(self, x: TreeNode) -> Tuple[int, int]:
        """Evict a device leaf node, choosing the right strategy:

        - backuped: demote to host via _evict_to_host (node stays in tree)
        - not backuped + write_back: write_backup first, then demote
        - not backuped + write_through: _evict_regular (delete from tree)
        """
        if not x.backuped:
            if self.cache_controller.write_policy == "write_back":
                self.write_backup(x, write_back=True)
                self.writing_check(write_back=True)
                return self._evict_to_host(x)
            else:
                return self._evict_regular(x)
        return self._evict_to_host(x)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_tokens = params.num_tokens
        full_num_evicted = 0
        mamba_num_evicted = 0

        if full_num_tokens > 0:
            leaves = list(self.evictable_full_device_leaves)
            eviction_heap = [(n.last_access_time, n) for n in leaves]
            heapq.heapify(eviction_heap)

            while full_num_evicted < full_num_tokens and eviction_heap:
                _, x = heapq.heappop(eviction_heap)
                if x not in self.evictable_full_device_leaves:
                    continue

                evicted_full, evicted_mamba = self._evict_device_leaf(x)
                full_num_evicted += evicted_full
                mamba_num_evicted += evicted_mamba

                parent = x.parent
                if parent in self.evictable_full_device_leaves:
                    heapq.heappush(eviction_heap, (parent.last_access_time, parent))

        if params.mamba_num > 0:
            mamba_num_evicted += self.evict_mamba(params.mamba_num)

        return EvictResult(
            num_tokens_evicted=full_num_evicted,
            mamba_num_evicted=mamba_num_evicted,
        )

    def evict_host(self, num_tokens: int):
        """Evict host-resident leaf nodes: free host KV + mamba, delete from tree, cascade."""
        heap = [(n.last_access_time, n) for n in self.evictable_full_host_leaves]
        heapq.heapify(heap)

        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x not in self.evictable_full_host_leaves:
                continue

            num_evicted += self._evict_host_leaf(x)

            if x.parent in self.evictable_full_host_leaves:
                heapq.heappush(heap, (x.parent.last_access_time, x.parent))

    def evict_mamba_host(self, num_mamba_hosts: int) -> int:
        """Evict host mamba states.

        Internal host node: free host mamba only (tombstone).
        Host leaf node: same as Full host evict — _evict_host_leaf_node frees
                        host KV + mamba, deletes from tree, cascades.
        """
        if self.disable or num_mamba_hosts <= 0:
            return 0

        x = self.mamba_host_lru_list.get_lru_no_lock()
        num_evicted = 0
        while num_evicted < num_mamba_hosts and self.mamba_host_lru_list.in_list(x):
            x_next = self.mamba_host_lru_list.get_prev_no_lock(x)
            if x.host_ref_counter > 0:
                x = x_next
                continue

            if x in self.evictable_full_host_leaves:
                self._evict_host_leaf(x)
                num_evicted += 1
            else:
                # internal host node: free host mamba only (tombstone)
                self.mamba_host_lru_list.remove_node(x)
                self.mamba_pool_host.free(x.mamba_host_value)
                x.mamba_host_value = None
                num_evicted += 1

            x = x_next
        return num_evicted

    def evict_mamba(self, mamba_num: int) -> int:
        """Evict mamba states.

        Internal node: tombstone — free GPU mamba only, KV stays on GPU.
        Leaf node: same as Full evict — _evict_to_host moves KV+mamba to host,
                   node stays in tree, then cascade tombstone parent device leaves.
        """
        if self.disable or mamba_num <= 0:
            return 0

        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        while mamba_num_evicted < mamba_num and self.mamba_lru_list.in_list(x):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use, {x.id=}"
            assert (
                not x.evicted
            ), f"evicted node should not be in mamba_lru_list, {x.id=}"

            if len(x.children) > 0:
                # Internal: free device mamba only, KV stays on device (tombstone)
                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                mamba_num_evicted += len(x.mamba_value)
                self.req_to_token_pool.mamba_pool.free(x.mamba_value)
                self.mamba_lru_list.remove_node(x)
                self._tombstone_internal_node(x)
            else:
                # Leaf: evict KV + mamba atomically
                assert (
                    x.full_lock_ref == 0
                ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=}"

                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                _, mamba_evicted = self._evict_device_leaf(x)
                mamba_num_evicted += mamba_evicted

                if not self.mamba_lru_list.in_list(x_next):
                    x_next = self.mamba_lru_list.get_lru_no_lock()

            x = x_next

        return mamba_num_evicted

    def _unevict_node(self, node: TreeNode, fresh_value: torch.Tensor):
        assert node.evicted, f"not evicted, {node.id=}"
        assert node.mamba_value is None, f"evicted node has device mamba, {node.id=}"
        n = len(fresh_value)

        node.value = fresh_value.clone()
        self.full_lru_list.insert_mru(node)
        self.full_evictable_size_ += n

        self._update_leaf_status(node)
        if node.parent is not None:
            self._update_leaf_status(node.parent)

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        mamba_value,
        chunked: bool = False,
        prev_prefix_len: int = 0,
    ) -> Tuple[int, bool]:
        assert mamba_value is not None, "Mamba value should not be None here."
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            if not node.evicted:
                self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0, True

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()

            if not node.evicted:
                self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)

            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if node.evicted:
                self._unevict_node(node, value[:prefix_len])
            else:
                if prev_prefix_len < total_prefix_length + prefix_len:
                    start = max(0, prev_prefix_len - total_prefix_length)
                    self.token_to_kv_pool_allocator.free(value[start:prefix_len])
                total_prefix_length += prefix_len
                self._inc_hit_count(node, chunked)

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        mamba_value_exist = False
        if len(key):
            new_node = self._add_new_node(node, key, value, mamba_value)
            self._inc_hit_count(new_node, chunked)
        elif node.mamba_value is None:
            node.mamba_value = mamba_value
            if not node.evicted:
                self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.insert_mru(node)
            self.mamba_evictable_size_ += len(mamba_value)
            node.last_access_time = get_last_access_time()
        else:
            mamba_value_exist = True
            if not node.evicted:
                self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()

        return total_prefix_length, mamba_value_exist

    def _add_new_node(
        self,
        parent: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        mamba_value: torch.Tensor,
    ) -> TreeNode:
        child_key = self.get_child_key_fn(key)
        new_node = TreeNode()
        new_node.parent = parent
        new_node.key = key
        new_node.value = value.clone()
        new_node.mamba_value = mamba_value
        self.full_lru_list.insert_mru(new_node)
        self.mamba_lru_list.insert_mru(new_node)
        parent.children[child_key] = new_node
        self.full_evictable_size_ += len(value)
        self.mamba_evictable_size_ += len(mamba_value)
        if self.enable_storage:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
        self._update_full_device_leaf_status(new_node)
        self._update_full_device_leaf_status(parent)
        return new_node

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key

        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, best_last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, best_last_node, best_value_len)

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """Walk tree to find best_last_node (mamba boundary)."""
        node = self.root_node
        child_key = self.get_child_key_fn(key)

        value: List[torch.Tensor] = []
        best_value_len = 0
        best_last_node = node

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]

            if child.evicted and not child.backuped:
                break

            if node.mamba_value is not None or node.mamba_backuped:
                best_value_len = len(value)
                best_last_node = node

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]
                if len(key):
                    child_key = self.get_child_key_fn(key)

        if node.mamba_value is not None or node.mamba_backuped:
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        best_last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        cow_mamba = params.cow_mamba
        req = params.req

        # Full LRU: skip evicted nodes for full_lru_list
        lru_node = best_last_node
        while lru_node != self.root_node and lru_node.evicted:
            lru_node = lru_node.parent
        self.full_lru_list.reset_node_and_parents_mru(lru_node, self.root_node)
        self.mamba_lru_list.reset_node_and_parents_mru(best_last_node, self.root_node)

        cur_time = get_last_access_time()
        node_update = best_last_node
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        if len(value) > best_value_len:
            from sglang.srt.server_args import get_global_server_args

            mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
            mamba_cache_chunk_aligned_seqlen = (
                sum(len(v) for v in value) // mamba_cache_chunk_size
            ) * mamba_cache_chunk_size
            mamba_branching_seqlen = (
                mamba_cache_chunk_aligned_seqlen
                if mamba_cache_chunk_aligned_seqlen > 0
                else None
            )
        else:
            mamba_branching_seqlen = None

        kv_host_hit_length = 0
        last_device_node = best_last_node
        while last_device_node is not self.root_node and last_device_node.evicted:
            kv_host_hit_length += len(last_device_node.host_value)
            last_device_node = last_device_node.parent

        last_host_node = best_last_node
        while last_host_node is not self.root_node and not last_host_node.backuped:
            last_host_node = last_host_node.parent

        mamba_host_hit = (
            1 if (last_host_node.mamba_evicted and last_host_node.mamba_backuped) else 0
        )
        host_hit_length = max(kv_host_hit_length, mamba_host_hit)

        mamba_node = best_last_node
        if cow_mamba and mamba_node.mamba_value is not None:
            if req.mamba_pool_idx is None:
                dst_index = self._alloc_with_evict(
                    self.req_to_token_pool.mamba_pool,
                    1,
                    self.evict_mamba,
                    lock_node=mamba_node,
                    error_message="Can not alloc mamba cache",
                )
                src_index = mamba_node.mamba_value
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)
                req.mamba_pool_idx = dst_index[0]
            else:
                src_index = mamba_node.mamba_value
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_device_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        if child.evicted:
            return self._split_evicted_node(key, child, split_len)

        self.evictable_full_device_leaves.discard(child)

        new_node = super()._split_node(key, child, split_len)

        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        self._update_leaf_status(new_node)
        self._update_leaf_status(child)

        return new_node

    def _split_evicted_node(
        self, key: RadixKey, child: TreeNode, split_len: int
    ) -> TreeNode:
        self.evictable_full_host_leaves.discard(child)

        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.value = None
        new_node.mamba_value = None
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = 0
        new_node.key = child.key[:split_len]

        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        child.last_access_time = get_last_access_time()
        if child.mamba_value is not None:
            self.mamba_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        if child.mamba_value is not None:
            self.mamba_lru_list.insert_mru(child)

        self._update_full_host_leaf_status(new_node)
        self._update_full_host_leaf_status(child)

        return new_node

    def _collect_all_nodes(self) -> list:
        ret = []
        stack = [self.root_node]
        while stack:
            cur = stack.pop()
            if not cur.evicted:
                ret.append(cur)
            stack.extend(cur.children.values())
        return ret

    def _collect_mamba_nontombstone_nodes(self) -> list:
        ret = []
        stack = [self.root_node]
        while stack:
            cur = stack.pop()
            if cur.mamba_value is not None:
                ret.append(cur)
            stack.extend(cur.children.values())
        return ret

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: TreeNode):
            for child in node.children.values():
                if not child.evicted:
                    values.append(child.value)
                _dfs(child)

        _dfs(self.root_node)
        return torch.cat(values) if values else torch.tensor([])

    def sanity_check(self):
        """Skip if async operations are pending (those nodes are still locked)."""
        self.loading_check()
        if self.ongoing_load_back or self.ongoing_write_through:
            return
        super().sanity_check()

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1

        while node != self.root_node:
            if node.evicted:
                node = node.parent
                continue

            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
                delta -= len(node.value)
                self.evictable_full_device_leaves.discard(node)
            node.full_lock_ref += 1
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0

        if node.mamba_value is not None and node.mamba_lock_ref > 0:
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
            node.mamba_lock_ref -= 1

        while node != self.root_node:
            if node.evicted:
                node = node.parent
                continue

            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
                delta += len(node.value)
            node.full_lock_ref -= 1
            if node.full_lock_ref == 0:
                self._update_full_device_leaf_status(node)
            node = node.parent
        return DecLockRefResult(delta=delta)

    # ---- L3 Support ----

    def shutdown(self):
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        storage_metrics_collector = None
        if enable_storage_metrics:
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            storage_metrics_collector = StorageMetricsCollector(labels=labels)

        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics
        if self.enable_storage_metrics:
            self.storage_metrics_collector = storage_metrics_collector
        else:
            self.storage_metrics_collector = None

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type
            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )
            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json "
                f"'{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                host_pools=self.host_pool_group.entries,
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple:
        try:
            self._drain_storage_control_queues_local()
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            return False, f"Failed to detach HiCache storage backend: {e}"

        self._drain_storage_control_queues_local()
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        if hasattr(self, "storage_metrics_collector"):
            self.storage_metrics_collector = None
        return True, "Detached HiCache storage backend successfully."

    def prefetch_abort(self, pool_transfers: Optional[list[PoolTransfer]]) -> None:
        """Free any allocated mamba host slots on prefetch abort/revoke."""
        for transfer in pool_transfers or []:
            if transfer.name == PoolName.MAMBA:
                if transfer.host_indices is not None:
                    self.mamba_pool_host.free(transfer.host_indices)
                break

    def _force_release_pending_storage_ops(self):
        cc = self.cache_controller

        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    self.ongoing_prefetch.pop(req_id, None)
                    continue
                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )
                try:
                    self.prefetch_abort(getattr(_operation, "pool_transfers", None))
                except Exception:
                    logger.exception(
                        "Failed to release mamba host indices for prefetch %s", req_id
                    )
                try:
                    self._release_host_node(last_host_node)
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )
                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass
                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    self._release_host_node(node)
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, operation = info
                    self.prefetch_abort(operation.pool_transfers)
                    self._release_host_node(last_host_node)
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    self._release_host_node(entry)
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend "
                        f"{type(self.cache_controller.storage_backend).__name__} "
                        "does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def drain_storage_control_queues(self):
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            return True

        operation_terminated = operation.is_terminated()
        if self.tp_world_size > 1:
            states = torch.tensor(
                [1 - int(can_terminate), int(operation_terminated)],
                dtype=torch.int,
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )
            can_terminate = states[0].item() == 0
            operation_terminated = states[1].item() == 1
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def write_backup_storage(self, node: TreeNode):
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )
        extra_pools = self.mamba_archive_transfers(node)
        operation_id = self.cache_controller.write_storage(
            node.host_value,
            node.key,
            node.hash_value,
            prefix_keys,
            extra_pools=extra_pools,
        )
        self.ongoing_backup[operation_id] = node
        self._protect_host_node(node)

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        self._protect_host_node(last_host_node)

        # Allocate host KV memory
        host_indices = self._alloc_with_evict(
            self.cache_controller.mem_pool_host,
            prefetch_length,
            self.evict_host,
        )
        if host_indices is None:
            self._release_host_node(last_host_node)
            return

        # Allocate host mamba slot
        extra_pools = self.mamba_prefetch_alloc(new_input_tokens, last_hash)
        if extra_pools is None:
            self.cache_controller.mem_pool_host.free(host_indices)
            self._release_host_node(last_host_node)
            return

        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys,
            extra_pools=extra_pools,
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = completed_tokens_tensor.item()

        mamba_host_indices = None
        mamba_loaded = False
        for transfer in operation.pool_transfers or []:
            if transfer.name == PoolName.MAMBA:
                mamba_host_indices = transfer.host_indices
                mamba_loaded = (
                    operation.pool_storage_result.extra_pool_hit_pages.get(
                        PoolName.MAMBA, 0
                    )
                    >= 1
                )
                break

        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids,
                extra_key=last_host_node.key.extra_key,
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
            mamba_host_indices,
            mamba_loaded,
        )

        # Free host KV memory: matched portion is already in tree, tail was unused
        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )

        # Free mamba host slot if it wasn't inserted into the tree
        if mamba_host_indices is not None:
            inserted_new = matched_length < min_completed_tokens
            if not inserted_new or not mamba_loaded:
                self.mamba_pool_host.free(mamba_host_indices)

        self._release_host_node(last_host_node)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        if loaded_from_storage > 0 and operation.pool_transfers:
            logger.debug(
                "HiCache mamba prefetch completed for request %s: prefetched_tokens=%s mamba_states=%s",
                req_id,
                loaded_from_storage,
                int(mamba_loaded),
            )

        return True

    def _insert_helper_host(
        self,
        node: TreeNode,
        key: RadixKey,
        host_value,
        hash_value,
        mamba_host_value: Optional[torch.Tensor] = None,
        mamba_loaded: bool = False,
    ):
        node.last_access_time = get_last_access_time()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            if node != self.root_node and node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
            prefix_len = self.key_match_fn(node.key, key)

            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        leaf_node: Optional[TreeNode] = None
        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.mamba_value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            leaf_node = new_node
            self._update_full_host_leaf_status(new_node)
            self._update_full_host_leaf_status(node)

        # Attach mamba state to the new leaf
        if leaf_node is not None and mamba_host_value is not None and mamba_loaded:
            leaf_node.mamba_host_value = mamba_host_value.clone()
            if not self.mamba_host_lru_list.in_list(leaf_node):
                self.mamba_host_lru_list.insert_mru(leaf_node)
        return matched_length

    def release_aborted_request(self, rid: str):
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)
        self._release_host_node(last_host_node)
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.prefetch_abort(operation.pool_transfers)
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

    def _flush_pending_storage_backups_before_reset(self) -> None:
        if not self.enable_storage:
            return

        self.writing_check(write_back=True)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            self.drain_storage_control_queues()
            backup_qsize = self.cache_controller.backup_queue.qsize()
            ack_backup_qsize = self.cache_controller.ack_backup_queue.qsize()
            ongoing_backup = len(self.ongoing_backup)
            ongoing_write = len(self.ongoing_write_through)
            if (
                backup_qsize == 0
                and ack_backup_qsize == 0
                and ongoing_backup == 0
                and ongoing_write == 0
            ):
                return
            time.sleep(0.05)

        logger.warning(
            "Timed out waiting for HiCache storage backups to drain before reset: "
            "ongoing_write=%s ongoing_backup=%s backup_queue=%s ack_backup_queue=%s",
            len(self.ongoing_write_through),
            len(self.ongoing_backup),
            self.cache_controller.backup_queue.qsize(),
            self.cache_controller.ack_backup_queue.qsize(),
        )

    def _alloc_with_evict(
        self,
        pool,
        size: int,
        evict_fn,
        lock_node: Optional[TreeNode] = None,
        error_message: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        indices = pool.alloc(size)
        if indices is None:
            if lock_node is not None:
                self.inc_lock_ref(lock_node)
            evict_fn(size)
            indices = pool.alloc(size)
            if lock_node is not None:
                self.dec_lock_ref(lock_node)
        if indices is None and error_message is not None:
            raise RuntimeError(error_message)
        return indices

    # -- mamba PoolTransfer builders (D↔H↔S) ----------------------------------

    def mamba_backup_transfers(self, node: TreeNode) -> Optional[list[PoolTransfer]]:
        # build D→H transfer descriptor for mamba state
        if node.mamba_value is None:
            return None
        return [
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=node.mamba_host_value,
                device_indices=node.mamba_value,
            )
        ]

    def mamba_backup_commit(
        self, node: TreeNode, transfers: list[PoolTransfer]
    ) -> None:
        # store auto-allocated mamba host indices into the node after D→H backup
        if not transfers:
            return
        host_indices = transfers[0].host_indices
        if node.mamba_host_value is None and host_indices is not None:
            node.mamba_host_value = host_indices
            self.mamba_host_lru_list.insert_mru(node)

    def mamba_archive_transfers(self, node: TreeNode) -> Optional[list[PoolTransfer]]:
        # build H→Storage transfer descriptor for mamba state
        if node.mamba_host_value is None or not node.hash_value:
            return None
        return [
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=node.mamba_host_value,
                keys=[node.hash_value[-1]],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def mamba_prefetch_alloc(
        self,
        token_ids: List[int],
        last_hash: Optional[str],
    ) -> Optional[list[PoolTransfer]]:
        # allocate a mamba host slot and build Storage→H transfer descriptor
        if not token_ids:
            return None
        host_indices = self._alloc_with_evict(
            self.mamba_pool_host, 1, self.evict_mamba_host
        )
        if host_indices is None:
            return None
        # placeholder key; I/O thread replaces with correct hash after hit query
        return [
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=host_indices,
                keys=["__placeholder__"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def mamba_restore_transfers(
        self,
        last_hit_node: TreeNode,
        nodes_to_restore: list[TreeNode],
        req,
    ) -> Optional[list[PoolTransfer]]:
        # build H→D transfer descriptors for mamba state
        backed_up_host_indices: list[torch.Tensor] = []
        for node in nodes_to_restore:
            if not node.mamba_backuped:
                continue
            backed_up_host_indices.append(node.mamba_host_value)

        transfers: list[PoolTransfer] = []
        if backed_up_host_indices:
            transfers.append(
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=torch.cat(backed_up_host_indices),
                    device_indices=None,
                )
            )

        if (
            req is not None
            and last_hit_node in nodes_to_restore
            and last_hit_node.mamba_host_value is not None
        ):
            if req.mamba_pool_idx is None:
                req.mamba_pool_idx = self._alloc_with_evict(
                    self.req_to_token_pool.mamba_pool,
                    len(last_hit_node.mamba_host_value),
                    self.evict_mamba,
                    lock_node=last_hit_node,
                    error_message="Cannot alloc request mamba cache for host load back",
                )[0]
            transfers.append(
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=last_hit_node.mamba_host_value,
                    device_indices=req.mamba_pool_idx.unsqueeze(0),
                )
            )

        return transfers if transfers else None

    def mamba_restore_commit(
        self,
        restored_nodes: list[TreeNode],
        transfers: Optional[list[PoolTransfer]],
    ) -> None:
        # write back controller-allocated device indices after H→D restore
        if not restored_nodes or not transfers or transfers[0].device_indices is None:
            return
        device_indices = transfers[0].device_indices
        offset = 0
        for node in restored_nodes:
            count = len(node.mamba_host_value)
            node.mamba_value = device_indices[offset : offset + count].clone()
            offset += count
