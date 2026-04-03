from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.kv_connector import BaseKVConnector, LoadOperation
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


class ExtendedRadixCache(BasePrefixCache):
    """RadixCache decorator with external KV storage connector."""

    def __init__(
        self,
        params: CacheInitParams,
        connector: Optional[BaseKVConnector] = None,
    ):
        self._inner_radixtree = RadixCache(params)
        self._connector = connector

        self._load_task_id_counter = 0
        self._load_queue: List[LoadOperation] = []
        self._ongoing_load_tasks: Dict[int, List[TreeNode]] = {}
        self._ongoing_store_tasks: Dict[int, TreeNode] = {}

    # -- Forward PrefixCacheTrait properties to inner cache --

    @property
    def req_to_token_pool(self):
        return self._inner_radixtree.req_to_token_pool

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):
        self._inner_radixtree.req_to_token_pool = value

    @property
    def token_to_kv_pool_allocator(self):
        return self._inner_radixtree.token_to_kv_pool_allocator

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):
        self._inner_radixtree.token_to_kv_pool_allocator = value

    @property
    def page_size(self):
        return self._inner_radixtree.page_size

    @page_size.setter
    def page_size(self, value):
        self._inner_radixtree.page_size = value

    @property
    def disable(self):
        return self._inner_radixtree.disable

    @property
    def device(self):
        return self._inner_radixtree.device

    @property
    def metrics_collector(self):
        return self._inner_radixtree.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value):
        self._inner_radixtree.metrics_collector = value

    @property
    def layer_done_counter(self):
        if self._connector is None:
            return None
        return self._connector.layer_done_counter

    # -- Core methods with connector logic --

    def reset(self):
        self._ongoing_store_tasks.clear()
        self._ongoing_load_tasks.clear()
        self._load_queue.clear()

        if self._connector is not None:
            self._connector.reset()

        self._inner_radixtree.reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        device_match_result = self._inner_radixtree.match_prefix(params)
        if self._connector is None:
            return device_match_result

        key = params.key
        device_indices: torch.Tensor = device_match_result.device_indices
        last_device_node = device_match_result.last_device_node

        uncached_len = len(key) - device_indices.numel()
        if uncached_len <= 0:
            return device_match_result

        token_mask = torch.zeros(len(key), dtype=torch.bool)
        token_mask[device_indices.numel() :] = True

        new_hit_length = self._connector.get_new_hit_length(
            token_ids=key.token_ids,
            token_mask=token_mask,
            update_state_for_load=params.update_connector_state,
            rid=params.req.rid if params.req is not None else None,
        )

        return MatchResult(
            device_indices=device_indices,
            last_device_node=last_device_node,
            last_host_node=last_device_node,
            host_hit_length=new_hit_length,
        )

    def init_load_back(
        self,
        req: Req,
        mem_quota: Optional[int] = None,
    ) -> None:
        if self._connector is None:
            return

        host_hit_length = req.host_hit_length

        if host_hit_length <= 0 or (
            mem_quota is not None and host_hit_length > mem_quota
        ):
            self._connector.release_load_state(req.rid)
            return

        device_indices = self._inner_radixtree.token_to_kv_pool_allocator.alloc(
            host_hit_length
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=host_hit_length))
            device_indices = self._inner_radixtree.token_to_kv_pool_allocator.alloc(
                host_hit_length
            )
        if device_indices is None:
            logger.warning(
                "Failed to allocate %d GPU slots for external load",
                host_hit_length,
            )
            self._connector.release_load_state(req.rid)
            return

        gpu_cached_len = len(req.prefix_indices)
        key = RadixKey(
            token_ids=req.fill_ids[gpu_cached_len : gpu_cached_len + host_hit_length],
            extra_key=req.extra_key,
        )

        last_node = req.last_node
        new_node = TreeNode()
        new_node.key = key
        new_node.value = device_indices
        new_node.parent = last_node
        last_node.children[self._inner_radixtree.get_child_key_fn(new_node.key)] = (
            new_node
        )
        self._inner_radixtree.evictable_size_ += len(device_indices)
        self._inner_radixtree._record_store_event(new_node)

        self._inner_radixtree.inc_lock_ref(new_node)

        self._load_queue.append(
            LoadOperation(
                rid=req.rid,
                device_indices=device_indices,
                node=new_node,
            )
        )

        req.prefix_indices = torch.cat([req.prefix_indices, device_indices])
        req.last_node = new_node

    def ready_to_load_host_cache(self) -> int:
        if self._connector is None or not self._load_queue:
            return -1

        task_id = self._load_task_id_counter
        self._load_task_id_counter += 1

        self._connector.start_load_kv(task_id, self._load_queue)

        nodes = [op.node for op in self._load_queue]
        self._ongoing_load_tasks[task_id] = nodes

        self._load_queue.clear()
        return task_id

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        self._inner_radixtree.cache_finished_req(req, is_insert=is_insert, **kwargs)

        if self._connector is None or not is_insert:
            return

        req_id = req.req_pool_idx
        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self._inner_radixtree.req_to_token_pool.req_to_token[
            req_id, : len(token_ids)
        ]

        task_id = self._load_task_id_counter
        self._load_task_id_counter += 1

        self._inner_radixtree.inc_lock_ref(req.last_node)
        
        self._connector.start_store_kv(
            task_id=task_id,
            token_ids=token_ids,
            kv_indices=kv_indices,
        )

        self._ongoing_store_tasks[task_id] = req.last_node

    def evict(self, params: EvictParams) -> EvictResult:
        return self._inner_radixtree.evict(params)

    def check_kv_events(self):
        if self._connector is None:
            return
        self._check_store_completion()
        self._check_load_completion()

    def prefetch(self, req: Req) -> None:
        if self._connector is None:
            return
        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        self._connector.prefetch(req.rid, token_ids)

    def can_be_scheduled(self, req: Req) -> bool:
        if self._connector is None:
            return True
        return self._connector.check_prefetch_completed(req.rid)

    def release_aborted_request(self, req: Req) -> None:
        if self._connector is None:
            return
        self._connector.cancel_prefetch(req.rid)

    # -- Private helpers --

    def _check_store_completion(self) -> None:
        completed_ids = self._connector.check_completed_store_tasks()
        for task_id in completed_ids:
            node = self._ongoing_store_tasks.pop(task_id, None)
            if node is not None:
                self._inner_radixtree.dec_lock_ref(node)

    def _check_load_completion(self) -> None:
        completed_ids = self._connector.check_completed_load_tasks()
        for task_id in completed_ids:
            nodes = self._ongoing_load_tasks.pop(task_id, None)
            if nodes is not None:
                for node in nodes:
                    self._inner_radixtree.dec_lock_ref(node)

    # -- Pass-through methods --

    def insert(self, key, value=None, **kwargs):
        return self._inner_radixtree.insert(key, value=value, **kwargs)

    def inc_lock_ref(self, node):
        return self._inner_radixtree.inc_lock_ref(node)

    def dec_lock_ref(self, node):
        return self._inner_radixtree.dec_lock_ref(node)

    def cache_unfinished_req(self, req: Req, **kwargs):
        return self._inner_radixtree.cache_unfinished_req(req, **kwargs)

    def evictable_size(self):
        return self._inner_radixtree.evictable_size()

    def protected_size(self):
        return self._inner_radixtree.protected_size()

    def total_size(self):
        return self._inner_radixtree.total_size()

    def pretty_print(self):
        return self._inner_radixtree.pretty_print()

    def all_values_flatten(self):
        return self._inner_radixtree.all_values_flatten()

    def take_events(self):
        return self._inner_radixtree.take_events()

    def __getattr__(self, name):
        return getattr(self._inner_radixtree, name)
