import heapq
import logging
import os
import threading
import time
from functools import partial
from typing import List, Optional

import torch
import yaml

from sglang.srt.managers.eic_cache_controller import (
    EICCacheController,
    get_content_hash,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.eic_memory_pool import (
    EICMHATokenToKVPoolHost,
    EICMLATokenToKVPoolHost,
    EICNSATokenToKVPoolHost,
    MemoryStateInt,
    get_eic_config_file_path,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EICHiRadixCacheBuilder:
    @staticmethod
    def build(
        params: CacheInitParams,
        server_args: ServerArgs,
    ):
        if server_args.disable_eic_shared:
            return EICHiRadixCache(
                params,
                server_args,
            )
        else:
            return EICPagedHiRadixCache(
                params,
                server_args,
            )


def mha_pool_get_flat_data(self: MHATokenToKVPool, indices: torch.Tensor):
    flatten = torch.stack(
        [
            torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
            torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
        ]
    )
    return flatten


def mha_pool_transfer(
    self: MHATokenToKVPool, indices: torch.Tensor, flat_data: torch.Tensor
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    k_data, v_data = flat_data[0], flat_data[1]
    for i in range(self.layer_num):
        self.k_buffer[i][indices] = k_data[i]
        self.v_buffer[i][indices] = v_data[i]


def mla_pool_get_flat_data(self: MLATokenToKVPool, indices: torch.Tensor):
    return torch.stack([self.kv_buffer[i][indices] for i in range(self.layer_num)])


def mla_pool_transfer(
    self: MLATokenToKVPool, indices: torch.Tensor, flat_data: torch.Tensor
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    for i in range(self.layer_num):
        self.kv_buffer[i][indices] = flat_data[i]


def nsa_pool_get_flat_data(self: NSATokenToKVPool, indices: torch.Tensor):
    num_pages = len(indices) // self.page_size

    # Gather MLA (num_tokens, 1, kv_cache_dim) -> (layer_num, num_tokens, kv_cache_dim)
    mla_data = torch.stack([self.kv_buffer[i][indices] for i in range(self.layer_num)])
    # (layer_num, num_pages, mla_page_bytes)
    mla_bytes = mla_data.view(self.layer_num, num_pages, -1).view(torch.uint8)

    # Gather NSA
    page_indices = indices.reshape(-1, self.page_size)[:, 0] // self.page_size
    # (layer_num, num_pages, 8448)
    nsa_packed = torch.stack(
        [self.index_k_with_scale_buffer[i][page_indices] for i in range(self.layer_num)]
    )

    combined = torch.cat(
        [mla_bytes, nsa_packed], dim=-1
    )  # (layer_num, num_pages, final_dim)
    return combined


def nsa_pool_transfer(
    self: NSATokenToKVPool,
    indices: torch.Tensor,
    flat_data: torch.Tensor,
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    # flat_data: (layer_num, num_pages, final_dim)
    num_pages = len(indices) // self.page_size
    mla_page_bytes = self.kv_cache_dim * self.store_dtype.itemsize * self.page_size
    page_indices = indices.reshape(-1, self.page_size)[:, 0] // self.page_size

    for i in range(self.layer_num):
        layer_data = flat_data[i]  # (num_pages, final_dim)

        # Split
        mla_bytes = layer_data[:, :mla_page_bytes]
        nsa_bytes = layer_data[:, mla_page_bytes:]

        # Write MLA
        mla_part = (
            mla_bytes.reshape(-1)
            .view(self.store_dtype)
            .reshape(num_pages * self.page_size, 1, self.kv_cache_dim)
        )
        self.kv_buffer[i][indices] = mla_part

        # Write NSA
        self.index_k_with_scale_buffer[i][page_indices] = nsa_bytes


class EICHiRadixCache(RadixCache):

    def __init__(
        self,
        params: CacheInitParams,
        server_args: ServerArgs,
    ):
        self.tp_group = params.tp_cache_group
        self.tp_size = self.tp_group.size()
        self.rank = self.tp_group.rank()
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = EICMHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                "cpu",
                params.page_size,
                self.rank,
                extra_info=self.get_extra_info(params, server_args),
            )
            self.kv_cache.get_flat_data = partial(mha_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(mha_pool_transfer, self.kv_cache)
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            self.token_to_kv_pool_host = EICNSATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                "cpu",
                params.page_size,
                self.rank,
                extra_info=self.get_extra_info(params, server_args),
            )
            self.kv_cache.get_flat_data = partial(nsa_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(nsa_pool_transfer, self.kv_cache)
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = EICMLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                "cpu",
                params.page_size,
                self.rank,
                extra_info=self.get_extra_info(params, server_args),
            )
            self.kv_cache.get_flat_data = partial(mla_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(mla_pool_transfer, self.kv_cache)
        else:
            raise ValueError(f"HiRadixCache only supports MHA, MLA and NSA yet")

        self.load_cache_event = threading.Event()
        self.cache_controller = EICCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            params.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            server_args=server_args,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # EIC is native to this cache (no pluggable L3 HiCacheStorage backend);
        # is_fully_idle() uses this flag to skip ongoing_prefetch/backup checks.
        self.enable_storage = False
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 3
        )
        self.load_back_threshold = 10
        super().__init__(params)

        self.save_decode_cache = True
        config_file = get_eic_config_file_path()
        if os.path.exists(config_file):
            with open(config_file, "r") as fin:
                config = yaml.safe_load(fin)
            self.init_hyper_params(config)

    def get_extra_info(self, params: CacheInitParams, server_args: ServerArgs):
        # TODO update when sglang support pp
        extra_info = {
            "model_path": server_args.model_path,
            "world_size": self.tp_size,
            "tp_rank": self.rank,
            "framework": "sglang",
            "pp_rank": params.pp_rank,
            "pp_size": params.pp_size,
        }
        return extra_info

    def init_hyper_params(self, config: dict):
        self.save_decode_cache = config.get("save_decode_cache", True)
        logger.info(
            f"EICHiRadixCache save_decode_cache set to {self.save_decode_cache}"
        )
        self.load_back_threshold = config.get("load_back_threshold", 10)
        logger.info(
            f"EICHiRadixCache load_back_threshold set to {self.load_back_threshold}"
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.ongoing_load_back = {}
        self.ongoing_write_through = {}
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def write_backup(self, node: TreeNode, write_back=False):
        logger.debug(f"write backup for node {node.id}")
        if node.evicted:
            return 0
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
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def inc_hit_count(self, node: TreeNode, chunked: bool = False):
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1
        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                self.write_backup(node)
                node.hit_count = 0

    def get_tp_result(self, flag):
        if isinstance(flag, bool):
            flag = [flag]
        if self.tp_size <= 1:
            return flag
        # synchronize the result across TP workers
        temp = [0 if x else 1 for x in flag]
        temp_tensor = torch.tensor(temp, dtype=torch.int64, device="cpu")
        torch.distributed.all_reduce(
            temp_tensor, op=torch.distributed.ReduceOp.SUM, group=self.tp_group
        )
        result_list = temp_tensor.tolist()
        result = []
        for i in range(len(result_list)):
            result.append(result_list[i] == 0)
        return result

    def writing_check(self, write_back=False):
        if len(self.ongoing_write_through) == 0:
            return
        write_check_start_time = time.perf_counter()
        if write_back:
            while (
                len(self.ongoing_write_through)
                != self.cache_controller.ack_write_queue.qsize()
            ):
                time.sleep(0.01)
        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue.qsize(), dtype=torch.int
        )
        # may skip synchronize queue_size for write
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        ack_list = []
        flags = []
        for _ in range(queue_size.item()):
            ack_id, success = self.cache_controller.ack_write_queue.get_nowait()
            ack_list.append(ack_id)
            flags.append(success)
        for ack_id, success in zip(ack_list, flags):
            if (
                not success
                and not isinstance(self, EICPagedHiRadixCache)
                and self.ongoing_write_through[ack_id].host_value is not None
            ):
                if (
                    self.cache_controller.mem_pool_host.get_state(
                        self.ongoing_write_through[ack_id].host_value
                    )
                    != MemoryStateInt.IDLE
                ):
                    self.cache_controller.mem_pool_host.free(
                        self.ongoing_write_through[ack_id].host_value
                    )
                self.ongoing_write_through[ack_id].host_value = None
            if not write_back:
                self.dec_lock_ref(self.ongoing_write_through[ack_id])
            # clear the reference
            del self.ongoing_write_through[ack_id]
        cost_time = time.perf_counter() - write_check_start_time
        if cost_time > 1:
            logger.warning(
                f"writing check cost {cost_time:.3f} seconds, "
                f"queue size {queue_size.item()}"
            )

    def loading_check(self):
        if len(self.ongoing_load_back) == 0:
            return
        loading_check_start_time = time.perf_counter()
        queue_size = torch.tensor(
            self.cache_controller.ack_load_queue.qsize(), dtype=torch.int
        )
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        ack_list = []
        complete_tokens = []
        for _ in range(queue_size.item()):
            ack_id, complete_token = self.cache_controller.ack_load_queue.get_nowait()
            ack_list.append(ack_id)
            complete_tokens.append(complete_token)
        for ack_id, complete_token in zip(ack_list, complete_tokens):
            start_node, end_node, total_token_num = self.ongoing_load_back[ack_id]
            self.dec_lock_ref(end_node)
            failed_token_num = total_token_num - complete_token
            while end_node != start_node:
                if failed_token_num >= len(end_node.value):
                    # node load back full fail
                    # no need to delete failed node because the kvcache will be set after compute
                    self.cache_controller.mem_pool_device_allocator.free(end_node.value)
                    self.evictable_size_ -= len(end_node.value)
                    failed_token_num -= len(end_node.value)
                    end_node.value = None
                    end_node.host_value = None
                    self._update_leaf_status(end_node)
                    self._update_leaf_status(end_node.parent)
                elif failed_token_num > 0:
                    # node load back partial fail, split node
                    split_len = len(end_node.value) - failed_token_num
                    self._split_node(end_node.key, end_node, split_len)
                    self.evictable_size_ -= failed_token_num
                    self.cache_controller.mem_pool_device_allocator.free(end_node.value)
                    failed_token_num -= len(end_node.value)
                    end_node.value = None
                    end_node.host_value = None
                    self._update_leaf_status(end_node)
                    self._update_leaf_status(end_node.parent)
                    assert failed_token_num == 0, "failed_token_num should be zero"
                end_node = end_node.parent
            # clear the reference
            del self.ongoing_load_back[ack_id]
        cost_time = time.perf_counter() - loading_check_start_time
        if cost_time > 1:
            logger.warning(
                f"loading check cost {cost_time:.3f} seconds, "
                f"queue size {queue_size.item()}"
            )

    # TODO: is not correct for eic, but neednt to be fixed rightnow
    def evictable_size(self):
        return self.evictable_size_

    def evict(self, params: EvictParams, retry_times: int = 3) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        while len(self.ongoing_write_through) > 50 or len(self.ongoing_load_back) > 50:
            self.writing_check()
            self.loading_check()
            time.sleep(0.1)

        num_evicted = 0
        while retry_times > 0:
            retry_times -= 1
            leaves = list(self.evictable_leaves)
            eviction_heap = [
                (self.eviction_strategy.get_priority(node), node) for node in leaves
            ]
            heapq.heapify(eviction_heap)

            write_back_nodes = []
            idx = 0

            logger.debug(
                f"evict {num_tokens} tokens, current evictable size {self.evictable_size_}, protect_size {self.protected_size_}, leaves {len(leaves)}"
            )
            while num_evicted < num_tokens and len(eviction_heap):
                _priority, x = heapq.heappop(eviction_heap)
                logger.debug(
                    f"evicting {idx} node {x.id}, access {x.last_access_time}, value {x.value} {x.host_value}"
                )
                idx += 1

                if x.lock_ref > 0:
                    logger.debug(f"node {x.id} is locked, skip eviction")
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
                    new_priority = self.eviction_strategy.get_priority(x.parent)
                    heapq.heappush(eviction_heap, (new_priority, x.parent))

            if self.cache_controller.write_policy == "write_back":
                # blocking till all write back complete
                self.writing_check(write_back=True)
                for node in write_back_nodes:
                    if node.backuped:
                        self._evict_backuped(node)
                    else:
                        self._evict_regular(node)

            if num_evicted < num_tokens:
                logger.info(
                    f"only evicted {num_evicted} tokens, less than requested {num_tokens}"
                )
            return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_backuped(self, node: TreeNode):
        if node.host_value is None:
            logger.error(f"host value is None for node {node.id}")
            return self._evict_regular(node)
        num_evicted = self.cache_controller.evict_device(node.value, node.host_value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_leaf_status(node.parent)
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
        start_time = time.perf_counter()
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
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (
            ancester_node,
            last_hit_node,
            len(device_indices),
        )
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices

    def loading_complete(self, node: TreeNode):
        self.loading_check()
        return node.id not in self.ongoing_load_back.keys()

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self):
        producer_index = self.cache_controller.layer_done_counter.update_producer()
        self.load_cache_event.set()
        return producer_index

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()

    def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            while not last_node.backuped and last_node.parent is not None:
                last_node = last_node.parent
                last_host_node = last_node
                host_hit_length = 0
            if not last_node.evicted:
                break
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = child.key.match(key, page_size=self.page_size)
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
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
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
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        if priority is None:
            priority = 0
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    if not isinstance(self, EICPagedHiRadixCache):
                        self.token_to_kv_pool_host.free(node.host_value)
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_leaf_status(node.parent)
                    self.inc_hit_count(node, chunked)
                else:
                    self.inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    if not isinstance(self, EICPagedHiRadixCache):
                        self.token_to_kv_pool_host.free(new_node.host_value)
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_leaf_status(new_node.parent)
                    self.inc_hit_count(new_node, chunked)
                else:
                    self.inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            if self.cache_controller.write_policy != "write_back":
                self.inc_hit_count(new_node, chunked)
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


def _need_calculate_hash(node: TreeNode, page_size: int):
    if node is None or node.key is None or len(node.key) == 0:
        return False
    return node.content_hash is None or len(node.key) // page_size != len(
        node.content_hash
    )


class EICPagedHiRadixCache(EICHiRadixCache):
    def __init__(
        self,
        params: CacheInitParams,
        server_args: ServerArgs,
    ):
        self.calculate_hash_fn = get_content_hash
        self.load_remote_threshold = 100
        self.match_req_set = []
        self.eic_check_max_num = -1
        super().__init__(params, server_args)

    def init_hyper_params(self, config):
        super().init_hyper_params(config)
        self.load_remote_threshold = max(
            config.get("load_remote_threshold", 100), self.page_size
        )
        logger.info(
            f"EICPagedHiRadixCache load_remote_threshold set to {self.load_remote_threshold}"
        )
        self.eic_check_max_num = config.get("eic_check_max_num", -1)
        logger.info(
            f"EICPagedHiRadixCache eic_check_max_num set to {self.eic_check_max_num}"
        )
        self.load_back_check = config.get("load_back_check", False)

    def _calculate_content_hash(self, node: TreeNode):
        if _need_calculate_hash(node.parent, self.page_size):
            self._calculate_content_hash(node.parent)
        if node.parent is not None and node.parent.content_hash is not None:
            prev_node_hash = node.parent.content_hash[-1]
        else:
            prev_node_hash = None
        node.content_hash = self.calculate_hash_fn(
            node.key, self.page_size, prev_node_hash
        )

    def _split_node(self, key, child: TreeNode, split_len: int):
        assert (
            split_len % self.page_size == 0
        ), f"split_len {split_len} is not page aligned"
        # child node split into new_node -> child
        if _need_calculate_hash(child, self.page_size):
            self._calculate_content_hash(child)
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count
        split_hash_nums = split_len // self.page_size
        new_node.content_hash = child.content_hash[:split_hash_nums]
        child.content_hash = child.content_hash[split_hash_nums:]

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        return new_node

    def match_prefix_extend(self, key: RadixKey, last_node):
        cache_prefix_len = 0
        temp_node = last_node
        while temp_node:
            cache_prefix_len += len(temp_node.key)
            temp_node = temp_node.parent

        # if the cache prefix is too long, or the remaining key is too short, we can skip loading from eic
        if (len(key) - cache_prefix_len) < self.load_remote_threshold:
            return last_node

        logger.debug(
            f"few cache in radix, try load from eic, cache len {cache_prefix_len}, total len {len(key)}"
        )
        if _need_calculate_hash(last_node, self.page_size):
            self._calculate_content_hash(last_node)
        last_prev_hash = None
        if last_node.content_hash is not None and len(last_node.content_hash) > 0:
            last_prev_hash = last_node.content_hash[-1]
        need_compute_key = key[cache_prefix_len:]
        eic_hash, eic_key = self.cache_controller.find_longest_prefix_in_eic(
            need_compute_key, last_prev_hash
        )
        if self.tp_size > 1:
            eic_hash_len_tensor = torch.tensor(
                [len(eic_hash)], dtype=torch.int64, device="cpu"
            )
            torch.distributed.all_reduce(
                eic_hash_len_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            eic_hash_len = eic_hash_len_tensor.item()
            eic_hash = eic_hash[:eic_hash_len]
            eic_key = eic_key[: eic_hash_len * self.page_size]
        if len(eic_key) < self.load_remote_threshold:
            logger.debug(
                f"eic key is too short, skip loading from eic, eic cache len {len(eic_key)}, need compute key len {len(need_compute_key)}"
            )
            return last_node
        load_node = TreeNode()
        load_node.key = eic_key
        load_node.content_hash = eic_hash
        load_node.host_value = torch.arange(
            len(eic_key), dtype=torch.int32, device="cpu"
        )
        assert (
            last_node.children.get(eic_key.child_key(self.page_size)) is None
        ), f"eic key {eic_key} already exists in radix cache"
        logger.debug(
            f"load token from eic: {len(eic_key)}, node {load_node.id}, parent {last_node.id}"
        )
        last_node.children[eic_key.child_key(self.page_size)] = load_node
        load_node.parent = last_node
        return load_node

    def _match_for_remote_fetch(self, node: TreeNode, key: RadixKey):
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)
        local_prefix_len = 0

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = child.key.match(key, page_size=self.page_size)
            local_prefix_len += prefix_len
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                node = new_node
                break
            else:
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)
        temp_node = node
        local_evict_len = 0
        while temp_node.evicted:
            while not temp_node.backuped and temp_node.parent is not None:
                temp_node = temp_node.parent
                local_evict_len = 0
            if not temp_node.evicted:
                break
            local_evict_len += len(temp_node.host_value)
            temp_node = temp_node.parent
        return local_prefix_len, local_evict_len, node

    def _insert_remote_node(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                if node.evicted and node.host_value is None:
                    node.host_value = torch.arange(
                        len(node.key), dtype=torch.int32, device="cpu"
                    )
                if not node.evicted:
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                if new_node.evicted and new_node.host_value is None:
                    new_node.host_value = torch.arange(
                        len(new_node.key), dtype=torch.int32, device="cpu"
                    )
                if not new_node.evicted:
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.host_value = torch.arange(
                len(key), dtype=torch.int32, device="cpu"
            )
            node.children[child_key] = new_node
            self._calculate_content_hash(new_node)
        return total_prefix_length

    def match_from_remote(self, waiting_queue: List[Req]):
        compute_keys = []
        prev_hashes = []
        fetch_list = []
        if len(self.match_req_set) > 1000:
            self.match_req_set = self.match_req_set[500:]
        eic_keys = 0
        for req in waiting_queue:
            logger.debug(f"req {req.rid} match from eic")
            if req.rid in self.match_req_set:
                continue
            fill_ids = req.origin_input_ids + req.output_ids
            req_tokens = fill_ids[: len(fill_ids) - 1]
            req_key = RadixKey(req_tokens, req.extra_key)
            local_prefix_len, local_evict_len, last_node = self._match_for_remote_fetch(
                self.root_node, req_key
            )
            if (
                len(req_key) - local_prefix_len + local_evict_len
                < self.load_remote_threshold
            ):
                # skip loading from eic if the remaining key is too short
                logger.debug(f"req {req.rid} skip loading from eic")
                continue
            compute_keys.append(req_key[local_prefix_len:])
            if _need_calculate_hash(last_node, self.page_size):
                self._calculate_content_hash(last_node)
            prev_hashes.append(
                last_node.content_hash[-1] if last_node.content_hash else None
            )
            fetch_list.append((last_node, req, local_prefix_len, local_evict_len))
            self.match_req_set.append(req.rid)
            eic_keys += (len(req_key) - local_prefix_len) // self.page_size
            if self.eic_check_max_num > 0 and eic_keys >= self.eic_check_max_num:
                logger.info(
                    f"eic check max num {self.eic_check_max_num} reached, stop matching"
                )
                break
        if len(fetch_list) == 0:
            return
        # batch exist
        eic_prefix_lens = self.cache_controller.batch_find_longest_prefix_in_eic(
            compute_keys, prev_hashes
        )
        if len(eic_prefix_lens) == 0 or len(eic_prefix_lens) != len(fetch_list):
            return
        if self.tp_size > 1:
            eic_prefix_len_tensor = torch.tensor(
                eic_prefix_lens, dtype=torch.int64, device="cpu"
            )
            torch.distributed.all_reduce(
                eic_prefix_len_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            eic_prefix_lens = eic_prefix_len_tensor.tolist()
        for i, (last_node, req, local_prefix_len, local_evict_len) in enumerate(
            fetch_list
        ):
            eic_prefix_len = eic_prefix_lens[i]
            if eic_prefix_len + local_evict_len < self.load_remote_threshold:
                continue
            eic_key = compute_keys[i][:eic_prefix_len]
            logger.debug(
                f"req {req.rid} match from eic, "
                f"last node {last_node.id}, "
                f"local prefix len {local_prefix_len}, "
                f"eic prefix len {eic_prefix_len}"
            )
            self._insert_remote_node(last_node, eic_key)

    def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        # last_node = self.match_prefix_extend(key, last_node)
        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            while not last_node.backuped and last_node.parent is not None:
                last_node = last_node.parent
                last_host_node = last_node
                host_hit_length = 0
            if not last_node.evicted:
                break
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def write_backup(self, node: TreeNode, write_back=False):
        if node.evicted:
            return 0
        if _need_calculate_hash(node, self.page_size):
            self._calculate_content_hash(node)
        host_indices = self.cache_controller.write_page(
            device_indices=node.value,
            priority=-self.get_height(node),
            node_id=node.id,
            content_hash=node.content_hash,
        )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies
        start_time = time.perf_counter()
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
        host_content_hash = []
        for n in nodes_to_load:
            host_content_hash.extend(n.content_hash)

        # check key existed
        if self.load_back_check:
            check_keys = host_content_hash[
                : self.load_back_threshold // self.page_size + 1
            ]
            mask = self.cache_controller.mem_pool_host.batch_exist_page(check_keys)
            check_ret = all(mask)
            if self.tp_size > 1:
                check_tensor = torch.tensor(
                    0 if check_ret else 1, dtype=torch.bool, device="cpu"
                )
                torch.distributed.all_reduce(
                    check_tensor,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.tp_group,
                )
                check_ret = check_tensor.item() == 0
            if not check_ret:
                logger.warning(f"key has been evicted, skip load back")
                self.dec_lock_ref(ancester_node)
                return None

        device_indices = self.cache_controller.load_page(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            content_hash=host_content_hash,
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load_page(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                content_hash=host_content_hash,
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (
            ancester_node,
            last_hit_node,
            len(device_indices),
        )
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices
