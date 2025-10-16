from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Set

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.cpp_radix_tree.radix_tree import (
    IOHandle,
    RadixTreeCpp,
    TreeNodeCpp,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)


class RadixCacheCpp(BasePrefixCache):
    def _merge_tensor(self, l: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge a list of tensors into a single tensor.
        Args:
            l (List[torch.Tensor]): List of tensors to merge.
        Returns:
            torch.Tensor: Merged tensor.
        """
        if len(l) == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        elif len(l) == 1:
            return l[0]
        else:
            return torch.cat(l)

    def __init__(
        self,
        disable: bool,
        use_hicache: bool,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
        enable_kv_cache_events: bool = False,
        hicache_oracle: bool = False,
        enable_write_cancel: bool = False,
    ):
        self.disable = disable
        self.enable_write_cancel = enable_write_cancel

        assert (
            enable_kv_cache_events is False
        ), "HiRadixCache does not support kv cache events yet"
        self.kv_cache = token_to_kv_pool.get_kvcache()

        # record the nodes with ongoing write through
        self.ongoing_write_through: Set[IOHandle] = set()
        # record the node segments with ongoing load back
        self.ongoing_load_back: Set[IOHandle] = set()
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 2
        )
        self.device = token_to_kv_pool.device
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token_pool = req_to_token_pool
        self.page_size = page_size

        self.tp_group = tp_cache_group

        if not use_hicache:
            self.tree = RadixTreeCpp(
                disabled=self.disable,
                page_size=page_size,
                host_size=None,  # no host cache, this should be removed in the future
                write_through_threshold=self.write_through_threshold,
            )
            self.cache_controller = None
            return  # early return if hicache is not used

        raise NotImplementedError("Host cache is not supported yet")

    def reset(self):
        if self.cache_controller is not None:
            # need to clear the acks before resetting the cache controller
            raise NotImplementedError("Host cache is not supported yet")
        self.tree.reset()

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        device_indices_vec, host_indices_length, node_gpu, node_cpu = (
            self.tree.match_prefix(key.token_ids)
        )
        return MatchResult(
            device_indices=self._merge_tensor(device_indices_vec),
            last_device_node=node_gpu,
            last_host_node=node_cpu,
            host_hit_length=host_indices_length,
        )

    def _insert(self, key: RadixKey, value: torch.Tensor) -> int:
        """
        Insert a key-value pair into the radix tree.
        Args:
            key (RadixKey): The key to insert, represented as a RadixKey.
            value (torch.Tensor): The value to associate with the key.
        Returns:
            int: Number of device indices that were already present in the tree before the insertion.
        """
        ongoing_write, length = self.tree.writing_through(key.token_ids, value)
        if self.cache_controller is None:
            assert len(ongoing_write) == 0, "Implementation error"
            return length

        raise NotImplementedError("Host cache is not supported yet")

    def dec_lock_ref(self, node: TreeNodeCpp):
        """
        Decrement the reference count of a node to root of the radix tree.
        Args:
            node (TreeNodeCpp): The handle of the node to decrement the reference count for.
        """
        self.tree.lock_ref(node, False)  # do not increment

    def inc_lock_ref(self, node: TreeNodeCpp):
        """
        Increment the reference count of from a node to root of the radix tree.
        Args:
            node (TreeNodeCpp): The handle of the node to increment the reference count for.
        """
        self.tree.lock_ref(node, True)

    def evict(self, num_tokens: int):
        evicted_device_indices = self.tree.evict(num_tokens)
        for indice in evicted_device_indices:
            self.token_to_kv_pool.free(indice)

    def evictable_size(self):
        return self.tree.evictable_size()

    def protected_size(self):
        return self.tree.protected_size()

    def total_size(self):
        return self.tree.total_size()

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        assert req.req_pool_idx is not None
        all_token_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        token_ids = (req.origin_input_ids + req.output_ids)[:all_token_len]
        overall_len = len(token_ids)  # prefill + decode
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :overall_len]

        # NOTE: our C++ implementation don't need `token_ids` and `kv_indices` to be page-aligned
        # it will automatically align them, but length of them should be equal
        old_prefix_len = len(req.prefix_indices) // self.page_size * self.page_size
        page_aligned_overall_len = overall_len // self.page_size * self.page_size

        if is_insert:
            new_prefix_len = self._insert(
                RadixKey(token_ids, req.extra_key), kv_indices
            )
            # NOTE: kv_indices[:old_prefix_len] == req.prefix_indices
            assert old_prefix_len <= new_prefix_len, "Wrong prefix indices"
            # Free duplicates that were already in the pool
            if old_prefix_len < new_prefix_len:
                self.token_to_kv_pool.free(kv_indices[old_prefix_len:new_prefix_len])
        else:
            self.token_to_kv_pool.free(
                kv_indices[old_prefix_len:page_aligned_overall_len]
            )

        # need to free the unaligned part, since it cannot be inserted into the radix tree
        if page_aligned_overall_len < overall_len:
            # NOTE: sglang PagedAllocator support unaligned free (which will automatically align it)
            self.token_to_kv_pool.free(kv_indices[page_aligned_overall_len:])

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)
        self.req_to_token_pool.free(req.req_pool_idx)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        assert req.req_pool_idx is not None
        token_ids = req.fill_ids
        prefill_len = len(token_ids)  # prefill only (maybe chunked)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :prefill_len]

        # NOTE: our C++ implementation don't need `token_ids` and `kv_indices` to be page-aligned
        # it will automatically align them, but length of them should be equal
        old_prefix_len = len(req.prefix_indices) // self.page_size * self.page_size
        new_prefix_len = self._insert(RadixKey(token_ids, req.extra_key), kv_indices)

        # NOTE: kv_indices[:old_prefix_len] == req.prefix_indices
        assert old_prefix_len <= new_prefix_len, "Wrong prefix indices"

        # TODO(dark): optimize the `insert` and `match` (e.g. merge into 1 function)
        # The prefix indices need to updated to reuse the kv indices in the pool
        new_indices_vec, _, new_last_node, _ = self.tree.match_prefix(
            RadixKey(token_ids, req.extra_key).token_ids
        )
        new_indices = self._merge_tensor(new_indices_vec)
        assert new_prefix_len <= len(new_indices)

        # KVCache between old & new is newly generated, but already exists in the pool
        # we need to free this newly generated kv indices and reuse the indices in the pool
        if old_prefix_len < new_prefix_len:
            self.token_to_kv_pool.free(kv_indices[old_prefix_len:new_prefix_len])
            reused_indices = new_indices[old_prefix_len:new_prefix_len]
            self.req_to_token_pool.req_to_token[
                req.req_pool_idx, old_prefix_len:new_prefix_len
            ] = reused_indices

        if req.last_node != new_last_node:
            self.dec_lock_ref(req.last_node)
            self.inc_lock_ref(new_last_node)

        # NOTE: there might be unaligned tail, so we may need to append it
        assert len(new_indices) <= prefill_len < len(new_indices) + self.page_size
        if self.page_size != 1 and len(new_indices) < prefill_len:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        return self.tree.debug_print()
