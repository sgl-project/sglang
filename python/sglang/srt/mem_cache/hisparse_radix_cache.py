"""CPU host radix cache for HiSparse prefix sharing.

Implements BasePrefixCache so it can serve as the scheduler's tree_cache.
All KV data lives on the CPU host pool; match_prefix returns host_hit_length
so the scheduler can trigger init_load_back to copy prefix KV from CPU to GPU
before prefill, reducing redundant GPU computation.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


class HiSparseRadixCache(BasePrefixCache):
    """Radix tree managing CPU host KV indices for HiSparse prefix sharing.

    The internal host-side RadixCache stores host pool indices in TreeNode.value.
    match_prefix returns empty device_indices (nothing cached on GPU) but sets
    host_hit_length so the scheduler's add_one_req path can call init_load_back
    to copy the prefix KV from CPU to GPU before the forward pass.
    """

    def __init__(self, params: CacheInitParams):
        # Device-side pools from CacheInitParams (used by cache_finished_req,
        # cache_unfinished_req, init_load_back, and available_and_evictable_str)
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.tp_cache_group = params.tp_cache_group
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # Host pool, internal radix tree, and cache controller are set up
        # lazily via set_host_pool() because the hisparse coordinator (which
        # owns the host pool) is created after the scheduler's tree_cache.
        self.host_pool: Optional[HostKVCache] = None
        self._host_cache: Optional[RadixCache] = None
        self.cache_controller: Optional[HiCacheController] = None

        # Cached host indices from the last match_prefix call, keyed by
        # the matched host node id.  Used by init_load_back to avoid a
        # fragile bottom-up tree walk.
        self._last_match_host_indices: dict = {}

    # -- Lazy initialisation of the host-side radix tree --

    def set_host_pool(self, host_pool: HostKVCache) -> None:
        """Attach the host memory pool, create the internal radix tree,
        and set up the HiCacheController for async overlap DMA.

        Must be called once after the HiSparseCoordinator is available.
        """
        self.host_pool = host_pool
        host_params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=host_pool,
            page_size=host_pool.page_size,
        )
        self._host_cache = RadixCache(host_params)

        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=host_pool,
            page_size=self.page_size,
            tp_group=self.tp_cache_group,
            load_cache_event=threading.Event(),
            io_backend="kernel",
        )

    @property
    def _cache(self) -> RadixCache:
        assert (
            self._host_cache is not None
        ), "HiSparseRadixCache: host pool not set yet — call set_host_pool() first"
        return self._host_cache

    # -- BasePrefixCache required properties --

    @property
    def disable(self) -> bool:
        return False

    # -- BasePrefixCache abstract method implementations --

    def reset(self):
        pass

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        empty = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self._host_cache is None or len(params.key) == 0:
            return MatchResult(
                device_indices=empty,
                last_device_node=None,
                last_host_node=None,
                host_hit_length=0,
            )

        key = RadixKey(token_ids=params.key.token_ids, extra_key=params.key.extra_key)
        result = self._cache.match_prefix(MatchPrefixParams(key=key))
        raw_host_hit_len = len(result.device_indices)
        host_hit_len = (raw_host_hit_len // self.page_size) * self.page_size

        last_node = result.last_device_node
        if host_hit_len > 0 and last_node is not None:
            self._last_match_host_indices[id(last_node)] = result.device_indices

        return MatchResult(
            device_indices=empty,
            last_device_node=None,
            last_host_node=last_node,
            host_hit_length=host_hit_len,
        )

    def init_load_back(self, params: InitLoadBackParams) -> Tuple[torch.Tensor, Any]:
        """Load prefix KV from CPU host pool to GPU device pool.

        HiSparse uses a dual-allocator model (logical pool + device buffer)
        with a mapping between them.  This method performs the full three-step
        allocation so that attention kernels can translate logical indices to
        valid device buffer locations.
        """
        last_node = params.last_host_node
        host_hit_length = params.host_hit_length
        empty = torch.empty((0,), dtype=torch.int64, device=self.device)

        if last_node is None or host_hit_length <= 0 or self.host_pool is None:
            return empty, last_node

        host_indices = self._last_match_host_indices.pop(id(last_node), None)
        if host_indices is None or host_indices.numel() == 0:
            return empty, last_node

        page_aligned_len = (host_hit_length // self.page_size) * self.page_size
        if page_aligned_len <= 0:
            return empty, last_node
        host_indices = host_indices[:page_aligned_len]

        allocator = self.token_to_kv_pool_allocator
        logical_alloc = getattr(allocator, "logical_attn_allocator", None)
        hisparse_alloc = getattr(allocator, "hisparse_attn_allocator", None)

        if logical_alloc is None or hisparse_alloc is None:
            return empty, last_node

        # Step 1: allocate logical indices
        logical_indices = logical_alloc.alloc(page_aligned_len)
        if logical_indices is None:
            logger.warning(
                "init_load_back: failed to allocate %d logical slots",
                page_aligned_len,
            )
            return empty, last_node

        # Step 2: allocate hisparse device buffer indices
        hisparse_indices = hisparse_alloc.alloc(page_aligned_len)
        if hisparse_indices is None:
            logical_alloc.free(logical_indices)
            logger.warning(
                "init_load_back: failed to allocate %d hisparse device slots",
                page_aligned_len,
            )
            return empty, last_node

        # Step 3: set up logical → hisparse device mapping
        allocator.full_to_hisparse_device_index_mapping[logical_indices] = (
            hisparse_indices
        )

        # Queue async DMA instead of synchronous per-layer copy.
        # start_loading() will move host_indices to GPU via move_indices().
        self.cache_controller.load_queue.append(
            CacheOperation(host_indices, hisparse_indices, node_id=-1)
        )

        return logical_indices, last_node

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Release device KV pool slots.  Host indices are managed by the
        coordinator's request_finished which inserts into the host tree."""
        kv_committed_len = req.pop_committed_kv_cache()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Set prefix_indices for chunked prefill scheduling."""
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def evict(self, params: EvictParams) -> EvictResult:
        if self._host_cache is None:
            return EvictResult()
        result = self._cache.evict(params)
        return result

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if self._host_cache is None or node is None:
            return IncLockRefResult(delta=0)
        return self._cache.inc_lock_ref(node)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self._host_cache is None or node is None:
            return DecLockRefResult(delta=0)
        return self._cache.dec_lock_ref(node, params)

    def evictable_size(self) -> int:
        # Device-side evictable size is 0: HiSparseRadixCache does not retain
        # device KV slots in the tree (they are freed in cache_finished_req).
        # The host tree's evictable size is exposed via host_evictable_size().
        return 0

    def protected_size(self):
        return 0

    def ready_to_load_host_cache(self) -> int:
        if self.cache_controller is None:
            return -1
        return self.cache_controller.start_loading()

    def check_hicache_events(self):
        if self.cache_controller is None:
            return
        finish_count = 0
        for _, finish_event, _ in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                break
            finish_count += 1
        del self.cache_controller.ack_load_queue[:finish_count]

    def pretty_print(self):
        if self._host_cache is None:
            return "<HiSparseRadixCache: host pool not initialised>"
        return self._cache.pretty_print()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        host_evictable = self.host_evictable_size() if self._host_cache else 0
        return (
            f"Available device tokens: {available_size}, "
            f"host evictable: {host_evictable}\n"
        )

    # -- Convenience methods for the coordinator --

    def host_match_prefix(
        self,
        token_ids: list,
        extra_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Any, int]:
        """Direct host-tree match returning (host_indices, last_node, matched_len)."""
        key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        result = self._cache.match_prefix(MatchPrefixParams(key=key))
        return (
            result.device_indices,
            result.last_device_node,
            len(result.device_indices),
        )

    def host_insert(
        self,
        token_ids: list,
        host_indices: torch.Tensor,
        extra_key: Optional[str] = None,
    ) -> int:
        """Insert host pool indices into the host tree.

        Returns the prefix_len (tokens already present).
        """
        key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        result = self._cache.insert(InsertParams(key=key, value=host_indices))
        return result.prefix_len

    def host_evict(self, num_tokens: int):
        return self._cache.evict(EvictParams(num_tokens=num_tokens))

    def host_evictable_size(self) -> int:
        return self._cache.evictable_size()

    def host_inc_lock_ref(self, node: TreeNode):
        return self._cache.inc_lock_ref(node)

    def host_dec_lock_ref(self, node: TreeNode):
        return self._cache.dec_lock_ref(node)

    @property
    def root_node(self) -> TreeNode:
        return self._cache.root_node
