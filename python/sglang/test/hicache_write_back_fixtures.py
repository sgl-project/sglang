"""Fakes that drive a UnifiedRadixCache's HiCache write-back path on CPU.

A slot-only host pool (the real slot accounting, no KV buffers), a write-back
controller whose D->H copies land on submit and whose H->D load-backs land on
start_loading, and builders of a Full-only and a FULL+SWA write-back cache
over them.
"""

import threading
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

PAGE = 4
# A write-back batch: eight two-page sequences.
SEQ_TOKENS = 2 * PAGE
SEQS_PER_BATCH = 8
BATCH_TOKENS = SEQS_PER_BATCH * SEQ_TOKENS


class SlotOnlyHostPool(HostKVCache):
    """The real host-pool slot accounting, without KV buffers."""

    def __init__(self, size: int):
        self.size = size
        self.page_size = PAGE
        self.dcp_size = 1
        self.layout = "page_first"
        self.device = "cpu"
        self.can_use_write_back_jit = False
        self.lock = threading.RLock()
        self.clear()

    def get_size_per_token(self):
        return 1

    def init_kv_buffer(self):
        return None

    def load_to_device_per_layer(self, *args, **kwargs):
        raise AssertionError("the fake controller never copies")

    def backup_from_device_all_layer(self, *args, **kwargs):
        raise AssertionError("the fake controller never copies")

    def get_data_page(self, *args, **kwargs):
        raise AssertionError

    def get_dummy_flat_data_page(self):
        raise AssertionError

    def set_from_flat_data_page(self, *args, **kwargs):
        raise AssertionError

    def destroy(self):
        pass


class DoneEvent:
    def synchronize(self):
        pass

    def query(self):
        return True


class InstantWriteBackController:
    """HybridCacheController's host allocation with D->H copies that finish
    on submit, and H->D load-backs that take device slots (side pools' too)
    from the cache's allocators and finish on start_loading. Acks count each
    pool's tokens as HybridCacheController's do."""

    write_policy = "write_back"
    enable_storage = False
    write = HybridCacheController.write
    allocate_host_transfers = HybridCacheController.allocate_host_transfers
    _uses_shared_host_domain = HybridCacheController._uses_shared_host_domain
    _resolve_device_transfers = HybridCacheController._resolve_device_transfers
    _num_tokens_by_pool = HybridCacheController._num_tokens_by_pool

    def __init__(self, host_group: HostPoolGroup, device_allocator):
        self.mem_pool_host = host_group
        self.mem_pool_device_allocator = device_allocator
        self.write_queue = []
        self.ack_write_queue = []
        self.load_queue = []
        self.ack_load_queue = []

    def _done_ack(self, queue):
        op = CacheOperation.merge_ops(queue)
        queue.clear()
        return SimpleNamespace(
            finish_event=DoneEvent(),
            node_ids=op.node_ids,
            num_tokens_by_pool=self._num_tokens_by_pool(op),
            num_bytes=0,
            timing_enabled=False,
        )

    def start_writing(self):
        if self.write_queue:
            self.ack_write_queue.append(self._done_ack(self.write_queue))

    def load(
        self,
        host_indices,
        priority=None,
        node_id=-1,
        extra_pools=None,
        *,
        device_indices=None,
        anchor_indices=None,
    ):
        assert device_indices is None and anchor_indices is None
        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if host_indices.numel() == 0:
            device_indices = torch.empty((0,), dtype=torch.int64)
        else:
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None
        pool_transfers = None
        if extra_pools:
            pool_transfers = self._resolve_device_transfers(
                extra_pools,
                kv_device_indices=device_indices,
                kv_host_indices=host_indices,
            )
            if pool_transfers is None:
                if host_indices.numel() > 0:
                    full_allocator.free(device_indices)
                return None
        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def start_loading(self):
        if not self.load_queue:
            return -1
        self.ack_load_queue.append(self._done_ack(self.load_queue))
        return 0


def write_back_cache(
    tree_core_backend: str,
    *,
    device_tokens: int,
    host_tokens: int,
    enable_kv_cache_events: bool = False,
):
    """A Full-only write-back UnifiedRadixCache on ``tree_core_backend``:
    ``device_tokens`` of paged device KV, a slot-only host pool of
    ``host_tokens`` and an InstantWriteBackController. Returns the cache, its
    device allocator and its host pool."""
    allocator = PagedTokenToKVPoolAllocator(
        size=device_tokens,
        page_size=PAGE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE,
            disable=False,
            tree_components=(ComponentType.FULL,),
            tree_core_backend=tree_core_backend,
            enable_kv_cache_events=enable_kv_cache_events,
        )
    )
    host_pool = SlotOnlyHostPool(host_tokens)
    host_group = HostPoolGroup(
        [
            PoolEntry(
                name=PoolName.KV,
                host_pool=host_pool,
                device_pool=None,
                layer_mapper=lambda layer_id: layer_id,
                is_primary_index_anchor=True,
            )
        ]
    )
    cache.host_pool_group = host_group
    cache.cache_controller = InstantWriteBackController(host_group, allocator)
    cache.components[ComponentType.FULL]._full_kv_pool_host = host_pool
    cache.tree_core.set_hicache_enabled()
    cache.is_write_back = True
    cache.write_through_threshold = 1 << 30
    # init_hicache's default; these fixtures never call init_hicache.
    cache.load_back_threshold = 10
    return cache, allocator, host_pool


def hybrid_write_back_cache(
    tree_core_backend: str,
    *,
    device_tokens: int,
    swa_device_tokens: int,
    host_tokens: int,
    swa_host_tokens: int,
    sliding_window: int,
    enable_kv_cache_events: bool = False,
):
    """A FULL+SWA write-back UnifiedRadixCache on ``tree_core_backend``, wired
    as hybrid_pool_assembler wires it (separate KV and SWA host pools, SWA
    host and device reclaim hooks) over slot-only host pools. Returns the
    cache, its device allocator and its KV and SWA host pools."""
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _evict_swa_for_device_alloc,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    kv_pool = SWAKVPool(
        size=device_tokens,
        size_swa=swa_device_tokens,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=1,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cpu",
    )
    allocator = SWATokenToKVPoolAllocator(
        size=device_tokens,
        size_swa=swa_device_tokens,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    # CPU pools: the paged allocators' Triton path cannot read them on a GPU host.
    allocator.full_attn_allocator.use_triton_kernels = False
    allocator.swa_attn_allocator.use_triton_kernels = False
    cache = UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE,
            disable=False,
            sliding_window_size=sliding_window,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
            tree_core_backend=tree_core_backend,
            enable_kv_cache_events=enable_kv_cache_events,
        )
    )
    kv_host = SlotOnlyHostPool(host_tokens)
    swa_host = SlotOnlyHostPool(swa_host_tokens)
    host_group = HostPoolGroup(
        [
            PoolEntry(
                name=PoolName.KV,
                host_pool=kv_host,
                device_pool=kv_pool.full_kv_pool,
                layer_mapper=lambda layer_id: layer_id,
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.SWA,
                host_pool=swa_host,
                device_pool=kv_pool.swa_kv_pool,
                layer_mapper=lambda layer_id: layer_id,
                host_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
                device_evict_fn=lambda n: _evict_swa_for_device_alloc(cache, n),
                device_alloc_fn=allocator.swa_attn_allocator.alloc,
                device_free_fn=allocator.swa_attn_allocator.free,
            ),
        ]
    )
    cache.host_pool_group = host_group
    cache.cache_controller = InstantWriteBackController(host_group, allocator)
    cache.full_kv_pool_host = kv_host
    cache.swa_kv_pool_host = swa_host
    cache.components[ComponentType.FULL]._full_kv_pool_host = kv_host
    cache.components[ComponentType.SWA]._swa_kv_pool_host = swa_host
    cache.tree_core.set_hicache_enabled()
    cache.tree_core.has_swa_host_pool = True
    cache.is_write_back = True
    cache.write_through_threshold = 1 << 30
    cache.load_back_threshold = 10
    return cache, allocator, kv_host, swa_host


def batch_seqs(batch: int) -> list[list[int]]:
    """Batch ``batch``'s sequences, disjoint from every other batch's."""
    return [
        [1000 * (batch * SEQS_PER_BATCH + i) + t for t in range(SEQ_TOKENS)]
        for i in range(SEQS_PER_BATCH)
    ]


def insert_seq(cache, allocator, seq):
    value = allocator.alloc(len(seq))
    assert value is not None
    cache.insert(InsertParams(key=RadixKey(array("q", seq)), value=value))


def insert_hybrid_seq(cache, allocator, seq):
    """insert_seq for a FULL+SWA cache: extend ``seq``'s device-resident
    prefix by the rest of ``seq`` through the SWA allocator's alloc_extend."""
    prefix = cache.path_device_indices(match_seq(cache, seq).last_device_node)
    prefix_len = len(prefix)
    assert prefix_len % PAGE == 0 and prefix_len < len(seq)
    value = allocator.alloc_extend(
        torch.tensor([prefix_len]),
        torch.tensor([prefix_len]),
        torch.tensor([len(seq)]),
        torch.tensor([len(seq)]),
        torch.tensor([-1]),
        len(seq) - prefix_len,
    )
    assert value is not None
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", seq)),
            value=torch.cat([prefix.to(torch.int64), value.to(torch.int64)]),
            prev_prefix_len=prefix_len,
        )
    )


def match_seq(cache, seq):
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
