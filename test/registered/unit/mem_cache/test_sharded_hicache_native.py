"""CPU checks of native DSA host declarations and sharded allocation.

Only the device-buffer allocation and host payload allocation are omitted:
device pools retain their real DSA declaration methods, and host pools use
their native ``is_dummy`` constructors. The assembler, host layer mappings,
sharded allocator, localization and clear/free paths are the production ones.

These tests do not exercise pinned allocations, GPU transfer kernels, CUDA
events or gather-stream fences. Those require separate GPU integration tests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import prepare_host_pool_config
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host import dsa as pool_host_dsa
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.sharded_hicache import (
    ShardedDSAHostPoolGroup,
    local_transfer_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64
DEVICE_TOKENS = 4096
# Ratio 2 produces 8192 usable rows plus the native host's alignment page.
HOST_TOKENS = 8256
ALLOCATABLE_HOST_TOKENS = 8192


def _device_pool(*, layers, rank, shards, skip_topk_layers):
    """Supply geometry to the real DSA class without creating CUDA buffers."""
    pool = object.__new__(DSATokenToKVPool)
    pool.layer_num = layers
    pool.size = DEVICE_TOKENS
    pool.start_layer = 0
    pool.end_layer = layers - 1
    pool.store_dtype = torch.bfloat16
    pool.kv_lora_rank = 512
    pool.qk_rope_head_dim = 64
    pool.kv_cache_dim = 576
    pool.index_head_dim = 128
    pool.page_size = PAGE_SIZE
    pool.index_page_size = PAGE_SIZE
    pool.skip_topk_layers = skip_topk_layers
    pool.index_key_cache = SimpleNamespace(buffer=[object()] * layers)
    # Page ownership and layer ownership are different mechanisms.
    pool.layer_shard_enabled = False
    pool.shard_rank = rank
    pool.shard_size = shards
    return pool


def _native_group(*, rank, shards, layout, mtp):
    target = _device_pool(
        layers=3, rank=rank, shards=shards, skip_topk_layers=[False, True, False]
    )
    drafts = (
        (_device_pool(layers=1, rank=rank, shards=shards, skip_topk_layers=[False]),)
        if mtp
        else ()
    )
    config = prepare_host_pool_config(
        decls=target.host_pool_decls(),
        full_layer_mapping={0: 0, 1: 1, 2: 2},
        transfer_layer_id_max=3,
        transfer_page_size=PAGE_SIZE,
        packed_draft_device_pools=drafts,
    )
    anchor = MLATokenToKVPoolHost(
        target,
        host_to_device_ratio=2,
        host_size=0,
        page_size=PAGE_SIZE,
        layout=layout,
        pin_memory=False,
        is_dummy=True,
        override_kv_cache_dim=576,
        mtp_draft_device_pools=drafts,
    )
    native_indexer_constructor = pool_host_dsa.DSAIndexerPoolHost

    def dummy_indexer(*args, **kwargs):
        return native_indexer_constructor(
            *args, **kwargs, pin_memory=False, is_dummy=True
        )

    # Keep this CPU-only on Linux and macOS; do not change production platform
    # detection or substitute the declaration/assembly/allocation algorithms.
    with (
        patch.object(pool_host_dsa, "DSAIndexerPoolHost", dummy_indexer),
        patch.object(
            hybrid_pool_assembler, "_get_allocator_type", return_value="default"
        ),
    ):
        entries = hybrid_pool_assembler._build_pool_entries(
            config=config, root_host_pool=anchor
        )
    return ShardedDSAHostPoolGroup(entries, target), drafts


class TestShardedHiCacheNative(unittest.TestCase):
    def test_native_declarations_capacity_clear_and_packed_layers(self):
        # 48 cases: every rank in CP4/CP8, two layouts, MTP on/off.
        for shards in (4, 8):
            for rank in range(shards):
                for layout in ("layer_first", "page_first"):
                    for mtp in (False, True):
                        with self.subTest(cp=shards, rank=rank, layout=layout, mtp=mtp):
                            self._check_native_group(rank, shards, layout, mtp)

    def _check_native_group(self, rank, shards, layout, mtp):
        group, drafts = _native_group(rank=rank, shards=shards, layout=layout, mtp=mtp)
        kv = group.entry_map[PoolName.KV]
        indexer = group.entry_map[PoolName.INDEXER]
        self.assertIsInstance(kv.host_pool, MLATokenToKVPoolHost)
        self.assertIsInstance(indexer.host_pool, pool_host_dsa.DSAIndexerPoolHost)
        self.assertEqual(group.physical_size, HOST_TOKENS)
        self.assertEqual(group.logical_size, ALLOCATABLE_HOST_TOKENS * shards)
        self.assertEqual(group.available_size(), group.logical_size)
        self.assertEqual(kv.host_pool.size, HOST_TOKENS)
        self.assertEqual(indexer.host_pool.size, HOST_TOKENS)
        self.assertEqual(kv.host_pool.layout, layout)
        self.assertEqual(indexer.host_pool.layout, layout)
        self.assertEqual(kv.host_pool.layer_num, 3 + int(mtp))
        self.assertEqual(indexer.host_pool.layer_num, 2 + int(mtp))
        self.assertEqual(kv.packed_draft_device_pools, drafts)
        self.assertEqual(indexer.packed_draft_device_pools, drafts)
        self.assertEqual(kv.host_pool.mtp_draft_device_pools, drafts)
        self.assertEqual(indexer.host_pool.mtp_draft_device_pools, drafts)
        self.assertEqual(indexer.host_pool.indexer_page_stride_size, 8448)
        self.assertEqual(indexer.host_pool._live_target_layers, [0, 2])
        self.assertEqual(indexer.host_pool._host_layer_index(2), 1)
        self.assertEqual(indexer.layer_mapper(0), 0)
        self.assertIsNone(indexer.layer_mapper(1))
        self.assertEqual(indexer.layer_mapper(2), 2)
        if mtp:
            self.assertEqual(kv.layer_mapper(3), 3)
            self.assertEqual(indexer.layer_mapper(3), 3)
            self.assertEqual(indexer.host_pool._draft_host_layer(3), 2)
        else:
            self.assertIsNone(kv.layer_mapper(3))
            self.assertIsNone(indexer.layer_mapper(3))
        self.assertIsNone(indexer.layer_mapper(4))

        # Two successive backups can reuse the same physical device addresses
        # while keeping disjoint host pages. Each source fits the device pool.
        source_pages = torch.arange(shards, (DEVICE_TOKENS // PAGE_SIZE + 1) * shards)
        source = (source_pages[:, None] * PAGE_SIZE + torch.arange(PAGE_SIZE)).flatten()
        first = group.alloc_matching(source)
        second = group.alloc_matching(source)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(first), DEVICE_TOKENS * shards)
        allocated = torch.cat((first, second))
        self.assertEqual(len(torch.unique(allocated)), group.logical_size)
        self.assertEqual(group.available_size(), 0)
        self.assertIsNone(group.alloc_matching(source[:PAGE_SIZE]))
        host, device = local_transfer_indices(
            allocated,
            torch.cat((source, source)),
            page_size=PAGE_SIZE,
            rank=rank,
            size=shards,
        )
        self.assertEqual(len(host), ALLOCATABLE_HOST_TOKENS)
        self.assertEqual(host.min().item(), PAGE_SIZE)
        self.assertEqual(host.max().item(), HOST_TOKENS - 1)
        self.assertEqual(device.min().item(), PAGE_SIZE)
        self.assertEqual(device.max().item(), DEVICE_TOKENS + PAGE_SIZE - 1)
        self.assertEqual(len(torch.unique(host)), ALLOCATABLE_HOST_TOKENS)

        self.assertEqual(group.free(first, pool=PoolName.KV), len(first))
        self.assertEqual(group.available_size(), len(first))
        self.assertEqual(group.free(second), len(second))
        self.assertEqual(group.available_size(), group.logical_size)
        # Clear while something is still allocated, rather than clearing only
        # an already empty logical allocator; both native pools reset as well.
        self.assertIsNotNone(group.alloc_matching(source))
        group.clear()
        self.assertEqual(group.available_size(), group.logical_size)
        for entry in group.entries:
            self.assertEqual(entry.host_pool.available_size(), HOST_TOKENS)
            self.assertFalse(entry.host_pool.slot_used.any())
        self.assertIsNotNone(group.alloc_matching(source))


if __name__ == "__main__":
    unittest.main()
