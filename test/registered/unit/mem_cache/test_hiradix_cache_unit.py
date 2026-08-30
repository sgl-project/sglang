"""Unit tests for srt/mem_cache/hiradix_cache.py KV cache events and
storage-prefetch host-slot ownership."""

import os
import unittest
from array import array

import torch

from sglang.srt.disaggregation.kv_events import BlockStored, StorageMedium
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import PrefetchOperation
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd")

PAGE_SIZE = 2


def _require_cuda_and_process_group():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for HiRadixCache tests.")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)


def _build_cache():
    server_args = ServerArgs(
        model_path="dummy",
        page_size=PAGE_SIZE,
        hicache_io_backend="direct",
        hicache_mem_layout="layer_first",
        hicache_write_policy="write_through",
    )
    set_global_server_args_for_scheduler(server_args)
    req_to_token_pool = ReqToTokenPool(
        size=10,
        max_context_len=512,
        device="cuda",
        enable_memory_saver=False,
    )
    kv_pool = MHATokenToKVPool(
        size=256,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=64,
        layer_num=4,
        device="cuda",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=256,
        dtype=torch.bfloat16,
        device="cuda",
        kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=PAGE_SIZE,
        disable=False,
        enable_kv_cache_events=True,
        tp_cache_group=torch.distributed.group.WORLD,
    )
    cache = HiRadixCache(params, server_args)
    # Disable hit-count-driven write-through; tests back up explicitly.
    cache.write_through_threshold = 1 << 30
    return cache, allocator


class TestHiRadixCacheKVEvents(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _require_cuda_and_process_group()

    def _insert(self, cache, allocator, tokens):
        key = RadixKey(array("q", tokens))
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        return cache.insert(InsertParams(key=key, value=value[: len(tokens)]))

    def _leaf_for(self, cache, tokens):
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertIsNot(match.last_device_node, cache.root_node)
        return match.last_device_node

    def _stored_cpu_events(self, cache):
        return [
            e
            for e in cache.take_events()
            if isinstance(e, BlockStored) and e.medium == StorageMedium.CPU
        ]

    def test_split_pending_write_through_publishes_fragments(self):
        cache, allocator = _build_cache()
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4])
        node = self._leaf_for(cache, [1, 2, 3, 4])
        backed_up = cache.write_backup(node, write_back=True)
        self.assertGreater(backed_up, 0)

        # Split the node while its write-through DMA is still pending.
        self._insert(cache, allocator, [1, 2, 5, 6])
        self.assertEqual(self._stored_cpu_events(cache), [])

        cache.writing_check(write_back=True)

        # Both split fragments must be published as one parent-linked batch.
        stored_cpu = self._stored_cpu_events(cache)
        self.assertEqual(len(stored_cpu), 1)
        self.assertEqual(list(stored_cpu[0].token_ids), [1, 2, 3, 4])
        self.assertIsNone(stored_cpu[0].parent_block_hash)
        self.assertEqual(len(stored_cpu[0].block_hashes), 2)


class TestHiRadixCachePrefetchHostOwnership(CustomTestCase):
    """A storage prefetch on a DSA / MiniMax stack must publish its fetched span
    and leave no host slot stranded.

    Regression: the fetch completed, but nothing reached the tree and the host
    slots it held could never be reclaimed -- L3 hit rate read as zero while the
    host pool drained away request by request.
    """

    @classmethod
    def setUpClass(cls):
        _require_cuda_and_process_group()

    @staticmethod
    def _kv_derived_indexer_transfer():
        """The transfer HiRadixCache._get_extra_pools builds for DSA / MiniMax."""
        return PoolTransfer(
            name=PoolName.INDEXER,
            hit_policy=PoolHitPolicy.ALL_PAGES,
            indices_from_pool=PoolName.KV,
        )

    def _prefetch_operation(self, prefetch_key, completed_tokens, pool_transfers):
        operation = PrefetchOperation(
            "rid-0",
            prefetch_key,
            pool_transfers=pool_transfers,
        )
        operation.completed_tokens = completed_tokens
        # Hex digests: the KV cache event path parses these back into int64.
        operation.hash_value = [
            f"{i:032x}" for i in range(completed_tokens // PAGE_SIZE)
        ]
        return operation

    def test_independent_sidecar_still_clamps_the_usable_prefix(self):
        """A sidecar owning its own slots (SWA / Mamba) still shortens the prefix.

        Guards the opposite degradation from the leak below: dropping the clamp
        altogether would publish pages the sidecar never fetched.
        """
        cache, _ = _build_cache()
        completed_tokens = 4 * PAGE_SIZE
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.arange(PAGE_SIZE, dtype=torch.int64),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        operation = self._prefetch_operation(
            RadixKey(array("q", list(range(completed_tokens)))),
            completed_tokens,
            [swa_transfer],
        )
        operation.pool_storage_result.update_extra_pool_hit_pages({PoolName.SWA: 1})

        self.assertEqual(cache._clamp_prefetch_result(operation), PAGE_SIZE)

    def test_prefetch_with_kv_derived_sidecar_leaks_no_host_slots(self):
        cache, allocator = _build_cache()
        host_pool = cache.cache_controller.mem_pool_host

        # Device-resident prefix; the fetched suffix grafts underneath it.
        prefix_tokens = [1, 2, 3, 4]
        value = allocator.alloc(len(prefix_tokens))
        self.assertIsNotNone(value)
        cache.insert(InsertParams(key=RadixKey(array("q", prefix_tokens)), value=value))
        anchor = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", prefix_tokens)))
        ).last_device_node
        self.assertIsNot(anchor, cache.root_node)

        available_before = host_pool.available_size()

        completed_tokens = 4 * PAGE_SIZE
        prefetch_key = RadixKey(array("q", list(range(100, 100 + completed_tokens))))
        operation = self._prefetch_operation(
            prefetch_key, completed_tokens, [self._kv_derived_indexer_transfer()]
        )
        host_indices = host_pool.alloc(completed_tokens)
        self.assertIsNotNone(host_indices)
        operation.host_indices = host_indices

        anchor.protect_host()
        cache.ongoing_prefetch["rid-0"] = (anchor, prefetch_key, operation)
        # Only initialized once a storage backend is attached; seed it the way
        # prefetch_from_storage would.
        cache.cache_controller.prefetch_tokens_occupied = len(prefetch_key)

        cache._handle_prefetch_result(operation)
        # The completed_req ack releases the untransferred tail.
        host_pool.free(operation.host_indices[operation.completed_tokens :])

        # The whole fetched span reached the tree.
        self.assertEqual(cache.pop_prefetch_loaded_tokens("rid-0"), completed_tokens)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)

        # Ownership conservation: every slot the prefetch took is reclaimable
        # through the normal host-eviction path.
        cache.evict_host(completed_tokens)
        self.assertEqual(host_pool.available_size(), available_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
