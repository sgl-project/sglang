"""Unit tests for srt/mem_cache/hiradix_cache.py KV cache events."""

import os
import unittest
from array import array

import torch

from sglang.srt.disaggregation.kv_events import BlockStored, StorageMedium
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd")

PAGE_SIZE = 2


class TestHiRadixCacheKVEvents(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for HiRadixCache tests.")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29601")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

    def _build_cache(self):
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
        cache, allocator = self._build_cache()
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4])
        node = self._leaf_for(cache, [1, 2, 3, 4])
        backed_up = cache.write_backup(node, write_back=True)
        self.assertGreater(backed_up, 0)

        # Split the node while its write-through DMA is still pending.
        self._insert(cache, allocator, [1, 2, 5, 6])
        self.assertEqual(self._stored_cpu_events(cache), [])

        cache.writing_check(write_back=True)

        # Both split fragments must be published, with intact parentage.
        stored_cpu = self._stored_cpu_events(cache)
        self.assertEqual(
            [list(e.token_ids) for e in stored_cpu],
            [[1, 2], [3, 4]],
        )
        self.assertIsNone(stored_cpu[0].parent_block_hash)
        self.assertEqual(stored_cpu[1].parent_block_hash, stored_cpu[0].block_hashes[0])

    def test_prefetch_host_insert_skips_unbacked_device_parent(self):
        # Regression: prefetch refill (_insert_helper_host) must NOT attach a
        # host-only child under a device node that is not backed up. Doing so
        # breaks the backup invariant (backed-up nodes form a contiguous prefix
        # from root) and later trips the `_evict_regular` non-leaf assertion.
        cache, allocator = self._build_cache()

        # A device-only, not-backed-up leaf (write_through_threshold disabled).
        self._insert(cache, allocator, [1, 2])
        node = self._leaf_for(cache, [1, 2])
        self.assertFalse(node.backuped)
        self.assertEqual(len(node.children), 0)

        host_pool = cache.cache_controller.mem_pool_host
        avail_before = host_pool.available_size()
        host_indices = host_pool.alloc(PAGE_SIZE)
        self.assertIsNotNone(host_indices)

        # Attempt to refill a host-only suffix [3, 4] under the un-backed-up node.
        matched_length, inserted_length = cache._insert_helper_host(
            node,
            RadixKey(array("q", [3, 4])),
            host_indices,
            ["0123456789abcdef"],
        )

        # Invariant kept: nothing attached, nothing reported as loaded, and the
        # prefetched host pages were released (no leak).
        self.assertEqual(len(node.children), 0)
        self.assertEqual(inserted_length, 0)
        self.assertEqual(host_pool.available_size(), avail_before)

    def test_prefetch_host_insert_allows_backed_up_parent(self):
        # Positive control: under a backed-up parent the invariant holds, so the
        # host-only suffix is attached normally.
        cache, allocator = self._build_cache()

        self._insert(cache, allocator, [1, 2])
        node = self._leaf_for(cache, [1, 2])
        self.assertGreater(cache.write_backup(node, write_back=True), 0)
        self.assertTrue(node.backuped)

        host_pool = cache.cache_controller.mem_pool_host
        host_indices = host_pool.alloc(PAGE_SIZE)
        self.assertIsNotNone(host_indices)

        matched_length, inserted_length = cache._insert_helper_host(
            node,
            RadixKey(array("q", [3, 4])),
            host_indices,
            ["0123456789abcdef"],
        )

        self.assertEqual(inserted_length, PAGE_SIZE)
        self.assertEqual(len(node.children), 1)
        child = next(iter(node.children.values()))
        self.assertTrue(child.evicted)  # host-only: no device value
        self.assertTrue(child.backuped)


if __name__ == "__main__":
    unittest.main(verbosity=2)
