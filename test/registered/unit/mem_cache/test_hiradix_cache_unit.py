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

    def _stored_gpu_events(self, cache):
        return [
            e
            for e in cache.take_events()
            if isinstance(e, BlockStored) and e.medium == StorageMedium.GPU
        ]

    def _backup_and_evict(self, cache, tokens):
        node = self._leaf_for(cache, tokens)
        backed_up = cache.write_backup(node, write_back=True)
        self.assertGreater(backed_up, 0)
        cache.writing_check(write_back=True)
        cache.take_events()
        self.assertGreater(cache._evict_backuped(node), 0)
        cache.take_events()
        self.assertTrue(node.evicted)
        self.assertTrue(node.backuped)
        return node

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

    def test_exact_recomputation_publishes_gpu_store(self):
        cache, allocator = self._build_cache()
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4])
        self._backup_and_evict(cache, [1, 2, 3, 4])

        self._insert(cache, allocator, [1, 2, 3, 4])
        stored_gpu = self._stored_gpu_events(cache)
        self.assertEqual(
            [list(e.token_ids) for e in stored_gpu],
            [[1, 2], [3, 4]],
        )
        self.assertIsNone(stored_gpu[0].parent_block_hash)
        self.assertEqual(stored_gpu[1].parent_block_hash, stored_gpu[0].block_hashes[0])

    def test_partial_recomputation_publishes_only_materialized_gpu_blocks(self):
        cache, allocator = self._build_cache()
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4])
        self._backup_and_evict(cache, [1, 2, 3, 4])

        self._insert(cache, allocator, [1, 2, 5, 6])
        stored_gpu = self._stored_gpu_events(cache)
        self.assertEqual(
            [list(e.token_ids) for e in stored_gpu],
            [[1, 2], [5, 6]],
        )
        self.assertNotIn([3, 4], [list(e.token_ids) for e in stored_gpu])
        self.assertIsNone(stored_gpu[0].parent_block_hash)
        self.assertEqual(stored_gpu[1].parent_block_hash, stored_gpu[0].block_hashes[0])

    def test_existing_device_node_does_not_publish_duplicate_gpu_store(self):
        cache, allocator = self._build_cache()
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4])
        cache.take_events()
        self._insert(cache, allocator, [1, 2, 3, 4])

        self.assertEqual(self._stored_gpu_events(cache), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
