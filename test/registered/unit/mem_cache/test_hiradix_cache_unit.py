"""Unit tests for srt/mem_cache/hiradix_cache.py KV cache events."""

import os
import unittest
from array import array

import torch

from sglang.srt.disaggregation.kv_events import BlockStored, StorageMedium
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
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

        # Both split fragments must be published as one parent-linked batch.
        stored_cpu = self._stored_cpu_events(cache)
        self.assertEqual(len(stored_cpu), 1)
        self.assertEqual(list(stored_cpu[0].token_ids), [1, 2, 3, 4])
        self.assertIsNone(stored_cpu[0].parent_block_hash)
        self.assertEqual(len(stored_cpu[0].block_hashes), 2)

    def test_proactive_write_back_preserves_kv_on_gpu_round_trip(self):
        cache, allocator = self._build_cache()
        cache.cache_controller.write_policy = "write_back"
        cache.write_back_threshold = 0.7
        nodes = []
        for branch in range(6):
            tokens = list(range(1000 * branch, 1000 * branch + 32))
            self._insert(cache, allocator, tokens)
            node = self._leaf_for(cache, tokens)
            nodes.append(node)
            for layer in range(4):
                cache.kv_cache.get_key_buffer(layer)[node.value] = branch + layer + 1
                cache.kv_cache.get_value_buffer(layer)[node.value] = branch + layer + 10

        self.assertEqual(allocator.available_size(), 64)
        cache._write_back_proactively(cache._write_back_budget())
        self.assertTrue(cache.ongoing_write_through)
        self.assertEqual(allocator.available_size(), 64)
        self.assertTrue(all(not node.evicted for node in nodes))
        torch.cuda.synchronize()
        cache.check_hicache_events()
        self.assertTrue(all(not node.evicted for node in nodes))
        self.assertEqual(allocator.available_size(), 64)
        cache.evict(EvictParams(num_tokens=32))
        evicted = [(i, node) for i, node in enumerate(nodes) if node.evicted]
        self.assertTrue(evicted)
        self.assertGreater(allocator.available_size(), 64)

        branch, node = evicted[0]
        restored = cache.load_back(node)
        self.assertIsNotNone(restored)
        cache.ready_to_load_host_cache()
        torch.cuda.synchronize()
        cache.loading_check()
        for layer in range(4):
            actual_k = cache.kv_cache.get_key_buffer(layer)[restored]
            actual_v = cache.kv_cache.get_value_buffer(layer)[restored]
            torch.testing.assert_close(
                actual_k, torch.full_like(actual_k, branch + layer + 1)
            )
            torch.testing.assert_close(
                actual_v, torch.full_like(actual_v, branch + layer + 10)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
