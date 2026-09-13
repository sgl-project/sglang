"""Unit tests for srt/mem_cache/hiradix_cache.py KV cache events."""

import os
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

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

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd")

PAGE_SIZE = 2


class TestHiRadixPrefetchAttribution(unittest.TestCase):
    def make_completed_prefetch(self, matched_length=128):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = SimpleNamespace(parent=None)
        parent = SimpleNamespace(parent=root, key=list(range(128)))
        anchor = SimpleNamespace(
            parent=parent, key=list(range(128)), release_host=MagicMock()
        )
        operation = SimpleNamespace(
            request_id="prefetch",
            completed_tokens=512,
            host_indices=list(range(512)),
            hash_value=["hash"] * 4,
        )
        cache.page_size = 128
        cache.ongoing_prefetch = {"prefetch": (anchor, list(range(512)), operation)}
        cache.cache_controller = MagicMock()
        cache.cache_controller.prefetch_tokens_occupied = 512
        cache.enable_storage_metrics = False
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_start_by_reqid = {}
        cache._clamp_prefetch_result = MagicMock(return_value=512)
        # Another request has already inserted the first 128 fetched tokens.
        cache._insert_helper_host = MagicMock(return_value=matched_length)
        cache._handle_prefetch_result(operation)
        return cache

    def test_storage_span_includes_concurrently_inserted_prefix(self):
        for matched in (0, 128, 512):
            with self.subTest(matched=matched):
                cache = self.make_completed_prefetch(matched)
                self.assertEqual(cache.pop_prefetch_loaded_span("prefetch"), (512, 256))
                self.assertEqual(cache.pop_prefetch_loaded_span("prefetch"), (0, None))

    def test_legacy_pop_cleans_up_span(self):
        cache = self.make_completed_prefetch()
        self.assertEqual(cache.pop_prefetch_loaded_tokens("prefetch"), 512)
        self.assertEqual(cache.prefetch_loaded_start_by_reqid, {})

    def test_abort_cleans_up_span(self):
        cache = self.make_completed_prefetch()
        cache.release_aborted_request("prefetch")
        self.assertEqual(cache.pop_prefetch_loaded_span("prefetch"), (0, None))


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
