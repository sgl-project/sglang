"""Unit tests for srt/mem_cache/hiradix_cache.py KV cache events."""

import os
import unittest
from array import array
from types import SimpleNamespace

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
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

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

    def _tag(self, cache, tokens, session_id):
        leaf = self._leaf_for(cache, tokens)
        cache._tag_session_leaf(
            SimpleNamespace(session_id=session_id),
            RadixKey(array("q", tokens)),
            node=leaf,
        )
        return leaf

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

    def test_session_survives_hicache_eviction_and_release(self):
        cache, allocator = self._build_cache()
        first_turn = list(range(12))
        second_turn = list(range(24))
        result = self._insert(cache, allocator, first_turn)
        self.assertIs(result.last_device_node, self._leaf_for(cache, first_turn))
        self._tag(cache, first_turn, "S")
        self._insert(cache, allocator, second_turn)
        leaf = self._tag(cache, second_turn, "S")

        cache.cache_controller.write_policy = "write_back"
        self.assertEqual(
            cache.evict(EvictParams(num_tokens=len(second_turn))).num_tokens_evicted,
            len(second_turn),
        )
        self.assertTrue(leaf.evicted)
        self.assertTrue(leaf.backuped)

        device_indices = cache.load_back(leaf)
        self.assertEqual(len(device_indices), len(second_turn))
        consumer = cache.ready_to_load_host_cache()
        cache.cache_controller.layer_done_counter.events[
            consumer
        ].finish_event.synchronize()
        cache.loading_check()
        self.assertFalse(leaf.evicted)
        self.assertEqual(cache.release_session("S"), 2)
        self.assertNotIn("S", cache._session_leaves)

    def test_session_shared_leaf_kept_until_last_holder(self):
        cache, allocator = self._build_cache()
        tokens = list(range(12))
        self._insert(cache, allocator, tokens)
        leaf = self._tag(cache, tokens, "A")
        self._tag(cache, tokens, "B")

        cache.cache_controller.write_policy = "write_back"
        cache.evict(EvictParams(num_tokens=len(tokens)))
        self.assertEqual(cache.release_session("A"), 0)
        self.assertEqual(leaf.session_ids, {"B"})
        self.assertEqual(cache.release_session("B"), 1)
        self.assertNotIn(leaf, cache.evictable_host_leaves)

    def test_locked_session_release_retries_after_unlock(self):
        cache, allocator = self._build_cache()
        tokens = list(range(12))
        self._insert(cache, allocator, tokens)
        leaf = self._tag(cache, tokens, "S")
        cache.inc_lock_ref(leaf)

        self.assertEqual(cache.release_session("S"), 0)
        self.assertIn("S", cache._pending_session_releases)
        cache.dec_lock_ref(leaf)
        cache.check_hicache_events()
        self.assertNotIn("S", cache._pending_session_releases)
        self.assertNotIn("S", cache._session_leaves)

    def test_reopen_cancels_pending_release(self):
        cache, allocator = self._build_cache()
        tokens = list(range(12))
        self._insert(cache, allocator, tokens)
        leaf = self._tag(cache, tokens, "S")
        cache.inc_lock_ref(leaf)
        cache.release_session("S")

        cache.register_session("S")
        cache.dec_lock_ref(leaf)
        cache.check_hicache_events()
        self.assertNotIn("S", cache._pending_session_releases)
        self.assertIn(leaf, cache._session_leaves["S"])

    def test_host_eviction_discards_session_index(self):
        cache, allocator = self._build_cache()
        tokens = list(range(12))
        self._insert(cache, allocator, tokens)
        leaf = self._tag(cache, tokens, "S")
        cache.cache_controller.write_policy = "write_back"
        cache.evict(EvictParams(num_tokens=len(tokens)))

        cache.evict_host(len(tokens))
        self.assertNotIn("S", cache._session_leaves)
        self.assertFalse(hasattr(leaf, "session_ids"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
