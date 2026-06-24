import unittest
from array import array

import torch

from sglang.srt.mem_cache import deepseek_v4_memory_pool as dsv4_memory_pool
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_radix_cache import RadixKey, SWARadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


class FakeDSV4KVPool(BaseSWAKVPool):
    def __init__(self):
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self.restored = []
        self.cleared = []
        self.snapshotted = []

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError

    def get_kv_buffer(self, layer_id: int):
        raise NotImplementedError

    def set_kv_buffer(self, layer, loc, cache_k, cache_v) -> None:
        raise NotImplementedError

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        pass

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return kv_indices

    def get_state_buf_infos(self):
        return [], [], []

    def snapshot_c128_radix_state(self, req_pool_idx: int, seq_len: int):
        self.snapshotted.append((req_pool_idx, seq_len))
        return [torch.tensor([req_pool_idx, seq_len], dtype=torch.int64)]

    def restore_c128_radix_state(self, req_pool_idx: int, snapshots):
        self.restored.append((req_pool_idx, snapshots))

    def clear_c128_radix_state(self, req_pool_idx: int):
        self.cleared.append(req_pool_idx)


class FakeReq:
    def __init__(self, req_pool_idx: int):
        self.req_pool_idx = req_pool_idx
        self.prefix_indices = []
        self.last_node = None


class FakeKVScoreBuffer:
    def __init__(self, kv_score: torch.Tensor):
        self.kv_score = kv_score


class FakeCompressStatePool:
    ratio = 128

    def __init__(self, ring_size: int, kv_score: torch.Tensor):
        self.ring_size = ring_size
        self.kv_score_buffer = FakeKVScoreBuffer(kv_score)


class TestDSV4C128RadixState(unittest.TestCase):
    def _make_cache(self):
        kv_pool = FakeDSV4KVPool()
        allocator = SWATokenToKVPoolAllocator(
            size=512,
            size_swa=512,
            page_size=1,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        req_to_token_pool = ReqToTokenPool(
            size=4, max_context_len=512, device="cpu", enable_memory_saver=False
        )
        cache = SWARadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                sliding_window_size=512,
            )
        )
        return cache, kv_pool

    def _insert_tokens(self, cache: SWARadixCache, length: int):
        key = RadixKey(array("q", range(length)))
        value = torch.arange(length, dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=value))
        match = cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(len(match.device_indices), length)
        return match.last_device_node

    def test_aligned_c128_boundary_does_not_need_snapshot(self):
        cache, kv_pool = self._make_cache()
        self._insert_tokens(cache, 260)

        req = FakeReq(req_pool_idx=2)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(128))), req=req)
        )

        self.assertEqual(len(match.device_indices), 128)
        self.assertIsNone(match.last_device_node.c128_state_snapshots)

        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        cache.restore_c128_state_for_reqs([req])
        self.assertEqual(kv_pool.restored, [])
        self.assertEqual(kv_pool.cleared, [2])

    def test_partial_c128_snapshot_moves_to_split_node_and_restores(self):
        cache, kv_pool = self._make_cache()
        leaf = self._insert_tokens(cache, 260)

        snapshot = [torch.tensor([7, 130], dtype=torch.int64)]
        cache._store_c128_snapshot_at_prefix_len(leaf, 130, snapshot)

        req = FakeReq(req_pool_idx=3)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(130))), req=req)
        )

        self.assertEqual(len(match.device_indices), 130)
        self.assertIs(
            match.last_device_node.c128_state_snapshots[130],
            snapshot,
        )

        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        cache.restore_c128_state_for_reqs([req])

        self.assertEqual(len(kv_pool.restored), 1)
        restored_req_pool_idx, restored_snapshot = kv_pool.restored[0]
        self.assertEqual(restored_req_pool_idx, 3)
        self.assertIs(restored_snapshot, snapshot)

    def test_missing_partial_snapshot_falls_back_to_aligned_boundary(self):
        cache, kv_pool = self._make_cache()
        self._insert_tokens(cache, 260)

        req = FakeReq(req_pool_idx=3)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(130))), req=req)
        )

        self.assertEqual(len(match.device_indices), 128)
        self.assertEqual(cache._node_prefix_len(match.last_device_node), 128)

        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        cache.restore_c128_state_for_reqs([req])
        self.assertEqual(kv_pool.restored, [])
        self.assertEqual(kv_pool.cleared, [3])

    def test_chunked_prefill_live_tail_skips_restore(self):
        cache, kv_pool = self._make_cache()
        self._insert_tokens(cache, 260)

        req = FakeReq(req_pool_idx=3)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(128))), req=req)
        )

        req.prefix_indices = torch.cat(
            [match.device_indices, torch.arange(128, 160, dtype=torch.int64)]
        )
        req.last_node = match.last_device_node
        cache.restore_c128_state_for_reqs([req])

        self.assertEqual(kv_pool.restored, [])
        self.assertEqual(kv_pool.cleared, [])

    def test_store_partial_snapshot_uses_requested_prefix_length(self):
        cache, kv_pool = self._make_cache()
        leaf = self._insert_tokens(cache, 260)
        req = FakeReq(req_pool_idx=4)

        cache._store_c128_partial_snapshot_for_req(req, leaf, 130)

        self.assertEqual(kv_pool.snapshotted, [(4, 130)])
        self.assertIn(130, leaf.c128_state_snapshots)

    def test_store_aligned_boundary_skips_snapshot(self):
        cache, kv_pool = self._make_cache()
        leaf = self._insert_tokens(cache, 260)
        req = FakeReq(req_pool_idx=4)

        cache._store_c128_partial_snapshot_for_req(req, leaf, 128)

        self.assertEqual(kv_pool.snapshotted, [])
        self.assertIsNone(leaf.c128_state_snapshots)

    def test_offline_snapshot_only_copies_current_c128_block(self):
        kv_score = torch.arange(512 * 4, dtype=torch.float32).reshape(512, 4)
        kv_pool = dsv4_memory_pool.DeepSeekV4TokenToKVPool.__new__(
            dsv4_memory_pool.DeepSeekV4TokenToKVPool
        )
        kv_pool.compress_state_pools = [FakeCompressStatePool(256, kv_score)]

        old_online_c128 = dsv4_memory_pool.ONLINE_C128
        dsv4_memory_pool.ONLINE_C128 = False
        try:
            snapshot = kv_pool.snapshot_c128_radix_state(req_pool_idx=1, seq_len=130)[0]
        finally:
            dsv4_memory_pool.ONLINE_C128 = old_online_c128

        start = 256
        expected = kv_score.new_empty((256, 4))
        expected[:, :2].zero_()
        expected[:, 2:].fill_(float("-inf"))
        expected[128:130] = kv_score[start + 128 : start + 130]
        torch.testing.assert_close(snapshot, expected)


if __name__ == "__main__":
    unittest.main()
