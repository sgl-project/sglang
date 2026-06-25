import unittest
from array import array

import torch

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
        self.cleared = []

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

    def clear_c128_radix_state(self, req_pool_idx: int):
        self.cleared.append(req_pool_idx)


class FakeReq:
    def __init__(self, req_pool_idx: int):
        self.req_pool_idx = req_pool_idx
        self.prefix_indices = []
        self.last_node = None
        self.kv_committed_len = 0


class TestDSV4C128RadixState(unittest.TestCase):
    def _make_cache(self):
        kv_pool = FakeDSV4KVPool()
        allocator = SWATokenToKVPoolAllocator(
            size=512,
            size_swa=512,
            page_size=256,
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
                page_size=256,
                sliding_window_size=512,
            )
        )
        return cache, kv_pool

    def _insert_tokens(self, cache: SWARadixCache, length: int):
        key = RadixKey(array("q", range(length)))
        value = torch.arange(length, dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=value))
        match = cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(len(match.device_indices), length // 256 * 256)
        return match.last_device_node

    def test_page_aligned_radix_hit_resets_c128_state(self):
        cache, kv_pool = self._make_cache()
        self._insert_tokens(cache, 260)

        req = FakeReq(req_pool_idx=2)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(256))), req=req)
        )

        self.assertEqual(len(match.device_indices), 256)

        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        cache.reset_c128_state_for_reqs([req])
        self.assertEqual(kv_pool.cleared, [2])

    def test_unaligned_match_uses_aligned_boundary_and_resets(self):
        cache, kv_pool = self._make_cache()
        self._insert_tokens(cache, 260)

        req = FakeReq(req_pool_idx=3)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(260))), req=req)
        )

        self.assertEqual(len(match.device_indices), 256)
        self.assertEqual(cache._node_prefix_len(match.last_device_node), 256)

        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        cache.reset_c128_state_for_reqs([req])
        self.assertEqual(kv_pool.cleared, [3])

    def test_chunked_prefill_live_tail_skips_reset(self):
        cache, kv_pool = self._make_cache()
        self._insert_tokens(cache, 260)

        req = FakeReq(req_pool_idx=3)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(256))), req=req)
        )

        req.prefix_indices = torch.cat(
            [match.device_indices, torch.arange(256, 288, dtype=torch.int64)]
        )
        req.kv_committed_len = len(req.prefix_indices)
        req.last_node = match.last_device_node
        cache.reset_c128_state_for_reqs([req])

        self.assertEqual(kv_pool.cleared, [])

    def test_non_live_tail_length_mismatch_asserts(self):
        cache, kv_pool = self._make_cache()
        self._insert_tokens(cache, 260)

        req = FakeReq(req_pool_idx=3)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(256))), req=req)
        )

        req.prefix_indices = torch.cat(
            [match.device_indices, torch.arange(256, 288, dtype=torch.int64)]
        )
        req.last_node = match.last_device_node
        with self.assertRaises(AssertionError):
            cache.reset_c128_state_for_reqs([req])
        self.assertEqual(kv_pool.cleared, [])


if __name__ == "__main__":
    unittest.main()
