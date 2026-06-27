"""Unit tests for chunk_cache.py — component-level, no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede srt imports that may pull in sgl_kernel

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_chunk_cache(req_to_token=None):
    """Build a ChunkCache with mocked pools (no GPU, no server)."""
    allocator = MagicMock()
    allocator.device = torch.device("cpu")

    req_to_token_pool = MagicMock()
    if req_to_token is not None:
        req_to_token_pool.req_to_token = req_to_token

    params = CacheInitParams(
        disable=True,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
    )
    return ChunkCache(params)


class TestChunkCacheBasics(CustomTestCase):
    def test_is_chunk_cache_true(self):
        self.assertTrue(_make_chunk_cache().is_chunk_cache())

    def test_disable_property_true(self):
        # ChunkCache has no prefix matching, so `disable` is always True.
        self.assertTrue(_make_chunk_cache().disable)

    def test_device_taken_from_allocator(self):
        self.assertEqual(_make_chunk_cache().device, torch.device("cpu"))

    def test_device_defaults_to_cpu_without_allocator(self):
        params = CacheInitParams(
            disable=True,
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=None,
            page_size=1,
        )
        self.assertEqual(ChunkCache(params).device, torch.device("cpu"))

    def test_reset_is_noop(self):
        self.assertIsNone(_make_chunk_cache().reset())

    def test_protected_size_is_zero(self):
        self.assertEqual(_make_chunk_cache().protected_size(), 0)

    def test_pretty_print_is_empty(self):
        self.assertEqual(_make_chunk_cache().pretty_print(), "")


class TestChunkCacheNoPrefixMatching(CustomTestCase):
    def test_match_prefix_returns_miss(self):
        result = _make_chunk_cache().match_prefix(MagicMock())
        self.assertEqual(result.device_indices.numel(), 0)
        self.assertEqual(result.device_indices.dtype, torch.int64)
        self.assertIsNone(result.last_device_node)
        self.assertIsNone(result.last_host_node)
        self.assertIsNone(result.best_match_node)

    def test_insert_is_noop(self):
        self.assertEqual(_make_chunk_cache().insert(MagicMock()).prefix_len, 0)

    def test_evict_is_noop(self):
        self.assertEqual(_make_chunk_cache().evict(MagicMock()).num_tokens_evicted, 0)

    def test_inc_lock_ref_zero_delta(self):
        self.assertEqual(_make_chunk_cache().inc_lock_ref(MagicMock()).delta, 0)

    def test_dec_lock_ref_zero_delta(self):
        self.assertEqual(_make_chunk_cache().dec_lock_ref(MagicMock()).delta, 0)


class TestChunkCacheRequestLifecycle(CustomTestCase):
    def test_cache_finished_req_frees_committed_kv(self):
        # req_to_token[req_pool_idx, :committed_len] are the slots to free.
        req_to_token = torch.arange(20, dtype=torch.int64).reshape(4, 5)
        cache = _make_chunk_cache(req_to_token)

        req = MagicMock()
        req.pop_committed_kv_cache.return_value = 3
        req.req_pool_idx = 1

        cache.cache_finished_req(req)

        req.pop_committed_kv_cache.assert_called_once()
        free = cache.token_to_kv_pool_allocator.free
        free.assert_called_once()
        self.assertTrue(torch.equal(free.call_args.args[0], torch.tensor([5, 6, 7])))

    def test_cache_unfinished_req_sets_prefix_indices_as_copy(self):
        req_to_token = torch.arange(20, dtype=torch.int64).reshape(4, 5)
        cache = _make_chunk_cache(req_to_token)

        req = MagicMock()
        req.req_pool_idx = 2
        req.extend_range.end = 4

        cache.cache_unfinished_req(req)

        self.assertEqual(req.prefix_indices.dtype, torch.int64)
        self.assertTrue(torch.equal(req.prefix_indices, torch.tensor([10, 11, 12, 13])))

        # It must be a copy: mutating the source pool must not change prefix_indices.
        req_to_token[2, 0] = 999
        self.assertEqual(int(req.prefix_indices[0]), 10)


if __name__ == "__main__":
    unittest.main()
