"""
CPU unit tests for page, eviction, and unfinished-cache helpers in mem_cache/common.py.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    evict_from_tree_cache,
    kv_to_page_indices,
    kv_to_page_num,
    maybe_cache_unfinished_req,
    page_align_floor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKVToPageIndices(CustomTestCase):
    def test_maps_contiguous_rows_to_page_ids(self):
        kv_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        page_indices = kv_to_page_indices(kv_indices, page_size=4)
        np.testing.assert_array_equal(page_indices, np.array([0, 1]))

    def test_maps_offset_rows_to_page_ids(self):
        kv_indices = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15])
        page_indices = kv_to_page_indices(kv_indices, page_size=4)
        np.testing.assert_array_equal(page_indices, np.array([2, 3]))

    def test_page_size_one_keeps_row_indices(self):
        kv_indices = torch.tensor([0, 1, 2, 3])
        page_indices = kv_to_page_indices(kv_indices, page_size=1)
        np.testing.assert_array_equal(page_indices, np.array([0, 1, 2, 3]))


class TestKVToPageNum(CustomTestCase):
    def test_ceil_division(self):
        self.assertEqual(kv_to_page_num(17, 4), 5)
        self.assertEqual(kv_to_page_num(16, 4), 4)
        self.assertEqual(kv_to_page_num(1, 4), 1)
        self.assertEqual(kv_to_page_num(0, 4), 0)


class TestPageAlignFloor(CustomTestCase):
    def test_floor_align(self):
        self.assertEqual(page_align_floor(17, 4), 16)
        self.assertEqual(page_align_floor(16, 4), 16)
        self.assertEqual(page_align_floor(15, 4), 12)
        self.assertEqual(page_align_floor(3, 4), 0)
        self.assertEqual(page_align_floor(0, 4), 0)


class TestEvictFromTreeCache(CustomTestCase):
    def test_when_tree_cache_is_empty(self):
        # Should return without error.
        for n in (0, 1, 10, 100, 1000):
            result = evict_from_tree_cache(tree_cache=None, num_tokens=n)
            self.assertIsNone(result)

    def test_when_tree_cache_is_chunk_cache(self):
        cache = MagicMock()
        cache.is_chunk_cache.return_value = True
        for n in (0, 1, 10, 100, 1000):
            result = evict_from_tree_cache(tree_cache=cache, num_tokens=n)
            self.assertIsNone(result)
        cache.evict.assert_not_called()

    def test_swa_evicts_only_swa_deficit_when_full_has_enough(self):
        cache = MagicMock()
        cache.is_chunk_cache.return_value = False
        allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
        cache.token_to_kv_pool_allocator = allocator
        allocator.full_available_size.return_value = 10
        allocator.swa_available_size.return_value = 5
        evict_from_tree_cache(tree_cache=cache, num_tokens=6)
        cache.evict.assert_called_once()
        params = cache.evict.call_args[0]
        self.assertEqual(params[0].num_tokens, 0)
        self.assertEqual(params[0].swa_num_tokens, 1)

    def test_swa_evicts_both_deficits_when_both_pools_short(self):
        cache = MagicMock()
        cache.is_chunk_cache.return_value = False
        allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
        cache.token_to_kv_pool_allocator = allocator
        allocator.full_available_size.return_value = 10
        allocator.swa_available_size.return_value = 5
        evict_from_tree_cache(tree_cache=cache, num_tokens=20)
        cache.evict.assert_called_once()
        params = cache.evict.call_args[0]
        self.assertEqual(params[0].num_tokens, 10)
        self.assertEqual(params[0].swa_num_tokens, 15)

    def test_swa_noop_when_both_pools_have_enough(self):
        cache = MagicMock()
        cache.is_chunk_cache.return_value = False
        allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
        cache.token_to_kv_pool_allocator = allocator
        allocator.full_available_size.return_value = 10
        allocator.swa_available_size.return_value = 10
        evict_from_tree_cache(tree_cache=cache, num_tokens=5)
        cache.evict.assert_not_called()

    def test_standard_evicts_when_available_below_need(self):
        cache = MagicMock()
        cache.is_chunk_cache.return_value = False
        allocator = MagicMock()
        cache.token_to_kv_pool_allocator = allocator
        allocator.available_size.return_value = 5
        evict_from_tree_cache(cache, num_tokens=6)
        cache.evict.assert_called_once()
        params = cache.evict.call_args.args[0]
        self.assertEqual(params.num_tokens, 6)
        self.assertEqual(params.swa_num_tokens, 0)

    def test_standard_noop_when_available_has_enough(self):
        cache = MagicMock()
        cache.is_chunk_cache.return_value = False
        allocator = MagicMock()
        cache.token_to_kv_pool_allocator = allocator
        allocator.available_size.return_value = 10
        evict_from_tree_cache(cache, num_tokens=6)
        cache.evict.assert_not_called()


class TestMaybeCacheUnfinishedReq(CustomTestCase):
    def test_noop_when_skip_radix_cache_insert_exists(self):
        req = SimpleNamespace(skip_radix_cache_insert=True)
        tree_cache = MagicMock()
        maybe_cache_unfinished_req(req, tree_cache)
        tree_cache.cache_unfinished_req.assert_not_called()

    def test_caches_when_skip_flag_is_false(self):
        req = SimpleNamespace(skip_radix_cache_insert=False)
        tree_cache = MagicMock()
        maybe_cache_unfinished_req(req, tree_cache)
        tree_cache.cache_unfinished_req.assert_called_once()

    def test_caches_when_skip_flag_is_missing(self):
        req = SimpleNamespace()
        tree_cache = MagicMock()
        maybe_cache_unfinished_req(req, tree_cache)
        tree_cache.cache_unfinished_req.assert_called_once()


if __name__ == "__main__":
    unittest.main()
