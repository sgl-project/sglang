"""Unit tests for LoRAPagePool — page allocation, eviction, build_page_table_tensor.

Covers the page lifecycle (allocate/free/evict/page_in), the page_generation
counter, and the build_page_table_tensor method that constructs the dense
page-table tensor for kernel usage.

Tests are hermetic: CPU-only, no CUDA, no model loading. LoRAPagePool is
instantiated via __new__ with only the fields the tested methods read.

Usage:
    python -m pytest test/registered/unit/lora/test_paged_mem_pool.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.lora.paged_mem_pool import LoRAPagePool
from sglang.test.test_utils import CustomTestCase


def _make_bare_pool(total_pages=8, page_rank_size=8, device="cpu"):
    """Construct a minimal LoRAPagePool for unit tests.

    Bypasses __init__ (which requires base model, HF config, GPU allocations)
    and sets only the fields consulted by the methods under test.
    """
    pool = LoRAPagePool.__new__(LoRAPagePool)
    pool.total_pages = total_pages
    pool.PAGE_RANK_SIZE = page_rank_size
    pool.device = device
    pool.free_page_indices = set(range(total_pages))
    pool.phys_page_to_uid = {}
    pool.page_table = {}
    pool.adapter_ranks = {}
    pool.page_generation = 0
    pool.page_access_times = [0.0] * pool.total_pages
    pool.eviction_events = 0
    pool.bytes_paged_in = 0
    pool.pinned_uids = set()
    return pool


class TestBuildPageTableTensor(CustomTestCase):
    """build_page_table_tensor: dense int32 page-table for kernel usage."""

    def setUp(self):
        self.pool = _make_bare_pool(total_pages=8, page_rank_size=8)

    def test_basic_tensor(self):
        self.pool.allocate_pages("a", 8)
        a_page = self.pool.page_table["a"][0]
        tensor = self.pool.build_page_table_tensor(["a", None], 2)
        self.assertEqual(tensor.shape, (2, 2))
        self.assertEqual(tensor.dtype, torch.int32)
        self.assertEqual(tensor[0, 0].item(), a_page)
        self.assertEqual(tensor[0, 1].item(), -1)
        self.assertEqual(tensor[1, 0].item(), -1)

    def test_none_uid_row(self):
        self.pool.allocate_pages("a", 8)
        tensor = self.pool.build_page_table_tensor(["a", None, None], 2)
        self.assertEqual(tensor.shape, (3, 2))
        for row in (1, 2):
            for col in range(2):
                self.assertEqual(tensor[row, col].item(), -1)

    def test_evicted_page_shows_neg1(self):
        self.pool.allocate_pages("a", 16)
        self.pool.mark_adapter_pages_accessed("a")
        self.pool.evict_pages(1, set())
        tensor = self.pool.build_page_table_tensor(["a"], 2)
        count_neg = (tensor == -1).sum().item()
        self.assertEqual(count_neg, 1)

    def test_max_pages_truncation(self):
        self.pool = _make_bare_pool(total_pages=16, page_rank_size=8)
        self.pool.allocate_pages("a", 64)
        tensor = self.pool.build_page_table_tensor(["a"], 2)
        self.assertEqual(tensor.shape, (1, 2))
        self.assertEqual(tensor[0, 0].item(), self.pool.page_table["a"][0])
        self.assertEqual(tensor[0, 1].item(), self.pool.page_table["a"][1])

    def test_empty_uids(self):
        tensor = self.pool.build_page_table_tensor([], 2)
        self.assertEqual(tensor.shape, (0, 2))

    def test_output_on_correct_device(self):
        self.pool.allocate_pages("a", 8)
        tensor = self.pool.build_page_table_tensor(["a"], 1)
        self.assertEqual(tensor.device.type, "cpu")


class TestPageGeneration(CustomTestCase):
    """page_generation: increments on every page modification."""

    def setUp(self):
        self.pool = _make_bare_pool(total_pages=16, page_rank_size=8)

    def test_allocate_increments(self):
        gen0 = self.pool.page_generation
        self.pool.allocate_pages("a", 8)
        self.assertEqual(self.pool.page_generation, gen0 + 1)

    def test_free_increments(self):
        self.pool.allocate_pages("a", 8)
        gen0 = self.pool.page_generation
        self.pool.free_pages("a")
        self.assertEqual(self.pool.page_generation, gen0 + 1)

    def test_evict_increments(self):
        self.pool.allocate_pages("a", 16)
        self.pool.mark_adapter_pages_accessed("a")
        gen0 = self.pool.page_generation
        self.pool.evict_pages(1, set())
        self.assertEqual(self.pool.page_generation, gen0 + 1)

    def test_page_in_increments(self):
        self.pool.allocate_pages("a", 16)
        self.pool.mark_adapter_pages_accessed("a")
        self.pool.evict_pages(1, set())
        gen0 = self.pool.page_generation
        self.pool.page_in("a", 0)
        self.assertEqual(self.pool.page_generation, gen0 + 1)

    def test_readonly_no_increment(self):
        self.pool.allocate_pages("a", 8)
        gen0 = self.pool.page_generation
        self.pool.get_protected_pages({"a"})
        self.pool.max_pages_per_lora_for_batch(["a", None])
        self.pool.is_complete("a", 8)
        self.assertEqual(self.pool.page_generation, gen0)

    def test_getattr_fallback(self):
        pool = LoRAPagePool.__new__(LoRAPagePool)
        pool.total_pages = 8
        pool.PAGE_RANK_SIZE = 8
        pool.device = "cpu"
        pool.free_page_indices = set(range(8))
        pool.phys_page_to_uid = {}
        pool.page_table = {}
        pool.adapter_ranks = {}
        pool.page_access_times = [0.0] * pool.total_pages
        pool.eviction_events = 0
        pool.bytes_paged_in = 0
        pool.pinned_uids = set()
        self.assertFalse(hasattr(pool, "page_generation"))
        pool.allocate_pages("a", 8)
        self.assertEqual(pool.page_generation, 1)


class TestPageLifecycle(CustomTestCase):
    """Page allocation, eviction, and reload lifecycle."""

    def setUp(self):
        self.pool = _make_bare_pool(total_pages=16, page_rank_size=8)

    def test_allocate_then_free(self):
        self.pool.allocate_pages("a", 8)
        self.assertIn("a", self.pool.page_table)
        self.assertEqual(len(self.pool.page_table["a"]), 1)
        self.assertNotIn(self.pool.page_table["a"][0], self.pool.free_page_indices)
        self.pool.free_pages("a")
        self.assertNotIn("a", self.pool.page_table)
        self.assertEqual(len(self.pool.free_page_indices), 16)

    def test_evict_then_page_in(self):
        self.pool.allocate_pages("a", 16)
        self.pool.mark_adapter_pages_accessed("a")
        phys0 = self.pool.page_table["a"][0]
        self.pool.evict_pages(1, set())
        self.assertEqual(self.pool.page_table["a"][0], -1)
        self.assertIn(phys0, self.pool.free_page_indices)
        phys1 = self.pool.page_in("a", 0)
        self.assertNotEqual(phys1, phys0)
        self.assertNotIn(phys1, self.pool.free_page_indices)
        self.assertEqual(self.pool.page_table["a"][0], phys1)

    def test_allocate_insufficient_pages(self):
        pool = _make_bare_pool(total_pages=1, page_rank_size=8)
        result = pool.allocate_pages("a", 16)
        self.assertFalse(result)
        self.assertNotIn("a", pool.page_table)

    def test_is_complete(self):
        self.pool.allocate_pages("a", 16)
        self.pool.mark_adapter_pages_accessed("a")
        self.assertTrue(self.pool.is_complete("a", 16))
        self.pool.evict_pages(1, set())
        self.assertFalse(self.pool.is_complete("a", 16))

    def test_get_missing_pages(self):
        self.pool.allocate_pages("a", 16)
        self.pool.mark_adapter_pages_accessed("a")
        self.pool.evict_pages(1, set())
        missing = self.pool.get_missing_pages("a", 16)
        self.assertEqual(len(missing), 1)


if __name__ == "__main__":
    unittest.main()
