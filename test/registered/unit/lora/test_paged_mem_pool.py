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
    pool.A_pages = {}
    pool.B_pages = {}
    pool.page_generation = 0
    pool.page_access_times = [0] * pool.total_pages
    pool._access_counter = 0
    pool.total_bytes_evicted = 0
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
        pool.A_pages = {}
        pool.B_pages = {}
        pool.page_access_times = [0] * pool.total_pages
        pool._access_counter = 0
        pool.total_bytes_evicted = 0
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

    def test_evict_updates_total_bytes(self):
        """evict_pages must update total_bytes_evicted."""
        self.pool.total_bytes_evicted = 100
        self.pool.allocate_pages("a", 16)  # 2 pages
        self.pool.mark_adapter_pages_accessed("a")
        self.pool.evict_pages(1, set())
        # _compute_page_bytes returns 0 on bare pool (no real A_pages/B_pages
        # tensors), but the counter should still be updated (0 bytes added).
        self.assertEqual(self.pool.total_bytes_evicted, 100)

    def test_evict_zero_pages_no_bytes_update(self):
        """Evicting 0 pages must not touch total_bytes_evicted."""
        self.pool.total_bytes_evicted = 50
        self.pool.evict_pages(0, set())
        self.assertEqual(self.pool.total_bytes_evicted, 50)


class TestCanEnsureAdapterReady(CustomTestCase):
    """Dry-run check: can_ensure_adapter_ready predicts without side effects."""

    def setUp(self):
        self.pool = _make_bare_pool(total_pages=16, page_rank_size=8)

    def test_already_complete(self):
        """Already-resident adapter returns True immediately."""
        self.pool.allocate_pages("a", 8)
        self.assertTrue(self.pool.can_ensure_adapter_ready("a", 8, set()))

    def test_enough_free_after_new_adapter(self):
        """New adapter (not in page_table) with enough free pages → True."""
        self.assertTrue(self.pool.can_ensure_adapter_ready("new_adapter", 8, set()))

    def test_not_enough_when_pool_full_and_protected(self):
        """All pages in-use and protected → cannot make room → False."""
        self.pool.allocate_pages("full", 128)  # 16 pages (rank 128 / 8)
        self.pool.mark_adapter_pages_accessed("full")
        protected = self.pool.get_protected_pages({"full"})
        # 0 free, 0 evictable (all protected) → False
        self.assertFalse(self.pool.can_ensure_adapter_ready("new", 8, protected))

    def test_evictable_counted(self):
        """Evictable (unprotected, unpinned) pages are counted toward budget."""
        # Allocate two adapters, make one evictable
        self.pool.allocate_pages("keep", 64)  # 8 pages
        self.pool.mark_adapter_pages_accessed("keep")
        self.pool.allocate_pages("evictable", 64)  # 8 pages
        self.pool.mark_adapter_pages_accessed("evictable")
        # 16 pages used, 0 free. A new r=8 needs 1 page.
        # "evictable" pages should count → True
        self.assertTrue(self.pool.can_ensure_adapter_ready("new", 8, set()))

    def test_protected_not_counted(self):
        """Protected pages are NOT counted as evictable."""
        self.pool.allocate_pages("keep", 64)  # 8 pages
        self.pool.mark_adapter_pages_accessed("keep")
        self.pool.allocate_pages("prot", 64)  # 8 pages
        self.pool.mark_adapter_pages_accessed("prot")
        # 16 pages used, 0 free. Protect "prot" and "keep" pages.
        protected = self.pool.get_protected_pages({"keep", "prot"})
        # No evictable pages → new r=8 should fail
        self.assertFalse(self.pool.can_ensure_adapter_ready("new", 8, protected))

    def test_pinned_not_counted(self):
        """Pinned pages are NOT counted as evictable."""
        self.pool.allocate_pages("keep", 64)  # 8 pages
        self.pool.mark_adapter_pages_accessed("keep")
        self.pool.allocate_pages("pin", 64)  # 8 pages
        self.pool.mark_adapter_pages_accessed("pin")
        self.pool.pinned_uids.add("pin")
        # 16 pages used, 0 free. "pin" pages are pinned, "keep" pages are
        # protected → nothing evictable → False
        protected = self.pool.get_protected_pages({"keep"})
        self.assertFalse(self.pool.can_ensure_adapter_ready("new", 8, protected))

    def test_partial_missing_counted(self):
        """Only missing pages are counted, not all pages of the adapter."""
        self.pool.allocate_pages("partial", 16)  # 2 pages
        self.pool.mark_adapter_pages_accessed("partial")
        # Evict 1 page
        self.pool.evict_pages(1, set())
        # 1 page missing, 15 free (16 total - 2 allocated + 1 freed = 15 free)
        # Should return True (15 free >= 1 needed)
        self.assertTrue(self.pool.can_ensure_adapter_ready("partial", 16, set()))

    def test_uid_none_always_true(self):
        """None uid returns True (base model path)."""
        self.assertTrue(self.pool.can_ensure_adapter_ready(None, 0, set()))

    def test_rank_zero_always_true(self):
        """rank <= 0 returns True."""
        self.assertTrue(self.pool.can_ensure_adapter_ready("noop", 0, set()))

    def test_no_side_effects(self):
        """can_ensure_adapter_ready must not mutate pool state."""
        self.pool.allocate_pages("a", 16)  # 2 pages
        self.pool.mark_adapter_pages_accessed("a")
        self.pool.evict_pages(1, set())  # 1 page free, 1 still allocated
        gen_before = self.pool.page_generation
        free_before = set(self.pool.free_page_indices)
        pt_before = dict(self.pool.page_table)
        phys_to_uid_before = dict(getattr(self.pool, "phys_page_to_uid", {}))

        self.pool.can_ensure_adapter_ready("a", 16, set())

        self.assertEqual(self.pool.page_generation, gen_before)
        self.assertEqual(self.pool.free_page_indices, free_before)
        self.assertEqual(self.pool.page_table, pt_before)
        self.assertEqual(getattr(self.pool, "phys_page_to_uid", {}), phys_to_uid_before)


if __name__ == "__main__":
    unittest.main()
