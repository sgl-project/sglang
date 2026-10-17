"""Token allocator capacity, release ownership and reuse order."""

import unittest

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTokenAllocator(CustomTestCase):
    def _allocator(self, need_sort):
        return TokenToKVPoolAllocator(8, torch.float16, "cpu", None, need_sort)

    def test_releases_preserve_capacity_and_reuse_order(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = self._allocator(need_sort)
                owned = allocator.alloc(6)
                allocator.free(owned[4:6])
                allocator.free(owned[:2])
                self.assertEqual(allocator.available_size(), 6)
                self.assertCountEqual(
                    allocator.get_all_free_pages().tolist(), [1, 2, 5, 6, 7, 8]
                )
                self.assertEqual(allocator.alloc(2).tolist(), [7, 8])
                expected = [1, 2, 5, 6] if need_sort else [5, 6, 1, 2]
                self.assertEqual(allocator.alloc(4).tolist(), expected)
                self.assertEqual(allocator.available_size(), 0)

    def test_failed_allocation_leaves_released_slots_reusable(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = self._allocator(need_sort)
                owned = allocator.alloc(8)
                allocator.free(owned[4:6])
                self.assertIsNone(allocator.alloc(3))
                self.assertEqual(allocator.available_size(), 2)
                self.assertEqual(allocator.alloc(2).tolist(), [5, 6])

    def test_release_owns_a_snapshot_of_the_callers_view(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = self._allocator(need_sort)
                owned = allocator.alloc(8)
                allocator.free(owned[:2])
                owned[:2].fill_(0)
                self.assertEqual(allocator.alloc(2).tolist(), [1, 2])

    def test_grouped_releases_wait_for_group_end_and_own_their_views(self):
        allocator = self._allocator(False)
        owned = allocator.alloc(8)
        allocator.free_group_begin()
        allocator.free(owned[:2])
        owned[:2].fill_(0)
        self.assertEqual(allocator.available_size(), 0)
        self.assertIsNone(allocator.alloc(1))
        allocator.free_group_end()
        self.assertEqual(allocator.alloc(2).tolist(), [1, 2])

    def test_clear_discards_releases_and_allows_full_reallocation(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = self._allocator(need_sort)
                allocator.free(allocator.alloc(3))
                allocator.clear()
                self.assertEqual(allocator.available_size(), 8)
                self.assertEqual(allocator.alloc(8).tolist(), list(range(1, 9)))


if __name__ == "__main__":
    unittest.main()
