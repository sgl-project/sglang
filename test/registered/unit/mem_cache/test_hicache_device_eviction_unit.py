"""CPU unit tests for HiCache device eviction safeguards."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self, *, need_sort=True, debug_mode=False):
        self.need_sort = need_sort
        self.debug_mode = debug_mode
        self.is_not_in_free_group = True
        self.free_pages = torch.empty((0,), dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)
        self.free_group = []
        self.free = mock.Mock()


class TestHiRadixSyncFreeDeviceRelease(unittest.TestCase):
    def _make_cache(self, *, page_size=4, need_sort=True):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = page_size
        cache.cache_controller = SimpleNamespace(
            mem_pool_device_allocator=_FakeAllocator(need_sort=need_sort)
        )
        return cache

    def test_debug_mode_accepts_page_aligned_contiguous_run(self):
        cache = self._make_cache(page_size=4)
        allocator = cache.cache_controller.mem_pool_device_allocator
        allocator.debug_mode = True
        device_indices = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23])

        released = HiRadixCache._free_device_indices_sync_free(cache, device_indices)

        self.assertEqual(released, 8)
        self.assertTrue(torch.equal(allocator.release_pages, torch.tensor([2, 5])))

    def test_debug_mode_rejects_non_contiguous_run(self):
        cache = self._make_cache(page_size=4)
        allocator = cache.cache_controller.mem_pool_device_allocator
        allocator.debug_mode = True
        device_indices = torch.tensor([8, 9, 10, 99, 20, 21, 22, 23])

        with self.assertRaises(AssertionError):
            HiRadixCache._free_device_indices_sync_free(cache, device_indices)

    def test_debug_mode_rejects_non_page_aligned_run(self):
        cache = self._make_cache(page_size=4)
        allocator = cache.cache_controller.mem_pool_device_allocator
        allocator.debug_mode = True
        device_indices = torch.tensor([1, 2, 3, 4, 20, 21, 22, 23])

        with self.assertRaises(AssertionError):
            HiRadixCache._free_device_indices_sync_free(cache, device_indices)

    def test_sync_free_release_uses_release_pages_when_sorting(self):
        cache = self._make_cache(page_size=4, need_sort=True)
        device_indices = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23])

        released = HiRadixCache._free_device_indices_sync_free(cache, device_indices)

        allocator = cache.cache_controller.mem_pool_device_allocator
        self.assertEqual(released, 8)
        self.assertTrue(torch.equal(allocator.release_pages, torch.tensor([2, 5])))
        self.assertEqual(allocator.free_pages.numel(), 0)
        allocator.free.assert_not_called()

    def test_sync_free_release_uses_free_pages_without_sorting(self):
        cache = self._make_cache(page_size=4, need_sort=False)
        device_indices = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23])

        released = HiRadixCache._free_device_indices_sync_free(cache, device_indices)

        allocator = cache.cache_controller.mem_pool_device_allocator
        self.assertEqual(released, 8)
        self.assertTrue(torch.equal(allocator.free_pages, torch.tensor([2, 5])))
        self.assertEqual(allocator.release_pages.numel(), 0)
        allocator.free.assert_not_called()

    def test_sync_free_release_preserves_open_free_group(self):
        cache = self._make_cache(page_size=4)
        allocator = cache.cache_controller.mem_pool_device_allocator
        allocator.is_not_in_free_group = False
        device_indices = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23])

        released = HiRadixCache._free_device_indices_sync_free(cache, device_indices)

        self.assertEqual(released, 8)
        self.assertEqual(len(allocator.free_group), 1)
        self.assertIs(allocator.free_group[0], device_indices)
        self.assertEqual(allocator.release_pages.numel(), 0)
        allocator.free.assert_not_called()

    def test_non_page_multiple_falls_back_to_allocator_free(self):
        cache = self._make_cache(page_size=4)
        device_indices = torch.tensor([8, 9, 10, 11, 20, 21])

        released = HiRadixCache._free_device_indices_sync_free(cache, device_indices)

        allocator = cache.cache_controller.mem_pool_device_allocator
        self.assertEqual(released, 6)
        allocator.free.assert_called_once()
        self.assertIs(allocator.free.call_args.args[0], device_indices)
        self.assertEqual(allocator.release_pages.numel(), 0)

    def test_page_size_one_falls_back_to_allocator_free(self):
        cache = self._make_cache(page_size=1)
        device_indices = torch.tensor([8, 9, 10])

        released = HiRadixCache._free_device_indices_sync_free(cache, device_indices)

        allocator = cache.cache_controller.mem_pool_device_allocator
        self.assertEqual(released, 3)
        allocator.free.assert_called_once()
        self.assertIs(allocator.free.call_args.args[0], device_indices)
        self.assertEqual(allocator.release_pages.numel(), 0)


class TestHiRadixPendingWriteThroughEviction(unittest.TestCase):
    def _make_cache(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 4
        cache.ongoing_write_through = {}
        cache._detach_backuped = mock.Mock(return_value=8)
        cache.cache_controller = SimpleNamespace(
            mem_pool_device_allocator=_FakeAllocator(need_sort=True)
        )
        return cache

    def test_pending_write_through_node_is_not_device_evicted(self):
        cache = self._make_cache()
        node = SimpleNamespace(
            id=42,
            value=torch.tensor([8, 9, 10, 11, 20, 21, 22, 23]),
            write_through_pending_id=42,
        )
        cache.ongoing_write_through[42] = (node, 8, [node])

        evicted = HiRadixCache._evict_backuped(cache, node)

        allocator = cache.cache_controller.mem_pool_device_allocator
        self.assertEqual(evicted, 0)
        cache._detach_backuped.assert_not_called()
        allocator.free.assert_not_called()
        self.assertEqual(allocator.release_pages.numel(), 0)
        self.assertIsNotNone(node.value)

    def test_non_pending_backuped_node_is_device_evicted_sync_free(self):
        cache = self._make_cache()
        node = SimpleNamespace(
            id=42,
            value=torch.tensor([8, 9, 10, 11, 20, 21, 22, 23]),
            write_through_pending_id=None,
        )

        evicted = HiRadixCache._evict_backuped(cache, node)

        allocator = cache.cache_controller.mem_pool_device_allocator
        self.assertEqual(evicted, 8)
        cache._detach_backuped.assert_called_once_with(node)
        self.assertTrue(torch.equal(allocator.release_pages, torch.tensor([2, 5])))
        allocator.free.assert_not_called()


class TestHiRadixRegularEvictionPendingGuard(unittest.TestCase):
    def _make_cache(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 4
        cache.ongoing_write_through = {}
        cache._record_remove_event = mock.Mock()
        cache._delete_leaf = mock.Mock()
        cache.cache_controller = SimpleNamespace(
            mem_pool_device_allocator=_FakeAllocator(need_sort=True)
        )
        return cache

    def test_pending_write_through_node_skips_regular_eviction(self):
        cache = self._make_cache()
        node = SimpleNamespace(
            id=7,
            value=torch.tensor([8, 9, 10, 11, 20, 21, 22, 23]),
            children={},
            write_through_pending_id=7,
        )
        cache.ongoing_write_through[7] = (node, 8, [node])

        evicted = HiRadixCache._evict_regular(cache, node)

        allocator = cache.cache_controller.mem_pool_device_allocator
        self.assertEqual(evicted, 0)
        cache._record_remove_event.assert_not_called()
        cache._delete_leaf.assert_not_called()
        allocator.free.assert_not_called()
        self.assertEqual(allocator.release_pages.numel(), 0)
        self.assertIsNotNone(node.value)

    def test_non_pending_node_is_regular_evicted_sync_free(self):
        cache = self._make_cache()
        node = SimpleNamespace(
            id=7,
            value=torch.tensor([8, 9, 10, 11, 20, 21, 22, 23]),
            children={},
            write_through_pending_id=None,
        )

        evicted = HiRadixCache._evict_regular(cache, node)

        allocator = cache.cache_controller.mem_pool_device_allocator
        self.assertEqual(evicted, 8)
        cache._record_remove_event.assert_called_once_with(node)
        cache._delete_leaf.assert_called_once_with(node)
        self.assertTrue(torch.equal(allocator.release_pages, torch.tensor([2, 5])))
        allocator.free.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
