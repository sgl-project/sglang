import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_DEV = "cpu"
_PAGE_SIZE = 4
_SIZE = 64


def _make_allocator(
    *,
    page_size: int = _PAGE_SIZE,
    need_sort: bool = False,
    debug_mode: bool = False,
) -> PagedTokenToKVPoolAllocator:
    allocator = PagedTokenToKVPoolAllocator(
        _SIZE,
        page_size,
        torch.float16,
        _DEV,
        MagicMock(),
        need_sort,
    )
    allocator.debug_mode = debug_mode
    return allocator


def _page_slots(page_id: int, page_size: int = _PAGE_SIZE) -> torch.Tensor:
    return torch.arange(
        page_id * page_size, (page_id + 1) * page_size, dtype=torch.int64, device=_DEV
    )


class TestPagedAllocatorFree(CustomTestCase):
    def test_free_folds_pages_in_caller_block_order(self):
        """Whole-page free reports one page id per block, in the caller's block order."""
        allocator = _make_allocator()
        allocator.alloc(2 * _PAGE_SIZE)

        allocator.free(torch.cat([_page_slots(2), _page_slots(1)]))

        self.assertEqual(allocator.free_pages[:2].tolist(), [2, 1])

    def test_free_with_need_sort_routes_pages_to_release_pages(self):
        """need_sort sends freed pages to release_pages, and merge_and_sort_free folds them back."""
        allocator = _make_allocator(need_sort=True)
        allocator.alloc(2 * _PAGE_SIZE)

        allocator.free(torch.cat([_page_slots(2), _page_slots(1)]))

        self.assertEqual(sorted(allocator.release_pages.tolist()), [1, 2])
        allocator.merge_and_sort_free()
        self.assertEqual(allocator.release_pages.numel(), 0)
        self.assertEqual(allocator.free_pages.tolist(), list(range(1, 17)))

    def test_free_rejects_partial_page_before_touching_any_state(self):
        """A sub-page length is refused, and no free list or free group is mutated."""
        allocator = _make_allocator()
        allocator.alloc(2 * _PAGE_SIZE)
        free_pages_before = allocator.free_pages.clone()

        with self.assertRaises(AssertionError):
            allocator.free(_page_slots(1)[:3])

        self.assertEqual(allocator.free_pages.tolist(), free_pages_before.tolist())
        self.assertEqual(allocator.release_pages.numel(), 0)
        self.assertEqual(allocator.free_group, [])

    def test_free_empty_tensor_is_noop(self):
        """An empty free is accepted and changes nothing."""
        allocator = _make_allocator()
        allocator.alloc(2 * _PAGE_SIZE)
        free_pages_before = allocator.free_pages.clone()

        allocator.free(torch.empty((0,), dtype=torch.int64, device=_DEV))

        self.assertEqual(allocator.free_pages.tolist(), free_pages_before.tolist())

    def test_page_size_one_free_folds_every_slot_to_its_own_page(self):
        """With page_size 1 every slot is its own page, so free is the identity fold."""
        allocator = _make_allocator(page_size=1)
        allocator.alloc(3)

        allocator.free(torch.tensor([3, 1, 2], dtype=torch.int64, device=_DEV))

        self.assertEqual(allocator.free_pages[:3].tolist(), [3, 1, 2])


class TestPagedAllocatorFreeGroup(CustomTestCase):
    def test_free_group_rejects_partial_fragment_before_appending(self):
        """A sub-page fragment is refused when offered, not when the group is concatenated."""
        allocator = _make_allocator()
        allocator.alloc(2 * _PAGE_SIZE)
        allocator.free_group_begin()

        with self.assertRaises(AssertionError):
            allocator.free(_page_slots(1)[:3])

        self.assertEqual(allocator.free_group, [])

    def test_free_group_end_frees_whole_page_fragments(self):
        """Individually whole-page fragments concatenate and free their pages at group end."""
        allocator = _make_allocator()
        allocator.alloc(2 * _PAGE_SIZE)
        allocator.free_group_begin()

        allocator.free(_page_slots(1))
        allocator.free(_page_slots(2))
        allocator.free_group_end()

        self.assertEqual(sorted(allocator.free_pages.tolist()), list(range(1, 17)))


class TestPagedAllocatorDebugChecks(CustomTestCase):
    def test_a_block_spanning_two_pages_is_rejected(self):
        """The unconditional block-homogeneity assert catches a page-crossing block the length assert cannot see."""
        allocator = _make_allocator()
        allocator.alloc(2 * _PAGE_SIZE)

        misaligned = torch.arange(7, 11, dtype=torch.int64, device=_DEV)

        self.assertEqual(misaligned.numel() % _PAGE_SIZE, 0)
        with self.assertRaises(RuntimeError):
            allocator.free(misaligned)

    def test_debug_mode_rejects_the_same_page_freed_twice_in_one_call(self):
        """Debug mode catches a repeated page, which strided folding would double-free."""
        allocator = _make_allocator(debug_mode=True)
        allocator.alloc(2 * _PAGE_SIZE)

        with self.assertRaises(AssertionError):
            allocator.free(torch.cat([_page_slots(1), _page_slots(1)]))


class TestPagedAllocatorLegacyFree(CustomTestCase):
    def test_legacy_free_reclaims_pages_from_arbitrary_members(self):
        """The legacy entry inflates an arbitrary subset of page members to whole pages."""
        allocator = _make_allocator()
        allocator.alloc(2 * _PAGE_SIZE)

        allocator.free_pages_by_any_member_legacy(
            torch.tensor([5, 9], dtype=torch.int64, device=_DEV)
        )

        self.assertEqual(sorted(allocator.free_pages.tolist()), list(range(1, 17)))

    def test_legacy_free_reclaims_pages_from_page_head_singletons(self):
        """The legacy entry accepts one slot per page, the shape HiSparse's surplus release uses."""
        allocator = _make_allocator()
        allocator.alloc(2 * _PAGE_SIZE)

        pure_surplus = torch.tensor([1, 2], dtype=torch.int64, device=_DEV)
        allocator.free_pages_by_any_member_legacy(pure_surplus * _PAGE_SIZE)

        self.assertEqual(sorted(allocator.free_pages.tolist()), list(range(1, 17)))

    def test_legacy_free_matches_free_on_whole_page_input(self):
        """Both entries share _free_raw, so a whole-page input reclaims the same pages."""
        strided_allocator = _make_allocator()
        legacy_allocator = _make_allocator()
        strided_allocator.alloc(2 * _PAGE_SIZE)
        legacy_allocator.alloc(2 * _PAGE_SIZE)
        whole_pages = torch.cat([_page_slots(1), _page_slots(2)])

        strided_allocator.free(whole_pages)
        legacy_allocator.free_pages_by_any_member_legacy(whole_pages)

        self.assertEqual(
            sorted(strided_allocator.free_pages.tolist()),
            sorted(legacy_allocator.free_pages.tolist()),
        )


if __name__ == "__main__":
    unittest.main()
