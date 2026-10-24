"""need_sort allocators stage released pages and merge only under allocation pressure."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

PAGE_SIZE = 2
NUM_PAGES = 16


def _make_allocator(*, need_sort: bool) -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _page_ids(indices: torch.Tensor) -> torch.Tensor:
    return indices[::PAGE_SIZE] // PAGE_SIZE


class TestPagedAllocatorLazyRelease(CustomTestCase):
    def test_generations_are_opt_in_for_external_linkers(self):
        allocator = _make_allocator(need_sort=False)
        allocated = allocator.alloc(2)

        self.assertIsNone(allocator.allocation_generations(allocated))

    def test_allocation_generations_change_only_on_writable_reuse(self):
        allocator = _make_allocator(need_sort=False)
        allocator.enable_allocation_generations()
        allocated = allocator.alloc(4)
        first = allocator.allocation_generations(allocated)

        self.assertIsNotNone(first)
        self.assertTrue(torch.equal(allocator.allocation_generations(allocated), first))
        allocator.free(allocated)
        self.assertTrue(torch.equal(allocator.allocation_generations(allocated), first))

        reused = allocator.alloc(4)
        self.assertTrue(torch.equal(reused, allocated))
        second = allocator.allocation_generations(reused)
        self.assertIsNotNone(second)
        self.assertFalse(torch.equal(first, second))

    def test_token_and_paged_overflow_preserve_capacity(self):
        allocators = (
            TokenToKVPoolAllocator(8, torch.float16, "cpu", None, False),
            _make_allocator(need_sort=False),
        )
        for allocator in allocators:
            with self.subTest(type=type(allocator).__name__):
                allocator.enable_allocation_generations()
                allocator._next_allocation_generation = torch.iinfo(torch.int64).max + 1
                before = allocator.available_size()
                with self.assertRaisesRegex(RuntimeError, "generation exhausted"):
                    allocator.alloc(allocator.page_size)
                self.assertEqual(allocator.available_size(), before)

    def test_paged_extend_and_decode_issue_generations(self):
        allocator = _make_allocator(need_sort=False)
        allocator.enable_allocation_generations()
        allocator.use_triton_kernels = False
        extended = allocator.alloc_extend(
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([2]),
            seq_lens_cpu=torch.tensor([2]),
            last_loc=torch.tensor([-1]),
            extend_num_tokens=2,
            num_new_pages=1,
        )
        self.assertIsNotNone(allocator.allocation_generations(extended))

        decoded = allocator.alloc_decode(
            seq_lens=torch.tensor([1]),
            seq_lens_cpu=torch.tensor([1]),
            last_loc=torch.tensor([-1]),
        )
        page_start = (decoded // PAGE_SIZE) * PAGE_SIZE
        decoded_page = page_start[:, None] + torch.arange(PAGE_SIZE)
        self.assertIsNotNone(allocator.allocation_generations(decoded_page.reshape(-1)))

    def test_resize_rebuilds_enabled_generation_ledger(self):
        allocator = _make_allocator(need_sort=False)
        allocator.enable_allocation_generations()
        allocator.resize(SimpleNamespace(max_total_num_tokens=8))

        self.assertEqual(allocator._allocation_generations.numel(), 5)

    def test_static_swa_resize_rebuilds_both_enabled_ledgers(self):
        allocator = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
        allocator._swa_req_ring = False
        allocator.page_size = PAGE_SIZE
        allocator.full_attn_allocator = _make_allocator(need_sort=False)
        allocator.swa_attn_allocator = _make_allocator(need_sort=False)
        allocator.full_attn_allocator.enable_allocation_generations()
        allocator.swa_attn_allocator.enable_allocation_generations()
        allocator.clear = MagicMock()

        allocator.resize(
            SimpleNamespace(full_max_total_num_tokens=8, swa_max_total_num_tokens=6)
        )

        self.assertEqual(
            allocator.full_attn_allocator._allocation_generations.numel(), 5
        )
        self.assertEqual(
            allocator.swa_attn_allocator._allocation_generations.numel(), 4
        )
        allocator.clear.assert_called_once_with()

    def test_clear_keeps_process_generation_monotonic(self):
        allocator = _make_allocator(need_sort=False)
        allocator.enable_allocation_generations()
        allocated = allocator.alloc(2)
        first = allocator.allocation_generations(allocated)

        allocator.clear()
        replacement = allocator.alloc(2)
        second = allocator.allocation_generations(replacement)

        self.assertTrue(torch.equal(replacement, allocated))
        self.assertGreater(int(second[0]), int(first[0]))

    def test_release_stages_until_free_pages_run_dry(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(24)
        free_pages_before = allocator.free_pages

        with patch.object(torch, "cat", wraps=torch.cat) as cat_mock:
            allocator.free(allocated[:2])
            allocator.free(allocated[4:6])
            self.assertIs(allocator.free_pages, free_pages_before)
            self.assertEqual(allocator.num_staged_pages, 2)
            self.assertEqual(allocator.available_size(), 12)
            cat_mock.assert_not_called()

            # free_pages still covers this: no merge.
            primary = allocator.alloc(8)
            self.assertTrue(torch.equal(_page_ids(primary), torch.arange(13, 17)))
            cat_mock.assert_not_called()

            # Pressure: one merge, staged pages come back sorted.
            reused = allocator.alloc(4)
            self.assertTrue(torch.equal(_page_ids(reused), torch.tensor([1, 3])))
            self.assertEqual(cat_mock.call_count, 1)

        self.assertEqual(allocator.num_staged_pages, 0)
        self.assertEqual(allocator.staged_pages, [])

    def test_clear_drops_staged_pages(self):
        allocator = _make_allocator(need_sort=True)
        allocator.free(allocator.alloc(4))

        allocator.clear()
        self.assertEqual(allocator.available_size(), allocator.size)
        self.assertEqual(allocator.staged_pages, [])


if __name__ == "__main__":
    unittest.main()
