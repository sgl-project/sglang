import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.utils.common import get_num_new_pages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_DEV = "cpu"


def _fake_kvcache() -> SimpleNamespace:
    return SimpleNamespace()


def _make_allocator(
    *, page_size: int, full_size: int, swa_size: int
) -> SWATokenToKVPoolAllocator:
    """Build the composite over real sub-allocators, without a real SWA KV pool.

    __init__ demands a BaseSWAKVPool purely to register the mapping; every code
    path under test reads only the sub-allocators and the mapping table.
    """
    allocator = object.__new__(SWATokenToKVPoolAllocator)
    allocator.page_size = page_size
    allocator.device = _DEV
    allocator.dtype = torch.float16
    allocator._size_full = full_size
    allocator._size_swa = swa_size
    allocator.need_sort = False
    allocator.is_not_in_free_group = True
    allocator.free_group = []
    allocator.free_pages = None
    allocator.release_pages = None

    if page_size == 1:
        allocator.full_attn_allocator = TokenToKVPoolAllocator(
            full_size, torch.float16, _DEV, _fake_kvcache(), False
        )
        allocator.swa_attn_allocator = TokenToKVPoolAllocator(
            swa_size, torch.float16, _DEV, _fake_kvcache(), False
        )
    else:
        allocator.full_attn_allocator = PagedTokenToKVPoolAllocator(
            full_size, page_size, torch.float16, _DEV, _fake_kvcache(), False
        )
        allocator.swa_attn_allocator = PagedTokenToKVPoolAllocator(
            swa_size, page_size, torch.float16, _DEV, _fake_kvcache(), False
        )

    allocator.full_to_swa_index_mapping = torch.cat(
        [
            torch.zeros(full_size + page_size, dtype=torch.int64, device=_DEV),
            torch.tensor([-1], dtype=torch.int64, device=_DEV),
        ]
    )
    return allocator


def _swa_tail_len(*, seq_len: int, window_size: int, page_size: int) -> int:
    """Mirror of DecodePreallocQueue._swa_tail_len, the only producer of the arg."""
    window_start = max(0, seq_len - window_size)
    window_start = (window_start // page_size) * page_size
    return seq_len - window_start


class TestSWAAllocExtendSwaTail(CustomTestCase):
    def test_alloc_extend_swa_tail_allocates_aligned_swa_window(self):
        """page 16 / window 32 / seq 50: SWA must cover [16, 64), not [18, 50)."""
        allocator = _make_allocator(page_size=16, full_size=1024, swa_size=1024)
        swa_returns = []
        real_swa_alloc = allocator.swa_attn_allocator.alloc

        def _spy_swa_alloc(need_size: int):
            out = real_swa_alloc(need_size)
            swa_returns.append(out)
            return out

        allocator.full_attn_allocator.alloc = MagicMock(
            wraps=allocator.full_attn_allocator.alloc
        )
        allocator.swa_attn_allocator.alloc = MagicMock(side_effect=_spy_swa_alloc)

        out = allocator.alloc_extend_swa_tail(seq_len=50, swa_tail_len=34)

        self.assertIsNotNone(out)
        # ceil(50/16)*16 == 64 full tokens; 64 - (50-34) == 48 swa tokens.
        allocator.full_attn_allocator.alloc.assert_called_once_with(64)
        allocator.swa_attn_allocator.alloc.assert_called_once_with(48)
        self.assertEqual(int(out.numel()), 64)

        mapping = allocator.full_to_swa_index_mapping
        # [0, 16) is the evicted head: mapped to the 0 tombstone.
        self.assertTrue(torch.all(mapping[out[:16].to(torch.int64)] == 0))
        # [16, 64) carries the whole swa allocation, in order.
        torch.testing.assert_close(mapping[out[16:].to(torch.int64)], swa_returns[0])

    def test_alloc_extend_swa_tail_rejects_unaligned_swa_start(self):
        """An unaligned swa_start would tear a page across the mapped/tombstoned boundary."""
        allocator = _make_allocator(page_size=16, full_size=1024, swa_size=1024)

        # seq 50 / tail 32 => swa_start 18, which is not on a page boundary.
        with self.assertRaises(AssertionError):
            allocator.alloc_extend_swa_tail(seq_len=50, swa_tail_len=32)

    def test_alloc_extend_swa_tail_returns_none_when_swa_pages_exhausted(self):
        """A refused allocation must not leave the full side already taken."""
        # 3 swa pages needed, 2 available.
        allocator = _make_allocator(page_size=16, full_size=1024, swa_size=32)
        full_free_before = int(allocator.full_attn_allocator.free_pages.numel())

        out = allocator.alloc_extend_swa_tail(seq_len=50, swa_tail_len=34)

        self.assertIsNone(out)
        self.assertEqual(
            int(allocator.full_attn_allocator.free_pages.numel()), full_free_before
        )

    def test_alloc_extend_swa_tail_page_estimation_matches_legacy(self):
        """Page counts must not regress against the get_num_new_pages / ceil(tail) formulas."""
        page_size = 16
        for seq_len, window_size in [
            (50, 32),
            (49, 32),
            (63, 32),
            (64, 32),
            (65, 32),
            (200, 128),
            (16, 32),
        ]:
            with self.subTest(seq_len=seq_len, window_size=window_size):
                allocator = _make_allocator(
                    page_size=page_size, full_size=4096, swa_size=4096
                )
                tail = _swa_tail_len(
                    seq_len=seq_len, window_size=window_size, page_size=page_size
                )
                full_before = int(allocator.full_attn_allocator.free_pages.numel())
                swa_before = int(allocator.swa_attn_allocator.free_pages.numel())

                out = allocator.alloc_extend_swa_tail(
                    seq_len=seq_len, swa_tail_len=tail
                )

                self.assertIsNotNone(out)
                full_used = full_before - int(
                    allocator.full_attn_allocator.free_pages.numel()
                )
                swa_used = swa_before - int(
                    allocator.swa_attn_allocator.free_pages.numel()
                )
                legacy_full = get_num_new_pages(
                    seq_lens=torch.tensor([seq_len], dtype=torch.int64),
                    page_size=page_size,
                    prefix_lens=torch.tensor([0], dtype=torch.int64),
                )
                legacy_swa = (tail + page_size - 1) // page_size
                self.assertEqual(full_used, int(legacy_full))
                self.assertEqual(swa_used, legacy_swa)


class TestSWAAllocExtendSwaTailMappingIsWholePages(CustomTestCase):
    """R2: the mapping's non-zero region must start and end on a page boundary.

    Under the old real-length allocation the SWA side took 3 whole pages but
    published only the first 34 of their 48 slot ids into the mapping. free_swa
    expands its input to whole full-side pages and then keeps the non-zero
    mapping entries, so it handed the SWA allocator 34 ids -- not a whole number
    of pages. The old free only survived that by reducing the ids to page
    numbers with torch.unique.
    """

    def test_free_swa_returns_whole_pages_after_swa_tail_prealloc(self):
        """The old semantics publish 34 of the 48 allocated slots; free_swa then sees 34."""
        allocator = _make_allocator(page_size=16, full_size=1024, swa_size=1024)
        freed = []
        allocator.swa_attn_allocator.free = MagicMock(
            side_effect=lambda idx: freed.append(idx)
        )

        out = allocator.alloc_extend_swa_tail(seq_len=50, swa_tail_len=34)
        allocator.free_swa(out)

        self.assertEqual(len(freed), 1)
        self.assertEqual(int(freed[0].numel()), 48)
        self.assertEqual(int(freed[0].numel()) % 16, 0)

    def test_swa_mapping_nonzero_region_endpoints_are_page_aligned(self):
        """Either endpoint off a page boundary hands free_swa a partial page."""
        page_size = 16
        for fill_len in (49, 50, 63, 64, 65):
            for window_size in (32, 128):
                with self.subTest(fill_len=fill_len, window_size=window_size):
                    allocator = _make_allocator(
                        page_size=page_size, full_size=4096, swa_size=4096
                    )
                    tail = _swa_tail_len(
                        seq_len=fill_len,
                        window_size=window_size,
                        page_size=page_size,
                    )
                    out = allocator.alloc_extend_swa_tail(
                        seq_len=fill_len, swa_tail_len=tail
                    )
                    self.assertIsNotNone(out)

                    mapped = allocator.full_to_swa_index_mapping[out.to(torch.int64)]
                    nonzero = torch.nonzero(mapped > 0).flatten()
                    self.assertGreater(int(nonzero.numel()), 0)
                    first = int(nonzero[0])
                    last = int(nonzero[-1])
                    # The region is contiguous...
                    self.assertEqual(int(nonzero.numel()), last - first + 1)
                    # ...and both cut points sit on a page boundary.
                    self.assertEqual(first % page_size, 0)
                    self.assertEqual((last + 1) % page_size, 0)

    def test_free_swa_partial_page_range_still_whole_pages(self):
        """A mid-sequence eviction frees some pages, and must still cut on page boundaries."""
        allocator = _make_allocator(page_size=16, full_size=1024, swa_size=1024)
        freed = []
        allocator.swa_attn_allocator.free = MagicMock(
            side_effect=lambda idx: freed.append(idx)
        )

        out = allocator.alloc_extend_swa_tail(seq_len=50, swa_tail_len=34)
        # Evict the middle two full pages only, [32, 64) of the sequence.
        allocator.free_swa(out[32:64])

        self.assertEqual(len(freed), 1)
        self.assertEqual(int(freed[0].numel()), 32)
        self.assertEqual(int(freed[0].numel()) % 16, 0)


class TestSWAAllocPageAligned(CustomTestCase):
    def test_alloc_page_aligned_allocates_both_pools(self):
        """page_size > 1 used to be refused outright, leaving hybrid SWA no alloc entry."""
        allocator = _make_allocator(page_size=16, full_size=1024, swa_size=1024)

        out = allocator.alloc(32)

        self.assertIsNotNone(out)
        self.assertEqual(int(out.numel()), 32)
        mapping = allocator.full_to_swa_index_mapping
        swa_ids = mapping[out.to(torch.int64)]
        # Both sides gave 32 distinct slots, and every full slot maps to one.
        self.assertEqual(int(torch.unique(swa_ids).numel()), 32)
        self.assertTrue(torch.all(swa_ids > 0))

    def test_alloc_rejects_a_size_that_is_not_whole_pages(self):
        """A partial-page alloc would publish a mapping the whole-page free cannot undo."""
        allocator = _make_allocator(page_size=16, full_size=1024, swa_size=1024)

        with self.assertRaises(AssertionError):
            allocator.alloc(20)

    def test_alloc_returns_none_when_a_pool_is_exhausted(self):
        """The joint pre-check must refuse before either side is taken."""
        allocator = _make_allocator(page_size=16, full_size=1024, swa_size=32)
        full_free_before = int(allocator.full_attn_allocator.free_pages.numel())

        self.assertIsNone(allocator.alloc(64))
        self.assertEqual(
            int(allocator.full_attn_allocator.free_pages.numel()), full_free_before
        )

    def test_alloc_page_size_one_matches_legacy_behavior(self):
        """page_size == 1 is relaxed, not rewritten: its old behaviour must be untouched."""
        allocator = _make_allocator(page_size=1, full_size=64, swa_size=64)

        out = allocator.alloc(5)

        self.assertIsNotNone(out)
        self.assertEqual(int(out.numel()), 5)
        mapping = allocator.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[out.to(torch.int64)] > 0))

    def test_alloc_page_size_one_returns_none_when_swa_is_short(self):
        """The old token-count capacity check must survive the switch to a page check."""
        allocator = _make_allocator(page_size=1, full_size=64, swa_size=4)

        self.assertIsNone(allocator.alloc(5))
        self.assertEqual(int(allocator.full_attn_allocator.available_size()), 64)


if __name__ == "__main__":
    unittest.main()
