import unittest

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_DEV = "cpu"


def _make_expander(
    *, page_size: int, uses_legacy: bool = False
) -> SWATokenToKVPoolAllocator:
    allocator = object.__new__(SWATokenToKVPoolAllocator)
    allocator.page_size = page_size
    if uses_legacy:
        allocator.uses_legacy_real_length_alloc = True
    return allocator


def _tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=_DEV)


class TestExpandToFullPagesPageAligned(CustomTestCase):
    def test_rejects_a_partial_page_length(self):
        """Under the page-aligned contract a sub-page input is refused rather than folded."""
        expander = _make_expander(page_size=4)

        with self.assertRaises(AssertionError):
            expander._expand_to_full_pages(_tensor([4]))

    def test_expands_whole_pages_in_caller_block_order(self):
        """Whole-page blocks expand to their own pages, keeping the caller's block order."""
        expander = _make_expander(page_size=4)

        out = expander._expand_to_full_pages(_tensor([12, 13, 14, 15, 4, 5, 6, 7]))

        self.assertEqual(out.tolist(), [12, 13, 14, 15, 4, 5, 6, 7])


class TestExpandToFullPagesLegacy(CustomTestCase):
    def test_legacy_accepts_a_real_length_interval(self):
        """The legacy path folds a page-aligned real-length interval up to whole pages."""
        expander = _make_expander(page_size=16, uses_legacy=True)

        out = expander._expand_to_full_pages(torch.arange(16, 50, device=_DEV))

        self.assertEqual(out.tolist(), list(range(16, 64)))

    def test_legacy_recovers_pages_that_strided_folding_would_drop(self):
        """A block starting mid-page owns two pages; only the legacy unique fold finds both."""
        expander = _make_expander(page_size=4, uses_legacy=True)
        indices = _tensor([2, 3, 4, 5])

        self.assertEqual(indices.numel() % 4, 0)
        out = expander._expand_to_full_pages(indices)

        self.assertEqual(out.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])

    def test_legacy_recovers_pages_from_unordered_members(self):
        """The legacy path takes an arbitrary member set, not a page-ordered run."""
        expander = _make_expander(page_size=4, uses_legacy=True)

        out = expander._expand_to_full_pages(_tensor([5, 1]))

        self.assertEqual(out.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
