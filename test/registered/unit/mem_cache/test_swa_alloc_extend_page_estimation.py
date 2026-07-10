"""Regression for SWA alloc page estimation.

Old gate in SWATokenToKVPoolAllocator.alloc_extend added one full page_size
per request unconditionally, refusing extends that fit inside the request's
last partial page. Fix replaces with get_num_new_pages-based gating.

Migrated to the page-granular ``alloc_pages`` surface (op6): callers now
compute the exact page count on the host and the composite gates on
``new_pages_available(num_pages, num_pages)``, so the contract under test is
that zero-page requests always succeed and the joint full/swa pre-check
refuses exactly when either pool lacks pages.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.base import expand_page_ids_to_token_ids
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _page_alloc_mock(start: int) -> MagicMock:
    def _alloc(num_pages: int) -> torch.Tensor:
        return torch.arange(start, start + num_pages, dtype=torch.int64)

    return MagicMock(side_effect=_alloc)


def _make_self(*, page_size: int, full_available: int, swa_available: int):
    full_to_swa_index_mapping = torch.zeros(page_size * 64, dtype=torch.int64)

    def new_pages_available(num_full_pages: int, num_swa_pages: int) -> bool:
        return (
            num_full_pages <= full_available // page_size
            and num_swa_pages <= swa_available // page_size
        )

    def set_full_to_swa_mapping(
        full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        full_to_swa_index_mapping[full_indices] = swa_indices

    return SimpleNamespace(
        page_size=page_size,
        full_attn_allocator=SimpleNamespace(
            available_size=lambda: full_available,
            alloc_pages=_page_alloc_mock(10),
        ),
        swa_attn_allocator=SimpleNamespace(
            available_size=lambda: swa_available,
            alloc_pages=_page_alloc_mock(20),
        ),
        new_pages_available=new_pages_available,
        set_full_to_swa_mapping=set_full_to_swa_mapping,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )


def _call(stub, *, num_pages: int):
    return SWATokenToKVPoolAllocator.alloc_pages(stub, num_pages)


class TestSWAAllocPagesEstimation(CustomTestCase):
    def test_zero_new_pages_must_succeed(self):
        """A zero-page request must succeed even when both pools are exhausted."""
        stub = _make_self(page_size=8, full_available=0, swa_available=0)
        result = _call(stub, num_pages=0)
        self.assertIsNotNone(result)
        self.assertEqual(int(result.numel()), 0)

    def test_new_pages_fit_available_pages(self):
        """A request that fits both pools' free pages succeeds."""
        stub = _make_self(page_size=8, full_available=16, swa_available=16)
        result = _call(stub, num_pages=2)
        self.assertIsNotNone(result)
        stub.full_attn_allocator.alloc_pages.assert_called_once_with(2)
        stub.swa_attn_allocator.alloc_pages.assert_called_once_with(2)

    def test_full_pool_genuinely_insufficient(self):
        """A full-pool shortfall refuses before touching either sub-allocator."""
        stub = _make_self(page_size=8, full_available=8, swa_available=64)
        result = _call(stub, num_pages=5)
        self.assertIsNone(result)
        stub.full_attn_allocator.alloc_pages.assert_not_called()

    def test_swa_pool_genuinely_insufficient(self):
        """A swa-pool shortfall refuses before touching either sub-allocator."""
        stub = _make_self(page_size=8, full_available=64, swa_available=8)
        result = _call(stub, num_pages=5)
        self.assertIsNone(result)
        stub.swa_attn_allocator.alloc_pages.assert_not_called()

    def test_exactly_at_capacity_succeeds(self):
        """A request that exactly consumes both pools' free pages succeeds."""
        stub = _make_self(page_size=8, full_available=16, swa_available=16)
        result = _call(stub, num_pages=2)
        self.assertIsNotNone(result)

    def test_one_over_capacity_refuses(self):
        """A request one page over either pool's capacity is refused."""
        stub = _make_self(page_size=8, full_available=16, swa_available=16)
        result = _call(stub, num_pages=3)
        self.assertIsNone(result)

    def test_mapping_written_for_whole_pages(self):
        """alloc_pages maps every token of each full page to its swa twin."""
        page_size = 8
        stub = _make_self(page_size=page_size, full_available=16, swa_available=16)
        result = _call(stub, num_pages=2)
        self.assertIsNotNone(result)
        full_tokens = expand_page_ids_to_token_ids(result, page_size)
        expected_swa_tokens = expand_page_ids_to_token_ids(
            torch.arange(20, 22, dtype=torch.int64), page_size
        )
        self.assertTrue(
            torch.equal(
                stub.full_to_swa_index_mapping[full_tokens], expected_swa_tokens
            )
        )

    def test_capacity_check_in_pages_across_page_sizes(self):
        """The pre-check counts pages, not tokens, for every page size."""
        for page_size in (16, 32, 64, 128):
            stub = _make_self(
                page_size=page_size,
                full_available=page_size * 2,
                swa_available=page_size * 2,
            )
            self.assertIsNotNone(
                _call(stub, num_pages=2), f"page_size={page_size} at capacity"
            )
            stub = _make_self(
                page_size=page_size,
                full_available=page_size * 2,
                swa_available=page_size * 2,
            )
            self.assertIsNone(
                _call(stub, num_pages=3), f"page_size={page_size} over capacity"
            )


if __name__ == "__main__":
    unittest.main()
