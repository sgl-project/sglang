"""Regression for SWA alloc_extend page estimation.

Old gate in SWATokenToKVPoolAllocator.alloc_extend added one full page_size
per request unconditionally, refusing extends that fit inside the request's
last partial page. Fix replaces with get_num_new_pages-based gating.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.test.test_utils import CustomTestCase


def _make_self(*, page_size: int, full_available: int, swa_available: int):
    full_indices = torch.tensor([10, 11], dtype=torch.int64)
    swa_indices = torch.tensor([20, 21], dtype=torch.int64)
    return SimpleNamespace(
        page_size=page_size,
        full_attn_allocator=SimpleNamespace(
            available_size=lambda: full_available,
            alloc_extend=MagicMock(return_value=full_indices),
        ),
        swa_attn_allocator=SimpleNamespace(
            available_size=lambda: swa_available,
            alloc_extend=MagicMock(return_value=swa_indices),
        ),
        translate_loc_from_full_to_swa=lambda last_loc: last_loc,
        full_to_swa_index_mapping=torch.zeros(64, dtype=torch.int64),
    )


def _call(stub, *, prefix_lens_cpu, seq_lens_cpu, extend_num_tokens):
    return SWATokenToKVPoolAllocator.alloc_extend(
        stub,
        prefix_lens=prefix_lens_cpu,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        last_loc=torch.tensor(
            [int(p) - 1 for p in prefix_lens_cpu.tolist()], dtype=torch.int64
        ),
        extend_num_tokens=extend_num_tokens,
    )


class TestSWAAllocExtendPageEstimation(CustomTestCase):
    def test_zero_new_pages_must_succeed(self):
        # Old: 2 + 2*8 = 18 > 16 -> would refuse.
        # New: prefix 5 -> 6 stays in page 0, 0 new pages.
        stub = _make_self(page_size=8, full_available=16, swa_available=16)
        result = _call(
            stub,
            prefix_lens_cpu=torch.tensor([5, 5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([6, 6], dtype=torch.int64),
            extend_num_tokens=2,
        )
        self.assertIsNotNone(result)
        stub.full_attn_allocator.alloc_extend.assert_called_once()
        stub.swa_attn_allocator.alloc_extend.assert_called_once()

    def test_one_new_page_fits(self):
        # Old: 6 + 2*8 = 22 > 16. New: 2 new pages == 16 // 8.
        stub = _make_self(page_size=8, full_available=16, swa_available=16)
        result = _call(
            stub,
            prefix_lens_cpu=torch.tensor([7, 7], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 10], dtype=torch.int64),
            extend_num_tokens=6,
        )
        self.assertIsNotNone(result)

    def test_full_pool_genuinely_insufficient(self):
        stub = _make_self(page_size=8, full_available=8, swa_available=64)
        result = _call(
            stub,
            prefix_lens_cpu=torch.tensor([8, 8, 8, 8, 8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([9, 9, 9, 9, 9], dtype=torch.int64),
            extend_num_tokens=5,
        )
        self.assertIsNone(result)
        stub.full_attn_allocator.alloc_extend.assert_not_called()

    def test_swa_pool_genuinely_insufficient(self):
        stub = _make_self(page_size=8, full_available=64, swa_available=8)
        result = _call(
            stub,
            prefix_lens_cpu=torch.tensor([8, 8, 8, 8, 8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([9, 9, 9, 9, 9], dtype=torch.int64),
            extend_num_tokens=5,
        )
        self.assertIsNone(result)
        stub.swa_attn_allocator.alloc_extend.assert_not_called()

    def test_exactly_at_capacity_succeeds(self):
        stub = _make_self(page_size=8, full_available=16, swa_available=16)
        result = _call(
            stub,
            prefix_lens_cpu=torch.tensor([8, 8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([9, 9], dtype=torch.int64),
            extend_num_tokens=2,
        )
        self.assertIsNotNone(result)

    def test_one_over_capacity_refuses(self):
        stub = _make_self(page_size=8, full_available=16, swa_available=16)
        result = _call(
            stub,
            prefix_lens_cpu=torch.tensor([8, 8, 8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([9, 9, 9], dtype=torch.int64),
            extend_num_tokens=3,
        )
        self.assertIsNone(result)

    def test_zero_new_pages_across_page_sizes(self):
        # Over-estimation gap grows with page_size; sweep to confirm fix
        # doesn't depend on the page_size=8 numbers above.
        for page_size in (16, 32, 64, 128):
            stub = _make_self(
                page_size=page_size,
                full_available=page_size * 2,
                swa_available=page_size * 2,
            )
            prefix = torch.tensor([page_size - 2] * 4, dtype=torch.int64)
            seq = torch.tensor([page_size - 1] * 4, dtype=torch.int64)
            result = _call(
                stub, prefix_lens_cpu=prefix, seq_lens_cpu=seq, extend_num_tokens=4
            )
            self.assertIsNotNone(result, f"page_size={page_size}")


if __name__ == "__main__":
    unittest.main()
