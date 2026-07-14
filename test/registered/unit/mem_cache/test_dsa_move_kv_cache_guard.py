"""Unit tests for the DSATokenToKVPool.move_kv_cache page-index guard.

Regression guard for the DSA index-K move bug: ``index_k_with_scale_buffer`` is
PAGE-indexed (dim-0 is ``num_pages``, with ``page_size`` tokens packed along
dim-1), but ``move_kv_cache`` receives per-TOKEN locations. The per-token row
copy is correct only for ``page_size == 1``; for ``page_size > 1`` (all CUDA DSA
uses ``page_size == 64``, HIP preshuffle uses a multiple of 16) it would index
the wrong rows / go out of bounds and silently corrupt the indexer cache. The
fix guards that path with an explicit ``NotImplementedError`` for
``page_size != 1``.

These tests pin the behavior without a GPU by bypassing the heavy
(GPU/allocator) ``__init__`` and stubbing the base MLA move (which is
token-indexed and correct), so only the DSA-specific guard is exercised.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


# The base MLA move (super().move_kv_cache) is token-indexed and correct, but it
# touches self.kv_buffer / self.size, which the bypassed __init__ never set.
# Stub it to a no-op so the tests isolate the DSA index-K guard on the CPU.
def _no_base_move(self, tgt_loc, src_loc):
    return None


class TestDSAMoveKVCacheGuard(CustomTestCase):
    @staticmethod
    def _make_pool(page_size: int) -> DSATokenToKVPool:
        # __new__ gives a real DSATokenToKVPool instance (so the zero-arg super()
        # call inside move_kv_cache resolves) without running the GPU __init__.
        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.page_size = page_size
        return pool

    def test_move_kv_cache_raises_for_page_size_gt_1(self):
        """page_size > 1 must raise instead of doing a corrupting per-token move.

        Covers the real configs: 64 (all CUDA DSA) and 16 (a HIP preshuffle
        page_size, which must be a multiple of 16). A guard written as
        ``== 64`` instead of ``!= 1`` would wrongly pass page_size 16 through.
        """
        tgt_loc = torch.tensor([0, 1], dtype=torch.int64)
        src_loc = torch.tensor([2, 3], dtype=torch.int64)
        for page_size in (16, 64):
            with self.subTest(page_size=page_size):
                pool = self._make_pool(page_size)
                with patch.object(MLATokenToKVPool, "move_kv_cache", _no_base_move):
                    with self.assertRaises(NotImplementedError):
                        pool.move_kv_cache(tgt_loc, src_loc)

    def test_move_kv_cache_page_size_1_moves_without_guard(self):
        """page_size == 1 (HIP legacy): token index == page index, so the guard
        must NOT fire and the per-token move must still copy the right row."""
        pool = self._make_pool(1)
        # (num_tokens=4, packed_dim=6); with page_size==1 a "page" is one token.
        buf = torch.arange(4 * 6, dtype=torch.uint8).view(4, 6)
        pool.index_k_with_scale_buffer = [buf]
        expected_row = buf[2].clone()

        tgt_loc = torch.tensor([0], dtype=torch.int64)
        src_loc = torch.tensor([2], dtype=torch.int64)
        with patch.object(MLATokenToKVPool, "move_kv_cache", _no_base_move):
            pool.move_kv_cache(tgt_loc, src_loc)  # must not raise

        # Row 0 now holds the old contents of row 2: the per-token move ran.
        self.assertTrue(torch.equal(buf[0], expected_row))


if __name__ == "__main__":
    unittest.main()
