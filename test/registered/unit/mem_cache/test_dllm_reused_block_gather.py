"""Unit tests for the batched retained-block gather used by dLLM FDFO KV reuse.

``_alloc_extend_loc_with_kv_reuse`` re-reads each reuse row's retained KV block
out of ``req_to_token``. It used to do that one row at a time (a slice + cast
kernel per row); the rows share a block length, so they are now fetched with a
single flat index op. The index arithmetic (``row * width + start``, flattened
and reshaped back) is what these tests pin down -- it is the part a
"looks equivalent" rewrite silently gets wrong.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.allocation import _gather_reused_block_locs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROWS, WIDTH = 8, 64


def _fingerprint_pool() -> torch.Tensor:
    # Slot at [row, col] is row*1000 + col, so a wrong row or a wrong start
    # offset is visible in the value itself rather than only in the shape.
    rows = torch.arange(ROWS, dtype=torch.int32).unsqueeze(1) * 1000
    return rows + torch.arange(WIDTH, dtype=torch.int32)


def _per_row_reference(req_to_token, req_pool_indices, prefix_lens, block_len, dtype):
    return torch.stack(
        [
            req_to_token[idx, start : start + block_len].to(dtype)
            for idx, start in zip(req_pool_indices, prefix_lens)
        ]
    )


class TestGatherReusedBlockLocs(CustomTestCase):
    def _gather(self, req_pool_indices, prefix_lens, block_len, dtype=torch.int64):
        pool = _fingerprint_pool()
        got = _gather_reused_block_locs(
            req_to_token=pool,
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            block_len=block_len,
            device=torch.device("cpu"),
            out_dtype=dtype,
        )
        want = _per_row_reference(pool, req_pool_indices, prefix_lens, block_len, dtype)
        return got, want

    def test_matches_per_row_slicing_at_distinct_offsets(self):
        # Rows are non-contiguous and each starts at a different offset: the
        # flat index must combine the right row with the right start, not
        # broadcast one row's offset across all of them.
        got, want = self._gather([1, 3, 5], [4, 8, 0], 4)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(
            got.tolist(),
            [[1004, 1005, 1006, 1007], [3008, 3009, 3010, 3011], [5000, 5001, 5002, 5003]],
        )

    def test_preserves_row_order_when_pool_indices_are_unsorted(self):
        # The gather feeds out_cache_loc positionally, so it must follow the
        # caller's row order rather than the pool index order.
        got, want = self._gather([5, 0, 3], [0, 12, 4], 4)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual([row[0] for row in got.tolist()], [5000, 12, 3004])

    def test_block_touching_the_row_end_does_not_bleed_into_the_next_row(self):
        # A block ending exactly at WIDTH is the case a flat index gets wrong
        # by reading past the row boundary.
        got, want = self._gather([2], [WIDTH - 4], 4)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(got.tolist(), [[2060, 2061, 2062, 2063]])


if __name__ == "__main__":
    unittest.main()
