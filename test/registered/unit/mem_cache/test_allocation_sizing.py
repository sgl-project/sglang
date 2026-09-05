"""CPU tests for page-aligned decode allocation sizing."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(kv_allocated_len, kv_committed_len):
    return SimpleNamespace(
        kv=SimpleNamespace(kv_allocated_len=kv_allocated_len),
        kv_committed_len=kv_committed_len,
    )


class TestPageAlignedDecodeAllocLens(CustomTestCase):
    def test_empty_requests(self):
        self.assertEqual(
            page_aligned_decode_alloc_lens([], reserve=16, page_size=4),
            ([], [], 0),
        )

    def test_rounds_target_up_to_page(self):
        self.assertEqual(
            page_aligned_decode_alloc_lens(
                [_req(kv_allocated_len=8, kv_committed_len=10)],
                reserve=4,
                page_size=4,
            ),
            ([8], [16], 8),
        )

    def test_exact_boundary_is_not_rounded_to_next_page(self):
        self.assertEqual(
            page_aligned_decode_alloc_lens(
                [_req(kv_allocated_len=4, kv_committed_len=8)],
                reserve=4,
                page_size=4,
            ),
            ([4], [12], 8),
        )

    def test_allocated_length_is_never_reduced(self):
        self.assertEqual(
            page_aligned_decode_alloc_lens(
                [_req(kv_allocated_len=32, kv_committed_len=10)],
                reserve=4,
                page_size=4,
            ),
            ([32], [32], 0),
        )

    def test_page_size_one(self):
        self.assertEqual(
            page_aligned_decode_alloc_lens(
                [_req(kv_allocated_len=7, kv_committed_len=9)],
                reserve=3,
                page_size=1,
            ),
            ([7], [12], 5),
        )

    def test_multiple_requests_sum_needed_tokens(self):
        reqs = [
            _req(kv_allocated_len=0, kv_committed_len=5),
            _req(kv_allocated_len=8, kv_committed_len=5),
            _req(kv_allocated_len=20, kv_committed_len=30),
        ]
        cur, nxt, needed = page_aligned_decode_alloc_lens(reqs, reserve=4, page_size=4)
        self.assertEqual(cur, [0, 8, 20])
        self.assertEqual(nxt, [12, 12, 36])
        self.assertEqual(needed, 32)

    def test_large_reserve_crosses_multiple_pages(self):
        self.assertEqual(
            page_aligned_decode_alloc_lens(
                [_req(kv_allocated_len=8, kv_committed_len=8)],
                reserve=17,
                page_size=8,
            ),
            ([8], [32], 24),
        )


if __name__ == "__main__":
    unittest.main()
