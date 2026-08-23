"""Unit tests for srt/mem_cache/allocation_sizing - CPU only, no server, no model.

Covers `page_aligned_decode_alloc_lens`, the pure page-rounding arithmetic used to
size per-request decode allocations; the runtime-config getters (get_alloc_page_size,
get_alloc_len_per_decode, ...) read the runtime context and are exercised via E2E
paths instead.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(kv_allocated_len, kv_committed_len):
    # Mirrors the real request object layout: kv_allocated_len is nested under
    # r.kv while kv_committed_len sits directly on the request.
    return SimpleNamespace(
        kv=SimpleNamespace(kv_allocated_len=kv_allocated_len),
        kv_committed_len=kv_committed_len,
    )


class TestPageAlignedDecodeAllocLens(CustomTestCase):
    def test_empty_requests(self):
        self.assertEqual(page_aligned_decode_alloc_lens([], reserve=16, page_size=4), ([], [], 0))

    def test_rounds_up_to_page(self):
        # committed 10 + reserve 4 = 14 -> ceil(14/4)*4 = 16
        cur, nxt, needed = page_aligned_decode_alloc_lens(
            [_req(kv_allocated_len=8, kv_committed_len=10)], reserve=4, page_size=4
        )
        self.assertEqual(cur, [8])
        self.assertEqual(nxt, [16])
        self.assertEqual(needed, 8)

    def test_exact_page_boundary_unchanged(self):
        # committed 8 + reserve 4 = 12, already page-aligned -> 12
        cur, nxt, needed = page_aligned_decode_alloc_lens(
            [_req(kv_allocated_len=4, kv_committed_len=8)], reserve=4, page_size=4
        )
        self.assertEqual(nxt, [12])
        self.assertEqual(needed, 8)

    def test_never_shrinks_below_allocated(self):
        # committed is far behind allocated; must not shrink back
        cur, nxt, needed = page_aligned_decode_alloc_lens(
            [_req(kv_allocated_len=32, kv_committed_len=10)], reserve=4, page_size=4
        )
        self.assertEqual(nxt, [32])
        self.assertEqual(needed, 0)

    def test_page_size_one_is_identity_plus_reserve(self):
        cur, nxt, needed = page_aligned_decode_alloc_lens(
            [_req(kv_allocated_len=7, kv_committed_len=9)], reserve=3, page_size=1
        )
        self.assertEqual(cur, [7])
        self.assertEqual(nxt, [12])
        self.assertEqual(needed, 5)

    def test_multiple_requests_accumulate(self):
        reqs = [
            _req(kv_allocated_len=0, kv_committed_len=5),
            _req(kv_allocated_len=8, kv_committed_len=5),
            _req(kv_allocated_len=20, kv_committed_len=30),
        ]
        cur, nxt, needed = page_aligned_decode_alloc_lens(
            reqs, reserve=4, page_size=4
        )
        # r0: ceil(9/4)*4 = 12 -> +12; r1: ceil(9/4)*4 = 12 >= 8 -> +4;
        # r2: ceil(34/4)*4 = 36 >= 20 -> +16; sum = 32
        self.assertEqual(cur, [0, 8, 20])
        self.assertEqual(nxt, [12, 12, 36])
        self.assertEqual(needed, 32)

    def test_large_reserve_crosses_pages(self):
        cur, nxt, needed = page_aligned_decode_alloc_lens(
            [_req(kv_allocated_len=8, kv_committed_len=8)], reserve=17, page_size=8
        )
        # 8+17=25 -> ceil(25/8)*8 = 32
        self.assertEqual(nxt, [32])
        self.assertEqual(needed, 24)


if __name__ == "__main__":
    unittest.main()
