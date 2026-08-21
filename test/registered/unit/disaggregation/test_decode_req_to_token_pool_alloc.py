"""Unit tests for srt/disaggregation/decode.py — DecodeReqToTokenPool.alloc().

Companion to test/registered/unit/mem_cache/test_req_to_token_pool_alloc.py:
DecodeReqToTokenPool.alloc() had the same free_slots head-slicing pattern as
ReqToTokenPool.alloc() (see python/sglang/srt/mem_cache/memory_pool.py), fixed
by popping from the tail instead.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


import unittest

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.test.test_utils import CustomTestCase


class _FakeReq:
    """Only carries the fields DecodeReqToTokenPool.alloc() reads."""

    def __init__(self, req_pool_idx=None, inflight_middle_chunks=0, kv_committed_len=0):
        self.req_pool_idx = req_pool_idx
        self.inflight_middle_chunks = inflight_middle_chunks
        self.kv_committed_len = kv_committed_len


def _make_pool(size=8, pre_alloc_size=0):
    return DecodeReqToTokenPool(
        size=size,
        max_context_len=16,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=pre_alloc_size,
    )


class TestDecodeReqToTokenPoolAlloc(CustomTestCase):
    def test_alloc_exhausts_pool_then_returns_none(self):
        pool = _make_pool(size=2, pre_alloc_size=0)
        self.assertEqual(pool.available_size(), 2)

        reqs = [_FakeReq(), _FakeReq()]
        indices = pool.alloc(reqs)
        self.assertEqual(len(indices), 2)
        self.assertEqual(pool.available_size(), 0)

        self.assertIsNone(pool.alloc([_FakeReq()]))

    def test_alloc_free_alloc_roundtrip_no_duplicates(self):
        pool = _make_pool(size=8, pre_alloc_size=0)

        reqs = [_FakeReq() for _ in range(5)]
        indices = pool.alloc(reqs)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(pool.available_size(), 3)

        for r in reqs:
            pool.free(r)
        self.assertEqual(pool.available_size(), 8)

        # Every slot must be reusable and still yield no duplicates.
        reqs2 = [_FakeReq() for _ in range(8)]
        indices2 = pool.alloc(reqs2)
        self.assertEqual(len(indices2), len(set(indices2)))
        self.assertEqual(pool.available_size(), 0)

    def test_alloc_with_all_reqs_reusing_is_noop_on_free_slots(self):
        """Regression test for the `free_slots[-0:]` slicing footgun.

        A naive tail-pop fix (`select_index = free_slots[-need_size:]` without
        special-casing need_size == 0) silently drains the entire free pool
        whenever every request in the batch is reusing its existing
        req_pool_idx (e.g. a chunked-prefill continuation batch), because
        `-0 == 0` and `list[-0:]` is the whole list, not `[]`.
        """
        pool = _make_pool(size=4, pre_alloc_size=0)
        self.assertEqual(pool.available_size(), 4)

        reusing_req = _FakeReq(req_pool_idx=1, inflight_middle_chunks=1)
        indices = pool.alloc([reusing_req])

        self.assertEqual(indices, [1])
        self.assertEqual(pool.available_size(), 4)


if __name__ == "__main__":
    unittest.main()
