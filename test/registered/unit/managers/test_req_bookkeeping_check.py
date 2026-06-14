"""Unit tests for the per-request bookkeeping clock checks.

Covers the runtime invariants behind SGLANG_ENABLE_REQ_BOOKKEEPING_CHECK:
at most one decode_batch_idx tick and one maybe_evict_swa pass per decode
iteration, a live clock at resolve, and KV watermark ordering. Static
companion: test/registered/unit/spec/test_decode_bookkeeping_ownership.py.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.invariant_checker import (
    bk_check_watermarks,
    bk_on_clock_tick,
    bk_on_evict_swa,
    bk_on_prepare_decode,
    bk_on_resolve_decode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_ROW_WIDTH = 64


def _make_req(rid="r0", committed=3, allocated=5):
    return SimpleNamespace(
        rid=rid,
        decode_batch_idx=0,
        kv_committed_len=committed,
        kv_allocated_len=allocated,
    )


def _make_batch(reqs):
    return SimpleNamespace(
        reqs=reqs,
        req_to_token_pool=SimpleNamespace(
            req_to_token=SimpleNamespace(shape=(8, _ROW_WIDTH))
        ),
    )


class TestReqBookkeepingCheck(CustomTestCase):
    def test_disabled_is_noop(self):
        req = _make_req()
        batch = _make_batch([req])
        bk_on_prepare_decode(batch)
        self.assertFalse(hasattr(batch, "_bk_iter_id"))
        # Hooks without an iter id are no-ops.
        bk_on_clock_tick(req, batch)
        bk_on_evict_swa(batch)

    def test_one_tick_per_iter_ok_double_tick_raises(self):
        req = _make_req()
        batch = _make_batch([req])
        with envs.SGLANG_ENABLE_REQ_BOOKKEEPING_CHECK.override(True):
            bk_on_prepare_decode(batch)
            bk_on_clock_tick(req, batch)
            with self.assertRaisesRegex(AssertionError, "ticked twice"):
                bk_on_clock_tick(req, batch)
            # A new iteration accepts a new tick.
            bk_on_prepare_decode(batch)
            bk_on_clock_tick(req, batch)

    def test_double_evict_raises(self):
        batch = _make_batch([_make_req()])
        with envs.SGLANG_ENABLE_REQ_BOOKKEEPING_CHECK.override(True):
            bk_on_prepare_decode(batch)
            bk_on_evict_swa(batch)
            with self.assertRaisesRegex(AssertionError, "ran twice"):
                bk_on_evict_swa(batch)
            bk_on_prepare_decode(batch)
            bk_on_evict_swa(batch)

    def test_resolve_requires_live_clock(self):
        req = _make_req()
        batch = _make_batch([req])
        with envs.SGLANG_ENABLE_REQ_BOOKKEEPING_CHECK.override(True):
            bk_on_prepare_decode(batch)
            bk_on_clock_tick(req, batch)
            bk_on_resolve_decode(req)
            never_ticked = _make_req(rid="r1")
            with self.assertRaisesRegex(AssertionError, "never ticked"):
                bk_on_resolve_decode(never_ticked)

    def test_watermark_ordering(self):
        ok = _make_batch([_make_req(committed=4, allocated=4)])
        bk_check_watermarks(ok)
        with self.assertRaisesRegex(AssertionError, "KV watermark"):
            bk_check_watermarks(_make_batch([_make_req(committed=6, allocated=5)]))
        with self.assertRaisesRegex(AssertionError, "KV watermark"):
            bk_check_watermarks(
                _make_batch([_make_req(committed=5, allocated=_ROW_WIDTH + 1)])
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
