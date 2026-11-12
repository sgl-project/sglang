"""Tests for gating the prefill/decode interval on the running batch."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(
    *, interval: int, min_bs: int, require_mlp_sync: bool = False
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.prefill_decode_interval_min_bs = min_bs
    scheduler.require_mlp_sync = require_mlp_sync
    return scheduler


def _arm(scheduler: Scheduler) -> None:
    scheduler._arm_prefill_decode_interval(
        SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=True,
        )
    )


def _make_running_batch(batch_size: int, *, prefill_only: bool = False):
    return SimpleNamespace(
        batch_size=lambda: batch_size,
        is_empty=lambda: batch_size == 0,
        is_prefill_only=prefill_only,
    )


class TestPrefillDecodeIntervalMinBs(unittest.TestCase):
    def test_small_running_batch_admits_prefill_and_clears_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(7)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)
        # The cleared interval does not resume once the batch grows.
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(8)))

    def test_large_running_batch_defers_prefill(self):
        scheduler = _make_scheduler(interval=2, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(9)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(9)))

    def test_running_batch_shrinking_below_threshold_ends_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(5)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)


class TestPrefillDecodeIntervalNothingDecodes(unittest.TestCase):
    # These use the default min_bs=0, where the interval applies at any batch size.

    def test_empty_running_batch_does_not_defer(self):
        # The next chunk of a long prompt on an idle server runs right away.
        scheduler = _make_scheduler(interval=4, min_bs=0)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(0)))

    def test_prefill_only_running_batch_does_not_defer(self):
        scheduler = _make_scheduler(interval=4, min_bs=0)
        _arm(scheduler)

        running = _make_running_batch(3, prefill_only=True)
        self.assertFalse(scheduler._should_defer_prefill(running))

    def test_single_decoding_request_keeps_interval(self):
        scheduler = _make_scheduler(interval=2, min_bs=0)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(1)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(1)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(1)))

    def test_dp_defers_without_local_decode(self):
        # Under DP attention, peer ranks may still be decoding.
        scheduler = _make_scheduler(interval=2, min_bs=0, require_mlp_sync=True)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(0)))
        running = _make_running_batch(3, prefill_only=True)
        self.assertTrue(scheduler._should_defer_prefill(running))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(0)))


if __name__ == "__main__":
    unittest.main()
