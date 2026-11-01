"""Tests for gating the prefill/decode interval on running batch size."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(*, interval: int, min_bs: int) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.prefill_decode_interval_min_bs = min_bs
    scheduler.require_mlp_sync = False
    scheduler.spec_verify_between_prefills = False
    scheduler._spec_verify_owed = False
    return scheduler


def _arm(scheduler: Scheduler) -> None:
    scheduler._arm_prefill_decode_interval(
        SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=False,
        )
    )


def _make_running_batch(batch_size: int):
    return SimpleNamespace(batch_size=lambda: batch_size)


class TestPrefillDecodeIntervalMinBs(CustomTestCase):
    def test_spec_verify_runs_once_between_prefills(self):
        scheduler = _make_scheduler(interval=0, min_bs=0)
        scheduler.spec_verify_between_prefills = True
        _arm(scheduler)
        batch = SimpleNamespace(is_empty=lambda: False, is_prefill_only=False)
        self.assertTrue(scheduler._should_defer_prefill(batch))
        self.assertFalse(scheduler._should_defer_prefill(batch))

    def test_no_decode_work_does_not_block_prefill(self):
        for empty, prefill_only in ((True, False), (False, True)):
            scheduler = _make_scheduler(interval=0, min_bs=0)
            scheduler.spec_verify_between_prefills = True
            _arm(scheduler)
            batch = SimpleNamespace(
                is_empty=lambda: empty, is_prefill_only=prefill_only
            )
            self.assertFalse(scheduler._should_defer_prefill(batch))
            self.assertFalse(scheduler._spec_verify_owed)

    def test_disabled_spec_policy_preserves_zero_interval(self):
        scheduler = _make_scheduler(interval=0, min_bs=0)
        _arm(scheduler)
        self.assertFalse(scheduler._spec_verify_owed)
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(16)))

    def test_explicit_interval_takes_precedence_over_spec_policy(self):
        scheduler = _make_scheduler(interval=2, min_bs=8)
        scheduler.spec_verify_between_prefills = True
        _arm(scheduler)
        self.assertFalse(scheduler._spec_verify_owed)
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(8)))

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


if __name__ == "__main__":
    unittest.main()
