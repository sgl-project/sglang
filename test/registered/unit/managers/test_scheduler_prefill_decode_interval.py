"""Tests for gating the prefill/decode interval on running batch size."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_SLOT = ScheduleSlot()


def _make_planner(*, interval: int, min_bs: int) -> StepPlanner:
    planner = StepPlanner.__new__(StepPlanner)
    planner.prefill_decode_interval = interval
    planner._prefill_decode_interval_remaining = 0
    planner.prefill_decode_interval_min_bs = min_bs
    planner.require_mlp_sync = False
    planner.core_lists = CoreLists()
    return planner


def _arm(planner: StepPlanner) -> None:
    planner.arm_prefill_decode_interval(
        SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=False,
        )
    )


def _should_defer(planner: StepPlanner, num_running: int) -> bool:
    planner.core_lists.decoding = [SimpleNamespace() for _ in range(num_running)]
    return planner.should_defer_prefill(_SLOT)


class TestPrefillDecodeIntervalMinBs(unittest.TestCase):
    def test_small_running_batch_admits_prefill_and_clears_interval(self):
        planner = _make_planner(interval=4, min_bs=8)
        _arm(planner)

        self.assertFalse(_should_defer(planner, 7))
        self.assertEqual(planner._prefill_decode_interval_remaining, 0)
        # The cleared interval does not resume once the batch grows.
        self.assertFalse(_should_defer(planner, 8))

    def test_large_running_batch_defers_prefill(self):
        planner = _make_planner(interval=2, min_bs=8)
        _arm(planner)

        self.assertTrue(_should_defer(planner, 8))
        self.assertTrue(_should_defer(planner, 9))
        self.assertFalse(_should_defer(planner, 9))

    def test_running_batch_shrinking_below_threshold_ends_interval(self):
        planner = _make_planner(interval=4, min_bs=8)
        _arm(planner)

        self.assertTrue(_should_defer(planner, 8))
        self.assertFalse(_should_defer(planner, 5))
        self.assertEqual(planner._prefill_decode_interval_remaining, 0)


if __name__ == "__main__":
    unittest.main()
