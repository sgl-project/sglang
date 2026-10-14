"""Tests for scheduler prefill/decode interleaving."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.step_planner import StepPlanner

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_planner(*, interval: int, require_mlp_sync: bool) -> StepPlanner:
    planner = StepPlanner.__new__(StepPlanner)
    planner.prefill_decode_interval = interval
    planner._prefill_decode_interval_remaining = 0
    planner.require_mlp_sync = require_mlp_sync
    return planner


def _make_batch(*, local_extend: bool, global_extend: bool):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: local_extend),
        is_extend_in_batch=global_extend,
    )


class TestPrefillDecodeInterval(unittest.TestCase):
    def test_disabled_interval_does_not_arm(self):
        planner = _make_planner(interval=0, require_mlp_sync=False)

        planner.arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertFalse(planner.should_defer_prefill())

    def test_non_dp_interval_uses_local_forward_mode(self):
        planner = _make_planner(interval=2, require_mlp_sync=False)

        planner.arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertTrue(planner.should_defer_prefill())
        self.assertTrue(planner.should_defer_prefill())
        self.assertFalse(planner.should_defer_prefill())

    def test_dp_interval_uses_globally_synchronized_extend_flag(self):
        planner = _make_planner(interval=2, require_mlp_sync=True)

        # This rank is locally decoding, but another DP rank is prefilling.
        planner.arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=True)
        )

        self.assertEqual(planner._prefill_decode_interval_remaining, 2)
        self.assertTrue(planner.should_defer_prefill())

    def test_decode_batch_does_not_rearm_interval(self):
        planner = _make_planner(interval=2, require_mlp_sync=True)
        planner._prefill_decode_interval_remaining = 1

        planner.arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=False)
        )

        self.assertEqual(planner._prefill_decode_interval_remaining, 1)


if __name__ == "__main__":
    unittest.main()
