"""Live schedule_conservativeness updates through /set_internal_state."""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt import runtime_context as rc
from sglang.srt.constants import DP_ATTENTION_CONSERVATIVENESS_SCALE
from sglang.srt.managers.io_struct import SetInternalStateReq
from sglang.srt.managers.scheduler_components import internal_state
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLiveScheduleConservativeness(CustomTestCase):
    def setUp(self):
        rc.reset_context()
        rc.publish(ServerArgs(model_path="dummy"), role="test")
        self.scheduler = SimpleNamespace(
            new_token_ratio_tracker=NewTokenRatioTracker.from_config()
        )
        self.controller = SchedulerInternalStateController(
            scheduler=self.scheduler, record_step_time=False
        )

    def tearDown(self):
        rc.reset_context()

    def _set(self, **server_args) -> bool:
        req = SetInternalStateReq(server_args=server_args)
        return self.controller.set_internal_state(req).updated

    def _readback(self):
        return rc.get_context().resolved_server_args_dict()["schedule_conservativeness"]

    def test_update_rederives_the_tracker_as_at_startup(self):
        self.assertTrue(self._set(schedule_conservativeness=0.5))

        tracker = self.scheduler.new_token_ratio_tracker
        expected = NewTokenRatioTracker.from_conservativeness(0.5)
        self.assertEqual(tracker.init, expected.init)
        self.assertEqual(tracker.min, expected.min)
        self.assertEqual(tracker.decay, expected.decay)
        # Startup reads the live value, so a rebuild from config agrees.
        self.assertEqual(rc.get_schedule().schedule_conservativeness, 0.5)
        self.assertEqual(NewTokenRatioTracker.from_config().init, expected.init)
        self.assertEqual(self._readback(), 0.5)

    def test_int_is_accepted_and_reported_as_float(self):
        self.assertTrue(self._set(schedule_conservativeness=2))
        self.assertEqual(self._readback(), 2.0)
        self.assertIsInstance(self._readback(), float)

    def test_more_conservative_value_raises_the_running_ratio(self):
        self.scheduler.new_token_ratio_tracker.current = 0.1
        self.assertTrue(self._set(schedule_conservativeness=1.3))
        tracker = self.scheduler.new_token_ratio_tracker
        self.assertEqual(tracker.current, tracker.init)

    def test_less_conservative_value_keeps_a_higher_running_ratio(self):
        # A retract can leave the running ratio above init.
        self.scheduler.new_token_ratio_tracker.current = 0.95
        self.assertTrue(self._set(schedule_conservativeness=0.3))
        tracker = self.scheduler.new_token_ratio_tracker
        self.assertEqual(tracker.current, 0.95)
        tracker.decay_step()
        self.assertAlmostEqual(tracker.current, 0.95 - tracker.decay)

    def test_less_conservative_value_does_not_raise_a_decayed_ratio(self):
        # Under load the running ratio decays to min. Lowering the value must
        # not lift it back up to the new (higher) init.
        old = self.scheduler.new_token_ratio_tracker
        old.current = old.min
        self.assertTrue(self._set(schedule_conservativeness=0.3))
        tracker = self.scheduler.new_token_ratio_tracker
        self.assertLess(old.min, tracker.init)
        self.assertEqual(tracker.current, max(old.min, tracker.min))

    def test_less_conservative_value_floors_at_the_new_min(self):
        self.scheduler.new_token_ratio_tracker.current = 0.0
        self.assertTrue(self._set(schedule_conservativeness=0.3))
        tracker = self.scheduler.new_token_ratio_tracker
        self.assertEqual(tracker.current, tracker.min)

    def test_same_value_leaves_the_running_ratio_unchanged(self):
        old = self.scheduler.new_token_ratio_tracker
        old.current = old.min
        self.assertTrue(self._set(schedule_conservativeness=1.0))
        self.assertEqual(self.scheduler.new_token_ratio_tracker.current, old.min)

    def test_dp_attention_scales_the_live_value_as_at_startup(self):
        parallel = SimpleNamespace(attn_dp_enabled=True)
        with mock.patch.object(internal_state, "get_parallel", return_value=parallel):
            self.assertTrue(self._set(schedule_conservativeness=1.0))
        scaled = 1.0 * DP_ATTENTION_CONSERVATIVENESS_SCALE
        expected = NewTokenRatioTracker.from_conservativeness(scaled)
        self.assertEqual(self.scheduler.new_token_ratio_tracker.init, expected.init)
        self.assertEqual(self._readback(), scaled)

    def test_rejects_values_that_are_not_finite_positive_numbers(self):
        before_tracker = self.scheduler.new_token_ratio_tracker
        before_readback = self._readback()
        for value in (True, False, "1.0", None, math.nan, math.inf, 0, 0.0, -1.0):
            with self.subTest(value=value):
                self.assertFalse(self._set(schedule_conservativeness=value))
                self.assertIs(self.scheduler.new_token_ratio_tracker, before_tracker)
                self.assertEqual(self._readback(), before_readback)

    def test_rejected_key_in_the_same_request_leaves_the_value_unchanged(self):
        before_tracker = self.scheduler.new_token_ratio_tracker
        self.assertFalse(self._set(schedule_conservativeness=0.5, not_a_live_knob=1))
        self.assertIs(self.scheduler.new_token_ratio_tracker, before_tracker)
        self.assertEqual(rc.get_schedule().schedule_conservativeness, 1.0)


if __name__ == "__main__":
    unittest.main()
