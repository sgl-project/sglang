"""Unit tests for managers/scheduler_recv_skipper.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-b-test-cpu")

import os
import unittest

from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


def _set_weights(default=1, decode=1, target_verify=1, none_weight=1):
    return {
        "SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT": str(default),
        "SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE": str(decode),
        "SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY": str(target_verify),
        "SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE": str(none_weight),
    }


class TestMaybeCreate(CustomTestCase):
    def test_interval_le_1_returns_none(self):
        self.assertIsNone(
            SchedulerRecvSkipper.maybe_create(
                ServerArgs(model_path="none", scheduler_recv_interval=1)
            )
        )
        self.assertIsNone(
            SchedulerRecvSkipper.maybe_create(
                ServerArgs(model_path="none", scheduler_recv_interval=0)
            )
        )

    def test_interval_gt_1_creates_instance(self):
        with unittest.mock.patch.dict(os.environ, _set_weights()):
            skipper = SchedulerRecvSkipper.maybe_create(
                ServerArgs(model_path="none", scheduler_recv_interval=3)
            )
        self.assertIsInstance(skipper, SchedulerRecvSkipper)

    def test_enable_dp_attention_asserts(self):
        with self.assertRaises(AssertionError):
            SchedulerRecvSkipper.maybe_create(
                ServerArgs(
                    model_path="none",
                    scheduler_recv_interval=3,
                    enable_dp_attention=True,
                )
            )


class TestHandle(CustomTestCase):
    def _make(self, interval=4, **weights):
        defaults = {"default": 1, "decode": 1, "target_verify": 1, "none_weight": 1}
        defaults.update(weights)
        self.env_ctx = unittest.mock.patch.dict(os.environ, _set_weights(**defaults))
        self.env_ctx.__enter__()
        self.addCleanup(self.env_ctx.__exit__)
        return SchedulerRecvSkipper(
            ServerArgs(model_path="none", scheduler_recv_interval=interval)
        )

    def test_four_calls_trigger_on_threshold_4(self):
        skipper = self._make(4)
        self.assertFalse(skipper.handle(ForwardMode.DECODE))
        self.assertFalse(skipper.handle(ForwardMode.DECODE))
        self.assertFalse(skipper.handle(ForwardMode.DECODE))
        self.assertTrue(skipper.handle(ForwardMode.DECODE))

    def test_counter_resets_after_trigger(self):
        skipper = self._make(4)
        for _ in range(4):
            skipper.handle(ForwardMode.DECODE)
        self.assertFalse(skipper.handle(ForwardMode.DECODE))

    def test_decode_weight_2_triggers_faster(self):
        skipper = self._make(4, decode=2)
        self.assertFalse(skipper.handle(ForwardMode.DECODE))
        self.assertTrue(skipper.handle(ForwardMode.DECODE))

    def test_target_verify_uses_own_weight(self):
        skipper = self._make(4, target_verify=4)
        self.assertTrue(skipper.handle(ForwardMode.TARGET_VERIFY))

    def test_none_forward_mode_uses_own_weight(self):
        skipper = self._make(4, none_weight=4)
        self.assertTrue(skipper.handle(None))

    def test_zero_weight_never_triggers(self):
        skipper = self._make(4, default=0, decode=0, target_verify=0, none_weight=0)
        for _ in range(100):
            self.assertFalse(skipper.handle(ForwardMode.DECODE))

    def test_unknown_forward_mode_falls_back_to_default(self):
        skipper = self._make(4, default=2, decode=1)
        self.assertFalse(skipper.handle("some_unknown_mode"))  # type: ignore
        self.assertTrue(skipper.handle("some_unknown_mode"))  # type: ignore


if __name__ == "__main__":
    unittest.main()
