"""Explicit scheduler scheduling leaves defaults and CPU placement alone."""

import sys
import unittest
from unittest.mock import patch

from sglang.srt.managers.scheduler_components.thread_scheduling import (
    configure_python_thread_scheduling,
    validate_python_thread_switch_interval,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPythonThreadScheduling(CustomTestCase):
    def test_default_does_not_modify_the_interpreter(self):
        with patch("sys.setswitchinterval") as setter:
            configure_python_thread_scheduling(None)
        setter.assert_not_called()

    def test_explicit_setting_applies_and_records_actual_process_value(self):
        before = sys.getswitchinterval()
        self.addCleanup(sys.setswitchinterval, before)
        with self.assertLogs(
            "sglang.srt.managers.scheduler_components.thread_scheduling", level="INFO"
        ) as logs:
            configure_python_thread_scheduling(0.001)
        self.assertAlmostEqual(sys.getswitchinterval(), 0.001, places=8)
        self.assertIn("requested_seconds=0.001 actual_seconds=0.001", logs.output[0])

    def test_invalid_values_rejected_before_mutation(self):
        for value in (0, -1, float("nan"), float("inf"), -float("inf"), True, "0.001"):
            with self.subTest(value=value), patch("sys.setswitchinterval") as setter:
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    configure_python_thread_scheduling(value)
                setter.assert_not_called()

    def test_finite_positive_intervals_validate(self):
        for value in (None, 0.0001, 0.001, 0.005, 1):
            with self.subTest(value=value):
                validate_python_thread_switch_interval(value)


if __name__ == "__main__":
    unittest.main()
