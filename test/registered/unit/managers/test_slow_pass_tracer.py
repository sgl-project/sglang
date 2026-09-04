"""Unit tests for SlowPassTracer (SGLANG_DEBUG_SLOW_SCHEDULER_PASS_MS)."""

import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.slow_pass_tracer import SlowPassTracer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TRACER_MOD = "sglang.srt.managers.scheduler_components.slow_pass_tracer"


class TestSlowPassTracer(CustomTestCase):
    def test_disabled_by_default_no_watchdog(self):
        tracer = SlowPassTracer()
        self.assertFalse(tracer.enabled)
        with patch(f"{TRACER_MOD}.faulthandler") as mock_fh:
            with tracer.trace("process_batch_result"):
                pass
        mock_fh.dump_traceback_later.assert_not_called()

    def test_fast_pass_arms_and_cancels_without_warning(self):
        with envs.SGLANG_DEBUG_SLOW_SCHEDULER_PASS_MS.override(100.0):
            tracer = SlowPassTracer()
        self.assertEqual(tracer.timeout_s, 0.1)
        with (
            patch(f"{TRACER_MOD}.faulthandler") as mock_fh,
            patch(f"{TRACER_MOD}.logger") as mock_log,
        ):
            with tracer.trace("get_next_batch_to_run"):
                pass
        mock_fh.dump_traceback_later.assert_called_once_with(
            0.1, repeat=False, exit=False
        )
        mock_fh.cancel_dump_traceback_later.assert_called_once()
        mock_log.warning.assert_not_called()

    def test_slow_pass_logs_attribution_line(self):
        with envs.SGLANG_DEBUG_SLOW_SCHEDULER_PASS_MS.override(1.0):
            tracer = SlowPassTracer()
        clock = [0.0]
        with (
            patch(f"{TRACER_MOD}.faulthandler") as mock_fh,
            patch(f"{TRACER_MOD}.logger") as mock_log,
            patch(f"{TRACER_MOD}.time") as mock_time,
        ):
            mock_time.perf_counter.side_effect = lambda: clock[0]
            with tracer.trace("process_batch_result"):
                clock[0] += 0.35
        mock_fh.cancel_dump_traceback_later.assert_called_once()
        mock_log.warning.assert_called_once()
        args = mock_log.warning.call_args[0]
        self.assertIn("process_batch_result", args)

    def test_cancel_runs_when_pass_raises(self):
        with envs.SGLANG_DEBUG_SLOW_SCHEDULER_PASS_MS.override(50.0):
            tracer = SlowPassTracer()
        with patch(f"{TRACER_MOD}.faulthandler") as mock_fh:
            with self.assertRaises(RuntimeError):
                with tracer.trace("process_batch_result"):
                    raise RuntimeError("boom")
        mock_fh.cancel_dump_traceback_later.assert_called_once()


if __name__ == "__main__":
    unittest.main()
