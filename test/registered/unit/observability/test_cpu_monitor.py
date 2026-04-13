import threading
import time
import unittest
from collections import namedtuple
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-test-cpu", nightly=True)


class TestCpuMonitor(unittest.TestCase):
    def test_cpu_monitor(self):
        from prometheus_client import REGISTRY

        from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread

        thread = start_cpu_monitor_thread("test", interval=0.1)
        self.assertTrue(thread.is_alive())
        self.assertTrue(thread.daemon)

        end_time = time.monotonic() + 0.3
        while time.monotonic() < end_time:
            _ = sum(i * i for i in range(1000))
        time.sleep(0.2)

        value = None
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if (
                    sample.name == "sglang:process_cpu_seconds_total"
                    and sample.labels.get("component") == "test"
                ):
                    value = sample.value
        print(f"sglang:process_cpu_seconds_total = {value}")
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)


class TestCpuMonitorMocked(unittest.TestCase):
    """Fast, deterministic tests for start_cpu_monitor_thread using mocks."""

    @patch("prometheus_client.Counter")
    @patch("sglang.srt.observability.cpu_monitor.psutil.Process")
    @patch("sglang.srt.observability.cpu_monitor.time.sleep")
    def test_delta_calculation_over_two_iterations(
        self, mock_sleep, MockProcess, MockCounter
    ):
        """Verify delta=(user_diff+system_diff) and last_times update across iterations."""
        from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread

        CpuTimes = namedtuple("CpuTimes", ["user", "system"])
        mock_process = MockProcess.return_value
        mock_process.cpu_times.side_effect = [
            CpuTimes(user=1.0, system=0.5),  # initial (L18)
            CpuTimes(user=2.5, system=1.0),  # iteration 1 (L22)
            CpuTimes(user=4.0, system=2.0),  # iteration 2 (L22)
        ]

        # Allow 2 loop iterations, then stop the thread.
        # Override threading.excepthook to suppress the pytest warning from
        # the intentional exception used to terminate the monitor loop.
        remaining = [2]
        orig_hook = threading.excepthook

        def controlled_sleep(seconds):
            if remaining[0] <= 0:
                raise SystemExit
            remaining[0] -= 1

        mock_sleep.side_effect = controlled_sleep
        threading.excepthook = lambda args: None

        mock_labeled = MagicMock()
        MockCounter.return_value.labels.return_value = mock_labeled

        thread = start_cpu_monitor_thread("my_component", interval=3.0)
        thread.join(timeout=1.0)
        threading.excepthook = orig_hook

        # Thread is daemon (L29)
        self.assertTrue(thread.daemon)

        # Sleep called with correct interval (L21)
        mock_sleep.assert_called_with(3.0)

        # Counter labeled with component (L26)
        MockCounter.return_value.labels.assert_called_with(component="my_component")

        # Delta calculation (L23-24) and counter increment (L26)
        inc_calls = mock_labeled.inc.call_args_list
        self.assertEqual(len(inc_calls), 2)
        # Iteration 1: (2.5 - 1.0) + (1.0 - 0.5) = 2.0
        self.assertAlmostEqual(inc_calls[0].args[0], 2.0)
        # Iteration 2: (4.0 - 2.5) + (2.0 - 1.0) = 2.5 (proves last_times updated)
        self.assertAlmostEqual(inc_calls[1].args[0], 2.5)


if __name__ == "__main__":
    unittest.main()
