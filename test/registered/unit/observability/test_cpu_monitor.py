import threading
import time
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=23, suite="base-a-test-cpu", nightly=True)


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

    def test_repeated_cpu_clock_resets_preserve_counter(self):
        """CPU clock resets must keep the monitor running and its counter growing."""
        from prometheus_client import CollectorRegistry, Counter

        from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread

        registry = CollectorRegistry()
        counter = Counter(
            "test_process_cpu_seconds_total",
            "CPU time",
            ["component"],
            registry=registry,
        )
        labeled = counter.labels(component="restored")
        CpuTimes = namedtuple("CpuTimes", ["user", "system"])
        readings = [
            CpuTimes(10.0, 5.0),
            CpuTimes(10.25, 5.25),  # Normal increment.
            CpuTimes(0.1, 0.05),  # First restore.
            CpuTimes(0.3, 0.15),  # Increment from the new baseline.
            CpuTimes(0.01, 0.02),  # Second restore.
            CpuTimes(0.11, 0.12),
        ]
        test_thread = threading.current_thread()
        sleep = Mock(side_effect=[None] * 5 + [SystemExit])

        def monitor_sleep(interval):
            if threading.current_thread() is test_thread:
                return sleep(interval)
            # The smoke test leaves a daemon monitor running in this process.
            time.sleep(interval)

        with (
            patch("prometheus_client.Counter", return_value=counter),
            patch(
                "sglang.srt.observability.cpu_monitor.psutil.Process"
            ) as mock_process,
            patch(
                "sglang.srt.observability.cpu_monitor.time",
                SimpleNamespace(sleep=monitor_sleep),
            ),
            patch(
                "sglang.srt.observability.cpu_monitor.threading.Thread"
            ) as mock_thread,
            patch.object(labeled, "inc", wraps=labeled.inc) as increment,
        ):
            mock_process.return_value.cpu_times.side_effect = readings
            start_cpu_monitor_thread("restored")
            # Run the monitor synchronously so any counter error fails the test.
            with self.assertRaises(SystemExit):
                mock_thread.call_args.kwargs["target"]()

        self.assertEqual(increment.call_count, 5)
        for call, expected in zip(
            increment.call_args_list, (0.5, 0.15, 0.3, 0.03, 0.2)
        ):
            self.assertAlmostEqual(call.args[0], expected)
        self.assertAlmostEqual(
            registry.get_sample_value(
                "test_process_cpu_seconds_total", {"component": "restored"}
            ),
            1.18,
        )


if __name__ == "__main__":
    unittest.main()
