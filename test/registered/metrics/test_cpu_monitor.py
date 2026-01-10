import os
import tempfile
import threading
import time
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="default", nightly=True)


class TestCpuMonitor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = self.temp_dir

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_cpu_monitor_thread_starts(self):
        from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread

        thread = start_cpu_monitor_thread("test_component", interval=0.1)
        self.assertTrue(thread.is_alive())
        self.assertTrue(thread.daemon)

    def test_cpu_monitor_increments_counter(self):
        from prometheus_client import REGISTRY

        from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread

        start_cpu_monitor_thread("test_counter_component", interval=0.1)

        def busy_work():
            end_time = time.monotonic() + 0.3
            while time.monotonic() < end_time:
                _ = sum(i * i for i in range(1000))

        busy_thread = threading.Thread(target=busy_work)
        busy_thread.start()
        busy_thread.join()
        time.sleep(0.2)

        found = False
        for metric in REGISTRY.collect():
            if metric.name == "sglang:process_cpu_seconds_total":
                for sample in metric.samples:
                    if (
                        sample.labels.get("component") == "test_counter_component"
                        and sample.name == "sglang:process_cpu_seconds_total_total"
                    ):
                        found = True
                        self.assertGreater(sample.value, 0)
                        break
        self.assertTrue(found, "CPU seconds counter not found or not incremented")

    def test_cpu_monitor_counter_monotonic(self):
        from prometheus_client import REGISTRY

        from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread

        start_cpu_monitor_thread("test_monotonic", interval=0.1)

        def get_counter_value():
            for metric in REGISTRY.collect():
                if metric.name == "sglang:process_cpu_seconds_total":
                    for sample in metric.samples:
                        if (
                            sample.labels.get("component") == "test_monotonic"
                            and sample.name == "sglang:process_cpu_seconds_total_total"
                        ):
                            return sample.value
            return None

        time.sleep(0.2)
        value1 = get_counter_value()

        _ = sum(i * i for i in range(100000))
        time.sleep(0.2)
        value2 = get_counter_value()

        self.assertIsNotNone(value1)
        self.assertIsNotNone(value2)
        self.assertGreaterEqual(value2, value1)


if __name__ == "__main__":
    unittest.main()
