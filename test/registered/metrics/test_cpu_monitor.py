import os
import tempfile
import threading
import time
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-a-unit-test")
register_amd_ci(est_time=5, suite="stage-a-unit-test-amd")


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

    def test_cpu_monitor_updates_gauge(self):
        from prometheus_client import REGISTRY

        from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread

        start_cpu_monitor_thread("test_gauge_component", interval=0.1)
        time.sleep(0.3)

        found = False
        for metric in REGISTRY.collect():
            if metric.name == "sglang:process_cpu_busy_ratio":
                for sample in metric.samples:
                    if sample.labels.get("component") == "test_gauge_component":
                        found = True
                        self.assertGreaterEqual(sample.value, 0)
                        self.assertLessEqual(sample.value, os.cpu_count())
                        break
        self.assertTrue(found, "CPU busy ratio metric not found")

    def test_cpu_monitor_measures_busy_time(self):
        from prometheus_client import REGISTRY

        from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread

        start_cpu_monitor_thread("busy_test", interval=0.1)

        def busy_work():
            end_time = time.monotonic() + 0.5
            while time.monotonic() < end_time:
                _ = sum(i * i for i in range(1000))

        busy_thread = threading.Thread(target=busy_work)
        busy_thread.start()
        busy_thread.join()
        time.sleep(0.2)

        for metric in REGISTRY.collect():
            if metric.name == "sglang:process_cpu_busy_ratio":
                for sample in metric.samples:
                    if sample.labels.get("component") == "busy_test":
                        self.assertGreater(sample.value, 0)
                        break


if __name__ == "__main__":
    unittest.main()
