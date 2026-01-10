import time
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="default", nightly=True)


class TestCpuMonitor(unittest.TestCase):
    def test_cpu_monitor(self):
        from prometheus_client import REGISTRY

        from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread

        thread = start_cpu_monitor_thread("test", interval=0.1)
        self.assertTrue(thread.is_alive())
        self.assertTrue(thread.daemon)

        for _ in range(100000):
            pass
        time.sleep(0.2)

        for metric in REGISTRY.collect():
            if metric.name == "sglang:process_cpu_seconds_total":
                for sample in metric.samples:
                    if sample.labels.get("component") == "test":
                        self.assertGreater(sample.value, 0)
                        return
        self.fail("CPU seconds counter not found")


if __name__ == "__main__":
    unittest.main()
