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

        metrics = REGISTRY.collect()
        print(f"{metrics=}")

        value = None
        for metric in metrics:
            for sample in metric.samples:
                if sample.labels.get("component") == "test":
                    value = sample.value

        print(f"CPU seconds value: {value}")
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)


if __name__ == "__main__":
    unittest.main()
