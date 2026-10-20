"""Real Prometheus registration and subprocess aggregation, without a server."""

import os
import subprocess
import sys
import tempfile
import unittest

from prometheus_client import CollectorRegistry, multiprocess

from sglang.srt.observability import kv_transfer_metrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Load only the collector in each fresh interpreter so import timing is observable.
LOAD_COLLECTOR = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("kv_transfer_metrics", sys.argv[1])
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
"""


class TestKVTransferMetrics(CustomTestCase):
    def run_collector(self, code, *, args=(), env=None):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                LOAD_COLLECTOR + code,
                kv_transfer_metrics.__file__,
                *args,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_disabled_metrics_do_not_import_or_register_prometheus(self):
        self.run_collector(
            """
assert "prometheus_client" not in sys.modules
assert metrics.get_kv_transfer_timeout_counter(enabled=False) is None
assert "prometheus_client" not in sys.modules
"""
        )

    def test_enabled_metrics_seed_both_stages_and_reuse_the_counter(self):
        self.run_collector(
            """
assert metrics.get_kv_transfer_timeout_counter(enabled=False) is None
counter = metrics.get_kv_transfer_timeout_counter(enabled=True)
assert counter is metrics.get_kv_transfer_timeout_counter(enabled=True)
assert metrics.get_kv_transfer_timeout_counter(enabled=False) is None
from prometheus_client import REGISTRY
for stage in ("bootstrap", "transfer"):
    assert REGISTRY.get_sample_value(
        "sglang:kv_transfer_timeouts_total", {"stage": stage}
    ) == 0
counter.labels(stage="bootstrap").inc()
assert REGISTRY.get_sample_value(
    "sglang:kv_transfer_timeouts_total", {"stage": "bootstrap"}
) == 1
"""
        )

    def test_multiprocess_export_sums_observers_without_peer_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": directory}
            for increments in (1, 2):
                self.run_collector(
                    """
counter = metrics.get_kv_transfer_timeout_counter(enabled=True)
counter.labels(stage="bootstrap").inc(int(sys.argv[2]))
counter.labels(stage="transfer").inc()
""",
                    args=(str(increments),),
                    env=env,
                )
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry, path=directory)
            samples = {
                tuple(sample.labels.items()): sample.value
                for metric in registry.collect()
                for sample in metric.samples
                if sample.name == "sglang:kv_transfer_timeouts_total"
            }
            self.assertEqual(
                samples,
                {(("stage", "bootstrap"),): 3, (("stage", "transfer"),): 2},
            )

    def test_new_multiprocess_directory_starts_at_zero(self):
        for expected in (1, 0):
            with tempfile.TemporaryDirectory() as directory:
                self.run_collector(
                    """
counter = metrics.get_kv_transfer_timeout_counter(enabled=True)
counter.labels(stage="bootstrap").inc(int(sys.argv[2]))
""",
                    args=(str(expected),),
                    env={**os.environ, "PROMETHEUS_MULTIPROC_DIR": directory},
                )
                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry, path=directory)
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:kv_transfer_timeouts_total", {"stage": "bootstrap"}
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
