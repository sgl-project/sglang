"""Unit tests for disaggregation load metric payload fields."""

import dataclasses
import unittest

from sglang.srt.managers.io_struct import DisaggregationMetrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestDisaggregationMetrics(CustomTestCase):
    def test_kv_transfer_load_fields_are_serializable(self):
        metrics = DisaggregationMetrics(
            mode="prefill",
            kv_transfer_latency_ms=100.0,
            kv_transfer_speed_gb_s=1.5,
            kv_transfer_total_mb=16.0,
            kv_transfer_bootstrap_ms=2.5,
            kv_transfer_alloc_ms=1.0,
        )

        payload = dataclasses.asdict(metrics)

        self.assertEqual(payload["kv_transfer_latency_ms"], 100.0)
        self.assertEqual(payload["kv_transfer_speed_gb_s"], 1.5)
        self.assertEqual(payload["kv_transfer_total_mb"], 16.0)
        self.assertEqual(payload["kv_transfer_bootstrap_ms"], 2.5)
        self.assertEqual(payload["kv_transfer_alloc_ms"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
