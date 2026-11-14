"""Unit tests for the gRPC HTTP sidecar."""

import unittest

from prometheus_client import CollectorRegistry, Gauge

from sglang.srt.entrypoints.grpc_server import _encode_metrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGrpcMetricsSerialization(CustomTestCase):
    def test_colon_delimited_metric_name_is_preserved(self):
        registry = CollectorRegistry()
        Gauge(
            "sglang:num_running_reqs",
            "The number of running requests.",
            registry=registry,
        ).set(1)

        accept_headers = (
            "*/*",
            "application/openmetrics-text; version=1.0.0; escaping=allow-utf-8",
        )
        for accept_header in accept_headers:
            with self.subTest(accept_header=accept_header):
                data, _ = _encode_metrics(registry, accept_header)
                output = data.decode("utf-8")

                self.assertIn("# HELP sglang:num_running_reqs ", output)
                self.assertIn("# TYPE sglang:num_running_reqs gauge", output)
                self.assertIn("\nsglang:num_running_reqs 1.0\n", output)
                self.assertNotIn("sglang_num_running_reqs", output)


if __name__ == "__main__":
    unittest.main()
