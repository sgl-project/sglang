import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import QueueCount

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _calculate_utilization(
    disaggregation_mode, num_running_reqs, token_usage, max_running_requests=32
):
    reporter = SimpleNamespace(
        scheduler=SimpleNamespace(
            disaggregation_mode=disaggregation_mode,
            max_running_requests=max_running_requests,
        ),
        stats=SimpleNamespace(
            utilization=0.0,
            num_running_reqs=QueueCount(total=num_running_reqs),
            token_usage=token_usage,
        ),
    )
    SchedulerMetricsReporter._calculate_utilization(reporter)
    return reporter.stats.utilization


class TestMetricsReporterUtilization(CustomTestCase):
    def test_decode_utilization_tracks_load(self):
        """sglang:utilization stayed at 0 under load because the capacity it
        divides by had no open-source setter; it must follow the more
        saturated of the request slots and the KV tokens."""
        self.assertAlmostEqual(
            _calculate_utilization(DisaggregationMode.NULL, 8, token_usage=0.1),
            8 / 32,
        )
        self.assertAlmostEqual(
            _calculate_utilization(DisaggregationMode.DECODE, 8, token_usage=0.45),
            0.45 / 0.9,
        )

    def test_prefill_utilization_is_not_reported(self):
        self.assertEqual(
            _calculate_utilization(DisaggregationMode.PREFILL, 8, token_usage=0.9),
            -1,
        )


if __name__ == "__main__":
    unittest.main()
