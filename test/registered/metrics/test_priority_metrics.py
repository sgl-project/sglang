import unittest
from typing import Dict, List

import requests
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=60, suite="stage-b-test-small-1-gpu-amd")

_MODEL_NAME = "Qwen/Qwen3-0.6B"


def _parse_prometheus_metrics(metrics_text: str) -> Dict[str, List[Sample]]:
    result = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name not in result:
                result[sample.name] = []
            result[sample.name].append(sample)
    return result


def _get_samples_by_name(metrics: Dict[str, List[Sample]], name: str) -> List[Sample]:
    return metrics.get(name, [])


def _get_sample_value_by_labels(samples: List[Sample], labels: Dict[str, str]) -> float:
    for sample in samples:
        if all(sample.labels.get(k) == v for k, v in labels.items()):
            return sample.value
    raise KeyError(f"No sample found with labels {labels}")


class TestPriorityMetrics(CustomTestCase):
    """Test that priority-based metrics are correctly emitted when
    --enable-priority-scheduling is enabled."""

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            _MODEL_NAME,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-metrics",
                "--enable-priority-scheduling",
                "--default-priority-value",
                "0",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_priority_label_in_gauge_metrics(self):
        """Send requests with different priorities and verify that
        gauge metrics (num_running_reqs, num_queue_reqs) contain
        the priority label dimension."""

        # Send requests with different priorities to populate metrics
        for priority in [1, 5, 10]:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "Hello",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 5},
                    "priority": priority,
                },
            )
            self.assertEqual(response.status_code, 200)

        # Fetch metrics
        metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics = _parse_prometheus_metrics(metrics_response.text)

        # Verify priority label exists on queue gauge metrics
        for metric_name in ["sglang:num_running_reqs", "sglang:num_queue_reqs"]:
            samples = _get_samples_by_name(metrics, metric_name)
            self.assertGreater(len(samples), 0, f"No samples found for {metric_name}")

            # Should have at least one sample with a non-empty priority label
            # (the total has priority="" and per-priority has priority="<int>")
            priority_labels = {s.labels.get("priority", "") for s in samples}
            self.assertIn(
                "",
                priority_labels,
                f"{metric_name}: missing total (priority='') sample",
            )

    def test_priority_label_in_histogram_metrics(self):
        """Send requests with different priorities and verify that
        histogram metrics (TTFT, ITL, e2e latency) contain the priority label."""

        for priority in [1, 5]:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 20},
                    "priority": priority,
                },
            )
            self.assertEqual(response.status_code, 200)

        metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics = _parse_prometheus_metrics(metrics_response.text)

        # Check histogram metrics have priority label
        histogram_metrics = [
            "sglang:time_to_first_token_seconds",
            "sglang:e2e_request_latency_seconds",
        ]
        for metric_name in histogram_metrics:
            # Histogram metrics are emitted as _sum, _count, _bucket
            sum_name = f"{metric_name}_sum"
            count_name = f"{metric_name}_count"
            for suffix_name in [sum_name, count_name]:
                samples = _get_samples_by_name(metrics, suffix_name)
                if not samples:
                    continue
                # At least one sample should have a non-empty priority label
                priority_values = {s.labels.get("priority", "") for s in samples}
                non_empty = priority_values - {""}
                self.assertGreater(
                    len(non_empty),
                    0,
                    f"{suffix_name}: expected per-priority samples, "
                    f"got priority labels: {priority_values}",
                )

    def test_default_priority_value(self):
        """Requests without explicit priority should use --default-priority-value (0)."""

        # Send request WITHOUT priority — should get default priority 0
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "Hello world",
                "sampling_params": {"temperature": 0, "max_new_tokens": 5},
            },
        )
        self.assertEqual(response.status_code, 200)

        metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics = _parse_prometheus_metrics(metrics_response.text)

        # Check that e2e latency has samples with priority="0" (the default)
        e2e_count = _get_samples_by_name(
            metrics, "sglang:e2e_request_latency_seconds_count"
        )
        priority_values = {s.labels.get("priority", "") for s in e2e_count}
        self.assertIn(
            "0",
            priority_values,
            f"Expected priority='0' from default, got: {priority_values}",
        )


if __name__ == "__main__":
    unittest.main()
