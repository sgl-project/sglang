import unittest
from typing import Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEnableMetrics(CustomTestCase):
    def _collect_metrics_content(self, extra_args: Optional[list[str]] = None) -> str:
        other_args = ["--enable-metrics", "--cuda-graph-max-bs", "2"]
        if extra_args:
            other_args.extend(str(arg) for arg in extra_args)

        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            # Make some requests to generate some metrics
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                    "stream": True,
                },
                stream=True,
            )
            for _ in response.iter_lines(decode_unicode=False):
                pass

            # Get metrics
            metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
            self.assertEqual(metrics_response.status_code, 200)
            metrics_content = metrics_response.text

            print(f"metrics_content=\n{metrics_content}")
            return metrics_content
        finally:
            kill_process_tree(process.pid)

    def _assert_metrics_content(
        self, metrics_content: str, expect_extra_metric_labels: bool
    ) -> None:
        """Validate common expectations for the metrics payload."""
        # Verify essential metrics are present
        essential_metrics = [
            "sglang:num_running_reqs",
            "sglang:num_used_tokens",
            "sglang:token_usage",
            "sglang:gen_throughput",
            "sglang:num_queue_reqs",
            "sglang:num_grammar_queue_reqs",
            "sglang:cache_hit_rate",
            "sglang:spec_accept_length",
            "sglang:prompt_tokens_total",
            "sglang:generation_tokens_total",
            "sglang:cached_tokens_total",
            "sglang:num_requests_total",
            "sglang:num_requests_received_total",
            "sglang:time_to_first_token_seconds",
            "sglang:inter_token_latency_seconds",
            "sglang:e2e_request_latency_seconds",
        ]

        for metric in essential_metrics:
            self.assertIn(metric, metrics_content, f"Missing metric: {metric}")

        # Verify model name label is present and correct
        expected_model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        self.assertIn(f'model_name="{expected_model_name}"', metrics_content)

        # Verify metrics have values (not empty)
        self.assertIn("_sum{", metrics_content)
        self.assertIn("_count{", metrics_content)
        self.assertIn("_bucket{", metrics_content)

        # Check that metric labels enabled by server launch args are present.
        if expect_extra_metric_labels:
            self.assertRegex(
                metrics_content,
                r'sglang:num_requests_received_total\{[^\n}]*request_type="/generate"',
            )
            self.assertRegex(
                metrics_content,
                r'sglang:num_requests_received_total\{[^\n}]*http_status="200"',
            )
        else:
            self.assertNotIn('request_type="', metrics_content)
            self.assertNotIn('http_status="', metrics_content)

    def test_metrics_enabled(self):
        """Test that metrics endpoint returns data when enabled with default labels."""
        extra_args: list[str] = []
        metrics_content = self._collect_metrics_content(extra_args or None)
        self._assert_metrics_content(metrics_content, expect_extra_metric_labels=False)

    def test_metrics_enabled_with_metric_labels(self):
        """Test that metrics include extra labels when the server enables them."""
        extra_args = ["--metrics-label-request-type", "--metrics-label-http-status"]
        metrics_content = self._collect_metrics_content(extra_args)
        self._assert_metrics_content(metrics_content, expect_extra_metric_labels=True)


if __name__ == "__main__":
    unittest.main()
