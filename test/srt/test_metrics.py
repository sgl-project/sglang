import unittest

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestEnableMetrics(unittest.TestCase):
    def test_metrics_enabled(self):
        """Test that metrics endpoint returns data when enabled"""
        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-metrics"],
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

            # Verify essential metrics are present
            essential_metrics = [
                "sglang:num_running_reqs",
                "sglang:token_usage",
                "sglang:gen_throughput",
                "sglang:cache_hit_rate",
                "sglang:func_latency_seconds",
                "sglang:prompt_tokens_total",
                "sglang:generation_tokens_total",
                "sglang:time_to_first_token_seconds",
                "sglang:time_per_output_token_seconds",
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

        finally:
            kill_child_process(process.pid, include_self=True)


if __name__ == "__main__":
    unittest.main()
