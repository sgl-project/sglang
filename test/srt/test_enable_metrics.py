import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

TEST_MODEL = DEFAULT_MODEL_NAME_FOR_TEST # I used "google/gemma-2-2b-it" for testing locally
class TestEnableMetrics(unittest.TestCase):
    def test_metrics_enabled(self):
        """Test that metrics endpoint returns data when enabled"""
        # Launch server with metrics enabled
        process = popen_launch_server(
            model=TEST_MODEL,
            base_url=DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            enable_metrics=True
        )

        try:
            # Make a request to generate some metrics
            response = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            # Get metrics
            metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
            self.assertEqual(metrics_response.status_code, 200)
            metrics_content = metrics_response.text

            # Verify essential metrics are present
            essential_metrics = [
                "sglang:prompt_tokens_total",
                "sglang:generation_tokens_total", 
                "sglang:max_total_num_tokens",
                "sglang:context_len",
                "sglang:time_to_first_token_seconds",
                "sglang:time_per_output_token_seconds",
                "sglang:e2e_request_latency_seconds"
            ]
            
            for metric in essential_metrics:
                self.assertIn(metric, metrics_content, f"Missing metric: {metric}")

            # Verify model name label is present and correct
            expected_model_name = TEST_MODEL
            self.assertIn(f'model_name="{expected_model_name}"', metrics_content)
            # Verify metrics have values (not empty)
            self.assertIn("_sum{", metrics_content)
            self.assertIn("_count{", metrics_content)
            self.assertIn("_bucket{", metrics_content)

        finally:
            kill_child_process(process.pid, include_self=True)

    def test_metrics_disabled(self):
        """Test that metrics endpoint returns 404 when disabled"""
        # Launch server with metrics disabled
        process = popen_launch_server(
            model=TEST_MODEL,
            base_url=DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            enable_metrics=False
        )

        try:
            response = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)
            # Verify metrics endpoint is not available
            metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
            self.assertEqual(metrics_response.status_code, 404)

        finally:
            kill_child_process(process.pid, include_self=True)
