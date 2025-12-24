import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    is_in_ci,
)

from python.sglang.srt.environ import envs


class TestEnableMetrics(CustomTestCase):
    def test_metrics_1gpu(self):
        """Test that metrics endpoint returns data when enabled"""

        def _verify_metrics(metrics_content):
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

        self._execute_core(
            other_args=[],
            verify_metrics=_verify_metrics,
        )

    def test_metrics_2gpu(self):
        # TODO enable when we have 2-gpu runner in nightly CI
        if is_in_ci():
            print("Skip test_metrics_2gpu since in 1-gpu CI")
            return

        def _verify_metrics(metrics_content):
            TODO

        with (
            envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.override(True),
            envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.override(True),
        ):
            self._execute_core(
                other_args=["--tp", "2", "--dp", "2", "--enable-dp-attention"],
                verify_metrics=_verify_metrics,
            )

    def _execute_core(self, other_args, verify_metrics):
        process = popen_launch_server(
            "Qwen/Qwen3-0.6B",
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-metrics", "--cuda-graph-max-bs", 2, *other_args],
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

            verify_metrics(metrics_content)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
