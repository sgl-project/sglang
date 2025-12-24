import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDPAttentionMetrics(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with (
            envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.override(True),
            envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.override(True),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp",
                    "1",
                    "--dp",
                    "2",
                    "--enable-dp-attention",
                    "--enable-metrics",
                    "--cuda-graph-max-bs",
                    "2",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_dp_attention_metrics(self):
        response = requests.get(f"{self.base_url}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{self.base_url}/generate",
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

        metrics_response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics_content = metrics_response.text

        print(f"metrics_content=\n{metrics_content}")

        essential_metrics = [
            "sglang:realtime_tokens_total",
            "sglang:gpu_execution_seconds_total",
            "sglang:dp_cooperation_realtime_tokens_total",
            "sglang:dp_cooperation_gpu_execution_seconds_total",
        ]

        for metric in essential_metrics:
            self.assertIn(metric, metrics_content, f"Missing metric: {metric}")

        self.assertIn('mode="prefill_compute"', metrics_content)
        self.assertIn('mode="decode"', metrics_content)
        self.assertIn('category="forward_prefill"', metrics_content)
        self.assertIn('category="forward_decode"', metrics_content)
        self.assertIn("num_prefill_ranks=", metrics_content)


if __name__ == "__main__":
    unittest.main()

