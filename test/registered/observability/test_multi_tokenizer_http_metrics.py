"""HTTP response-tracking metrics must be emitted in multi-tokenizer
(``--tokenizer-worker-num`` > 1) mode.

In multi-tokenizer mode each worker process re-imports the FastAPI ``app``, so
the response-tracking middleware must be registered per worker from the app
lifespan — middleware registered only on the parent process's ``app`` never
reaches the workers that actually serve requests, leaving
``sglang:http_responses_total`` / ``sglang:http_requests_total`` unemitted. This
test guards that those metrics are emitted with more than one tokenizer worker.
"""

import unittest

import requests
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Backend-independent middleware behavior: CUDA-only is enough coverage.
register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


class TestMultiTokenizerHttpMetrics(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-metrics",
                # > 1 forces multi-tokenizer (Granian multi-worker) mode, the path
                # that previously dropped the HTTP response-tracking middleware.
                "--tokenizer-worker-num",
                2,
                "--mem-fraction-static",
                0.7,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _scrape_http_response_samples(self):
        metrics_text = requests.get(f"{self.base_url}/metrics", timeout=30).text
        samples = []
        for family in text_string_to_metric_families(metrics_text):
            for sample in family.samples:
                if sample.name == "sglang:http_responses_total":
                    samples.append(sample)
        return metrics_text, samples

    def test_http_responses_metric_emitted_in_multi_tokenizer_mode(self):
        # Drive a request through the HTTP path so the middleware records it.
        gen = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=60,
        )
        self.assertEqual(gen.status_code, 200)

        metrics_text, samples = self._scrape_http_response_samples()

        # The whole family must be present (it was entirely absent before the fix).
        self.assertIn(
            "sglang:http_responses_total",
            metrics_text,
            "sglang:http_responses_total missing — response-tracking middleware "
            "was not registered in multi-tokenizer mode.",
        )

        # And it must have actually counted the successful /generate response.
        generate_ok = [
            s
            for s in samples
            if s.labels.get("endpoint") == "/generate"
            and s.labels.get("status_code") == "200"
        ]
        self.assertTrue(
            generate_ok,
            f"No /generate 200 sample in http_responses_total; samples={samples!r}",
        )
        self.assertGreater(generate_ok[0].value, 0)


if __name__ == "__main__":
    unittest.main()
