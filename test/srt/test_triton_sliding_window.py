import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


class TestSlidingWindowAttentionTriton(CustomTestCase):
    """Test sliding window attention functionality with triton backend."""

    @classmethod
    def setUpClass(cls):
        """Set up the test server with Gemma3 model and triton backend."""
        # Gemma3 model supports sliding window attention
        cls.model = "google/gemma-3-4b-it"
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "triton",
            "--context-length",
            "8192",
            "--random-seed",
            "42",
        ]

        cls.short_context_prompt = "The capital of France is"

        # Test prompt longer than window size
        cls.long_context_prompt = (
            """
        Once upon a time, there was a mountain. In the mountain, there was a temple. In the temple, there was an old monk telling a story. The story was:
        """
            * 100
        )
        cls.long_context_prompt += "\nNow, summarize the story in one sentence:"

    def _test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=200,
            num_threads=32,
        )

        metrics = run_eval(args)
        print(f"MMLU metrics with sliding window: {metrics}")

        self.assertGreaterEqual(metrics["score"], 0.60)

    def _test_short_context_generation(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": self.short_context_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("paris", result["text"].lower())
        print(f"Short context generation result: {result['text']}")

    def _test_long_context_generation(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": self.long_context_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertGreater(len(result["text"].strip()), 0)
        print(f"Long context generation result: {result['text'][:100]}...")

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_no_cuda_graph(self):
        self.no_cuda_graph_process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.common_args + ["--disable-cuda-graph"],
        )

        try:
            self._test_short_context_generation()
            self._test_long_context_generation()
            self._test_mmlu()
        finally:
            kill_process_tree(self.no_cuda_graph_process.pid)

    def test_cuda_graph(self):
        self.cuda_graph_process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.common_args,
        )

        try:
            self._test_short_context_generation()
            self._test_long_context_generation()
            self._test_mmlu()
        finally:
            kill_process_tree(self.cuda_graph_process.pid)


if __name__ == "__main__":
    unittest.main()
