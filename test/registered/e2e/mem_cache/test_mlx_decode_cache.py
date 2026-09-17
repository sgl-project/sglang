"""Chained MLX decode must publish its generated prefix to the radix cache."""

import unittest

import requests

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_mlx_ci(est_time=40, suite="stage-b-e2e-mlx")


class TestMlxDecodeCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            try_cached_model("mlx-community/Qwen3-0.6B-4bit"),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--max-total-tokens",
                "1024",
                "--context-length",
                "512",
                "--max-running-requests",
                "4",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--nccl-port",
                "29500",
            ],
            env={"SGLANG_USE_MLX": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def _generate(self, input_ids, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        self.assertEqual(result["meta_info"]["completion_tokens"], max_new_tokens)
        return result

    def test_generated_prefix_is_reused_without_changing_output(self):
        prompt = list(range(1, 9))
        first = self._generate(prompt, max_new_tokens=32)
        continuation = prompt + first["output_ids"]
        cached = self._generate(continuation, max_new_tokens=8)
        # The last input token is reserved for computing the next-token logits.
        self.assertEqual(cached["meta_info"]["cached_tokens"], len(continuation) - 1)

        response = requests.post(self.base_url + "/flush_cache", timeout=10)
        response.raise_for_status()
        fresh = self._generate(continuation, max_new_tokens=8)
        self.assertEqual(fresh["meta_info"]["cached_tokens"], 0)
        self.assertEqual(cached["output_ids"], fresh["output_ids"])


if __name__ == "__main__":
    unittest.main()
