"""A request's sampling seed is honored without --enable-deterministic-inference."""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=65, stage="base-b", runner_config="1-gpu-small")


class TestSeededGeneration(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            # A cached prefix recomputes the last prompt token through another path,
            # which can move its logits enough to flip a near-tie draw.
            other_args=["--disable-radix-cache"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, n):
        # One request at a time, so every draw sees the same batch and logits.
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Write a short story about a lighthouse.",
                "sampling_params": {
                    "temperature": 1.0,
                    "max_new_tokens": 32,
                    "sampling_seed": 1234,
                    "n": n,
                },
            },
        )
        response.raise_for_status()
        body = response.json()
        return [choice["text"] for choice in (body if n > 1 else [body])]

    def test_seed_repeats_and_samples_differ(self):
        self.assertEqual(self._generate(n=1), self._generate(n=1))
        first, second = self._generate(n=2)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
