import json
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


class TestPyTorchSamplingBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--sampling-backend", "pytorch"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.65

    def test_greedy(self):

        first_text = None

        # ensure the answer is identical across single response
        for _ in range(5):
            response_single = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            ).json()
            text = response_single["text"]
            if first_text is None:
                first_text = text

            assert text == first_text, f'"{text}" is not identical to "{first_text}"'

        first_text = None

        response_batch = requests.post(
            self.base_url + "/generate",
            json={
                "text": ["The capital of France is"] * 10,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        ).json()

        # ensure the answer is identical among the batch
        for i in range(10):
            text = response_batch[i]["text"]
            if first_text is None:
                first_text = text
            assert text == first_text, f'"{text}" is not identical to "{first_text}"'


if __name__ == "__main__":
    unittest.main()
