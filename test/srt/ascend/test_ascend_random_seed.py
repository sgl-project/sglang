import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestRandomSeed(CustomTestCase):
    random_seed = 0

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--random-seed",
                cls.random_seed,
            ]
            if is_npu()
            else ["--random-seed", cls.random_seed]
        )
        return other_args

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
                if is_npu()
                else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_random_seed(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        print(response.text)
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        print(response.json())
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIsInstance(
            response.json()["random_seed"], int, "--random-seed is not taking effect."
        )


class TestRandomSeedOne(TestRandomSeed):
    random_seed = 1


if __name__ == "__main__":
    unittest.main()
