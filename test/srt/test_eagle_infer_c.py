import time
from types import SimpleNamespace
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST_EAGLE3,
    DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

class TestEAGLE3Server(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST_EAGLE3,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                6,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--cuda-graph-max-bs",
                5,
                "--tp",
                4,
                "--dp",
                4,
                "--enable-dp-attention",
                "--dtype",
                "float16",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.74)

        server_info = requests.get(self.base_url + "/get_server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        self.assertGreater(avg_spec_accept_length, 1.65)

        # Wait a little bit so that the memory check happens.
        time.sleep(4)

class TestEAGLE3Server2(TestEAGLE3Server):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST_EAGLE3,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                6,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--cuda-graph-max-bs",
                5,
                "--tp",
                4,
                "--dp",
                2,
                "--enable-dp-attention",
                "--dtype",
                "float16",
            ],
        )

class TestEAGLE3Server3(TestEAGLE3Server):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST_EAGLE3,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                6,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--cuda-graph-max-bs",
                5,
                "--tp",
                4,
                "--dp",
                2,
                "--ep",
                4,
                "--enable-dp-attention",
                "--dtype",
                "float16",
            ],
        )

if __name__ == "__main__":
    unittest.main()
