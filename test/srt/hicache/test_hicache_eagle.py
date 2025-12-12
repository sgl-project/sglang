import unittest
from types import SimpleNamespace

import requests

from sglang.bench_serving import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST_EAGLE3,
    DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestHiCacheEagle(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST_EAGLE3
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = get_tokenizer(cls.model)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                1.2,
                "--mem-fraction-static",
                0.7,
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
                "--speculative-num-steps",
                2,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                3,
                "--dtype",
                "float16",
                "--chunked-prefill-size",
                1024,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.72)

        server_info = requests.get(self.base_url + "/get_server_info")
        print(f"{server_info=}")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 2.26)


if __name__ == "__main__":
    unittest.main()
