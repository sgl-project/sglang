import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestAscendL2CacheReused(CustomTestCase):
    """Testcase: Test enabled L2 cache (enable-hierarchical-cache) inference request reuse succeddfully
    [Test Category] Parameter
    [Test Target] --enable-hierarchical-cache
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        common_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            0.8,
            "--tp-size",
            2,
        ]
        other_args = common_args + (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_L2_cache_resused(self):
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?What is The capital of France?"
                    "What is The capital of France?What is The capital of France?",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 10,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            # If the same request is made, the token will be reused.
            # cached_tokens: Number of tokens cached in KV Cache
            if i == 0:
                self.assertEqual(int(response.json()["meta_info"]["cached_tokens"]), 0)
            else:
                self.assertGreater(
                    int(response.json()["meta_info"]["cached_tokens"]), 0
                )


if __name__ == "__main__":
    unittest.main()
