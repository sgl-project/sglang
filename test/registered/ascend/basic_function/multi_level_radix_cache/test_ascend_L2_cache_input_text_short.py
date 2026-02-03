import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

class TestL1Cache(CustomTestCase):
    """Testcase: Test shows that L2 cache is enabled,
    and inference request outputs shorter than the page size will not be reused

    [Test Category] Parameter
    [Test Target] --enable-hierarchical-cache
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.8,
                "--tp-size",
                2,
                "--enable-hierarchical-cache",
                "--base-gpu-id",
                4,
            ]
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_L2_cache(self):
        # with two identical short text input requests, the token will not be reused.
        texts=["who am i?","who am i?"]
        for text in texts:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                        "text": text,
                        "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 10,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(int(response.json()["meta_info"]["cached_tokens"]) == 0)


if __name__ == "__main__":
    unittest.main()
