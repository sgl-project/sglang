import unittest

import requests
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestEnableCacheReport(CustomTestCase):
    """Testcase：Verify set --enable-cache-report, sent openai request prompt_tokens_details will return cached_tokens.

       [Test Category] Parameter
       [Test Target] --enable-cache-report
       """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-cache-report",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-hierarchical-cache",
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_enable_cache_report(self):
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/completions",
                json={
                    "prompt": "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, ",
                    "max_tokens": 260,

                },
            )
            self.assertEqual(response.status_code, 200)
            if i == 1:
                cached_tokens = response.json()["usage"]['prompt_tokens_details']['cached_tokens']
                self.assertEqual(256, cached_tokens)


if __name__ == "__main__":
    unittest.main()
