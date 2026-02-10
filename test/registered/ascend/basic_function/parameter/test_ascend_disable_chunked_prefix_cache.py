import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestDisableChunkedPrefixCache(CustomTestCase):
    """Testcase: Verify that inference requests are processed successfully when the chunked prefix cache is disabled.

    [Test Category] Parameter
    [Test Target] --disable-chunked-prefix-cache
    """

    def setUpClass(cls):
        other_args = (
            [
                "--disable-chunked-prefix-cache",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.9",
                "--quantization",
                "modelslim",
                "--tp-size",
                "16",
            ]
        )
        cls.process = popen_launch_server(
            DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=3600,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chunk_prefix_cache(self):
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
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()
