import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestDisableCudaGraphPadding(CustomTestCase):
    """Testcase: Verify that both fixed-length and variable-length requests generate correct results with --disable-cuda-graph-padding enabled.

    [Test Category] Parameter Validation
    [Test Target] --disable-cuda-graph-padding
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph-padding",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate_without_padding(self):
        """Test text generation without padding (CUDA Graph should work normally)"""
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                },
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Paris", resp.text)

    def test_generate_with_padding(self):
        """Test text generation with variable-length inputs (padding required)"""
        prompts = [
            "A",
            "Hello world",
            "What is the capital of France?",
            "This is a longer test prompt to ensure padding is needed in batch",
        ]

        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 8,
                },
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Paris", resp.text)


if __name__ == "__main__":
    unittest.main()
