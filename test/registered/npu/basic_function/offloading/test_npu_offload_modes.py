import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=800, suite="nightly-2-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH,
}


class TestAscendOffloadModes(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.common_args = [
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.9,
            "--attention-backend",
            "ascend",
            "--offload-group-size",
            4,
            "--offload-num-in-group",
            1,
            "--offload-prefetch-step",
            1,
            "--dp-size",
            2,
        ]

    def run_a_test(self, offload_mode, additional_args=None):
        """Run test for a specific offload mode."""
        for model in self.models:
            with self.subTest(model=model, offload_mode=offload_mode):
                print(f"##=== Testing {offload_mode} offload: {model} ===##")

                args = [
                    *self.common_args,
                    "--offload-mode",
                    offload_mode,
                ]

                if additional_args:
                    args.extend(additional_args)

                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=args,
                )

                try:
                    # Check if server is running (basic functionality test)
                    response = requests.post(
                        f"{DEFAULT_URL_FOR_TEST}/generate",
                        json={
                            "text": "Where is the capital of France?",
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 32,
                            },
                        },
                    )
                    self.assertEqual(
                        response.status_code,
                        200,
                        f"The request status code is not 200, server failed to respond for {offload_mode}",
                    )
                    self.assertIn(
                        "Paris",
                        response.text,
                        f"The inference result does not include Paris, server failed to respond for {offload_mode}",
                    )
                finally:
                    kill_process_tree(process.pid)

    def test_offload_mode_cpu(self):
        """Test offload mode: cpu"""
        self.run_a_test("cpu")

    def test_offload_mode_sharded_gpu(self):
        """Test offload mode: sharded_gpu"""
        self.run_a_test("sharded_gpu")


if __name__ == "__main__":
    unittest.main()
