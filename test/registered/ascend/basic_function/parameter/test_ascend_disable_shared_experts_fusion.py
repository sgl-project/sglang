import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="nightly-16-npu-a3", nightly=True)


class TestDisableSharedExpertsFusion(CustomTestCase):
    """
    Testcase：Verify the inference request is successfully processed when the --disable-shared-experts-fusion parameter is set

    [Test Category] Parameter
    [Test Target] --disable-shared-experts-fusion
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.common_args = [
            "--disable-shared-experts-fusion",
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

        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 6,
            other_args=[
                *cls.common_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_disable_shared_experts_fusion(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        # Verify that the inference request is successfully processed when --disable-shared-experts-fusion parameter is set
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        response = requests.get(self.base_url + "/get_server_info")
        # Verify that --disable-shared-experts-fusion parameter takes effect
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["disable_shared_experts_fusion"], True)


if __name__ == "__main__":
    unittest.main()
