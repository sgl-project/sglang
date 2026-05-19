import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestEnableReturnRoutedExperts(CustomTestCase):
    """
    Testcase：When the service startup configuration --enable-return-routed-experts is enabled and the request sets
    return_routed_experts to true, the response body will contain "routed_experts" information.

    [Test Category] Parameter
    [Test Target] --enable-return-routed-experts
    """

    model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
        "--disable-radix-cache",
        "--enable-return-routed-experts",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3000,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_enable_return_routed_experts(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        text1 = "The capital of France is"

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text1,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "return_routed_experts": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        print(response.json()["meta_info"])
        self.assertIn("routed_experts", response.json()["meta_info"])


if __name__ == "__main__":
    unittest.main()
