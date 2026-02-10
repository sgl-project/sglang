import unittest
from abc import ABC

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH, \
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestLoraBackend(ABC):
    """Testcase: Test configuration of lora-backend parameters, and inference request successful.

    [Test Category] Parameter
    [Test Target] --lora-backend
    """

    lora = "triton"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--lora-backend",
            f"{cls.lora}",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--lora-path",
            f"tool_calling={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_backend(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

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
        response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lora_backend"], f"{self.lora}")


class TestLoraBackendCsgmv(TestLoraBackend, CustomTestCase):
    lora = "csgmv"


class TestLoraBackendAscend(TestLoraBackend, CustomTestCase):
    lora = "ascend"


class TestLoraBackendTorchNative(TestLoraBackend, CustomTestCase):
    lora = "torch_native"


class TestLoraBackendTorchTriton(TestLoraBackend, CustomTestCase):
    lora = "triton"


if __name__ == "__main__":
    unittest.main()
