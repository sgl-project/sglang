import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMoreRunnerBackendTriton(CustomTestCase):
    """Testcase：Verify set --moe-runner-backend, the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --moe-runner-backend
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    moe_runner_backend = "triton"

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--moe-runner-backend",
                cls.moe_runner_backend,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_moe_runner_backend(self):
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
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertEqual(
            response.json()["moe_runner_backend"],
            self.moe_runner_backend,
            "--moe-runner-backend is not taking effect.",
        )


class TestMoreRunnerBackendTritonDefault(TestMoreRunnerBackendTriton):
    moe_runner_backend = "auto"

    @classmethod
    def get_server_args(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        return other_args


if __name__ == "__main__":
    unittest.main()
