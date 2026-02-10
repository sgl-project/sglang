import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import run_command
from sglang.test.ascend.test_ascend_utils import LLAMA_2_7B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestNcclPort(CustomTestCase):
    """Testcase: Test the basic functions of nccl-port
                 Test nccl-port configured, the inference request successful.

    [Test Category] Parameter
    [Test Target] --nccl-port
    """

    model = LLAMA_2_7B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--nccl-port",
            "9111",
            "--tp-size",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
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

    def test_nccl_port(self):
        """Test the --nccl-port argument."""
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
        result = run_command("lsof -i:9111")
        self.assertIn("*:9111 (LISTEN)", result)


if __name__ == "__main__":
    unittest.main()
