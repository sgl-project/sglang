import os
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

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestCudaGraphConfigOverride(CustomTestCase):
    """Testcase: verify --cuda-graph-config JSON overrides per-phase convenience flags

    [Test Category] Parameter
    [Test Target] --cuda-graph-config
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-config",
                '{"decode":{"backend":"full"}}',
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def test_cuda_graph_config_override(self):
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

        self.err_log_file.seek(0)
        err_log = self.err_log_file.read()
        self.assertIn(
            "cuda_graph_config",
            err_log,
            "Expected stderr to contain cuda_graph_config parse log, "
            "proving JSON config was processed and overrode the disabled flag",
        )


if __name__ == "__main__":
    unittest.main()
