import os
import unittest
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=50, suite="nightly-1-npu-a3", nightly=True)


class TestEnableProfileCudaGraph(CustomTestCase):
    """
    Testcase：Verify that the --enable-profile-cuda-graph parameter is correctly configured and the inference result is correctly

    [Test Category] Parameter
    [Test Target] --enable-profile-cuda-graph
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.out_log_file_name = "./tmp_out_log.txt"
        cls.err_log_file_name = "./tmp_err_log.txt"
        cls.out_log_file = open(cls.out_log_file_name, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.err_log_file_name, "w+", encoding="utf-8")

        cls.other_args = [
            "--attention-backend",
            "ascend",
            "--enable-profile-cuda-graph",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove(cls.out_log_file_name)
        os.remove(cls.err_log_file_name)

    def test_enable_profile_cuda_graph(self):
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
        self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        self.assertIn("Paris", response.text, "The inference result does not include Paris.")

        response = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        self.assertTrue(
            response.json()["enable_profile_cuda_graph"],
            "--enable-profile-cuda-graph is not taking effect.",
        )

        self.out_log_file.seek(0)
        content = self.out_log_file.read()
        self.assertTrue(len(content) > 0)
        self.assertIn("profiler.py: Start parsing profiling data:", content)


if __name__ == "__main__":
    unittest.main()
