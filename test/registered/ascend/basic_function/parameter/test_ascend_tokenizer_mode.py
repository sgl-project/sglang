import os
import unittest
from shutil import copy2
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestEnableTokenizerModeSlow(CustomTestCase):
    """
    Testcase：Verify that the inference is successful when tokenizer path is modified and the tokenizer mode is set

    [Test Category] Parameter
    [Test Target] --tokenizer-path; --tokenizer-mode; --tokenizer-worker-num
    """

    tokenizer_mode = "slow"

    @classmethod
    def setUpClass(cls):
        cls.model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.tokenizer_path = "/tmp"
        cls.file_names = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        for file_name in cls.file_names:
            if not os.path.exists(cls.tokenizer_path + "/" + file_name):
                copy2(cls.model_path + "/" + file_name, cls.tokenizer_path)

        cls.tokenizer_worker_num = 4
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tokenizer-mode",
            cls.tokenizer_mode,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tokenizer-path",
            cls.tokenizer_path,
            "--tokenizer-worker-num",
            cls.tokenizer_worker_num,
        ]
        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        for file_name in cls.file_names:
            if os.path.exists(cls.tokenizer_path + "/" + file_name):
                os.remove(cls.tokenizer_path + "/" + file_name)

    def test_tokenzier_mode(self):
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
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        response = requests.get(self.base_url + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tokenizer_path"], self.tokenizer_path)
        self.assertEqual(response.json()["tokenizer_mode"], self.tokenizer_mode)
        self.assertEqual(response.json()["tokenizer_worker_num"], self.tokenizer_worker_num)


class TestEnableTokenizerModeAuto(TestEnableTokenizerModeSlow):
    tokenizer_mode = "auto"


if __name__ == "__main__":
    unittest.main()
