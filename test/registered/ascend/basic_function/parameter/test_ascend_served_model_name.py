import unittest
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

register_npu_ci(est_time=50, suite="nightly-1-npu-a3", nightly=True)


class TestEnableTokenizerMode(CustomTestCase):
    """
    Testcase：Verify that the --served-model-name parameter is used to override the model name returned by the v1/models
    endpoint in OpenAI API server

    [Test Category] Parameter
    [Test Target] --served-model-name
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.served_model_name = "Llama3.2"
        other_args = [
            "--served-model-name",
            cls.served_model_name,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
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
        # Verify that the inference request is successfully processed when --served-model-name parameter is set
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        response = requests.get(self.base_url + "/get_server_info")
        # Verify that the model name is override by setting --served-model-name parameter
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["served_model_name"], self.served_model_name)


if __name__ == "__main__":
    unittest.main()
