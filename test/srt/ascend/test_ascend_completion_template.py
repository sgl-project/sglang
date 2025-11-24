import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestCompletionTemplate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_npu():
            cls.model = (
                "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V2-Lite-W8A8"
            )
        else:
            cls.model = "deepseek-ai/deepseek-coder-1.3b-base"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            [
                "--completion-template",
                "deepseek_coder",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--quantization",
                "w8a8_int8",
                "--mem-fraction-static",
                0.6,
            ]
            if is_npu()
            else ["--completion-template", "deepseek_coder"]
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_completion_template(self):
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


if __name__ == "__main__":
    unittest.main()
