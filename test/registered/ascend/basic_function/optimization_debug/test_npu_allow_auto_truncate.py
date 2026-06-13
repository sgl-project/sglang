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


class TestAllowAutoTruncate(CustomTestCase):
    """Testcase：Verify set --allow-auto-truncate parameter, request exceeding the service's context-length setting is sent without excessive length error
    and inference request is successful.

        [Test Category] Parameter
        [Test Target] --allow-auto-truncate
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    allow_auto_truncate = True

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--context-length",
            "1000",
        ]
        if cls.allow_auto_truncate:
            other_args.append("--allow-auto-truncate")
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def send_long_text_request(self):
        text = "hello " * 1200
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        return response

    def test_allow_auto_truncate(self):
        response = self.send_long_text_request()
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertNotIn("is longer than the model's context length", response.text)


class TestNoAllowAutoTruncate(TestAllowAutoTruncate):
    """
    Verify --allow-auto-truncate = False over --context-length request is rejected
    """

    allow_auto_truncate = False

    def test_allow_auto_truncate(self):
        response = self.send_long_text_request()
        self.assertEqual(
            response.status_code, 400, "The request status code is not 400."
        )
        self.assertIn("is longer than the model's context length", str(response.json()))


if __name__ == "__main__":
    unittest.main()
