import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-2-npu-a3",
    nightly=True,
)


class TestReturnRoutedExperts(CustomTestCase):
    """Testcase: Testing with the 'enable_thinking' feature enabled/disabled,
                 both streaming and non-streaming input requests successful

    [Test Category] Interface
    [Test Target] /v1/chat/completions
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.other_args = [
           "--disable-overlap-schedule",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-return-routed-experts",
            "--mem-fraction-static",
            0.95,
            "--tp",
            2,
            "--dp",
            2,
            "--enable-dp-attention",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        cls.additional_chat_kwargs = {}

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_completion_with_return_routed_experts(self):
        client = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "max_tokens": 100,
                "return_routed_experts": True,
            },
        )

        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        data = client.json()

        self.assertIn("sglext", data)
        self.assertIn("routed_experts", data['sglext'])


if __name__ == "__main__":
    unittest.main()