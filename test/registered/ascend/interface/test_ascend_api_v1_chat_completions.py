import json
import requests
import unittest
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestChatCompletionsInterface(CustomTestCase):
    """Testcase: The test is to verify whether the functions of each parameter of the v1/chat/completions interface are normal.

    [Test Category] Interface
    [Test Target] v1/chat/completions
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.other_args = [
            "--reasoning-parser",
            "qwen3",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--tp-size",
            2,
            "--enable-return-hidden-states",
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

    def test_model_and_messages(self):
        # Test model and messages parameter; configured model returns correct name, unconfigured defaults to "default", reasoning works
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        self.assertEqual(data["model"], self.model)
        self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        self.assertEqual(data["model"], "default")
        self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

    def test_max_completion_tokens(self):
        # Test max_completion_tokens parameter; setting to 1 token forces immediate truncation, verify finish_reason is "length"
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_completion_tokens": 1,
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")

    def test_stream(self):
        # Test stream parameter; verify streaming response contains both reasoning_content and normal content chunks
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        has_reasoning = False
        has_content = False

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})

                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            has_reasoning = True

                        if "content" in delta and delta["content"]:
                            has_content = True

        self.assertTrue(
            has_reasoning,
            "The reasoning content is not included in the stream response",
        )
        self.assertTrue(
            has_content, "The stream response does not contain normal content"
        )

    def test_temperature(self):
        # Test temperature parameter; temperature=0 yields identical outputs across requests, temperature=2 yields varied outputs
        response1 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "temperature": 0,
            },
        )
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        content1 = response1.json()["choices"][0]["message"]["content"]

        response2 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "temperature": 0,
            },
        )
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")
        content2 = response2.json()["choices"][0]["message"]["content"]
        self.assertEqual(content1, content2)

        response3 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "temperature": 2,
            },
        )
        self.assertEqual(response3.status_code, 200, f"Failed with: {response3.text}")
        content3 = response3.json()["choices"][0]["message"]["content"]

        response4 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "temperature": 2,
            },
        )
        self.assertEqual(response4.status_code, 200, f"Failed with: {response4.text}")
        content4 = response4.json()["choices"][0]["message"]["content"]
        self.assertNotEqual(content3, content4)

    def test_return_hidden_states(self):
        # Test return_hidden_states parameter; verify hidden_states field appears when enabled and is absent when disabled
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "return_hidden_states": True,
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertIn("hidden_states", response.json()["choices"][0])

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertNotIn("hidden_states", response.json()["choices"][0])

    def test_top_k(self):
        # Test top_k parameter; with k=20, outputs vary between identical requests due to token sampling
        response1 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "top_k": 20,
            },
        )
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        content1 = response1.json()["choices"][0]["message"]["content"]

        response2 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "top_k": 20,
            },
        )
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")
        content2 = response2.json()["choices"][0]["message"]["content"]
        self.assertNotEqual(content1, content2)

    def test_stop_token_ids(self):
        # Test stop_token_ids parameter; verify response stops at specified token ID (13) and matched_stop field is correct
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stop_token_ids": [1, 13],
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()['choices'][0]['matched_stop'], 13)

    def test_rid(self):
        # Test rid parameter; verify response ID matches the requested rid value 'sssss'
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "rid": "sssss",
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()['id'], 'sssss')


if __name__ == "__main__":
    unittest.main()
