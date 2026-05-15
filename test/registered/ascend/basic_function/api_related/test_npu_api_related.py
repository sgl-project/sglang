import os
import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import C4AI_COMMAND_R_V01_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestNpuApiRelated(CustomTestCase):
    """The API test with combined parameters returned the correct values for model_name and weight_version, indicating successful inference.

    [Test Category] Functional
    [Test Target] Api related on NPU
    --served-model-name; --weight-version; --hf-chat-template-name; --enable-cache-report
    """

    SYSTEM_MESSAGE = (
        "You are a helpful assistant with tool calling capabilities. "
        "Only reply with a tool call if the function exists in the library provided by the user. "
        "If it doesn't exist, just reply directly in natural language. "
        "When you receive a tool call response, use the output to format an answer to the original user question. "
        "You have access to the following functions. "
        "To call a function, please respond with JSON for a function call. "
        'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. '
        "Do not use variables.\n\n"
    )

    @classmethod
    def setUpClass(cls):
        cls.model = C4AI_COMMAND_R_V01_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.custom_model_name = "Llama3.2"
        cls.weight_version = "v1.0.0"
        cls.hf_chat_template_name = "tool_use"
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--tp-size",
            2,
            "--disable-cuda-graph",
            "--served-model-name",
            cls.custom_model_name,
            "--weight-version",
            cls.weight_version,
            "--hf-chat-template-name",
            cls.hf_chat_template_name,
            "--enable-cache-report",
        ]

        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )
        cls.base_url_v1 = cls.base_url + "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def test_served_model_weight_version(self):
        # Verify the weight version identifier and the served-model-name covered model name.
        response = requests.get(
            f"{self.base_url}/v1/models",
            headers={"Authorization": "Bearer sk-123456"},
        )
        result = response.json()

        self.assertIn("data", result)
        self.assertEqual(result["data"][0]["id"], self.custom_model_name)

        response1 = requests.get(
            f"{self.base_url}/model_info",
            headers={"Authorization": "Bearer sk-123456"},
        )
        self.assertEqual(response1.json()["weight_version"], self.weight_version)

        # Verify that the hf-chat-template-name configuration is effective, including the configuration value tool_use.
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn("Using specified chat template: 'tool_use'", content)
        self.out_log_file.close()
        self.err_log_file.close()

    def test_chat_template_request(self):
        """Send inference request"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url_v1)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Compute the sum of two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "integer",
                                "description": "A number",
                            },
                            "b": {
                                "type": "integer",
                                "description": "A number",
                            },
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": "Compute (3+5)"},
        ]
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
        )
        reason = response.choices[0].finish_reason
        self.assertEqual(reason, "stop")

    def test_cache_report(self):
        """Return number of cached tokens in prompt_tokens_details for each openai request."""
        for i in range(2):
            response = requests.post(
                f"{self.base_url}/v1/completions",
                headers={"Authorization": "Bearer sk-123456"},
                json={
                    "prompt": "just return me a string with of 5000 characters, " * 24,
                    "max_tokens": 260,
                },
            )
            self.assertEqual(response.status_code, 200)
            if i == 1:
                cached_tokens = response.json()["usage"]["prompt_tokens_details"][
                    "cached_tokens"
                ]
                self.assertEqual(256, cached_tokens)


if __name__ == "__main__":
    unittest.main()
