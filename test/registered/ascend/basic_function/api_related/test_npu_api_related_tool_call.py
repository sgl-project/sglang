import os
import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_5_27B_MODEL_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestApiRelatedToolCallParserLlama(CustomTestCase):
    """Test --chat-template, indicating successful inference.

    [Test Category] Functional
    [Test Target] Api related on NPU
    --sampling-defaults; --chat-template; --tool-call-parser
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
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--sampling-defaults",
            "openai",
            "--chat-template",
            "llama-4",
            "--tool-call-parser",
            "llama3",
        ]

        # Start the local OpenAI Server. If necessary, you can add other parameters such as --enable-tools.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_function_calling_format(self):
        """
        Test: Whether the function call format returned by the AI is correct.
        When returning a tool call, message.content should be None, and tool_calls should be a list.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

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

        tool_calls = response.choices[0].message.tool_calls
        assert (
            isinstance(tool_calls, list) and len(tool_calls) > 0
        ), "tool_calls should be a non-empty list"

        function_name = tool_calls[0].function.name
        assert function_name == "add", "Function name should be 'add'"


class TestApiRelatedToolCallParserPythonic(CustomTestCase):
    """Test configuration of tool-call-parser as pythonic, sending requests"""

    PYTHONIC_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The name of the city or location.",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_tourist_attractions",
                "description": "Get a list of top tourist attractions for a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city to find attractions for.",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
    ]

    PYTHONIC_MESSAGES = [
        {
            "role": "system",
            "content": (
                "You are a travel assistant. "
                "When asked to call functions, ALWAYS respond ONLY with a python list of function calls, "
                "using this format: [func_name1(param1=value1, param2=value2), func_name2(param=value)]. "
                "Do NOT use JSON, do NOT use variables, do NOT use any other format. "
                "Here is an example:\n"
                '[get_weather(location="Paris"), get_tourist_attractions(city="Paris")]'
            ),
        },
        {
            "role": "user",
            "content": (
                "I'm planning a trip to Tokyo next week. What's the weather like and what are some top tourist attractions? "
                "Propose parallel tool calls at once, using the python list of function calls format as shown above."
            ),
        },
    ]

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        other_args = [
            "--tool-call-parser",
            "pythonic",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_pythonic_tool_call_prompt(self):
        """
        Test: Explicit prompt for pythonic tool call format without chat template.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=self.PYTHONIC_MESSAGES,
            tools=self.PYTHONIC_TOOLS,
            temperature=0.1,
            stream=False,
        )
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsInstance(tool_calls, list, "No tool_calls found")
        self.assertGreaterEqual(len(tool_calls), 1)
        names = [tc.function.name for tc in tool_calls]
        self.assertTrue(
            "get_weather" in names or "get_tourist_attractions" in names,
            f"Function name '{names}' should container either 'get_weather' or 'get_tourist_attractions'",
        )


class TestApiRelatedAdminApiKey(CustomTestCase):
    """Test --admin-api-key and --chat-template, and send requests using the key.

    [Test Category] Functional
    [Test Target] Api related on NPU
    --admin-api-key; --chat-template
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.chat_template_path = f"{QWEN3_5_27B_MODEL_WEIGHTS_PATH}/chat_template.jinja"
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--admin-api-key",
            "123456",
            "--chat-template",
            cls.chat_template_path,
        ]

        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")

        # Start the local OpenAI Server. If necessary, you can add other parameters such as --enable-tools.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def test_function_admin_api_key(self):
        # Successfully invoked the interface using the key.
        response = requests.get(
            f"{DEFAULT_URL_FOR_TEST}/hicache/storage-backend",
            headers={"Authorization": "Bearer 123456"},
        )
        self.assertEqual(response.status_code, 200)

        # Calling the API without a key returns a 401 (Unauthorized) error.
        response1 = requests.get(
            f"{DEFAULT_URL_FOR_TEST}/hicache/storage-backend",
        )
        self.assertEqual(response1.status_code, 401)

    def test_function_chat_template(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/completions",
            headers={"Authorization": "Bearer 123456"},
            json={
                "prompt": "The capital of France is",
                "max_tokens": 128,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        # The chat template used is consistent with the configuration.
        self.assertIn(
            f"Loading chat template from argument: {self.chat_template_path}", content
        )
        self.out_log_file.close()
        self.err_log_file.close()


if __name__ == "__main__":
    unittest.main()
