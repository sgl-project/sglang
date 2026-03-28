"""
python3 -m unittest test.srt.openai_server.function_call.test_paralell_tool_calls
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestParallelToolCall(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--tool-call-parser",
            "qwen",
            "--reasoning-parser",
            "qwen3",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.weather_tool = {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Retrieve the current weather for a given city and state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name, e.g. 'San Francisco'.",
                        },
                        "state": {
                            "type": "string",
                            "description": "Two-letter state abbreviation, e.g. 'CA'.",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit.",
                            "enum": ["CELSIUS", "FAHRENHEIT"],
                        },
                    },
                    "required": ["city", "state", "unit"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        cls.name_selection_tool = {
            "type": "function",
            "function": {
                "name": "select_name",
                "description": "Select a name and provide its associated age.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": ["NAME_A", "NAME_B"]},
                        "age": {
                            "type": "integer",
                            "description": "Age between 0 and 23.",
                            "minimum": 0,
                            "maximum": 23,
                        },
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_parallel_tool_calls_required(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")
        prompt = (
            "Please call both tools. My name is NAME_B and I am 20 years old. "
            "Tell me the weather in Boston, MA, using Celsius. "
            "Please note: you must call both tools!"
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],  # parallel_tool_calls is True by default
            tool_choice="required",
            tools=[self.name_selection_tool, self.weather_tool],
        )
        self.assertEqual(len(response.choices[0].message.tool_calls), 2)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            tool_choice="required",
            tools=[self.name_selection_tool, self.weather_tool],
            parallel_tool_calls=False,
        )
        self.assertEqual(len(response.choices[0].message.tool_calls), 1)

    def test_parallel_tool_calls_auto(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")
        prompt = (
            "Please call both tools. My name is NAME_B and I am 20 years old. "
            "Tell me the weather in Boston, MA, using Celsius. "
            "Please note: you must call both tools!"
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],  # parallel_tool_calls is True by default
            tool_choice="auto",
            tools=[self.name_selection_tool, self.weather_tool],
        )
        self.assertEqual(len(response.choices[0].message.tool_calls), 2)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            tool_choice="auto",
            tools=[self.name_selection_tool, self.weather_tool],
            parallel_tool_calls=False,
        )
        self.assertEqual(len(response.choices[0].message.tool_calls), 1)


if __name__ == "__main__":
    unittest.main()
