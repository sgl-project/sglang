"""
Test script for tool_choice functionality through router with gRPC worker.

This test suite validates tool_choice behavior through the router → gRPC worker
pathway to catch potential issues at streaming boundaries and transport conversions
(e.g., partial JSON frames, interleaving, retries, backpressure).

Focuses on critical tool choice scenarios.
"""

import json
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_router_with_grpc_worker,
)


class TestToolChoiceLlama32GRPCRouter(CustomTestCase):
    """Test tool_choice functionality through router with gRPC worker."""

    @classmethod
    def setUpClass(cls):
        # Mark flaky tests for this model
        cls.flaky_tests = {
            "test_multi_tool_scenario_auto",
            "test_multi_tool_scenario_required",
        }

        cls.model = "meta-llama/Llama-3.2-1B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Launch router with single gRPC worker
        # Note: This can take 5+ minutes due to router connection time
        print("Launching router with gRPC worker (may take 5+ minutes)...")
        cls.router_process, cls.worker_process, cls.worker_port = (
            popen_launch_router_with_grpc_worker(
                cls.model,
                cls.base_url,
                timeout=600,  # 10 minutes for router + worker startup
                api_key=cls.api_key,
                worker_other_args=[
                    "--tool-call-parser",
                    "llama3",
                ],
                router_other_args=[],
            )
        )

        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        # Clean up both router and worker processes
        kill_process_tree(cls.router_process.pid)
        kill_process_tree(cls.worker_process.pid)

    def setUp(self):
        self.client = openai.Client(base_url=self.base_url, api_key=self.api_key)
        self.model_name = self.client.models.list().data[0].id

    def get_test_tools(self):
        """Get the test tools for function calling"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "use this to get latest weather information for a city given its name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "name of the city to get weather for",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pokemon_info",
                    "description": "get detailed information about a pokemon given its name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "name of the pokemon to get info for",
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
        ]

    def get_test_messages(self):
        """Get test messages that should trigger tool usage"""
        return [
            {
                "role": "user",
                "content": "What is the weather in Tokyo?",
            }
        ]

    def test_tool_choice_required_streaming(self):
        """
        Test: tool_choice='required' in streaming mode through gRPC router.
        This validates streaming tool calls across transport boundaries.
        """
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        # Collect streaming response
        tool_call_chunks = []

        for chunk in response:
            if chunk.choices[0].delta.tool_calls:
                tool_call_chunks.extend(chunk.choices[0].delta.tool_calls)

        # With required, we should get tool call chunks
        self.assertGreater(len(tool_call_chunks), 0)

    def test_required_streaming_arguments_chunks_json(self):
        """
        Test: In streaming required mode through gRPC, complete tool call arguments
        should be valid JSON when all chunks are combined.
        This is critical for catching partial JSON frame issues.
        """
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        # Collect all tool call chunks and reconstruct complete tool calls
        tool_calls_by_index = {}
        for chunk in response:
            if chunk.choices[0].delta.tool_calls:
                for tool_call_delta in chunk.choices[0].delta.tool_calls:
                    tool_index = tool_call_delta.index

                    # Initialize tool call if not seen before
                    if tool_index not in tool_calls_by_index:
                        tool_calls_by_index[tool_index] = {
                            "id": tool_call_delta.id,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    # Update function name if present (first chunk)
                    if tool_call_delta.function and tool_call_delta.function.name:
                        tool_calls_by_index[tool_index]["function"][
                            "name"
                        ] = tool_call_delta.function.name

                    # Accumulate arguments (all chunks)
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        tool_calls_by_index[tool_index]["function"][
                            "arguments"
                        ] += tool_call_delta.function.arguments

        self.assertGreater(len(tool_calls_by_index), 0)

        # Validate that complete tool calls have valid JSON arguments
        for tool_call in tool_calls_by_index.values():
            self.assertIsNotNone(tool_call["function"]["name"])
            self.assertIsNotNone(tool_call["function"]["arguments"])

            # The complete arguments should be valid JSON
            try:
                args = json.loads(tool_call["function"]["arguments"])
                self.assertIsInstance(args, dict)
            except json.JSONDecodeError:
                self.fail(
                    f"Invalid JSON in complete tool call arguments: {tool_call['function']['arguments']}"
                )

    def test_tool_choice_specific_function_streaming(self):
        """
        Test: tool_choice with specific function in streaming mode through gRPC router.
        """
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )

        # Collect streaming response
        tool_call_chunks = []

        for chunk in response:
            if chunk.choices[0].delta.tool_calls:
                tool_call_chunks.extend(chunk.choices[0].delta.tool_calls)

        # Should get tool call chunks for the specific function
        self.assertGreater(len(tool_call_chunks), 0)

        # Find function name in chunks
        found_name = None
        for chunk in tool_call_chunks:
            if chunk.function and chunk.function.name:
                found_name = chunk.function.name
                break

        self.assertEqual(found_name, "get_weather")


if __name__ == "__main__":
    unittest.main()
