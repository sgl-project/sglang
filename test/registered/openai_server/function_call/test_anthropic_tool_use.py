"""
Tests for Anthropic-compatible tool use via the /v1/messages endpoint.

python3 -m unittest openai_server.function_call.test_anthropic_tool_use.TestAnthropicToolUse.test_tool_use_format
python3 -m unittest openai_server.function_call.test_anthropic_tool_use.TestAnthropicToolUse.test_tool_use_streaming
python3 -m unittest openai_server.function_call.test_anthropic_tool_use.TestAnthropicToolUse.test_tool_use_streaming_args_parsing
python3 -m unittest openai_server.function_call.test_anthropic_tool_use.TestAnthropicToolUse.test_tool_choice_auto
python3 -m unittest openai_server.function_call.test_anthropic_tool_use.TestAnthropicToolUse.test_tool_choice_any
python3 -m unittest openai_server.function_call.test_anthropic_tool_use.TestAnthropicToolUse.test_tool_choice_specific
python3 -m unittest openai_server.function_call.test_anthropic_tool_use.TestAnthropicToolUse.test_tool_result_multi_turn
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")
register_amd_ci(est_time=140, suite="stage-b-test-small-1-gpu-amd")

# System message to guide Llama3.2 to produce proper tool call format
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

ADD_TOOL = {
    "name": "add",
    "description": "Compute the sum of two integers",
    "input_schema": {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First integer"},
            "b": {"type": "integer", "description": "Second integer"},
        },
        "required": ["a", "b"],
    },
}

WEATHER_TOOL = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city to find the weather for",
            },
            "unit": {
                "type": "string",
                "description": "Weather unit (celsius or fahrenheit)",
                "enum": ["celsius", "fahrenheit"],
            },
        },
        "required": ["city", "unit"],
    },
}


class TestAnthropicToolUse(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tool-call-parser",
                "llama3",
            ],
        )
        cls.messages_url = cls.base_url + "/v1/messages"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _make_request(self, payload, stream=False):
        """Send a request to the /v1/messages endpoint."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        return requests.post(
            self.messages_url,
            headers=headers,
            json=payload,
            stream=stream,
        )

    def _parse_sse_events(self, response):
        """Parse SSE events from a streaming response."""
        events = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    events.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass
        return events

    # ---- Non-streaming tool use tests ----

    def test_tool_use_format(self):
        """Test that tool use returns proper Anthropic content blocks."""
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "Compute (3+5)"},
            ],
            "tools": [ADD_TOOL],
            "temperature": 0.8,
            "top_p": 0.8,
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")

        # Find tool_use content blocks
        tool_use_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        self.assertTrue(
            len(tool_use_blocks) > 0,
            f"Expected tool_use content blocks, got: {body['content']}",
        )

        tool_block = tool_use_blocks[0]
        self.assertEqual(tool_block["name"], "add", "Tool name should be 'add'")
        self.assertIn("id", tool_block, "Tool use block should have an id")
        self.assertIn("input", tool_block, "Tool use block should have input")
        self.assertIsInstance(tool_block["input"], dict)

        # Verify stop_reason is tool_use
        self.assertEqual(
            body["stop_reason"],
            "tool_use",
            f"Expected stop_reason 'tool_use', got: {body['stop_reason']}",
        )

    def test_tool_choice_auto(self):
        """Test tool_choice type=auto (default when tools provided)."""
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "Compute (3+5)"},
            ],
            "tools": [ADD_TOOL],
            "tool_choice": {"type": "auto"},
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        # With auto, model may or may not use tools - just verify valid response
        self.assertIsInstance(body["content"], list)

    def test_tool_choice_any(self):
        """Test tool_choice type=any (maps to required)."""
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Paris in celsius?",
                },
            ],
            "tools": [WEATHER_TOOL],
            "tool_choice": {"type": "any"},
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")

        # With 'any', the model must use a tool
        tool_use_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        self.assertTrue(
            len(tool_use_blocks) > 0,
            f"Expected tool_use blocks with tool_choice=any, got: {body['content']}",
        )

    def test_tool_choice_specific(self):
        """Test tool_choice type=tool with specific tool name."""
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "tools": [ADD_TOOL, WEATHER_TOOL],
            "tool_choice": {"type": "tool", "name": "get_current_weather"},
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")

        # With specific tool choice, the model should call that specific tool
        tool_use_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        self.assertTrue(
            len(tool_use_blocks) > 0,
            f"Expected tool_use blocks with specific tool_choice, got: {body['content']}",
        )
        for block in tool_use_blocks:
            self.assertEqual(
                block["name"],
                "get_current_weather",
                f"Expected tool name 'get_current_weather', got: {block['name']}",
            )

    def test_tool_result_multi_turn(self):
        """Test multi-turn conversation with tool_result messages."""
        # First turn: request a tool call
        payload_1 = {
            "model": self.model,
            "max_tokens": 2048,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "Compute (3+5)"},
            ],
            "tools": [ADD_TOOL],
            "temperature": 0.8,
        }
        resp_1 = self._make_request(payload_1)
        self.assertEqual(resp_1.status_code, 200, f"Response: {resp_1.text}")
        body_1 = resp_1.json()

        # Extract tool call info
        tool_use_blocks = [b for b in body_1["content"] if b["type"] == "tool_use"]
        self.assertTrue(len(tool_use_blocks) > 0, "Expected tool_use in first response")
        tool_call_id = tool_use_blocks[0]["id"]

        # Second turn: send tool_result back
        payload_2 = {
            "model": self.model,
            "max_tokens": 2048,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "Compute (3+5)"},
                {
                    "role": "assistant",
                    "content": body_1["content"],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "id": tool_call_id,
                            "content": "8",
                        }
                    ],
                },
            ],
            "tools": [ADD_TOOL],
        }
        resp_2 = self._make_request(payload_2)
        self.assertEqual(resp_2.status_code, 200, f"Response: {resp_2.text}")

        body_2 = resp_2.json()
        self.assertEqual(body_2["type"], "message")
        self.assertTrue(
            len(body_2["content"]) > 0, "Second response should have content"
        )

    def test_tool_use_with_text_content(self):
        """Test that response can contain both text and tool_use blocks."""
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "Compute (3+5)"},
            ],
            "tools": [ADD_TOOL],
            "tool_choice": {"type": "auto"},
            "temperature": 0.8,
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertIsInstance(body["content"], list)
        # Verify that content has valid block types
        for block in body["content"]:
            self.assertIn(
                block["type"],
                ["text", "tool_use"],
                f"Unexpected content block type: {block['type']}",
            )

    # ---- Streaming tool use tests ----

    def test_tool_use_streaming(self):
        """Test streaming tool use returns proper Anthropic events."""
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "stream": True,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the temperature in Paris in celsius?",
                },
            ],
            "tools": [WEATHER_TOOL],
            "tool_choice": {"type": "any"},
        }
        resp = self._make_request(payload, stream=True)
        self.assertEqual(resp.status_code, 200)

        events = self._parse_sse_events(resp)
        event_types = [e["type"] for e in events]

        # Verify basic event sequence
        self.assertIn("message_start", event_types)
        self.assertIn("message_stop", event_types)

        # Check for tool use content block events
        block_starts = [e for e in events if e["type"] == "content_block_start"]
        tool_use_starts = [
            e
            for e in block_starts
            if e.get("content_block", {}).get("type") == "tool_use"
        ]

        self.assertTrue(
            len(tool_use_starts) > 0,
            "Expected tool_use content_block_start events with tool_choice=any",
        )

        # Verify tool_use content_block_start has proper structure
        tool_start = tool_use_starts[0]
        self.assertIn("content_block", tool_start)
        self.assertEqual(tool_start["content_block"]["type"], "tool_use")
        self.assertIn("id", tool_start["content_block"])
        self.assertIn("name", tool_start["content_block"])

        # Check for input_json_delta events
        input_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        # Tool calls should have at least some argument deltas
        self.assertTrue(
            len(input_deltas) > 0,
            "Expected input_json_delta events for tool call",
        )

        # Verify message_delta has stop_reason=tool_use
        message_deltas = [e for e in events if e["type"] == "message_delta"]
        self.assertTrue(len(message_deltas) > 0)
        self.assertEqual(
            message_deltas[-1]["delta"]["stop_reason"],
            "tool_use",
            "Expected stop_reason 'tool_use' in streaming",
        )

    def test_tool_use_streaming_args_parsing(self):
        """Test that streaming tool call arguments can be concatenated into valid JSON."""
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "stream": True,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": "Please sum 5 and 7, just call the function.",
                },
            ],
            "tools": [
                {
                    "name": "add",
                    "description": "Compute the sum of two integers",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "First integer"},
                            "b": {"type": "integer", "description": "Second integer"},
                        },
                        "required": ["a", "b"],
                    },
                }
            ],
        }
        resp = self._make_request(payload, stream=True)
        self.assertEqual(resp.status_code, 200)

        events = self._parse_sse_events(resp)

        # Collect tool call data from stream
        tool_name = None
        argument_fragments = []

        for event in events:
            if event["type"] == "content_block_start":
                cb = event.get("content_block", {})
                if cb.get("type") == "tool_use":
                    tool_name = cb.get("name")
            elif event["type"] == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "input_json_delta":
                    partial = delta.get("partial_json", "")
                    if partial:
                        argument_fragments.append(partial)

        if tool_name is not None:
            # If we got a tool call, verify arguments are valid JSON
            self.assertEqual(tool_name, "add", "Tool name should be 'add'")
            joined_args = "".join(argument_fragments)
            self.assertTrue(
                len(joined_args) > 0,
                "No argument fragments returned for tool call",
            )

            try:
                args_obj = json.loads(joined_args)
            except json.JSONDecodeError:
                self.fail(
                    f"Concatenated tool call arguments are not valid JSON: {joined_args}"
                )

            self.assertIn("a", args_obj, "Missing parameter 'a'")
            self.assertIn("b", args_obj, "Missing parameter 'b'")

    def test_tool_use_streaming_event_sequence(self):
        """Test that streaming tool use events follow the correct order."""
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "stream": True,
            "system": SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "Compute (3+5)"},
            ],
            "tools": [ADD_TOOL],
            "tool_choice": {"type": "any"},
        }
        resp = self._make_request(payload, stream=True)
        self.assertEqual(resp.status_code, 200)

        events = self._parse_sse_events(resp)
        event_types = [e["type"] for e in events]

        # message_start must be first
        self.assertEqual(
            event_types[0],
            "message_start",
            "First event should be message_start",
        )

        # message_stop must be last
        self.assertEqual(
            event_types[-1],
            "message_stop",
            "Last event should be message_stop",
        )

        # message_delta should come before message_stop
        self.assertIn("message_delta", event_types)
        delta_idx = event_types.index("message_delta")
        stop_idx = event_types.index("message_stop")
        self.assertLess(
            delta_idx, stop_idx, "message_delta should come before message_stop"
        )

        # For each content block, start should come before stop
        block_start_indices = [
            i for i, t in enumerate(event_types) if t == "content_block_start"
        ]
        block_stop_indices = [
            i for i, t in enumerate(event_types) if t == "content_block_stop"
        ]
        self.assertEqual(
            len(block_start_indices),
            len(block_stop_indices),
            "Number of content_block_start should equal content_block_stop",
        )
        for start_i, stop_i in zip(block_start_indices, block_stop_indices):
            self.assertLess(
                start_i,
                stop_i,
                "content_block_start should come before content_block_stop",
            )

    def test_no_tools_no_tool_use(self):
        """Test that without tools, no tool_use blocks appear."""
        payload = {
            "model": self.model,
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        tool_use_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        self.assertEqual(
            len(tool_use_blocks),
            0,
            "Should not have tool_use blocks when no tools provided",
        )
        self.assertIn(
            body["stop_reason"],
            ["end_turn", "max_tokens"],
            "Stop reason should be end_turn or max_tokens without tools",
        )


if __name__ == "__main__":
    unittest.main()
