# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for /v1/responses request validation and input conversion."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.harmony_utils import parse_response_input
from sglang.srt.entrypoints.openai.protocol import (
    ResponseInputFunctionCallItem,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class ResponsesRequestValidationTest(unittest.TestCase):
    def test_accepts_codex_structured_input_shapes(self):
        request = ResponsesRequest(
            model="test-model",
            input=[
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_1",
                    "content": [
                        {"type": "reasoning_text", "text": "thinking"},
                        {"type": "output_text", "text": "done"},
                        {
                            "type": "input_image",
                            "image_url": {
                                "url": "https://example.com/a.png",
                                "detail": "high",
                            },
                        },
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "/tmp/project"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/log.txt",
                        },
                    ],
                },
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "earlier trace"}],
                },
            ],
        )

        self.assertEqual(len(request.input), 4)
        self.assertEqual(request.input[0].content[1].text, "done")
        self.assertEqual(request.input[1].call_id, "call_1")
        self.assertEqual(request.input[2].output[0].text, "/tmp/project")
        self.assertEqual(request.input[3].content[0].text, "earlier trace")

    def test_rejects_malformed_structured_input(self):
        invalid_payloads = [
            {"model": "test-model", "input": [{"type": "message", "content": []}]},
            {
                "model": "test-model",
                "input": [{"type": "function_call", "name": "x", "arguments": "{}"}],
            },
            {
                "model": "test-model",
                "input": [{"type": "function_call_output", "call_id": "x"}],
            },
            {
                "model": "test-model",
                "input": [{"type": "computer_call", "role": "user", "content": []}],
            },
            {
                "model": "test-model",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_audio", "audio_url": "x"}],
                    }
                ],
            },
            {
                "model": "test-model",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            },
            {
                "model": "test-model",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "x",
                        "name": "n",
                        "arguments": {"cmd": "pwd"},
                    }
                ],
            },
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    ResponsesRequest(**payload)


class ResponsesInputConversionTest(unittest.TestCase):
    def setUp(self):
        self.serving = object.__new__(OpenAIServingResponses)

    def test_non_harmony_conversion_preserves_supported_shapes(self):
        request = ResponsesRequest(
            model="test-model",
            input=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning_text", "text": "thinking"},
                        {"type": "output_text", "text": "done"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/a.png",
                        },
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "/tmp/project"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/log.txt"},
                        },
                    ],
                },
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "earlier trace"}],
                },
            ],
        )

        message_item, function_call_item, function_output_item, reasoning_item = (
            request.input
        )

        self.assertEqual(
            self.serving._response_input_item_to_chat_messages(message_item),
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "done"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/a.png",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
        )
        self.assertEqual(
            self.serving._response_input_item_to_chat_messages(function_call_item)[0][
                "tool_calls"
            ][0]["function"]["name"],
            "exec_command",
        )
        self.assertEqual(
            self.serving._response_input_item_to_chat_messages(function_output_item),
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "/tmp/project\nhttps://example.com/log.txt",
                }
            ],
        )
        self.assertEqual(
            self.serving._response_input_item_to_chat_messages(reasoning_item), []
        )

    def test_harmony_parser_handles_structured_message_and_tool_output(self):
        request = ResponsesRequest(
            model="test-model",
            input=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning_text", "text": "thinking"},
                        {"type": "output_text", "text": "done"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/a.png",
                        },
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "/tmp/project"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/log.txt",
                        },
                    ],
                },
            ],
        )

        prev_outputs = []
        parsed_messages = []
        for item in request.input:
            parsed_messages.append(parse_response_input(item, prev_outputs))
            if isinstance(item, ResponseInputFunctionCallItem):
                prev_outputs.append(
                    self.serving._response_function_call_item_to_output(item)
                )

        self.assertEqual(parsed_messages[0].content[0].text, "thinking")
        self.assertEqual(parsed_messages[0].content[1].text, "done")
        self.assertEqual(parsed_messages[0].content[2].text, "https://example.com/a.png")
        self.assertEqual(
            parsed_messages[2].content[0].text,
            "/tmp/project\nhttps://example.com/log.txt",
        )


if __name__ == "__main__":
    unittest.main()
