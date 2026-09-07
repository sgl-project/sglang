"""Responses API custom (freeform) tool support.

A client that declares ``tools: [{"type": "custom"}]`` must be able to
round-trip the conversation: the declaration reaches the chat template, the
model's call comes back as a ``custom_tool_call`` output item carrying the raw
string input, and replaying ``custom_tool_call`` / ``custom_tool_call_output``
items on the next turn is accepted instead of 400ing (the OpenAI Codex CLI
``apply_patch_tool_type = freeform`` flow).
"""

import unittest

import orjson

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _request_with_custom_tool() -> ResponsesRequest:
    return ResponsesRequest(
        model="test-model",
        input="patch README.md",
        tools=[
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "freeform patch tool",
            },
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
    )


class TestResponsesCustomTools(CustomTestCase):
    def test_custom_tool_declaration_reaches_chat_tools(self):
        request = _request_with_custom_tool()
        chat_tools = OpenAIServingResponses._response_tools_to_chat_tools(request)
        by_name = {tool.function.name: tool for tool in chat_tools}
        self.assertIn("apply_patch", by_name)
        self.assertIn("get_weather", by_name)
        params = by_name["apply_patch"].function.parameters
        self.assertEqual(params["required"], ["input"])
        self.assertEqual(params["properties"]["input"]["type"], "string")

    def test_custom_tool_call_replay_normalizes_to_assistant_tool_call(self):
        patch = "*** Begin Patch\n*** End Patch\n"
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "apply_patch",
                "input": patch,
            }
        )
        self.assertEqual(message["role"], "assistant")
        (tool_call,) = message["tool_calls"]
        self.assertEqual(tool_call["function"]["name"], "apply_patch")
        arguments = orjson.loads(tool_call["function"]["arguments"])
        self.assertEqual(arguments, {"input": patch})

    def test_custom_tool_call_output_replay_normalizes_to_tool_message(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "done",
            }
        )
        self.assertEqual(
            message, {"role": "tool", "tool_call_id": "call_1", "content": "done"}
        )

    def test_unknown_item_type_still_rejected(self):
        with self.assertRaises(ValueError):
            OpenAIServingResponses._normalize_response_message_for_chat(
                {"type": "additional_tools"}
            )

    def test_build_tool_call_item_maps_declared_custom_tools(self):
        patch = "*** Begin Patch\nline\n*** End Patch\n"
        wrapped = orjson.dumps({"input": patch}).decode("utf-8")
        item = OpenAIServingResponses._build_tool_call_item(
            "apply_patch", wrapped, {"apply_patch"}
        )
        self.assertEqual(item.type, "custom_tool_call")
        self.assertEqual(item.name, "apply_patch")
        self.assertEqual(item.input, patch)
        self.assertEqual(item.status, "completed")

        function_item = OpenAIServingResponses._build_tool_call_item(
            "get_weather", "{}", {"apply_patch"}
        )
        self.assertEqual(function_item.type, "function_call")

    def test_custom_tool_input_unwrap_falls_back_to_raw(self):
        unwrap = OpenAIServingResponses._custom_tool_input_from_arguments
        self.assertEqual(unwrap('{"input": "abc"}'), "abc")
        self.assertEqual(unwrap("not json"), "not json")
        self.assertEqual(unwrap('{"other": 1}'), '{"other": 1}')
        self.assertEqual(unwrap(""), "")


if __name__ == "__main__":
    unittest.main()
