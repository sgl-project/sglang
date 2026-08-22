"""Responses API Codex-compatibility items: namespace tools, agent_message,
and array-shaped tool output.

Namespace tools flatten to ``f"{namespace}.{inner}"`` chat functions and the
emitted call item splits the pair back apart; ``agent_message`` items from
multi-agent threads render as text instead of 400ing; tool-output content
arrays preserve non-text parts (``input_image``) as chat content parts instead
of silently dropping them (issues #33867 / #34927).
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _request_with_namespace_tool() -> ResponsesRequest:
    return ResponsesRequest(
        model="test-model",
        input="spawn an agent",
        tools=[
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Multi-agent orchestration tools.",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawn a sub-agent.",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"task": {"type": "string"}},
                            "required": ["task"],
                        },
                    },
                    {
                        "type": "function",
                        "name": "send_message",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            },
            {
                "type": "function",
                "name": "dotted.but.flat",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
    )


class TestNamespaceTools(CustomTestCase):
    def test_namespace_flattens_to_qualified_chat_functions(self):
        request = _request_with_namespace_tool()
        chat_tools = OpenAIServingResponses._response_tools_to_chat_tools(request)
        by_name = {tool.function.name: tool for tool in chat_tools}
        self.assertIn("multi_agent_v1.spawn_agent", by_name)
        self.assertIn("multi_agent_v1.send_message", by_name)
        spawn = by_name["multi_agent_v1.spawn_agent"].function
        self.assertEqual(spawn.description, "Spawn a sub-agent.")
        self.assertEqual(spawn.parameters["required"], ["task"])
        # ``strict`` is part of the inner function definition and must survive
        # flattening.
        self.assertTrue(spawn.strict)
        self.assertFalse(by_name["multi_agent_v1.send_message"].function.strict)
        # Inner without a description inherits the namespace's.
        self.assertEqual(
            by_name["multi_agent_v1.send_message"].function.description,
            "Multi-agent orchestration tools.",
        )

    def test_emitted_call_item_splits_namespace(self):
        request = _request_with_namespace_tool()
        namespace_names = OpenAIServingResponses._namespace_names(request)
        item = OpenAIServingResponses._build_tool_call_item(
            "multi_agent_v1.spawn_agent", '{"task": "t"}', set(), namespace_names
        )
        self.assertEqual(item.type, "function_call")
        self.assertEqual(item.name, "spawn_agent")
        self.assertEqual(item.model_dump()["namespace"], "multi_agent_v1")

    def test_undeclared_dotted_name_passes_through(self):
        request = _request_with_namespace_tool()
        namespace_names = OpenAIServingResponses._namespace_names(request)
        item = OpenAIServingResponses._build_tool_call_item(
            "dotted.but.flat", "{}", set(), namespace_names
        )
        self.assertEqual(item.name, "dotted.but.flat")
        self.assertNotIn("namespace", item.model_dump())

    def test_replayed_namespaced_call_requalifies_for_chat(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "spawn_agent",
                "namespace": "multi_agent_v1",
                "arguments": '{"task": "t"}',
            }
        )
        self.assertEqual(
            message["tool_calls"][0]["function"]["name"],
            "multi_agent_v1.spawn_agent",
        )


class TestAgentMessage(CustomTestCase):
    def test_agent_message_renders_text_with_routing_header(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "agent_message",
                "id": "amsg_123",
                "author": "/root/agent_a",
                "recipient": "/root",
                "agent": {"agent_name": "/root"},
                "content": [
                    {"type": "input_text", "text": "sub-agent result"},
                    {"type": "encrypted_content", "encrypted_content": "plain reply"},
                ],
            }
        )
        self.assertEqual(message["role"], "user")
        self.assertIn("[agent message from /root/agent_a to /root]", message["content"])
        self.assertIn("sub-agent result", message["content"])
        self.assertIn("plain reply", message["content"])

    def test_agent_message_ciphertext_becomes_placeholder(self):
        blob = "QWJj" * 100  # 400 chars of unbroken base64
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "agent_message",
                "author": "/root/agent_a",
                "content": [{"type": "encrypted_content", "encrypted_content": blob}],
            }
        )
        self.assertNotIn(blob, message["content"])
        self.assertIn(
            "[encrypted agent message content unavailable]", message["content"]
        )

    def test_empty_agent_message_drops(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {"type": "agent_message", "author": "/root/agent_a", "content": []}
        )
        self.assertIsNone(message)


class TestToolOutputContentParts(CustomTestCase):
    def test_text_only_array_output_flattens_to_string(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "input_text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        )
        self.assertEqual(message["content"], "ab")

    def test_image_part_survives_as_chat_content_part(self):
        data_url = "data:image/png;base64,iVBORw0KGgo="
        for item_type in ("function_call_output", "custom_tool_call_output"):
            message = OpenAIServingResponses._normalize_response_message_for_chat(
                {
                    "type": item_type,
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "screenshot:"},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            )
            self.assertEqual(message["role"], "tool")
            parts = message["content"]
            self.assertIsInstance(parts, list)
            self.assertEqual(parts[0], {"type": "text", "text": "screenshot:"})
            self.assertEqual(parts[1]["type"], "image_url")
            self.assertEqual(parts[1]["image_url"]["url"], data_url)


if __name__ == "__main__":
    unittest.main()
