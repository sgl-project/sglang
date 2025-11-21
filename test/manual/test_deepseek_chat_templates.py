"""
Unit tests for DeepSeek chat template tool call handling.

Tests verify that the DeepSeek chat templates (v3, v3.1, v3.2) correctly handle
both dict and string types for tool['function']['arguments'] without double-escaping,
addressing issue #11700.
"""

import os
import unittest

from jinja2 import Template


class TestDeepSeekChatTemplateToolCalls(unittest.TestCase):
    """Test DeepSeek chat templates handle tool calls correctly."""

    @classmethod
    def setUpClass(cls):
        """Load all DeepSeek chat templates."""
        base_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "chat_template"
        )

        cls.templates = {}
        template_files = {
            "v3": "tool_chat_template_deepseekv3.jinja",
            "v3.1": "tool_chat_template_deepseekv31.jinja",
            "v3.2": "tool_chat_template_deepseekv32.jinja",
        }

        for version, filename in template_files.items():
            template_path = os.path.join(base_path, filename)
            with open(template_path, "r") as f:
                template_content = f.read()
            cls.templates[version] = Template(template_content)

    def _render_template(
        self, version, messages, tools=None, add_generation_prompt=True
    ):
        """Helper method to render a template with given messages and tools."""
        template = self.templates[version]

        # Common template variables
        context = {
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": "<｜begin▁of▁sentence｜>",
        }

        if tools is not None:
            context["tools"] = tools

        return template.render(**context)

    def test_tool_arguments_as_dict(self):
        """Test that tool arguments as dict are properly JSON-encoded (normal case)."""
        # This tests the normal case where arguments come from OpenAI API as dict

        for version in ["v3", "v3.1", "v3.2"]:
            with self.subTest(version=version):
                messages = [
                    {"role": "user", "content": "What's the weather in NYC?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {
                                        "city": "New York",
                                        "unit": "celsius",
                                    },  # Dict
                                },
                            }
                        ],
                    },
                ]

                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                    "unit": {"type": "string"},
                                },
                            },
                        },
                    }
                ]

                output = self._render_template(version, messages, tools)

                # Should contain properly formatted JSON (not double-escaped)
                self.assertIn('"city"', output, f"{version}: Should contain city key")
                self.assertIn(
                    '"New York"', output, f"{version}: Should contain city value"
                )

                # Should NOT contain double-escaped quotes
                self.assertNotIn(
                    '\\"city\\"', output, f"{version}: Should not double-escape"
                )
                self.assertNotIn(
                    '\\\\"', output, f"{version}: Should not have escaped backslashes"
                )

    def test_tool_arguments_as_string(self):
        """Test that tool arguments as string are used as-is (multi-round case)."""
        # This tests the multi-round function calling case from issue #11700
        # where arguments might already be JSON strings from previous model output

        for version in ["v3", "v3.1", "v3.2"]:
            with self.subTest(version=version):
                messages = [
                    {"role": "user", "content": "What's the stock price of NVDA?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_stock_info",
                                    "arguments": '{"symbol": "NVDA"}',  # Already a JSON string
                                },
                            }
                        ],
                    },
                ]

                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_stock_info",
                            "description": "Get stock information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                },
                            },
                        },
                    }
                ]

                output = self._render_template(version, messages, tools)

                # Should contain the JSON string as-is
                self.assertIn(
                    '{"symbol": "NVDA"}',
                    output,
                    f"{version}: Should contain JSON as-is",
                )

                # Should NOT double-escape (the bug from issue #11700)
                # Bad output would look like: "{\"symbol\": \"NVDA\"}" or "{\\"symbol\\": \\"NVDA\\"}"
                self.assertNotIn(
                    '{\\"symbol\\"', output, f"{version}: Should not double-escape"
                )
                self.assertNotIn(
                    '"{\\"symbol', output, f"{version}: Should not wrap and escape"
                )

                # Verify it's not triple-quoted or escaped
                self.assertNotIn(
                    '""{"', output, f"{version}: Should not have extra quotes"
                )

    def test_multiple_tool_calls_mixed_types(self):
        """Test multiple tool calls with mixed dict and string argument types."""
        # This tests a complex scenario with multiple tools, some with dict args, some with string

        for version in ["v3", "v3.1", "v3.2"]:
            with self.subTest(version=version):
                messages = [
                    {"role": "user", "content": "Get weather and stock info"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "Boston"},  # Dict
                                },
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_stock_info",
                                    "arguments": '{"symbol": "TSLA"}',  # String
                                },
                            },
                        ],
                    },
                ]

                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_stock_info",
                            "description": "Get stock info",
                            "parameters": {
                                "type": "object",
                                "properties": {"symbol": {"type": "string"}},
                            },
                        },
                    },
                ]

                output = self._render_template(version, messages, tools)

                # First tool (dict) should be properly JSON-encoded
                self.assertIn(
                    '"city"', output, f"{version}: First tool should have city key"
                )
                self.assertIn(
                    '"Boston"',
                    output,
                    f"{version}: First tool should have Boston value",
                )

                # Second tool (string) should be used as-is
                self.assertIn(
                    '{"symbol": "TSLA"}',
                    output,
                    f"{version}: Second tool should use string as-is",
                )

                # Neither should be double-escaped
                self.assertNotIn(
                    '\\"city\\"',
                    output,
                    f"{version}: First tool should not double-escape",
                )
                self.assertNotIn(
                    '\\"symbol\\"',
                    output,
                    f"{version}: Second tool should not double-escape",
                )

    def test_tool_call_with_content(self):
        """Test tool calls that also include content text."""
        # Some models include explanatory text along with tool calls

        for version in ["v3", "v3.1", "v3.2"]:
            with self.subTest(version=version):
                messages = [
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "assistant",
                        "content": "Let me check the weather for you.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "Seattle"},
                                },
                            }
                        ],
                    },
                ]

                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ]

                output = self._render_template(version, messages, tools)

                # Should contain both the content and the tool call
                self.assertIn(
                    "Let me check the weather",
                    output,
                    f"{version}: Should include content",
                )
                self.assertIn(
                    '"city"', output, f"{version}: Should include tool arguments"
                )
                self.assertNotIn(
                    '\\"city\\"', output, f"{version}: Should not double-escape"
                )


if __name__ == "__main__":
    unittest.main()
