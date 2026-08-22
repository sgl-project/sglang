"""Unit tests for DeepSeekV3Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestDeepSeekV3Detector(CustomTestCase):
    def setUp(self):
        """Set up test tools and detector for DeepSeekV3 format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV3Detector()

    def test_parse_streaming_multiple_tool_calls_with_multi_token_chunk(self):
        """Test parsing multiple tool calls when streaming chunks contains multi-tokens (e.g. DeepSeekV3 enable MTP)"""
        # Simulate streaming chunks with multi-tokens for two consecutive tool calls
        chunks = [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>function",
            "<｜tool▁sep｜>get",
            "_weather\n",
            "```json\n",
            '{"city":',
            '"Shanghai',
            '"}\n```<｜tool▁call▁end｜>',
            "\n<｜tool▁call▁begin｜>",
            "function<｜tool▁sep｜>",
            "get_tour",
            "ist_att",
            "ractions\n```" 'json\n{"',
            'city": "',
            'Beijing"}\n',
            "```<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ]

        tool_calls_seen = []
        tool_calls_parameters = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.name:
                        tool_calls_seen.append(call.name)
                    if call.parameters:
                        tool_calls_parameters.append(call.parameters)

        # Should see both tool names
        self.assertIn("get_weather", tool_calls_seen, "Should process first tool")
        self.assertIn(
            "get_tourist_attractions", tool_calls_seen, "Should process second tool"
        )

        # Verify that the parameters are valid JSON and contain the expected content
        params1 = json.loads(tool_calls_parameters[0])
        params2 = json.loads(tool_calls_parameters[1])
        self.assertEqual(params1["city"], "Shanghai")
        self.assertEqual(params2["city"], "Beijing")


if __name__ == "__main__":
    import unittest

    unittest.main()
