"""Unit tests for GptOssDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestGptOssDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = GptOssDetector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = '<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city":"Beijing"}<|call|>'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city":"Beijing"}<|call|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_normal_text_before_tool_call(self):
        text = 'Let me check.<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city":"Beijing"}<|call|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Let me check.")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    def test_multiple_tool_calls(self):
        text = (
            '<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city":"Beijing"}<|call|>'
            '<|start|>assistant<|channel|>commentary to=functions.search<|constrain|>json<|message|>{"query":"restaurants"}<|call|>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    # ==================== structure_info Tests ====================

    def test_structure_info_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        chunks = [
            "<|start|>",
            "assistant<|channel|>",
            "commentary to=functions.get_weather",
            '<|constrain|>json<|message|>{"city":"Beijing"}<|call|>',
        ]

        all_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_normal_text_before_tool(self):
        result = self.detector.parse_streaming_increment(
            "Let me check the weather. ", self.tools
        )
        self.assertEqual(result.normal_text, "Let me check the weather. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        chunks = [
            "I'll look",
            " that up. <|start|>",
            "assistant<|channel|>",
            "commentary to=functions.get_weather",
            '<|constrain|>json<|message|>{"city":"Beijing", "unit":"celsius"}<|call|>',
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "I'll look that up. ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")
        self.assertEqual(params["unit"], "celsius")


if __name__ == "__main__":
    import unittest

    unittest.main()
