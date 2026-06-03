"""Unit tests for Qwen25Detector — no server, no model loading.

Covers the Qwen 2.5 / Qwen 3 tool-call format:

    <tool_call>
    {"name": "func", "arguments": {...}}
    </tool_call>
"""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestQwen25Detector(CustomTestCase):
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
        self.detector = Qwen25Detector()

    @staticmethod
    def _tool_call(name: str, arguments: dict) -> str:
        """Build a single ``<tool_call>...</tool_call>`` block."""
        body = json.dumps({"name": name, "arguments": arguments})
        return f"<tool_call>\n{body}\n</tool_call>"

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = self._tool_call("get_weather", {"city": "Beijing"})
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_with_leading_text(self):
        text = "Let me check. " + self._tool_call("get_weather", {"city": "Beijing"})
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = self._tool_call("get_weather", {"city": "Beijing"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        text = (
            self._tool_call("get_weather", {"city": "Beijing"})
            + "\n"
            + self._tool_call("search", {"query": "restaurants"})
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")
        self.assertEqual(json.loads(result.calls[1].parameters)["query"], "restaurants")

    def test_tool_call_with_leading_text(self):
        text = "I will check that for you." + self._tool_call(
            "get_weather", {"city": "Tokyo"}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "I will check that for you.")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    def test_tool_call_with_nested_json(self):
        text = self._tool_call(
            "get_weather", {"city": "Beijing", "options": {"detailed": True}}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["options"]["detailed"], True)

    def test_invalid_json_inside_block_is_skipped(self):
        # Malformed JSON inside the block should be ignored, not raise.
        text = "<tool_call>\n{not valid json}\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "")

    def test_valid_and_invalid_blocks_mixed(self):
        # A malformed block is dropped while a valid sibling block is kept.
        text = "<tool_call>\n{bad}\n</tool_call>\n" + self._tool_call(
            "search", {"query": "news"}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        info = self.detector.structure_info()("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertIn("<tool_call>", info.trigger)
        self.assertEqual(info.end, "}\n</tool_call>")

    # ==================== Streaming Tests ====================

    def test_streaming_normal_text_only(self):
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment("Let me check. ", self.tools)
        self.assertEqual(result.normal_text, "Let me check. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_single_tool_call(self):
        detector = Qwen25Detector()
        chunks = [
            '<tool_call>\n{"name": "get',
            '_weather", "arguments": {"city": "Paris"}}',
            "\n</tool_call>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        self.assertEqual(json.loads(full_params)["city"], "Paris")

    def test_streaming_text_then_tool_call(self):
        detector = Qwen25Detector()
        chunks = [
            "I will help. ",
            '<tool_call>\n{"name": "get_weather", ',
            '"arguments": {"city": "Tokyo"}}',
            "\n</tool_call>",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "I will help. ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        self.assertEqual(json.loads(full_params)["city"], "Tokyo")


if __name__ == "__main__":
    import unittest

    unittest.main()
