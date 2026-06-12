"""Unit tests for Qwen3CoderDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _wrap(func_name: str, params: str) -> str:
    """Wrap a function body in the Qwen3-Coder tool-call envelope."""
    return f"<tool_call>\n<function={func_name}>\n{params}</function>\n</tool_call>"


def _param(name: str, value: str) -> str:
    return f"<parameter={name}>\n{value}\n</parameter>\n"


class TestQwen3CoderDetector(CustomTestCase):
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
                            "days": {"type": "integer"},
                            "verbose": {"type": "boolean"},
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
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = Qwen3CoderDetector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = _wrap("get_weather", _param("city", "Beijing"))
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse (non-streaming) Tests ====================

    def test_single_tool_call(self):
        text = _wrap("get_weather", _param("city", "Beijing"))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_parameter_type_conversion(self):
        # Values are emitted as text; the detector should coerce them to the
        # types declared in the tool schema (int / bool), leaving strings as-is.
        text = _wrap(
            "get_weather",
            _param("city", "Beijing") + _param("days", "3") + _param("verbose", "true"),
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(args["days"], 3)
        self.assertIs(args["verbose"], True)

    def test_leading_normal_text(self):
        text = "Let me check.\n" + _wrap("get_weather", _param("city", "Tokyo"))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.normal_text, "Let me check.\n")

    def test_multiple_tool_calls(self):
        text = (
            _wrap("get_weather", _param("city", "Beijing"))
            + "\n"
            + _wrap("search", _param("query", "restaurants"))
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")
        self.assertEqual(
            json.loads(result.calls[1].parameters), {"query": "restaurants"}
        )

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    # ==================== parse_streaming_increment Tests ====================

    def test_streaming_reconstructs_tool_call(self):
        detector = Qwen3CoderDetector()
        chunks = [
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>\nBei",
            "jing\n</parameter>\n",
            "<parameter=days>\n3\n</parameter>\n",
            "</function>\n</tool_call>",
        ]
        names = []
        param_parts = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            for item in result.calls:
                if item.name:
                    names.append(item.name)
                if item.parameters:
                    param_parts.append(item.parameters)

        self.assertEqual(names, ["get_weather"])
        params = json.loads("".join(param_parts))
        self.assertEqual(params, {"city": "Beijing", "days": 3})

    def test_streaming_normal_text_before_tool(self):
        detector = Qwen3CoderDetector()
        result = detector.parse_streaming_increment("Sure! ", self.tools)
        self.assertEqual(result.normal_text, "Sure! ")
        self.assertEqual(len(result.calls), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
