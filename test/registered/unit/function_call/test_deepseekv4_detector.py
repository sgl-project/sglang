"""Unit tests for DeepSeekV4Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _xml_invoke(name: str, *params: str) -> str:
    """Build one DSML invoke block with XML-style parameter tags."""
    return (
        '<｜DSML｜invoke name="'
        + name
        + '">\n'
        + "".join(params)
        + "</｜DSML｜invoke>\n"
    )


def _xml_param(name: str, value: str) -> str:
    return (
        '<｜DSML｜parameter name="'
        + name
        + '" string="true">'
        + value
        + "</｜DSML｜parameter>\n"
    )


def _wrap(*invokes: str) -> str:
    return "<｜DSML｜tool_calls>\n" + "".join(invokes) + "</｜DSML｜tool_calls>"


class TestDeepSeekV4Detector(CustomTestCase):
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
        self.detector = DeepSeekV4Detector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = _wrap(_xml_invoke("get_weather", _xml_param("city", "Beijing")))
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call_xml_params(self):
        text = _wrap(_xml_invoke("get_weather", _xml_param("city", "Beijing")))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_single_tool_call_json_params(self):
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">{"city": "Beijing"}</｜DSML｜invoke>\n'
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_multiple_tool_calls(self):
        text = _wrap(
            _xml_invoke("get_weather", _xml_param("city", "Beijing")),
            _xml_invoke("search", _xml_param("query", "restaurants")),
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_leading_text(self):
        text = "I will check the weather. " + _wrap(
            _xml_invoke("get_weather", _xml_param("city", "Tokyo"))
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.normal_text, "I will check the weather. ")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertEqual(info.trigger, "<｜DSML｜invoke")
        self.assertIn("get_weather", info.begin)
        self.assertEqual(info.end, "</｜DSML｜invoke>")

    def test_structural_tag_name(self):
        self.assertEqual(self.detector.get_structural_tag_name(), "deepseek_v4")

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        detector = DeepSeekV4Detector()
        chunks = [
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n',
            '<｜DSML｜parameter name="city" string="true">Beijing</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_normal_text_before_tool(self):
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment("Hello! Let me help. ", self.tools)
        self.assertEqual(result.normal_text, "Hello! Let me help. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = DeepSeekV4Detector()
        chunks = [
            "Sure, let me check. ",
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n',
            '<｜DSML｜parameter name="city" string="true">Tokyo</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "Sure, let me check. ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")


if __name__ == "__main__":
    import unittest

    unittest.main()
