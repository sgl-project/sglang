"""Unit tests for K2V3Detector (BBQ 0518 IFM tool-call format).

Ported from vLLM's K2V3ToolParser tests
(tests/entrypoints/openai/tool_parsers/test_multi_format_tool_parser.py).
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.multi_format_detector import (
    K2V3Detector,
    MultiFormatDetector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "stage-a-test-cpu")


def _make_tool(name, parameters=None):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters=parameters
            or {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        ),
    )


class TestK2V3DetectorConstruction(unittest.TestCase):
    """K2V3Detector defaults to the IFM 'xml' dialect with no delegate."""

    def test_default_dialect_is_xml(self):
        det = K2V3Detector()
        self.assertEqual(det.tool_format, "xml")
        self.assertIsNone(det._delegate)

    def test_subclass_of_multi_format(self):
        self.assertIsInstance(K2V3Detector(), MultiFormatDetector)

    def test_tool_call_format_kwarg_selects_dialect(self):
        det = K2V3Detector(chat_template_kwargs={"tool_call_format": "xml_typed"})
        self.assertEqual(det.tool_format, "xml_typed")

    def test_json_dialect_via_positional(self):
        det = K2V3Detector(tool_format="json")
        self.assertEqual(det.tool_format, "json")

    def test_unknown_dialect_errors(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tool_format"):
            K2V3Detector(tool_format="not-a-dialect")


class TestK2V3XmlExtraction(unittest.TestCase):
    def setUp(self):
        self.tools = [
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            )
        ]
        self.det = K2V3Detector()

    def test_has_tool_call(self):
        text = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        self.assertTrue(self.det.has_tool_call(text))
        self.assertFalse(self.det.has_tool_call("no markers here"))

    def test_single_call(self):
        text = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})
        self.assertEqual(result.normal_text, "")

    def test_schema_typed_value_coercion(self):
        # 'days' is declared integer in the schema -> deserialized to int 3.
        text = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>days</ifm|arg_key><ifm|arg_value>3</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"days": 3})

    def test_no_args(self):
        text = "<ifm|tool_call>get_weather</ifm|tool_call>"
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_no_markers_returns_full_text(self):
        result = self.det.detect_and_parse("plain text", self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "plain text")

    def test_unknown_tool_is_forwarded(self):
        text = (
            "<ifm|tool_call>not_registered"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, -1)
        self.assertEqual(result.calls[0].name, "not_registered")


class TestK2V3XmlTyped(unittest.TestCase):
    """The <ifm|arg_type> hint forces a value to stay a string."""

    def test_arg_type_string_keeps_numeric_as_string(self):
        det = K2V3Detector(tool_format="xml_typed")
        # No schema type for user_id; the inline <ifm|arg_type>string</...> wins.
        text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>user_id</ifm|arg_key>"
            "<ifm|arg_type>string</ifm|arg_type>"
            "<ifm|arg_value>12345</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = det.detect_and_parse(text, [_make_tool("study_args")])
        self.assertEqual(json.loads(result.calls[0].parameters), {"user_id": "12345"})


class TestK2V3ReasoningPrefix(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]
        self.det = K2V3Detector()

    def test_ifm_reasoning_prefix_is_stripped(self):
        text = (
            "<ifm|think>need lookup</ifm|think>\n"
            "<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_all_three_effort_blocks_are_stripped(self):
        for think in ("think", "think_fast", "think_faster"):
            with self.subTest(think=think):
                text = (
                    f"<ifm|{think}>reasoning</ifm|{think}>"
                    "<ifm|tool_call>get_weather"
                    "<ifm|arg_key>city</ifm|arg_key>"
                    "<ifm|arg_value>Tokyo</ifm|arg_value>"
                    "</ifm|tool_call>"
                )
                result = self.det.detect_and_parse(text, self.tools)
                self.assertEqual(result.normal_text, "")
                self.assertEqual(result.calls[0].name, "get_weather")

    def test_legacy_think_prefix_is_not_stripped(self):
        text = (
            "<think>legacy reasoning</think>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "<think>legacy reasoning</think>\n")
        self.assertEqual(result.calls[0].name, "get_weather")


class TestK2V3JsonDialect(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]
        self.det = K2V3Detector(tool_format="json")

    def test_json_object_tool_call(self):
        text = (
            "<ifm|tool_call>"
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_json_arguments_as_string(self):
        text = (
            "<ifm|tool_call>"
            '{"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"}'
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_json_list_of_tool_calls(self):
        text = (
            "<ifm|tool_call>"
            '[{"name": "get_weather", "arguments": {"city": "Tokyo"}},'
            ' {"name": "get_weather", "arguments": {"city": "Osaka"}}]'
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather", "get_weather"])
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "Osaka"})


class TestK2V3Streaming(unittest.TestCase):
    """IFM is an embedded (non-streaming) dialect: increments buffer, emit nothing."""

    def test_streaming_emits_nothing(self):
        det = K2V3Detector()
        tools = [_make_tool("get_weather")]
        text = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = det.parse_streaming_increment(text, tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "")


class TestK2V3RegistryWiring(unittest.TestCase):
    """FunctionCallParser resolves 'k2_v3' to K2V3Detector end-to-end."""

    def test_registry_builds_k2v3_detector(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        parser = FunctionCallParser(
            tools=tools,
            tool_call_parser="k2_v3",
            chat_template_kwargs={"tool_call_format": "xml"},
        )
        self.assertIsInstance(parser.detector, K2V3Detector)
        self.assertEqual(parser.detector.tool_format, "xml")

    def test_full_pipeline_non_stream(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        parser = FunctionCallParser(tools=tools, tool_call_parser="k2_v3")
        wire_output = (
            "<ifm|think>looking it up</ifm|think>"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        normal_text, calls = parser.parse_non_stream(wire_output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Tokyo"})
        self.assertEqual(normal_text, "")


if __name__ == "__main__":
    unittest.main()
