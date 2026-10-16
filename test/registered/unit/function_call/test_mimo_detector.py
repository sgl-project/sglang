"""Unit tests for MiMoDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

BOT = "<" + "tool_call" + ">"
EOT = "</" + "tool_call" + ">"


def _make_tools(param_properties):
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": param_properties,
                    "required": ["city"],
                },
            ),
        )
    ]


def _tool_call_text(param_pairs):
    inner = "".join(
        "<parameter=%s>%s</parameter>\n" % (name, value) for name, value in param_pairs
    )
    return BOT + "\n<function=get_weather>\n" + inner + "</function>\n" + EOT


class TestMiMoDetector(CustomTestCase):
    def test_union_type_param_converts_like_first_non_null(self):
        # JSON Schema union types (e.g. ["string", "null"]) must not crash
        tools = _make_tools(
            {
                "city": {"type": "string"},
                "unit": {"type": ["string", "null"]},
            }
        )
        detector = MiMoDetector()
        result = detector.detect_and_parse(
            _tool_call_text([("city", "Beijing"), ("unit", "celsius")]), tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters)["unit"], "celsius")

    def test_union_type_param_null_value(self):
        tools = _make_tools(
            {
                "city": {"type": "string"},
                "unit": {"type": ["string", "null"]},
            }
        )
        detector = MiMoDetector()
        result = detector.detect_and_parse(
            _tool_call_text([("city", "Beijing"), ("unit", "null")]), tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertIsNone(json.loads(result.calls[0].parameters)["unit"])

    def test_union_type_integer_converts_to_int(self):
        tools = _make_tools(
            {
                "city": {"type": "string"},
                "days": {"type": ["integer", "null"]},
            }
        )
        detector = MiMoDetector()
        result = detector.detect_and_parse(
            _tool_call_text([("city", "Beijing"), ("days", "7")]), tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters)["days"], 7)

    def test_union_type_all_null_falls_back_to_string(self):
        tools = _make_tools(
            {
                "city": {"type": "string"},
                "unit": {"type": ["null"]},
            }
        )
        detector = MiMoDetector()
        result = detector.detect_and_parse(
            _tool_call_text([("city", "Beijing"), ("unit", "celsius")]), tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters)["unit"], "celsius")

    def test_plain_types_unchanged(self):
        tools = _make_tools(
            {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "unit": {"type": "string"},
            }
        )
        detector = MiMoDetector()
        result = detector.detect_and_parse(
            _tool_call_text([("city", "Beijing"), ("days", "7"), ("unit", "celsius")]),
            tools,
        )
        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Beijing")
        self.assertEqual(params["days"], 7)
        self.assertEqual(params["unit"], "celsius")

    def test_no_tool_call_returns_normal_text(self):
        detector = MiMoDetector()
        result = detector.detect_and_parse("plain answer", _make_tools({}))
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "plain answer")


if __name__ == "__main__":
    import unittest

    unittest.main()
