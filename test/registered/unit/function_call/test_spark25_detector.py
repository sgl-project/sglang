"""Unit tests for Spark25Detector - no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.spark25_detector import Spark25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _xml(name: str, arguments: list[tuple[str, str]]) -> str:
    pairs = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        for key, value in arguments
    )
    return f"<tool_call>{name}{pairs}</tool_call>"


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="set_state",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "active": {"type": "boolean"},
                        "items": {"type": "array"},
                        "metadata": {"type": "object"},
                    },
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="now",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    ]


class TestSpark25DetectorDetectAndParse(CustomTestCase):
    def setUp(self):
        self.tools = _tools()
        self.detector = Spark25Detector()

    def test_spark25_parser_is_registered(self):
        self.assertIs(FunctionCallParser.ToolCallParserEnum["spark25"], Spark25Detector)

    def test_nonstream_parses_multiple_calls_and_preserves_normal_text(self):
        text = (
            "before"
            + _xml(
                "set_state",
                [
                    ("name", "上海"),
                    ("count", "42"),
                    ("ratio", "2.5"),
                    ("active", "1"),
                    ("items", '["a", "b"]'),
                    ("metadata", '{"source":"spark"}'),
                ],
            )
            + "middle"
            + _xml("now", [])
            + "after"
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "beforemiddleafter")
        self.assertEqual([call.tool_index for call in result.calls], [0, 1])
        self.assertEqual([call.name for call in result.calls], ["set_state", "now"])
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {
                "name": "上海",
                "count": 42,
                "ratio": 2.5,
                "active": True,
                "items": ["a", "b"],
                "metadata": {"source": "spark"},
            },
        )
        self.assertEqual(json.loads(result.calls[1].parameters), {})

    def test_null_and_conversion_fallbacks_match_spark2_5_protocol(self):
        text = _xml(
            "set_state",
            [
                ("name", "null"),
                ("count", "not-an-int"),
                ("active", "false"),
                ("undeclared", "42"),
            ],
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {
                "name": None,
                "count": "not-an-int",
                "active": False,
                "undeclared": "42",
            },
        )

    def test_malformed_block_is_text_and_unknown_tool_honors_policy(self):
        malformed = "x<tool_call></tool_call>y"
        result = self.detector.detect_and_parse(malformed, self.tools)
        self.assertEqual(result.normal_text, malformed)
        self.assertEqual(result.calls, [])

        unknown = _xml("missing", [("value", "1")])
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            result = self.detector.detect_and_parse(unknown, self.tools)
            self.assertEqual(result.normal_text, "")
            self.assertEqual(result.calls, [])
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = self.detector.detect_and_parse(unknown, self.tools)
            self.assertEqual(result.calls[0].name, "missing")
            self.assertEqual(json.loads(result.calls[0].parameters), {"value": "1"})

    def test_stream_end_flushes_partial_marker_and_required_stays_native(self):
        result = self.detector.parse_streaming_increment("plain<tool_", self.tools)
        self.assertEqual(result.normal_text, "plain")
        self.assertEqual(self.detector.finish(self.tools).normal_text, "<tool_")

        truncated = Spark25Detector()
        result = truncated.parse_streaming_increment(
            "plain<tool_call>set_state<arg_key>count</arg_key>", self.tools
        )
        self.assertEqual(result.normal_text, "plain")
        self.assertEqual(truncated.finish(self.tools).normal_text, "")

        self.assertFalse(self.detector.supports_structural_tag())
        self.assertTrue(self.detector.parses_required_natively())
        self.assertIs(
            FunctionCallParser(self.tools, "spark25").get_structure_constraint(
                "required"
            ),
            None,
        )


class TestSpark25DetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = _tools()

    def test_streaming_character_chunks_match_nonstream_result(self):
        text = (
            "answer:"
            + _xml("set_state", [("count", "42"), ("active", "0")])
            + _xml("now", [])
            + "done"
        )
        detector = Spark25Detector()
        normal_parts = []
        calls = []

        for character in text:
            result = detector.parse_streaming_increment(character, self.tools)
            normal_parts.append(result.normal_text)
            calls.extend(result.calls)
        end = detector.finish(self.tools)
        normal_parts.append(end.normal_text)
        calls.extend(end.calls)

        self.assertEqual("".join(normal_parts), "answer:done")
        self.assertEqual([call.tool_index for call in calls], [0, 1])
        self.assertEqual(
            json.loads(calls[0].parameters), {"count": 42, "active": False}
        )
        self.assertEqual(json.loads(calls[1].parameters), {})
        self.assertEqual(
            detector.prev_tool_call_arr,
            [
                {"name": "set_state", "arguments": {"count": 42, "active": False}},
                {"name": "now", "arguments": {}},
            ],
        )
        self.assertEqual(
            detector.streamed_args_for_tool,
            ['{"count":42,"active":false}', "{}"],
        )


class TestSpark25NumericArguments(CustomTestCase):
    def test_numeric_arguments_preserve_precision_and_fallbacks(self):
        cases = [
            ("9007199254740993", 9007199254740993),
            ("-9007199254740993", -9007199254740993),
            ("9223372036854775807", 9223372036854775807),
            ("18446744073709551615", 18446744073709551615),
            ("1700000000000000001", 1700000000000000001),
            (" 9007199254740993 ", 9007199254740993),
            ("9007199254740992", 9007199254740992),
            ("42", 42),
            ("+42", 42),
            ("1.25", 1.25),
            ("-2.5", -2.5),
            ("42.0", 42),
            ("1e3", 1000),
            ("not-a-number", "not-a-number"),
            ("null", None),
        ]
        for param_type in ("number", "float"):
            tools = _tools()
            tools[0].function.parameters["properties"]["ratio"]["type"] = param_type
            for raw_value, expected in cases:
                text = _xml("set_state", [("ratio", raw_value)])
                for chunk_size in (None, 1, 7, len(text)):
                    with self.subTest(
                        param_type=param_type,
                        raw_value=raw_value,
                        chunk_size=chunk_size,
                    ):
                        parser = FunctionCallParser(tools, "spark25")
                        if chunk_size is None:
                            normal_text, calls = parser.parse_non_stream(text)
                        else:
                            normal_parts = []
                            calls = []
                            for start in range(0, len(text), chunk_size):
                                normal, emitted = parser.parse_stream_chunk(
                                    text[start : start + chunk_size]
                                )
                                normal_parts.append(normal)
                                calls.extend(emitted)
                            normal, emitted = parser.parse_stream_end()
                            normal_parts.append(normal)
                            calls.extend(emitted)
                            normal_text = "".join(normal_parts)

                        self.assertEqual(normal_text, "")
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(calls[0].name, "set_state")
                        self.assertEqual(calls[0].tool_index, 0)
                        arguments = json.loads(calls[0].parameters)
                        self.assertEqual(arguments, {"ratio": expected})
                        self.assertIs(type(arguments["ratio"]), type(expected))


if __name__ == "__main__":
    unittest.main()
