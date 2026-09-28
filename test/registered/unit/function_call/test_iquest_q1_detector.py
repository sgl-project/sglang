import json
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.iquest_q1_detector import IQuestQ1Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestIQuestQ1Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="run",
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "count": {"type": "integer"},
                            "enabled": {"type": "boolean"},
                            "items": {"type": "array"},
                        },
                    },
                ),
            )
        ]
        self.call = (
            "<iquest_tool_call>run\n"
            '<arg_key>code</arg_key><arg_value>  "123"\n</arg_value>'
            "<arg_key>count</arg_key> <arg_value>42</arg_value>\n"
            "<arg_key>enabled</arg_key><arg_value>true</arg_value>"
            "<arg_key>items</arg_key><arg_value>[1, null]</arg_value>"
            "</iquest_tool_call>"
        )
        self.expected = {
            "code": '  "123"\n',
            "count": 42,
            "enabled": True,
            "items": [1, None],
        }

    def test_non_stream_typed_arguments_and_leading_text(self):
        parser = FunctionCallParser(self.tools, "iquest_q1")
        text, calls = parser.parse_non_stream("Running now. " + self.call)
        self.assertEqual(text, "Running now. ")
        self.assertEqual(calls[0].name, "run")
        self.assertEqual(json.loads(calls[0].parameters), self.expected)

    def test_streaming_every_boundary_and_mtp_finish(self):
        wire = "Running now. " + self.call + self.call
        for size in (1, 2, 3, 7, 13, len(wire)):
            with self.subTest(size=size):
                parser = FunctionCallParser(self.tools, "iquest_q1")
                normal, calls = [], []
                for start in range(0, len(wire), size):
                    text, chunk_calls = parser.parse_stream_chunk(
                        wire[start : start + size]
                    )
                    normal.append(text)
                    calls.extend(chunk_calls)
                text, final_calls = parser.parse_stream_end()
                normal.append(text)
                calls.extend(final_calls)
                self.assertEqual("".join(normal), "Running now. ")
                self.assertEqual([call.tool_index for call in calls], [0, 1])
                self.assertEqual(
                    [json.loads(call.parameters) for call in calls],
                    [self.expected, self.expected],
                )
                detector = parser.detector
                self.assertEqual(len(detector.prev_tool_call_arr), 2)
                for action, sent in zip(
                    detector.prev_tool_call_arr, detector.streamed_args_for_tool
                ):
                    self.assertEqual(action["arguments"], json.loads(sent))

    def test_surrounding_text_is_preserved_in_each_mode(self):
        for wire, expected_text, call_count in (
            ("before" + self.call + "after", "beforeafter", 1),
            (self.call + "\nafter", "\nafter", 1),
            (
                "before" + self.call + "between" + self.call + "after",
                "beforebetweenafter",
                2,
            ),
        ):
            with self.subTest(wire=wire):
                parser = FunctionCallParser(self.tools, "iquest_q1")
                text, calls = parser.parse_non_stream(wire)
                self.assertEqual(text, expected_text)
                self.assertEqual(
                    [json.loads(call.parameters) for call in calls],
                    [self.expected] * call_count,
                )
                for size in (1, 5, len(wire)):
                    with self.subTest(size=size):
                        parser = FunctionCallParser(self.tools, "iquest_q1")
                        normal, streamed = [], []
                        for start in range(0, len(wire), size):
                            chunk, chunk_calls = parser.parse_stream_chunk(
                                wire[start : start + size]
                            )
                            normal.append(chunk)
                            streamed.extend(chunk_calls)
                        tail, tail_calls = parser.parse_stream_end()
                        normal.append(tail)
                        streamed.extend(tail_calls)
                        self.assertEqual("".join(normal), expected_text)
                        self.assertEqual(streamed, calls)

    def test_non_stream_preserves_unparsed_text_around_valid_calls(self):
        malformed = (
            "<iquest_tool_call>run<arg_key>count</arg_key>"
            "<arg_value>1</iquest_tool_call>"
        )
        incomplete = "<iquest_tool_call>run<arg_key>code</arg_key>"
        for wire, expected_text in (
            (malformed + self.call + "after", malformed + "after"),
            (self.call + malformed + self.call, malformed),
            (self.call + "after" + incomplete, "after" + incomplete),
        ):
            with self.subTest(wire=wire):
                parser = FunctionCallParser(self.tools, "iquest_q1")
                text, calls = parser.parse_non_stream(wire)
                self.assertEqual(text, expected_text)
                self.assertTrue(calls)
                for call in calls:
                    self.assertEqual(json.loads(call.parameters), self.expected)

    def test_empty_arguments_and_malformed_call(self):
        detector = IQuestQ1Detector()
        result = detector.detect_and_parse(
            "<iquest_tool_call>run</iquest_tool_call>", self.tools
        )
        self.assertEqual(json.loads(result.calls[0].parameters), {})
        malformed = (
            "<iquest_tool_call>run<arg_key>count</arg_key>"
            "<arg_value>1</iquest_tool_call>"
        )
        result = detector.detect_and_parse(malformed, self.tools)
        self.assertEqual(result.normal_text, malformed)
        self.assertFalse(result.calls)
        result = detector.parse_streaming_increment(malformed, self.tools)
        self.assertEqual(result.normal_text, malformed)
        self.assertFalse(result.calls)

    def test_disabled_tools_and_truncated_marker(self):
        parser = FunctionCallParser([], "iquest_q1")
        self.assertEqual(parser.parse_non_stream(self.call), (self.call, []))
        self.assertEqual(parser.parse_stream_chunk(self.call), (self.call, []))
        parser = FunctionCallParser(self.tools, "iquest_q1")
        text, calls = parser.parse_stream_chunk("Text <iquest_")
        self.assertEqual((text, calls), ("Text ", []))
        self.assertEqual(parser.parse_stream_end(), ("<iquest_", []))
        self.assertEqual(parser.parse_stream_end(), ("", []))

    def test_incomplete_call_is_preserved_at_stream_end(self):
        wire = "before<iquest_tool_call>run<arg_key>code</arg_key>"
        parser = FunctionCallParser(self.tools, "iquest_q1")
        self.assertEqual(parser.parse_non_stream(wire), (wire, []))
        self.assertEqual(parser.parse_stream_chunk(wire), ("before", []))
        self.assertEqual(parser.parse_stream_end(), (wire[len("before") :], []))

    def test_unknown_tool_name_is_filtered(self):
        wire = self.call.replace("<iquest_tool_call>run", "<iquest_tool_call>unknown")
        parser = FunctionCallParser(self.tools, "iquest_q1")
        text, calls = parser.parse_non_stream(wire)
        self.assertFalse(calls)
        parser = FunctionCallParser(self.tools, "iquest_q1")
        text, calls = parser.parse_stream_chunk(wire)
        self.assertFalse(calls)

    def test_schema_reference_preserves_numeric_string(self):
        self.tools[0].function.parameters = {
            "$defs": {"code": {"type": "string"}},
            "type": "object",
            "properties": {"code": {"$ref": "#/$defs/code"}},
        }
        wire = "<iquest_tool_call>run<arg_key>code</arg_key><arg_value>42</arg_value></iquest_tool_call>"
        parser = FunctionCallParser(self.tools, "iquest_q1")
        self.assertEqual(
            json.loads(parser.parse_non_stream(wire)[1][0].parameters), {"code": "42"}
        )
        parser = FunctionCallParser(self.tools, "iquest_q1")
        self.assertEqual(
            json.loads(parser.parse_stream_chunk(wire)[1][0].parameters), {"code": "42"}
        )

    def test_required_and_named_constraints_match_release(self):
        parser = FunctionCallParser(self.tools, "iquest_q1")
        kind, schema = parser.get_structure_constraint("required")
        self.assertEqual(kind, "json_schema")
        self.assertEqual(schema["type"], "array")
        self.assertEqual(schema["minItems"], 1)
        named = ToolChoice(function=ToolChoiceFuncName(name="run"))
        kind, schema = parser.get_structure_constraint(named)
        self.assertEqual(kind, "json_schema")
        self.assertEqual(schema, self.tools[0].function.parameters)
        self.assertIsNone(parser.get_structure_constraint("auto"))

    def test_required_streaming_parallel_calls_at_every_boundary(self):
        wire = json.dumps(
            [
                {"name": "run", "parameters": self.expected},
                {"name": "run", "parameters": {"code": "你好"}},
            ],
            ensure_ascii=False,
        )
        for size in (1, 2, 7, len(wire)):
            with self.subTest(size=size):
                parser = FunctionCallParser(
                    self.tools, "iquest_q1", tool_choice="required"
                )
                names, arguments = {}, {}
                for start in range(0, len(wire), size):
                    text, calls = parser.parse_stream_chunk(wire[start : start + size])
                    self.assertEqual(text, "")
                    for call in calls:
                        if call.name:
                            self.assertNotIn(call.tool_index, names)
                            names[call.tool_index] = call.name
                        arguments[call.tool_index] = (
                            arguments.get(call.tool_index, "") + call.parameters
                        )
                self.assertEqual(names, {0: "run", 1: "run"})
                self.assertEqual(json.loads(arguments[0]), self.expected)
                self.assertEqual(json.loads(arguments[1]), {"code": "你好"})
                self.assertEqual(parser.parse_stream_end(), ("", []))

    def test_named_streaming_keeps_raw_arguments_and_emits_name_once(self):
        wire = '{ "code": "你好", "items": [1, null] }'
        for size in (1, 5, len(wire)):
            with self.subTest(size=size):
                named = ToolChoice(function=ToolChoiceFuncName(name="run"))
                parser = FunctionCallParser(self.tools, "iquest_q1", tool_choice=named)
                self.assertEqual(
                    parser.get_structure_constraint(named),
                    ("json_schema", self.tools[0].function.parameters),
                )
                calls = []
                for start in range(0, len(wire), size):
                    calls.extend(
                        parser.parse_stream_chunk(wire[start : start + size])[1]
                    )
                self.assertEqual([call.name for call in calls if call.name], ["run"])
                self.assertTrue(all(call.tool_index == 0 for call in calls))
                self.assertEqual("".join(call.parameters for call in calls), wire)
                self.assertEqual(parser.parse_stream_end(), ("", []))


class TestIQuestQ1StopMarkers(CustomTestCase):
    def _tools(self):
        return [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Query the weather of a city",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def test_parser_preserves_stop_marker_text_like_release(self):
        detector = IQuestQ1Detector()
        wire = (
            "查询中。<iquest_tool_call>get_weather\n"
            "<arg_key>city</arg_key><arg_value>北京</arg_value>\n"
            "</iquest_tool_call><|iquest_end|>"
        )
        content, calls = "", []
        for size in (1, 5, len(wire)):
            detector = IQuestQ1Detector()
            content, calls = "", []
            for start in range(0, len(wire), size):
                result = detector.parse_streaming_increment(
                    wire[start : start + size], self._tools()
                )
                content += result.normal_text or ""
                calls.extend(result.calls)
            with self.subTest(chunk=size):
                self.assertEqual(content, "查询中。<|iquest_end|>")
                self.assertEqual(len(calls), 1)

    def test_no_global_stop_marker_replacement(self):
        detector = IQuestQ1Detector()
        result = detector.detect_and_parse("好的。<|iquest_end|>", self._tools())
        self.assertEqual(result.normal_text, "好的。<|iquest_end|>")

    def test_plain_reply_stop_markers_at_every_stream_split(self):
        for marker in ("<|iquest_end|>", "<|endoftext|>"):
            wire = "好的。" + marker
            for split in range(len(wire) + 1):
                with self.subTest(marker=marker, split=split):
                    parser = FunctionCallParser(self._tools(), "iquest_q1")
                    content, calls = [], []
                    for chunk in (wire[:split], wire[split:]):
                        text, parsed = parser.parse_stream_chunk(chunk)
                        content.append(text)
                        calls.extend(parsed)
                    text, parsed = parser.parse_stream_end()
                    content.append(text)
                    calls.extend(parsed)
                    self.assertEqual("".join(content), wire)
                    self.assertEqual(calls, [])
                    self.assertEqual(parser.parse_non_stream(wire), (wire, []))


class TestIQuestQ1NonStreamingServingPath(CustomTestCase):
    class _Serving:
        tool_call_parser = "iquest_q1"
        tokenizer_manager = SimpleNamespace(tokenizer=None)

        @staticmethod
        def _process_tool_call_id(call_info, history_tool_calls_cnt):
            return f"functions.{call_info.name}:{history_tool_calls_cnt}"

    def _tools(self):
        return [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            )
        ]

    def _process(self, text, tool_choice="auto", finish_type="stop"):
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        return OpenAIServingChat._process_tool_calls(
            self._Serving(),
            text,
            self._tools(),
            {"type": finish_type, "matched": None},
            tool_choice=tool_choice,
        )

    def test_reply_without_a_call_is_unchanged_by_serving(self):
        result = self._process("好的。<|iquest_end|>")
        self.assertEqual(result.remaining_text, "好的。<|iquest_end|>")
        self.assertIsNone(result.tool_calls)
        self.assertEqual(result.finish_reason["type"], "stop")

    def test_serving_preserves_text_around_auto_calls(self):
        call = (
            "<iquest_tool_call>get_weather\n"
            "<arg_key>city</arg_key><arg_value>北京</arg_value>\n"
            "</iquest_tool_call>"
        )
        for wire, expected_text, call_count in (
            ("before" + call + "after", "beforeafter", 1),
            (call + "\nafter", "\nafter", 1),
            (
                "before" + call + "between" + call + "after",
                "beforebetweenafter",
                2,
            ),
        ):
            with self.subTest(wire=wire):
                result = self._process(wire)
                self.assertEqual(result.remaining_text, expected_text)
                self.assertEqual(
                    [call.function.name for call in result.tool_calls],
                    ["get_weather"] * call_count,
                )
                self.assertEqual(result.finish_reason["type"], "tool_calls")

    def test_required_array_and_named_arguments(self):
        named = ToolChoice(function=ToolChoiceFuncName(name="get_weather"))
        for choice, wire in (
            (
                "required",
                '[{"name":"get_weather","parameters":{"city":"北京"}}]',
            ),
            (named, '{ "city": "北京" }'),
        ):
            with self.subTest(choice=choice):
                result = self._process(wire, choice)
                self.assertEqual(result.remaining_text, "")
                self.assertEqual(len(result.tool_calls), 1)
                self.assertEqual(result.tool_calls[0].function.name, "get_weather")
                self.assertEqual(
                    json.loads(result.tool_calls[0].function.arguments),
                    {"city": "北京"},
                )
                if choice == named:
                    self.assertEqual(result.tool_calls[0].function.arguments, wire)
                self.assertEqual(result.finish_reason["type"], "tool_calls")

    def test_finish_reason_keeps_length_with_tool_calls(self):
        named = ToolChoice(function=ToolChoiceFuncName(name="get_weather"))
        for choice, wire in (
            (
                "auto",
                "<iquest_tool_call>get_weather</iquest_tool_call>",
            ),
            ("required", '[{"name":"get_weather","parameters":{}}]'),
            (named, "{}"),
        ):
            with self.subTest(choice=choice):
                result = self._process(wire, choice, "length")
                self.assertEqual(result.finish_reason["type"], "length")


if __name__ == "__main__":
    unittest.main()
