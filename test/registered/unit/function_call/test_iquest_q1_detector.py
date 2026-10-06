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

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


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
        result = detector.parse_streaming_increment(malformed, self.tools)
        self.assertEqual(result.normal_text, malformed)
        self.assertFalse(result.calls)

    def test_truncated_marker_is_flushed_at_stream_end(self):
        parser = FunctionCallParser(self.tools, "iquest_q1")
        text, calls = parser.parse_stream_chunk("Text <iquest_")
        self.assertEqual((text, calls), ("Text ", []))
        self.assertEqual(parser.parse_stream_end(), ("<iquest_", []))
        self.assertEqual(parser.parse_stream_end(), ("", []))
        parser = FunctionCallParser(self.tools, "iquest_q1")
        self.assertEqual(parser.parse_stream_chunk("a <"), ("a ", []))
        self.assertEqual(parser.parse_stream_chunk("| b"), ("<| b", []))

    def test_incomplete_call_is_preserved_at_stream_end(self):
        wire = "before<iquest_tool_call>run<arg_key>code</arg_key>"
        parser = FunctionCallParser(self.tools, "iquest_q1")
        self.assertEqual(parser.parse_non_stream(wire), (wire, []))
        self.assertEqual(parser.parse_stream_chunk(wire), ("before", []))
        self.assertEqual(parser.parse_stream_end(), (wire[len("before") :], []))

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


if __name__ == "__main__":
    unittest.main()
