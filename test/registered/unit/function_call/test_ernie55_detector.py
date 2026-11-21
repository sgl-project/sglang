# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
import re
import sys
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.ernie55_detector import Ernie5_5Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="get_weather",
            description="weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        ),
    )
]
_CALL = (
    "<ernie:tool_call>\n"
    '<invoke name="get_weather">\n'
    '<parameter name="city">Beijing</parameter>\n'
    '<parameter name="days">3</parameter>\n'
    "</invoke>\n"
    "</ernie:tool_call>"
)


class TestErnie5_5ToolCallParser(CustomTestCase):
    def test_types_come_from_the_tool_schema(self):
        text, calls = FunctionCallParser(_TOOLS, "ernie55").parse_non_stream(
            "I will look it up. " + _CALL
        )
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(calls[0].parameters), {"city": "Beijing", "days": 3}
        )
        self.assertEqual(text.strip(), "I will look it up.")

    def test_call_without_block_end_still_parses(self):
        text, calls = FunctionCallParser(_TOOLS, "ernie55").parse_non_stream(
            "hi " + _CALL.removesuffix("</ernie:tool_call>")
        )
        self.assertEqual(text.strip(), "hi")
        self.assertEqual(
            json.loads(calls[0].parameters), {"city": "Beijing", "days": 3}
        )

    def test_value_is_parsed_per_schema_type(self):
        for ptype, raw, expected in (
            (["string", "null"], "sglang", "sglang"),
            ("string", "null", "null"),
            ("string", "none", "none"),
            ("string", "nil", "nil"),
        ):
            with self.subTest(ptype=ptype, raw=raw):
                tools = [
                    Tool(
                        type="function",
                        function=Function(
                            name="run",
                            parameters={
                                "type": "object",
                                "properties": {"arg": {"type": ptype}},
                            },
                        ),
                    )
                ]
                call = (
                    '<ernie:tool_call>\n<invoke name="run">\n'
                    f'<parameter name="arg">{raw}</parameter>\n'
                    "</invoke>\n</ernie:tool_call>"
                )
                _, calls = FunctionCallParser(tools, "ernie55").parse_non_stream(call)
                self.assertEqual(json.loads(calls[0].parameters), {"arg": expected})


_S, _E = "<ernie:tool_call>", "</ernie:tool_call>"
_STREAM_TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="f",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "n": {"type": "integer"},
                    "x": {"type": "number"},
                },
            },
        ),
    )
]


def _block(*invokes, after=""):
    return f"{_S}\n" + "".join(invokes) + f"{_E}{after}"


def _invoke(name, *params, close=True):
    body = "".join(params)
    return f'<invoke name="{name}">\n{body}' + ("</invoke>\n" if close else "")


def _param(name, value, close=True):
    return f'<parameter name="{name}">{value}' + ("</parameter>\n" if close else "")


_STREAM_CASES = {
    "brace_in_value": _block(_invoke("f", _param("path", "\na}\n")), after="Done."),
    "no_parameters": _block(_invoke("f")),
    "parallel_calls": _block(
        _invoke("f", _param("path", "x}")), _invoke("f", _param("n", "3"))
    ),
    "undefined_then_defined": _block(
        _invoke("g", _param("a", "1")), _invoke("f", _param("n", "2")), after="after"
    ),
    "two_blocks": _block(_invoke("f", _param("n", "1")), after="mid")
    + _block(_invoke("f", _param("n", "2")), after="end"),
    "missing_invoke_end": _block(
        _invoke("f", _param("path", "x"), close=False), after="tail"
    ),
    "missing_parameter_end": _block(
        _invoke("f", _param("path", "x\n", close=False), close=False), after="tail"
    ),
    "undefined_missing_end": _block(
        _invoke("g", _param("a", "1"), close=False), after="tail"
    ),
    "null_word_string": _block(_invoke("f", _param("path", "null"))),
    "infinite_number": _block(_invoke("f", _param("x", "inf"))),
    "missing_block_end": f"{_S}\n" + _invoke("f", _param("n", "4")),
}


def _chunks(text, size):
    # The block tags are single tokens and arrive whole.
    for part in re.split(f"({re.escape(_S)}|{re.escape(_E)})", text):
        if part in (_S, _E):
            yield part
        else:
            yield from (part[i : i + size] for i in range(0, len(part), size))


def _stream(text, size):
    detector, normal, args = Ernie5_5Detector(), "", {}
    for chunk in _chunks(text, size):
        result = detector.parse_streaming_increment(chunk, _STREAM_TOOLS)
        normal += result.normal_text
        for call in result.calls:
            args[call.tool_index] = args.get(call.tool_index, "") + call.parameters
    return normal, [json.loads(a) for a in args.values()]


class TestErnie5_5ToolCallStreaming(CustomTestCase):
    def test_streaming_matches_non_streaming(self):
        for name, text in _STREAM_CASES.items():
            result = Ernie5_5Detector().detect_and_parse(text, _STREAM_TOOLS)
            expected = (
                result.normal_text,
                [json.loads(c.parameters) for c in result.calls],
            )
            for size in (1, 3, len(text)):
                with self.subTest(case=name, chunk=size):
                    self.assertEqual(_stream(text, size), expected)

    def test_streaming_yields_the_expected_calls(self):
        for name, normal, params in (
            ("brace_in_value", "Done.", [{"path": "a}"}]),
            ("parallel_calls", "", [{"path": "x}"}, {"n": 3}]),
        ):
            with self.subTest(case=name):
                self.assertEqual(_stream(_STREAM_CASES[name], 1), (normal, params))
                result = Ernie5_5Detector().detect_and_parse(
                    _STREAM_CASES[name], _STREAM_TOOLS
                )
                self.assertEqual([c.name for c in result.calls], ["f"] * len(params))

    def test_string_value_streams_before_its_end_tag(self):
        detector = Ernie5_5Detector()
        sent = ""
        for chunk in (
            _S,
            '\n<invoke name="f">\n<parameter name="path">',
            "/tmp/",
            "abc",
        ):
            for call in detector.parse_streaming_increment(chunk, _STREAM_TOOLS).calls:
                sent += call.parameters
        self.assertEqual(sent, '{"path": "/tmp/abc')

    def test_start_tag_split_over_chunks(self):
        text = "hi " + _block(_invoke("f", _param("n", "5")))
        for cut in range(len("hi ") + 1, len("hi ") + len(_S)):
            with self.subTest(cut=cut):
                detector, normal, args = Ernie5_5Detector(), "", ""
                for chunk in (text[:cut], text[cut:]):
                    result = detector.parse_streaming_increment(chunk, _STREAM_TOOLS)
                    normal += result.normal_text
                    args += "".join(c.parameters for c in result.calls)
                self.assertEqual((normal, json.loads(args)), ("hi ", {"n": 5}))

    def test_buffered_values_preserve_split_terminators_and_history(self):
        """Buffered typed values must not leak into cutoff history, lose a
        split terminator, or carry buffered text into the next parameter.
        """
        for kind, value in (
            ("object", {"text": "雪<not-a-tag>" * 512, "nested": [1, None, True]}),
            ("array", list(range(1200))),
            (["integer", "string"], "not-a-number" * 512),
        ):
            tools = [
                Tool(
                    type="function",
                    function=Function(
                        name="f",
                        parameters={
                            "type": "object",
                            "properties": {
                                "payload": {"type": kind},
                                "path": {"type": "string"},
                            },
                        },
                    ),
                )
            ]
            raw = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
            for tag in ("</parameter>", "</invoke>", _E):
                for cut in range(1, len(tag)):
                    with self.subTest(kind=kind, tag=tag, cut=cut):
                        detector = Ernie5_5Detector()
                        emitted = []

                        def feed(chunk):
                            result = detector.parse_streaming_increment(chunk, tools)
                            emitted.extend(c.parameters for c in result.calls)

                        feed(f'{_S}<invoke name="f"><parameter name="payload">')
                        # Leading/trailing newlines span several input chunks.
                        text = "\n" * 24 + raw + "\n" * 24
                        for pos in range(0, len(text), 7):
                            feed(text[pos : pos + 7])
                        feed(tag[:cut])
                        self.assertEqual(
                            detector.prev_tool_call_arr,
                            [{"name": "f", "arguments": {}}],
                        )
                        self.assertEqual(detector.streamed_args_for_tool, [""])
                        feed(tag[cut:])
                        expected = {"payload": value}
                        if tag == "</parameter>":
                            # A second buffered parameter and a streamed string
                            # exercise reset on both paths within the same call.
                            feed('<parameter name="payload">' + raw)
                            feed("</parameter>")
                            feed(_param("path", "after") + "</invoke>" + _E)
                            expected["path"] = "after"
                        elif tag == "</invoke>":
                            feed(_E)
                        args = "".join(emitted)
                        self.assertEqual(json.loads(args), expected)
                        self.assertEqual(detector.streamed_args_for_tool, [args])
                        self.assertEqual(
                            detector.prev_tool_call_arr,
                            [{"name": "f", "arguments": expected}],
                        )

    def test_buffered_conversion_error_preserves_pending_value(self):
        """A conversion error must not discard the buffered prefix and turn
        its leftover closing brackets into a new value on the next increment.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="f",
                    parameters={
                        "type": "object",
                        "properties": {"payload": {"type": "object"}},
                    },
                ),
            )
        ]
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        raw = '{"n":' + "9" * 5000 + "}"
        detector = Ernie5_5Detector()
        detector.parse_streaming_increment(
            f'{_S}<invoke name="f"><parameter name="payload">' + raw, tools
        )
        with self.assertLogs(
            "sglang.srt.function_call.ernie55_detector", level="ERROR"
        ):
            for chunk in ("</parameter>", "</invoke>", _E):
                self.assertEqual(
                    detector.parse_streaming_increment(chunk, tools).calls, []
                )
                self.assertEqual(detector.streamed_args_for_tool, [""])
                self.assertEqual(
                    detector.prev_tool_call_arr, [{"name": "f", "arguments": {}}]
                )


if __name__ == "__main__":
    unittest.main()
