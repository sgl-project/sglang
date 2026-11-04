import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minicpm5_detector import (
    MiniCPM5Detector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def make_tools_weather():
    return [
        Tool(
            function=Function(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["city"],
                },
            )
        )
    ]


def make_tools_sum():
    return [
        Tool(
            function=Function(
                name="sum_values",
                parameters={
                    "type": "object",
                    "properties": {
                        "nums": {"type": "array"},
                        "exact": {"type": "boolean"},
                    },
                    "required": ["nums"],
                },
            )
        )
    ]


def make_tools_config():
    return [
        Tool(
            function=Function(
                name="set_config",
                parameters={
                    "type": "object",
                    "properties": {
                        "config": {"type": "object"},
                    },
                    "required": ["config"],
                },
            )
        )
    ]


def make_tools_no_required():
    return [
        Tool(
            function=Function(
                name="noop",
                parameters={
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                    "required": [],
                },
            )
        )
    ]


def test_detect_and_parse_single_call_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = (
        "Intro before.\n"
        '<function name="get_weather">'
        '<param name="city">上海</param>'
        '<param name="date">2024-06-27</param>'
        "</function>\n"
        "Outro after.\n"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["city"] == "上海"
    assert args["date"] == "2024-06-27"
    assert "Intro before." in res.normal_text and "Outro after." in res.normal_text
    assert "<tool_sep>" not in res.normal_text


def test_detect_and_parse_cdata_multiline_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = (
        '<function name="get_weather">'
        '<param name="city"><![CDATA[北\n京]]></param>'
        '<param name="date">2024-06-27</param>'
        "</function>\n"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["city"] == "北\n京"
    assert args["date"] == "2024-06-27"


def test_unknown_tool_block_preserved_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = '<function name="unknown"><param name="x">1</param></function>\n'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert "unknown" in res.normal_text


def test_non_string_types_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_sum()
    text = (
        '<function name="sum_values">'
        '<param name="nums">[1, 2, 3]</param>'
        '<param name="exact">true</param>'
        "</function>\n"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["nums"] == [1, 2, 3]
    assert args["exact"] is True


def test_multiple_calls_interleaved_text_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather() + make_tools_sum()
    text = (
        "Head\n"
        '<function name="get_weather"><param name="city">北京</param></function>\n'
        "TXT\n"
        '<function name="sum_values"><param name="nums">[7,8,9]</param><param name="exact">false</param></function>\n'
        "Tail\n"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 2
    args0 = json.loads(res.calls[0].parameters)
    assert args0["city"] == "北京"
    args1 = json.loads(res.calls[1].parameters)
    assert args1["nums"] == [7, 8, 9]
    assert args1["exact"] is False
    assert (
        "Head" in res.normal_text
        and "TXT" in res.normal_text
        and "Tail" in res.normal_text
    )
    assert "<tool_sep>" not in res.normal_text


def test_incomplete_missing_function_end_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = '<function name="get_weather"><param name="city">北京</param>'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert "get_weather" in res.normal_text


def test_param_missing_name_invalid_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = (
        '<function name="get_weather">'
        "<param>北京</param>"
        '<param name="date">2024-06-27</param>'
        "</function>\n"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert "<param>北京</param>" in res.normal_text


def test_duplicate_param_names_invalid_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = (
        '<function name="get_weather">'
        '<param name="city">北京</param>'
        '<param name="city">上海</param>'
        "</function>\n"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0


def test_case_sensitive_param_name_invalid_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = '<function name="get_weather"><param name="City">北京</param></function>\n'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0


def test_no_required_and_zero_param_valid_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_no_required()
    text = '<function name="noop"></function>\n'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args == {}


def test_streaming_increment_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    c1 = 'Hello\n<function name="get_weather">\n  <param name="city">'
    c2 = '北京</param>\n  <param name="date">2024-06-27</param>\n</function>\n'

    r1 = detector.parse_streaming_increment(c1, tools)
    assert r1.normal_text == "Hello\n"
    assert len(r1.calls) == 0

    r2 = detector.parse_streaming_increment(c2, tools)
    assert len(r2.calls) == 1
    args = json.loads(r2.calls[0].parameters)
    assert args["city"] == "北京"
    assert args["date"] == "2024-06-27"


def test_streaming_split_bot_token():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = '<function name="get_weather"><param name="city">北京</param></function>'

    r1 = detector.parse_streaming_increment("<", tools)
    assert r1.normal_text == ""
    assert len(r1.calls) == 0

    r2 = detector.parse_streaming_increment(text[1:], tools)
    assert len(r2.calls) == 1
    args = json.loads(r2.calls[0].parameters)
    assert args["city"] == "北京"


def test_streaming_multiple_complete_blocks_in_one_delta():
    detector = MiniCPM5Detector()
    tools = make_tools_weather() + make_tools_sum()
    text = (
        '<function name="get_weather"><param name="city">北京</param></function>'
        '<function name="sum_values"><param name="nums">[1,2]</param></function>'
    )

    result = detector.parse_streaming_increment(text, tools)
    assert len(result.calls) == 2
    assert json.loads(result.calls[0].parameters)["city"] == "北京"
    assert json.loads(result.calls[1].parameters)["nums"] == [1, 2]


def test_malformed_xml_with_unescaped_ampersand_falls_back_to_regex():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = '<function name="get_weather"><param name="city">A & B</param></function>'

    result = detector.detect_and_parse(text, tools)
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters)["city"] == "A & B"


class TestMiniCPM5CDATA(CustomTestCase):
    def test_function_end_inside_cdata_is_argument_text(self):
        """Writing XML containing </function> must not truncate a CDATA argument
        or lose the tool call, including when that token spans stream chunks."""
        value = '<function name="inner">content</function>'
        text = (
            '<function name="noop"><param name="note"><![CDATA['
            + value
            + "]]></param></function>"
        )
        tools = make_tools_no_required()
        result = MiniCPM5Detector().detect_and_parse(text, tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"note": value})
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                detector = MiniCPM5Detector()
                calls = []
                for chunk in (text[:split], text[split:]):
                    calls.extend(detector.parse_streaming_increment(chunk, tools).calls)
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(calls[0].parameters), {"note": value})

    def test_unfinished_cdata_never_emits_a_partial_call(self):
        """A literal closing-function tag inside an unfinished CDATA region is
        not permission to invoke a tool with incomplete arguments."""
        tools = make_tools_no_required()
        prefix = '<function name="noop"><param name="note"><![CDATA[text </function>'
        detector = MiniCPM5Detector()
        self.assertEqual(detector.parse_streaming_increment(prefix, tools).calls, [])
        result = detector.parse_streaming_increment(
            " tail]]></param></function>", tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"note": "text </function> tail"}
        )

    def test_cdata_markers_in_prose_or_comments_do_not_hide_calls(self):
        """Only CDATA inside a function frames its arguments; prose and XML
        comments can describe its opener without swallowing a later call."""
        call = '<function name="noop"><param name="note">ok</param></function>'
        cases = [
            ("Example <![CDATA[ marker.\n" + call, 1),
            (call + " Example <![CDATA[ marker. " + call, 2),
            (call.replace("<param", "<!-- <![CDATA[ -->" + "<param", 1), 1),
        ]
        for text, count in cases:
            for streaming in (False, True):
                with self.subTest(text=text, streaming=streaming):
                    detector = MiniCPM5Detector()
                    if streaming:
                        calls = detector.parse_streaming_increment(
                            text, make_tools_no_required()
                        ).calls
                    else:
                        calls = detector.detect_and_parse(
                            text, make_tools_no_required()
                        ).calls
                    self.assertEqual(len(calls), count)
                    self.assertEqual(
                        [json.loads(item.parameters) for item in calls],
                        [{"note": "ok"}] * count,
                    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
