import json
import sys

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minicpm4_xml_detector import MiniCPM4XmlFormatDetector
from sglang.test.ci.ci_register import register_cpu_ci

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


def make_tools_exec():
    return [
        Tool(
            function=Function(
                name="exec",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
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


def make_tools_untyped_command():
    return [
        Tool(
            function=Function(
                name="run",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            )
        )
    ]


def make_tools_string_list():
    return [
        Tool(
            function=Function(
                name="store",
                parameters={
                    "type": "object",
                    "properties": {
                        "items": {"type": "string"},
                    },
                    "required": ["items"],
                },
            )
        )
    ]


def test_detect_and_parse_single_call_v3():
    detector = MiniCPM4XmlFormatDetector()
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
    detector = MiniCPM4XmlFormatDetector()
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
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = '<function name="unknown"><param name="x">1</param></function>\n'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert "unknown" in res.normal_text


def test_missing_required_param_call_dropped_v3():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = 'a<function name="get_weather"><param name="date">x</param></function>b'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert res.normal_text == "ab"


def test_non_string_types_v3():
    detector = MiniCPM4XmlFormatDetector()
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
    detector = MiniCPM4XmlFormatDetector()
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
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = '<function name="get_weather"><param name="city">北京</param>'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert "get_weather" in res.normal_text


def test_param_missing_name_invalid_v3():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = (
        '<function name="get_weather">'
        "<param>北京</param>"
        '<param name="date">2024-06-27</param>'
        "</function>\n"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert "<function" not in res.normal_text


def test_duplicate_param_names_invalid_v3():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = (
        '<function name="get_weather">'
        '<param name="city">北京</param>'
        '<param name="city">上海</param>'
        "</function>\n"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["city"] == "北京"


def test_case_sensitive_param_name_invalid_v3():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = '<function name="get_weather"><param name="City">北京</param></function>\n'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0


def test_no_required_and_zero_param_valid_v3():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_no_required()
    text = '<function name="noop"></function>\n'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args == {}


def test_streaming_increment_v3():
    detector = MiniCPM4XmlFormatDetector()
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


def _exec_block(command: str) -> str:
    return f'<function name="exec"><param name="command">{command}</param></function>'


@pytest.mark.parametrize(
    "command",
    [
        "<![CDATA[ls 2>&1]]>",
        "ls 2>&1",
        "<![CDATA[cat '<stdin>']]>",
        "cat '<stdin>'",
    ],
)
def test_exec_special_chars_non_stream(command):
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    res = detector.detect_and_parse(_exec_block(command), tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert "2>&1" in args["command"] or "<stdin>" in args["command"]
    assert "<function" not in res.normal_text


@pytest.mark.parametrize(
    "text,expected_calls",
    [
        (_exec_block("ls 2>&1"), 1),
        (_exec_block("echo hi") + _exec_block("ls 2>&1"), 2),
        (_exec_block("ls 2>&1") + _exec_block("echo hi"), 2),
        (
            '<function name="exec"><param name="nope">x</param></function>'
            + _exec_block("ls 2>&1"),
            1,
        ),
        (_exec_block("ls 2>&1") + _exec_block("ls 2>&1"), 2),
    ],
)
def test_multi_block_no_state_leak(text, expected_calls):
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == expected_calls
    assert "<function" not in res.normal_text


def test_unknown_param_ignored_but_call_kept():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    text = (
        '<function name="exec">'
        '<param name="nope">x</param>'
        '<param name="command">echo ok</param>'
        "</function>"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["command"] == "echo ok"


def test_entity_not_decoded():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    text = _exec_block("echo a &amp; b")
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["command"] == "echo a &amp; b"


def test_untyped_param_stays_string():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_untyped_command()
    text = '<function name="run"><param name="command">2</param></function>'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["command"] == "2"
    assert isinstance(args["command"], str)


def test_string_param_not_json_parsed():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_string_list()
    text = '<function name="store"><param name="items">[1,2,3]</param></function>'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["items"] == "[1,2,3]"


def test_missing_param_close_stops_at_next_param():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = (
        '<function name="get_weather">'
        '<param name="city">北京'
        '<param name="date">2024-06-27</param>'
        "</function>"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["city"] == "北京"
    assert args["date"] == "2024-06-27"


def test_cdata_with_literal_close_tags():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    text = (
        '<function name="exec">'
        '<param name="command"><![CDATA[doc </param> and </function>]]></param>'
        "</function>"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["command"] == "doc </param> and </function>"


def test_broken_cdata_falls_back_to_nearest_boundary():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    text = (
        '<function name="exec">'
        '<param name="command"><![CDATA[ls 2>&1</param>'
        '<param name="ignored">x</param>'
        "</function>"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["command"] == "ls 2>&1"


def test_mixed_cdata_and_plain_in_one_call():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    text = (
        '<function name="exec">'
        '<param name="command">echo hi</param>'
        '<param name="note"><![CDATA[a & b]]></param>'
        "</function>"
    )
    tools = [
        Tool(
            function=Function(
                name="exec",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["command"],
                },
            )
        )
    ]
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["command"] == "echo hi"
    assert args["note"] == "a & b"


def test_mixed_compliant_and_violating_blocks():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    text = _exec_block("<![CDATA[ls 2>&1]]>") + _exec_block("ls 2>&1")
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 2


def test_empty_param_value():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_no_required()
    text = '<function name="noop"><param name="note"></param></function>'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["note"] == ""


def test_single_quote_param_name():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = "<function name='get_weather'><param name='city'>北京</param></function>"
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["city"] == "北京"


def test_function_tag_with_extra_space():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = '<function name="get_weather" ><param name="city">北京</param></function>'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1


def test_plain_text_without_function():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = "Just a normal answer."
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert res.normal_text == text


def test_empty_tools_list():
    detector = MiniCPM4XmlFormatDetector()
    text = _exec_block("echo hi")
    res = detector.detect_and_parse(text, tools=[])
    assert len(res.calls) == 0
    assert "<function" in res.normal_text


@pytest.mark.parametrize(
    "command",
    ["ls 2>&1", "cat '<stdin>'"],
)
def test_streaming_exec_special_chars(command):
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    block = _exec_block(command)
    res = detector.parse_streaming_increment(block, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert command in args["command"] or args["command"] == command.strip()
    assert "<function" not in (res.normal_text or "")


def test_streaming_value_split_across_chunks():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    c1 = '<function name="exec"><param name="command">ls 2>'
    c2 = "&1</param></function>"
    r1 = detector.parse_streaming_increment(c1, tools)
    assert len(r1.calls) == 0
    r2 = detector.parse_streaming_increment(c2, tools)
    assert len(r2.calls) == 1
    args = json.loads(r2.calls[0].parameters)
    assert args["command"] == "ls 2>&1"


def test_streaming_multiple_calls_across_chunks():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    c1 = _exec_block("echo hi")
    c2 = _exec_block("ls 2>&1")
    r1 = detector.parse_streaming_increment(c1, tools)
    assert len(r1.calls) == 1
    r2 = detector.parse_streaming_increment(c2, tools)
    assert len(r2.calls) == 1
    assert detector.current_tool_id == 2


def test_streaming_cdata_across_chunks():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    c1 = '<function name="exec"><param name="command"><![CDATA[ls 2>&'
    c2 = "1]]></param></function>"
    r1 = detector.parse_streaming_increment(c1, tools)
    assert len(r1.calls) == 0
    r2 = detector.parse_streaming_increment(c2, tools)
    assert len(r2.calls) == 1
    args = json.loads(r2.calls[0].parameters)
    assert args["command"] == "ls 2>&1"


def test_streaming_partial_function_token_buffered():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    c1 = "Hello <func"
    c2 = 'tion name="exec"><param name="command">echo hi</param></function>'
    r1 = detector.parse_streaming_increment(c1, tools)
    assert r1.normal_text == ""
    assert len(r1.calls) == 0
    r2 = detector.parse_streaming_increment(c2, tools)
    assert r2.normal_text == "Hello "
    assert len(r2.calls) == 1


def test_streaming_broken_cdata_uses_function_close():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_exec()
    c1 = '<function name="exec"><param name="command"><![CDATA[ls 2>&1'
    c2 = "</function>"
    r1 = detector.parse_streaming_increment(c1, tools)
    assert len(r1.calls) == 0
    r2 = detector.parse_streaming_increment(c2, tools)
    assert len(r2.calls) == 1
    args = json.loads(r2.calls[0].parameters)
    assert "2>&1" in args["command"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
