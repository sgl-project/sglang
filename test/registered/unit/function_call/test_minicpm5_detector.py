import json
import sys

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minicpm5_detector import MiniCPM5Detector
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


def test_missing_required_param_call_dropped_v3():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = 'a<function name="get_weather"><param name="date">x</param></function>b'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert res.normal_text == "ab"


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
    assert "<function" not in res.normal_text


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
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["city"] == "北京"


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


def _exec_block(command: str) -> str:
    return f'<function name="exec"><param name="command">{command}</param></function>'


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<![CDATA[ls 2>&1]]>", "ls 2>&1"),
        ("ls 2>&1", "ls 2>&1"),
        ("<![CDATA[cat '<stdin>']]>", "cat '<stdin>'"),
        ("cat '<stdin>'", "cat '<stdin>'"),
    ],
)
def test_exec_special_chars_exact_stream_and_non_stream(raw, expected):
    tools = make_tools_exec()
    block = _exec_block(raw)

    non_stream = MiniCPM5Detector().detect_and_parse(block, tools)
    assert len(non_stream.calls) == 1
    assert json.loads(non_stream.calls[0].parameters) == {"command": expected}
    assert non_stream.normal_text == ""

    cut = block.index(expected) + len(expected) // 2
    detector = MiniCPM5Detector()
    r1 = detector.parse_streaming_increment(block[:cut], tools)
    assert r1.calls == []
    r2 = detector.parse_streaming_increment(block[cut:], tools)
    calls = r2.calls + detector.finish(tools).calls
    assert len(calls) == 1
    assert json.loads(calls[0].parameters) == {"command": expected}
    assert r1.normal_text + r2.normal_text == ""


def test_dropped_block_does_not_affect_following_call():
    tools = make_tools_exec()
    text = (
        '<function name="exec"><param name="nope">x</param></function>'
        + _exec_block("ls 2>&1")
    )
    res = MiniCPM5Detector().detect_and_parse(text, tools)
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {"command": "ls 2>&1"}
    assert res.normal_text == ""


def test_unknown_param_ignored_but_call_kept():
    tools = make_tools_exec()
    text = (
        '<function name="exec">'
        '<param name="nope">x</param>'
        '<param name="command">echo ok</param>'
        "</function>"
    )
    res = MiniCPM5Detector().detect_and_parse(text, tools)
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {"command": "echo ok"}


def test_entity_not_decoded():
    tools = make_tools_exec()
    res = MiniCPM5Detector().detect_and_parse(_exec_block("echo a &amp; b"), tools)
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {"command": "echo a &amp; b"}


def test_untyped_and_string_params_not_json_parsed():
    untyped_res = MiniCPM5Detector().detect_and_parse(
        '<function name="run"><param name="command">2</param></function>',
        make_tools_untyped_command(),
    )
    assert json.loads(untyped_res.calls[0].parameters) == {"command": "2"}

    str_res = MiniCPM5Detector().detect_and_parse(
        '<function name="store"><param name="items">[1,2,3]</param></function>',
        make_tools_string_list(),
    )
    assert json.loads(str_res.calls[0].parameters) == {"items": "[1,2,3]"}


@pytest.mark.parametrize(
    ("schema", "raw", "expected"),
    [
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "5", 5),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "5", "5"),
        ({"type": ["string", "null"]}, "5", "5"),
        ({"oneOf": [{"type": "boolean"}, {"type": "null"}]}, "true", True),
        ({"allOf": [{"type": "integer"}, {"minimum": 0}]}, "5", 5),
        ({"items": {"type": "integer"}}, "[1,2]", [1, 2]),
    ],
)
def test_parameter_schema_types(schema, raw, expected):
    tools = [
        Tool(
            function=Function(
                name="lookup",
                parameters={"type": "object", "properties": {"value": schema}},
            )
        )
    ]
    opening = '<function name="lookup"><param name="value">'
    closing = "</param></function>"
    full = MiniCPM5Detector().detect_and_parse(opening + raw + closing, tools)
    detector = MiniCPM5Detector()
    assert detector.parse_streaming_increment(opening + raw, tools).calls == []
    stream = detector.parse_streaming_increment(closing, tools)
    for result in (full, stream):
        assert [json.loads(call.parameters) for call in result.calls] == [
            {"value": expected}
        ]


def test_missing_param_close_stops_at_next_param():
    tools = make_tools_weather()
    text = (
        '<function name="get_weather">'
        '<param name="city">北京'
        '<param name="date">2024-06-27</param>'
        "</function>"
    )
    res = MiniCPM5Detector().detect_and_parse(text, tools)
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {"city": "北京", "date": "2024-06-27"}


def test_cdata_with_literal_close_tags():
    tools = make_tools_exec()
    text = (
        '<function name="exec">'
        '<param name="command"><![CDATA[doc </param> and </function>]]></param>'
        "</function>"
    )
    res = MiniCPM5Detector().detect_and_parse(text, tools)
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {
        "command": "doc </param> and </function>"
    }


def test_broken_cdata_falls_back_to_nearest_boundary():
    tools = make_tools_exec()
    text = (
        '<function name="exec">'
        '<param name="command"><![CDATA[ls 2>&1</param>'
        '<param name="ignored">x</param>'
        "</function>"
    )
    res = MiniCPM5Detector().detect_and_parse(text, tools)
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {"command": "ls 2>&1"}


def test_empty_param_value():
    res = MiniCPM5Detector().detect_and_parse(
        '<function name="noop"><param name="note"></param></function>',
        make_tools_no_required(),
    )
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {"note": ""}


def test_single_quotes_and_extra_space_in_tags():
    res = MiniCPM5Detector().detect_and_parse(
        "<function name='get_weather' ><param name='city'>北京</param></function>",
        make_tools_weather(),
    )
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {"city": "北京"}


def test_streaming_partial_function_token_buffered():
    detector = MiniCPM5Detector()
    tools = make_tools_exec()
    r1 = detector.parse_streaming_increment("Hello <func", tools)
    assert r1.normal_text == "Hello "
    assert r1.calls == []
    r2 = detector.parse_streaming_increment(
        'tion name="exec"><param name="command">echo hi</param></function>', tools
    )
    assert r2.normal_text == ""
    assert len(r2.calls) == 1


def test_streaming_broken_cdata_recovers_at_finish():
    detector = MiniCPM5Detector()
    tools = make_tools_exec()
    c1 = '<function name="exec"><param name="command"><![CDATA[ls 2>&1'
    c2 = "</function>"
    assert detector.parse_streaming_increment(c1, tools).calls == []
    assert detector.parse_streaming_increment(c2, tools).calls == []
    result = detector.finish(tools)
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == {"command": "ls 2>&1"}


def test_final_delta_drains_multiple_complete_calls():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    beijing = (
        '<function name="get_weather"><param name="city">Beijing</param></function>'
    )
    shanghai = beijing.replace("Beijing", "Shanghai")
    result = detector.parse_streaming_increment(beijing + shanghai, tools)
    assert [call.tool_index for call in result.calls] == [0, 1]
    assert [json.loads(call.parameters) for call in result.calls] == [
        {"city": "Beijing"},
        {"city": "Shanghai"},
    ]
    assert detector.finish(tools).calls == []


def test_streaming_text_between_and_after_calls():
    tools = make_tools_weather()
    detector = MiniCPM5Detector()
    block = '<function name="get_weather"><param name="city">Beijing</param></function>'
    first = detector.parse_streaming_increment(block, tools)
    between = detector.parse_streaming_increment(" Between. <func", tools)
    second = detector.parse_streaming_increment(
        block[len("<func") :].replace("Beijing", "Shanghai"), tools
    )
    trailing = detector.parse_streaming_increment(" After. <fun", tools)
    end = detector.finish(tools)
    assert [
        first.normal_text,
        between.normal_text,
        second.normal_text,
        trailing.normal_text,
        end.normal_text,
    ] == ["", " Between. ", "", " After. ", "<fun"]
    calls = first.calls + between.calls + second.calls + trailing.calls + end.calls
    assert [(c.tool_index, json.loads(c.parameters)) for c in calls] == [
        (0, {"city": "Beijing"}),
        (1, {"city": "Shanghai"}),
    ]


def test_cdata_literal_function_close_does_not_emit_early():
    detector = MiniCPM5Detector()
    tools = make_tools_exec()
    c1 = '<function name="exec"><param name="command"><![CDATA[doc </function>'
    c2 = " tail]]></param></function>"
    assert detector.parse_streaming_increment(c1, tools).calls == []
    result = detector.parse_streaming_increment(c2, tools)
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == {
        "command": "doc </function> tail",
    }


def test_literal_function_marker_does_not_hide_later_call():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    prefix = "Use <function> to describe a tool call."
    block = '<function name="get_weather"><param name="city">Beijing</param></function>'
    non_stream = detector.detect_and_parse(prefix + block, tools)
    assert len(non_stream.calls) == 1
    assert non_stream.normal_text == prefix

    first = detector.parse_streaming_increment(prefix, tools)
    second = detector.parse_streaming_increment(block, tools)
    assert first.normal_text + second.normal_text == prefix
    assert [json.loads(call.parameters) for call in second.calls] == [
        {"city": "Beijing"},
    ]


def test_finish_releases_partial_marker_in_plain_text():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    text = "Example: <func"
    streamed = detector.parse_streaming_increment(text, tools)
    finished = detector.finish(tools)
    assert streamed.normal_text + finished.normal_text == text
    assert streamed.calls + finished.calls == []


def test_cdata_preserves_literal_adjacent_function_markup():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    city = 'Beijing</function><function name="get_weather"><param name="city">example'
    opening = '<function name="get_weather"><param name="city"><![CDATA['
    first = detector.parse_streaming_increment(opening + city, tools)
    assert first.calls == []
    second = detector.parse_streaming_increment("]]></param></function>", tools)
    assert [json.loads(call.parameters) for call in second.calls] == [{"city": city}]


def test_finish_preserves_unclosed_function_text():
    detector = MiniCPM5Detector()
    tools = make_tools_weather()
    prefix = "Reading an example: "
    suffix = '<function name="get_weather"> has no closing tag. Keep this text.'
    streamed = detector.parse_streaming_increment(prefix + suffix, tools)
    finished = detector.finish(tools)
    assert streamed.calls + finished.calls == []
    assert streamed.normal_text + finished.normal_text == prefix + suffix
    assert detector.finish(tools).normal_text == ""


def test_complete_cdata_owns_all_text_through_its_terminator():
    tools = make_tools_weather()
    first = (
        '<function name="get_weather"><param name="city"><![CDATA[Beijing'
        "</param></function>"
    )
    second = (
        '\n<function name="get_weather">'
        '<param name="city"><![CDATA[Shanghai]]></param></function>'
    )
    expected = [
        {
            "city": 'Beijing</param></function>\n<function name="get_weather">'
            '<param name="city"><![CDATA[Shanghai'
        }
    ]
    non_stream = MiniCPM5Detector().detect_and_parse(first + second, tools)
    detector = MiniCPM5Detector()
    first_result = detector.parse_streaming_increment(first, tools)
    assert first_result.calls == []
    second_result = detector.parse_streaming_increment(second, tools)
    calls = second_result.calls + detector.finish(tools).calls
    assert [json.loads(call.parameters) for call in calls] == expected
    assert [call.tool_index for call in calls] == [0]
    assert [json.loads(call.parameters) for call in non_stream.calls] == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
