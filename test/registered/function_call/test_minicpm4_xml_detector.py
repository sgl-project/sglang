import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minicpm4_xml_detector import (
    MiniCPM4XmlFormatDetector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


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
    text = '<function name="unknown">' '<param name="x">1</param>' "</function>\n"
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert "unknown" in res.normal_text


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
    text = (
        '<function name="get_weather">'
        '<param name="city">北京</param>'
        # Missing </function>
    )
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
    assert "<param>北京</param>" in res.normal_text


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
    assert len(res.calls) == 0


def test_case_sensitive_param_name_invalid_v3():
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    text = (
        '<function name="get_weather">'
        '<param name="City">北京</param>'
        "</function>\n"
    )
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


def test_build_ebnf_contains_rules_v3():
    """Test build_ebnf - requires EBNFComposer (available in sglang >= 0.5.3.post1)."""
    detector = MiniCPM4XmlFormatDetector()
    tools = make_tools_weather()
    try:
        ebnf = detector.build_ebnf(tools)
        assert "call_get_weather" in ebnf
        assert "arguments_get_weather" in ebnf
        # V3 should not include <arguments> wrapper
        assert '"<arguments>"' not in ebnf
    except NotImplementedError:
        pytest.skip("EBNFComposer not available in this sglang version")


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
