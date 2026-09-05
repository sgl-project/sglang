"""Unit tests for GigaChat3Detector - no server, no model loading."""

import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1.0, suite="base-a-test-cpu")


def make_tools_weather():
    return [
        Tool(
            function=Function(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            )
        )
    ]


def test_plain_text_no_tool_call():
    detector = GigaChat3Detector()
    text = "Just a normal assistant reply with no tool call."
    tools = []
    res = detector.detect_and_parse(text, tools)
    assert res.normal_text == text
    assert res.calls == []


def test_valid_tool_call():
    detector = GigaChat3Detector()
    tools = make_tools_weather()
    text = 'function call<|role_sep|>\n{"name": "get_weather", "arguments": {"city": "Moscow"}}'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"


def test_malformed_json():
    detector = GigaChat3Detector()
    tools = make_tools_weather()
    text = 'function call<|role_sep|>\n{"name": "get_weather", "arguments": {"city": "Moscow"'
    res = detector.detect_and_parse(text, tools)
    assert res.calls == []
    assert res.normal_text == text


def test_missing_arguments_field():
    detector = GigaChat3Detector()
    tools = make_tools_weather()
    text = 'function call<|role_sep|>\n{"name": "get_weather"}'
    res = detector.detect_and_parse(text, tools)
    assert res.calls == []
    assert res.normal_text == text


def test_unknown_tool_name():
    detector = GigaChat3Detector()
    tools = make_tools_weather()
    text = 'function call<|role_sep|>\n{"name": "delete_everything", "arguments": {"confirm": true}}'
    res = detector.detect_and_parse(text, tools)
    assert res.calls == []


def test_eos_token_stripped():
    detector = GigaChat3Detector()
    tools = make_tools_weather()
    text = 'function call<|role_sep|>\n{"name": "get_weather", "arguments": {"city": "Moscow"}}</s>'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"


def test_content_before_marker_preserved():
    detector = GigaChat3Detector()
    tools = make_tools_weather()
    text = 'Sure, let me check.<|message_sep|>function call<|role_sep|>\n{"name": "get_weather", "arguments": {"city": "Moscow"}}'
    res = detector.detect_and_parse(text, tools)
    assert res.normal_text == "Sure, let me check."
    assert len(res.calls) == 1


def test_arguments_not_a_dict():
    detector = GigaChat3Detector()
    tools = make_tools_weather()
    text = 'function call<|role_sep|>\n{"name": "get_weather", "arguments": "none", "city": "Moscow"}'
    res = detector.detect_and_parse(text, tools)
    assert res.calls == []


def test_has_tool_call_detection():
    detector = GigaChat3Detector()
    assert detector.has_tool_call("function call<|role_sep|>\n") is True
    assert detector.has_tool_call("just a normal reply with no tool call") is False


def test_supports_structural_tag_is_false():
    detector = GigaChat3Detector()
    assert detector.supports_structural_tag() is False


def test_structure_info_not_implemented():
    detector = GigaChat3Detector()
    with pytest.raises(NotImplementedError):
        detector.structure_info()


def test_streaming_no_marker():
    detector = GigaChat3Detector()
    tools = make_tools_weather()
    res = detector.parse_streaming_increment("Just chatting", tools)
    assert res.normal_text == "Just chatting"
    assert res.calls == []


def test_streaming_incremental_tool_call():
    detector = GigaChat3Detector()
    tools = make_tools_weather()

    chunk1 = "function call<|role_sep|>\n"
    r1 = detector.parse_streaming_increment(chunk1, tools)
    assert r1.calls == []

    chunk2 = '{"name": "get_weather", "arguments": {"city": "Perm"}}'
    r2 = detector.parse_streaming_increment(chunk2, tools)
    assert len(r2.calls) == 1
    assert r2.calls[0].name == "get_weather"


def test_streaming_unknown_tool_name():
    detector = GigaChat3Detector()
    tools = make_tools_weather()

    chunk1 = "function call<|role_sep|>\n"
    detector.parse_streaming_increment(chunk1, tools)

    chunk2 = '{"name": "delete_everything", "arguments": {"confirm": true}}'
    r2 = detector.parse_streaming_increment(chunk2, tools)
    assert len(r2.calls) == 1
    assert r2.calls[0].name == "delete_everything"


def test_streaming_eos_stripped_from_arguments():
    detector = GigaChat3Detector()
    tools = make_tools_weather()

    detector.parse_streaming_increment("function call<|role_sep|>\n", tools)
    detector.parse_streaming_increment('{"name": "get_weather", ', tools)
    r = detector.parse_streaming_increment('"arguments": {"city": "Kirov"}}</s>', tools)

    assert "</s>" not in r.calls[0].parameters


def test_unicode_arguments_round_trip():
    detector = GigaChat3Detector()
    tools = make_tools_weather()
    text = 'function call<|role_sep|>\n{"name": "get_weather", "arguments": {"city": "Москва"}}'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["city"] == "Москва"
