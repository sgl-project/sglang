import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


def make_tool(tags_schema: dict) -> list:
    return [
        Tool(
            function=Function(
                name="tag_items",
                parameters={
                    "type": "object",
                    "properties": {"tags": tags_schema},
                    "required": ["tags"],
                },
            )
        )
    ]


def make_call_text(func_name: str, key: str, value: str) -> str:
    return (
        "<｜tool_calls_begin｜>"
        "<｜tool_call_begin｜>function<｜tool_sep｜>"
        f'<steptml:invoke name="{func_name}">'
        f'<steptml:parameter name="{key}">{value}</steptml:parameter>'
        "</steptml:invoke>"
        "<｜tool_call_end｜>"
        "<｜tool_calls_end｜>"
    )


def parse_single_argument(tags_schema: dict, value: str):
    detector = Step3Detector()
    text = make_call_text("tag_items", "tags", value)
    res = detector.detect_and_parse(text, make_tool(tags_schema))
    assert len(res.calls) == 1
    return json.loads(res.calls[0].parameters)["tags"]


def test_implicit_array_schema_parsed_as_list():
    # JSON Schema may declare an array through ``items`` without a top-level type.
    parsed = parse_single_argument({"items": {"type": "string"}}, '["a", "b"]')
    assert parsed == ["a", "b"]
    assert isinstance(parsed, list)


def test_nullable_type_array_schema_parsed_as_string():
    # ``type: ["string", "null"]`` is a string schema, so the literal must not
    # be JSON-decoded into a bool.
    parsed = parse_single_argument({"type": ["string", "null"]}, "true")
    assert parsed == "true"
    assert isinstance(parsed, str)


def test_implicit_object_schema_parsed_as_dict():
    # A schema declaring only ``properties`` is an object schema.
    parsed = parse_single_argument(
        {"properties": {"a": {"type": "string"}}}, '{"a": "b"}'
    )
    assert parsed == {"a": "b"}
    assert isinstance(parsed, dict)


def test_nullable_anyof_schema_stays_string_guard():
    # Guard, not a regression repro: c67efab427 already kept this value a str.
    parsed = parse_single_argument(
        {"anyOf": [{"type": "string"}, {"type": "null"}]}, "true"
    )
    assert parsed == "true"
    assert isinstance(parsed, str)


def test_plain_string_argument_unchanged():
    parsed = parse_single_argument({"type": "string"}, "[1, 2]")
    assert parsed == "[1, 2]"
    assert isinstance(parsed, str)


def test_explicit_array_schema_parsed_as_list():
    parsed = parse_single_argument({"type": "array"}, "[1, 2]")
    assert parsed == [1, 2]
    assert isinstance(parsed, list)


def test_unknown_tool_ignored():
    detector = Step3Detector()
    text = make_call_text("unknown_tool", "tags", "x")
    res = detector.detect_and_parse(text, make_tool({"type": "string"}))
    assert len(res.calls) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
