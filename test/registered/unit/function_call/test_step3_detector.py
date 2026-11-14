"""Unit tests for Step3Detector argument typing - no server, no model loading."""

import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.step3_detector import Step3Detector, get_argument_type
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")

PROPERTIES = {
    "city": {"type": "string"},
    "days": {"type": "integer"},
    "zip_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "area_code": {"type": ["string", "null"]},
    "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    "metric": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
}


def make_tools():
    return [
        Tool(
            function=Function(
                name="lookup",
                parameters={"type": "object", "properties": PROPERTIES},
            )
        )
    ]


def make_call(detector, params):
    body = "".join(
        f'<steptml:parameter name="{k}">{v}</steptml:parameter>'
        for k, v in params.items()
    )
    return (
        f"{detector.bot_token}{detector.tool_call_begin}function{detector.tool_sep}"
        f'<steptml:invoke name="lookup">{body}</steptml:invoke>'
        f"{detector.tool_call_end}{detector.eot_token}"
    )


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("city", "string"),
        ("days", "integer"),
        ("zip_code", "string"),
        ("area_code", "string"),
        ("limit", "integer"),
        ("metric", "boolean"),
    ],
)
def test_get_argument_type_resolves_nullable_schemas(arg, expected):
    assert get_argument_type("lookup", arg, make_tools()) == expected


def test_detect_and_parse_coerces_by_schema_type():
    detector = Step3Detector()
    text = make_call(
        detector,
        {
            "city": "Paris",
            "days": "3",
            "zip_code": "10001",
            "area_code": "212",
            "limit": "5",
            "metric": "true",
        },
    )
    result = detector.detect_and_parse(text, make_tools())
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == {
        "city": "Paris",
        "days": 3,
        "zip_code": "10001",
        "area_code": "212",
        "limit": 5,
        "metric": True,
    }


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
