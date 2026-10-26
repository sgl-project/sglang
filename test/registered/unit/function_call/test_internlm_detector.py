import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name=name,
                parameters={"type": "object"},
            ),
        )
        for name in ("get_weather", "get_time")
    ]


def test_streaming_multiple_calls_with_split_start_marker():
    chunks = [
        "Checking.",
        "<|action_start|>",
        " ",
        "<|plugin|>",
        "\n",
        '{"name":"get_weather","parameters":{"city":"Tokyo"}}',
        "<|action_end|>",
        "\n",
        "<|action_start|>",
        " ",
        "<|plugin|>",
        "\n",
        '{"name":"get_time","parameters":{"city":"Paris"}}',
        "<|action_end|>",
    ]

    detector = InternlmDetector()
    calls = []
    for chunk in chunks:
        calls.extend(detector.parse_streaming_increment(chunk, _make_tools()).calls)

    assert [(call.tool_index, call.name) for call in calls] == [
        (0, "get_weather"),
        (1, "get_time"),
    ]
    assert json.loads(calls[0].parameters) == {"city": "Tokyo"}
    assert json.loads(calls[1].parameters) == {"city": "Paris"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
