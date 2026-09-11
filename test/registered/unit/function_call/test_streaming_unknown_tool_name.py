"""Streaming counterpart to test_unknown_tool_name.py.

``parse_base_json`` drops only the call that names an unknown tool and keeps
going (``continue``). The streaming state machine used to clear the whole
buffer instead, which also discarded every call batched behind the bad one --
and when the bad call came first, the client received no tool calls at all.
"""

import logging

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")

TOOLS = [
    Tool(function=Function(name="get_weather", parameters={})),
    Tool(function=Function(name="get_time", parameters={})),
]

WEATHER = '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
TIME = '{"name": "get_time", "arguments": {"tz": "JST"}}'
UNKNOWN = '{"name": "rm_rf", "arguments": {"path": "/"}}'


def _stream(detector, text, chunk_size=1):
    """Feed ``text`` through the detector and rebuild what a client would see."""
    calls = {}
    order = []
    for i in range(0, len(text), chunk_size):
        result = detector.parse_streaming_increment(text[i : i + chunk_size], TOOLS)
        for call in result.calls or []:
            if call.tool_index not in calls:
                calls[call.tool_index] = {"name": None, "arguments": ""}
                order.append(call.tool_index)
            if call.name:
                calls[call.tool_index]["name"] = call.name
            if call.parameters:
                calls[call.tool_index]["arguments"] += call.parameters
    return [(idx, calls[idx]["name"], calls[idx]["arguments"]) for idx in order]


def _json_array(*calls):
    return "[" + ",".join(c.replace('"arguments"', '"parameters"') for c in calls) + "]"


def _tag_wrapped(*calls):
    return "".join(f"<tool_call>\n{c}\n</tool_call>\n" for c in calls)


@pytest.mark.parametrize("chunk_size", [1, 3, 17])
def test_unknown_tool_first_does_not_drop_the_rest(chunk_size):
    """The batch opens with an unknown tool. Every valid call behind it used to
    disappear, leaving tool_choice="required" with nothing to return."""
    streamed = _stream(
        JsonArrayParser(), _json_array(UNKNOWN, WEATHER, TIME), chunk_size
    )

    assert [name for _, name, _ in streamed] == ["get_weather", "get_time"]
    assert [idx for idx, _, _ in streamed] == [0, 1]
    assert streamed[0][2] == '{"city": "Tokyo"}'
    assert streamed[1][2] == '{"tz": "JST"}'


@pytest.mark.parametrize("chunk_size", [1, 3, 17])
def test_unknown_tool_in_the_middle_does_not_drop_the_rest(chunk_size):
    streamed = _stream(
        JsonArrayParser(), _json_array(WEATHER, UNKNOWN, TIME), chunk_size
    )

    assert [name for _, name, _ in streamed] == ["get_weather", "get_time"]
    assert [idx for idx, _, _ in streamed] == [0, 1]


@pytest.mark.parametrize("detector_cls", [Qwen25Detector, HermesDetector])
def test_tag_wrapped_format_does_not_merge_calls(detector_cls):
    """Formats that wrap each call in its own tag resynchronise on the next
    tag, so instead of losing the trailing call they replayed it at index 0:
    the reset sent current_tool_id back to -1 and popped the previous tool's
    entry. The client saw a single call whose name had been overwritten and
    whose arguments were two JSON objects concatenated.
    """
    text = _tag_wrapped(WEATHER, UNKNOWN, TIME)
    streamed = _stream(detector_cls(), text)

    assert [(idx, name) for idx, name, _ in streamed] == [
        (0, "get_weather"),
        (1, "get_time"),
    ]
    assert [args for _, _, args in streamed] == [
        '{"city": "Tokyo"}',
        '{"tz": "JST"}',
    ]


def test_dropped_call_does_not_reuse_a_delivered_index():
    """Whatever the format, an index handed to the client must never be
    reused by a later call."""
    for text, cls in (
        (_json_array(WEATHER, UNKNOWN, TIME), JsonArrayParser),
        (_tag_wrapped(WEATHER, UNKNOWN, TIME), Qwen25Detector),
    ):
        streamed = _stream(cls(), text)
        indices = [idx for idx, _, _ in streamed]
        assert len(indices) == len(set(indices)), f"index collision: {streamed}"


def test_previous_tool_bookkeeping_survives_a_dropped_call():
    """The reset popped streamed_args_for_tool, deleting the record of the
    previous, already-streamed tool. serving_chat._check_for_unstreamed_tool_args
    bails out when that list is empty, so the end-of-stream flush went missing."""
    detector = JsonArrayParser()
    _stream(detector, _json_array(WEATHER, UNKNOWN))

    assert len(detector.prev_tool_call_arr) == len(detector.streamed_args_for_tool)
    assert detector.streamed_args_for_tool == ['{"city": "Tokyo"}']


def test_unknown_tool_is_still_dropped():
    """The unknown call itself must not reach the client."""
    streamed = _stream(JsonArrayParser(), _json_array(WEATHER, UNKNOWN, TIME))

    assert "rm_rf" not in [name for _, name, _ in streamed]


def test_unknown_tool_is_logged_once(caplog):
    """Non-streaming warns via parse_base_json; streaming was silent, so an
    operator had no signal that calls were being discarded."""
    with caplog.at_level(
        logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
    ):
        _stream(JsonArrayParser(), _json_array(WEATHER, UNKNOWN, TIME))

    matching = [
        m
        for m in caplog.messages
        if "Model attempted to call undefined function: rm_rf" in m
    ]
    assert len(matching) == 1, f"expected exactly one warning, got {matching}"


def test_llama32_separator_format_recovers():
    """Llama 3.2 separates calls with ';' rather than wrapping each one, so it
    hit the same buffer-clearing loss as the JSON array format."""
    text = f"<|python_tag|>{UNKNOWN};{WEATHER}"
    streamed = _stream(Llama32Detector(), text)

    assert [name for _, name, _ in streamed] == ["get_weather"]


def test_all_valid_calls_are_unaffected():
    """Baseline: nothing about the ordinary parallel path changes."""
    streamed = _stream(JsonArrayParser(), _json_array(WEATHER, TIME))

    assert [(idx, name) for idx, name, _ in streamed] == [
        (0, "get_weather"),
        (1, "get_time"),
    ]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
