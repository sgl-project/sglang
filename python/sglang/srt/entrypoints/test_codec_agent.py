"""
Unit tests for codec_agent (server-side ToolWatcher + parse_tool_call).

Mirrors the test cases from the libcodec / @codecai/web / codecai /
Codec.Net suites — same state machine, same edge cases, same expected
outputs. If any of these regress, server-side detection won't agree
with what the client-side watchers produce.

Run:
    pytest -xvs python/sglang/srt/entrypoints/test_codec_agent.py
"""

from __future__ import annotations

from sglang.srt.entrypoints.codec_agent import (
    ToolWatcher,
    make_call_id,
    parse_tool_call,
)

# Synthetic markers — small ints chosen for readable assertions.
START = 90
END = 91


def test_passthrough_then_region_then_passthrough():
    w = ToolWatcher(start_id=START, end_id=END)
    # "hello world <tool_call> foo bar </tool_call> hello !"
    pt, regs = w.feed([0, 1, START, 3, 4, END, 0, 2])
    assert pt == [0, 1, 0, 2]  # markers consumed, body withheld
    assert regs == [[3, 4]]
    assert not w.inside


def test_region_split_across_feeds():
    w = ToolWatcher(START, END)
    # Feed 1: opens region, no close.
    pt, regs = w.feed([0, START, 3])
    assert pt == [0]
    assert regs == []
    assert w.inside

    # Feed 2: closes region, then more passthrough.
    pt, regs = w.feed([4, END, 1])
    assert pt == [1]
    assert regs == [[3, 4]]
    assert not w.inside


def test_multiple_regions_in_one_feed():
    w = ToolWatcher(START, END)
    pt, regs = w.feed([0, START, 3, END, 1, START, 4, END, 2])
    assert pt == [0, 1, 2]
    assert regs == [[3], [4]]


def test_stray_end_passes_through():
    w = ToolWatcher(START, END)
    pt, regs = w.feed([0, END, 1])
    # End with no preceding start → ordinary token.
    assert pt == [0, END, 1]
    assert regs == []


def test_nested_start_ignored():
    w = ToolWatcher(START, END)
    # <tool_call> body1 <tool_call> body2 </tool_call> trailing
    # Per all four reference implementations (libcodec, @codecai/web,
    # codecai, Codec.Net): the nested start is fully ignored — neither
    # forwarded nor buffered into the body. The first </tool_call>
    # closes the outer region with body [3, 4].
    pt, regs = w.feed([START, 3, START, 4, END, 99])
    assert pt == [99]
    assert regs == [[3, 4]]


def test_reset_drops_in_flight_region():
    w = ToolWatcher(START, END)
    w.feed([START, 3, 4])
    assert w.inside
    w.reset()
    assert not w.inside
    pt, regs = w.feed([END, 1])
    assert pt == [END, 1]
    assert regs == []


def test_no_decode_path():
    """The watcher operates on uint32 IDs only — never reads any vocab,
    never invokes the tokenizer. Feed IDs that have no plausible vocab
    entry; the watcher must still emit them verbatim. This mirrors the
    test_watcher_does_not_decode_tokens case in libcodec."""
    w = ToolWatcher(START, END)
    BIG_A = 0xFFFFFF00
    BIG_B = 0xDEADBEEF
    pt, regs = w.feed([12345, BIG_A, START, BIG_B, END, 99999])
    assert pt == [12345, BIG_A, 99999]
    assert regs == [[BIG_B]]


# ── parse_tool_call ─────────────────────────────────────────────────────────


def test_parse_tool_call_well_formed():
    body = '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
    ev = parse_tool_call(body)
    assert ev.name == "get_weather"
    assert ev.arguments_json == body
    assert ev.id is None


def test_parse_tool_call_with_id():
    ev = parse_tool_call('{"name": "search"}', call_id="tc_00000001")
    assert ev.name == "search"
    assert ev.id == "tc_00000001"


def test_parse_tool_call_malformed_json():
    """Malformed JSON: keep raw body so the caller can return an
    'invalid_arguments' error to the model."""
    body = '{"name": "search"'  # unterminated
    ev = parse_tool_call(body)
    assert ev.name is None
    assert ev.arguments_json == body  # raw body preserved


def test_parse_tool_call_empty():
    ev = parse_tool_call("   ")
    assert ev.name is None
    assert ev.arguments_json == ""


def test_parse_tool_call_compact():
    body = '{"name":"f","arguments":{"a":1}}'
    ev = parse_tool_call(body)
    assert ev.name == "f"


def test_parse_tool_call_no_name_key():
    """Body is JSON but doesn't follow the standard tool-call shape.
    We still return arguments_json; name=None. Caller decides what
    to do with it."""
    body = '{"foo": "bar"}'
    ev = parse_tool_call(body)
    assert ev.name is None
    assert ev.arguments_json == body


# ── call id ─────────────────────────────────────────────────────────────────


def test_make_call_id_format():
    assert make_call_id(1) == "tc_00000001"
    assert make_call_id(0xDEADBEEF) == "tc_deadbeef"
