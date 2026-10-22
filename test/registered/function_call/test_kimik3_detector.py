import json
import random
import sys

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _make_tool(name: str) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
            },
        ),
    )


def _call_block(tool: str, index: int, args: dict[str, tuple[str, str]]) -> str:
    parts = [f'<|open|>call tool="{tool}" index="{index}"<|sep|>']
    for key, (arg_type, value) in args.items():
        parts.append(
            f'<|open|>argument key="{key}" type="{arg_type}"<|sep|>'
            f"{value}<|close|>argument<|sep|>"
        )
    parts.append("<|close|>call<|sep|>")
    return "".join(parts)


def _chunks(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def _stream(
    detector: KimiK3Detector, chunks: list[str], tools: list[Tool]
) -> tuple[str, list[ToolCallItem]]:
    text = ""
    calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        text += result.normal_text
        calls.extend(result.calls)
    return text, calls


def _collect(calls: list[ToolCallItem]) -> list[tuple[str, str]]:
    """Fold streamed items into (name, arguments) per tool index."""
    merged: dict[int, list[str]] = {}
    for call in calls:
        if call.name is not None:
            assert call.tool_index not in merged, "name sent twice"
            merged[call.tool_index] = [call.name, ""]
        merged[call.tool_index][1] += call.parameters
    assert list(merged) == list(range(len(merged)))
    return [tuple(merged[i]) for i in range(len(merged))]


def test_detect_and_parse_single_call() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        f"{RESPONSE_OPEN}Let me run it.{RESPONSE_CLOSE}{TOOLS_OPEN}"
        + _call_block(
            "python",
            1,
            {"code": ("string", "print(1)"), "opts": ("object", '{"a": 1}')},
        )
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, tools)
    assert result.normal_text == "Let me run it."
    assert len(result.calls) == 1
    assert result.calls[0].name == "python"
    assert json.loads(result.calls[0].parameters) == {
        "code": "print(1)",
        "opts": {"a": 1},
    }


def test_detect_and_parse_no_tools_channel() -> None:
    detector = KimiK3Detector()
    result = detector.detect_and_parse(
        f"{RESPONSE_OPEN}hi there{RESPONSE_CLOSE}{MESSAGE_CLOSE}",
        [_make_tool("python")],
    )
    assert result.normal_text == "hi there"
    assert result.calls == []


def test_detect_and_parse_multiple_calls() -> None:
    detector = KimiK3Detector()
    text = (
        TOOLS_OPEN
        + _call_block("python", 1, {"code": ("string", "a")})
        + _call_block("python", 2, {"code": ("string", "b")})
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert [call.tool_index for call in result.calls] == [0, 1]
    assert json.loads(result.calls[1].parameters) == {"code": "b"}


def test_detect_and_parse_unclosed_tools_section() -> None:
    detector = KimiK3Detector()
    text = TOOLS_OPEN + _call_block("python", 1, {"code": ("string", "x")})
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == {"code": "x"}


def test_attr_unescaping_and_raw_string_args() -> None:
    detector = KimiK3Detector()
    text = (
        f"{TOOLS_OPEN}"
        '<|open|>call tool="a&amp;b" index="1"<|sep|>'
        '<|open|>argument key="q" type="string"<|sep|>'
        "say &quot;hi&quot;<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        f"{TOOLS_CLOSE}"
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert result.calls[0].name == "a&b"
    assert json.loads(result.calls[0].parameters) == {"q": "say &quot;hi&quot;"}


def test_non_string_arg_json_decoding() -> None:
    detector = KimiK3Detector()
    text = (
        TOOLS_OPEN
        + _call_block(
            "python",
            1,
            {
                "n": ("number", "42"),
                "flag": ("boolean", "true"),
                "bad": ("object", "{not json"),
            },
        )
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert json.loads(result.calls[0].parameters) == {
        "n": 42,
        "flag": True,
        "bad": "{not json",
    }


@pytest.mark.parametrize("chunk_size", [1, 7, 23])
def test_streaming_split_markers(chunk_size: int) -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        f"{RESPONSE_OPEN}Hello!{RESPONSE_CLOSE}{TOOLS_OPEN}"
        + _call_block("python", 1, {"code": ("string", "print(2)")})
        + TOOLS_CLOSE
    )
    normal_text, calls = _stream(detector, _chunks(text, chunk_size), tools)
    assert normal_text == "Hello!"
    assert _collect(calls) == [("python", '{"code": "print(2)"}')]


def test_streaming_two_calls() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        TOOLS_OPEN
        + _call_block("python", 1, {"code": ("string", "a")})
        + _call_block("python", 2, {"code": ("string", "b")})
        + TOOLS_CLOSE
    )
    _, calls = _stream(detector, _chunks(text, 7), tools)
    assert _collect(calls) == [("python", '{"code": "a"}'), ("python", '{"code": "b"}')]


def test_streaming_plain_text_only() -> None:
    detector = KimiK3Detector()
    text, calls = _stream(
        detector, ["just a ", "plain ", "reply"], [_make_tool("python")]
    )
    assert text == "just a plain reply"
    assert calls == []


def test_streaming_bookkeeping_for_serving_layer() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        TOOLS_OPEN + _call_block("python", 1, {"code": ("string", "a")}) + TOOLS_CLOSE
    )
    _stream(detector, _chunks(text, 9), tools)
    assert detector.current_tool_id == 0
    assert detector.prev_tool_call_arr[0] == {
        "name": "python",
        "arguments": {"code": "a"},
    }
    assert json.loads(detector.streamed_args_for_tool[0]) == {"code": "a"}


def test_stream_end_reports_truncated_tools_section(caplog) -> None:
    """A tools section cut off with no call must be reported, and never leak
    XTML markup as text."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    truncated = TOOLS_OPEN + "<|open|>call"
    text, calls = _stream(detector, _chunks(truncated, 7), tools)
    assert calls == [] and text == ""
    with caplog.at_level("WARNING", logger="sglang.srt.function_call.kimik3_detector"):
        result = detector.finish(tools)
    assert result.calls == []
    assert TOOLS_OPEN not in (result.normal_text or "")
    assert "no complete tool call" in caplog.text


def test_stream_end_closes_truncated_call(caplog) -> None:
    """A call cut off mid-arguments has already been announced to the client, so
    its JSON is closed with what arrived instead of being left as a prefix."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    truncated = (
        TOOLS_OPEN
        + '<|open|>call tool="python" index="1"<|sep|>'
        + '<|open|>argument key="code" type="string"<|sep|>print('
    )
    text, calls = _stream(detector, _chunks(truncated, 7), tools)
    assert text == ""
    assert _collect(calls) == [("python", '{"code": "print(')]
    with caplog.at_level("WARNING", logger="sglang.srt.function_call.kimik3_detector"):
        result = detector.finish(tools)
    assert not result.normal_text
    assert _collect(calls + result.calls) == [("python", '{"code": "print("}')]
    assert "inside tool call 0" in caplog.text
    assert detector.prev_tool_call_arr == [
        {"name": "python", "arguments": {"code": "print("}}
    ]
    assert detector.streamed_args_for_tool == ['{"code": "print("}']


def test_stream_end_releases_held_back_text() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text, _ = _stream(detector, ["all done", "<"], tools)
    assert text == "all done"
    result = detector.finish(tools)
    assert text + (result.normal_text or "") == "all done<"


def test_stream_end_drops_truncated_marker() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text, _ = _stream(detector, ["all done", "<|open|>"], tools)
    result = detector.finish(tools)
    assert text + (result.normal_text or "") == "all done"


_TRICKY_STRINGS = [
    "print(1)",
    "",
    'say "hi" \\ back\\slash',
    "line1\nline2\r\n\ttab",
    "unicode: \u4e2d\u6587 \U0001f600 \u00e9",
    "a < b and c > d, <|not a marker, <|close|> alone",
    "ends with <",
    "ends with <|close|>argument",
    "control \x00\x1f chars",
]


def _random_call(rng: random.Random, index: int) -> tuple[str, str, dict]:
    name = rng.choice(["write_file", "python", "a&amp;b"])
    args: dict[str, tuple[str, str]] = {}
    for k in range(rng.randint(0, 3)):
        kind = rng.choice(["string", "string", "number", "object", "boolean", "bad"])
        if kind == "string":
            args[f"k{k}"] = ("string", rng.choice(_TRICKY_STRINGS))
        elif kind == "number":
            args[f"k{k}"] = ("number", str(rng.randint(-5, 5000)))
        elif kind == "object":
            args[f"k{k}"] = ("object", '{"a": [1, "\u00e9"], "b": null}')
        elif kind == "boolean":
            args[f"k{k}"] = ("boolean", "true")
        else:
            args[f"k{k}"] = ("object", "{not json")
    return name, _call_block(name, index, args), args


def _random_output(rng: random.Random) -> str:
    text = rng.choice(["", f"{RESPONSE_OPEN}Let me <b>do</b> it.{RESPONSE_CLOSE}"])
    text += TOOLS_OPEN
    for index in range(1, rng.randint(1, 3) + 1):
        text += rng.choice(["", "\n", " "]) + _random_call(rng, index)[1]
    return text + rng.choice(["", TOOLS_CLOSE, TOOLS_CLOSE + MESSAGE_CLOSE])


def _random_chunks(rng: random.Random, text: str) -> list[str]:
    chunks, pos = [], 0
    while pos < len(text):
        size = rng.choice([1, 1, 2, 3, 5, 8, 13, 40])
        chunks.append(text[pos : pos + size])
        pos += size
    return chunks


@pytest.mark.parametrize("seed", range(300))
def test_streaming_matches_detect_and_parse(seed: int) -> None:
    """Streamed name/argument fragments concatenate byte-for-byte to the
    non-streaming result, for any chunking."""
    rng = random.Random(seed)
    tools = [_make_tool("python")]
    text = _random_output(rng)
    expected = KimiK3Detector().detect_and_parse(text, tools)

    detector = KimiK3Detector()
    normal_text, calls = _stream(detector, _random_chunks(rng, text), tools)
    end = detector.finish(tools)
    normal_text += end.normal_text
    calls += end.calls

    assert normal_text == expected.normal_text
    assert _collect(calls) == [(c.name, c.parameters) for c in expected.calls]
    for i, call in enumerate(expected.calls):
        assert detector.streamed_args_for_tool[i] == call.parameters
        assert detector.prev_tool_call_arr[i] == {
            "name": call.name,
            "arguments": json.loads(call.parameters),
        }


def test_streaming_emits_name_and_arguments_before_call_closes() -> None:
    """The client must receive data while a long argument is generated, not only
    when the whole call has closed."""
    detector = KimiK3Detector()
    tools = [_make_tool("write_file")]
    head = (
        TOOLS_OPEN
        + '<|open|>call tool="write_file" index="1"<|sep|>'
        + '<|open|>argument key="path" type="string"<|sep|>a.py<|close|>argument<|sep|>'
        + '<|open|>argument key="content" type="string"<|sep|>'
    )
    result = detector.parse_streaming_increment(head, tools)
    assert result.calls[0].name == "write_file"
    assert _collect(result.calls) == [("write_file", '{"path": "a.py", "content": "')]

    for line in ["import os\n", 'print("x")\n']:
        result = detector.parse_streaming_increment(line, tools)
        assert [c.name for c in result.calls] == [None]
        assert result.calls[0].parameters == json.dumps(line)[1:-1]

    result = detector.parse_streaming_increment(
        "<|close|>argument<|sep|><|close|>call<|sep|>" + TOOLS_CLOSE, tools
    )
    assert result.calls[0].parameters == '"}'
    assert json.loads(detector.streamed_args_for_tool[0]) == {
        "path": "a.py",
        "content": 'import os\nprint("x")\n',
    }


def test_streaming_holds_back_partial_close_marker() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    detector.parse_streaming_increment(
        TOOLS_OPEN
        + '<|open|>call tool="python" index="1"<|sep|>'
        + '<|open|>argument key="code" type="string"<|sep|>',
        tools,
    )
    result = detector.parse_streaming_increment("x = 1<|close|>argu", tools)
    assert result.calls[0].parameters == "x = 1"
    result = detector.parse_streaming_increment("ment<|sep|>", tools)
    assert result.calls[0].parameters == '"'


def test_streaming_skips_nameless_call() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        TOOLS_OPEN
        + '<|open|>call index="1"<|sep|><|close|>call<|sep|>'
        + _call_block("python", 2, {"code": ("string", "b")})
        + TOOLS_CLOSE
    )
    _, calls = _stream(detector, _chunks(text, 5), tools)
    assert _collect(calls) == [("python", '{"code": "b"}')]


def test_detector_capabilities_and_registration() -> None:
    detector = KimiK3Detector()
    assert detector.supports_structural_tag()
    assert not detector.parses_required_natively()
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    assert isinstance(parser.detector, KimiK3Detector)
    assert parser.get_structure_constraint("required") is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
