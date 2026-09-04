import json
import sys

import pytest
from openai.types.responses.response_output_text import Logprob

from sglang.srt.entrypoints.openai.protocol import Function, ResponsesRequest, Tool
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
    strip_response_wrappers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
    assert len(calls) == 1
    assert calls[0].name == "python"
    assert json.loads(calls[0].parameters) == {"code": "print(2)"}


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
    assert [call.tool_index for call in calls] == [0, 1]
    assert [json.loads(call.parameters) for call in calls] == [
        {"code": "a"},
        {"code": "b"},
    ]


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
    """A tools section cut off before its closing tag used to vanish at
    end-of-stream: no call, no text, no log. It must at least be reported."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    truncated = TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>'
    text, calls = _stream(detector, _chunks(truncated, 7), tools)
    assert calls == []
    with caplog.at_level("WARNING", logger="sglang.srt.function_call.kimik3_detector"):
        result = detector.finish(tools)
    assert result.calls == []
    assert TOOLS_OPEN not in (result.normal_text or "")
    assert "no complete tool call" in caplog.text


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


def test_detector_capabilities_and_registration() -> None:
    detector = KimiK3Detector()
    assert detector.supports_structural_tag()
    assert not detector.parses_required_natively()
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    assert isinstance(parser.detector, KimiK3Detector)
    assert parser.get_structure_constraint("required") is not None


LEAKED_TOOLS_SECTION = (
    f'{TOOLS_OPEN}<|open|>call tool="read_agent" index="1"<|sep|>'
    f'<|open|>argument key="id" type="string"<|sep|>worker_1'
    f"<|close|>argument<|sep|><|close|>call<|sep|>{TOOLS_CLOSE}{MESSAGE_CLOSE}"
)


def _strip(text: str, reasoning_separated: bool = True) -> str:
    return KimiK3Detector().strip_template_artifacts(text, reasoning_separated)


def test_strip_template_artifacts_whitespace_behavior() -> None:
    detector = KimiK3Detector()

    assert detector.strip_template_artifacts("   ") == "   "

    marker_only = f"{RESPONSE_OPEN}   {RESPONSE_CLOSE}"
    assert detector.strip_template_artifacts(marker_only) == ""


def test_strip_response_wrappers_preserves_whitespace() -> None:
    text = " \t\n "

    assert strip_response_wrappers(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "<|open|>",
        "<|close|>",
        "<|",
        "<|open|>tools",
        "<|close|>response",
        f"{RESPONSE_OPEN}<|close|",
    ],
)
def test_truncated_markers_leave_no_text(text: str) -> None:
    assert _strip(text) == ""


def test_unparsed_tools_section_dropped_with_reply_kept() -> None:
    assert _strip(LEAKED_TOOLS_SECTION) == ""
    assert (
        _strip(f"{RESPONSE_OPEN}on it{RESPONSE_CLOSE}{LEAKED_TOOLS_SECTION}") == "on it"
    )


def test_narration_before_truncated_marker_survives() -> None:
    assert (
        _strip("I am delegating the docs subagent.<|close|>response")
        == "I am delegating the docs subagent."
    )


def test_think_channel_content_is_dropped_not_unwrapped() -> None:
    assert _strip(f"{THINK_OPEN}secret{THINK_CLOSE}visible") == "visible"
    assert _strip(f"{THINK_OPEN}cut off mid-thought") == ""


def test_think_content_kept_when_reasoning_is_not_separated() -> None:
    assert (
        _strip(f"weighing{THINK_CLOSE}visible", reasoning_separated=False)
        == "weighingvisible"
    )
    assert (
        _strip(f"{THINK_OPEN}weighing{THINK_CLOSE}visible", reasoning_separated=False)
        == "weighingvisible"
    )


def test_response_channel_kept_inline_when_reasoning_is_not_separated() -> None:
    assert (
        _strip(
            f"trace{THINK_CLOSE}{RESPONSE_OPEN}answer{RESPONSE_CLOSE}",
            reasoning_separated=False,
        )
        == "traceanswer"
    )
    assert (
        _strip(f"trace{THINK_CLOSE}{RESPONSE_OPEN}answer{RESPONSE_CLOSE}") == "answer"
    )


def test_think_close_without_open_drops_reasoning() -> None:
    assert _strip(f"deliberating{THINK_CLOSE}visible") == "visible"
    assert _strip(f"deliberating{THINK_CLOSE}{LEAKED_TOOLS_SECTION}") == ""


def test_quoted_call_token_without_attributes_survives() -> None:
    assert (
        _strip("calls open with <|open|>call, then attributes")
        == "calls open with , then attributes"
    )


def test_markers_only_strip_keeps_reasoning_text() -> None:
    detector = KimiK3Detector()
    assert (
        detector.strip_template_markers(f"weighing options{THINK_CLOSE}")
        == "weighing options"
    )
    assert (
        detector.strip_template_markers("weighing options<|open|>argu")
        == "weighing options"
    )
    assert detector.strip_template_markers('weighing <|open|>call tool="rea') == (
        "weighing "
    )


@pytest.mark.parametrize(
    "text", ["", "hello world", "a < b and c |> d", "is 3 < 4? yes <"]
)
def test_plain_text_is_untouched(text: str) -> None:
    assert _strip(text) == text


def test_parser_strips_when_detection_misses() -> None:
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    assert parser.parse_non_stream("done<|open|>tool") == ("done", [])


def test_parser_strips_when_no_tools_declared() -> None:
    parser = FunctionCallParser([], "kimi_k3")
    assert parser.parse_non_stream(LEAKED_TOOLS_SECTION) == ("", [])


def _logprobs(*tokens: str) -> list[Logprob]:
    return [
        Logprob(
            token=token,
            logprob=-0.5,
            bytes=list(token.encode("utf-8")),
            top_logprobs=[],
        )
        for token in tokens
    ]


def _output_items(text: str, output_logprobs: list[Logprob] | None = None):
    server = object.__new__(OpenAIServingResponses)
    server.reasoning_parser = None
    server.tool_call_parser = None
    server._artifact_detector = KimiK3Detector()
    return server._make_response_output_items(
        ResponsesRequest(input="hi"),
        text,
        tokenizer=None,
        output_logprobs=output_logprobs,
        require_reasoning=False,
    )


def test_syntax_only_text_still_yields_a_message() -> None:
    items = _output_items(f"{RESPONSE_OPEN}<|close|")
    assert len(items) == 1
    assert items[0].content[0].text == ""


def test_logprobs_dropped_when_cleanup_changes_text() -> None:
    items = _output_items(
        f"trace{THINK_CLOSE}answer",
        output_logprobs=_logprobs("trace", THINK_CLOSE, "answer"),
    )
    assert items[0].content[0].text == "traceanswer"
    assert items[0].content[0].logprobs is None


def test_logprobs_kept_verbatim_when_cleanup_changes_nothing() -> None:
    items = _output_items("answer", output_logprobs=_logprobs("something", "else"))
    assert [entry.token for entry in items[0].content[0].logprobs] == [
        "something",
        "else",
    ]


def test_logprobs_dropped_when_they_cannot_be_reconciled() -> None:
    items = _output_items(
        f"trace{THINK_CLOSE}answer", output_logprobs=_logprobs("something", "else")
    )
    assert items[0].content[0].logprobs is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
