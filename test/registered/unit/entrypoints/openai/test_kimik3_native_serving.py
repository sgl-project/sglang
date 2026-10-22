"""Exercise native failures through the actual Chat and Responses handlers."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from utils import StreamFixture, engine_chunk, event_payloads, make_serving

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Function,
    ResponsesRequest,
    Tool,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_native import KimiK3ParseError
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

O, C, S = "<|open|>", "<|close|>", "<|sep|>"
CALL = (
    f'{O}call tool="bash" index="1"{S}'
    f'{O}argument key="command" type="string"{S}echo synthetic{C}argument{S}'
    f"{C}call{S}"
)
VALID = f"{O}tools{S}{CALL}{C}tools{S}"
BROKEN = f".{C}think0,60:080\u21923433/te" + VALID
TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="bash",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        ),
    )
]


def request():
    return ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "synthetic"}],
        tools=TOOLS,
        stream=True,
        separate_reasoning=True,
    )


def chat(monkeypatch, chunks, *, reasoning=False):
    from sglang.srt.entrypoints.openai import serving_chat

    monkeypatch.setattr(
        serving_chat,
        "get_serving",
        lambda: SimpleNamespace(
            stream_response_default_include_usage=False,
            incremental_streaming_output=True,
            enable_cache_report=False,
        ),
    )
    obj = OpenAIServingChat.__new__(OpenAIServingChat)
    obj._encoding_adapter = None
    obj.tool_call_parser = "kimi_k3"
    obj.reasoning_parser = "kimi_k3" if reasoning else None
    obj.template_manager = SimpleNamespace(force_reasoning=True)
    obj._get_reasoning_from_request = lambda _: True
    obj._should_return_input_ids = lambda _: False
    obj._should_return_output_ids = lambda _: False
    obj._reported_prompt_tokens = lambda _: 10

    async def generate(*args):
        for text, finish in chunks:
            yield {
                "index": 0,
                "text": text,
                "meta_info": {
                    "id": "synthetic",
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "finish_reason": (
                        finish
                        if isinstance(finish, dict)
                        else {"type": finish, "matched": None}
                    )
                    if finish
                    else None,
                },
            }

    obj.tokenizer_manager = SimpleNamespace(tokenizer=None, generate_request=generate)
    return obj


def wire(obj, req=None):
    async def collect():
        return "".join(
            [
                chunk
                async for chunk in obj._generate_chat_stream(
                    None, req or request(), None
                )
            ]
        )

    return asyncio.run(collect())


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize(
    "chunks",
    [
        [(BROKEN, "stop")],
        [(VALID.replace('tool="bash"', 'tool="undeclared"'), "stop")],
        [(char, None) for char in BROKEN] + [("", "stop")],
        [(f"{O}tools{S}" + CALL, None), (f'{O}call tool="bash" index="2"{S}', "stop")],
        [(f"{O}tools{S}" + CALL, None)],  # EOF must still validate missing closure.
        [(VALID, None)],  # A complete call alone is not a successful stream terminal.
    ],
)
def test_chat_native_errors_never_emit_success_or_done(monkeypatch, reasoning, chunks):
    output = wire(chat(monkeypatch, chunks, reasoning=reasoning))
    assert "KimiK3NativeOutputError" in output
    assert '"code": 502' in output or '"code":502' in output
    assert "[DONE]" not in output
    frames = [
        json.loads(line[6:])
        for line in output.splitlines()
        if line.startswith("data: {")
    ]
    assert not any(
        choice.get("finish_reason")
        for frame in frames
        for choice in frame.get("choices", [])
    )


@pytest.mark.parametrize("reasoning", [False, True])
def test_valid_chat_preserves_calls_and_terminal(monkeypatch, reasoning):
    output = wire(chat(monkeypatch, [(VALID, "stop")], reasoning=reasoning))
    assert '"tool_calls"' in output
    assert "echo synthetic" in output
    assert "[DONE]" in output
    assert "KimiK3NativeOutputError" not in output


def test_chat_with_unseparated_reasoning(monkeypatch):
    req = request()
    req.separate_reasoning = False
    output = wire(
        chat(
            monkeypatch,
            [(f"{O}think{S}thought{C}think{S}" + VALID, "stop")],
            reasoning=True,
        ),
        req,
    )
    assert "echo synthetic" in output
    assert "[DONE]" in output
    assert "KimiK3NativeOutputError" not in output


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("matched", [C + "tools" + S, 123])
def test_chat_restores_proven_stop_before_reasoning_guard(
    monkeypatch, reasoning, incremental, matched
):
    from sglang.srt.entrypoints.openai import serving_chat

    raw = VALID[: -len(C + "tools" + S)] if isinstance(matched, str) else VALID[:-1]
    chunks = [
        (raw[:-4], None),
        (raw[-4:] if incremental else raw, {"type": "stop", "matched": matched}),
    ]
    obj = chat(monkeypatch, chunks, reasoning=reasoning)
    obj.tokenizer_manager.tokenizer = SimpleNamespace(decode=lambda *a, **kw: ">")
    monkeypatch.setattr(
        serving_chat,
        "get_serving",
        lambda: SimpleNamespace(
            stream_response_default_include_usage=False,
            incremental_streaming_output=incremental,
            enable_cache_report=False,
        ),
    )
    output = wire(obj)
    assert "[DONE]" in output and "KimiK3NativeOutputError" not in output
    assert "echo synthetic" in output


@pytest.mark.parametrize("reasoning", [False, True])
def test_responses_restores_proven_tools_close(reasoning):
    serving = make_serving()
    serving.tool_call_parser = "kimi_k3"
    serving.reasoning_parser = "kimi_k3" if reasoning else None
    chunk = engine_chunk(VALID[: -len(C + "tools" + S)], finish=True)
    chunk["meta_info"]["finish_reason"] = {"type": "stop", "matched": C + "tools" + S}
    frames = event_payloads(
        StreamFixture(serving, responses_request(), require_reasoning=reasoning).run(
            [chunk]
        )
    )
    assert frames[-1]["type"] == "response.completed"


@pytest.mark.parametrize("reasoning", [False, True])
def test_length_is_preserved_instead_of_promoted_to_tool_success(
    monkeypatch, reasoning
):
    output = wire(
        chat(monkeypatch, [(f"{O}tools{S}" + CALL, "length")], reasoning=reasoning)
    )
    assert '"finish_reason":"length"' in output or '"finish_reason": "length"' in output
    assert "[DONE]" in output
    assert "KimiK3NativeOutputError" not in output


@pytest.mark.parametrize("complete_first", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_chat_length_at_call_header_never_completes_arguments(
    monkeypatch, complete_first, choice, reasoning, incremental
):
    from sglang.srt.entrypoints.openai import serving_chat

    header = f'{O}call tool="bash" index="{2 if complete_first else 1}"{S}'
    raw = f"{O}tools{S}" + (CALL if complete_first else "") + header
    tool_choice = (
        {"type": "function", "function": {"name": "bash"}}
        if choice == "named"
        else choice
    )
    req = ChatCompletionRequest.model_validate(
        dict(request().model_dump(), tool_choice=tool_choice)
    )
    for split in (0, len(raw) // 2, len(raw)):
        chunks = [
            (raw[:split], None),
            (raw[split:] if incremental else raw, "length"),
        ]
        obj = chat(monkeypatch, chunks, reasoning=reasoning)
        monkeypatch.setattr(
            serving_chat,
            "get_serving",
            lambda: SimpleNamespace(
                stream_response_default_include_usage=False,
                incremental_streaming_output=incremental,
                enable_cache_report=False,
            ),
        )
        output = wire(obj, req)
        arguments = {}
        finishes = []
        for line in output.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            frame = json.loads(line[6:])
            assert "error" not in frame
            for item in frame.get("choices", []):
                if item.get("finish_reason"):
                    finishes.append(item["finish_reason"])
                for delta in item.get("delta", {}).get("tool_calls", []):
                    index = delta["index"]
                    arguments[index] = arguments.get(index, "") + delta.get(
                        "function", {}
                    ).get("arguments", "")
        assert finishes == ["length"]
        assert arguments[int(complete_first)] == "{"
        if complete_first:
            assert json.loads(arguments[0]) == {"command": "echo synthetic"}
        assert "[DONE]" in output


def test_nonstream_does_not_swallow_native_failure(monkeypatch):
    obj = chat(monkeypatch, [])
    with pytest.raises(KimiK3ParseError, match="invalid_header"):
        obj._process_tool_calls(BROKEN, TOOLS, {"type": "stop"}, "auto")


@pytest.mark.parametrize("as_dict", [False, True])
def test_nonstream_rejects_undeclared_tool_name(monkeypatch, as_dict):
    obj = chat(monkeypatch, [])
    tools = [tool.model_dump() for tool in TOOLS] if as_dict else TOOLS
    with pytest.raises(KimiK3ParseError, match="unknown_tool"):
        obj._process_tool_calls(
            VALID.replace('tool="bash"', 'tool="undeclared"'),
            tools,
            {"type": "stop"},
            "auto",
        )


def test_nonstream_native_failure_maps_to_502(monkeypatch):
    obj = chat(monkeypatch, [("", "stop")])
    obj._build_chat_response = Mock(side_effect=KimiK3ParseError("invalid_header"))
    response = asyncio.run(obj._handle_non_streaming_request(None, request(), None))
    assert response.status_code == 502
    assert "invalid_header" in response.body.decode()


def test_facades_preserve_length_semantics():
    parser = FunctionCallParser(TOOLS, "kimi_k3")
    _, calls = parser.parse_non_stream(f"{O}tools{S}" + CALL, finish_reason="length")
    assert len(calls) == 1
    parser = FunctionCallParser(TOOLS, "kimi_k3")
    parser.parse_stream_chunk(f"{O}tools{S}" + CALL)
    assert parser.parse_stream_end(finish_reason="length") == ("", [])


@pytest.mark.parametrize("text", [BROKEN, f"thought{C}think0" + VALID])
def test_reasoning_guard_catches_bad_headers_before_channel_cleanup(text):
    for split in range(len(text) + 1):
        parser = ReasoningParser("kimi_k3", tool_call_parser_active=True)
        with pytest.raises(KimiK3ParseError, match="invalid_header"):
            parser.parse_stream_chunk(text[:split])
            parser.parse_stream_chunk(text[split:])
            parser.parse_stream_end()
    parser = ReasoningParser("kimi_k3", tool_call_parser_active=True)
    with pytest.raises(KimiK3ParseError, match="invalid_header"):
        parser.parse_non_stream(text)


def responses_request():
    return ResponsesRequest(
        model="x",
        input="synthetic",
        stream=True,
        store=False,
        tools=[
            {
                "type": "function",
                "name": "bash",
                "parameters": TOOLS[0].function.parameters,
            }
        ],
    )


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize(
    "mode", ["bad_header", "unknown_tool", "partial_second", "eof", "valid_eof"]
)
def test_responses_native_failure_never_completes(reasoning, mode):
    serving = make_serving()
    serving.tool_call_parser = "kimi_k3"
    serving.reasoning_parser = "kimi_k3" if reasoning else None
    if mode == "valid_eof":
        chunks = [engine_chunk(VALID)]
    elif mode == "bad_header":
        chunks = [engine_chunk(BROKEN, finish=True)]
    elif mode == "unknown_tool":
        chunks = [
            engine_chunk(VALID.replace('tool="bash"', 'tool="undeclared"'), finish=True)
        ]
    else:
        first = f"{O}tools{S}" + CALL
        chunks = [engine_chunk(first)]
        if mode == "partial_second":
            chunks.append(
                engine_chunk(first + f'{O}call tool="bash" index="2"{S}', finish=True)
            )
    frames = event_payloads(
        StreamFixture(serving, responses_request(), require_reasoning=reasoning).run(
            chunks
        )
    )
    assert frames[-1]["type"] == "response.failed"
    assert "Invalid Kimi K3 native output" in frames[-1]["response"]["error"]["message"]
    assert not any(frame["type"] == "response.completed" for frame in frames)


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("finish", ["stop", "length"])
def test_responses_valid_and_length_terminals(reasoning, finish):
    serving = make_serving()
    serving.tool_call_parser = "kimi_k3"
    serving.reasoning_parser = "kimi_k3" if reasoning else None
    text = VALID if finish == "stop" else f"{O}tools{S}" + CALL
    chunk = engine_chunk(text, finish=True)
    chunk["meta_info"]["finish_reason"] = {"type": finish}
    frames = event_payloads(
        StreamFixture(serving, responses_request(), require_reasoning=reasoning).run(
            [chunk]
        )
    )
    assert frames[-1]["type"] == (
        "response.completed" if finish == "stop" else "response.incomplete"
    )
    assert not any(frame["type"] == "response.failed" for frame in frames)


def test_responses_nonstream_does_not_swallow_native_failure():
    serving = make_serving()
    serving.tool_call_parser = "kimi_k3"
    serving.reasoning_parser = None
    with pytest.raises(KimiK3ParseError, match="invalid_header"):
        serving._make_response_output_items(
            responses_request(), BROKEN, None, require_reasoning=False
        )


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_responses_nonstream_passes_finish_metadata_through_text_path(finish):
    serving = make_serving()
    serving.tool_call_parser = "kimi_k3"
    raw = VALID[: -len(C + "tools" + S)]
    items = serving._make_response_output_items(
        responses_request(),
        raw,
        None,
        require_reasoning=False,
        finish_reason={"type": finish, "matched": C + "tools" + S},
    )
    assert len(items) == 1
    assert items[0].name == "bash"
    assert json.loads(items[0].arguments) == {"command": "echo synthetic"}


def test_responses_token_adapter_still_receives_ids_and_finish_metadata():
    serving = make_serving()
    output = Mock()
    output.parse_responses_output.return_value = (None, "token result", [])
    serving._encoding_output = lambda: output
    serving._parse_text_output = Mock(side_effect=AssertionError("text fallback"))
    finish = {"type": "stop", "matched": 123}
    items = serving._make_response_output_items(
        responses_request(),
        "unused",
        None,
        require_reasoning=False,
        output_ids=[7, 8],
        finish_reason=finish,
    )
    assert items[0].content[0].text == "token result"
    assert output.parse_responses_output.call_args.kwargs["output_ids"] == [7, 8]
    assert output.parse_responses_output.call_args.kwargs["finish_reason"] == finish


UNCLOSED_RESPONSE = {
    "before_tool_call": f"{O}response{S}hello" + VALID,
    "at_stop": f"{O}response{S}hello",
}


@pytest.mark.parametrize("case", sorted(UNCLOSED_RESPONSE))
def test_reasoning_cleanup_cannot_hide_unclosed_response(case):
    text = UNCLOSED_RESPONSE[case]
    reason = "invalid_boundary" if case == "before_tool_call" else "unclosed_channel"
    for split in range(len(text) + 1):
        parser = ReasoningParser("kimi_k3", tool_call_parser_active=True)
        with pytest.raises(KimiK3ParseError, match=reason):
            parser.parse_stream_chunk(text[:split])
            parser.parse_stream_chunk(text[split:])
            parser.parse_stream_end(finish_reason="stop")
    parser = ReasoningParser("kimi_k3", tool_call_parser_active=True)
    with pytest.raises(KimiK3ParseError, match=reason):
        parser.parse_non_stream(text, finish_reason="stop")


@pytest.mark.parametrize("case", sorted(UNCLOSED_RESPONSE))
@pytest.mark.parametrize("per_char", [False, True])
def test_chat_reasoning_unclosed_response_is_native_error(monkeypatch, case, per_char):
    text = UNCLOSED_RESPONSE[case]
    chunks = (
        [(char, None) for char in text] + [("", "stop")]
        if per_char
        else [(text, "stop")]
    )
    output = wire(chat(monkeypatch, chunks, reasoning=True))
    assert "KimiK3NativeOutputError" in output
    assert '"code": 502' in output or '"code":502' in output
    assert "[DONE]" not in output
    frames = [
        json.loads(line[6:])
        for line in output.splitlines()
        if line.startswith("data: {")
    ]
    assert not any(
        choice.get("finish_reason")
        for frame in frames
        for choice in frame.get("choices", [])
    )


def test_chat_reasoning_closed_response_before_tool_call_succeeds(monkeypatch):
    text = f"{O}response{S}hello{C}response{S}" + VALID
    output = wire(chat(monkeypatch, [(text, "stop")], reasoning=True))
    assert "echo synthetic" in output
    assert "[DONE]" in output
    assert "KimiK3NativeOutputError" not in output


@pytest.mark.parametrize("case", sorted(UNCLOSED_RESPONSE))
def test_responses_reasoning_unclosed_response_never_completes(case):
    serving = make_serving()
    serving.tool_call_parser = "kimi_k3"
    serving.reasoning_parser = "kimi_k3"
    frames = event_payloads(
        StreamFixture(serving, responses_request(), require_reasoning=True).run(
            [engine_chunk(UNCLOSED_RESPONSE[case], finish=True)]
        )
    )
    assert frames[-1]["type"] == "response.failed"
    assert "Invalid Kimi K3 native output" in frames[-1]["response"]["error"]["message"]
    assert not any(frame["type"] == "response.completed" for frame in frames)
