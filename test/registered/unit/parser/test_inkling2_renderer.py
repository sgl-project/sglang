"""Chat rendering and response parsing across token boundaries."""

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.parser.inkling_output import (
    InklingOutputDelta,
    InklingOutputParser,
    native_byte_ids,
)
from sglang.srt.parser.inkling_renderer import (
    load_tml_renderers,
    render_inkling_messages,
)
from sglang.srt.parser.inkling_tokenizer import INKLING_SPECIAL_TOKEN_IDS
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _ids(*parts, spec="inkling2"):
    """Spellings are specials, bytes are raw byte tokens."""
    native = load_tml_renderers(spec).tokenizer
    ids = []
    for part in parts:
        if isinstance(part, list):
            ids.extend(part)
        elif isinstance(part, bytes):
            ids.extend(native_byte_ids(spec)[byte] for byte in part)
        elif part.startswith("<|") and part.endswith("|>"):
            ids.append(native.encode_special(part[2:-2]))
        else:
            ids.extend(native.encode_ordinary(part))
    return ids


def _parse(output_ids, *, spec="inkling2", per_token=False):
    parser = InklingOutputParser(
        separate_reasoning=True, parse_tool_calls=True, spec=spec
    )
    chunks = [[token_id] for token_id in output_ids] if per_token else [output_ids]
    delta = InklingOutputDelta()
    for chunk in chunks:
        delta = delta.merge(parser.feed(chunk))
    return delta.merge(parser.finish())


def _served_tokenizer(*, ordinary=None, shift=None):
    """The checkpoint tokenizer as the adapter sees it: only encode(), as on the
    tiktoken tokenizer wrapper."""
    native = load_tml_renderers("inkling2").tokenizer

    def encode(text, add_special_tokens):
        if text.startswith("<|") and text.endswith("|>"):
            return [native.encode_special(text[2:-2]) + (text == shift)]
        return list((ordinary or native.encode_ordinary)(text))

    return SimpleNamespace(encode=encode)


@pytest.fixture(scope="module")
def served():
    return _served_tokenizer()


@pytest.fixture
def serving_context():
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    yield
    reset_context()


def _inkling2_serving(tokenizer):
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    serving = object.__new__(OpenAIServingChat)
    serving.chat_encoding_spec = "inkling2"
    serving.reasoning_parser = "inkling"
    serving.tool_call_parser = "inkling2"
    serving.supports_inline_system = False
    serving.tokenizer_manager = Mock(tokenizer=tokenizer)
    serving._encoding_adapter = serving._build_chat_encoding_adapter()
    return serving


_THINKING_CALL_OUTPUT = (
    "<|message_model|>",
    "<|content_thinking|>",
    "plan",
    "<|end_message|>",
    "<|message_model|>",
    "weather",
    "<|content_invoke_tool_json|>",
    '{"name":"weather","args":{"city":"SF"}}',
    "<|end_message|>",
    "<|content_model_end_sampling|>",
)
_WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "weather", "parameters": {"type": "object"}},
}


def _message(kind, body, *, name=None, closed=True):
    return (
        "<|message_model|>",
        *([name] if name else []),
        f"<|content_{kind}|>",
        body,
        *(["<|end_message|>"] if closed else []),
    )


def _call(body, *, closed=True):
    return _message("invoke_tool_json", body, name="lookup", closed=closed)


_END_SAMPLING = ("<|content_model_end_sampling|>",)
_LITERAL = '<|end_message|><|content_invoke_tool_json|> 中文 "quote" \\ tail'
_MALFORMED_CALLS = [
    '{"name":"lookup","args":{"key":"<|end_message|>"}',
    '{"name":"lookup","args":[]}',
    '{"name":7,"args":{}}',
    '"<|content_invoke_tool_json|>"',
]


@pytest.mark.parametrize("stream_reasoning", [False, True])
@pytest.mark.parametrize(
    "blocks, reasoning, content, calls",
    [
        pytest.param(
            [_message("thinking", "🦉中文"), _message("text", "answer"), _END_SAMPLING],
            "🦉中文",
            "answer",
            [],
            id="unicode-reasoning",
        ),
        pytest.param(
            [
                _call('{"name":"lookup","args":{"id":1}}'),
                _message("thinking", "Consider 雪"),
                _message("text", "First 中文"),
                _call('{"name":"lookup","args":{"id":2}}'),
                _message("text", "Last"),
                _END_SAMPLING,
            ],
            "Consider 雪",
            "First 中文Last",
            [{"id": 1}, {"id": 2}],
            id="interleaved",
        ),
        pytest.param(
            [
                _message(kind, body)
                for kind, body in [
                    ("text", ""),
                    ("text", "One"),
                    ("text", ""),
                    ("text", "二"),
                    ("thinking", ""),
                    ("thinking", "Why"),
                    ("thinking", ""),
                    ("thinking", "雪"),
                    ("text", ""),
                ]
            ]
            + [_END_SAMPLING],
            "Why雪",
            "One二",
            [],
            id="adjacent-and-empty",
        ),
        pytest.param(
            [
                _call('{"name":"lookup","args":'),
                _message("tool_error", "Failed"),
                _call('{"name":"lookup","args":{}}'),
                _message("text", "Recovered"),
                _END_SAMPLING,
            ],
            "",
            '{"name":"lookup","args":FailedRecovered',
            [{}],
            id="tool-recovery",
        ),
        pytest.param(
            [
                _message("thinking", "Consider " + _LITERAL),
                _message("text", "Literal answer: " + _LITERAL),
                _call(
                    json.dumps(
                        {"name": "lookup", "args": {"key": _LITERAL}},
                        ensure_ascii=False,
                    )
                ),
                _END_SAMPLING,
            ],
            "Consider " + _LITERAL,
            "Literal answer: " + _LITERAL,
            [{"key": _LITERAL}],
            id="literal-framing",
        ),
        *(
            pytest.param(
                [
                    _call(body),
                    _call('{"name":"lookup","args":{"id":1}}'),
                    _call('{"name":"lookup","args":{"id":2}}', closed=False),
                ],
                "",
                body,
                [{"id": 1}, {"id": 2}],
                id=f"malformed-call-{index}",
            )
            for index, body in enumerate(_MALFORMED_CALLS)
        ),
    ],
)
def test_blocks_parse_identically_at_every_token_boundary(
    blocks, reasoning, content, calls, stream_reasoning
):
    """Every split and per-token feed yields the batch result: ordered calls,
    literal marker spellings as text, and malformed calls recovering as text."""
    token_ids = _ids(*(piece for block in blocks for piece in block))
    boundaries = [[split, len(token_ids)] for split in range(len(token_ids) + 1)]
    boundaries.append(list(range(1, len(token_ids) + 1)))
    for ends in boundaries:
        parser = InklingOutputParser(
            separate_reasoning=True,
            parse_tool_calls=True,
            stream_reasoning=stream_reasoning,
            spec="inkling2",
        )
        delta, start = InklingOutputDelta(), 0
        for end in ends[:-1]:
            delta = delta.merge(parser.feed(token_ids[start:end]))
            start = end
        delta = delta.merge(parser.finish(token_ids[start:]))
        assert (delta.reasoning, delta.content) == (reasoning, content), ends
        assert [
            (call.index, call.name, json.loads(call.arguments))
            for call in delta.tool_calls
        ] == [(index, "lookup", arguments) for index, arguments in enumerate(calls)]
        assert parser.finish() == InklingOutputDelta()


@pytest.mark.parametrize("spec", ["inkling", "inkling2"])
def test_leading_ignored_special_keeps_the_header_author(spec):
    """TMLv0 skips a special such as <|begin_of_text|>, and any text after it,
    before a header; that header still decides the author, so non-model text
    never streams."""
    for prefix in (("<|begin_of_text|>",), ("<|begin_of_text|>", "noise")):
        for role in ("model", "user", "system", "tool"):
            output_ids = _ids(
                *prefix,
                f"<|message_{role}|>",
                "<|content_text|>",
                "echoed text",
                "<|end_message|>",
                spec=spec,
            )
            for per_token in (False, True):
                parsed = _parse(output_ids, spec=spec, per_token=per_token)
                assert (parsed.reasoning, parsed.content) == (
                    "",
                    "echoed text" if role == "model" else "",
                ), (prefix, role, per_token)


@pytest.mark.parametrize("spec", ["inkling", "inkling2"])
def test_end_message_before_a_header_is_skipped(spec):
    """Before any header TMLv0 skips <|end_message|> too, so it closes nothing
    and the text after it is still pre-header text, not headerless model output."""
    output_ids = _ids(
        "<|begin_of_text|>", "noise", "<|end_message|>", "echoed text", spec=spec
    )
    for per_token in (False, True):
        assert _parse(output_ids, spec=spec, per_token=per_token).content == ""


@pytest.mark.parametrize("spec", ["inkling", "inkling2"])
def test_renders_and_parses_off_the_creating_thread(spec):
    """Prompts render on executor threads and outputs parse on the event loop,
    so neither may depend on the thread that first loaded tml-renderers."""
    messages = [{"role": "user", "content": "Weather in SF?"}]
    prompt_ids = render_inkling_messages(messages, tools=[_WEATHER_TOOL], spec=spec)
    parser = InklingOutputParser(
        separate_reasoning=True, parse_tool_calls=True, spec=spec
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        rendered = executor.submit(
            render_inkling_messages, messages, tools=[_WEATHER_TOOL], spec=spec
        ).result()
        parsed = executor.submit(
            parser.finish, _ids(*_THINKING_CALL_OUTPUT, spec=spec)
        ).result()
    assert rendered == prompt_ids
    assert parsed.reasoning == "plan"
    assert [(call.name, json.loads(call.arguments)) for call in parsed.tool_calls] == [
        ("weather", {"city": "SF"})
    ]


def test_inkling2_rejects_what_the_checkpoint_cannot_represent(serving_context):
    """The checkpoint keeps the native audio IDs as unused tokens and must match
    native H200k on every structural ID and on ordinary BPE."""
    from sglang.srt.entrypoints.openai.chat_encoding import encode_simple_chat

    audio = {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}
    with pytest.raises(ValueError, match="does not accept audio input"):
        render_inkling_messages([{"role": "user", "content": [audio]}], spec="inkling2")
    parser = InklingOutputParser(
        separate_reasoning=True, parse_tool_calls=True, spec="inkling2"
    )
    reserved = INKLING_SPECIAL_TOKEN_IDS["<|content_audio_input|>"]
    with pytest.raises(ValueError, match=f"{reserved} is a reserved token"):
        parser.feed(_ids("<|message_model|>", "<|content_text|>", "a", [reserved]))
    with pytest.raises(ValueError, match="H200kBaseChatTokenizer on <\\|message_user"):
        _inkling2_serving(_served_tokenizer(shift="<|message_user|>"))
    o200k = _served_tokenizer(ordinary=load_tml_renderers().tokenizer.encode_ordinary)
    with pytest.raises(ValueError, match="H200kBaseChatTokenizer on ordinary text"):
        _inkling2_serving(o200k)
    with pytest.raises(ValueError, match="H200kBaseChatTokenizer on ordinary text"):
        encode_simple_chat(
            tokenizer=o200k,
            spec="inkling2",
            messages=[{"role": "user", "content": "q"}],
        )


@pytest.mark.parametrize(
    "reasoning_parser,accepted", [(None, True), ("inkling", True), ("qwen3", False)]
)
def test_inkling2_tool_parser_aliases_token_output(reasoning_parser, accepted):
    """inkling2 selects token-level output like inkling, and mixing it with
    another format's reasoning parser stays a startup error."""
    from sglang.srt.entrypoints.openai.serving_inkling import (
        InklingChatEncodingAdapter,
    )

    def build():
        return InklingChatEncodingAdapter(
            tokenizer=None,
            reasoning_parser=reasoning_parser,
            tool_call_parser="inkling2",
            chat_hooks=Mock(),
        )

    if not accepted:
        with pytest.raises(ValueError, match="Inkling token parsing requires"):
            build()
        return
    adapter = build()
    assert adapter.output is adapter


@pytest.mark.parametrize("incremental", [False, True])
def test_held_reasoning_streams_only_at_the_thinking_boundary(
    serving_context, served, incremental
):
    """Bug regression: with stream_reasoning=False, invalid bytes inside thinking
    streamed partial reasoning chunks before the block closed."""
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.runtime_context import get_context

    adapter = _inkling2_serving(served)._encoding_adapter
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "q"}],
        stream_reasoning=False,
    )
    chunks = [
        _ids("<|message_model|>", "<|content_thinking|>", "left", b"\xff"),
        _ids("mid", b"\xff\xfe"),
        _ids("right", "<|end_message|>"),
        _ids(
            "<|message_model|>", "<|content_text|>", "A", "<|end_message|>",
            "<|content_model_end_sampling|>",
        ),
    ]  # fmt: skip
    output_ids = [token_id for chunk in chunks for token_id in chunk]
    finish = {"type": "stop", "matched": output_ids[-1]}
    parser_dict, per_chunk = {}, []
    with get_context().override_server_args(incremental_streaming_output=incremental):
        for index, chunk in enumerate(chunks):
            sent = sum(len(previous) for previous in chunks[: index + 1])
            finish_reason = finish if index == len(chunks) - 1 else None
            lines = adapter.stream_chat_chunks(
                content={
                    "output_ids": chunk if incremental else output_ids[:sent],
                    "meta_info": {
                        "id": "chatcmpl-1",
                        "completion_tokens": sent,
                        "finish_reason": finish_reason,
                    },
                },
                index=0,
                request=request,
                created=0,
                parser_dict=parser_dict,
                has_tool_calls={},
                choice_logprobs=None,
                finish_reason_type=finish_reason and finish_reason["type"],
                usage=None,
            )
            deltas = [
                json.loads(line[len("data: ") :])["choices"][0]["delta"]
                for line in lines
            ]
            per_chunk.append(
                (
                    "".join(d.get("reasoning_content") or "" for d in deltas),
                    "".join(d.get("content") or "" for d in deltas),
                )
            )
    assert per_chunk == [("", ""), ("", ""), ("left�mid��right", ""), ("", "A")]


@pytest.mark.parametrize("incremental", [False, True])
def test_abort_never_completes_an_open_tool_call(serving_context, served, incremental):
    """Bug regression: an abort while a call with complete JSON was still open
    emitted it as an executable tool call through Chat and Responses."""
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ResponsesRequest,
    )
    from sglang.srt.runtime_context import get_context

    adapter = _inkling2_serving(served)._encoding_adapter
    output_ids = _ids(*_THINKING_CALL_OUTPUT[:8])
    abort = {"type": "abort"}
    chat = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[_WEATHER_TOOL],
    )
    steps = (
        (output_ids, None),
        (output_ids[-1:] if incremental else output_ids, abort),
    )
    with get_context().override_server_args(incremental_streaming_output=incremental):
        parser_dict, lines = {}, []
        for ids, finish_reason in steps:
            lines += adapter.stream_chat_chunks(
                content={
                    "output_ids": ids,
                    "meta_info": {
                        "id": "chatcmpl-1",
                        "completion_tokens": len(output_ids),
                        "finish_reason": finish_reason,
                    },
                },
                index=0,
                request=chat,
                created=0,
                parser_dict=parser_dict,
                has_tool_calls={},
                choice_logprobs=None,
                finish_reason_type=finish_reason and finish_reason["type"],
                usage=None,
            )
        responses = adapter.new_responses_stream(
            request=ResponsesRequest(model="m", input="weather?"),
            require_reasoning=False,
            separate_reasoning=True,
            parse_tool_calls=True,
        )
        items = [
            item
            for ids, finish_reason in steps
            for item in responses.step(
                {
                    "output_ids": ids,
                    "meta_info": {"completion_tokens": len(output_ids)},
                },
                finish_reason,
            )[2]
        ]
    assert not any(
        json.loads(line[len("data: ") :])["choices"][0]["delta"].get("tool_calls")
        for line in lines
    )
    assert items == []
    assert (
        adapter.parse_chat_response(
            request=chat, output_ids=output_ids, finish_reason=abort
        )[2]
        is None
    )


def _chat_stream_lines(adapter, request, output_ids, finish, split=4):
    parser_dict, lines = {}, []
    for ids, finish_reason in ((output_ids[:split], None), (output_ids, finish)):
        lines += adapter.stream_chat_chunks(
            content={
                "output_ids": ids,
                "meta_info": {
                    "id": "chatcmpl-1",
                    "completion_tokens": len(ids),
                    "finish_reason": finish_reason,
                },
            },
            index=0,
            request=request,
            created=0,
            parser_dict=parser_dict,
            has_tool_calls={},
            choice_logprobs=None,
            finish_reason_type=finish_reason and finish_reason["type"],
            usage=None,
        )
    return lines


def test_default_parsers_parse_chat_and_responses_from_tokens(serving_context, served):
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ResponsesRequest,
    )

    adapter = _inkling2_serving(served)._encoding_adapter
    output_ids = _ids(*_THINKING_CALL_OUTPUT)
    finish = {"type": "stop", "matched": output_ids[-1]}
    chat = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[_WEATHER_TOOL],
    )
    reasoning, content, calls, finish_reason = adapter.parse_chat_response(
        request=chat, output_ids=output_ids, finish_reason=finish
    )
    assert (reasoning, content, finish_reason["type"]) == ("plan", "", "tool_calls")
    assert [(c.function.name, json.loads(c.function.arguments)) for c in calls] == [
        ("weather", {"city": "SF"})
    ]
    deltas = [
        json.loads(line[len("data: ") :])["choices"][0]["delta"]
        for line in _chat_stream_lines(adapter, chat, output_ids, finish)
    ]
    assert "".join(d.get("reasoning_content") or "" for d in deltas) == "plan"
    assert [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for d in deltas
        for call in d.get("tool_calls") or []
    ] == [("weather", {"city": "SF"})]

    responses = ResponsesRequest(model="m", input="weather?")
    expected = ("plan", "", [("weather", {"city": "SF"})])
    reasoning, content, items = adapter.parse_responses_output(
        request=responses,
        require_reasoning=False,
        output_ids=output_ids,
        finish_reason=finish,
        separate_reasoning=True,
        parse_tool_calls=True,
    )
    assert (
        reasoning,
        content,
        [(i.name, json.loads(i.parameters)) for i in items],
    ) == expected
    stream = adapter.new_responses_stream(
        request=responses,
        require_reasoning=False,
        separate_reasoning=True,
        parse_tool_calls=True,
    )
    steps = [
        stream.step(
            {"output_ids": ids, "meta_info": {"completion_tokens": len(ids)}},
            finish_reason,
        )
        for ids, finish_reason in ((output_ids[:4], None), (output_ids, finish))
    ]
    assert (
        "".join(step[0] for step in steps),
        "".join(step[1] for step in steps),
        [(i.name, json.loads(i.parameters)) for step in steps for i in step[2]],
    ) == expected


def test_anthropic_messages_render_and_parse_through_inkling2(serving_context, served):
    from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatMessage,
        UsageInfo,
    )

    serving = _inkling2_serving(served)
    anthropic = AnthropicServing(serving)
    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 16,
            "tools": [{"name": "weather", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Consider", "signature": "s"},
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "weather",
                            "input": {"city": "SF"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "sunny",
                        }
                    ],
                },
            ],
        }
    )
    chat = anthropic._convert_to_chat_completion_request(request)
    prompt = serving._encode_messages(
        [message.model_dump() for message in chat.messages],
        chat,
        thinking_mode=None,
        image_data=[],
    )
    history = load_tml_renderers("inkling2").tokenizer.decode(prompt)
    assert history.endswith(
        "<|message_model|><|content_thinking|>Consider<|end_message|>"
        '<|message_model|>weather<|content_invoke_tool_json|>{"name":"weather","args":{"city":"SF"}}<|end_message|>'
        "<|content_model_end_sampling|>"
        "<|message_tool|>weather<|content_text|>sunny<|end_message|>"
    )

    adapter = serving._encoding_adapter
    output_ids = _ids(*_THINKING_CALL_OUTPUT)
    finish = {"type": "stop", "matched": output_ids[-1]}
    reasoning, content, calls, finish_reason = adapter.parse_chat_response(
        request=chat, output_ids=output_ids, finish_reason=finish
    )
    response = anthropic._convert_response(
        ChatCompletionResponse(
            id="chatcmpl-1",
            model="m",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=content,
                        reasoning_content=reasoning,
                        tool_calls=calls,
                    ),
                    finish_reason=finish_reason["type"],
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    blocks = [block.model_dump(exclude_none=True) for block in response.content]
    assert [(b["type"], b.get("thinking") or b.get("input")) for b in blocks] == [
        ("thinking", "plan"),
        ("tool_use", {"city": "SF"}),
    ]
    assert response.stop_reason == "tool_use"

    lines = _chat_stream_lines(adapter, chat, output_ids, finish)
    lines += [
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        )
        + "\n\n",
        "data: [DONE]\n\n",
    ]

    async def chat_stream(*args):
        for line in lines:
            yield line

    serving._generate_chat_stream = chat_stream

    async def collect():
        events = []
        async for sse in anthropic._generate_anthropic_stream(
            object(), object(), request.model_copy(update={"stream": True}), object()
        ):
            events += [
                json.loads(line[len("data: ") :])
                for line in sse.splitlines()
                if line.startswith("data: ")
            ]
        return events

    events = asyncio.run(collect())
    assert [
        e["content_block"]["type"] for e in events if e["type"] == "content_block_start"
    ] == ["thinking", "tool_use"]
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    assert "".join(d.get("thinking", "") for d in deltas) == "plan"
    assert json.loads("".join(d.get("partial_json", "") for d in deltas)) == {
        "city": "SF"
    }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
