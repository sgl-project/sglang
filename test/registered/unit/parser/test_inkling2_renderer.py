"""Chat rendering and response parsing across token boundaries."""

import functools
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


_TOOL_CALL_HISTORY = [
    {"role": "user", "content": "Weather in SF?"},
    {
        "role": "assistant",
        "reasoning_content": "plan",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city": "SF"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
]


def _contains(ids, sub):
    return any(ids[i : i + len(sub)] == sub for i in range(len(ids)))


def _named(messages, name):
    return [
        {**message, "name": name} if message["role"] == "assistant" else message
        for message in messages
    ]


_render2 = functools.partial(
    render_inkling_messages, tools=[_WEATHER_TOOL], spec="inkling2"
)
_TOOL_NAME_HEADER = ("<|message_model|>", "weather", "<|content_invoke_tool_json|>")
_EMPTY_HEADER = ("<|message_model|>", "<|content_invoke_tool_json|>")


@pytest.mark.parametrize("mode", ["independent", "drop_name"])
def test_unnamed_tool_calls_render_with_empty_headers(mode):
    """Without assistant names, independent and drop_name both write
    <|message_model|><|content_invoke_tool_json|>: only the header name differs from
    tool_name; the declaration, thinking, payload and tool result do not."""
    tool_name = _render2(_TOOL_CALL_HISTORY)
    empty = _render2(_TOOL_CALL_HISTORY, tool_call_author_name_mode=mode)
    assert _contains(tool_name, _ids(*_TOOL_NAME_HEADER))
    assert not _contains(tool_name, _ids(*_EMPTY_HEADER))
    assert _contains(empty, _ids(*_EMPTY_HEADER))
    assert not _contains(empty, _ids(*_TOOL_NAME_HEADER))
    assert len(tool_name) - len(empty) == len(_ids("weather"))
    assert _contains(empty, _ids("<|message_tool|>", "weather", "<|content_text|>"))


def test_independent_mode_writes_the_assistant_name():
    """independent writes a named assistant's own name in every header of its turn,
    the tool call's included; tool_name refuses that name on a tool call."""
    named = _named(_TOOL_CALL_HISTORY, "helper")
    ids = _render2(named, tool_call_author_name_mode="independent")
    assert _contains(
        ids, _ids("<|message_model|>", "helper", "<|content_invoke_tool_json|>")
    )
    assert _contains(ids, _ids("<|message_model|>", "helper", "<|content_thinking|>"))
    assert not _contains(ids, _ids(*_TOOL_NAME_HEADER))
    with pytest.raises(Exception, match="incompatible"):
        _render2(named)


def test_drop_name_mode_drops_assistant_names():
    """drop_name renders a named conversation exactly as the unnamed one."""
    named = _named(_TOOL_CALL_HISTORY, "helper") + [
        {"role": "assistant", "name": "helper", "content": "It is sunny."}
    ]
    unnamed = [
        {key: value for key, value in message.items() if key != "name"}
        for message in named
    ]
    assert _render2(named, tool_call_author_name_mode="drop_name") == _render2(
        unnamed, tool_call_author_name_mode="drop_name"
    )


def test_tool_call_author_name_mode_needs_an_inkling2_checkpoint():
    with pytest.raises(ValueError, match="must be one of"):
        render_inkling_messages(
            _TOOL_CALL_HISTORY, spec="inkling2", tool_call_author_name_mode="anonymous"
        )
    for mode in ("independent", "drop_name"):
        with pytest.raises(ValueError, match="needs an inkling2 checkpoint"):
            render_inkling_messages(
                _TOOL_CALL_HISTORY, spec="inkling", tool_call_author_name_mode=mode
            )


@pytest.mark.parametrize("per_token", [False, True])
def test_parses_a_tool_call_with_an_anonymous_header(per_token):
    output_ids = _ids(
        "<|message_model|>",
        "<|content_invoke_tool_json|>",
        '{"name":"weather","args":{"city":"SF"}}',
        "<|end_message|>",
        "<|content_model_end_sampling|>",
    )
    parsed = _parse(output_ids, per_token=per_token)
    assert [(call.name, json.loads(call.arguments)) for call in parsed.tool_calls] == [
        ("weather", {"city": "SF"})
    ]


@pytest.mark.parametrize("mode", ["tool_name", "independent", "drop_name"])
def test_tool_call_author_name_mode_comes_from_the_server_flag(served, mode):
    """--inkling2-tool-call-author-name-mode switches the header chat prompts render."""
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    publish(
        ServerArgs(model_path="dummy", inkling2_tool_call_author_name_mode=mode),
        role="tokenizer",
    )
    try:
        adapter = _inkling2_serving(served)._encoding_adapter
    finally:
        reset_context()
    assert adapter._tool_call_author_name_mode == mode


@pytest.mark.parametrize(
    "flag,checkpoint,expected",
    [
        (None, None, "tool_name"),
        (None, "independent", "independent"),
        (None, "drop_name", "drop_name"),
        ("tool_name", "independent", "tool_name"),
        ("drop_name", "independent", "drop_name"),
    ],
)
def test_tool_call_author_name_mode_defaults_to_the_checkpoint_config(
    flag, checkpoint, expected
):
    """The flag wins; without it, config.json's text_config.tool_call_author_name_mode;
    without that, tool_name."""
    from sglang.srt.configs.inkling2 import Inkling2ModelConfig
    from sglang.srt.entrypoints.openai.chat_encoding_adapters import (
        _inkling_tool_call_author_name_mode,
    )

    text_config = (
        Inkling2ModelConfig()
        if checkpoint is None
        else Inkling2ModelConfig(tool_call_author_name_mode=checkpoint)
    )
    ctx = SimpleNamespace(
        inkling2_tool_call_author_name_mode=flag,
        hf_config=SimpleNamespace(text_config=text_config),
    )
    assert _inkling_tool_call_author_name_mode(ctx) == expected


def test_checkpoint_tool_call_author_name_mode_is_validated():
    from sglang.srt.configs.inkling2 import Inkling2ModelConfig

    with pytest.raises(ValueError, match="tool_call_author_name_mode"):
        Inkling2ModelConfig(tool_call_author_name_mode="anonymous")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
