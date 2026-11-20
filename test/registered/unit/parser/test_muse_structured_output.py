"""Muse Glimmer / Muse Spark Open structured output: the answer frame, its cache
key, commentary routing and the request plumbing that selects them.

A Muse turn is ``to=self`` / ``to=commentary`` messages, then either the
``to=user`` answer or tool calls. A JSON response_format must constrain only the
answer: started earlier, the model pads whitespace or writes its header into the
JSON; applied to a tool call, the model cannot call the tool.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import xgrammar as xgr
from xgrammar import TokenizerInfo
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarBackend
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ResponsesRequest,
)
from sglang.srt.parser.answer_frame import _muse_tool_recipient_ebnf
from sglang.srt.parser.reasoning_parser import MuseGlimmerDetector, ReasoningParser
from sglang.srt.sampling.sampling_params import get_request_answer_allows_tools
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Token IDs from the Muse checkpoints' tokenizer_config.json.
SPECIAL = {
    "<|end_of_text|>": 200001,
    "<|eom|>": 200007,
    "<|eot|>": 200008,
    "<|start|>": 200022,
    "<|message|>": 200023,
}
SCHEMA = {
    "type": "object",
    "properties": {"result": {"type": "integer"}},
    "required": ["result"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "answer", "schema": SCHEMA},
}
TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]

REASONING = " to=self<|message|>Plan.<|eom|><|start|>assistant"
COMMENTARY = " to=commentary<|message|>Working.<|eom|><|start|>assistant"
ANSWER = ' to=user<|message|>{"result": 4}'
TOOL_CALL = " to=get_weather<|message|><atem:function_calls>x</atem:function_calls>"


def _encode(text):
    """ASCII bytes, with Muse special tokens as their single ids."""
    ids, i = [], 0
    while i < len(text):
        token = next((t for t in SPECIAL if text.startswith(t, i)), None)
        if token:
            ids.append(SPECIAL[token])
            i += len(token)
        else:
            ids.append(ord(text[i]))
            i += 1
    return ids


def _make_reasoner(enable_strict_thinking=False):
    # ASCII plus Muse's framing IDs keeps these checks independent of model files.
    vocab = [""] * (max(SPECIAL.values()) + 1)
    for token_id in range(128):
        vocab[token_id] = chr(token_id)
    for token, token_id in SPECIAL.items():
        vocab[token_id] = token
    info = TokenizerInfo(
        vocab, stop_token_ids=[SPECIAL["<|eot|>"], SPECIAL["<|end_of_text|>"]]
    )
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: _encode(text),
        decode=lambda ids, **_: "".join(vocab[i] for i in ids),
        init_xgrammar=lambda: (info, None),
    )
    backend = XGrammarGrammarBackend(tokenizer, len(vocab))
    reasoner = ReasonerGrammarBackend(
        backend,
        SimpleNamespace(detector=MuseGlimmerDetector()),
        tokenizer,
        enable_strict_thinking=enable_strict_thinking,
    )
    return reasoner, len(vocab)


def _accepts(reasoner, vocab_size, key, turn):
    """Whether the request grammar for `key` admits `turn`, then <|eot|>."""
    grammar = reasoner._init_value_dispatch(key, reasoning=True)
    for token_id in _encode(turn) + [SPECIAL["<|eot|>"]]:
        mask = torch.full((1, (vocab_size + 31) // 32), -1, dtype=torch.int32)
        grammar.fill_vocab_mask(mask, 0)
        if not int(mask[0, token_id // 32].item()) & (1 << (token_id % 32)):
            return False
        grammar.accept_token(token_id)
    return grammar.is_terminated()


@pytest.mark.parametrize(
    ("recipient", "is_tool"),
    [
        ("get_weather", True),
        ("functions.get_weather", True),
        ("mcp:files.read_file", True),
        ("users_db", True),  # extends a role name
        ("u", True),  # prefix of a role name
        ("commentar", True),
        ("selfie", True),
        ("user", False),
        ("self", False),
        ("commentary", False),
        ("", False),
        ("get weather", False),
    ],
)
def test_tool_recipient_is_any_name_except_a_role(recipient, is_tool):
    """A tool branch that admitted `user` would reach the answer unconstrained;
    one that rejected a valid tool name would make the tool uncallable."""
    grammar = xgr.Grammar.from_ebnf(_muse_tool_recipient_ebnf())
    assert _is_grammar_accept_string(grammar, f" to={recipient}") is is_tool


@pytest.mark.parametrize(
    ("turn", "with_tools", "without_tools"),
    [
        (REASONING + ANSWER, True, True),
        (REASONING + COMMENTARY + ANSWER, True, True),
        (ANSWER, True, True),
        (REASONING + TOOL_CALL, True, False),
        (
            REASONING
            + " to=get_weather<|message|>a<|eom|><|start|>assistant"
            + " to=self<|message|>b<|eom|><|start|>assistant to=get_time<|message|>c",
            True,
            False,
        ),
        # No tool result yet, so a turn cannot call a tool and also answer.
        (" to=get_weather<|message|>a<|eom|><|start|>assistant" + ANSWER, False, False),
        (REASONING + " to=user<|message|>four", False, False),
        (ANSWER + "<|eom|><|start|>assistant" + COMMENTARY, False, False),
    ],
)
def test_answer_frame_ends_in_the_answer_or_offered_tools(
    turn, with_tools, without_tools
):
    """The JSON grammar starts at `to=user` only; with tools offered the turn
    may end in tool calls instead, never in an unconstrained answer."""
    reasoner, vocab_size = _make_reasoner()
    key = ("json", json.dumps(SCHEMA))
    no_tools = reasoner.get_request_grammar_key(key, input_ids=[], thinking_budget=None)
    tools = reasoner.get_request_grammar_key(
        key, input_ids=[], thinking_budget=None, allows_tools=True
    )
    assert _accepts(reasoner, vocab_size, no_tools, turn) is without_tools
    assert _accepts(reasoner, vocab_size, tools, turn) is with_tools


def test_cache_key_separates_tools_and_thinking_budget(monkeypatch):
    """Requests share compiled grammars by key: a collision between the tools
    and no-tools frames, or across budgets, would serve the wrong grammar."""
    monkeypatch.setenv("SGLANG_MAX_THINK_TOKENS", "64")
    reasoner, _ = _make_reasoner(enable_strict_thinking=True)
    key = ("json", json.dumps(SCHEMA))

    def request_key(budget, tools):
        return reasoner.get_request_grammar_key(
            key, input_ids=[], thinking_budget=budget, allows_tools=tools
        )

    keys = {
        (budget, tools): request_key(budget, tools)
        for budget in (None, 8, 16)
        for tools in (False, True)
    }
    assert len(set(keys.values())) == len(keys)
    assert keys[(None, False)] == key
    # A prompt that already opened the answer has no room for a tool call.
    open_answer = _encode("<|start|>") + [140680, 328, 76976, SPECIAL["<|message|>"]]
    assert reasoner.get_request_grammar_key(
        key, input_ids=open_answer, thinking_budget=None, allows_tools=True
    ) == reasoner.get_request_grammar_key(
        key, input_ids=open_answer, thinking_budget=None
    )


TURN = REASONING + COMMENTARY + ANSWER


def _parse(request, stream_chunk):
    parser = ReasoningParser(model_type="muse", stream_reasoning=True, request=request)
    if stream_chunk is None:
        return parser.parse_non_stream(TURN)
    reasoning, normal = "", ""
    for i in range(0, len(TURN), stream_chunk):
        r, n = parser.parse_stream_chunk(TURN[i : i + stream_chunk])
        reasoning += r or ""
        normal += n or ""
    return reasoning, normal


def _chat(**kwargs):
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **kwargs
    )


@pytest.mark.parametrize("stream_chunk", [None, 1, 3, 17])
def test_commentary_is_reasoning_under_a_json_format(stream_chunk):
    """Under a JSON format, content must be the constrained answer alone; a
    commentary update in content makes it invalid JSON."""
    for request in (
        _chat(response_format=RESPONSE_FORMAT),
        _chat(response_format={"type": "json_object"}),
        ResponsesRequest(
            model="m",
            input="hi",
            text={"format": {"type": "json_schema", "name": "a", "schema": SCHEMA}},
        ),
    ):
        reasoning, normal = _parse(request, stream_chunk)
        assert "Working." in reasoning
        assert json.loads(normal) == {"result": 4}


@pytest.mark.parametrize("stream_chunk", [None, 1, 17])
def test_commentary_stays_content_without_a_json_format(stream_chunk):
    for request in (_chat(), _chat(response_format={"type": "text"})):
        reasoning, normal = _parse(request, stream_chunk)
        assert "Working." not in reasoning
        assert "Working." in normal


@pytest.mark.parametrize(
    ("kwargs", "allows_tools"),
    [
        (dict(tools=TOOLS, response_format=RESPONSE_FORMAT), True),
        (dict(response_format=RESPONSE_FORMAT), False),
        (dict(tools=TOOLS, tool_choice="none", response_format=RESPONSE_FORMAT), False),
        (dict(tools=TOOLS), False),
    ],
)
def test_chat_request_sets_the_tools_flag(kwargs, allows_tools):
    params = _chat(**kwargs).to_sampling_params(
        stop=[], model_generation_config={}, tool_call_constraint=None
    )
    assert get_request_answer_allows_tools(params["custom_params"]) is allows_tools


@pytest.mark.parametrize(
    ("tools", "allows_tools"),
    [
        ([{"type": "function", "name": "get_weather", "parameters": {}}], True),
        ([], False),
    ],
)
def test_responses_request_sets_the_tools_flag(tools, allows_tools):
    request = ResponsesRequest(
        model="m",
        input="hi",
        tools=tools,
        text={"format": {"type": "json_schema", "name": "a", "schema": SCHEMA}},
    )
    params = request.to_sampling_params()
    assert get_request_answer_allows_tools(params.get("custom_params")) is allows_tools


@pytest.mark.parametrize(
    ("text_format", "response_format"),
    [
        (
            {"type": "json_schema", "name": "a", "schema": SCHEMA, "strict": True},
            {
                "type": "json_schema",
                "json_schema": {"name": "a", "schema": SCHEMA, "strict": True},
            },
        ),
        ({"type": "json_object"}, {"type": "json_object"}),
        ({"type": "text"}, None),
    ],
)
def test_responses_text_format_maps_to_the_chat_response_format(
    text_format, response_format
):
    """Chat templates read the chat-completions `response_format` shape, so the
    Responses API `text.format` must arrive in that shape."""
    request = ResponsesRequest(model="m", input="hi", text={"format": text_format})
    mapped = request.chat_response_format()
    if response_format is None:
        assert mapped is None
    else:
        assert mapped.model_dump(by_alias=True, exclude_none=True) == response_format
