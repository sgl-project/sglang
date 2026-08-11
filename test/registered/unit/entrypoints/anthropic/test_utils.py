"""Tests for the standalone Anthropic conversion utilities.

Request/response conversion in ``utils`` delegates to the original
``AnthropicServing`` methods through a runtime-detached instance, so the
delegation tests here compare against a normally constructed
``AnthropicServing``: if the serving methods ever grow instance state that
``__init__`` provides but the detached instance does not, these fail.
Conversion semantics themselves are covered by ``test_serving.py``; the
behavior tests below cover only what ``utils`` adds — feature gates, the
composite error map, the envelope DTO seam, eager fake-SSE synthesis, and
message-ID injection.
"""

import ast
import importlib.util
import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.entrypoints.anthropic import utils  # noqa: E402
from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_FIXED_UUID = uuid.UUID(int=0x1234)
_FIXED_MSG_ID = f"msg_{_FIXED_UUID.hex}"

# Permissive policy: the delegation matrix compares the conversion semantics
# of every typed feature, so no gate may reject the fixtures.
_ALL_FEATURES = dict(
    allow_images=True,
    allow_output_config=True,
    allow_beta_fields=True,
    allow_tool_references=True,
    allow_search_results=True,
    allow_server_tools=True,
)

_DEFAULT_CTX = utils.AnthropicRequestContext(merge_inline_system=True)


class _FakeOpenAIServingChat:
    """Just enough of OpenAIServingChat for ``AnthropicServing.__init__``."""

    def __init__(self, chat_template=None):
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=chat_template)
        )


def _real_serving(merge_inline_system: bool) -> AnthropicServing:
    """A normally constructed AnthropicServing with a forced policy.

    The matrix must cover both policy values regardless of what the fake's
    (absent) chat template probes to; template detection itself is covered
    by ``test_serving.py``.
    """
    serving = AnthropicServing(_FakeOpenAIServingChat())
    serving._merge_inline_system = merge_inline_system
    return serving


def _payload(messages, **extra) -> dict:
    return {"model": "claude-test", "max_tokens": 64, "messages": messages, **extra}


def _convert(payload: dict, context=_DEFAULT_CTX) -> dict:
    request = utils.parse_anthropic_request(json.dumps(payload).encode())
    openai_request = utils.to_openai_request(request, context=context)
    return openai_request.model_dump(mode="json", exclude_none=True, by_alias=True)


_TOOLS = [
    {
        "name": "get_weather",
        "description": "w",
        "input_schema": {"type": "object", "properties": {}},
    }
]

_REQUEST_CASES = {
    "text_system_sampling": _payload(
        [{"role": "user", "content": "hello"}],
        system="be brief",
        temperature=0.5,
        top_k=20,
        top_p=0.9,
        stop_sequences=["END", "STOP"],
    ),
    "system_blocks_and_inline_system": _payload(
        [
            {"role": "user", "content": "q"},
            {"role": "system", "content": [{"type": "text", "text": " inline "}]},
        ],
        system=[{"type": "text", "text": "s1"}, {"type": "text", "text": "s2"}],
    ),
    "stream_with_options": _payload([{"role": "user", "content": "hi"}], stream=True),
    "tool_roundtrip": _payload(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "pre"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "sunny",
                        "is_error": True,
                    },
                    {"type": "text", "text": "post"},
                ],
            },
        ],
        tools=_TOOLS,
        tool_choice={"type": "auto"},
    ),
    "tool_result_variants": _payload(
        [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "id": "legacy_id", "content": None}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": "inner"}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t3",
                        "content": [
                            {"type": "text", "text": "a"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "eA==",
                                },
                            },
                            {"type": "tool_reference", "tool_name": "deferred_fn"},
                            {
                                "type": "search_result",
                                "title": "T",
                                "source": "https://s",
                            },
                        ],
                    }
                ],
            },
        ]
    ),
    "tool_choice_required": _payload(
        [{"role": "user", "content": "q"}], tools=_TOOLS, tool_choice={"type": "any"}
    ),
    "tool_choice_named": _payload(
        [{"role": "user", "content": "q"}],
        tools=_TOOLS,
        tool_choice={"type": "tool", "name": "get_weather"},
    ),
    "tool_choice_none": _payload(
        [{"role": "user", "content": "q"}], tools=_TOOLS, tool_choice={"type": "none"}
    ),
    "empty_placeholders": _payload(
        [
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {"role": "assistant", "content": []},
            {"role": "user", "content": "q"},
        ]
    ),
    "images": _payload(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "eA==",
                        },
                    },
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://x/y.png"},
                    },
                ],
            }
        ]
    ),
    "search_result_block": _payload(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "search_result",
                        "title": "T",
                        "source": "https://s",
                        "content": [{"type": "text", "text": "C"}],
                    }
                ],
            }
        ]
    ),
    "output_config_and_betas": _payload(
        [{"role": "user", "content": "q"}],
        output_config={
            "effort": "xhigh",
            "task_budget": {"type": "tokens", "total": 1000},
        },
        betas=["thinking-2025-08-04"],
    ),
    "server_tools_skipped": _payload(
        [{"role": "user", "content": "q"}],
        tools=[{"type": "web_search_20250305", "name": "web_search"}, *_TOOLS],
    ),
    "unknown_extra_keys": _payload(
        [{"role": "user", "content": "hi", "unknown_msg_key": 1}], unknown_top_key="x"
    ),
}


def _chat_response(
    message: dict, finish_reason: str = "stop", usage: dict = None
) -> ChatCompletionResponse:
    return ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "served-alias",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", **message},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage
            or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


_RESPONSE_CASES = {
    "text": ({"content": "hi"}, "stop", None),
    "reasoning_and_tools": (
        {
            "content": "text",
            "reasoning_content": "thought",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "g", "arguments": "not json"},
                },
            ],
        },
        "tool_calls",
        None,
    ),
    "empty_content": ({"content": ""}, "stop", None),
    "length_stop": ({"content": "x"}, "length", None),
    "unmapped_finish_reason": ({"content": "x"}, "content_filter", None),
    "cached_usage": (
        {"content": "hi"},
        "stop",
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    ),
}


class TestRequestDelegation(unittest.TestCase):
    def test_matches_normally_constructed_serving(self):
        for case in sorted(_REQUEST_CASES):
            for merge_inline_system in (True, False):
                with self.subTest(case=case, merge_inline_system=merge_inline_system):
                    payload = _REQUEST_CASES[case]
                    serving_request = AnthropicMessagesRequest.model_validate(payload)
                    utils_request = utils.parse_anthropic_request(
                        json.dumps(payload).encode()
                    )
                    context = utils.AnthropicRequestContext(
                        merge_inline_system=merge_inline_system, **_ALL_FEATURES
                    )
                    with patch("uuid.uuid4", return_value=_FIXED_UUID):
                        expected = _real_serving(
                            merge_inline_system
                        )._convert_to_chat_completion_request(serving_request)
                        actual = utils.to_openai_request(utils_request, context=context)
                    self.assertEqual(actual.model_dump(), expected.model_dump())

    def test_conversion_failures_match_serving(self):
        failure_payloads = {
            "named_tool_missing": _payload(
                [{"role": "user", "content": "q"}],
                tools=_TOOLS,
                tool_choice={"type": "tool", "name": "nope"},
            ),
            "required_without_tools": _payload(
                [{"role": "user", "content": "q"}], tool_choice={"type": "any"}
            ),
        }
        context = utils.AnthropicRequestContext(
            merge_inline_system=True, **_ALL_FEATURES
        )
        for case, payload in sorted(failure_payloads.items()):
            with self.subTest(case=case):
                serving_request = AnthropicMessagesRequest.model_validate(payload)
                with self.assertRaises(ValueError) as serving_error:
                    _real_serving(True)._convert_to_chat_completion_request(
                        serving_request
                    )
                utils_request = utils.parse_anthropic_request(
                    json.dumps(payload).encode()
                )
                with self.assertRaises(utils.AnthropicRequestError) as utils_error:
                    utils.to_openai_request(utils_request, context=context)
                self.assertEqual(
                    str(utils_error.exception), str(serving_error.exception)
                )


class TestRequestGoldens(unittest.TestCase):
    """Absolute value pins for the gated launch surface.

    The delegation matrix proves utils == serving at the same commit; these
    goldens pin WHAT that shared behavior is, so a deliberate serving.py
    semantic change must update them explicitly instead of moving both sides
    of the delegation comparison silently.
    """

    def test_sampling_params_and_system(self):
        dump = _convert(_REQUEST_CASES["text_system_sampling"])
        self.assertEqual(
            (dump["temperature"], dump["top_k"], dump["top_p"], dump["stop"]),
            (0.5, 20, 0.9, ["END", "STOP"]),
        )
        self.assertEqual(dump["max_tokens"], 64)
        self.assertIs(dump["stream"], False)
        self.assertNotIn("stream_options", dump)
        self.assertEqual(
            dump["messages"],
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_stream_true_sets_stream_options(self):
        dump = _convert(_REQUEST_CASES["stream_with_options"])
        self.assertIs(dump["stream"], True)
        self.assertEqual(
            dump["stream_options"],
            {"include_usage": True, "continuous_usage_stats": True},
        )

    def test_system_blocks_and_inline_system_merge(self):
        payload = _REQUEST_CASES["system_blocks_and_inline_system"]
        merged = _convert(
            payload, utils.AnthropicRequestContext(merge_inline_system=True)
        )
        self.assertEqual(
            merged["messages"][0], {"role": "system", "content": "s1\ns2\ninline"}
        )
        self.assertEqual([m["role"] for m in merged["messages"]], ["system", "user"])

        unmerged = _convert(
            payload, utils.AnthropicRequestContext(merge_inline_system=False)
        )
        self.assertEqual(
            unmerged["messages"][0], {"role": "system", "content": "s1\ns2"}
        )
        self.assertEqual(
            [m["role"] for m in unmerged["messages"]], ["system", "user", "system"]
        )

    def test_tool_use_roundtrip_and_wire_order(self):
        dump = _convert(_REQUEST_CASES["tool_roundtrip"])
        self.assertEqual(
            dump["messages"][1],
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
        )
        # Wire order preserved: user(pre) -> tool -> user(post); is_error is
        # dropped from the OpenAI tool message.
        self.assertEqual(dump["messages"][2], {"role": "user", "content": "pre"})
        self.assertEqual(
            dump["messages"][3],
            {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
        )
        self.assertEqual(dump["messages"][4], {"role": "user", "content": "post"})
        self.assertEqual(dump["tools"][0]["function"]["name"], "get_weather")

    def test_tool_result_variants(self):
        dump = _convert(
            _payload(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "id": "legacy_id", "content": None}
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t2",
                                "content": "inner",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t3",
                                "content": [
                                    {"type": "text", "text": "a"},
                                    {"type": "text", "text": "b"},
                                ],
                            }
                        ],
                    },
                ]
            )
        )
        # None content -> ""; legacy ``id`` used as tool_call_id fallback.
        self.assertEqual(
            dump["messages"][0],
            {"role": "tool", "tool_call_id": "legacy_id", "content": ""},
        )
        # Assistant-role tool_result folds into text.
        self.assertEqual(
            dump["messages"][1], {"role": "assistant", "content": "Tool result: inner"}
        )
        # Multi-text list keeps the parts list.
        self.assertEqual(
            dump["messages"][2]["content"],
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        )

    def test_tool_choice_mappings(self):
        msgs = [{"role": "user", "content": "q"}]
        self.assertEqual(
            _convert(_payload(msgs, tools=_TOOLS, tool_choice={"type": "any"}))[
                "tool_choice"
            ],
            "required",
        )
        self.assertEqual(
            _convert(_payload(msgs, tools=_TOOLS, tool_choice={"type": "none"}))[
                "tool_choice"
            ],
            "none",
        )
        self.assertEqual(
            _convert(
                _payload(
                    msgs,
                    tools=_TOOLS,
                    tool_choice={"type": "tool", "name": "get_weather"},
                )
            )["tool_choice"],
            {"type": "function", "function": {"name": "get_weather"}},
        )
        self.assertEqual(_convert(_payload(msgs, tools=_TOOLS))["tool_choice"], "auto")
        with self.assertRaisesRegex(
            utils.AnthropicRequestError, "not in the forwarded tools list"
        ):
            _convert(
                _payload(
                    msgs, tools=_TOOLS, tool_choice={"type": "tool", "name": "missing"}
                )
            )
        with self.assertRaisesRegex(
            utils.AnthropicRequestError, "requires at least one custom"
        ):
            _convert(_payload(msgs, tool_choice={"type": "any"}))

    def test_empty_text_and_empty_assistant_placeholders(self):
        dump = _convert(_REQUEST_CASES["empty_placeholders"])
        self.assertEqual(dump["messages"][0], {"role": "assistant", "content": ""})
        self.assertEqual(dump["messages"][1], {"role": "assistant", "content": ""})


class TestResponseDelegation(unittest.TestCase):
    def test_matches_normally_constructed_serving(self):
        for case in sorted(_RESPONSE_CASES):
            with self.subTest(case=case):
                message, finish_reason, usage = _RESPONSE_CASES[case]
                response = _chat_response(message, finish_reason, usage)
                with patch("uuid.uuid4", return_value=_FIXED_UUID):
                    expected = _real_serving(True)._convert_response(response)
                actual = utils.to_anthropic_response(
                    response, id_factory=lambda: _FIXED_MSG_ID
                )
                self.assertEqual(actual.model_dump(), expected.model_dump())

    def test_default_id_factory_keeps_wire_format(self):
        result = utils.to_anthropic_response(_chat_response({"content": "hi"}))
        self.assertRegex(result.id, r"^msg_[0-9a-f]{32}$")


class TestResponseGoldens(unittest.TestCase):
    """Absolute value pins for response conversion (see TestRequestGoldens)."""

    def test_empty_choices_response(self):
        no_choices = ChatCompletionResponse.model_validate(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )
        result = utils.to_anthropic_response(no_choices, id_factory=lambda: "m")
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual([b.type for b in result.content], ["text"])
        self.assertEqual(result.content[0].text, "")
        self.assertEqual(
            (result.usage.input_tokens, result.usage.output_tokens), (0, 0)
        )

    def test_block_order_and_invalid_json_tool_arguments(self):
        message, finish_reason, _ = _RESPONSE_CASES["reasoning_and_tools"]
        result = utils.to_anthropic_response(
            _chat_response(message, finish_reason), id_factory=lambda: "m"
        )
        types = [(b.type, getattr(b, "name", None)) for b in result.content]
        self.assertEqual(
            types,
            [("thinking", None), ("text", None), ("tool_use", "f"), ("tool_use", "g")],
        )
        self.assertEqual(result.content[2].input, {"a": 1})
        # Invalid JSON arguments -> empty input, never a crash.
        self.assertEqual(result.content[3].input, {})
        self.assertEqual(result.stop_reason, "tool_use")

    def test_stop_reason_mapping(self):
        length = utils.to_anthropic_response(
            _chat_response({"content": "x"}, "length"), id_factory=lambda: "m"
        )
        self.assertEqual(length.stop_reason, "max_tokens")
        unmapped = utils.to_anthropic_response(
            _chat_response({"content": "x"}, "content_filter"), id_factory=lambda: "m"
        )
        self.assertEqual(unmapped.stop_reason, "end_turn")


class TestParse(unittest.TestCase):
    def test_parse_failures_are_request_errors(self):
        with self.assertRaisesRegex(utils.AnthropicRequestError, "invalid JSON body"):
            utils.parse_anthropic_request(b"{not json")
        with self.assertRaises(utils.AnthropicRequestError):
            # missing required max_tokens
            utils.parse_anthropic_request(
                json.dumps({"model": "m", "messages": []}).encode()
            )

    def test_unknown_extra_keys_keep_pydantic_ignore_behavior(self):
        payload = _payload(
            [{"role": "user", "content": "hi", "unknown_msg_key": 1}],
            unknown_top_key="x",
        )
        request = utils.parse_anthropic_request(json.dumps(payload).encode())
        self.assertEqual(request.model, "claude-test")
        self.assertFalse(hasattr(request, "unknown_top_key"))


class TestFeatureGates(unittest.TestCase):
    def test_disabled_features_fail_closed(self):
        msgs = [{"role": "user", "content": "q"}]
        cases = {
            "thinking_param": _payload(
                msgs, thinking={"type": "enabled", "budget_tokens": 2048}
            ),
            "output_config": _payload(msgs, output_config={"effort": "high"}),
            "betas": _payload(msgs, betas=["b-1"]),
            "image_block": _payload(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "data": "eA=="},
                            }
                        ],
                    }
                ]
            ),
            "thinking_block": _payload(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "t"}],
                    }
                ]
            ),
            "redacted_thinking_block": _payload(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "x"}],
                    }
                ]
            ),
            "search_result_block": _payload(
                [{"role": "user", "content": [{"type": "search_result", "title": "t"}]}]
            ),
            "nested_tool_reference": _payload(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t",
                                "content": [
                                    {"type": "tool_reference", "tool_name": "f"}
                                ],
                            }
                        ],
                    }
                ]
            ),
            "server_tool": _payload(
                msgs, tools=[{"type": "web_search_20250305", "name": "web_search"}]
            ),
        }
        for case, payload in sorted(cases.items()):
            with self.subTest(case=case):
                with self.assertRaises(utils.AnthropicRequestError):
                    _convert(payload)

    def test_thinking_rejected_even_with_all_gates_open(self):
        context = utils.AnthropicRequestContext(
            merge_inline_system=True, **_ALL_FEATURES
        )
        payload = _payload(
            [{"role": "user", "content": "q"}],
            thinking={"type": "enabled", "budget_tokens": 2048},
        )
        with self.assertRaises(utils.AnthropicRequestError):
            _convert(payload, context)

    def test_enabled_image_conversion(self):
        context = utils.AnthropicRequestContext(
            merge_inline_system=True, allow_images=True
        )
        dump = _convert(_REQUEST_CASES["images"], context)
        parts = dump["messages"][0]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "image_url", "image_url"])
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/jpeg;base64,eA==")
        self.assertEqual(parts[2]["image_url"]["url"], "https://x/y.png")

    def test_enabled_output_config_effort_mapping(self):
        context = utils.AnthropicRequestContext(
            merge_inline_system=True, allow_output_config=True
        )
        msgs = [{"role": "user", "content": "q"}]
        high = _convert(_payload(msgs, output_config={"effort": "high"}), context)
        xhigh = _convert(_payload(msgs, output_config={"effort": "xhigh"}), context)
        self.assertEqual(high["reasoning_effort"], "high")
        self.assertEqual(xhigh["reasoning_effort"], "max")

    def test_enabled_server_tools_are_skipped_not_forwarded(self):
        context = utils.AnthropicRequestContext(
            merge_inline_system=True, allow_server_tools=True
        )
        dump = _convert(_REQUEST_CASES["server_tools_skipped"], context)
        self.assertEqual(
            [t["function"]["name"] for t in dump["tools"]], ["get_weather"]
        )

    def test_enabled_search_result_flattening(self):
        context = utils.AnthropicRequestContext(
            merge_inline_system=True, allow_search_results=True
        )
        dump = _convert(_REQUEST_CASES["search_result_block"], context)
        self.assertEqual(
            dump["messages"][0],
            {"role": "user", "content": "Title: T\nSource: https://s\nContent: C"},
        )

    def test_enabled_tool_reference_translation(self):
        context = utils.AnthropicRequestContext(
            merge_inline_system=True, allow_tool_references=True
        )
        dump = _convert(
            _payload(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t",
                                "content": [
                                    {
                                        "type": "tool_reference",
                                        "tool_name": "deferred_fn",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            ),
            context,
        )
        self.assertEqual(
            dump["messages"][0]["content"],
            [{"type": "tool_reference", "name": "deferred_fn"}],
        )

    def test_gate_depth_matches_conversion_depth_canary(self):
        """Gated blocks nested at depth >= 2 (tool_result inside tool_result)
        pass the closed gates today ONLY because conversion reads exactly one
        nesting level and drops them — wire-safe by convention. The moment
        serving.py deepens ``_convert_tool_result_content`` this fails: deepen
        ``_iter_typed_content_blocks`` together with it."""
        payload = _payload(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "outer",
                            "content": [
                                {"type": "text", "text": "depth1"},
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "inner",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": "eA==",
                                            },
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        )
        dump = _convert(payload)  # closed gates: must not raise
        self.assertEqual(
            dump["messages"][0],
            {"role": "tool", "tool_call_id": "outer", "content": "depth1"},
        )
        self.assertNotIn("image_url", json.dumps(dump))


_ERROR_BODIES = [
    b'{"error": "boom"}',
    b'{"error": {"message": "m", "type": "custom_type"}}',
    b'{"message": "top-level"}',
    b"<html>gateway</html>",
    b"",
]


class TestErrorConversion(unittest.TestCase):
    def test_matches_serving_for_shared_statuses(self):
        # Statuses present in serving.py's own ERROR_TYPE_MAP; 413/422 are
        # composite additions with no serving.py equivalent (below).
        for status in (400, 401, 403, 404, 408, 429, 500, 502, 503, 504):
            for body in _ERROR_BODIES:
                with self.subTest(status=status, body=body):
                    serving_response = _real_serving(
                        True
                    )._convert_openai_error_response(
                        SimpleNamespace(status_code=status, body=body)
                    )
                    self.assertEqual(serving_response.status_code, status)
                    envelope = utils.to_anthropic_error(status, body)
                    self.assertEqual(
                        envelope.model_dump(), json.loads(bytes(serving_response.body))
                    )

    def test_composite_statuses_follow_http_layer_policy(self):
        self.assertEqual(utils.ERROR_TYPE_MAP[413], "request_too_large")
        self.assertEqual(utils.ERROR_TYPE_MAP[422], "invalid_request_error")
        env_413 = utils.to_anthropic_error(413, b'{"error": "big"}')
        self.assertEqual(
            (env_413.error.type, env_413.error.message), ("request_too_large", "big")
        )
        env_422 = utils.to_anthropic_error(422, b'{"error": "bad"}')
        self.assertEqual(
            (env_422.error.type, env_422.error.message),
            ("invalid_request_error", "bad"),
        )
        # The composite statuses share the full message policy: custom
        # error.type honored for 4xx, empty body -> "Request failed".
        custom = utils.to_anthropic_error(
            413, b'{"error": {"message": "m", "type": "custom_type"}}'
        )
        self.assertEqual(
            (custom.error.type, custom.error.message), ("custom_type", "m")
        )
        self.assertEqual(
            utils.to_anthropic_error(422, b"").error.message, "Request failed"
        )
        # Unlisted statuses fall through to api_error in both sources.
        self.assertEqual(
            utils.to_anthropic_error(409, b'{"error": "conflict"}').error.type,
            "api_error",
        )

    def test_message_policy_goldens(self):
        """Absolute pins for the envelope message policy (see
        TestRequestGoldens for why value pins accompany the delegation
        matrix)."""
        # 4xx keeps the parsed upstream message; custom error.type honored.
        env = utils.to_anthropic_error(
            400, b'{"error": {"message": "m", "type": "custom_type"}}'
        )
        self.assertEqual((env.error.type, env.error.message), ("custom_type", "m"))
        # 5xx never echoes upstream detail or type.
        env = utils.to_anthropic_error(
            500, b'{"error": {"message": "secret", "type": "custom_type"}}'
        )
        self.assertEqual(
            (env.error.type, env.error.message), ("api_error", "Internal server error")
        )
        # Empty-body fallbacks.
        self.assertEqual(
            utils.to_anthropic_error(400, b"").error.message, "Request failed"
        )
        self.assertEqual(
            utils.to_anthropic_error(502, b"").error.message, "Internal server error"
        )
        # Non-JSON 4xx body passes through as a bounded hint.
        self.assertEqual(
            utils.to_anthropic_error(400, b"<html>gateway</html>").error.message,
            "<html>gateway</html>",
        )

    def test_composite_map_matches_http_server_source(self):
        """413/422 are a copy of the inline status->type dict in
        ``http_server.py``'s ``/v1/messages`` exception handler, which is not
        importable without loading the server app. Parse the source so a
        policy change there fails loudly here instead of silently diverging
        every external frontend."""
        origin = importlib.util.find_spec("sglang.srt.entrypoints.http_server").origin
        tree = ast.parse(Path(origin).read_text())
        handler_maps = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict) or not node.keys:
                continue
            if not all(
                key is not None
                and isinstance(key, ast.Constant)
                and isinstance(key.value, int)
                for key in node.keys
            ):
                continue
            mapping = {
                key.value: value.value
                for key, value in zip(node.keys, node.values)
                if isinstance(value, ast.Constant)
            }
            if 413 in mapping and 422 in mapping:
                handler_maps.append(mapping)
        self.assertTrue(
            handler_maps, "no status->type dict with 413/422 found in http_server.py"
        )
        for mapping in handler_maps:
            for status, error_type in mapping.items():
                self.assertEqual(
                    utils.ERROR_TYPE_MAP.get(status, "api_error"),
                    error_type,
                    f"status {status} diverged from http_server.py policy",
                )

    def test_4xx_scrub_strips_traceback_lines_and_truncates(self):
        upstream = "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "/app/handler.py", line 3, in run',
                "real cause: " + "x" * 600,
            ]
        )
        body = json.dumps({"error": {"message": upstream}}).encode()
        message = utils.to_anthropic_error(400, body).error.message
        self.assertNotIn("Traceback", message)
        self.assertNotIn('File "/', message)
        self.assertTrue(message.startswith("real cause: "))
        self.assertEqual(len(message), 501)
        self.assertTrue(message.endswith("…"))


class TestFakeSse(unittest.TestCase):
    def test_text_event_sequence_uses_request_model(self):
        events = utils.to_anthropic_fake_sse_events(
            _chat_response({"content": "hello"}),
            model="claude-test",
            id_factory=lambda: "msg_fixed",
        )
        self.assertEqual(
            [e.type for e in events],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        start = events[0].message
        # Model comes from the Anthropic request, not the backend alias.
        self.assertEqual(start.model, "claude-test")
        self.assertEqual(start.id, "msg_fixed")
        self.assertEqual(start.content, [])
        self.assertEqual(start.usage.input_tokens, 10)
        self.assertEqual(start.usage.output_tokens, 0)
        self.assertEqual(events[2].delta.text, "hello")
        self.assertEqual(events[4].delta.stop_reason, "end_turn")
        self.assertIsNone(events[4].usage.input_tokens)
        self.assertEqual(events[4].usage.output_tokens, 5)

    def test_multi_block_index_accounting(self):
        events = utils.to_anthropic_fake_sse_events(
            _chat_response(
                {
                    "content": "txt",
                    "reasoning_content": "think",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"a": 1}'},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "g", "arguments": ""},
                        },
                    ],
                },
                finish_reason="tool_calls",
            ),
            model="claude-test",
            id_factory=lambda: "m",
        )
        starts = [e for e in events if e.type == "content_block_start"]
        self.assertEqual(
            [(e.index, e.content_block.type) for e in starts],
            [(0, "thinking"), (1, "text"), (2, "tool_use"), (3, "tool_use")],
        )
        deltas = [e for e in events if e.type == "content_block_delta"]
        # A zero-argument tool call emits no input_json_delta (live stream
        # behavior).
        self.assertEqual(
            [(e.index, e.delta.type) for e in deltas],
            [(0, "thinking_delta"), (1, "text_delta"), (2, "input_json_delta")],
        )
        self.assertEqual(deltas[2].delta.partial_json, '{"a": 1}')
        stops = [e.index for e in events if e.type == "content_block_stop"]
        self.assertEqual(stops, [0, 1, 2, 3])
        self.assertEqual(events[-2].delta.stop_reason, "tool_use")
        self.assertEqual(events[-1].type, "message_stop")

    def test_unmapped_finish_reason_and_cached_usage(self):
        events = utils.to_anthropic_fake_sse_events(
            _chat_response(
                {"content": "x"},
                finish_reason="content_filter",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            ),
            model="claude-test",
            id_factory=lambda: "m",
        )
        start_usage = events[0].message.usage
        self.assertEqual(
            (
                start_usage.input_tokens,
                start_usage.cache_read_input_tokens,
                start_usage.output_tokens,
            ),
            (6, 4, 0),
        )
        message_delta = next(e for e in events if e.type == "message_delta")
        # content_filter is unmapped and falls through to end_turn.
        self.assertEqual(message_delta.delta.stop_reason, "end_turn")
        self.assertIsNone(message_delta.usage.cache_read_input_tokens)
        self.assertEqual(message_delta.usage.output_tokens, 5)


class TestFakeSseEdgeCases(unittest.TestCase):
    def test_empty_choices_emits_bare_envelope(self):
        no_choices = ChatCompletionResponse.model_validate(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )
        events = utils.to_anthropic_fake_sse_events(
            no_choices, model="claude-test", id_factory=lambda: "m"
        )
        self.assertEqual(
            [e.type for e in events], ["message_start", "message_delta", "message_stop"]
        )
        self.assertEqual(events[1].delta.stop_reason, "end_turn")


class TestImportHygiene(unittest.TestCase):
    def test_utils_import_loads_no_serving_runtime(self):
        # External frontends import this module without a serving runtime;
        # it must not pull the OpenAI serving chat, tokenizer manager, or
        # engine (serving.py keeps OpenAIServingChat under TYPE_CHECKING).
        code = (
            "import sys\n"
            "import sglang.srt.entrypoints.anthropic.utils\n"
            "banned = ('sglang.srt.entrypoints.openai.serving_chat',\n"
            "          'sglang.srt.managers.tokenizer_manager',\n"
            "          'sglang.srt.entrypoints.engine')\n"
            "loaded = [m for m in sys.modules for b in banned "
            "if m == b or m.startswith(b + '.')]\n"
            "assert not loaded, loaded\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
