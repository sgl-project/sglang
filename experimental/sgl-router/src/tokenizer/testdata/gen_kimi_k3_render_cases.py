#!/usr/bin/env python3
"""Generate render-parity fixtures from the model repo's own encoder.

Companion to `gen_kimi_k3_cases.py`, split off for one reason: this script needs
NOTHING but the reference encoder. `encoding_k3.py` imports only `json`,
`dataclasses` and `typing`, so its `build_chat_segments` runs in a bare
interpreter — no `tiktoken`, no `transformers`, no 2.8 MB rank file. That makes
these fixtures cheap to regenerate and lets them cover far more scenarios than the
id-level fixtures can reach without the full model repo — every prompt-shaping
path the encoder has, not just the ones a real client happens to exercise.

SEGMENTS ONLY, deliberately. The render is the half this repo reimplements; the
tokenizer half is already pinned id-for-id against the reference by
`full_vocab_parity_when_available`. Asserting segments therefore tests exactly the
half in question, against the reference as authority, and does it for inputs no
tokenizer fixture could afford to cover.

Every case is built with `image_prompts=None` because that is the ONLY way the
router can call the encoder: substituting a real image prompt requires the
pre-resize pixel dimensions, which the router never sees. `_ImagePromptState`
then yields `IMAGE_PLACEHOLDER` for each image part, and `_append_text`'s
placeholder-substitution branch is unreachable — so a literal placeholder typed
by a client stays literal text. Both properties are asserted below.

Usage:
    python3 gen_kimi_k3_render_cases.py [--model-dir DIR]

DIR defaults to `.kimi_k3_ref`, where `gen_kimi_k3_cases.py` caches the encoder
sources fetched from the model repo.
"""

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_URL = "http://x/y.png"
SECOND_URL = "http://x/z.jpg"


def load_encoder(model_dir):
    path = os.path.join(model_dir, "encoding_k3.py")
    if not os.path.isfile(path):
        sys.exit(
            f"{path} not found. Run gen_kimi_k3_cases.py first (it fetches the "
            f"encoder sources), or pass --model-dir."
        )
    spec = importlib.util.spec_from_file_location("encoding_k3", path)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: the encoder uses `from __future__ import annotations`,
    # so `@dataclass` resolves its field types by looking the module up in
    # `sys.modules`, and a module_from_spec that is not registered yet fails there.
    sys.modules["encoding_k3"] = module
    # `encoding_k3` is self-contained (json/dataclasses/typing only), so importing
    # it does not pull the model repo's tokenizer or any third-party package.
    spec.loader.exec_module(module)
    return module


def multimodal_cases(enc):
    """Image scenarios: (name, messages, tools, kwargs).

    Ordered roughly text-shape first, then image placement, then interactions
    with the other prompt-shaping options.
    """
    ph = enc.IMAGE_PLACEHOLDER
    return [
        # --- how an image reference is spelled -------------------------------
        # The reference keys ONLY on `part["type"]`, never on the payload shape,
        # so all three of these must render the identical placeholder.
        (
            "mm_image_url_string",
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": IMAGE_URL}],
                }
            ],
            None,
            {},
        ),
        (
            "mm_image_url_object",
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": IMAGE_URL}}],
                }
            ],
            None,
            {},
        ),
        (
            "mm_bare_image_type",
            [{"role": "user", "content": [{"type": "image", "image": IMAGE_URL}]}],
            None,
            {},
        ),
        (
            "mm_image_url_object_extra_fields",
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_URL, "detail": "high"},
                        }
                    ],
                }
            ],
            None,
            {},
        ),
        # --- placement within a message -------------------------------------
        (
            "mm_image_only",
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": IMAGE_URL}],
                }
            ],
            None,
            {},
        ),
        (
            "mm_image_before_text",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": IMAGE_URL},
                        {"type": "text", "text": "what is this?"},
                    ],
                }
            ],
            None,
            {},
        ),
        (
            "mm_text_image_text",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look: "},
                        {"type": "image_url", "image_url": IMAGE_URL},
                        {"type": "text", "text": " ok?"},
                    ],
                }
            ],
            None,
            {},
        ),
        (
            "mm_many_images_one_message",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": IMAGE_URL},
                        {"type": "image_url", "image_url": SECOND_URL},
                        {"type": "image", "image": IMAGE_URL},
                    ],
                }
            ],
            None,
            {},
        ),
        (
            "mm_empty_text_part_beside_image",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "image_url", "image_url": IMAGE_URL},
                    ],
                }
            ],
            None,
            {},
        ),
        ("mm_empty_content_array", [{"role": "user", "content": []}], None, {}),
        # --- placement across roles and turns -------------------------------
        (
            "mm_image_in_system_message",
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "ref: "},
                        {"type": "image_url", "image_url": IMAGE_URL},
                    ],
                },
                {"role": "user", "content": "compare"},
            ],
            None,
            {},
        ),
        (
            "mm_image_in_assistant_message",
            [
                {"role": "user", "content": "here"},
                {
                    "role": "assistant",
                    "content": [{"type": "image_url", "image_url": IMAGE_URL}],
                },
                {"role": "user", "content": "and?"},
            ],
            None,
            {},
        ),
        (
            "mm_multiturn_image_then_text",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first: "},
                        {"type": "image_url", "image_url": IMAGE_URL},
                    ],
                },
                {"role": "assistant", "content": "I see it."},
                {"role": "user", "content": "describe it"},
            ],
            None,
            {},
        ),
        (
            "mm_images_in_two_messages",
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": IMAGE_URL}],
                },
                {"role": "assistant", "content": "ok"},
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": SECOND_URL}],
                },
            ],
            None,
            {},
        ),
        # --- the placeholder as CLIENT text ---------------------------------
        # With `image_prompts=None` the substitution branch in `_append_text` is
        # unreachable, so this must stay ordinary text and consume no image slot.
        # The router additionally tokenizes it with specials DISABLED, which is
        # what stops it becoming a control token; that half is covered by
        # `user_text_control_markers_are_not_promoted`.
        (
            "mm_literal_placeholder_in_user_text",
            [{"role": "user", "content": f"before {ph} after"}],
            None,
            {},
        ),
        (
            "mm_literal_placeholder_beside_real_image",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"literal {ph} here"},
                        {"type": "image_url", "image_url": IMAGE_URL},
                    ],
                }
            ],
            None,
            {},
        ),
        # --- interaction with the other prompt-shaping options --------------
        (
            "mm_image_thinking_off",
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": IMAGE_URL}],
                }
            ],
            None,
            {"thinking": False},
        ),
        (
            "mm_image_effort_low",
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": IMAGE_URL}],
                }
            ],
            None,
            {"thinking_effort": "low"},
        ),
        (
            "mm_image_no_generation_prompt",
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": IMAGE_URL}],
                }
            ],
            None,
            {"add_generation_prompt": False},
        ),
        (
            "mm_image_with_tools",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "use a tool on "},
                        {"type": "image_url", "image_url": IMAGE_URL},
                    ],
                }
            ],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "see",
                        "description": "look at an image",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    },
                }
            ],
            {},
        ),
        (
            "mm_image_with_response_format",
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": IMAGE_URL}],
                }
            ],
            None,
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "out",
                        "schema": {
                            "type": "object",
                            "properties": {"a": {"type": "string"}},
                        },
                    },
                }
            },
        ),
        (
            "mm_image_with_tool_choice_required",
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": IMAGE_URL}],
                }
            ],
            [{"type": "function", "function": {"name": "see", "parameters": {}}}],
            {"tool_choice": "required"},
        ),
    ]


def assistant_cases(enc):
    """The assistant turn: the `<think>` channel and the `<tools>` call block.

    The channel is STRUCTURAL — in thinking mode the open/close tags are emitted
    even with nothing to put in them, and the body is filled only when the
    reasoning text is non-empty after `.strip()`. In non-thinking mode the whole
    channel disappears. Both `reasoning_content` and its `reasoning` alias are
    read, in that order.
    """
    return [
        (
            "as_thinking_empty_channel",
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "again"},
            ],
            None,
            {},
        ),
        (
            "as_reasoning_content_filled",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "a",
                    "reasoning_content": "let me think",
                },
                {"role": "user", "content": "again"},
            ],
            None,
            {},
        ),
        (
            "as_reasoning_alias_filled",
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning": "via the alias"},
                {"role": "user", "content": "again"},
            ],
            None,
            {},
        ),
        # Both present: `reasoning_content` wins (plain `or`, so an empty string
        # falls through to the alias).
        (
            "as_reasoning_both_present",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "a",
                    "reasoning_content": "primary",
                    "reasoning": "alias",
                },
                {"role": "user", "content": "again"},
            ],
            None,
            {},
        ),
        (
            "as_reasoning_content_empty_falls_back",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "a",
                    "reasoning_content": "",
                    "reasoning": "alias wins",
                },
                {"role": "user", "content": "again"},
            ],
            None,
            {},
        ),
        # Whitespace-only does NOT fill the channel, but the tags still appear.
        (
            "as_reasoning_whitespace_only",
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": "   \n  "},
                {"role": "user", "content": "again"},
            ],
            None,
            {},
        ),
        (
            "as_thinking_off_drops_channel",
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": "hidden"},
                {"role": "user", "content": "again"},
            ],
            None,
            {"thinking": False},
        ),
        (
            "as_content_none",
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": None},
                {"role": "user", "content": "again"},
            ],
            None,
            {},
        ),
        # --- the <tools> call block ------------------------------------------
        # `index` is 1-based, and each mapping argument becomes its own tag with a
        # `type` derived from the value.
        (
            "tc_single_call_object_args",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "f",
                                "arguments": {"a": 1, "b": "two"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            None,
            {},
        ),
        (
            "tc_two_calls_index_is_one_based",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": {"a": 1}},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "g", "arguments": {"b": 2}},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r1"},
                {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            ],
            None,
            {},
        ),
        # `arguments` as a JSON STRING goes through `normalize_tool_arguments`,
        # which may keep the raw text as an `_xtml_json_block` instead of
        # re-serializing it — a distinct render path from the mapping one.
        (
            "tc_arguments_json_string",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "f",
                                "arguments": '{"a": 1, "b": [2, 3]}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            None,
            {},
        ),
        (
            "tc_arguments_unparsable_string",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "not json at all"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            None,
            {},
        ),
        (
            "tc_arguments_empty_object",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            None,
            {},
        ),
        (
            "tc_arguments_missing",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            None,
            {},
        ),
        # Nested and scalar argument values, to pin `_xtml_type` / `_xtml_value`.
        (
            "tc_argument_value_shapes",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "f",
                                "arguments": {
                                    "s": "str",
                                    "i": 1,
                                    "f": 1.5,
                                    "b": True,
                                    "n": None,
                                    "arr": [1, "x"],
                                    "obj": {"k": "v"},
                                },
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            None,
            {},
        ),
        # `fn = tool_call.get("function", tool_call)` — a FLAT call with no
        # `function` wrapper is read directly.
        (
            "tc_flat_call_without_function_wrapper",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "f", "arguments": {"a": 1}}],
                },
                {"role": "user", "content": "next"},
            ],
            None,
            {},
        ),
        # Attribute values are escaped: `&` -> `&amp;`, `"` -> `&quot;`.
        (
            "tc_tool_name_needs_attr_escaping",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": 'a&b"c', "arguments": {'k"1': "v&2"}},
                        }
                    ],
                },
                {"role": "user", "content": "next"},
            ],
            None,
            {},
        ),
        (
            "tc_content_and_calls_together",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "thinking out loud",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": {"a": 1}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            None,
            {},
        ),
    ]


def tool_and_option_cases(enc):
    """Tool declaration, tool results, and the request-level options."""
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "description": "does f",
            "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
        },
    }
    return [
        # Top-level `tools` renders the "# Tools" system message.
        ("td_top_level_tools", [{"role": "user", "content": "q"}], [tool], {}),
        (
            "td_two_tools",
            [{"role": "user", "content": "q"}],
            [tool, {"type": "function", "function": {"name": "g", "parameters": {}}}],
            {},
        ),
        ("td_empty_tools_list", [{"role": "user", "content": "q"}], [], {}),
        # `deep_sort_dict` sorts tool keys, so declaration order of the SAME tool
        # spelled with keys in a different order must render identically.
        (
            "td_key_order_independence",
            [{"role": "user", "content": "q"}],
            [
                {
                    "function": {
                        "parameters": {
                            "properties": {"a": {"type": "string"}},
                            "type": "object",
                        },
                        "description": "does f",
                        "name": "f",
                    },
                    "type": "function",
                }
            ],
            {},
        ),
        # A MESSAGE-level `tools` is the lazy-loading form and renders the
        # "## New Tools Available" body instead.
        (
            "td_dynamic_tools_on_a_message",
            [
                {"role": "user", "content": "q"},
                {"role": "system", "tools": [tool], "content": "extra tools"},
                {"role": "user", "content": "again"},
            ],
            None,
            {},
        ),
        # --- tool results ----------------------------------------------------
        # `normalize_xtml_tool_result_messages` re-sorts results to the order of
        # the calls that produced them, so a swapped pair must render sorted.
        (
            "tr_results_out_of_order_are_resorted",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": {}},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "g", "arguments": {}},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "c2", "content": "second"},
                {"role": "tool", "tool_call_id": "c1", "content": "first"},
            ],
            None,
            {},
        ),
        (
            "tr_result_named_by_its_own_name_field",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "r"},
            ],
            None,
            {},
        ),
        (
            "tr_result_content_is_parts",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": {}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [{"type": "text", "text": "part r"}],
                },
            ],
            None,
            {},
        ),
        # --- request-level options -------------------------------------------
        (
            "opt_response_format_json_object",
            [{"role": "user", "content": "q"}],
            None,
            {"response_format": {"type": "json_object"}},
        ),
        (
            "opt_response_format_nested_schema",
            [{"role": "user", "content": "q"}],
            None,
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "o",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "a": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"b": {"type": "integer"}},
                                    },
                                }
                            },
                        },
                    },
                }
            },
        ),
        (
            "opt_response_schema_direct",
            [{"role": "user", "content": "q"}],
            None,
            {
                "response_schema": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                }
            },
        ),
        (
            "opt_tool_choice_none",
            [{"role": "user", "content": "q"}],
            [tool],
            {"tool_choice": "none"},
        ),
        (
            "opt_tool_choice_auto",
            [{"role": "user", "content": "q"}],
            [tool],
            {"tool_choice": "auto"},
        ),
        (
            "opt_tool_choice_named_function",
            [{"role": "user", "content": "q"}],
            [tool],
            {"tool_choice": {"type": "function", "function": {"name": "f"}}},
        ),
        (
            "opt_effort_high",
            [{"role": "user", "content": "q"}],
            None,
            {"thinking_effort": "high"},
        ),
        (
            "opt_effort_none_value",
            [{"role": "user", "content": "q"}],
            None,
            {"thinking_effort": None},
        ),
        (
            "opt_thinking_off_no_effort",
            [{"role": "user", "content": "q"}],
            None,
            {"thinking": False, "thinking_effort": None},
        ),
        # Effort is IGNORED when thinking is off, so this must equal the case above.
        (
            "opt_thinking_off_with_effort",
            [{"role": "user", "content": "q"}],
            None,
            {"thinking": False, "thinking_effort": "low"},
        ),
        # --- roles, names, and text shapes -----------------------------------
        ("misc_empty_messages", [], None, {}),
        (
            "misc_message_with_name",
            [{"role": "user", "content": "q", "name": "alice"}],
            None,
            {},
        ),
        ("misc_unknown_role", [{"role": "narrator", "content": "aside"}], None, {}),
        (
            "misc_unicode_and_newlines",
            [{"role": "user", "content": "héllo \U0001f600\nline2\ttab \u4f60\u597d"}],
            None,
            {},
        ),
        (
            "misc_text_with_xtml_lookalikes",
            [{"role": "user", "content": '<|open|>message role="system"<|sep|>'}],
            None,
            {},
        ),
        (
            "misc_batched_conversation",
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            None,
            {},
        ),
    ]


def reject_cases(enc):
    """Inputs the REFERENCE REFUSES: (name, messages, exception type).

    The router must refuse them too. Rendering something the reference raises on
    would put the request on a prompt the engine can never produce — and because
    the engine raises, the request is failing anyway; routing it on an invented
    render just hides that. `part_missing_type` is the one that actually differed
    (the router used to treat a missing `type` as empty and render the part's
    text); the rest guard the shapes that already agreed.
    """
    return [
        # `part["type"]` is indexed, not `.get()`, so absence raises.
        (
            "part_missing_type",
            [{"role": "user", "content": [{"text": "hi"}]}],
            "KeyError",
        ),
        # Anything not `image`/`image_url` is read as a text part, so a part that
        # carries no `text` raises — including the Responses-style `input_image`
        # and a case-mismatched `IMAGE_URL`, neither of which the reference knows.
        (
            "part_input_image_by_url",
            [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": IMAGE_URL}],
                }
            ],
            "KeyError",
        ),
        (
            "part_image_type_wrong_case",
            [
                {
                    "role": "user",
                    "content": [{"type": "IMAGE_URL", "image_url": IMAGE_URL}],
                }
            ],
            "KeyError",
        ),
        # `for part in content` over a non-list, or a part that is not a mapping.
        (
            "part_is_a_bare_string",
            [{"role": "user", "content": ["hello"]}],
            "TypeError",
        ),
        ("content_is_a_number", [{"role": "user", "content": 42}], "TypeError"),
        (
            "content_is_a_mapping",
            [{"role": "user", "content": {"type": "text", "text": "hi"}}],
            "TypeError",
        ),
        # `fn["name"]` is indexed, so a call with no name raises.
        (
            "tool_call_without_a_name",
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"arguments": {}}}
                    ],
                },
            ],
            "KeyError",
        ),
        # An unsupported effort is ASSERTED, not ignored, whenever thinking is on.
        (
            "thinking_effort_unsupported",
            [{"role": "user", "content": "q"}],
            "AssertionError",
            {"thinking_effort": "medium"},
        ),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(HERE, ".kimi_k3_ref"))
    args = ap.parse_args()
    enc = load_encoder(args.model_dir)

    out = []
    battery = multimodal_cases(enc) + assistant_cases(enc) + tool_and_option_cases(enc)
    for name, messages, tools, kwargs in battery:
        kwargs = dict(kwargs)
        # `apply_chat_template` sets this before calling the encoder, so a case
        # that does not mention effort must still get "max".
        kwargs.setdefault("thinking_effort", "max")
        # image_prompts is NEVER passed: see the module docstring.
        segments = enc.build_chat_segments(messages, tools=tools, **kwargs)
        out.append(
            {
                "name": name,
                "messages": messages,
                "tools": tools,
                "opts": {
                    "thinking": kwargs.get("thinking", True),
                    "thinking_effort": kwargs.get("thinking_effort"),
                    "tool_choice": kwargs.get("tool_choice"),
                    "response_format": kwargs.get("response_format"),
                    "add_generation_prompt": kwargs.get("add_generation_prompt", True),
                },
                "segments": [
                    {"text": s.text, "allow_special": s.allow_special} for s in segments
                ],
            }
        )

    for entry in reject_cases(enc):
        name, messages, expected = entry[0], entry[1], entry[2]
        kwargs = dict(entry[3]) if len(entry) > 3 else {}
        kwargs.setdefault("thinking_effort", "max")
        try:
            enc.build_chat_segments(messages, tools=None, **kwargs)
        except Exception as e:  # noqa: BLE001 - the exception TYPE is the fixture
            actual = type(e).__name__
            if actual != expected:
                sys.exit(f"{name}: expected {expected}, reference raised {actual}: {e}")
        else:
            sys.exit(f"{name}: the reference did NOT raise; it is not a reject case")
        out.append(
            {
                "name": name,
                "messages": messages,
                "tools": None,
                # The opts the reference was ACTUALLY refused with; recording a
                # different set would have the Rust side render a case the
                # reference never rejected.
                "opts": {
                    "thinking": kwargs.get("thinking", True),
                    "thinking_effort": kwargs.get("thinking_effort"),
                    "tool_choice": kwargs.get("tool_choice"),
                    "response_format": kwargs.get("response_format"),
                    "add_generation_prompt": kwargs.get("add_generation_prompt", True),
                },
                "segments": [],
                "raises": expected,
            }
        )

    dest = os.path.join(HERE, "kimi_k3_render_cases.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {len(out)} render cases to {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
