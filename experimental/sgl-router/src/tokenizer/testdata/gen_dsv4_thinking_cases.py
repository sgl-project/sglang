#!/usr/bin/env python3
"""Generate dsv4_thinking_cases.json — the golden fixtures for the router's
DeepSeek-V4 encoder parity test (`thinking_and_chat_parity_fixtures`).

Each `expected` is produced by the ENGINE itself: the real `encoding_dsv4.py`
encoder, fed the SAME preprocessing `serving_chat.py` applies before it
(empty-system insertion + `tool.model_dump()` canonicalization). That pydantic
canonicalization lives in serving_chat, NOT in encoding_dsv4 — running bare
`encode_messages` on raw tools produces the WRONG field order, so this
preprocessing MUST be replayed here. Re-run this whenever the engine encoder,
`TOOLS_TEMPLATE`, the reasoning-effort profile prompts, or the `Function` field
order changes:

    python3 gen_dsv4_thinking_cases.py <path-to-encoding_dsv4.py>

Each case is emitted for BOTH reasoning-effort profiles, with the profile
recorded in the fixture and replayed by the Rust test — so neither side can
silently assume one. Pass the encoder path EXPLICITLY when the checkout's own
`python/` tree is older than the engine you actually deploy: the default ENC
below points at this checkout, which on a long-lived branch can predate the
feature under test. A pre-#33140 encoder is rejected outright rather than
quietly emitting preview-only fixtures.

The Rust consts + these fixtures drift together if the oracle changes but this
isn't re-run — the test guards Rust-side regressions, and engine-side drift only
once this is re-run against the deployed engine.
"""

import importlib.util
import json
import os
import sys

# Default to the engine encoder shipping in THIS checkout (the router worktree is
# inside the sglang repo, which also holds the Python engine). On a long-lived
# branch that copy can lag the deployed engine badly enough that the guard below
# rejects it — pass the encoder path explicitly when it does.
# Repo root is five levels up: testdata/tokenizer/src/sgl-router/experimental/<root>.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *([os.pardir] * 5))
)
ENC = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(
        _REPO_ROOT,
        "python",
        "sglang",
        "srt",
        "entrypoints",
        "openai",
        "encoding_dsv4.py",
    )
)
spec = importlib.util.spec_from_file_location("encoding_dsv4", ENC)
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)


def fdump(f):
    """Replicate pydantic Function.model_dump(): fixed field order, defaults
    injected, extras dropped, defer_loading popped when None."""
    out = {
        "description": f.get("description"),
        "name": f["name"],
        "parameters": f.get("parameters"),
        "strict": f.get("strict", False),
    }
    if f.get("defer_loading") is not None:
        out["defer_loading"] = f["defer_loading"]
    return out


def tdump(ts):
    return [
        {"type": t.get("type", "function"), "function": fdump(t["function"])}
        for t in ts
    ]


def prep(msgs, tools):
    """Mirror serving_chat.py: flatten list content, None->'', insert empty
    system if the first message isn't system, attach model_dump'd tools to [0]."""
    msgs = [dict(m) for m in msgs]
    for m in msgs:
        if isinstance(m.get("content"), list):
            # Match process_content_for_template_format(_, "string"): text parts are
            # joined with a single space (jinja_template_utils), as the router's
            # content_to_string does — NOT concatenated.
            m["content"] = " ".join(
                p.get("text", "") for p in m["content"] if isinstance(p, dict)
            )
        if m.get("content") is None:
            m["content"] = ""
    if not msgs or msgs[0].get("role") != "system":
        msgs.insert(0, {"role": "system", "content": ""})
    if tools:
        msgs[0]["tools"] = tdump(tools)
    return msgs


PROFILES = getattr(enc, "REASONING_EFFORT_PROFILES", None)
if PROFILES is None:
    sys.exit(
        f"{ENC} has no REASONING_EFFORT_PROFILES (it predates the profile split).\n"
        "Regenerating against it would silently emit preview-only fixtures and\n"
        "drop all `official` coverage. Point this at a newer encoder:\n"
        "    python3 gen_dsv4_thinking_cases.py <newer>/encoding_dsv4.py"
    )


def render(msgs, tools, thinking, effort, profile):
    mode = "thinking" if thinking else "chat"
    # `reasoning_effort_profile` MUST be passed explicitly: upstream defaults it
    # to "preview", so omitting it silently yields preview output regardless of
    # which profile the fixture claims to cover.
    return enc.encode_messages(
        prep(msgs, tools),
        thinking_mode=mode,
        reasoning_effort=effort,
        reasoning_effort_profile=profile,
    )


READ_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
]

cases = []

# Every case is emitted once per profile: the two differ only in the block-0
# effort preamble, but that is exactly the surface #33140 added, and `official`
# is what the DeepSeek-V4-Flash-0731 checkpoint resolves to.
_PROFILES = ("preview", "official")


def add(name, msgs, tools, thinking, effort):
    for profile in _PROFILES:
        cases.append(
            {
                "name": f"{name}[{profile}]",
                "messages": msgs,
                "tools": tools,
                "thinking": thinking,
                "reasoning_effort": effort,
                "reasoning_effort_profile": profile,
                "expected": render(msgs, tools, thinking, effort, profile),
            }
        )


# --- chat-mode regression guards (must equal the pre-thinking output) ---
add("chat_single_user", [{"role": "user", "content": "ABCD"}], None, False, None)
add(
    "chat_with_tools",
    [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
    READ_TOOL,
    False,
    None,
)
# chat mode must SUPPRESS the max preamble (guarded on opts.thinking).
add(
    "chat_max_effort_suppresses_preamble",
    [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"}],
    None,
    False,
    "max",
)

# --- thinking transitions ---
add("thinking_single_user", [{"role": "user", "content": "ABCD"}], None, True, None)
add(
    "thinking_multiturn_no_tools_drops_prior_reasoning",
    [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        {"role": "user", "content": "U2"},
    ],
    None,
    True,
    None,
)
add(
    "thinking_assistant_after_last_user_keeps_reasoning_no_tools",
    [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
    ],
    None,
    True,
    None,
)
# multiple prior assistant turns before last user: both reasonings dropped.
add(
    "thinking_multiple_prior_assistants_no_tools",
    [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        {"role": "user", "content": "U3"},
    ],
    None,
    True,
    None,
)
# developer turn before last user is dropped entirely (index-shift), R1 dropped.
add(
    "thinking_developer_before_last_user_dropped_no_tools",
    [
        {"role": "user", "content": "U0"},
        {"role": "developer", "content": "D1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        {"role": "user", "content": "U2"},
    ],
    None,
    True,
    None,
)

# --- thinking WITH tools keeps ALL prior reasoning (effective_drop_thinking=False) ---
add(
    "thinking_multiturn_with_tools_keeps_reasoning",
    [
        {"role": "user", "content": "U1"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Let me read",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "/x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "FILE"},
        {"role": "user", "content": "U2"},
    ],
    READ_TOOL,
    True,
    None,
)
# the real agentic shape: TWO interleaved assistant tool-call rounds, reasoning on each.
add(
    "thinking_with_tools_two_tool_rounds_keeps_all_reasoning",
    [
        {"role": "user", "content": "U1"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "r1",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "/a"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "FA"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "r2",
            "tool_calls": [
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "/b"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "b", "content": "FB"},
        {"role": "user", "content": "U2"},
    ],
    READ_TOOL,
    True,
    None,
)
add(
    "thinking_empty_reasoning_still_emits_think_block",
    [
        {"role": "user", "content": "U1"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "R"},
    ],
    READ_TOOL,
    True,
    None,
)

# --- reasoning-effort preamble (thinking + max only) ---
add(
    "thinking_reasoning_effort_max_prefix",
    [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"}],
    None,
    True,
    "max",
)
add(
    "thinking_reasoning_effort_high_no_prefix",
    [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"}],
    None,
    True,
    "high",
)
# preamble + tools coexist: ordering is preamble < system < tools.
add(
    "thinking_max_effort_with_tools",
    [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"}],
    READ_TOOL,
    True,
    "max",
)

out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dsv4_thinking_cases.json"
)
with open(out, "w") as f:
    f.write(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {len(cases)} cases -> {out}")
