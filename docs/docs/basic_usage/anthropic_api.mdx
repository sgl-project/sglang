---
title: "Anthropic-Compatible API"
description: "Use the Anthropic Messages API (/v1/messages) with SGLang, including Claude Code integration and prefix-cache tuning."
---
SGLang ships an Anthropic-compatible `/v1/messages` endpoint so any client built for the Anthropic
Messages API — including the Anthropic SDKs and agentic CLIs such as Claude Code — can talk to a
self-hosted SGLang server without changes. A complete reference for the API is available in the
[Anthropic API Reference](https://docs.anthropic.com/en/api/messages).

The endpoint is registered automatically on every SGLang server; no extra flag is required to enable it.
It reuses the same model, chat template, and reasoning / tool-call parsers as the OpenAI-compatible
endpoint, and supports both non-streaming and streaming responses, tool use, and a `count_tokens` route.

This tutorial covers:

- `POST /v1/messages` (non-streaming and streaming)
- `POST /v1/messages/count_tokens`
- Pointing **Claude Code** at the server, including the `CLAUDE_CODE_ATTRIBUTION_HEADER` setting that is
  required for good prefix-cache reuse.

## Launch A Server

Launch the server in your terminal and wait for it to initialize. The Anthropic `/v1/messages` endpoint
is registered automatically — no extra flag is required beyond the usual server launch. The example below
is a single-node GLM-5.2-FP8 config; see the
[GLM-5.2 cookbook](/cookbook/autoregressive/GLM/GLM-5.2) for verified commands
across hardware and quantizations.

```bash Command
sglang serve \
    --model-path zai-org/GLM-5.2-FP8 \
    --tp 8 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --host 0.0.0.0 \
    --port 30000
```

<Note>
- **The endpoint is model-agnostic.** The `/v1/messages` route is on by default for any model; GLM-5.2 is
  used here because its reasoning + tool-use output is where Claude Code integration shines, but any model
  works.
- **Model name and `[1m]`.** SGLang does not validate the request `model` field, so Claude Code can send
  any name. The `[1m]` suffix is a **client-side hint**: Claude Code only enables its 1M-context beta when
  the model name ends in `[1m]` — without it, context is capped. Set the same `glm-5.2[1m]` in the
  `ANTHROPIC_DEFAULT_*_MODEL` env vars below.
- **`--reasoning-parser` / `--tool-call-parser` are optional.** Add them when the model emits reasoning
  content (GLM-5.2, Qwen3, DeepSeek-R1, …) or when you want tool calls parsed into structured `tool_use`
  blocks. Without a tool-call parser, tool schemas are still accepted but the model's tool calls come back
  as raw text, and Claude Code cannot execute them.
- **Context length** defaults to the model's own (1M for GLM-5.2); pass `--context-length` only to cap it.
</Note>

## Send A Message

### Non-Streaming

Use the Anthropic Python SDK pointed at the server. Unlike the OpenAI SDK, the Anthropic SDK appends
`/v1/messages` itself, so `base_url` is the server root **without** a `/v1` suffix.

```python Example
from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:30000",
    api_key="EMPTY",  # SGLang does not require a real key by default
)

message = client.messages.create(
    model="zai-org/GLM-5.2-FP8",
    max_tokens=512,
    messages=[{"role": "user", "content": "List 3 countries and their capitals."}],
)
# A reasoning model may emit a `thinking` block before the `text` block —
# pick the text block rather than assuming content[0].
print(next(b.text for b in message.content if b.type == "text"))
```

**Example Output:**
```text Output
Here are 3 countries and their capitals:

1. **France** - Paris
2. **Japan** - Tokyo
3. **Brazil** - Brasília
```

### Streaming

Set `stream=True` to receive Server-Sent Events as they are produced.

```python Example
with client.messages.stream(
    model="zai-org/GLM-5.2-FP8",
    max_tokens=512,
    messages=[{"role": "user", "content": "Say this is a test"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

**Example Output:**
```text Output
This is a test.
```

### System Prompt

The top-level `system` field is accepted as a string or as a list of text blocks, matching the Anthropic
API shape:

```python Example
message = client.messages.create(
    model="zai-org/GLM-5.2-FP8",
    max_tokens=512,
    system="You are a helpful assistant that answers concisely.",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(next(b.text for b in message.content if b.type == "text"))
```

**Example Output:**
```text Output
The capital of France is Paris.
```

### Tool Use

Tool definitions follow the Anthropic `tools` schema. When the server is launched with a
`--tool-call-parser`, the model's tool calls are returned as `tool_use` content blocks:

```python Example
message = client.messages.create(
    model="zai-org/GLM-5.2-FP8",
    max_tokens=512,
    tools=[
        {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ],
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
)
print(message.stop_reason)
print([b for b in message.content if b.type == "tool_use"])
```

**Example Output:**
```text Output
tool_use
[ToolUseBlock(type='tool_use', id='toolu_01XXXX', name='get_weather', input={'city': 'Paris'})]
```

### Counting Tokens

`POST /v1/messages/count_tokens` returns the tokenized length of a request without generating a
response. It reuses the same request conversion as `/v1/messages`, so system prompts, tools, and
multi-turn history are all accounted for.

```python Example
resp = client.messages.count_tokens(
    model="zai-org/GLM-5.2-FP8",
    messages=[{"role": "user", "content": "Hello, world"}],
)
print(resp.input_tokens)
```

**Example Output:**
```text Output
15
```

## Using Claude Code

Claude Code can be pointed at an SGLang server by setting a few env vars in the shell that starts it.
With the server already running on `:30000`, export the full set and launch `claude`:

```bash Command
export ANTHROPIC_BASE_URL="http://127.0.0.1:30000"
export ANTHROPIC_AUTH_TOKEN="dummy"                 # required by Claude Code; any non-empty string works
export API_TIMEOUT_MS="3000000"                     # long timeout — reasoning + 1M-context turns are slow
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="1000000"     # let auto-compact use the full 1M window
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1   # drop autoupdater/telemetry/error-reporting noise
export CLAUDE_CODE_ATTRIBUTION_HEADER=0             # required for prefix-cache reuse — see below
export ANTHROPIC_DEFAULT_HAIKU_MODEL="glm-5.2[1m]"    # [1m] suffix enables Claude Code's 1M-context beta
export ANTHROPIC_DEFAULT_SONNET_MODEL="glm-5.2[1m]"  # [1m] suffix enables Claude Code's 1M-context beta
export ANTHROPIC_DEFAULT_OPUS_MODEL="glm-5.2[1m]"     # [1m] suffix enables Claude Code's 1M-context beta
claude
```

Each var matters:

- **`ANTHROPIC_BASE_URL`** — points Claude Code at your SGLang server instead of the Anthropic API.
- **`ANTHROPIC_AUTH_TOKEN`** — Claude Code requires a non-empty auth token; SGLang accepts any value
  when launched without `--api-key`.
- **`API_TIMEOUT_MS`** — raise it; reasoning models with long outputs and 1M-context turns routinely
  exceed the default timeout.
- **`ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL`** — the model name Claude Code sends for each tier.
  SGLang does not validate this field, so any name works. Use `glm-5.2[1m]`: the `[1m]` suffix is a
  client-side hint that enables Claude Code's 1M-context beta (without it, context is capped).
- **`CLAUDE_CODE_AUTO_COMPACT_WINDOW`** — set to `1000000` so auto-compaction uses the full 1M window
  instead of the default, keeping long sessions alive.

<Tip>
Instead of exporting these in every shell, persist them in `~/.claude/settings.json` under the `env` key
— they apply to all Claude Code sessions:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:30000",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.2[1m]",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2[1m]",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.2[1m]"
  }
}
```
</Tip>

### Required: `CLAUDE_CODE_ATTRIBUTION_HEADER=0` for prefix-cache reuse

<Note>
**Set this whenever Claude Code routes through SGLang (or any non-Anthropic gateway).** Without it,
multi-turn conversations re-prefill the whole history every turn.
</Note>

Claude Code prepends a per-request attribution block to the start of the system prompt, of the form
`x-anthropic-billing-header: cc_version=<ver>.<per-request-hash>; cc_entrypoint=...; cch=<hash>;`. The
per-request hash is the **first token to differ between turns**, so the radix prefix cache can only reuse
the short prefix before that hash and re-prefills the system prompt plus the entire conversation history
on every turn.

Setting `CLAUDE_CODE_ATTRIBUTION_HEADER=0` removes the whole attribution line from the system prompt.
This is a documented Claude Code env var whose explicit purpose is to "improve prompt-cache hit rates when
routing through an [LLM gateway](https://code.claude.com/docs/en/llm-gateway)" (see the [Claude Code env-vars reference](https://code.claude.com/docs/en/env-vars)).

<Note>
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` does **not** remove the attribution block — it only covers
autoupdater/telemetry/error reporting. The attribution header is a separate code path; use
`CLAUDE_CODE_ATTRIBUTION_HEADER=0` for it.
</Note>

## Troubleshooting

**Connection refused / `fetch failed`** — Ensure the server is up and the port in `ANTHROPIC_BASE_URL`
matches `--port` (default 30000). If you set `ANTHROPIC_BASE_URL` to a remote host, confirm it's reachable
and not behind a proxy that blocks the connection.

**`Model not found` / 404 from the server** — SGLang does not validate the request `model` field and
serves whatever model was loaded at startup, so a 404 usually means the request did not reach the
`/v1/messages` route at all. Confirm `ANTHROPIC_BASE_URL` points at the server (not missing the port) and
that the server finished loading.

**Tool calls not working / returned as raw text** — Launch the server with the correct
`--tool-call-parser` for your model (e.g. `glm47`, `qwen3`). Without it the `tools` field is still accepted
but the model's tool calls come back as text instead of `tool_use` blocks, and Claude Code cannot execute
them.

**Slow / re-prefills the whole history every turn** — You are missing
`CLAUDE_CODE_ATTRIBUTION_HEADER=0`. Claude Code's per-request attribution hash in the system prompt
defeats radix prefix-cache reuse; see the section above.

**Context capped below 1M** — The model name must end in `[1m]` for Claude Code to enable its 1M-context
beta. Verify `ANTHROPIC_DEFAULT_*_MODEL` uses the `[1m]` suffix, and that the loaded model's native context
is 1M (GLM-5.2 is 1048576; pass `--context-length` only to cap it, not to extend).

## Parameters

The `/v1/messages` endpoint accepts the standard Anthropic Messages API parameters. Refer to the
[Anthropic Messages API reference](https://docs.anthropic.com/en/api/messages) for the full list.

Reasoning models are supported through the same `--reasoning-parser` mechanism as the OpenAI-compatible
endpoint; pass the model's reasoning kwarg via the request (e.g. `thinking` for DeepSeek-V3-style models,
`enable_thinking` for Qwen3-style models). See [OpenAI APIs - Completions](./openai_api_completions) for
the reasoning-parser / chat-template mapping.
