---
name: processor-model-parity
description: Add or verify a model's chat prompt rendering and tokenization in rust/sglang-processor so it matches SGLang's Python serving path exactly (same prompt text, same token ids). Use when adding a model to sglang-processor, porting a model's rendering from sgl-router, bumping Dynamo's renderer, or debugging a processor-vs-Python prompt mismatch.
---

# Processor model parity

`rust/sglang-processor` renders chat requests for SGLang's Rust hosts (the Rust
server, sgl-router, the renderer). A model is supported only when, for the same
request body, the processor produces **the same prompt text and the same prompt
token ids** as Python's `OpenAIServingChat`. Python is the reference, Dynamo is
the implementation, and SGLang code exists only where the two differ.

Scope: the request -> prompt -> token ids path, plus the checks that let hosts
serve the model's OpenAI requests through `/generate` (see "OpenAI layer").
Multimodal preprocessing is not covered here.

The worked example is DeepSeek-V4-Flash-0731: `src/render/models/deepseek_v4.rs`
plus `tests/fixtures/parity/deepseek-v4-flash-0731.json`. Paths below are relative
to `rust/sglang-processor/` unless they start with `python/`.

## Where a model's code goes

| Piece | File | Needed when |
|---|---|---|
| Identity -> formatter | `src/render/selection.rs` (`select_chat_formatter`) | always; mirror the order of `chat_encoding.resolve_chat_encoding_spec` |
| Formatter variant | `src/render/mod.rs` (`ChatFormatter`, its `render_request` arm, plus the `render_prompt`, `stop_strs` and `resolve_thinking` arms) | always |
| SGLang-specific steps | `src/render/models/<model>.rs`, re-exported in `models/mod.rs` | only where Dynamo diverges from Python |
| Parity cases | `tests/fixtures/parity/<model-id>.json` | always |
| README row | the `render` table in `README.md` | always |

No new test code: `tests/parity.rs` checks every fixture in the directory.

`render_request(Value, ..)` takes the SGLang request **body**, not Dynamo's
`OAIChatLikeRequest`, because Python reads fields the trait lacks (`task`,
`continue_final_message`, the `reasoning` object). Pick the shape by how Python
renders the model:
- **Python calls its own encoder** (a non-`None` spec such as `dsv4` or `dsv32`):
  give the model a dedicated `ChatFormatter` variant and `render_request` arm, like
  DeepSeek-V4. If `selection.rs` already wires the model to a Dynamo
  `PromptFormatter`, replace that branch.
- **Python calls `apply_chat_template`** (Jinja, spec `None`): the first such model
  adds one generic arm that adapts the body into `OAIChatLikeRequest`, in its own
  PR. sgl-router's `ChatRequest` in
  `experimental/sgl-router/src/tokenizer/chat_formatter.rs` is a working adapter.

## Method

0. **Pick the checkpoint.** Use the one the model's cookbook page recommends
   (`docs/cookbook/`), pin its Hugging Face commit as `revision`, and cache it.
   Note whether its tokenizer ships a `chat_template`; that can change the spec.
1. **Trace the Python path for the model.** Write down every transformation
   between the HTTP body and `prompt_ids`, in order:
   - `protocol.py::ChatCompletionRequest.normalize_reasoning_inputs`: `reasoning`
     and `reasoning_effort` set the default `thinking` / `enable_thinking`.
     Reuse `render/reasoning.rs` for this rather than porting it again.
   - `serving_chat._convert_to_internal_request`: `chat_template_kwargs.reasoning_effort`
     replaces the request effort; server `--default-chat-template-kwargs` merge in.
   - `chat_encoding.resolve_chat_encoding_spec`: which branch renders the model
     (a spec such as `dsv4`, `dsv32`, `kimi_k3`, `inkling`, or `None` for Jinja).
     It also reads the server's `--tool-call-parser` and whether the tokenizer has
     a chat template. If the deployment depends on the parser, set
     `tool_call_parser` in the fixture, and add it to `ChatFormatterOptions` in the
     same PR. Note any per-checkpoint state the server resolves once (DeepSeek-V4's
     effort profile reads `encoding/encoding_dsv4.py`).
   - That spec's branch in `serving_chat._apply_jinja_template` / `_encode_messages`:
     message dumping, content flattening, `_handle_last_assistant_message`, system
     insertion, tools, model-specific fields, the encoder or `apply_chat_template`
     call, and how the prompt is tokenized.
   - The encoder itself (for example `encoding_dsv4.py`), line by line. History
     rules live there: dropped turns, `</think>` on empty reasoning, argument
     formatting, and the errors it raises.

   This list is the **divergence inventory**.
2. **Find Dynamo's counterpart.** Check `dynamo_renderer::native_formatter_for`,
   the model's low-level encoder (for example `deepseek::v4::encode_messages_with_options`),
   or the Jinja template. Prefer the lowest-level Dynamo entry that lets SGLang own
   request semantics. Dynamo's OpenAI-level formatters apply their own defaults
   (they filter tools by `tool_choice`, inject `response_format`, and map effort),
   which often differ from SGLang's. Diff the low-level encoder against Python's
   for the rules in step 1.
3. **Write one request case per inventory item** in the fixture (`name` +
   `request` only), then generate the expected outputs from Python (below). Cover
   the checklist; reuse the request shapes from the DeepSeek-V4 fixture where they apply.
4. **Implement.** Start from a plain Dynamo call and run `tests/parity.rs`. For
   each failing case, add the smallest SGLang step that fixes it, with a one-line
   doc comment naming the Python source it mirrors. Never patch the fixture to
   match Rust.
5. **Verify** text offline, then token ids with the checkpoint cached (commands
   below). If sgl-router or the renderer uses the formatter, run their suites too.

## Checklist (DeepSeek-V4 case that covers each)

- **Message dump**: roles lowercased; unknown and null fields dropped; `user`
  reduced to role and content; null content becomes `""`. (`ignored_fields`, `tool_results`)
- **Content parts**: which part types survive and the join separator. (`parts`)
- **Tools**: all request tools or only those `tool_choice` allows; field order
  and defaults of `Function.model_dump`; message-level tools; empty lists.
  (`tools`, `tools_none`, `tools_named`, `message_tools`, `empty_message_tools`)
- **Pydantic coercion**: typed request fields are coerced before rendering. For
  example, `strict: 1` and `defer_loading: "false"` become booleans, and
  `strict: null` is rejected. Probe the pydantic model in Python to get its exact
  rules; don't guess them. (`tool_bool_coercion`, `tool_strict_null`, `continuation_coerced`)
- **Tool-call arguments**: how the encoder wants them. DeepSeek-V4 takes a compact
  JSON string; other encoders format them their own way. Keep key order
  (`serde_json` `preserve_order` is declared in `Cargo.toml`). Floats must print as
  Python's `json.dumps` prints them (`1e-06`, `1e+16`, `100.0`).
  (`tool_results`, `agentic_thinking`, `tool_float_values`, `history_float_arguments`)
- **Thinking**: kwargs `thinking` > request effort (`!= "none"`) > `reasoning.enabled`
  > server default kwargs > `SGLANG_DEFAULT_THINKING`. `enabled` follows Python
  truthiness (`1` is on); strings use Python's yes-word set.
  (`thinking_false`, `thinking_none`, `drop_thinking_ignored`, `reasoning_enabled_numeric`, `reasoning_enable_string`)
- **Effort**: precedence between kwargs, `reasoning`, `reasoning_effort` and env;
  the checkpoint's profile mapping. (`effort_high`, `effort_max`, `kwargs_effort`, `effort_conflict`, `reasoning_object`)
- **Final assistant turn**: becomes a user turn, or with `continue_final_message`
  a separately tokenized prefix. Check whether content is flattened before the
  split (it is for DeepSeek-V4). Run each check where Python runs it: a final
  assistant turn's tool calls are discarded, so object checks on their arguments
  belong after the split. Hosts reach the model through `render_prompt` too, so
  check the renderer's continuation handling (`sglang-renderer` `render()`) as well.
  (`final_assistant`, `continuation_parts`, `continuation_bos`, `continuation_only`, `final_tool_call_no_arguments`, `continuation_tool_call_null_arguments`)
- **Model-specific fields and turns**: `task` placement, a system turn mid-conversation,
  inserted empty system turns. (`task_action`, `task_after_developer`, `consecutive_task`, `mid_system`)
- **History**: the encoder's rules for earlier turns: reasoning kept or dropped,
  whole turns dropped, closing tags on empty reasoning. (`multi_turn`, `thinking_multi_turn`)
- **Tokens**: Python's `tokenizer.encode` adds special tokens by default; the
  continuation prefix is encoded alone and loses a leading BOS
  (`_append_assistant_prefix_to_prompt_ids`). `tests/parity.rs` does both, using the
  fixture's `bos_token_id`; `render_prompt` returns the prefix as its own segment so
  hosts keep that boundary. (`continuation_bos`, `continuation_after_text`)
- **Errors**: reject what Python rejects. A case Python raises on is recorded with
  `error`, and `tests/parity.rs` then requires `render_request` to fail. That
  includes pydantic rejections and Python crashes (a 500 is still a rejection).
  (`invalid_tool_arguments`, `continuation_only`) If Dynamo rejects a shape Python accepts, say in the
  PR that hosts fall back to the engine for it.
- **Known gaps**: when a mismatch is inside Dynamo and out of the processor's
  reach, fix it upstream and mark the case `known_gap` with the reason. The test
  reports the case without failing, and fails once it matches so the marker gets
  removed. (`tool_float_values`: Dynamo's `deepseek::common::to_json`)

## Generating fixtures

`tests/scripts/generate_parity.py` loads the pinned snapshot and resolves the
spec and per-checkpoint state with SGLang's own resolvers. It then renders every
case through the real `OpenAIServingChat._apply_jinja_template` and writes:
- `prompt`, `token_count` and `token_sha256` for each case, or `error` when
  Python raises (`name`, `request` and `known_gap` are kept as written). An
  `AttributeError` aborts instead: the stub server lacks something, so extend it;
- the fixture-level `config` (the `config.json` fields the processor reads, plus
  resolved overrides such as the effort profile) and `bos_token_id`.

```sh
cd rust/sglang-processor
PYTHONPATH=$REPO/python HF_HUB_CACHE=<hub cache> HF_HUB_OFFLINE=1 \
    python tests/scripts/generate_parity.py tests/fixtures/parity/<model-id>.json
```

When a new spec needs more, extend `serving_chat()` in the script (one place):
- **More server state:** set the extra attribute that `__init__` resolves, for
  example `_dsv41_default_reasoning_effort`.
- **Tokenizes through `apply_chat_template(tokenize=True)`:** the recorder sees no
  text. Record the template's text output instead.

Pin `revision` to a commit hash and regenerate only on purpose.

## Verify

```sh
cd rust
cargo test -p sglang-processor --locked                                       # text, offline
HF_HUB_CACHE=<hub cache> cargo test -p sglang-processor --test parity --locked  # + token ids
cargo clippy -p sglang-processor --all-targets --locked -- -D warnings
cargo check -p sglang-processor --no-default-features --features render,tokenizer --locked
```

The token check loads the pinned commit's snapshot from the cache. When the
snapshot is missing, the test prints `token ids not checked` (visible with
`-- --nocapture`), so confirm that line is absent before claiming token parity.
With the snapshot cached, the test also checks that the checkpoint resolves the
fixture's recorded DeepSeek-V4 profile.

## OpenAI layer

A host serves a model's OpenAI requests through `/generate` (`src/openai/`) only
once its render parity holds. Two more fixtures then cover the rest of Python's
OpenAI layer:
- `tests/fixtures/reasoning_parity/<parser>.json`, from
  `tests/scripts/generate_reasoning_parity.py`: SGLang's `ReasoningParser` on
  chunked outputs. A parser is served once `src/parser/models/` ports it.
- `tests/fixtures/openai_parity/<model-id>.json`, from
  `tests/scripts/generate_openai_parity.py --model <dir> --engine-url <engine>`:
  the `/generate` body `OpenAIServingChat` builds for each case, a live engine's
  output for it, and the response Python builds from that output. Launch the
  engine with the model's cookbook flags, including its parsers.

## Pitfalls

- Dynamo version bumps change rendering silently: rerun every fixture after bumping
  `dynamo-renderer` or `dynamo-tokenizers`.
- Env vars (`SGLANG_DEFAULT_THINKING`, `SGLANG_DSV4_REASONING_EFFORT`) are read per
  request, as in Python. The generator pins them and `tests/parity.rs` clears them,
  so fixtures do not depend on the machine.
- Typed hosts (the renderer's `OAIChatLikeRequest` path) cannot carry every field,
  such as `task` and message-level `tools`. Parity is defined on `render_request`;
  report host-adapter gaps separately rather than bending the model code.
- Keep model code in its own file: no model-specific branches in `render/mod.rs`
  beyond the dispatch arm.
