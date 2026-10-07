# SGLang processor

A thin wrapper over Dynamo's frontend crates (`dynamo-renderer`,
`dynamo-tokenizers`, `dynamo-parsers`), shared by SGLang's Rust hosts.

**Rule.** Each feature calls Dynamo directly. SGLang code exists only where
Dynamo differs from SGLang's Python serving behavior, or lacks a model; that
code names the Python source it mirrors. Hosts own request types, sampling,
validation, transport and runtime.

Hosts use the re-exported `dynamo_protocols`, `dynamo_renderer` and
`dynamo_tokenizers` so their Dynamo versions match the processor's.

The `render`, `tokenizer` and `parser` features (all default) gate each
component and its Dynamo crate, so a host that only renders and tokenizes
skips `dynamo-parsers`.

```text
src/
  model_files.rs   find files in a model dir or the HF cache
  tokenizer/       encode prompts
  render/          chat request -> prompt
    models/        model-specific SGLang code
    legacy/        conversation.py templates
  parser/          engine output -> chat events
```

## model_files

```rust
resolve_model_file(path, revision, filename) -> Option<String>
resolve_tokenizer_file(path, revision) -> Option<String>
```

SGLang-only, because Dynamo has no file discovery. `path` is a file, a model
directory, or an HF repo id that is read from the local HF cache without
network access. Tokenizer discovery prefers `tiktoken.model` / `*.tiktoken`
when `tokenizer_config.json` names a tiktoken class, then falls back to
`tokenizer.json`.

## tokenizer

```rust
load_tokenizer(path, revision, add_special_tokens) -> Result<dynamo_tokenizers::Tokenizer, String>
DynamoTokenizer::new(without_specials, with_specials)
trait TextTokenizer { encode(text, add_special_tokens), encode_segments(segments, add_special_tokens) }
```

Pass-through. `load_tokenizer` calls `Tokenizer::from_file_with_options`, and
`encode` / `encode_segments` call the Dynamo tokenizer. Dynamo fixes
`add_special_tokens` at load time, so `DynamoTokenizer` holds one handle per
setting and picks one per request.

## render

```rust
select_chat_formatter(&ChatFormatterOptions) -> (Option<ChatFormatter>, Option<String>)
load_chat_formatter(tokenizer_config, model_path, model_type, chat_template) -> Result<ChatFormatter, TemplateError>
ChatFormatter::render_prompt(&dyn OAIChatLikeRequest) -> Result<RenderedPrompt, TemplateError>
ChatFormatter::render_request(serde_json::Value, &default_kwargs) -> Result<(String, String), TemplateError>  // DeepSeek-V4
DeepSeekV4Profile::from_checkpoint(profile_override, encoder_source) -> Result<DeepSeekV4Profile, String>
ChatFormatter::resolve_thinking(&mut kwargs, tools_enabled, named_tool_choice) -> Option<bool>
ChatFormatter::stop_strs() -> Option<OneOrMany<String>>
requested_effort(&body) -> Option<&Value>
requested_thinking(&body) -> Option<bool>
```

`select_chat_formatter` (`selection.rs`) picks a formatter from
`config.json`. A `ChatFormatterOptions::chat_template` override skips the
native formatters.

| Model | Formatter |
|---|---|
| DeepSeek V4 | `models/deepseek_v4.rs` on Dynamo's V4 encoder |
| DeepSeek V3.2, Kimi K3, Inkling, other Dynamo native models | Dynamo native formatter, with SGLang's thinking defaults |
| Everything else | `load_chat_formatter` |

`load_chat_formatter` (`loader.rs`) follows Python's `template_manager.py`
order: a built-in name first; with no `--chat-template`, a legacy template
inferred from the model path, then the tokenizer config's Jinja template; with a
`--chat-template` file (`.jinja` or JSON), that file. On a missing template, the
error string is returned so the host can report it per request.

`render_prompt` is pass-through to `OAIPromptFormatter::render_prompt` for
Jinja and native formatters. It keeps Dynamo's segments so Kimi K3 can encode
with `encode_segments`.
`render_request` takes the SGLang request body, because DeepSeek-V4 reads
`task`, `continue_final_message` and the `reasoning` object, which
`OAIChatLikeRequest` lacks. It returns the continuation prefix separately, since
SGLang tokenizes it on its own. `default_kwargs` are the server's
`--default-chat-template-kwargs`, which apply after the request's own values.
On a DeepSeek-V4 formatter, `render_prompt` rebuilds the body from the trait.
`from_checkpoint` lets a host that fetches model files itself, such as
sgl-router, build `ChatFormatter::DeepSeekV4` directly. `requested_effort` and
`requested_thinking` (`reasoning.rs`) port `protocol.py`'s `reasoning` handling
once, for the renderer and DeepSeek-V4 alike. The SGLang additions hook in as follows:

| Code | Mirrors | Hook |
|---|---|---|
| `thinking.rs` | `parser/template_detection.py` | Reads the Jinja template's thinking toggle and its default; `resolve_thinking` writes that default into the kwargs before rendering. |
| `legacy/` | `parser/conversation.py` | Replaces Dynamo: a native port of `Conversation.get_prompt()`, the built-in templates, and model-path inference. |
| `models/deepseek_v4.rs` | `entrypoints/openai/serving_chat.py`, `encoding_dsv4.py`, `chat_encoding.py` | Replaces the OAI formatter. It normalises the request the way SGLang does (message dump, part flattening, tool field order, final assistant turn, `task`, thinking and effort, the checkpoint's effort profile), then calls `deepseek::v4::encode_messages_with_options`. |
| `models/kimi_k25.rs` | the Kimi K2.5 checkpoint's tool encoder | Adds `tools_ts_str` (TypeScript tool declarations) to the kwargs before the Jinja render. |

`stop_strs` returns the legacy template's stop strings; Jinja and native
formatters have none, as in Python.

**Parity.** Each `tests/fixtures/parity/<model>.json` holds requests that
SGLang's Python serving path rendered and tokenized, written by
`tests/scripts/generate_parity.py`. `tests/parity.rs` checks every fixture. To
add a model, follow the
[`processor-model-parity`](../../.claude/skills/processor-model-parity/SKILL.md)
skill.

## parser

```rust
ChatResponseProcessor::new(tool_parser, reasoning_parser, tools, tool_choice, uses_tool_call_structural_tag, parallel_tool_calls, choices)
    .with_reasoning_state(thinking)
    .process_stream(Stream<DecodedChatEvent>) -> Stream<ChatEvent>
split_reasoning(reasoning_parser, text, token_ids) -> (reasoning, normal)
ReasoningStreamSplitter::new(reasoning_parser, thinking).split(text, token_ids)
tool_constraint(tool_parser, &dynamo_tool_choice(&tool_choice), &tools, parallel_tool_calls)
    -> Result<Option<ToolConstraint>, ProcessorError>
dynamo_tool_parser_name(sglang_name) -> &str
```

`mod.rs` holds the events and the stream processor, `reasoning.rs` the
reasoning split, and `tools.rs` tool schemas, constraints and call deltas.

Pass-through for parsing:
- Reasoning goes through `ReasoningParserType::get_reasoning_parser_from_name`
  and `parse_reasoning_streaming_incremental`.
- Tool calls go through `apply_tool_calling_jail`. Decoded output is wrapped as
  OpenAI stream chunks only to feed the jail, then unwrapped into `ChatEvent`.

SGLang additions:
- **Parser names** from SGLang are mapped onto Dynamo's before construction,
  mirroring `parser/reasoning_parser.py` and
  `function_call/function_call_parser.py`.
- **Special tokens after a tool call** (qwen25 and glm47 terminators) are
  dropped.
- **`parallel_tool_calls=false`** keeps only the first call.
- **`tool_constraint`** turns `tool_choice` into the parser's structural tag,
  or a `json_schema` array of calls for `required` and named choices.

## Host example

```rust
let (formatter, error) = select_chat_formatter(&ChatFormatterOptions { tokenizer_path, model_path, ..Default::default() });
let formatter = formatter.ok_or(error)?;
let thinking = formatter.resolve_thinking(&mut kwargs, tools_enabled, named_tool_choice);
let prompt = formatter.render_prompt(&request)?;
let ids = match prompt.encode_segments() {
    Some(segments) => tokenizer.encode_segments(&segments, false)?,
    None => tokenizer.encode(prompt.as_str(), false)?,
};
let events = ChatResponseProcessor::new(/* ... */).with_reasoning_state(thinking).process_stream(decoded);
```

```sh
cargo test --manifest-path rust/Cargo.toml -p sglang-processor --locked
```
