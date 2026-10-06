# SGLang processor

Shared SGLang wrappers over Dynamo's `dynamo-tokenizers`, `dynamo-renderer`,
and `dynamo-parsers` crates. The standalone renderer currently uses this
library; adoption by sgl-router and the Rust server is follow-up work.

The library loads the model's chat template, including SGLang's legacy and
built-in templates and Dynamo's native DeepSeek, Kimi, and Inkling formatters.
It renders chat requests, tokenizes rendered prompts, and parses generated
output into chat events, including tool calls and reasoning.

Hosts own request types, sampling, tool-call constraints, validation, transport,
and runtime. They build their own generate requests from the rendered output.

## Usage

`select_chat_formatter` takes `ChatFormatterOptions` and returns an optional
`ChatFormatter` and an optional loading error. `ChatFormatter::render_prompt`
renders a Dynamo chat request; `stop_strs` exposes the template's stop strings.

`load_tokenizer` loads a Dynamo tokenizer. `DynamoTokenizer` implements
`TextTokenizer` for encoding rendered prompts with or without special tokens.

`ChatResponseProcessor::process_stream` parses a stream of `DecodedChatEvent`
values into `ChatEvent` values. Hosts supply decoded engine output and convert
the resulting events into their API responses.

```sh
cargo test --manifest-path rust/Cargo.toml -p sglang-processor --locked
```
