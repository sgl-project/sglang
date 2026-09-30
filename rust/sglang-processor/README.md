# SGLang processor

Reusable request preprocessing and output parsing for SGLang, built on
Dynamo's `dynamo-tokenizers`, `dynamo-renderer`, and `dynamo-parsers` crates.

The library renders chat requests with the model's chat template, lowers text
completions, tokenizes prompts, validates sampling parameters and limits, and
produces the token-in `GenerateRequest` contract consumed by SGLang. It also
parses generated output into chat events, including tool calls and reasoning.

It has no protocol handlers, transport, or runtime. Hosts such as sgl-router,
the Rust server, and the temporary
[`sglang-renderer-server`](../sglang-renderer-server) own those.

## Usage

`RendererService` is the entry point. It owns a tokenizer worker pool and
exposes `prepare_chat`, `prepare_text_requests`, `prepare_token_ids_requests`,
`tokenize_prompt`, and `tokenize_chat`. `ChatPreprocessor` renders chat
requests without a tokenizer or worker pool. See
[`tests/public_preprocessing.rs`](tests/public_preprocessing.rs) for an
example.

```sh
cargo test --manifest-path rust/Cargo.toml -p sglang-processor --locked
```
