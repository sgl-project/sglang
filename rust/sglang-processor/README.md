# SGLang processor

Shared SGLang wrappers over Dynamo's `dynamo-tokenizers`, `dynamo-renderer`,
and `dynamo-parsers` crates, used by every SGLang Rust host.

The library loads the model's chat template, including SGLang's legacy and
built-in templates and Dynamo's native DeepSeek, Kimi, and Inkling formatters.
It renders chat requests, derives tool-call constraints, tokenizes rendered
prompts, and parses generated output into chat events, including tool calls
and reasoning.

It has no SGLang request, sampling, or validation types, and no protocol
handlers, transport, or runtime. Hosts such as sgl-router, the Rust server,
and the standalone renderer own those and build their own generate requests
from the rendered output.

## Usage

`ChatPreprocessor::load` reads a model's chat template. `preprocess` renders
a `ChatRequest` into a `RenderedChat`:

- the prompt;
- the template stop strings;
- the tool-call constraint;
- whether reasoning output is expected;
- a `ChatResponseProcessor` factory for the generated output.

`load_tokenizer` and `DynamoTokenizer` tokenize the prompt. See
[`tests/public_preprocessing.rs`](tests/public_preprocessing.rs) for an
example.

```sh
cargo test --manifest-path rust/Cargo.toml -p sglang-processor --locked
```
