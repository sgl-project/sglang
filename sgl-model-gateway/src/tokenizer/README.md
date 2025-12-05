# Tokenizer Module

## Overview
The `sgl-model-gateway` tokenizer subsystem exposes a single `Tokenizer` facade around multiple backends
(Hugging Face JSON tokenizers, OpenAI/tiktoken models, and an in-memory mock).  It packages the
shared behaviours needed by the router–encoding user text, incrementally decoding streamed tokens,
tracking per-request state, and detecting stop conditions—behind trait objects so the rest of the
router can remain backend-agnostic.

Key capabilities:
- trait-based split between `Encoder`, `Decoder`, and `Tokenizer` for shared APIs across backends
- Hugging Face tokenizer loading (with optional chat templates) and HF Hub downloads
- heuristic selection of OpenAI/tiktoken encodings for GPT model names
- incremental decoding utilities (`DecodeStream`, `Sequence`) that handle UTF-8 boundaries
- stop sequence handling via `StopSequenceDecoder` with token-level and string-level triggers
- optional Jinja2 chat-template rendering that matches Hugging Face semantics

The implementation deliberately keeps the surface area small—metrics, batching, or SentencePiece
support mentioned in earlier drafts do **not** exist today.  This document reflects the actual code
as of `sgl-model-gateway/src/tokenizer/*`.

## Source Map
- `mod.rs` – module exports and the `Tokenizer` wrapper around `Arc<dyn Tokenizer>`
- `traits.rs` – shared traits and the `Encoding`/`SpecialTokens` helper types
- `factory.rs` – backend discovery, file/model heuristics, and tokio-aware creation helpers
- `hub.rs` – Hugging Face Hub downloads via `hf_hub`
- `huggingface.rs` – wrapper over `tokenizers::Tokenizer`, chat template loading, vocab access
- `tiktoken.rs` – wrapper over `tiktoken-rs` encoders for OpenAI model families
- `chat_template.rs` – AST-driven Jinja template inspection and rendering utilities
- `sequence.rs` – stateful incremental decoding helper used by router sequences
- `stream.rs` – stateless streaming decoder that yields textual chunks from token streams
- `stop.rs` – stop-sequence detection with "jail" buffering and a builder API
- `mock.rs` – lightweight tokenizer used by unit tests
- `tests.rs` – smoke tests covering the trait facade and helpers (largely with the mock backend)

## Core Traits and Types (`traits.rs`)
- `Encoder`, `Decoder`, and `Tokenizer` traits stay `Send + Sync` so instances can be shared across
  threads.  Concrete backends implement the minimal methods: `encode`, `encode_batch`, `decode`,
  `vocab_size`, special-token lookup, and optional token↔id conversions.
- `Encoding` wraps backend-specific results: `Hf` holds the Hugging Face encoding object,
  `Sp` is a plain ID vector reserved for future SentencePiece support, and `Tiktoken` stores u32 IDs
  from `tiktoken-rs`.  `Encoding::token_ids()` is the zero-copy accessor used everywhere.
- `SpecialTokens` collects optional BOS/EOS/etc. markers so upstream code can make backend-agnostic
  decisions.
- `Tokenizer` (in `mod.rs`) is a thin `Arc<dyn Tokenizer>` newtype that exposes convenience methods
  (`encode`, `decode`, `decode_stream`, etc.) while keeping cloning cheap.

## Backend Implementations
### HuggingFaceTokenizer (`huggingface.rs`)
- Loads `tokenizer.json` (or similar) using `tokenizers::Tokenizer::from_file`.
- Caches vocab forward and reverse maps for `token_to_id`/`id_to_token` support.
- Extracts special tokens using common patterns (e.g. `<s>`, `[CLS]`).
- Supports optional chat templates: either auto-discovered next to the tokenizer via
  `tokenizer_config.json` or overridable with an explicit template path.
- Exposes `apply_chat_template` which renders a minijinja template given JSON message payloads and
  template parameters.

### TiktokenTokenizer (`tiktoken.rs`)
- Wraps the `tiktoken-rs` `CoreBPE` builders (`cl100k_base`, `p50k_base`, `p50k_edit`, `r50k_base`).
- `from_model_name` heuristically maps OpenAI model IDs (e.g. `gpt-4`, `text-davinci-003`) to those
  bases. Unknown model names return an error rather than silently defaulting.
- Implements encode/decode operations; batch encode simply iterates sequentially.
- Provides approximate vocab sizes and common GPT special tokens.  Direct token↔id lookup is not
  implemented—the underlying library does not expose that mapping.

### MockTokenizer (`mock.rs`)
- Purely for tests; hard-codes a tiny vocabulary and simple whitespace tokenization.
- Implements the same trait surface so helpers can be exercised without pulling real tokenizer data.

## Factory and Backend Discovery (`factory.rs`)
- `create_tokenizer{,_async}` accept either a filesystem path or a model identifier.  Logic:
   1. Paths are loaded directly; the file extension (or JSON autodetection) selects the backend.
   2. Strings that look like OpenAI model names (`gpt-*`, `davinci`, `curie`, `babbage`, `ada`) use
      `TiktokenTokenizer`.
   3. Everything else attempts a Hugging Face Hub download via `download_tokenizer_from_hf`.
- Chat templates can be injected with `create_tokenizer_with_chat_template`.
- Async creation uses `tokio` for network access. The blocking variant reuses or spins up a runtime
  when called from synchronous contexts.
- SentencePiece (`.model`) and GGUF files are detected but currently return a clear `not supported`
  error.

## Hugging Face Hub Integration (`hub.rs`)
- Uses the async `hf_hub` API to list and download tokenizer-related files
  (`tokenizer.json`, `merges.txt`, `.model`, etc.), filtering out weights and docs.
- The helper returns the HF cache directory containing the fetched files; the factory then loads
  from disk using standard file paths.
- Honour the `HF_TOKEN` environment variable for private or rate-limited models.  Without it the
  download may fail with an authorization error.

## Chat Template Support (`chat_template.rs`)
- Detects whether a template expects raw string content or the structured OpenAI-style `content`
  list by walking the minijinja AST.  This matches the Python-side detection logic used elsewhere in
  SGLang.
- `ChatTemplateProcessor` (constructed per call) renders templates against JSON `messages` and
  `ChatTemplateParams` (system prompt, tools, EOS token handling, etc.).  Errors surface as
  `anyhow::Error`, keeping parity with Hugging Face error messages.
- The tokenizer wrapper stores both the template string and its detected content format so callers
  can pre-transform message content correctly.

## Streaming and Stateful Helpers
### `DecodeStream` (`stream.rs`)
- Maintains a sliding window (`prefix_offset`, `read_offset`) over accumulated token IDs.
- Each `step` decodes the known prefix and the new slice; when the new slice produces additional
  UTF-8 text (and does not end in the replacement character `�`), it returns the incremental chunk
  and updates offsets.  Otherwise it returns `None` and waits for more tokens.
- `step_batch` and `flush` offer convenience for batching and draining remaining text.

### `Sequence` (`sequence.rs`)
- Holds per-request decoding state: accumulated IDs plus offsets mirroring `DecodeStream`.
- `append_text` encodes extra prompt text; `append_token` decodes incremental output while
  respecting UTF-8 boundaries and replacing stray `�` characters.
- Designed for integration with router sequence management where decoded text must be replayed.

### `StopSequenceDecoder` (`stop.rs`)
- Extends the incremental decoding approach with a "jail" buffer that holds potential partial
  matches against configured stop sequences.
- Supports both token-level stops (visible or hidden) and arbitrary string sequences.  When a string
  stop is configured, the decoder emits only the safe prefix and keeps a suffix jailed until it can
  decide whether it completes a stop sequence.
- Provides `StopSequenceDecoderBuilder` for ergonomic configuration and exposes `process_token`,
  `process_tokens`, `flush`, `reset`, and `is_stopped` helpers.

## Testing
- Unit tests cover the mock tokenizer, the `Tokenizer` wrapper, incremental decoding helpers, and
  stop-sequence behaviour (`tests.rs`, `sequence.rs`, `stop.rs`, `tiktoken.rs`, `factory.rs`,
  `hub.rs`).  Network-dependent Hugging Face downloads are exercised behind a best-effort async test
  that skips in CI without credentials.
- Use `cargo test -p sgl-model-gateway tokenizer` to run the module’s test suite.

## Known Limitations & Future Work
- SentencePiece (`.model`) and GGUF tokenizers are detected but deliberately unimplemented.
- `Encoding::Sp` exists for future SentencePiece support but currently behaves as a simple `Vec<u32>`.
- `TiktokenTokenizer` cannot map individual tokens/IDs; the underlying library would need to expose
  its vocabulary to implement `token_to_id`/`id_to_token`.
- There is no metrics or batching layer inside this module; the router records metrics elsewhere.
- Dynamic batching / sequence pooling code that earlier READMEs mentioned never landed in Rust.

## Usage Examples
```rust
use std::sync::Arc;
use sgl_model_gateway::tokenizer::{
    create_tokenizer, SequenceDecoderOutput, StopSequenceDecoderBuilder, Tokenizer,
};

// Load a tokenizer from disk (Hugging Face JSON)
let tokenizer = Tokenizer::from_file("/path/to/tokenizer.json")?;
let encoding = tokenizer.encode("Hello, world!")?;
assert!(!encoding.token_ids().is_empty());

// Auto-detect OpenAI GPT tokenizer
let openai = create_tokenizer("gpt-4")?;
let text = openai.decode(&[1, 2, 3], true)?;

// Incremental decoding with stop sequences
let mut stream = tokenizer.decode_stream(&[], true);
let mut stop = StopSequenceDecoderBuilder::new(Arc::clone(&tokenizer))
    .stop_sequence("\nHuman:")
    .build();
for &token in encoding.token_ids() {
    if let Some(chunk) = stream.step(token)? {
        match stop.process_token(token)? {
            SequenceDecoderOutput::Text(t) => println!("{}", t),
            SequenceDecoderOutput::StoppedWithText(t) => {
                println!("{}", t);
                break;
            }
            SequenceDecoderOutput::Held | SequenceDecoderOutput::Stopped => {}
        }
    }
}
```

```rust
// Apply a chat template when one is bundled with the tokenizer
use sgl_model_gateway::tokenizer::{chat_template::ChatTemplateParams, HuggingFaceTokenizer};

let mut hf = HuggingFaceTokenizer::from_file_with_chat_template(
    "./tokenizer.json",
    Some("./chat_template.jinja"),
)?;
let messages = vec![
    serde_json::json!({"role": "system", "content": "You are concise."}),
    serde_json::json!({"role": "user", "content": "Summarise Rust traits."}),
];
let prompt = hf.apply_chat_template(
    &messages,
    ChatTemplateParams {
        add_generation_prompt: true,
        continue_final_message: false,
        tools: None,
        documents: None,
        template_kwargs: None,
    },
)?;
```

Set `HF_TOKEN` in the environment if you need to download private models from the Hugging Face Hub.
