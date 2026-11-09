# sglang-server

`sglang-server` is SGLang's Rust HTTP frontend and request-processing pipeline. It exchanges typed requests and responses with the Python scheduler while keeping latency-sensitive work outside Python.

## Returning token IDs without decoding text

Native HTTP `/generate` and gRPC `Generate` requests can set
`skip_detokenize: true` to bypass output text decoding for that request:

```json
{"text": "Hello", "sampling_params": {"max_new_tokens": 16}, "skip_detokenize": true}
```

The response keeps `output_ids`, token counts, finish reasons and numeric
logprobs. Its text is empty, and logprob token text remains absent even with
`return_text_in_logprobs: true`. Input text is still tokenized normally; other
requests served by the same process continue to decode their output. The flag
defaults to false and applies to every prompt in a batch, for both streaming
and non-streaming responses. This does not change scheduler stop handling.

## Code review principles

1. **Use strongly typed boundaries.** Model every supported protocol shape with structs, enums, and validated newtypes; avoid opaque values such as `serde_json::Value` and `rmpv::Value` in production paths.
2. **Keep one canonical schema.** Rust and Python must derive their wire contracts from one source of truth, aligned with `io_struct.py`, instead of independently duplicating field names, order, defaults, or validation.
3. **Make protocol declarations minimal and declarative.** A reviewer should be able to understand the wire format from its type declarations alone, without tracing fillers, conversion code, macros, or repeated field lists.
4. **Use one representation per semantic stage.** Separate external input, normalized domain data, and wire data, and convert between them once at explicit boundaries; do not keep multiple overlapping representations of the same state.
5. **Design compatibility and safety explicitly.** Version protocols, reject unsupported or malformed inputs clearly, validate lengths and resource bounds before allocation, preserve invariants in types, and test compatibility across the real Rust and Python codecs.
