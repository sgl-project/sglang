# Rust runtime.v1 support

The adapter exposes existing Rust frontend operations through the unchanged
`sglang.runtime.v1` contract. It does not implement the full Python frontend.
Runtime access and cancellation stay behind `FrontendHandle`.

| RPC | Existing Rust operation / boundary |
| --- | --- |
| TextGenerate | Native generation with a text prompt; requires a tokenizer. |
| Generate | Native generation with token IDs. |
| ChatComplete | Shared OpenAI chat preparation and response shaping; requires a tokenizer and chat template. |
| Complete | Shared OpenAI completion preparation and response shaping. Text prompts and token-ID echo require a tokenizer. |
| Detokenize | Existing detokenization operation (also used by completion echo); requires a tokenizer. |
| HealthCheck | Startup readiness and, when enabled, the same activity probe used by HTTP health checks. |
| GetIsReady | `UNIMPLEMENTED`: no pause-aware readiness operation in the Rust frontend. |
| GetModelInfo | Typed model metadata. |
| GetServerInfo | Typed, allowlisted server metadata and scheduler metrics; no private scheduler fields. |
| ListModels | The configured served model and context limit, as in HTTP model listing. |
| TextEmbed, Embed, Classify | `UNIMPLEMENTED`: no corresponding Rust inference operation. |
| Tokenize | `UNIMPLEMENTED`: Rust tokenizes internally but has no standalone frontend operation returning tokens and offsets. |
| GetLoad | `UNIMPLEMENTED`: no frontend operation for runtime.v1's load response. Server metadata is not a substitute. |
| Abort | `UNIMPLEMENTED`: no public abort-by-client-ID operation. Dropping an RPC still cancels the calls it owns. |
| FlushCache, PauseGeneration, ContinueGeneration | `UNIMPLEMENTED`: no corresponding Rust frontend control operation. |
| OpenAIEmbed, OpenAIClassify, Score, Rerank | `UNIMPLEMENTED`: no corresponding Rust OpenAI operation. |
| StartProfile, StopProfile, UpdateWeightsFromDisk | `UNIMPLEMENTED`: no corresponding Rust frontend control operation. |

## OpenAI framing and feature policy

- HTTP and gRPC share prompt preparation, sampling conversion, choice fan-out,
  tool/reasoning parsing, and OpenAI response shaping in `src/openai/`.
  HTTP retains JSON/SSE framing; gRPC does not call HTTP routes.
- Unary ChatComplete/Complete requests emit one JSON chunk with `finished=true`.
  Streaming requests emit JSON chunks followed by one empty chunk with
  `finished=true`, matching the existing runtime.v1 bridge. SSE prefixes and
  `[DONE]` do not appear on the gRPC wire.
- Unsupported request features return `UNIMPLEMENTED`, including trace headers,
  multimodal chat, unsupported OpenAI options, and tools without a configured
  parser. Invalid supported requests return `INVALID_ARGUMENT`.
  Existing HTTP validation behavior is unchanged.
- Client stream drop and response timeouts release the owned frontend calls.
  Unary/control operations and each streamed response wait are bounded.

The Tonic adapter and shared-operation approach build on Rain Jiang's
[multi-protocol prototype (#36923)](https://github.com/sgl-project/sglang/pull/36923).
This stack reuses the canonical runtime.v1 schema and existing Rust OpenAI code,
without adopting that prototype's new schema or proto-to-JSON migration.
