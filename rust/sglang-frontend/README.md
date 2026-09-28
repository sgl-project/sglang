# SGLang frontend

Reusable chat/completion request handling and HTTP adapters extracted from the
standalone renderer. Rendering, tokenization, and tool/reasoning parsing use
`sglang-renderer`, which builds on Dynamo's frontend crates.

A host constructs `OpenAIService` from a `RendererService` and a
`GenerationService`. The generation service accepts a `GenerateTransport` that
submits prepared token requests and returns token deltas. It owns incremental
decoding and stop handling; the transport owns engine submission and cancellation.
Dropping a submission future or response stream must release its engine work.

`http::inference_routes` supplies the chat/completion routes.
`http::renderer_routes` supplies the render and tokenize routes. Hosts own the
runtime, listener, middleware, health checks, and any proxy fallback.

`sglang-renderer-server` is the first consumer. This extraction does not change
`sglang-server` or add a local scheduler transport. Current API coverage and
standalone usage are documented in [the renderer README](../sglang-renderer/README.md).
