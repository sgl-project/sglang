# SGLang gRPC API

Protocol definitions for the in-process gRPC server exposed by SGLang.

- Native RPCs provide generation, embeddings, classification, tokenization, model information, and server control.
- OpenAI-compatible RPCs provide chat, completion, embeddings, classification, scoring, and reranking through JSON payloads.

The `nightly` label is updated daily from SGLang's `main` branch. The `main` label tracks the latest SGLang release, and version labels match tags such as `v0.5.15`. Pin a Buf commit or generated SDK version for reproducible builds.
