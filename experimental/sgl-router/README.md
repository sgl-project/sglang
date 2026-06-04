# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

**Status:** functional single-worker HTTP proxy. Exposes `/v1/tokenize`,
`/v1/detokenize`, `/v1/models`, `/v1/chat/completions` (buffered and SSE),
plus `/healthz` / `/readyz`. Forwards to one configured worker via reqwest;
parity-tested against `transformers.AutoTokenizer`. Multi-worker routing,
service discovery, and observability still pending.

## Building

```bash
cd experimental/sgl-router
cargo build --release
```

## License

Apache-2.0.
