# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

Serves a single model and routes across its workers. Exposes
`/v1/tokenize`, `/v1/detokenize`, `/v1/models`, `/v1/chat/completions`
(buffered and SSE), plus `/healthz` / `/readyz` and `/metrics`. Worker
pools come from either a static URL list or Kubernetes EndpointSlice
discovery.

## Building

```bash
cd experimental/sgl-router
cargo build --release
```

## Running

The router is configured entirely through CLI flags (run
`sgl-router --help` for the full list). It serves exactly one model, so
`--model-id` is required, along with exactly one discovery backend.
`--tokenizer-path` is optional: give it a local `tokenizer.json` path or a
HuggingFace repo id, and when omitted the router downloads the tokenizer
for `--model-id` from HuggingFace. Authentication uses `HF_TOKEN`, then the
file at `HF_TOKEN_PATH`, then `$HF_HOME/token`. Surrounding whitespace in
`HF_TOKEN` is trimmed; a whitespace-only, non-UTF-8, or malformed value reports
an error without logging the token. An unset or empty value allows token-file
fallback. `HF_HUB_DISABLE_IMPLICIT_TOKEN` disables ambient credentials.
`HF_ENDPOINT`, `HF_HOME`, and `HF_HUB_CACHE` configure the Hub and cache.
Downloads use cached files first; `HF_HUB_OFFLINE` is not interpreted.

Static worker list:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000
```

Kubernetes EndpointSlice discovery:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --service-discovery \
  --service-discovery-namespace prod \
  --selector app=engines-qwen3
```

Omit `--service-discovery-namespace` to watch all namespaces (requires
cluster-wide RBAC). For prefill/decode disaggregation, replace `--selector`
with `--prefill-selector` and `--decode-selector`.

External KV indexer as the cache-aware signal source:

```bash
sgl-router \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy cache_aware \
  --cache-prefix-provider indexer \
  --kv-indexer-endpoint http://10.0.0.10:50051 \
  --kv-indexer-query-timeout-ms 100 \
  --kv-indexer-query-max-inflight 32
```

The Indexer replaces the Router-local radix tree as the native Cache-Aware
signal. Query timeouts and local concurrency are bounded by the two Indexer
options, which default to 100 ms and 32 respectively.

## Chat rendering

The router renders chat requests with dynamo-render (`dynamo-renderer`): the model's
HF Jinja template from `tokenizer_config.json` or a sibling
`chat_template.jinja`, or dynamo-render's built-in DeepSeek encoder (V4 family, V3.2)
for template-less models. Cache-aware routing hashes the rendered tokens so its
prefix queries match the blocks the engine caches. Models the engine encodes in
code but dynamo-render cannot tokenize here (Inkling, Kimi K3) route via raw prompt
text, as does any model whose template fails to load or render.

Plain text chat requests (string `content`, no tools, no template kwargs or
reasoning controls or historical `reasoning_content`, no assistant continuation,
no consecutive users or non-leading system turns) additionally forward the
rendered tokens to the engine as `input_ids`, retaining the original messages,
so the engine skips re-tokenizing. Every other request shape is rendered for
routing only: the router renders with dynamo-render and does not replicate
SGLang's request normalization, so forwarding is enabled shape by shape as
parity is verified. Use matching model files on the router and workers; worker
template overrides and default kwargs are not observable from the request.

The Dynamo crates are pinned exactly and `Cargo.lock` is committed; CI builds
with `--locked`, so rendered bytes cannot change without a reviewed diff.

## Upgrading from `cache_aware_zmq`

The `cache_aware_zmq` policy has been removed. Configurations using it should
select `--policy cache_aware` and choose a native cache-prefix source: the
Router-local radix tree (the default), or the external Indexer shown above.

The legacy `--cache-threshold`, `--balance-abs-threshold`, and
`--balance-rel-threshold` flags have also been removed. They do not have
one-to-one replacements; remove them and review the current `sgl-router
--help` output when tuning Cache-Aware routing.

## License

Apache-2.0.
