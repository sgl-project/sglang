# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

Serves a single model and routes across its workers. Exposes
`/v1/tokenize`, `/v1/detokenize`, `/v1/models`, `/v1/chat/completions`
(buffered and SSE), plus `/healthz` / `/readyz` and `/metrics`. Worker pools
come from either a static URL list or Kubernetes EndpointSlice discovery.

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
for `--model-id` from HuggingFace (honoring `HF_TOKEN` / `HF_HOME`).

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

## Engine-reported load monitoring

Load monitoring is disabled by default. When enabled, the Router first binds
an independent gRPC listener, then asks every discovered worker to start or
renew reporting through `/v1/start_reporting`. Port `0` is supported and the
actual bound port is sent to the engine:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy round_robin \
  --load-monitor \
  --load-monitor-bind-host 0.0.0.0 \
  --load-monitor-bind-port 0 \
  --load-monitor-report-ip 10.0.0.10
```

`--load-monitor-report-ip` is required and must be reachable from the engine.
The first version uses a fixed 1-second report interval, 3-second freshness
window, 15-second lease, and 2-second registration timeout. This change keeps
routing policies unchanged; the immutable snapshot is the read-only boundary
for follow-up scheduling integrations.

The monitor maintains an internal immutable, versioned snapshot with worker
freshness, source and sequence metadata, complete DP-rank values, and aggregate
load. This snapshot is intentionally not exposed as a public HTTP endpoint;
follow-up scheduling policies consume it inside the Router process.

The Router intentionally sends no `Authorization` header to
`/v1/start_reporting`. It is therefore compatible with an unauthenticated
open-source or fake engine. Engine builds that enforce `ADMIN_FORCE` on this
endpoint currently reject registration with 401/403; authenticated reporting
is outside this Router-only change.

## License

Apache-2.0.
