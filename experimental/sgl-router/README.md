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

Load monitoring is disabled by default. When enabled, the Router derives each
Worker host from service discovery and dials the Worker's fixed Load Reporter
port over h2c. The Router sends the first `RegisterRequest`, maintains the
lease with `KeepAlive`, and consumes `LoadReport` frames on the same bidi
stream:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy round_robin \
  --load-monitor \
  --load-reporter-port 31000
```

`--load-reporter-port` is optional. When set, it is the fallback reporter
port for Workers whose `/server_info` does not advertise a
`load_reporter_port`; when unset, each Worker's reporter port is resolved
from its `/server_info` response. Workers with neither (e.g. engines built
without the load reporter) are simply not monitored.
The first version uses a fixed 1-second report interval, 3-second freshness
window, 15-second lease, and 2-second connection timeout. This change keeps
routing policies unchanged; the immutable snapshot is the read-only boundary
for follow-up scheduling integrations.

The monitor maintains an internal immutable, versioned snapshot with worker
freshness, source and sequence metadata, complete DP-rank values, and aggregate
load. This snapshot is intentionally not exposed as a public HTTP endpoint;
follow-up scheduling policies consume it inside the Router process.

The current transport is insecure h2c and does not provide TLS, mTLS, or gRPC
authentication. The report's `worker_addr` remains compatibility metadata;
the Router associates every report with the discovery-owned Worker whose
outbound task owns the stream.

## License

Apache-2.0.
