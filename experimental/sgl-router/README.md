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

Prefill/decode disaggregation with the pull-mode Load Monitor:

```bash
sgl-router \
  --model-id qwen3 \
  --service-discovery --service-discovery-namespace prod \
  --prefill-selector app=engines-qwen3,role=prefill \
  --decode-selector app=engines-qwen3,role=decode \
  --policy power_of_two \
  --decode-policy load_based \
  --load-monitor \
  --load-monitor-interval-ms 1000 \
  --load-monitor-stale-after-ms 3000
```

`--load-monitor` makes the router poll every worker's `GET /v1/loads?include=core`
(immediately on registration, once per routed request coalesced to at most one
in-flight plus one pending pull per worker, and on the interval as a fallback).
Only workers whose latest report is younger than `--load-monitor-stale-after-ms`
are routable — a never-reported, stale, or unreachable worker is skipped and an
empty pool answers `503 no_fresh_worker_load`. `power_of_two` and `load_based`
then score by the engine-reported `running + waiting` request count plus the
requests this router dispatched since that report, instead of the router-local
in-flight counter. `--decode-policy` picks the decode peer with a real policy
over the decode pool; without it the same-host affinity heuristic is used.

In PD mode a prefill worker is only routable once `/server_info` has disclosed
its `disaggregation_bootstrap_port`. The prefill leg is raced against the decode
leg: a prefill that is unreachable or answers non-2xx fails the request fast
with `502 prefill_upstream_failed` / `502 prefill_upstream_rejected` and the
paired decode request is cancelled, instead of the client waiting for the
engine-side `bootstrap_room` timeout. `/readyz` reports 503 for a PD deployment
until it has at least one complete transfer group.

Transfer groups: the EndpointSlice label named by `--transfer-group-label`
(default `sglang.ai/transfer-group`) assigns each worker to a PD transfer group.
The prefill worker is chosen across groups, the decode peer only inside the
chosen prefill's group; a group missing either side contributes no candidates,
and when no group is complete the request fails with `503 no_compatible_pd_group`.
Unlabelled slices (and all static-URL workers) are ungrouped and pair with each
other.

External KV indexer as the cache-aware signal source:

```bash
sgl-router \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy cache_aware_zmq \
  --kv-indexer-endpoint http://10.0.0.10:50051 \
  --kv-indexer-query-timeout-ms 100 \
  --kv-indexer-query-max-inflight 32
```

The existing cache-aware policy and thresholds are reused. When configured, the
Indexer replaces the Router-local radix tree as the cache signal. A successful
query with no usable match selects by minimum active load; connection failures,
timeouts, local admission rejection, and server rejection fail the Router
request with `503` rather than silently switching signals. The timeout and local
concurrency bound default to 100ms and 32 respectively.

## License

Apache-2.0.
