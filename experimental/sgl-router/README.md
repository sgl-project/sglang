# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

Serves a single model and routes across its workers. Exposes
`/v1/tokenize`, `/v1/detokenize`, `/v1/models`, `/v1/chat/completions`
(buffered and SSE), plus `/healthz` / `/readyz` and `/metrics`. Worker
pools come from either a static URL list or Kubernetes EndpointSlice
discovery. Both edges speak cleartext HTTP/2 where the peer does — see
[HTTP/2](#http2).

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

## HTTP/2

There is nothing to configure. The router negotiates per connection inbound and
resolves the protocol per worker outbound; every combination below is reached
automatically, and HTTP/1.1 remains a supported peer on both edges.

**Inbound.** The listener accepts cleartext HTTP/2 (h2c, prior knowledge) and
HTTP/1.1 on the same `--port`, chosen per connection. A mesh sidecar or
load balancer that prefers h2c multiplexes over one connection instead of
opening one per request; an HTTP/1.1 client is unaffected.

**Outbound.** At registration the router reads each worker's `/model_info` and
forwards over cleartext h2c only when that worker reports `--enable-http2` (the
engine's Granian server, which serves h2c and HTTP/1.1 together) **and** the
worker URL is cleartext. Everything else uses the default client, which speaks
HTTP/1.1 in cleartext and negotiates ALPN `h2, http/1.1` over TLS:

| `/model_info` | worker URL | router forwards over |
|---|---|---|
| `enable_http2: true` | `http://` | cleartext h2c |
| `enable_http2: true` | `https://` | HTTP/2 over TLS, by ALPN |
| `enable_http2: false`, or absent | any | HTTP/1.1 (cleartext) or ALPN (TLS) |

The choice is per worker, not fleet-wide, so a mixed fleet works and one
worker's state never changes another's. `/model_info` is the source — rather
than `/server_info`, which also reports the flag — because it answers from
manager-owned state while `/server_info` awaits a scheduler round-trip: a
warming engine can serve its identity long before its launch record, and a
protocol resolved late is one the router never applies. Reading it alongside
the model name means the two resolve together, so a worker that joined a model
pool carries the protocol from that same fetch. Admin fan-out
(`/flush_cache`) always uses the default client, because it addresses every
worker at once rather than a selected one.

Two things worth knowing when debugging. h2c is prior-knowledge only — there is
no negotiation and no fallback — which is why the router requires the engine's
own `enable_http2` report before using it. And a worker's protocol is fixed
for as long as that worker stays registered: it is read once, at registration,
and never re-read. Under the K8s backend a restarting engine flips its
EndpointSlice to `ready=false`, which is a `Removed` → `Added` cycle and so a
fresh reading; under `--worker-urls` the fan-out happens once at startup and
nothing re-registers. So a worker registered over h2c that later stops serving
it (a proxy interposed on its port, say) is not detected until it
is re-registered; its circuit breaker will open in the meantime.

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
