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

### Fleet-wide sampling contract

`--override-sampling-params` fixes the sampling configuration for every client
of this router, independently of what the engine's own defaults happen to be:

```bash
sgl-router \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 \
  --override-sampling-params '{"temperature": 1, "top_p": 0.95, "n": 1}' \
  --sampling-param-conflict reject
```

It takes one JSON object keyed by the request-body field names
(`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`,
`frequency_penalty`, `presence_penalty`, `n`). A configured value is injected
whenever the request omits that field, so the engine's defaults cannot drift
from what the operator declared. `temperature`, `top_p`, `top_k`, `min_p` and
`repetition_penalty` are the five the engine resolves from the model's own
`generation_config`, which is what makes them drift when a deployed image
changes; the rest have fixed API defaults.

An explicit `null` counts as omitting the field, not as a client-supplied
value: the OpenAI API types these parameters as nullable with a documented
default, so `null` asks for the default — and on a governed fleet the
configured value is what the default is.

For a request that does send a value, `--sampling-param-conflict` decides:
`reject` (the default) 400s a differing value before admission, while `allow`
forwards the client's value untouched — the router never silently rewrites what
a client sent. A `reject` response carries
`x-router-error-code: sampling_contract_violation` and is counted in
`sgl_router_sampling_contract_rejections_total{param}`, so a rollout's blast
radius is visible per parameter rather than folded into `bad_request`.

A value may also be an inclusive band, `{"min": LO, "max": HI}`, for a
parameter that stays tunable inside a range. A band names no value to inject,
so it constrains only the requests that name the parameter; one that omits it
gets the model's own `generation_config` default, which the router cannot see.
Because a band can only ever reject, combining one with `allow` is a startup
error.

Values are range-checked at startup, so a misconfiguration fails the launch
instead of 400ing every request at the engine. Repeating a key in the flag is
also a startup error, rather than silently enforcing whichever copy came last.

| parameter | accepted | notes |
| --- | --- | --- |
| `temperature` | `[0, 2]` | |
| `top_p` | `(0, 1]` | |
| `top_k` | `>= 1`, or exactly `-1` | `-1` is the engine's "whole vocabulary" spelling and its default. Being non-contiguous it cannot bound a band. Note `top_k: 1` is greedy decoding, not "disabled". |
| `min_p` | `[0, 1]` | not an OpenAI parameter; the engine's domain |
| `repetition_penalty` | `(0, 2]` | not an OpenAI parameter; the engine's domain |
| `frequency_penalty` | `[-2, 2]` | |
| `presence_penalty` | `[-2, 2]` | |
| `n` | `[1, 128]` | |

The OpenAI domains are deliberately narrower than what the engine itself
accepts (`SamplingParams.verify` would take `temperature: 5`): these values are
injected into request bodies, and a fleet contract outside the range every
OpenAI client library validates against is far more likely a typo than an
intent.

Cost: a request that named every configured field forwards its original bytes
untouched. One that omits a field has the scalars spliced directly into the
body bytes — no parse, no re-serialize — which on a 16 MiB body is ~0.24 ms
against ~5.7 ms for a `serde_json` round-trip. Only `input_ids` and PD
bootstrap injection still parse and re-serialize, because those may have to
overwrite a key the client sent.

### Relationship to the engine's own `--preferred-sampling-params`

The engine has an inject-when-absent flag of its own,
`--preferred-sampling-params`, merged in
`python/sglang/srt/managers/tokenizer_manager.py` as
`{**preferred, **obj.sampling_params}`. On `/v1/chat/completions` it is
currently a no-op: `ChatCompletionRequest.to_sampling_params`
(`python/sglang/srt/entrypoints/openai/protocol.py`) resolves every sampling
key eagerly through `generation_config` and then its own defaults, so the
right-hand side of that merge is always fully populated and always wins.
(`/v1/responses`, in the same file, already omits `None` entries for exactly
this reason.) If that is fixed engine-side, `--preferred-sampling-params`
covers the inject-when-absent half for a single engine.

What stays the router's to own either way is the enforcement half: the
`reject` immutability contract with its 400 before admission — costing no
queue slot and no engine round-trip — the `sampling_contract_violation` code
and per-parameter counter, bands, and one contract applied at a shared ingress
across engines whose own flags the router operator may not control.

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
