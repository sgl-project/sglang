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

### Reorg routing

Use `--chat-routing reorg` to select the new bucket engine. The existing `--policy`
and cache/session flags configure its policies; no separate file is required.

```bash
sgl-router --model-id qwen3 --worker-urls http://localhost:30001 \
  --chat-routing reorg --policy cache_aware
```

Reorg supports `power_of_two` (its default), `cache_aware`, and `session_aware`.
Discovery supplies the plain or PD workers; decode uses power-of-two. Cache
settings, external indexers, session headers/timeouts, and `--filter overloaded`
with `--max-in-flight` retain their existing flags. Unsupported legacy options
fail at startup. Legacy `--bucket-config` files cannot define complete reorg PD
buckets and are not accepted on this path.

Omitting `--chat-routing` keeps the existing policies and defaults.

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

`reject` also 400s a value it cannot read as a number, on a governed parameter
only. The engine coerces more than JSON numbers — a bool and a numeric string
both become numbers — by rules that are undocumented and need not match across
a fleet, so a value the router cannot read is one it cannot prove conforms, and
waving it through would make the pin bypassable. The common coercions are
matched exactly (`false` is 0, `"0.5"` and `"1_0"` are 0.5 and 10), so this
refuses only genuine garbage. `allow` is unaffected: it promises nothing, so
such a value keeps flowing to the engine, which owns the request schema.

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

## Chat rendering

The router renders chat requests with dynamo-render (`dynamo-renderer`): the model's
HF Jinja template from `tokenizer_config.json` or a sibling
`chat_template.jinja`, or dynamo-render's built-in DeepSeek encoder (V4 family, V3.2)
for template-less models. Cache-aware routing hashes the rendered tokens so its
prefix queries match the blocks the engine caches. Models the engine encodes in
code but dynamo-render cannot tokenize here (Inkling) route via raw prompt
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

Set `--disable-input-ids-forwarding` for this router's model when worker-side
rendering has not been verified to match. This disables router-generated IDs
for every routing policy; cache-aware routing still renders and tokenizes
locally, and the original messages reach the workers for engine processing.
Caller-supplied `input_ids` remain caller-owned and pass through unchanged.

Forwarding logs its assumptions at startup. In particular, disable it for
`SGLANG_DEFAULT_THINKING=true`, a non-default `SGLANG_DSV4_REASONING_EFFORT`,
worker parser overrides such as `--tool-call-parser deepseekv32` that select a
native encoder over a shipped template, or conversation templates with stop
strings (the engine's `input_ids` path skips those template stops). These worker
settings are not inferred from the router's environment. Disabling forwarding preserves
engine behavior but does not establish parity for local routing hashes.

Also set `--disable-input-ids-forwarding` for array-only templates: Dynamo may wrap
string content into arrays differently from the worker. The pinned Dynamo renderer does not expose
its conversion flag, so the router cannot automatically block these templates.
Detailed content-format parity coverage follows in #39133.

The Dynamo crates are pinned exactly and `Cargo.lock` is committed; CI builds
with `--locked`, so rendered bytes cannot change without a reviewed diff.

## DeepSeek V4

Native V4 rendering follows SGLang's serving path (`serving_chat.py`), not
Dynamo's OpenAI defaults: all declared tools are rendered with SGLang's schema
defaults, reasoning effort comes from `reasoning` / `reasoning_effort`, and the
official/preview effort profile is detected from the checkpoint's
`encoding/encoding_dsv4.py` or overridden by `dsv4_reasoning_effort_profile` in
`config.json`, as in SGLang. Reference prompts live in `tests/fixtures/deepseek/`
and are regenerated by `tests/scripts/generate_deepseek_parity.py`.

V4.1 Flash uses Dynamo's separate V4.1 encoder with SGLang's numeric reasoning
budgets, tool payloads, and `<｜System｜>` markers. Developer messages and media
are left to the worker (the pinned encoder renders them differently), and a
non-default `SGLANG_DSV41_REASONING_EFFORT` needs the forwarding precautions above.

## Kimi-K3

Kimi-K3 renders through dynamo-render's native XTML formatter with SGLang's
request semantics (reasoning controls, tools, `response_format`, continuations)
and the checkpoint's chunked tiktoken encoding. `--tokenizer-path` accepts a
local `tiktoken.model` or an HF repo id, whose `tiktoken.model`, `config.json`
and `tokenizer_config.json` are downloaded when it has no `tokenizer.json`.
An explicit null `thinking_effort` with thinking enabled is not representable
in the pinned formatter and falls back to engine-side rendering.

## HTTP/2

There is nothing to configure. The router negotiates per connection inbound and
resolves the protocol per worker outbound; every combination below is reached
automatically, and HTTP/1.1 remains a supported peer on both edges.

**Inbound.** The listener accepts cleartext HTTP/2 (h2c, prior knowledge) and
HTTP/1.1 on the same `--port`, chosen per connection. A mesh sidecar or
load balancer that prefers h2c multiplexes over one connection instead of
opening one per request; an HTTP/1.1 client is unaffected.

**Outbound.** At registration the router reads each worker's `/server_info` and
forwards over cleartext h2c only when that worker reports `--enable-http2` (the
engine's Granian server, which serves h2c and HTTP/1.1 together) **and** the
worker URL is cleartext. Everything else uses the default client, which speaks
HTTP/1.1 in cleartext and negotiates ALPN `h2, http/1.1` over TLS:

| `/server_info` | worker URL | router forwards over |
|---|---|---|
| `enable_http2: true` | `http://` | cleartext h2c |
| `enable_http2: true` | `https://` | HTTP/2 over TLS, by ALPN |
| `enable_http2: false`, or absent | any | HTTP/1.1 (cleartext) or ALPN (TLS) |

The choice is per worker, not fleet-wide, so a mixed fleet works and one
worker's state never changes another's. The flag comes from the `/server_info`
launch record, which has reported it all along, so no engine change is needed.
A worker whose `/server_info` does not answer has no readable protocol and
forwards over HTTP/1.1 for as long as it stays registered — a throughput cost,
never a correctness one. Admin fan-out (`/flush_cache`) always uses the default
client, because it addresses every worker at once rather than a selected one.

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
