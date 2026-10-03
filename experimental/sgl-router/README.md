# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

Serves a single model and routes across its workers. Exposes
`/v1/tokenize`, `/v1/detokenize`, `/v1/models`, [`/v1/embeddings`](#embeddings),
[`/v1/classify`](#classify), [`/v1/rerank`](#rerank), `/v1/chat/completions` and
SGLang's native [`/generate`](#native-generate) (buffered and SSE), plus
`/healthz` / `/readyz` and `/metrics`. Worker pools come from either a static URL
list or Kubernetes EndpointSlice discovery. Both edges speak cleartext HTTP/2
where the peer does — see [HTTP/2](#http2).

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

To run two engine versions side by side in PD mode (e.g. during a rollout),
pass `--pd-version-group-label <key>`. A prefill worker is then paired only with
decode workers that have the same value for that EndpointSlice label (inherited
from the Service, so use one Service per role and version), so KV never crosses
versions. Unlabeled workers form their own group.

```bash
sgl-router --model-id qwen3 --service-discovery \
  --prefill-selector app=engines-qwen3,role=prefill \
  --decode-selector app=engines-qwen3,role=decode \
  --pd-version-group-label sglang.ai/version-group
```

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

### Peer bootstrap (Kubernetes)

A replica that starts mid-fleet subscribes to each worker's KV topic
mid-stream, so everything already resident in the engines' caches is invisible
to it and it routes cache-blind until traffic re-stores those blocks —
degrading the engines' locality for the whole fleet, once per replica on every
rolling update. With a peer selector set, a booting replica instead pulls a
tree snapshot from a warm sibling over `/internal/kv_snapshot` and splices it
under its own live delta stream, and `/readyz` stays 503 until that bootstrap
settles. Off unless `--kv-peer-selector` is set; requires `--policy
cache_aware` with the Router-local radix tree (not an external Indexer) and
`--service-discovery`, which the CLI enforces.

```bash
sgl-router \
  --model-id qwen3 --policy cache_aware \
  --service-discovery --service-discovery-namespace prod \
  --selector app=engines-qwen3 \
  --kv-peer-selector kubernetes.io/service-name=sgl-router
```

- `--kv-peer-selector` is matched against **EndpointSlice labels** — the
  router Service's labels plus `kubernetes.io/service-name`, never the pods'
  labels, so `kubernetes.io/service-name=<router-service>` is the unambiguous
  choice. A pod-template label matches nothing and every replica boots cold.
- `--kv-bootstrap-timeout-ms` (default 600000) bounds the whole bootstrap;
  readiness waits on it, so startup/readiness probes must tolerate a replica
  staying unready this long.
- `--kv-bootstrap-fetch-timeout-cap-ms` (default 300000) caps one snapshot
  fetch within that budget.
- `--kv-bootstrap-seed-required` (off by default) keeps holding `/readyz` when
  siblings were present but their tree could not be pulled, bounded at
  max(3× the bootstrap timeout, 60s) — it delays a failed seed's replica (and
  with it a rolling update) rather than shipping it cache-blind; it cannot
  stall a rollout indefinitely.

The deployment needs two things beyond the flags: the router's ServiceAccount
must hold `get`/`list`/`watch` on `endpointslices` in the watched namespace
(the worker-discovery Role already grants this), and the pod spec should wire
`POD_NAME`, `POD_NAMESPACE` and `POD_IP` from the downward API so a replica
can exclude itself from its own peer list. A rolling update should surge
(`maxUnavailable: 0`) so new replicas always have warm siblings to copy from.
`tests/e2e/k8s_integration/manifests/kv-bootstrap.yaml` is a worked example of
all of it, and the `sgl_router_kv_bootstrap_*` series in
[monitoring/README.md](monitoring/README.md) make the whole path observable.

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

Both reorg affinity policies accept `--affinity-mode prefer` (default) or
`balanced`. Prefer keeps an admissible session binding or the best admissible
prefix owner. Balanced samples a power-of-two alternative and switches only
when the affinity engine's waiting uncached tokens exceed both
`alternative * --affinity-load-factor` (default 2) and
`alternative + --affinity-load-gap` (default 1024). Missing fresh native load
preserves admissible affinity; ties also preserve it.

Both modes fall back within the group when affinity fails admission, excluding
rejected engines. The fallback winner must pass admission; failure advances to
the next bucket. Session replacements are bound after admission during selection,
not after dispatch. Reorg rejects legacy pressure guards, cache switch margins,
and queue/saturation gates in favor of these shared affinity settings.

### Optional tokenizer for load-only routing

`--no-tokenizer` skips tokenizer loading for load-only policies such as
`power_of_two` and `session_aware`, on either routing path. Workers tokenize the
original messages, and `/v1/tokenize` and `/v1/detokenize` are unavailable.
Cache-aware routing, prefix-cache terms or filters, and `--bucket-config` still
require a tokenizer.

### DP-rank routing

An engine launched with `--dp-size` or `--attn-dp-size` runs several DP ranks,
each with its own KV cache, behind one endpoint. With `--dp-aware`, the router
also picks the rank inside the selected worker. It sends that rank as
`X-Data-Parallel-Rank`, which the engine honors, and overwrites any value the
client sent. The router picks the first of these that applies:

1. A hash of the sticky routing key or session id, so a conversation keeps
   its rank and router replicas agree.
2. The rank with the deepest cached prefix in the local KV tree.
3. The rank with the fewest requests this router has in flight on it.

In PD mode, decode is ranked by load only. The bootstrap room satisfies
`room % prefill_dp_size == prefill_rank`, which is how a decode engine finds
the prefill rank.

### Engines with `--api-key`

The router reads each worker's `/server_info` and `/model_info` to learn its
model, KV-event publisher, HTTP/2 support and DP size. An engine launched with
`--api-key` rejects those requests without the key, so pass the same key as
`--worker-api-key`. Chat requests and `/flush_cache` forward the caller's
`Authorization` header instead, so callers still need the engine key.

### Fleet-wide sampling contract

`--override-sampling-params` fixes the sampling configuration for every client
of this router, independently of what the engine's own defaults happen to be
(on native [`/generate`](#native-generate), in each prompt's `sampling_params`):

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

Some chats additionally forward the rendered tokens to the engine as
`input_ids`, retaining the original messages, so the engine skips
re-tokenizing. How many depends on the model's renderer. DeepSeek-V4's native
encoder is fixture-verified against SGLang's request normalization, so it
forwards every chat except multimodal ones and those with caller-provided
`input_ids`. Renderers without that verification (HF Jinja templates, Kimi-K3)
forward only plain text chat requests (string `content`, no tools, no template
kwargs or reasoning controls or historical `reasoning_content`, no assistant
continuation, no consecutive users or non-leading system turns) and warn
`UNVERIFIED` at startup; every other request shape is rendered for routing
only. DeepSeek-V4.1 forwards nothing — its renderer is not verified against
current SGLang — while routing tokenization keeps working.

Use matching model files on the router and workers, and set
the same `--default-chat-template-kwargs`, `SGLANG_DEFAULT_THINKING`,
`SGLANG_DSV4_REASONING_EFFORT`, and `SGLANG_DSV41_REASONING_EFFORT` on both.
The router reads these render defaults from its own configuration and environment;
it does not discover the workers' settings. Point `--tokenizer-path` at the workers'
model snapshot so the V4 effort profile is read from the same
`encoding/encoding_dsv4.py`.

Set `--disable-input-ids-forwarding` for this router's model when worker-side
rendering has not been verified to match. This disables router-generated IDs
for every routing policy; cache-aware routing still renders and tokenizes
locally, and the original messages reach the workers for engine processing.
Caller-supplied `input_ids` remain caller-owned and pass through unchanged.

Forwarding logs its assumptions at startup. In particular, disable it when the
router's render defaults differ from the workers', for worker template overrides
not reflected in the router's model files, for worker parser overrides such as
`--tool-call-parser deepseekv32` that select a native encoder over a shipped
template, or conversation templates with stop strings (the engine's `input_ids`
path skips those template stops). Disabling forwarding preserves engine behavior
but does not establish parity for local routing hashes.

Also set `--disable-input-ids-forwarding` for array-only templates: Dynamo may wrap
string content into arrays differently from the worker. The pinned Dynamo renderer does not expose
its conversion flag, so the router cannot automatically block these templates.
Detailed content-format parity coverage follows in #39133.

The Dynamo crates are pinned exactly and `Cargo.lock` is committed; CI builds
with `--locked`, so rendered bytes cannot change without a reviewed diff.

Router tokenization sits on the TTFT path for every chat it renders. Two opt-in
flags make it cheaper:

- `--tokenizer-backend fast` encodes with fastokens (decoding stays on HF). It
  needs a `tokenizer.json` and falls back to `hf` when fastokens cannot load it.
- `--tokenizer-l1-cache-mb N` caches prefix tokenizations at special-token
  boundaries, so a multi-turn chat encodes only the turns added since the
  previous request. Boundaries are unconditional, non-normalized, non-stripping
  special tokens with no overlapping added-token spellings. Unsafe candidates
  are excluded; if none remain, encoding proceeds without the cache.

On a ~69K-token DeepSeek-V4 chat, `hf` encodes in ~40 ms, `fast` in ~4 ms, and
a new turn on a cached history in ~0.2 ms. The DeepSeek fixtures check every
case under `hf`, `fast`, and `fast` with L1. Startup logs report the resolved
backend and cache state. `/metrics` exposes only
`sgl_router_tokenizer_l1_tokens_total{source="cached"|"encoded"}` to measure
how much tokenization work the cache reuses.

## Native `/generate`

`/generate` (`POST` or `PUT`) has the engine's interface: the same
`GenerateReqInput` body and the same response, buffered or SSE. The body names no
model, so requests go to the one this router serves. Worker selection
(`--chat-routing`, `--policy`), PD dispatch, and abort-on-disconnect are shared
with chat completions.

With a tokenizer loaded, the router tokenizes `text` (a string or a list) with
the special tokens SGLang adds (BOS per `add_bos_token` for Llama-, Gemma- and
Cohere-class tokenizers, otherwise the `tokenizer.json` post-processor's), and
forwards it as `input_ids`. The engine skips tokenizing and routing sees its exact
tokens. Multimodal requests keep `text`, since the engine expands placeholders
from it, and `--disable-input-ids-forwarding` keeps it for every request. So does
a model whose `tokenizer.json` normalizer transformers replaces on load (legacy
SentencePiece Llama files, bge-m3), or whose `tokenizer_config.json` the router
cannot read, as the router cannot reproduce its tokens. A batch goes to one
worker: load counts every prompt and each of its `n` samples, while bucket and
context limits bound the longest prompt plus its own `max_new_tokens`.

Otherwise the body passes through, plus PD bootstrap fields, a minted `rid` for
a single prompt that has none, and `--override-sampling-params` defaults. Under
`--dp-aware` each worker's body also carries the chosen `routed_dp_rank`; a PD
batch or `n > 1` request leaves the prefill rank to the engine, which gives item
`i` the bootstrap room `room + i`.

## Embeddings

`/v1/embeddings` has the engine's interface: the same OpenAI `EmbeddingRequest`
body and response. As for chat completions, `model` must name the served model.
A PD fleet answers 400, since prefill and decode engines serve no embeddings.

Text `input` (a string or a list) is tokenized and forwarded as token IDs, as for
[`/generate`](#native-generate), plus the EOS SGLang appends for EmbeddingGemma.
Blank prompts and multimodal items stay as sent, for the engine to reject or
render. A list is a batch for one worker, routed on load. A single prompt with no
`rid` gets a minted one. Under `--dp-aware` the engine picks the rank, since its
embeddings endpoint reads none.

## Classify

`/v1/classify` has the engine's interface: the same `ClassifyRequest` body and
response. It is served as embeddings are, except that a batch of text stays text,
since the engine takes token IDs for only one prompt.

## Rerank

`/v1/rerank` has the engine's interface: the same `V1RerankReqInput` body and
response, sent as `POST` or `PUT` to the model this router serves. The body is
forwarded as sent, since the engine renders and tokenizes each query-document
pair. Routing is on load, with each pair's size estimated from its text. As for
embeddings, a PD fleet answers 400 and `--dp-aware` pins no rank.

## DeepSeek V4

Native V4 rendering follows SGLang's serving path (`serving_chat.py`), not
Dynamo's OpenAI defaults: all declared tools are rendered with SGLang's schema
defaults, reasoning effort comes from `reasoning` / `reasoning_effort`, and the
official/preview effort profile is detected from the checkpoint's
`encoding/encoding_dsv4.py` or overridden by `dsv4_reasoning_effort_profile` in
`config.json`, as in SGLang. Reference prompts live in `tests/fixtures/deepseek/`
and are regenerated by `tests/scripts/generate_deepseek_parity.py`.

V4.1 Flash uses Dynamo's separate V4.1 encoder with SGLang's numeric reasoning
budgets, tool payloads, and `<｜System｜>` markers — for routing tokenization
only, since V4.1 never forwards `input_ids`. Developer messages and media are
left to the worker (the pinned encoder renders them differently), and a
non-default `SGLANG_DSV41_REASONING_EFFORT` still matters for cache-aware
routing-hash parity.

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
