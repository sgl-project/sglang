# sgl-router reference

Detailed behavior of the features summarized in [README.md](README.md). Routing
design, policies and bucket configuration are in [ARCHITECTURE.md](ARCHITECTURE.md).

- [Discovery and KV signals](#discovery-and-kv-signals)
- [Bucket file membership](#bucket-file-membership)
- [Routing and reliability](#routing-and-reliability)
- [Chat rendering](#chat-rendering)
- [Native `/generate`](#native-generate)
- [OpenAI over /generate](#openai-over-generate)
- [Embeddings](#embeddings), [Classify](#classify), [Completions](#completions), [Rerank](#rerank)
- [Anthropic Messages](#anthropic-messages)
- Per-model rendering: [DeepSeek V4 and V4.1](#deepseek-v4-and-v41), [GLM-5.3](#glm-53), [Qwen3.8](#qwen38), [Kimi-K3](#kimi-k3)
- [HTTP/2](#http2)
- [Native gRPC](#native-grpc)
- [Upgrading from `cache_aware_zmq`](#upgrading-from-cache_aware_zmq)

## Discovery and KV signals

### PD version groups

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

### External KV indexer

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

### LoRA cache identity

LoRA routing requires workers that publish `BlockStored.lora_name`. Requests
select the adapter through `model: "base:adapter"` on OpenAI routes, or
`lora_path` on native routes. Salted generation requests also carry `cache_salt`.

Worker block IDs are opaque and may differ across adapter loads or workers.
The router derives adapter-aware lookup keys from event tokens and keeps a
per-worker, per-DP-rank mapping for parent links and removals. Multiple block
IDs can own the same lookup key; removing one leaves the other owners and
storage tiers intact. The remote indexer bridge performs the same translation.
Base-model events retain their existing published-key path.

New replicas request `?include_lora=true` snapshots containing LoRA mappings
(version 2). Legacy consumers receive a base-only version-1 view, so mixed
snapshots do not prevent base-cache bootstrap during rolling upgrades. Base-only
snapshots retain version 1 for every consumer. LoRA stores buffered during bootstrap retain their
tokens; the original base-model buffering and replay behavior is unchanged.

Reloading different adapter weights under the same public name can still
credit stale engine blocks until their eviction. Names identify adapters;
versioned names are needed to distinguish different weights.

### KV-event gap replay

ZMQ drops KV events at the publisher's high-water mark, and a lost removal
leaves the router crediting a worker with blocks it no longer holds. Give each
engine a replay socket and the router re-fetches the batches a sequence gap
skipped:

```bash
python -m sglang.launch_server ... \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}'
```

Rank r publishes on 5557 + r and replays on 5558 + r, so with `dp_size > 1`
place the replay base at least dp_size ports past the PUB base. The engine keeps
the last `buffer_steps` batches (default 10000); older gaps stay unrepaired.
`sgl_router_kv_event_replays_total{outcome}` counts the attempts.

### Pending prefixes

Until the engine's KV events arrive, requests that share a cold prefix see no
cached owner and spread across workers, each prefilling the same prefix. This
is common with parallel sampling, RL rollouts and agent fan-out.
`--cache-pending-prefix-ttl-ms` credits a worker with a prompt's prefix for that
long after routing it there, so the burst stays together. It is off by default
and needs `--policy cache_aware` with the Router-local radix tree.
`sgl_router_cache_pending_prefix_hits_total` counts lookups where a pending
prefix matched deeper than any confirmed one; the policy's candidate and
admission filters still decide whether that worker is picked.

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

## Bucket file membership

`worker_services` matches Kubernetes Services by `namespace/name`, using the
`kubernetes.io/service-name` label on watched EndpointSlices. Replacement pods
and new replicas join the same group automatically. The router's discovery
selectors must include those EndpointSlices; this field does not expand the watch.
A worker selected by several Services belongs to each of them.

For static URL discovery, use `"worker_ids": ["http://worker:30000"]`: each ID
is the configured worker URL. Kubernetes worker IDs are `namespace/pod-UID`
(and change when a pod is replaced), so use `worker_services` for durable pools.
Set only one membership field, or omit both for every engine of the group's role.
Empty membership lists and blank worker IDs are rejected at startup.

Admission fields left unset or set to `null` inherit CLI defaults. To apply a
limit only to selected groups, omit that CLI default and set it on those groups.

## Routing and reliability

### Optional tokenizer for load-only routing

`--no-tokenizer` skips tokenizer loading for load-only policies such as
`power_of_two` and `session_aware`, on either routing path. Workers tokenize the
original messages, and `/v1/tokenize` and `/v1/detokenize` are unavailable.
Cache-aware routing, prefix-cache terms or filters, and buckets with token-length
or context limits require a tokenizer. Reorg buckets that only select membership
and load-based policies can use `--no-tokenizer`.

### DP-rank routing

An engine launched with `--dp-size` or `--attn-dp-size` runs several DP ranks,
each with its own KV cache, behind one endpoint. With `--dp-aware`, the router
also picks the rank inside the selected worker. It sends that rank as
`X-Data-Parallel-Rank`, which the engine honors, and overwrites any value the
client sent. The router picks the first of these that applies:

1. A hash of the sticky routing key or session id, so a conversation keeps
   its rank and router replicas agree.
2. The rank with the deepest cached prefix in the local KV tree.
3. The rank with the fewest requests this router has in flight on it, or with
   `--dp-rank-policy round_robin`, the worker's ranks in turn.

In PD mode, decode is ranked by step 3 only. The decode request names the
prefill rank in `disagg_prefill_dp_rank`, and the bootstrap room satisfies
`room % prefill_dp_size == prefill_rank`.

A decode rank picked on arrival only starts decoding once prefill has queued,
computed and sent the KV, so when prefill queues are long, even ranks with
equal requests in flight run unequal batches. `--decode-dispatch-limit N` still
sends every request to prefill on arrival, but sends at most N per prefill DP
rank on to decode before decode streams their first token; the rest wait at
the router in arrival order and get their decode rank when they are sent.
Prefill must start a request before decode connects, so prefill engines need
`--optimistic-prefill-attempts` above 0 or
`--disaggregation-decode-allocation-policy prefill_complete`; the router
refuses to register a prefill whose `/server_info` reports neither, since its
gated requests could never start. A request that expires or loses its
client while waiting has its prefill aborted; requests with a caller-set `rid`
and fan-out requests go to decode at once.

### Retries

`--retry-max-attempts N` (default 1, which disables retries; 3 is typical) lets
a request that fails before any response reaches the client be sent again to a
worker it has not tried yet. A failure is a transport error, an open circuit
breaker, a 5xx, a 429, or a timeout: `--request-timeout-secs` bounds a whole
non-streaming response, and a streaming one until its headers arrive. SGLang's
chat endpoint sends those headers with the first token, so for streaming chat
the same flag bounds time to first token, queueing included; raise it for
long-context or deeply queued fleets. Once a streaming response's 2xx status
has been sent, it is never retried, and neither is any other 2xx or 4xx. In PD mode the failed
side is excluded, or both sides when the caller set `rid`, which an engine still
running it would refuse; a new pair gets a new bootstrap room. Retries share the
request's `--stale-request-timeout-secs` deadline, and none starts after it. When
every eligible worker has failed, the client gets the last failure.
`sgl_router_retries_total` counts the retried attempts.

Each retry first waits out a jittered exponential backoff: a random delay in
`[d/2, d]`, where `d` starts at `--retry-initial-backoff-ms` (default 50),
doubles per retry, and is capped at `--retry-max-backoff-ms` (default 2000).
Set the initial backoff to 0 to retry immediately. A backoff never waits past
the request's stale deadline.

Separately, a send whose pooled HTTP/1.1 connection the worker closed before
answering (reset, broken pipe or EOF) is repeated once, immediately, on the same
worker: an engine closing an idle connection never read the request. Idle
connections leave the pool after 3 s, under SGLang's default 5 s keep-alive.

### Circuit breaker

Each worker has a circuit breaker: after `--cb-threshold` (default 3)
consecutive failures the worker is taken out of routing until a request already
sent to it succeeds or, once `--cb-cool-down-secs` (default 30) has passed, a
probe request does. Under `--worker-urls` it is the only thing that takes a
dead or hung worker out of routing, since there is no active health check.
`--disable-circuit-breaker` turns it off, for example when one worker serves a
role and an open breaker would reject all of that role's requests; a worker
that stops answering then keeps its share of traffic.

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
re-tokenizing. How many depends on the model's renderer. DeepSeek-V4's, V4.1's
and Kimi-K3's native encoders and GLM-5.3's and Qwen3.8's templates are
fixture-verified against SGLang's request normalization, so they forward every
chat except multimodal ones and those with caller-provided `input_ids`. Renderers
without that verification (HF Jinja templates)
forward only plain text chat requests (string `content`, no tools, no template
kwargs or reasoning controls or historical `reasoning_content`, no assistant
continuation, no consecutive users or non-leading system turns) and warn
`UNVERIFIED` at startup; every other request shape is rendered for routing
only.

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

## OpenAI over /generate

`/v1/completions` and `/v1/chat/completions` behave as the engine's own routes,
but the router runs SGLang's OpenAI layer itself (`sglang-processor`'s `openai` module): it lowers
the request to the `GenerateReqInput` SGLang would build, sends it to
`/generate` like a native request, and builds the OpenAI response, buffered or
SSE, from the engine's output. The engine's `/server_info` supplies the server
args that layer reads (`--enable-cache-report`,
`--stream-response-default-include-usage`, `--incremental-streaming-output`, the
custom-labels header). Requests it cannot reproduce exactly go to the engine's
own route as sent: workers that disagree on those args or never
reported them, `--completion-template`, `return_hidden_states` and other
`sglext` outputs, `echo` or `logprobs` without a router tokenizer, and bodies
SGLang would reject. Chat also needs a renderer verified against SGLang (the
`AllText` scope: DeepSeek-V4, V4.1, GLM-5.3, Kimi-K3 and Qwen3.8 today), router `--default-chat-template-kwargs`
equal to the workers', a reasoning parser the processor reproduces
(`deepseek-v4`, `deepseek-v41`, `glm45`, `kimi_k3`, `qwen3`) or none, and no media yet. Tools are served with `tool_choice`
`auto` or `none`, non-strict, with standard JSON-schema types, and a tool
parser the processor reproduces (`deepseekv4`, `deepseekv41`, `kimi_k3`, `qwen3_coder`); `required`, named and strict
tools take a constraint from the engine's xgrammar, so they go to the engine
(`kimi_k3`'s `auto` tag is reproduced).
With `--tool-call-parser glm47`, a chat with tools also goes to the engine: its
grammar needs `ebnf_full_assistant`, which `/generate` does not accept.
The router reads the model's `config.json` and `generation_config.json` beside
`--tokenizer-path` (or from the HF repo), as the engine reads them, so a local
path should point into the model's directory. Engine env vars that change the
layer (`SGLANG_TOOL_STRICT_LEVEL`, `SGLANG_FORWARD_UNKNOWN_TOOLS`) are taken as
unset. `sgl_router_openai_route_total` counts each outcome.

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

## Completions

`/v1/completions` has the engine's interface and is prepared like chat (sampling
contract, PD bootstrap, DP rank), with the prompt forwarded as sent: tokens the
router computes for routing do not replace it.

## Rerank

`/v1/rerank` has the engine's interface: the same `V1RerankReqInput` body and
response, sent as `POST` or `PUT` to the model this router serves. The body is
forwarded as sent, since the engine renders and tokenizes each query-document
pair. Routing is on load, with each pair's size estimated from its text. As for
embeddings, a PD fleet answers 400 and `--dp-aware` pins no rank.

## Anthropic Messages

`/v1/messages` converts each request to a chat completion at ingress, so it is
routed exactly like `/v1/chat/completions`, and converts the reply (or stream)
back to the Messages format. Consecutive turns of the same role are merged, as
the Messages API does, and a final assistant turn is sent as a prefill
(`continue_final_message`). Besides the Messages fields, only
`chat_template_kwargs` and `continue_final_message` reach the chat request.
Every error uses the Anthropic error envelope; 5xx messages are replaced with a
generic one, and a failed stream still closes its open block and ends with
`message_stop`.

`/v1/messages/count_tokens` is answered by the router from the model's chat
template, without calling an engine. Image tokens are not counted, and a model
without a chat template answers 500 rather than a raw-text estimate.

`thinking` sets the `thinking` / `enable_thinking` chat template kwargs. The
engine's own `/v1/messages` also adapts it per reasoning parser (for example
`reasoning_effort` for Hunyuan), which the router cannot see; on such models
use `output_config.effort` or `chat_template_kwargs` directly.

## DeepSeek V4 and V4.1

V4 and V4.1 render through `rust/sglang-processor` with deepseek-recipe, DeepSeek's
own encoder, as SGLang's serving path does (`serving_deepseek_recipe.py`): SGLang's
thinking default, request and server-default reasoning effort, V4.1 numeric budgets,
`task`, and the final-assistant rule wrap recipe's rendering. The processor also
encodes the prompt ids with the checkpoint's `tokenizer.json`, so text a user writes
never becomes a control token; the model fails to load without that file. Set
`SGLANG_DSV4_REASONING_EFFORT` / `SGLANG_DSV41_REASONING_EFFORT` on the router as on
the workers. Media is left to the worker. Reference prompts are the processor's
parity fixtures.

## GLM-5.3

GLM-5.3 and GLM-5.3-Flash render with the checkpoint's `chat_template.jinja` through
`rust/sglang-processor`, which builds the template context SGLang's Jinja path builds,
so every text chat forwards `input_ids`. Media, a `chat_template` kwarg, `-0` in tools
or tool arguments, and U+001C..U+001F in assistant history render on the worker. With
the cookbook's parsers, chats without tools are served through `/generate`; chats with
tools keep the engine's route (see above).

## Qwen3.8

Qwen3.8 (Max, 27B and Flash-Next) renders with the checkpoint's `chat_template.jinja`
through `rust/sglang-processor`, which picks it by its template and builds the
template context SGLang's Jinja path builds. Every text chat forwards `input_ids`.
Media, a `chat_template` key in `chat_template_kwargs`, and text or tool numbers
Python's `trim` or `json.dumps` treat unlike Dynamo's (such as `1e-06`) render on the
worker. With the cookbook's `qwen3` and `qwen3_coder` parsers, text chats are served
through `/generate`; thinking follows the template's `enable_thinking` (on by default).

## Kimi-K3

Kimi-K3 renders through `rust/sglang-processor` (dynamo-render's XTML formatter
with SGLang's request semantics) and is tokenized with the checkpoint's chunked
tiktoken encoding; shapes the formatter renders differently go to the worker.
`--tokenizer-path` accepts a
local `tiktoken.model` or an HF repo id, whose `tiktoken.model`, `config.json`
and `tokenizer_config.json` are downloaded when it has no `tokenizer.json`.
With the cookbook's `kimi_k3` parsers, text chats are served through `/generate`.

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

## Native gRPC

`--grpc-port` serves the engine's own gRPC API, `sglang.runtime.v1.SglangService`
(`proto/sglang/runtime/v1/sglang.proto`), so a gRPC client can point at the
router or at an engine. A call is routed exactly like its HTTP route and sent to
the same RPC on the selected engine's native gRPC server, which engines run
beside HTTP with their own `--grpc-port`; the router reads that port from each
worker's `/server_info`. HTTP requests are unaffected and still reach engines
over HTTP.

| RPC | Routed like |
|---|---|
| `ChatComplete` | `/v1/chat/completions`: same rendering, `input_ids` forwarding, PD and DP-rank handling |
| `Complete` | `/v1/completions` |
| `Generate`, `TextGenerate` | `/generate` with `input_ids` or `text` |
| `Embed`, `TextEmbed`, `OpenAIEmbed` | `/v1/embeddings` |
| `Classify`, `OpenAIClassify` | `/v1/classify` |
| `Rerank` | `/v1/rerank` |

The OpenAI RPCs carry the HTTP route's JSON body, so they are prepared exactly as
over HTTP. A typed RPC is prepared from its fields with the same rules, and the
router's additions (rid, PD bootstrap, DP rank, sampling defaults) are set on the
proto it forwards. A text prompt is tokenized once, as its HTTP route forwards
`input_ids`: `TextGenerate` is sent as `Generate` with `return_text`, so the engine
still supplies the text, and `TextEmbed` as `Embed`. They stay text RPCs under
`--disable-input-ids-forwarding`, without a tokenizer, or, for `TextGenerate`, with
`return_text_in_logprobs`. Other RPCs, such as
`Tokenize`, `HealthCheck` and the admin RPCs, answer `UNIMPLEMENTED`. Request metadata and `trace_headers`
act as HTTP headers for routing and forwarding, and a cancelled call aborts the
engine request. Router errors arrive as a gRPC status carrying the HTTP path's
message and an `x-router-error-code` trailer; an engine's status passes through
unchanged. An engine without `--grpc-port` (including one launched with
`--api-key`, which native gRPC rejects) cannot serve gRPC calls: the router warns
when it registers, and calls routed to it fail with `UNAVAILABLE`.

## Upgrading from `cache_aware_zmq`

The `cache_aware_zmq` policy has been removed. Configurations using it should
select `--policy cache_aware` and choose a native cache-prefix source: the
Router-local radix tree (the default), or the external Indexer shown above.

The legacy `--cache-threshold`, `--balance-abs-threshold`, and
`--balance-rel-threshold` flags have also been removed. They do not have
one-to-one replacements; remove them and review the current `sgl-router
--help` output when tuning Cache-Aware routing.
