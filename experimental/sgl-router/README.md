# sgl-router

Slim, KV-aware router for SGLang workers.

A router serves one model and spreads its requests across that model's SGLang
engines. It picks an engine per request from live engine load, global KV-cache prefix
matches, or session affinity, supports prefill/decode (P/D) disaggregation, bucketed deployments, and
discovers engines from a static URL list or Kubernetes EndpointSlices.

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Key interfaces](#key-interfaces)
- [Routing policies](#routing-policies)
- [Deployment examples](#deployment-examples)
- [Add a new policy](#add-a-new-policy)
- [Other features](#other-features)

Related docs: [ARCHITECTURE.md](ARCHITECTURE.md) (routing design and bucket
configuration), [monitoring/README.md](monitoring/README.md) (metrics and
Grafana), [BENCHMARKS.md](BENCHMARKS.md),
[sgl-kv-indexer/README.md](sgl-kv-indexer/README.md) (external KV indexer).

## Architecture

A deployment is a **router** in front of **buckets of SGLang engines**. A bucket
is a logical group defined in the router configuration. It holds either one
**plain** engine group, where each engine does prefill and decode, or separate
**prefill and decode (P/D)** groups. Buckets can be length tiers (for example
short and long context); without a bucket file there is one plain bucket and one
P/D bucket containing every discovered engine.

![Router deployment with short-context and long-context P/D engine buckets](./assets/architecture-deployment.png)

Engines publish load reports and KV events. The router uses them, plus its own
in-flight counts and session bindings, to pick engines. An optional
[external KV indexer](sgl-kv-indexer/README.md) can replace the router's local
radix tree as the source of prefix matches. In P/D, the router sends the request
to both the chosen prefill and decode engines, and prefill transfers KV directly
to decode.

For each request, the router renders and tokenizes it, tries the buckets that fit
its length (ordered by SLO preference, capacity, then `rank`), lets the group's
policy pick an engine that passes admission, and dispatches it, retrying on
another engine if it fails before any response. Output handling (OpenAI
responses, tool and reasoning parsing) also happens in the router.

**Scalability.** Routers are stateless: nothing is persisted or shared between
replicas. Each one rebuilds its routing state (engine load, KV prefix tree,
in-flight counts) from the engines' own reports, so you scale a large deployment
by running more router replicas behind a load balancer. A new replica can copy
the KV tree from a warm sibling at startup
([peer bootstrap](#peer-bootstrap-kubernetes)). Session bindings stay in each
replica's memory, so with `session_aware` and several replicas, route a session's
requests to the same replica (for example, a load balancer sticky on the session
header).

## Quick start

Build:

```bash
cd experimental/sgl-router
cargo build --release
```

Run in front of two engines with cache-aware routing:

```bash
./target/release/sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --chat-routing reorg --policy cache_aware
```

Send a request:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Hello"}]}'
```

`--model-id` is required. `--tokenizer-path` takes a local `tokenizer.json` or a
HuggingFace repo id; when omitted, the tokenizer for `--model-id` is downloaded
from HuggingFace (honoring `HF_TOKEN` / `HF_HOME`). `sgl-router --help` lists
every flag.

## Key interfaces

### HTTP endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions`, `POST /v1/completions` | OpenAI API, served [over `/generate`](#openai-over-generate) |
| `POST`/`PUT /generate` | SGLang [native generate](#native-generate), buffered or SSE |
| `POST /v1/messages`, `/v1/messages/count_tokens` | [Anthropic Messages](#anthropic-messages) |
| `POST /v1/embeddings`, `/v1/classify`, `/v1/rerank` | [Embeddings, classify and rerank](#embeddings-classify-and-rerank) |
| `POST /v1/tokenize`, `/v1/detokenize` | Tokenize with the router's tokenizer |
| `GET /v1/models` | The served model |
| `POST /flush_cache` | Fan out a cache flush to every worker |
| `GET /healthz`, `/readyz` | Liveness and readiness |
| `GET /metrics` | Prometheus metrics, see [monitoring/README.md](monitoring/README.md) |

Both edges speak cleartext HTTP/2 where the peer does; see [HTTP/2](#http2).

### Key flags

| Area | Flags |
| --- | --- |
| Model | `--model-id`, `--tokenizer-path`, `--no-tokenizer` |
| Engines | `--worker-urls`, or `--service-discovery` with `--selector` (plain) or `--prefill-selector` + `--decode-selector` (P/D) |
| Routing | `--chat-routing reorg`, `--policy`, `--bucket-config`, `--affinity-*`, `--session-id-header` |
| Admission | `--max-in-flight`, `--max-waiting-requests`, `--max-pending-prefill-tokens`, `--max-running-usage`, `--max-kv-usage` |
| Reliability | `--retry-max-attempts`, `--request-timeout-secs`, `--stale-request-timeout-secs` |
| KV signals | `--cache-prefix-provider`, `--kv-indexer-endpoint`, `--kv-peer-selector` |

Omitting `--chat-routing` keeps the legacy routing path (default
`--policy round_robin`). This README and [ARCHITECTURE.md](ARCHITECTURE.md)
describe `--chat-routing reorg`.

## Routing policies

Each engine group runs one policy, chosen by `--policy` (default `power_of_two`
under reorg) or per group in a bucket file:

- **`power_of_two`**: samples two engines and takes the less loaded. Prefill and
  plain engines compare estimated prefill queue time, waiting uncached tokens,
  waiting, running, then in-flight requests; decode engines compare waiting and
  running requests, then KV usage. Decode groups always default to it.
- **`cache_aware`**: prefers the engine holding the longest cached prefix of the
  prompt (at least `--cache-affinity-min-matched-tokens`, default 1024, and
  optionally `--cache-affinity-min-match-ratio` of the prompt). Plain and
  prefill groups only. Needs a tokenizer.
- **`session_aware`**: keeps requests with the same session header
  (`--session-id-header`, default `x-session-id`) on the engine they are bound
  to. Any stage, including decode via a bucket file.

Both affinity policies fall back to `power_of_two` among the remaining engines
when there is no preferred engine or it fails admission. With
`--affinity-mode prefer` (default) the preferred engine wins whenever it is
admitted. With `--affinity-mode balanced` the router also samples an
alternative and switches only when the preferred engine's load exceeds both
`alternative * --affinity-load-factor` (default 2) and
`alternative + --affinity-load-gap`, measured by `--affinity-balanced-by`
(`prefill-tokens` by default, or `running-requests`).

Admission flags set per-engine caps; an engine at a cap is skipped:

| Flag | Rejects an engine at |
|---|---|
| `--max-in-flight N` | `N` requests this router has in flight to it |
| `--max-waiting-requests N` | `N` engine-reported waiting requests |
| `--max-pending-prefill-tokens N` | `N` engine-reported waiting uncached tokens |
| `--max-running-usage F` | share `F` of its reported running-request capacity |
| `--max-kv-usage F` | share `F` of its reported KV capacity |

Engine-reported caps lag by one load report, so a burst can pass them;
`--max-in-flight` counts dispatches as they happen.

For the full selection rules see ARCHITECTURE.md:
[load-aware routing](ARCHITECTURE.md#13-example-load-aware-routing),
[cache- and session-aware routing](ARCHITECTURE.md#14-example-cache-aware-and-session-aware-routing),
and [configuration examples](ARCHITECTURE.md#2-configuration-examples).

## Deployment examples

Static workers, plain engines:

```bash
sgl-router --model-id qwen3 --chat-routing reorg --policy cache_aware \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000
```

Kubernetes discovery, plain engines (omit the namespace to watch all
namespaces, which needs cluster-wide RBAC):

```bash
sgl-router --model-id qwen3 --chat-routing reorg --policy cache_aware \
  --service-discovery --service-discovery-namespace prod \
  --selector app=engines-qwen3
```

Kubernetes discovery, P/D engines with session affinity:

```bash
sgl-router --model-id qwen3 --chat-routing reorg --policy session_aware \
  --service-discovery --service-discovery-namespace prod \
  --prefill-selector app=engines-qwen3,role=prefill \
  --decode-selector app=engines-qwen3,role=decode \
  --max-kv-usage 0.9
```

Length-tiered buckets from a file (schema and a full example in
[ARCHITECTURE.md](ARCHITECTURE.md#example-c-pd-buckets-from-a-file)):

```bash
sgl-router --model-id qwen3 --chat-routing reorg --policy cache_aware \
  --service-discovery --service-discovery-namespace prod \
  --prefill-selector app=engines-qwen3,role=prefill \
  --decode-selector app=engines-qwen3,role=decode \
  --bucket-config buckets.json
```

## Add a new policy

Reorg policies live in [`src/policies_reorg/`](src/policies_reorg/). Contributions
are welcome; read [ARCHITECTURE.md](ARCHITECTURE.md) first for how buckets,
admission and fallback fit together.

1. **Implement the `Policy` trait** in `src/policies_reorg/<name>.rs`. The trait
   is in [`src/policies_reorg/mod.rs`](src/policies_reorg/mod.rs):

   ```rust
   pub trait Policy: Send + Sync + Debug {
       fn pick<'a>(&'a self, engines: &'a [Arc<Worker>], request: &'a PickRequest<'a>)
           -> BoxFuture<'a, Result<Pick, PickError>>;
       fn supports(&self, _stage: Stage) -> bool { true }
       fn needs_request_tokens(&self) -> bool { false }
       fn fallback(&self) -> Option<&dyn Policy> { None }
   }
   ```

   - `pick` must return one of the supplied `engines` and check admission on it
     (`self.admission.check(&engine, &metrics)`), returning
     `PickError::AdmissionRejected` on rejection.
   - The policy holds its own shared-state handles (load table, KV source,
     session store), passed in at construction, and reads them inside `pick`.
   - Override `supports` to restrict stages (plain, prefill, decode),
     `needs_request_tokens` if it reads `request.token_ids` or `request.prefix`,
     and `fallback` if it delegates to another policy.
   - [`power_of_two.rs`](src/policies_reorg/power_of_two.rs) is a complete
     ~80-line example; [`session_aware.rs`](src/policies_reorg/session_aware.rs)
     shows affinity with a fallback.
2. **Declare the module** in `src/policies_reorg/mod.rs`.
3. **Add a `PolicyKind` variant** in [`src/config/types.rs`](src/config/types.rs)
   with its `--policy` name. `PolicyKind` is shared with the legacy path, so add
   an arm returning an error to `build_kind` in
   [`src/policies/factory.rs`](src/policies/factory.rs) (the compiler lists every
   exhaustive match to update).
4. **Wire it into** [`src/policies_reorg/factory.rs`](src/policies_reorg/factory.rs):
   add it to the allowlist in `validate()` and construct it in the `match kind`
   of `Groups::build`.
5. **Test it** in `tests/component/policies_reorg_<name>.rs`, registered in
   `tests/component/main.rs`, following `policies_reorg_power_of_two.rs`. For
   end-to-end routing, add a case under `tests/proxy/chat_routing/reorg/`.
6. **Document it**: a subsection in [ARCHITECTURE.md](ARCHITECTURE.md) next to
   section 1.4, and a bullet in [Routing policies](#routing-policies) above.

## Other features

### Native `/generate`

`/generate` (`POST` or `PUT`) takes the engine's `GenerateReqInput` body and
returns the same response, buffered or SSE. Worker selection, P/D dispatch and
abort-on-disconnect are shared with chat completions. With a tokenizer loaded,
the router tokenizes `text` (with the special tokens SGLang adds) and forwards
`input_ids`, so routing sees the engine's exact tokens. Multimodal requests,
`--disable-input-ids-forwarding`, and tokenizers the router cannot reproduce keep
`text`. A batch goes to one worker. Prompts without a `rid` get one minted.

### OpenAI over /generate

`/v1/completions` and `/v1/chat/completions` behave as the engine's own routes,
but the router runs SGLang's OpenAI layer itself (`sglang-processor`): it lowers
the request to `GenerateReqInput`, sends it to `/generate`, and builds the OpenAI
response. Requests it cannot reproduce exactly go to the engine's own route
unchanged, for example: workers disagreeing on server args, `sglext` outputs,
`logprobs` without a tokenizer, unverified chat renderers, media, or `required` /
named / strict tools. Point `--tokenizer-path` into the model directory so
`config.json` and `generation_config.json` are read as the engine reads them.
`sgl_router_openai_route_total` counts each outcome.

### Embeddings, classify and rerank

These take the engine's request bodies and return its responses, routed on load.
A P/D fleet answers 400 to all three.

- `/v1/embeddings` (OpenAI `EmbeddingRequest`; `model` must name the served
  model): text input is tokenized and forwarded as token IDs, and a list is one
  batch for one worker.
- `/v1/classify` (`ClassifyRequest`): served like embeddings, except a batch of
  text stays text.
- `/v1/rerank` (`POST` or `PUT`, `V1RerankReqInput`): forwarded as sent.

### Anthropic Messages

`/v1/messages` converts each request to a chat completion, routes it like
`/v1/chat/completions`, and converts the reply or stream back. Errors use the
Anthropic envelope. `/v1/messages/count_tokens` is answered by the router from
the chat template without calling an engine (image tokens are not counted).
`thinking` maps to the `thinking` / `enable_thinking` template kwargs; for
models whose reasoning parser needs more, use `output_config.effort` or
`chat_template_kwargs`.

### Bucket file membership

`--bucket-config buckets.json` replaces the default plain and P/D buckets. Each
group can set its engines, `policy` (`power_of_two` or the `--policy` value),
`admission` and `affinity`; unset fields inherit the CLI flags.

- `"worker_services": ["namespace/name"]` matches Kubernetes Services via the
  `kubernetes.io/service-name` EndpointSlice label; replacement pods join
  automatically. Discovery selectors must already include those EndpointSlices.
- `"worker_ids": ["http://worker:30000"]` names static worker URLs.
- Set at most one of them; omit both for every engine of the group's role.
- `"affinity": {"mode", "balanced_by", "load_factor", "load_gap"}`; the balanced
  fields require `"mode": "balanced"`, and `power_of_two` groups take none.

Unsupported legacy options (pressure guards, cache switch margins,
queue/saturation gates) fail at startup under reorg.

### Optional tokenizer for load-only routing

`--no-tokenizer` skips loading a tokenizer for load-only policies
(`power_of_two`, `session_aware`). Workers tokenize the original messages, and
`/v1/tokenize` / `/v1/detokenize` are unavailable. Cache-aware routing and
buckets with token-length or context limits need a tokenizer.

### External KV indexer

Use an external indexer instead of the router-local radix tree as the
cache-aware prefix source:

```bash
sgl-router --model-id qwen3 --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy cache_aware --cache-prefix-provider indexer \
  --kv-indexer-endpoint http://10.0.0.10:50051
```

`--kv-indexer-query-timeout-ms` (default 100) and
`--kv-indexer-query-max-inflight` (default 32) bound queries. See
[sgl-kv-indexer/README.md](sgl-kv-indexer/README.md).

### KV-event gap replay

ZMQ can drop KV events, leaving the router crediting blocks a worker no longer
holds. Give each engine a replay socket and the router re-fetches skipped
batches:

```bash
python -m sglang.launch_server ... \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}'
```

Rank r uses port base + r, so with `dp_size > 1` keep the replay base at least
`dp_size` ports past the publish base. The engine keeps the last `buffer_steps`
batches (default 10000). `sgl_router_kv_event_replays_total{outcome}` counts
attempts.

### Pending prefixes

`--cache-pending-prefix-ttl-ms` credits a worker with a prompt's prefix for that
long after routing it there, before its KV events arrive, so bursts sharing a
cold prefix (parallel sampling, RL rollouts, agent fan-out) stay on one worker.
Off by default; needs `--policy cache_aware` with the router-local radix tree.
`sgl_router_cache_pending_prefix_hits_total` counts matches.

### Peer bootstrap (Kubernetes)

A router replica that starts mid-fleet cannot see KV already cached in the
engines. With `--kv-peer-selector`, it pulls a tree snapshot from a warm sibling
over `/internal/kv_snapshot`, and `/readyz` stays 503 until that settles.
Requires `--policy cache_aware` with the local radix tree and
`--service-discovery`.

```bash
sgl-router --model-id qwen3 --policy cache_aware \
  --service-discovery --service-discovery-namespace prod \
  --selector app=engines-qwen3 \
  --kv-peer-selector kubernetes.io/service-name=sgl-router
```

- The selector matches EndpointSlice labels (not pod labels), so use
  `kubernetes.io/service-name=<router-service>`.
- `--kv-bootstrap-timeout-ms` (default 600000) bounds the bootstrap; readiness
  probes must tolerate it. `--kv-bootstrap-fetch-timeout-cap-ms` (default
  300000) caps one fetch. `--kv-bootstrap-seed-required` keeps the replica
  unready if siblings exist but the snapshot fails.
- Grant `get`/`list`/`watch` on `endpointslices`, wire `POD_NAME`,
  `POD_NAMESPACE` and `POD_IP` from the downward API, and roll out with
  `maxUnavailable: 0`. See
  `tests/e2e/k8s_integration/manifests/kv-bootstrap.yaml`.

### PD version groups

`--pd-version-group-label <key>` pairs a prefill worker only with decode
workers that have the same value for that EndpointSlice label, so KV never
crosses engine versions during a rollout. Use one Service per role and version.
Unlabeled workers form their own group.

### DP-rank routing

With `--dp-aware`, for engines running several DP ranks (`--dp-size` /
`--attn-dp-size`), the router also picks the rank and sends it as
`X-Data-Parallel-Rank`. It uses, in order: a hash of the routing key or session
id, the rank with the deepest cached prefix, then the rank with the fewest
in-flight requests. Decode ranks by load only.

### Retries

`--retry-max-attempts N` (default 1, no retries; 3 is typical) resends a request
that failed before any response reached the client to an untried worker.
Failures are transport errors, open circuit breakers, 5xx, 429 and timeouts.
`--request-timeout-secs` bounds a non-streaming response, or a stream until its
headers (for SGLang chat, the first token). Streams that started, 2xx and 4xx
are never retried. Backoff is jittered exponential from
`--retry-initial-backoff-ms` (default 50) up to `--retry-max-backoff-ms`
(default 2000), within `--stale-request-timeout-secs`.
`sgl_router_retries_total` counts retries.

### Engines with `--api-key`

The router reads each worker's `/server_info` and `/model_info`. If engines run
with `--api-key`, pass the same key as `--worker-api-key`. Chat requests and
`/flush_cache` forward the caller's `Authorization` header instead.

### Fleet-wide sampling contract

`--override-sampling-params` fixes sampling parameters for every client:

```bash
sgl-router --model-id qwen3 --worker-urls http://10.0.0.1:30000 \
  --override-sampling-params '{"temperature": 1, "top_p": 0.95, "n": 1}' \
  --sampling-param-conflict reject
```

- A configured value is injected when the request omits the field (or sends
  `null`).
- When the request sends a different value, `reject` (default) returns 400 with
  `x-router-error-code: sampling_contract_violation`, and `allow` forwards it.
  `sgl_router_sampling_contract_rejections_total{param}` counts rejections.
- A value may be a band `{"min": LO, "max": HI}`, which only rejects; it cannot
  be combined with `allow`.
- Supported keys and ranges, checked at startup: `temperature` [0, 2], `top_p`
  (0, 1], `top_k` >= 1 or -1, `min_p` [0, 1], `repetition_penalty` (0, 2],
  `frequency_penalty` / `presence_penalty` [-2, 2], `n` [1, 128].

### Chat rendering

The router renders chats with dynamo-render, using the model's HF Jinja template
or a native encoder for models that have one. Cache-aware routing hashes the
rendered tokens. Renderers verified against SGLang forward the tokens to the
engine as `input_ids` for every non-multimodal chat; unverified ones forward
only plain-text chats and log `UNVERIFIED` at startup.

- Use the same model files on router and workers, and the same
  `--default-chat-template-kwargs` and rendering-related `SGLANG_*` environment
  variables.
- Set `--disable-input-ids-forwarding` when worker rendering may differ (worker
  template or parser overrides, array-only templates, templates with stop
  strings). Caller-supplied `input_ids` always pass through.
- `--tokenizer-backend fast` encodes with fastokens, and
  `--tokenizer-l1-cache-mb N` caches multi-turn prefixes, both to cut
  tokenization latency.

### HTTP/2

Nothing to configure. Inbound, the listener accepts cleartext HTTP/2 (h2c) and
HTTP/1.1 on the same port. Outbound, the router uses h2c to a worker only when
its `/server_info` reports `enable_http2: true` and the URL is `http://`;
`https://` workers negotiate by ALPN, and everything else uses HTTP/1.1. The
protocol is read once per worker at registration.

## License

Apache-2.0.
