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
configuration), [REFERENCE.md](REFERENCE.md) (detailed feature behavior),
[monitoring/README.md](monitoring/README.md) (metrics and
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
([peer bootstrap](#kv-cache-signals)). Session bindings stay in each
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

Both edges speak cleartext HTTP/2 where the peer does; see
[operations](#operations). With `--grpc-port`, the router also serves the
engine's [native gRPC](#native-grpc) API.

### Key flags

| Area | Flags |
| --- | --- |
| Model | `--model-id`, `--tokenizer-path`, `--no-tokenizer` |
| Engines | `--worker-urls`, or `--service-discovery` with `--selector` (plain) or `--prefill-selector` + `--decode-selector` (P/D) |
| Routing | `--chat-routing reorg`, `--policy`, `--bucket-config`, `--affinity-*`, `--session-id-header` |
| Admission | `--max-inflight-requests`, `--max-waiting-requests`, `--max-pending-prefill-tokens`, `--max-running-usage`, `--max-kv-usage` |
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
| `--max-inflight-requests N` | `N` requests this router has in flight to it (`--max-in-flight` is an alias) |
| `--max-waiting-requests N` | `N` engine-reported waiting requests |
| `--max-pending-prefill-tokens N` | `N` engine-reported waiting uncached tokens |
| `--max-running-usage F` | share `F` of its reported running-request capacity |
| `--max-kv-usage F` | share `F` of its reported KV capacity |

Engine-reported caps lag by one load report, so a burst can pass them;
`--max-inflight-requests` counts dispatches as they happen.

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

Each item links to its full description in [REFERENCE.md](REFERENCE.md).

### Native `/generate`

Same body and response as the engine's `/generate`, buffered or SSE. With a
tokenizer loaded, `text` is forwarded as `input_ids` so routing sees the
engine's exact tokens (`--disable-input-ids-forwarding` turns this off). See
[native `/generate`](REFERENCE.md#native-generate).

### OpenAI over /generate

For `/v1/chat/completions` and `/v1/completions`, the router runs SGLang's
OpenAI layer itself: it sends the request to `/generate` and builds the OpenAI
response. Requests it cannot reproduce exactly (for example media or strict
tools) go to the engine's own route unchanged. See
[OpenAI over /generate](REFERENCE.md#openai-over-generate) for the exact
conditions, and [chat rendering](REFERENCE.md#chat-rendering) plus the
per-model sections ([DeepSeek V4 and V4.1](REFERENCE.md#deepseek-v4-and-v41),
[GLM-5.3](REFERENCE.md#glm-53), [Qwen3.8](REFERENCE.md#qwen38),
[Kimi-K3](REFERENCE.md#kimi-k3)) for which chats forward `input_ids`.

### Anthropic Messages

`/v1/messages` is converted to a chat completion and back.
`/v1/messages/count_tokens` is answered by the router from the chat template.
See [Anthropic Messages](REFERENCE.md#anthropic-messages).

### Embeddings, classify and rerank

Engine request and response bodies, routed on load. Not available on P/D fleets.
See [embeddings](REFERENCE.md#embeddings), [classify](REFERENCE.md#classify)
and [rerank](REFERENCE.md#rerank).

### Native gRPC

`--grpc-port` serves the engine's `sglang.runtime.v1.SglangService`. Each RPC is
routed like its HTTP route and forwarded to the same RPC on the engine's own
native gRPC port, so engines must run with `--grpc-port` too. See
[native gRPC](REFERENCE.md#native-grpc).

### Bucket file

`--bucket-config` defines buckets whose groups set `worker_services`
(`namespace/name`) or `worker_ids`, plus `policy`, `admission` and `affinity`;
unset fields inherit the CLI flags. See
[ARCHITECTURE.md](ARCHITECTURE.md#2-configuration-examples) and
[bucket file membership](REFERENCE.md#bucket-file-membership).

### KV cache signals

- **External KV indexer**: `--cache-prefix-provider indexer
  --kv-indexer-endpoint <url>`; see [sgl-kv-indexer](sgl-kv-indexer/README.md)
  and [external KV indexer](REFERENCE.md#external-kv-indexer).
- **KV-event gap replay**: add `"replay_endpoint"` to the engine's
  `--kv-events-config` so the router can re-fetch dropped KV events; see
  [gap replay](REFERENCE.md#kv-event-gap-replay).
- **Pending prefixes**: `--cache-pending-prefix-ttl-ms` keeps bursts that share
  a cold prefix on one worker until its KV events arrive; see
  [pending prefixes](REFERENCE.md#pending-prefixes).
- **Peer bootstrap (Kubernetes)**: `--kv-peer-selector
  kubernetes.io/service-name=<router-service>` makes a new replica copy the KV
  tree from a sibling before `/readyz` passes; see
  [peer bootstrap](REFERENCE.md#peer-bootstrap-kubernetes).
- **LoRA cache identity**: with workers that publish `BlockStored.lora_name`,
  prefix matches are kept per adapter; see
  [LoRA cache identity](REFERENCE.md#lora-cache-identity).

### Operations

- **No tokenizer**: `--no-tokenizer` for load-only policies (`power_of_two`,
  `session_aware`); see [optional tokenizer](REFERENCE.md#optional-tokenizer-for-load-only-routing).
- **Retries**: `--retry-max-attempts N` resends a request that failed before
  any response to another worker; see [retries](REFERENCE.md#retries).
- **Circuit breaker**: a worker leaves routing after `--cb-threshold`
  consecutive failures; `--disable-circuit-breaker` turns this off; see
  [circuit breaker](REFERENCE.md#circuit-breaker).
- **DP-rank routing**: `--dp-aware` also picks the DP rank inside multi-rank
  engines, and `--dp-rank-policy round_robin` cycles ranks that no routing key
  or cached prefix decides; see [DP-rank routing](REFERENCE.md#dp-rank-routing).
- **PD version groups**: `--pd-version-group-label <key>` pairs prefill and
  decode only within the same engine version; see
  [PD version groups](REFERENCE.md#pd-version-groups).
- **Engine API key**: `--worker-api-key` for engines launched with `--api-key`;
  see [engines with `--api-key`](REFERENCE.md#engines-with---api-key).
- **Sampling contract**: `--override-sampling-params '<json>'` pins sampling
  parameters fleet-wide; `--sampling-param-conflict reject|allow` decides what
  happens to requests that differ. See
  [fleet-wide sampling contract](REFERENCE.md#fleet-wide-sampling-contract)
  for parameter domains and null handling.
- **Chat rendering**: use the same model files and chat-template settings on
  router and workers, or set `--disable-input-ids-forwarding`.
  `--tokenizer-backend fast` and `--tokenizer-l1-cache-mb N` speed up
  tokenization.
- **HTTP/2**: no configuration. h2c is used inbound when the client speaks it,
  and outbound to workers that report `enable_http2`; see
  [HTTP/2](REFERENCE.md#http2).
- **Upgrading**: the `cache_aware_zmq` policy and the legacy cache and balance
  threshold flags were removed; see
  [upgrading](REFERENCE.md#upgrading-from-cache_aware_zmq).

## License

Apache-2.0.
