# Migrating from sgl-model-gateway to sgl-router

SGLang 1.0 removes `sgl-model-gateway` and replaces it with `sgl-router`. This
guide is for deployments that run the gateway, either as the `sgl-model-gateway`
binary or through `python -m sglang_router.launch_router`, and move to the
`sgl-router` binary.

- [What changes](#what-changes)
- [Flag mapping](#flag-mapping)
- [PD disaggregation](#pd-disaggregation)
- [DP-aware routing](#dp-aware-routing)
- [Cache-aware routing](#cache-aware-routing)
- [Unsupported features](#unsupported-features)
- [Check the migration](#check-the-migration)

Versions compared: the gateway as of SGLang
[v0.5.23](https://github.com/sgl-project/sglang/tree/v0.5.23/sgl-model-gateway),
and `sgl-router` on `main`. The `sgl-router` 0.1.0 release (`pip install
sgl-router`, `lmsysorg/sgl-router:v0.1.0`) accepts every router flag used here
except `--dp-rank-policy least_tokens` and `--decode-dp-rank-policy`, and it
answers `POST /v1/responses` with `501`. `sgl-router --version` prints the
version of a binary, and `sgl-router --help` lists the flags it accepts.

## What changes

- **One model per router.** `--model-id` is required. It is the `model` name
  clients send and must match the engines' served model name, which is their
  `--served-model-name`, or their `--model-path` when that flag is not set.
  Multi-model routing (`--enable-igw`) is gone: run one router per model.
- **Engines describe themselves.** The router reads each engine's `/server_info`
  and `/model_info` to learn its served model, PD role, prefill bootstrap port,
  DP size, and KV-event and load-report endpoints. `--pd-disaggregation`,
  `--prefill URL [PORT]` and `--decode URL` are replaced by one `--worker-urls`
  list. The engines' own flags still decide their roles; the router only reads
  them.
- **Two routing paths.** `--chat-routing legacy` (the default, with `--policy
  round_robin` by default) and `--chat-routing reorg` (`--policy power_of_two`
  by default) are two implementations inside `sgl-router`; neither runs the old
  gateway code. Reorg accepts only `power_of_two`, `cache_aware`, and
  `session_aware`, and requires `--decode-policy power_of_two`. The router
  [README](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) describe reorg, and
  the examples below use it.
- **Different defaults.** The gateway listened on `0.0.0.0` by default and
  `sgl-router` on `127.0.0.1`. The gateway defaulted to `--policy cache_aware`, five
  attempts per request, a 1800 s request timeout, a 512 MiB request body cap, and
  Prometheus on port 29000. `sgl-router` defaults to `round_robin` (legacy) or
  `power_of_two` (reorg), one attempt (no retries), a 300 s request timeout, a
  32 MiB body cap, and `/metrics` on its main port. Metric names change from
  `smg_*` to `sgl_router_*`; see [monitoring/README.md](monitoring/README.md).
- **No client authentication or rate limiting in the router.** See
  [Authentication](#authentication).
- **No dynamic worker API.** Workers come only from `--worker-urls` or
  Kubernetes discovery.

## Flag mapping

These flags keep their name and meaning: `--port`, `--worker-urls`,
`--service-discovery`, `--service-discovery-namespace`, `--selector`,
`--prefill-selector`, `--decode-selector`, `--dp-aware`, `--tokenizer-path`,
`--log-level`,
`--retry-initial-backoff-ms`, `--disable-circuit-breaker`.

### Renamed or replaced

| sgl-model-gateway | sgl-router | Notes |
| --- | --- | --- |
| `--host` | `--host` | Default `0.0.0.0` in the gateway, `127.0.0.1` in `sgl-router`. Pass `--host 0.0.0.0` to accept connections from other hosts. |
| `--model-path PATH` | `--model-id NAME`, plus `--tokenizer-path PATH` if the tokenizer is not at `NAME` | `--tokenizer-path` defaults to `--model-id`. `--model-alias NAME` accepts another `model` name from clients. |
| `--pd-disaggregation` | remove | Static workers: the PD role comes from each engine's `/server_info`. Kubernetes: from the selector an EndpointSlice matches. |
| `--prefill URL [BOOTSTRAP_PORT]`, `--decode URL` | `--worker-urls URL...`, prefill and decode together | The bootstrap port comes from the prefill engine's `/server_info` (its `--disaggregation-bootstrap-port`, default 8998). |
| `--prefill-policy P` | `--policy P` | `--policy` applies to plain and prefill engines. |
| `--decode-policy P` | `--decode-policy power_of_two` | The only other value is `legacy_host_affinity`, legacy path only. `cache_aware` never selects decode engines. |
| `--decode-dispatch-limit N` | `--decode-dispatch-limit N` | Now per prefill DP rank instead of per prefill worker; see [PD disaggregation](#pd-disaggregation). |
| `--request-timeout-secs` | `--request-timeout-secs` | Default 1800 in the gateway, 300 in `sgl-router`. For streaming chat it bounds the time to the first token; see [retries](REFERENCE.md#retries). |
| `--retry-max-retries N` | `--retry-max-attempts N` | Both count the first attempt. Default 5 in the gateway, 1 in `sgl-router`. |
| `--disable-retries` | `--retry-max-attempts 1` | Already the default. |
| `--retry-max-backoff-ms` | `--retry-max-backoff-ms` | Default 30000 in the gateway, 2000 in `sgl-router`. |
| `--cb-failure-threshold N` | `--cb-threshold N` | Consecutive failures that open the breaker. Default 10 in the gateway, 3 in `sgl-router`. |
| `--cb-timeout-duration-secs S` | `--cb-cool-down-secs S` | Default 60 in the gateway, 30 in `sgl-router`. |
| `--json-log` | `--log-format json` | |
| `--log-dir DIR` | remove | Logs go to stdout. |
| `--prometheus-port`, `--prometheus-host` | remove | `/metrics` is served on `--host` and `--port`. Update scrape targets. |
| `--max-payload-size BYTES` | `SGLANG_ROUTER_MAX_BODY_BYTES=BYTES` (environment variable) | Default 32 MiB instead of 512 MiB; larger bodies get `413`. See [request body size](REFERENCE.md#request-body-size). |
| `--tokenizer-cache-enable-l1`, `--tokenizer-cache-l1-max-memory BYTES` | `--tokenizer-l1-cache-mb MB` | Size in MiB; `0` (default) disables the cache. |
| `--chat-template` | set it on the engines | Give the router the same model files and chat-template settings as the engines, or pass `--disable-input-ids-forwarding`; see [chat rendering](REFERENCE.md#chat-rendering). |
| `--reasoning-parser`, `--tool-call-parser` | set them on the engines | The engines parse. When the router serves a chat over `/generate` itself, it applies the parsers the engines report; see [OpenAI over /generate](REFERENCE.md#openai-over-generate). |
| `--worker-startup-timeout-secs`, `--worker-startup-check-interval` | remove | Workers register as they come up; `/readyz` returns `503` until the model can be served. |
| `--service-discovery-port`, `--bootstrap-port-annotation` | remove | Ports come from the EndpointSlices and from `/server_info`. |
| `--policy manual`, `--max-idle-secs`, `--assignment-mode` | `--policy sticky`, `--sticky-idle-secs`, `--sticky-fallback-policy` | Legacy path only, and not the same algorithm. |

### No equivalent

| sgl-model-gateway | What to do |
| --- | --- |
| `--cache-threshold` | Remove. `--cache-affinity-min-matched-tokens` and `--cache-affinity-min-match-ratio` set the minimum match, in a different unit; see [Cache-aware routing](#cache-aware-routing). |
| `--balance-abs-threshold`, `--balance-rel-threshold` | Remove. Reorg's `--affinity-mode balanced` with `--affinity-load-factor` and `--affinity-load-gap` is the closest control. |
| `--eviction-interval`, `--max-tree-size` | Remove. The prefix index follows the engines' KV events, evictions included. |
| `--policy prefix_hash`, `--prefix-token-count`, `--prefix-hash-load-factor` | Choose `cache_aware`, `session_aware`, or `power_of_two`. |
| `--api-key` | See [Authentication](#authentication). `--worker-api-key` alone does not replace it. |
| `--max-concurrent-requests`, `--queue-size`, `--queue-timeout-secs`, `--rate-limit-tokens-per-second` | Enforce rate limits at an ingress. The router has no queue. Reorg admission flags (`--max-inflight-requests`, `--max-waiting-requests`, ...) cap each engine, and a request gets `503` at once when every engine is at a cap. |
| `--health-check-*`, `--disable-health-check` | Remove. There is no active health check; the [circuit breaker](REFERENCE.md#circuit-breaker) takes failing workers out of routing. |
| `--cb-success-threshold`, `--cb-window-duration-secs`, `--retry-backoff-multiplier`, `--retry-jitter-factor` | Remove. A single success closes the breaker, and the retry backoff doubles with a fixed jitter. |
| `--enable-igw` | Run one router per model. |
| `--tls-cert-path`, `--tls-key-path`, `--client-cert-path`, `--client-key-path`, `--ca-cert-paths` | Terminate TLS at an ingress; the router serves cleartext HTTP. |
| `--jwt-*`, `--control-plane-api-keys`, `--disable-audit-logging` | Not supported. |
| `--backend` (`openai` and other non-SGLang backends), `grpc://` worker URLs | Not supported. Workers must be SGLang engines at `http://` or `https://` URLs. `--grpc-port` serves gRPC to clients; see [native gRPC](REFERENCE.md#native-grpc). |
| `--history-backend`, `--oracle-*`, `--postgres-*`, `--redis-*`, `--mcp-config-path`, `--enable-wasm` | Not supported; the router stores no responses or conversations. |
| `--enable-mesh`, `--mesh-*` | Not supported. Router replicas share no state and run independently behind a load balancer. |
| `--enable-trace`, `--otlp-traces-endpoint` | Not supported. `traceparent` and `tracestate` are forwarded to the engines. |
| `--request-id-headers` | Not supported. `x-request-id` and `x-request-id-*` are forwarded to the engines. |
| `--shutdown-grace-period-secs` | Not supported. See `--shutdown-drain-secs` and `--termination-grace-secs` in `sgl-router --help`. |
| `--cors-allowed-origins`, `--pool-idle-timeout-secs`, `--pool-max-idle-per-host`, `--connect-timeout-secs`, `--tcp-keepalive-secs`, `--tokenizer-cache-enable-l0`, `--tokenizer-cache-l0-max-entries`, `--mini-lb`, `--test-external-dp-routing` | Not supported. |

On `main`, `sgl-router` rejects these flags with clap's generic `unexpected
argument` error. [#50643](https://github.com/sgl-project/sglang/pull/50643) (open)
makes the error name the replacement.

### Authentication

The gateway's `--api-key` did two things: clients had to send it as
`Authorization: Bearer <key>`, and the gateway sent it to the workers listed in
`--worker-urls`. `sgl-router` checks no client keys:

- It forwards the caller's `Authorization` header with every request it passes
  to an engine (chat, completions, `/generate`, and the others). Engines started
  with `--api-key` still reject requests without the key.
- `--worker-api-key KEY` is only for the router's own requests to the engines,
  such as reading `/server_info`. Set it when the engines run with `--api-key`,
  or the workers cannot register; see
  [engines with `--api-key`](REFERENCE.md#engines-with---api-key).
- The routes the router answers itself, such as `/v1/models`, `/v1/tokenize`,
  `/v1/messages/count_tokens`, `/metrics`, and `/readyz`, need no key.

A gateway that shared one key with its engines therefore maps to
`--worker-api-key KEY` with the engines unchanged: clients keep sending the same
key, and the engines check it. If the engines ran without a key, or no route may
be reachable without one, put authentication in an ingress in front of the router.

## PD disaggregation

The engines must already run as prefill and decode servers; the router does not
change an engine's role. This follows the
[Llama single-node example](../docs/docs/advanced_features/pd_disaggregation.mdx#llama-single-node),
which needs two GPUs and an RDMA device (replace `mlx5_roce0` with yours):

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill --port 30000 \
  --disaggregation-ib-device mlx5_roce0
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode --port 30001 --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0
```

For other transfer backends and multi-node layouts, see
[PD disaggregation](../docs/docs/advanced_features/pd_disaggregation.mdx).

Before:

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --policy power_of_two \
  --host 0.0.0.0 --port 8000
```

After:

```bash
sgl-router \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --worker-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --chat-routing reorg --policy power_of_two \
  --no-tokenizer \
  --host 0.0.0.0 --port 8000
```

- `--no-tokenizer` leaves tokenization to the engines. It works only with
  load-only policies (`power_of_two`, `session_aware`); drop it to use
  `cache_aware`.
- A prefill bootstrap port other than 8998 needs nothing extra: the router reads
  it from `/server_info`.
- With `--decode-dispatch-limit N`, the prefill engines must run with
  `--optimistic-prefill-attempts` above 0 or
  `--disaggregation-decode-allocation-policy prefill_complete`; the router
  refuses to register a prefill engine with neither. The limit now applies per
  prefill DP rank. See [DP-rank routing](REFERENCE.md#dp-rank-routing).

On Kubernetes, the gateway watched Pods and read the bootstrap port from the
`sglang.ai/bootstrap-port` annotation. `sgl-router` watches EndpointSlices, so the
selectors must match labels on the engines' Services, which their EndpointSlices
copy, and the router's service account needs `get`, `list`, and `watch` on
`endpointslices`:

```bash
sgl-router \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --service-discovery --service-discovery-namespace inference \
  --prefill-selector app=llama,role=prefill \
  --decode-selector app=llama,role=decode \
  --chat-routing reorg --policy power_of_two --no-tokenizer \
  --host 0.0.0.0
```

## DP-aware routing

`--dp-aware` makes the router also pick the DP rank inside each engine and send
it as the `X-Data-Parallel-Rank` header, replacing any value the client sent. It
only matters when the engines run several DP ranks. The router counts an engine's
ranks as `dp_size x attn_dp_size` from its `/server_info`, so either engine flag
works:

- `--dp-size N`: N full model replicas in one server, each on its own GPUs. The
  [DP and DPA guide](../docs/docs/advanced_features/dp_dpa_router_guide.mdx)
  recommends separate servers behind the router over this mode.
- `--attn-dp-size N`: attention data parallelism inside the TP group, mainly for
  MLA and MoE models such as DeepSeek; `--tp-size` must be divisible by N.

The engines reject `--dp-size` above 1 combined with `--attn-dp-size` above 1.
Example engines, two GPUs each:

```bash
# On 10.0.0.1 and 10.0.0.2
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --dp-size 2 --host 0.0.0.0 --port 30001
```

Before:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://10.0.0.1:30001 http://10.0.0.2:30001 \
  --policy power_of_two --dp-aware
```

After:

```bash
sgl-router \
  --model-id Qwen/Qwen3-0.6B \
  --worker-urls http://10.0.0.1:30001 http://10.0.0.2:30001 \
  --chat-routing reorg --policy power_of_two \
  --dp-aware --host 0.0.0.0
```

The router picks the rank from, in order: a hash of the session or sticky key,
the rank with the longest cached prefix, then `--dp-rank-policy`:
`least_in_flight` (default), `round_robin`, or `least_tokens` (uses the engines'
load reports; not in 0.1.0). In PD mode, decode ranks use only the last step,
with `--decode-dp-rank-policy` when set (not in 0.1.0). See
[DP-rank routing](REFERENCE.md#dp-rank-routing).

## Cache-aware routing

The gateway's `cache_aware` kept its own prefix tree of the request text it had
routed and stayed with the matching worker while at least `--cache-threshold`
(default 0.3) of the request matched. `sgl-router`'s `cache_aware` builds its
prefix index from KV events the engines publish, so it follows what each engine
actually holds. It needs:

- Every engine started with `--kv-events-config`. Without it, the router has no
  prefix data for that engine.
- A tokenizer in the router: `--tokenizer-path`, or the Hugging Face tokenizer of
  `--model-id` (set `HF_TOKEN` for gated models). `--no-tokenizer` does not work
  with `cache_aware`.
- Optionally, `--load-publish-endpoint auto` on every engine. The load reports
  feed the `power_of_two` fallback, balanced affinity, and the engine-reported
  admission caps.

KV events say which prefix blocks an engine holds; load reports say how busy it
is. Neither carries KV data through the router.

Engines, one per host:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 30001 \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv"}' \
  --load-publish-endpoint auto
```

`auto` takes two ports per DP rank (`dp_size x attn_dp_size` ranks) starting at
the KV port (5557 here), so keep the KV ports of engines on the same host far
enough apart. Check that an engine
advertises its publishers:

```bash
curl -s http://10.0.0.1:30001/server_info \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["kv_events"])'
```

Before:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://10.0.0.1:30001 http://10.0.0.2:30001 \
  --policy cache_aware --cache-threshold 0.3
```

After:

```bash
sgl-router \
  --model-id Qwen/Qwen3-0.6B \
  --worker-urls http://10.0.0.1:30001 http://10.0.0.2:30001 \
  --chat-routing reorg --policy cache_aware \
  --cache-affinity-min-matched-tokens 1024 \
  --host 0.0.0.0
```

- A request goes to the engine holding its longest cached prefix if the match is
  at least `--cache-affinity-min-matched-tokens` tokens (default 1024) and, when
  set, `--cache-affinity-min-match-ratio` of the prompt. Otherwise it falls back
  to `power_of_two`. The gateway's ratio does not translate into a token count;
  choose the minimum from the length of your shared prefixes.
- `--affinity-mode prefer` (default) keeps the prefix owner whenever it passes
  admission. `--affinity-mode balanced` moves a request off a busy owner, the
  closest match to the gateway's balance thresholds; see
  [prefer or balanced](EXAMPLES.md#prefer-or-balanced).
- In PD, `cache_aware` selects the prefill engine; decode uses `power_of_two`.
- A restarted router starts with an empty index, because engines do not
  re-announce blocks they already hold; see [gotchas](EXAMPLES.md#gotchas). On
  Kubernetes, `--kv-peer-selector` copies the index from a running replica; see
  [peer bootstrap](REFERENCE.md#peer-bootstrap-kubernetes).
- With attention DP across nodes, `/server_info` advertises one host for the KV
  publishers, so the router reaches only the head node's ranks. SGLang logs a
  warning at startup ([#50751](https://github.com/sgl-project/sglang/pull/50751)).
  Subscribing to every publishing rank is a "Multi-node cache-aware routing" item
  on the [SGLang 1.0 tracker](https://github.com/sgl-project/sglang/issues/50045).

## Unsupported features

| Gateway feature | In sgl-router |
| --- | --- |
| Several models behind one gateway (`--enable-igw`) | One model per router. |
| Client API keys, JWT, control-plane keys, TLS | None; use an ingress. Engines still check the forwarded `Authorization` header. |
| Rate limits and a request queue | None; use an ingress. Reorg admission caps answer `503`. |
| Active worker health checks | None; the circuit breaker takes failing workers out of routing. |
| Adding and removing workers over HTTP | None; use `--worker-urls` or Kubernetes discovery. |
| OpenAI and other non-SGLang backends, gRPC workers | None. |
| Stored responses, conversations, history databases, MCP, WASM | None. `POST /v1/responses` is served without storage on `main`. |
| Mesh and HA state sync | None; router replicas share no state. |
| OpenTelemetry export | None. |
| Separate metrics port, `smg_*` metrics | `/metrics` on the main port, `sgl_router_*` names. |

HTTP routes:

| Gateway route | In sgl-router |
| --- | --- |
| `/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/classify`, `/v1/rerank`, `/v1/models`, `/v1/tokenize`, `/v1/detokenize`, `/flush_cache`, `/server_info`, `/model_info`, `/get_server_info`, `/get_model_info`, `/health_generate`, `/health`, `/liveness`, `/readiness` | Served. |
| `POST /v1/responses` | Served without storage on `main`: `previous_response_id`, `conversation`, and `background` answer `400`. `501` in 0.1.0. See [OpenAI Responses](REFERENCE.md#openai-responses). |
| `/v1/responses/{id}` and its subroutes, `/v1/conversations` and its subroutes, `/v1/loads` | `501`, error code `unsupported_endpoint`. |
| `/workers`, `/get_loads`, `/engine_metrics`, `/parse/reasoning`, `/parse/function_call`, `/v1/tokenizers`, `/wasm`, `/ha/*` | `404`. |

New in `sgl-router`: `/healthz`, `/readyz`, and `/v1/messages` (Anthropic
Messages).

## Check the migration

1. `sgl-router --version` and `sgl-router --help` show the version you deploy
   and accept your flags.
2. `/readyz` returns `200` once the workers are registered, and `/v1/models`
   lists `--model-id`:

   ```bash
   curl -fsS http://127.0.0.1:8000/readyz
   curl -fsS http://127.0.0.1:8000/v1/models
   ```

3. A chat request works, with and without `"stream": true`:

   ```bash
   curl -sS http://127.0.0.1:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
   ```

4. Requests reach every engine:

   ```bash
   curl -s http://127.0.0.1:8000/metrics | grep '^sgl_router_worker_requests_total'
   ```

5. Authentication, rate limits, and TLS are in place at the ingress; Prometheus
   scrapes the main port; dashboards and alerts use the `sgl_router_*` names.
6. Request bodies stay under the 32 MiB cap, or `SGLANG_ROUTER_MAX_BODY_BYTES`
   is raised.
7. `--retry-max-attempts` and `--request-timeout-secs` are set for your traffic:
   the gateway retried by default and waited up to 1800 s; `sgl-router` does not
   retry and waits up to 300 s.
