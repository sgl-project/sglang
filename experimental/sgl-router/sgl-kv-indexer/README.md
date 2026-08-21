# SGL KV Indexer (in-memory build)

`sgl-kv-indexer` is an experimental metadata service for SGLang KV-cache
blocks. It records which worker and storage tier currently holds each
content-addressed block, allowing a router to query likely cache hits without
moving KV data itself.

Each Indexer uses a process-local in-memory index with no external storage
dependency. The state is soft, but a paired Bridge rebuilds it from every
Worker's Snapshot + Live Events before that replica reports READY.

## Architecture

```text
Worker fleet ── Snapshot + ZMQ Live ──> Bridge i ── gRPC ──> Indexer i
Router fleet <──── 100 ms status ───── Indexer fleet
Router ───────── MatchPrefix ─────────> one fresh READY Indexer
```

- SGLang publishes `BlockStored`, `BlockRemoved`, and `AllBlocksCleared` events.
- One `kv-indexer-bridge` is paired 1:1 with one `kv-indexer-server`; that
  Bridge follows every configured worker/rank stream.
- Multiple Indexer replicas independently hold the complete placement view.
- Every Indexer reports readiness, coverage, and normalized query saturation to
  every Router at a 100 ms default interval.
- Placement, worker metadata, reverse holdings, and hit counters live in that
  server process behind a single read/write lock.

Each snapshot replacement and apply RPC is atomic within the process, and a
query sees a consistent view. Recovery-aware applies are fenced by Worker epoch
and contiguous sequence: duplicates are acknowledged, while a gap invalidates
and clears that Worker until a new snapshot is installed. There is no Indexer
leader, consensus, persistence, or state sharing between replicas.

## Operational contract

Run any number of `Bridge + Indexer` pairs. Routers select the lowest-load fresh
READY report, retry another READY member on timeout/unreachable/overload, and
fall back to load-only Worker routing when none is usable.

Fleet mode requires snapshot-capable Workers. The Bridge subscribes to Live
Events before requesting `snapshot-v1`, waits for the exact barrier, installs
the snapshot atomically, and then applies only contiguous events from the same
epoch. Indexer restart (detected by a per-process epoch even while Workers are
idle), Worker epoch change, sequence gap, or Bridge reconnect starts a fresh
snapshot cycle. Snapshot v1 contains whole-block HBM placement;
tier/component-complete recovery remains a future protocol version.

## In-memory data model

The server stores:

- block hash → token count and `(worker, tier) → component mask`
- worker → router-facing address, `WorkerCacheSpec`, and reverse holdings by tier
- block hash → cumulative hit count

Component masks are `FULL=1`, `SWA=2`, `MAMBA=4`, and `0` for legacy
whole-block events. `REPORT` replaces the component snapshot for one
`(worker, tier, block)` placement. `REVOKE` removes that placement, and
`CLEAR_ALL_AT_TIER` removes every placement for the worker at that tier.

## Build

```bash
cd experimental/sgl-router
cargo build --release -p sgl-kv-indexer
```

This produces:

- `target/release/kv-indexer-server`
- `target/release/kv-indexer-bridge`

## End-to-end quickstart

Component-aware routing requires an SGLang engine build that supports
`component_types`. Start each command in its own terminal.

1. Start each Indexer server. Status reporting is enabled when Router URLs are
configured:

```bash
KV_INDEXER_LISTEN_ADDR=127.0.0.1:50051 \
KV_INDEXER_ID=indexer-1 \
KV_INDEXER_ADVERTISE_ENDPOINT=http://127.0.0.1:50051 \
KV_INDEXER_ROUTER_URLS=http://127.0.0.1:3001 \
  cargo run --release --bin kv-indexer-server
```

`KV_INDEXER_LISTEN_ADDR` defaults to `[::1]:50051`.
`KV_INDEXER_PREFIX_QUERY_MAX_INFLIGHT` sets the maximum number of prefix
queries executing concurrently and defaults to `32`. Requests above the limit
are rejected immediately with gRPC `RESOURCE_EXHAUSTED`.
`KV_INDEXER_STATUS_INTERVAL_MS` defaults to `100`; Router reports expire after
500 ms. The normalized load is current prefix-query saturation divided by the
configured in-flight capacity.

2. Start one Bridge paired with this Indexer. Fleet mode configures all Worker
streams in one JSON array:

```bash
KV_INDEXER_ENDPOINT=http://127.0.0.1:50051 \
KV_INDEXER_WORKERS_JSON='[
  {
    "worker_id":"worker-0",
    "worker_address":"http://127.0.0.1:30000",
    "event_endpoint":"tcp://127.0.0.1:5567",
    "snapshot_endpoint":"tcp://127.0.0.1:5767",
    "event_topic":"kv-events",
    "dp_rank":0
  },
  {
    "worker_id":"worker-1",
    "worker_address":"http://127.0.0.1:30001",
    "event_endpoint":"tcp://127.0.0.1:5568",
    "snapshot_endpoint":"tcp://127.0.0.1:5768",
    "event_topic":"kv-events",
    "dp_rank":0
  }
]' \
  cargo run --release --bin kv-indexer-bridge
```

3. Start the matching SGLang worker:

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --port 30000 \
  --kv-events-config \
    '{"publisher":"zmq","endpoint":"tcp://*:5567","snapshot_endpoint":"tcp://*:5767","topic":"kv-events"}'
```

4. Start the Router with the Indexer as the authoritative cache signal:

```bash
sgl-router \
  --model-id <model-id> \
  --tokenizer-path <huggingface-repo-or-tokenizer> \
  --worker-urls http://127.0.0.1:30000 \
  --policy cache_aware_zmq \
  --kv-indexer-endpoint http://127.0.0.1:50051 \
  --kv-indexer-query-timeout-ms 100 \
  --kv-indexer-query-max-inflight 32
```

For multiple Indexer replicas, repeat steps 1–2 with unique Indexer ports/IDs;
each Bridge uses the same Worker list and its paired Indexer endpoint.

## API

The protobuf service in `proto/kv_indexer.proto` provides:

- `ApplyExternalKvBatch`: ordered placement reports, revocations, and clears.
  Recovery-aware requests enforce epoch and contiguous sequence.
- `ConfigureExpectedWorkers`: atomically configures desired Worker coverage and
  removes scaled-in Worker state.
- `InvalidateWorker`: clears one Worker's placement and marks it NOT_READY
  before recovery/reconnect.
- `ReplaceExternalKvSnapshot`: atomically installs one Worker snapshot and
  sequence checkpoint.
- `MatchExternalKv`: workers and tiers holding requested block hashes.
- `MatchExternalKvPrefix`: per-worker longest contiguous reusable prefix.
- `GetExternalKvHitCounts`: per-block hit counters.

There is no gRPC health service in this build.

## Prefix routing semantics

For a legacy worker, `matched_prefix_blocks` is the largest `n` such that it
holds every block in `hashes[0..n)` without a gap. For a component-aware worker:

- FULL must be contiguous on every matched block.
- SWA must cover the trailing `swa_window_tokens` at the candidate boundary, or
  form an unbroken run from the prompt head.
- MAMBA must be present on the candidate boundary block.

Component placements without a worker spec fail closed. Workers with an empty
router-facing address are excluded.

The Indexer returns every candidate sorted by prefix length; it does not choose
a worker. When configured, it replaces the Router's local radix tree as the
cache signal: the Router intersects Indexer results with its healthy candidates,
and a successful query with no usable match selects by minimum active load.
Indexer connection failures, timeouts, overload, and a prompt too long to fit one
gRPC message fall back to that same minimum-active-load selection, so an
unreachable Indexer costs cache affinity rather than availability; a rejected RPC
still fails the Router request with `503`, because it means the two sides
disagree on the request contract. An
endpoint the Router could never dial is rejected at startup instead of failing
every query later. The local radix tree is used only when no Indexer endpoint is
configured. The per-query deadline defaults to 100ms and can be changed with
`--kv-indexer-query-timeout-ms`. The Router-side admission bound defaults to 32
concurrent calls and can be changed with `--kv-indexer-query-max-inflight`.

Prefix queries carry no Indexer-imposed block cap beyond the caller's
`max_blocks` ceiling — unlike applies and `MatchExternalKv`, which reject above
16,384 hashes. The in-memory backend scans the request in one pass over a single
consistent snapshot, holding O(1) matching state per candidate worker and
considering only workers that hold the first block, so request length costs time
but not memory. Block hashes use packed `sfixed64` encoding, and the server
accepts decoded gRPC messages up to 8 MiB (roughly one million hashes).
Server work is bounded by `max_blocks` when the caller supplies one and by that
transport limit. The Router's per-query deadline bounds how long it waits for an
answer, but does not cancel a synchronous scan already in progress. A first-block
miss returns immediately with `blocks_read=1`.

Message decoding happens before a request reaches the service, so
`KV_INDEXER_PREFIX_QUERY_MAX_INFLIGHT` bounds the scan but not the bytes a peer
makes the server buffer. That is bounded instead by the HTTP/2 stream limit: each
connection is capped at 64 concurrent streams, bounding that connection to
64 × 8 MiB of undecoded requests. For a query past the 8 MiB ceiling, the Router
sends only the leading hashes that fit and still divides the returned prefix by
the full request's block count. This preserves a useful lower-bound cache signal
without overstating the match rate. If an Indexer has a lower ceiling and returns
gRPC `OUT_OF_RANGE`, the Router falls back to minimum active load.

Long scans hold the read lock throughout. Operators serving very long prompts
should set `max_blocks` instead of relying on the message-size limit.

## Overload behavior and observability

The Router and server apply separate admission bounds. The Router rejects a
query locally when its `--kv-indexer-query-max-inflight` permits are exhausted;
the server returns gRPC `RESOURCE_EXHAUSTED` when
`KV_INDEXER_PREFIX_QUERY_MAX_INFLIGHT` is exhausted. Both leave the request
routed by minimum active load, logged at `WARN` on the Router.

Every Router query publishes its timeout through the gRPC `grpc-timeout` header.
The server timestamps arrival and returns `DEADLINE_EXCEEDED` before backend work
when queueing has already consumed that budget. Apply/event RPCs are never shed,
because dropping one would permanently diverge the soft-state index.

Deadline shedding is logged at `INFO`; server admission rejection is logged at
`WARN`. Each rejection class reports totals 1, 2, 4, 8, and so on, making the
first overload visible at the default log level without log volume growing
linearly with sustained overload.

## Bridge configuration

Required or commonly used bridge variables:

- `KV_INDEXER_WORKERS_JSON`: fleet-mode array shown above; each entry requires
  `worker_id`, `worker_address`, `event_endpoint`, and `snapshot_endpoint`, and
  optionally accepts `event_topic` and `dp_rank`
- `KV_INDEXER_ENDPOINT`: paired Indexer endpoint, default
  `http://[::1]:50051`

The variables below are the backward-compatible, non-recoverable single-Worker
mode. They are ignored when `KV_INDEXER_WORKERS_JSON` is present:

- `KV_INDEXER_WORKER_ID`: unique ID for the worker event stream
- `KV_INDEXER_WORKER_ADDRESS`: Router-facing worker URL
- `SGLANG_KV_EVENT_ENDPOINT`: worker PUB endpoint
- `SGLANG_KV_EVENT_TOPIC`: ZMQ subscription topic
- `KV_INDEXER_CLEAR_TIERS`: tiers affected by clear, default `HBM,DRAM,SSD`
- `KV_INDEXER_CACHE_COMPONENTS`: optional `full,swa` or `full,mamba`
- `KV_INDEXER_SWA_WINDOW_TOKENS`: required when SWA is configured
- `KV_INDEXER_FULL_TIERS`, `KV_INDEXER_SWA_TIERS`,
  `KV_INDEXER_MAMBA_TIERS`: servable component tiers
- `KV_INDEXER_CACHE_SPEC_VERSION`: component-rule version, default `1`

## Tests

No external service is needed:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```
