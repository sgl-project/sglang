# SGL KV Indexer (in-memory build)

`sgl-kv-indexer` is an experimental metadata service for SGLang KV-cache
blocks. It records which worker and storage tier currently holds each
content-addressed block, allowing a router to query likely cache hits without
moving KV data itself.

This build deliberately uses one process-local in-memory index. It has no
external storage dependency, but it is soft-state: restarting the Indexer loses
all placement metadata.

## Architecture

```text
SGLang worker ── ZMQ PUB ──> bridge ── gRPC ──> in-memory indexer
```

- SGLang publishes `BlockStored`, `BlockRemoved`, and `AllBlocksCleared` events.
- One `kv-indexer-bridge` follows each independent worker/rank event stream.
- One `kv-indexer-server` applies event batches and serves match queries.
- Placement, worker metadata, reverse holdings, and hit counters live in that
  server process behind a single read/write lock.

Each apply RPC is ordered and atomic within the process, and a query sees a
consistent snapshot. The bridge splits larger worker event batches into
ordered RPCs of at most 16,384 hashes and 256 actions while keeping per-hash
metadata aligned. If a later RPC fails, earlier chunks may already be applied;
there is no rollback or replay. There is no persistence, replication, or state
sharing between Indexer servers.

## Operational contract

Run exactly one Indexer server for a deployment. Multiple bridge processes and
workers may report to it, but active-active Indexer servers have independent
state and must not be treated as replicas.

This build has no sequence gate, incarnation fencing, replay recovery, worker
liveness TTL, or restart recovery:

- An Indexer restart starts with an empty index.
- A worker death is not detected; its last placements remain until revoked or
  until the Indexer restarts.
- Events published while a bridge is disconnected are not replayed.
- A publisher sequence gap is logged and otherwise ignored.
- Redelivered or reordered batches are applied again in arrival order.

Individual report, revoke, and clear mutations are idempotent. A future
Snapshot plus event-replay mechanism is required before production high
availability can rebuild state safely after restart or event loss.

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

1. Start the single Indexer server:

```bash
KV_INDEXER_LISTEN_ADDR=127.0.0.1:50051 \
  cargo run --release --bin kv-indexer-server
```

`KV_INDEXER_LISTEN_ADDR` defaults to `[::1]:50051`.
`KV_INDEXER_PREFIX_QUERY_MAX_INFLIGHT` sets the maximum number of prefix
queries executing concurrently and defaults to `32`. Requests above the limit
are rejected immediately with gRPC `RESOURCE_EXHAUSTED`.
There is no backend or storage configuration.

2. Start one bridge per worker event stream. This FULL+SWA example uses the
worker URL registered with the Router:

```bash
KV_INDEXER_WORKER_ID=worker-0 \
KV_INDEXER_WORKER_ADDRESS=http://127.0.0.1:30000 \
KV_INDEXER_ENDPOINT=http://127.0.0.1:50051 \
SGLANG_KV_EVENT_ENDPOINT=tcp://127.0.0.1:5567 \
SGLANG_KV_EVENT_TOPIC=kv-events \
KV_INDEXER_CACHE_COMPONENTS=full,swa \
KV_INDEXER_SWA_WINDOW_TOKENS=<model-window-tokens> \
KV_INDEXER_FULL_TIERS=HBM \
KV_INDEXER_SWA_TIERS=HBM \
  cargo run --release --bin kv-indexer-bridge
```

For FULL+MAMBA, use:

```bash
KV_INDEXER_CACHE_COMPONENTS=full,mamba
KV_INDEXER_FULL_TIERS=HBM
KV_INDEXER_MAMBA_TIERS=HBM
```

3. Start the matching SGLang worker:

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --port 30000 \
  --kv-events-config \
    '{"publisher":"zmq","endpoint":"tcp://*:5567","topic":"kv-events"}' \
  --enable-kv-events-component-types
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

For multiple workers, repeat steps 2–3 with unique worker IDs and ports.
`KV_INDEXER_WORKER_ADDRESS` must exactly match the corresponding Router URL.

The bridge sends its `WorkerCacheSpec` with every batch. Omitting
`KV_INDEXER_CACHE_COMPONENTS` clears any previously stored spec and uses legacy
whole-block matching.

## API

The protobuf service in `proto/kv_indexer.proto` provides:

- `ApplyExternalKvBatch`: ordered placement reports, revocations, and clears.
  The request `seq` is carried for observability only.
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

- `KV_INDEXER_WORKER_ID`: unique ID for the worker event stream
- `KV_INDEXER_WORKER_ADDRESS`: Router-facing worker URL
- `KV_INDEXER_ENDPOINT`: Indexer endpoint, default `http://[::1]:50051`
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
