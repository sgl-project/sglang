# SGL KV Indexer (basic build)

`sgl-kv-indexer` is an experimental shared metadata index for SGLang KV-cache
blocks. It records which worker and storage tier currently holds each
content-addressed block, allowing a router to query likely cache hits without
moving KV data itself.

> **This is the basic build**, kept deliberately small for bringing up and
> debugging the SGLang -> bridge -> indexer -> Redis chain. It has no fault
> tolerance: see [What this build does not do](#what-this-build-does-not-do).
> The full build adds sequence gating, incarnation fencing, replay recovery and
> worker liveness.

## Architecture

```text
SGLang worker ── ZMQ PUB ──> bridge ── gRPC ──> indexer ──> Redis
```

- SGLang publishes `BlockStored`, `BlockRemoved`, and `AllBlocksCleared` events.
- One `kv-indexer-bridge` follows each independent worker/rank event stream.
- `kv-indexer-server` applies event batches in order and serves match queries.
- Redis, Dragonfly, or Redis Cluster stores the placement metadata.

## What this build does not do

Every apply is unconditional. There is no per-worker sequence gate, no
incarnation or generation fencing, no restart reset, no replay of missed
batches, and no worker liveness TTL. Concretely:

| If this happens | The result is |
| --- | --- |
| A worker restarts | Its previous placement entries stay in Redis and `match` keeps returning them |
| A worker dies | It stays in `match` results indefinitely |
| The bridge is disconnected | Events published during the outage are lost, with no catch-up |
| The publisher's sequence jumps | The gap is logged and otherwise ignored |
| A batch is redelivered or reordered | It is applied again, in arrival order |
| A multi-slot apply partially fails on Redis Cluster | Nothing repairs the partial write |

Individual mutations are still idempotent (bit set/clear, `SADD`/`SREM`) and each
block-hash mutation is atomic on its own cluster slot, so re-delivering an
identical batch converges. What is missing is everything that detects and repairs
the cases above.

Two conveniences are kept because they materially affect debugging, even though
both are arguably resilience features:

- The bridge reconnects with backoff, so it can be started before SGLang and
  survives an indexer restart. It recovers the *connection* only, never the data.
- Redis connects and PINGs at startup and every operation is bounded by a
  timeout, so a misconfigured store fails loudly instead of hanging.

Redis Cluster support is retained in full, including explicit `ASKING` handling
for slot migration, since that is a correctness requirement rather than a
resilience feature.

## Data model

```text
kvidx:{<hash>}:p         HASH  worker_id -> tier bitmask
kvidx:{<hash>}:h         HASH  c (hit count), ls (last seen ms)
kvidx:{w:<worker>}:blocks SET  block hashes this worker holds
kvidx:{w:<worker>}:meta   HASH addr
```

A tier is one bit: `bit = 1 << tier` (HBM=1, DRAM=2, SSD=3). The hash tags keep
placement co-located with its hit counter, and per-worker keys co-located with
each other, so every Lua script stays within one cluster slot.

## Build

```bash
cd experimental/sgl-kv-indexer
cargo build --release --features redis-backend
```

This produces:

- `target/release/kv-indexer-server`
- `target/release/kv-indexer-bridge`

The `redis-backend` feature is required to run the Redis-backed server.

## Run locally

Start Redis:

```bash
redis-server --port 6379
```

Start the indexer:

```bash
KV_INDEXER_BACKEND=redis \
KV_INDEXER_REDIS_URL=redis://127.0.0.1:6379 \
KV_INDEXER_LISTEN_ADDR=127.0.0.1:50051 \
RUST_LOG=info \
  cargo run --release --features redis-backend --bin kv-indexer-server
```

For chain bring-up before Redis is involved, `KV_INDEXER_BACKEND=logging` runs an
in-memory backend that logs running block totals on every apply.

Start one bridge for a SGLang worker:

```bash
KV_INDEXER_WORKER_ID=worker-0 \
KV_INDEXER_WORKER_ADDRESS=http://127.0.0.1:30000 \
KV_INDEXER_ENDPOINT=http://127.0.0.1:50051 \
SGLANG_KV_EVENT_ENDPOINT=tcp://127.0.0.1:5567 \
RUST_LOG=info \
  cargo run --release --bin kv-indexer-bridge
```

The corresponding SGLang server must publish KV events at that endpoint, for
example with a `kv-events-config` containing a ZMQ publisher endpoint. The replay
endpoint is not used by this build.

Deploy one bridge per independent SGLang KV-event stream (for example, per DP
rank), each with a unique `KV_INDEXER_WORKER_ID`.

## Configuration

### Indexer server

| Variable | Default | Description |
| --- | --- | --- |
| `KV_INDEXER_LISTEN_ADDR` | `[::1]:50051` | gRPC listen address |
| `KV_INDEXER_BACKEND` | required | `redis`, or `logging` for an in-memory debug backend |
| `KV_INDEXER_REDIS_URL` | none | Single Redis/Dragonfly URL |
| `KV_INDEXER_REDIS_CLUSTER_NODES` | none | Comma-separated Redis Cluster seed URLs; takes precedence over the single URL |
| `KV_INDEXER_REDIS_NAMESPACE` | `kvidx` | Redis key prefix |

### Bridge

| Variable | Default | Description |
| --- | --- | --- |
| `KV_INDEXER_WORKER_ID` | required | Unique ID for this worker event stream |
| `KV_INDEXER_WORKER_ADDRESS` | empty | Address returned to match callers |
| `KV_INDEXER_ENDPOINT` | `http://[::1]:50051` | Indexer gRPC endpoint |
| `SGLANG_KV_EVENT_ENDPOINT` | `tcp://127.0.0.1:5557` | SGLang event PUB endpoint |
| `SGLANG_KV_EVENT_TOPIC` | empty | ZMQ subscription topic |
| `KV_INDEXER_CLEAR_TIERS` | `HBM,DRAM,SSD` | Tiers affected by `AllBlocksCleared` |

## API

The protobuf service in `proto/kv_indexer.proto` provides:

- `ApplyExternalKvBatch`: ordered placement reports, revocations, and clears.
  The response is empty; the request `seq` is carried for observability only.
- `MatchExternalKv`: workers and tiers holding requested block hashes.
- `GetExternalKvHitCounts`: per-block hit counters.

There is no gRPC health service in this build.

## Tests

Single Redis:

```bash
KV_INDEXER_REDIS_URL=redis://127.0.0.1:6379 \
  cargo test --features redis-backend
```

Tests skip themselves when no store is configured, so a plain
`cargo test --features redis-backend` stays green without Redis.

Static checks:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features redis-backend -- -D warnings
```
