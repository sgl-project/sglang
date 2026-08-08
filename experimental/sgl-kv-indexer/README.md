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
kvidx:{<hash>}:p          HASH  "<worker_id>\x1f<tier>" -> component bitmask
                                "\x00sz" -> block token count
kvidx:{<hash>}:h          HASH  c (hit count), ls (last seen ms)
kvidx:{w:<worker>}:blocks SET   block hashes this worker holds
kvidx:{w:<worker>}:meta   HASH  addr (router-facing URL),
                                spec (encoded WorkerCacheSpec)
```

Each placement field identifies one `(worker, tier)` pair; its value is a
component mask (`FULL=1`, `SWA=2`, `MAMBA=4`, and `0` for legacy whole-block
events). Tier masks (`1 << tier`) are stored separately inside
`WorkerCacheSpec`. The hash tags keep placement co-located with its hit counter,
and per-worker keys co-located with each other, so every Lua script stays within
one cluster slot.

## Build

```bash
cd experimental/sgl-kv-indexer
cargo build --release --features redis-backend
```

This produces:

- `target/release/kv-indexer-server`
- `target/release/kv-indexer-bridge`

The `redis-backend` feature is required to run the Redis-backed server.

## End-to-end quickstart

Component-aware routing requires an SGLang engine build that supports
`component_types`. Start each command in its own terminal.

1. Start Redis:

```bash
redis-server --port 6379
```

2. Start the indexer:

```bash
KV_INDEXER_BACKEND=redis \
KV_INDEXER_REDIS_URL=redis://127.0.0.1:6379 \
KV_INDEXER_LISTEN_ADDR=127.0.0.1:50051 \
  cargo run --release --features redis-backend --bin kv-indexer-server
```

3. Start one bridge per worker event stream. This FULL+SWA example uses the
worker URL that the router will register; set the window to the model's value:

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

For FULL+MAMBA, replace the component-specific variables with:

```bash
KV_INDEXER_CACHE_COMPONENTS=full,mamba
KV_INDEXER_FULL_TIERS=HBM
KV_INDEXER_MAMBA_TIERS=HBM
```

4. Start the matching SGLang worker. Component types are gated and off by
default:

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --port 30000 \
  --kv-events-config \
    '{"publisher":"zmq","endpoint":"tcp://*:5567","topic":"kv-events"}' \
  --enable-kv-events-component-types
```

5. Start the router with the indexer as the preferred cache signal:

```bash
sgl-router \
  --model-id <model-id> \
  --tokenizer-path <huggingface-repo-or-tokenizer> \
  --worker-urls http://127.0.0.1:30000 \
  --policy cache_aware_zmq \
  --kv-indexer-endpoint http://127.0.0.1:50051
```

For multiple workers, repeat steps 3–4 with a unique worker ID, HTTP port, and
ZMQ port, then include every HTTP URL in `--worker-urls`. Each
`KV_INDEXER_WORKER_ADDRESS` must match its router URL byte-for-byte.

The bridge stores component snapshots, `MatchExternalKvPrefix` converts them to
an effective reusable prefix, and the router consumes that prefix without
implementing SWA or MAMBA rules. RPC failures fall back to the router's local
cache-aware policy.

### Verify and troubleshoot

- Prefill a unique prompt directly on one worker, then send it through the
  router; the router should select that worker.
- Missing `--enable-kv-events-component-types` produces legacy whole-block
  events, so FULL/SWA/MAMBA cannot be distinguished.
- A wrong SWA window or cache-component list produces incorrect effective
  prefixes; copy both from the model/cache configuration.
- If the router gets no external signal, first compare
  `KV_INDEXER_WORKER_ADDRESS` with the registered worker URL.
- For chat requests, use a tokenizer configuration with the same chat template
  as the engine; a bare `tokenizer.json` may not contain it.

For non-component bring-up, omit the component variables and gated engine flag.
`KV_INDEXER_BACKEND=logging` can replace Redis with an in-memory debug backend.

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
| `KV_INDEXER_WORKER_ADDRESS` | empty | Router-facing worker URL; must exactly match the URL registered by the router |
| `KV_INDEXER_ENDPOINT` | `http://[::1]:50051` | Indexer gRPC endpoint |
| `SGLANG_KV_EVENT_ENDPOINT` | `tcp://127.0.0.1:5557` | SGLang event PUB endpoint |
| `SGLANG_KV_EVENT_TOPIC` | empty | ZMQ subscription topic |
| `KV_INDEXER_CLEAR_TIERS` | `HBM,DRAM,SSD` | Tiers affected by `AllBlocksCleared` |
| `KV_INDEXER_CACHE_COMPONENTS` | unset (legacy) | Comma-separated `full`, `swa`, and/or `mamba`; FULL is always included when set |
| `KV_INDEXER_SWA_WINDOW_TOKENS` | `0` | Required and greater than zero when `swa` is configured |
| `KV_INDEXER_FULL_TIERS` | `HBM,DRAM` | Servable tiers for FULL (`HBM`/`GPU`, `DRAM`/`CPU`/`CPU_PINNED`, `SSD`/`DISK`) |
| `KV_INDEXER_SWA_TIERS` | `HBM,DRAM` | Servable tiers for SWA, using the same aliases |
| `KV_INDEXER_MAMBA_TIERS` | `HBM,DRAM` | Servable tiers for MAMBA, using the same aliases |
| `KV_INDEXER_CACHE_SPEC_VERSION` | `1` | Component-rule version; versions newer than the server supports fail closed |

The bridge builds its `WorkerCacheSpec` once at startup and sends it with every
apply batch. Omitting `KV_INDEXER_CACHE_COMPONENTS` clears any previously stored
spec and preserves legacy whole-block matching.

## API

The protobuf service in `proto/kv_indexer.proto` provides:

- `ApplyExternalKvBatch`: ordered placement reports, revocations, and clears.
  The response is empty; the request `seq` is carried for observability only.
- `MatchExternalKv`: workers and tiers holding requested block hashes.
- `MatchExternalKvPrefix`: per-worker longest **contiguous** request prefix, for
  cache-aware routing (see below).
- `GetExternalKvHitCounts`: per-block hit counters.

There is no gRPC health service in this build.

## Prefix routing query

`MatchExternalKvPrefix` answers, for a request's block-hash chain (prompt order,
`hashes[0]` first), how much prefix each worker can actually reuse.

For a legacy worker, `matched_prefix_blocks` is the largest `n` such that it
holds every block in `hashes[0..n)` with no gap. For a component-aware worker,
the stored `WorkerCacheSpec` additionally applies the engine's fixed rules:

- FULL must be contiguous on every matched block.
- SWA, when configured, must cover the trailing `swa_window_tokens` at the
  candidate boundary (or form an unbroken run from the prompt head).
- MAMBA, when configured, must be present on the candidate boundary block.

Component placements without a worker spec fail closed instead of being treated
as legacy whole-block placements. A batch without a spec clears any older stored
spec, preventing stale component rules from carrying across worker changes.

The indexer does **not** pick a worker. It cannot see the router's health checks,
circuit breakers, PD-pool split, or in-flight load, so it returns every candidate
sorted by prefix length and leaves the final choice to the router (intersect the
cache-hit set with the router's own candidates, then pick by lowest load). It also
does not return tier information: routing selects on prefix length alone.

Prefix queries are capped at 2,048 blocks and at most 16 execute concurrently.
The first block is read separately: a miss returns immediately with
`blocks_read=1`. After a first-block hit, the indexer reads placement for every
block in the bounded prefix (in chunks of 256), loads candidate worker address
and spec metadata, and computes component-aware prefix lengths in memory.
Consequently, the fast path saves work only on a first-block miss; a hit costs
O(prefix blocks) placement reads plus O(candidate workers) metadata reads.

Because placement and worker metadata are read as an advisory snapshot, and this
basic build never fences restarted workers, a reported prefix can be longer than
a worker truly holds. The router must continue to intersect results with its own
healthy candidates and retain its normal fallback behavior.

### Worker address contract

`ExternalKvPrefixMatch.worker_address` is the worker's **router-facing routing
identity, not its KV-transfer address**. The router intersects it byte-for-byte
with the worker URLs it registered, so a mismatch makes the intersection always
empty and silently disables cache-aware routing — a failure that is hard to
diagnose. It is populated from the bridge's `KV_INDEXER_WORKER_ADDRESS`; set it to
exactly the URL the router registers. Workers with an empty address are unroutable
and are excluded from prefix results.

### Router client

`src/client.rs` is the minimal library the router links against: one trait
(`PrefixIndex::match_prefix`) taking an ordered `Vec<i64>` and returning
`PrefixOutcome`. The outcome has **no error variant** — every failure (empty
result, unreachable, timeout, rejected) becomes `NoSignal`, so an advisory-index
outage falls back to existing routing instead of failing a request. The connection
is lazy (the router does not depend on the indexer at startup) and each query has
its own short deadline (default 10 ms).

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
