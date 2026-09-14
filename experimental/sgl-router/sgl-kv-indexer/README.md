# Recoverable KV Placement Replicas

Each Indexer is an active-active, full in-memory placement replica. It stores
metadata, not tensors. One Bridge serves one Indexer and discovers all DP streams
of every configured Worker. Replicas recover independently; there is no leader,
durable database, hash sharding, or inter-Indexer replication.

```text
Workers -- Snapshot v2 + Replay v2 + Live --> Bridge 1 --> Indexer 1
        -- Snapshot v2 + Replay v2 + Live --> Bridge 2 --> Indexer 2
                                                         ^
Routers ---------------- random complete-coverage query --+
Routers -- inference / reporting registration --> Workers
Routers <-- independent leased gRPC load streams -- Workers
```

The Router's `indexer` provider does not subscribe to KV events or retain
placement. Empty shared policy infrastructure remains for compatibility; matching
uses only the new `KVReplica` service. Other legacy providers are unchanged.

## Build and protocol

From `experimental/sgl-router`, run `cargo build --release --bins`. Binaries are
`target/release/{sgl-router,kv-indexer-server,kv-indexer-bridge}`. Workers need this
branch's Python sources and declared grpcio/Protobuf 6 dependencies.
`load_report_pb2.py` is generated from `proto/load_report.proto` using
`grpcio-tools==1.78.0`.

The new `KVReplica` API in `proto/kv_indexer.proto` provides:

- `BeginSnapshot` / `SnapshotChunk`: owner-fenced, hidden per-stream staging.
- `ApplyLive`: original Worker epoch/sequence; duplicates ignored, gaps revoke
  READY, epoch changes require a new snapshot.
- `ConfirmStream`: validate the snapshot cut/replay cursor, atomically install
  complete staging and renew the lease.
- `InvalidateStream` / `RemoveStream`: invalidate only the owning stream.
- `MatchPrefix`: matches and explicit readiness, identity/schema, epoch and
  watermark per stream. A synchronized miss is not NOT_READY.

The old `KvIndexer` service and environment-only legacy Bridge remain available
with separate state. They cannot mutate recoverable replicas. Use
`KV_BRIDGE_CONFIG` and the new Router together; incompatible versions fail closed
to load-only routing instead of claiming complete coverage.

## Two-Worker, two-replica quickstart

Run long-running commands in separate terminals. Worker IDs must be stable and
unique per namespace. Use one served model name throughout. Each ZMQ base port
reserves consecutive ports for DP ranks; PUB/snapshot/replay/legacy-load ranges
must not overlap.

Worker A (change ID, HTTP port and all ZMQ ports for Worker B):

```bash
python -m sglang.launch_server --model-path <model> --served-model-name model \
  --host 0.0.0.0 --port 30000 \
  --kv-events-config '{"publisher":"zmq","worker_id":"worker-a","namespace":"default","endpoint":"tcp://*:5567","snapshot_endpoint":"tcp://*:5767","replay_endpoint":"tcp://*:5967","topic":"kv"}'
```

Worker B can use HTTP `30001`, PUB `6567`, snapshot `6767`, replay `6967` and
`worker_id=worker-b`. Scheduler supplies actual page size, model, bigram mode and
supported Unified Cache component spec. `/server_info` advertises discovery.

```bash
KV_INDEXER_LISTEN_ADDR=127.0.0.1:50051 target/release/kv-indexer-server
```

```bash
KV_INDEXER_LISTEN_ADDR=127.0.0.1:50052 target/release/kv-indexer-server
```

Create `bridge-1.json`:

```json
{
  "indexer_endpoint": "http://127.0.0.1:50051",
  "worker_urls": ["http://127.0.0.1:30000", "http://127.0.0.1:30001"],
  "queue_capacity": 256,
  "snapshot_concurrency": 4
}
```

Create `bridge-2.json` with the same Workers and Indexer port `50052`.

```bash
KV_BRIDGE_CONFIG=bridge-1.json target/release/kv-indexer-bridge
```

```bash
KV_BRIDGE_CONFIG=bridge-2.json target/release/kv-indexer-bridge
```

Create `indexers.json`:

```json
["http://127.0.0.1:50051", "http://127.0.0.1:50052"]
```

Router tokenizer path must be the actual `tokenizer.json` file:

```bash
SGL_ROUTER_LOAD_LISTEN_ADDR=0.0.0.0:50061 \
SGL_ROUTER_LOAD_ADVERTISE_URL=http://127.0.0.1:50061 \
target/release/sgl-router --model-id model --tokenizer-path /path/to/tokenizer.json \
  --worker-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --policy cache_aware --cache-prefix-provider indexer \
  --kv-indexer-endpoint @indexers.json --port 8000
```

A comma-separated endpoint list also works but is static. Additional Routers need
independent reachable Load Monitor ports/addresses. Router registers and renews
via `POST /v1/start_reporting`; Worker opens an independent gRPC stream to each
Router. Do not advertise wildcard or container-local addresses to remote Workers.

## Recovery, failures and horizontal scaling

Bridge subscribes before snapshot capture, buffers bounded events, checks the
Worker-generated live barrier, forwards chunks to hidden staging, and replays to
the Worker watermark before READY. Periodic replay repairs a lost final event
even without later PUB traffic. Short gaps replay; expired replay, changed epoch,
Bridge failure or overflow triggers full recovery. A failing pair cannot block
the Worker or another pair.

Indexer keeps `(namespace, hash) -> (worker, rank, tier) -> placement`, reverse
holdings and recovery state. Replacing one stream cannot erase another Worker's
copy of a shared hash. Partial recovery is not queryable.

Router shuffles endpoints per query, requires exact eligible-stream coverage and
current load generations, and retries failures/partial responses within one total
deadline. An HTTP Worker with multiple DP ranks uses the minimum safe prefix.
All replicas down means empty cache affinity: fresh load, health/breakers and
local request tracking still route inference. Stale reports do not enter normal
scoring; existing local-load policy is the last resort without fresh reports.

- Replica expansion: start Indexer and its Bridge with the full Worker list, then
  add endpoint to each Router discovery file. Partial coverage is rejected until
  recovery finishes. Remove endpoints before stopping pairs for scale-in.
- Worker expansion/removal: update every Bridge's `worker_urls` and Router's
  existing Worker registry. Bridge reloads every second; Router refreshes every
  two seconds. Removed stream leases expire and soft state is reclaimed.
- Router expansion/removal: changes independent reporting leases only, without
  adding KV subscriptions or requesting snapshots.

Update files atomically. Invalid JSON preserves the last valid configuration;
`[]` deliberately disables replica affinity. Bridge endpoint/queue-limit changes
require restart, while Worker list changes are hot-reloaded. Transient Worker
discovery errors do not erase its desired configuration.

## Bounds, compatibility and observability

Defaults: 256 live messages and 32 MiB per stream, 8 MiB per wire message, four
concurrent snapshot recoveries per pair, 4096 records/chunk, ten million snapshot
records/stream. Startup/retry jitter reduces snapshot bursts. Worker replay uses
nonblocking bounded replies; a slow reader may need a snapshot. Placement memory
still scales with current holdings and replica count.

`KV_INDEXER_STREAM_LEASE_MS` defaults to 5000. Missing renewal hides placement at
expiry; after two lease intervals soft state is reclaimed. Bridge probes every
500 ms: lease must exceed normal probe/transport latency.
`KV_INDEXER_PREFIX_QUERY_MAX_INFLIGHT` defaults to 32 and sheds excess requests.
Router excludes load older than two seconds; Reporter never relabels an old
scheduler measurement as fresh.

Snapshot v2 retains parent hash, block size, HBM/DRAM/SSD and FULL/SWA/MAMBA masks.
Native Python Unified Cache emits partial component eviction, host/load-back and
Mamba boundary updates. Unsupported specs (C128, hybrid alternative cores lacking
this contract) are rejected, not interpreted as whole-block hits. SSD metadata is
retained; existing prefix candidates use locally usable HBM/DRAM only. Snapshot
v1 remains available but cannot represent complete hybrid placement.

An explicit empty `component_types` list revokes placement at that tier in both
Snapshot v2 and live recovery. Omitted/nil component types retain legacy
whole-block semantics. Recoverable Bridges reject unknown event tags and
component labels instead of advancing the stream past an unhandled mutation.
Workers advertise snapshot versions only with a routable snapshot endpoint.

Router `/metrics` exports:

- `sgl_router_indexer_complete_queries_total`
- `sgl_router_indexer_fallback_queries_total`
- `sgl_router_indexer_failed_attempts_total`

`RUST_LOG=info,sgl_kv_indexer::fleet=debug` logs accepted endpoints. Bridge logs
READY, replay failure, overflow and recovery. Deploy internal endpoints on a
trusted network with access controls: this change does not add transport TLS or
tenant authentication. Tokens fence leases/generations, not an untrusted network.

## Tests

Install project/test dependencies and `grpcio-tools==1.78.0`. From repository root:

```bash
cargo test --manifest-path experimental/sgl-router/Cargo.toml --workspace --tests
cargo build --manifest-path experimental/sgl-router/Cargo.toml --bins
PYTHONPATH=python .venv/bin/python -m pytest \
  experimental/sgl-router/sgl-kv-indexer/tests/test_replica_processes.py -q
KV_REPLICA_GPU_TESTS=1 PYTHONPATH=python .venv/bin/python -m pytest \
  experimental/sgl-router/sgl-kv-indexer/tests/test_replica_processes.py \
  -k real_two_gpu -q
```

Default process tests use production publishers/reporters, real Rust processes
and synthetic HTTP inference responses. The opt-in test runs two actual GPU
SGLang Workers (GPUs 0/1), tiny randomly initialized Qwen2, a Router and two pairs,
covering failure/recovery/scaling without a model download. This is functional,
not quality/throughput validation. Cleanup targets only test-owned process groups;
logs/metrics remain in pytest's temporary directory.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for requirements, measured results,
commit breakdown and limitations.
See [SNAPSHOT_V2_VALIDATION.md](SNAPSHOT_V2_VALIDATION.md) for the September 14
source synchronization, protocol corrections and fresh validation results.
