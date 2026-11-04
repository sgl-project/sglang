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
- For LoRA, the bridge translates worker block IDs into lookup keys derived
  from tokens, `lora_name`, and `cache_salt`. It tracks each external ID and tier
  independently so removing one adapter load does not revoke another.
- Placement, worker metadata, reverse holdings, and hit counters live in that
  server process behind a single read/write lock.

Each apply RPC is ordered and atomic within the process, and a query sees a
consistent snapshot. The bridge splits larger worker event batches into
ordered RPCs of at most 16,384 hashes and 256 actions while keeping per-hash
metadata aligned. If a later RPC fails, earlier chunks remain committed and the query reports
`pending_seq`/incomplete coverage. Each registered chunk has its own index, so
several RPCs sharing one publisher sequence are all retained. Placement mutation,
source sequence state and query evidence use the same read/write lock.

## Operational contract and recovery capabilities

Run one Indexer server. State is process-local and is lost on restart. The
mutation endpoint is a **trusted administrative interface**: this build adds no
authentication or authorization layer. Isolate it to the bridges and source
lifecycle authority. An epoch string is an opaque identity, never a timestamp or
an ordering key.

| Capability | Guarantee |
| --- | --- |
| Registered sources | Explicit epoch CAS, per-batch/chunk sequence gates, bounded duplicate receipts |
| Sequence/chunk gap | Clear only that worker's registered tiers before accepting later chunk-zero data; coverage stays incomplete |
| Chunk failure | Earlier committed chunks remain visible; no rollback or complete-batch claim |
| Epoch replacement | `RegisterSource` must match `previous_epoch`; retired epochs cannot register again |
| Reset evidence | Only trusted `ReconcileSource(confirmed_empty=true)` at an explicit publisher sequence cut restores completeness |
| Worker death | Trusted `ReconcileSource(confirmed_empty=false)` clears registered tiers and suspends applies |
| Query evidence | Epoch, completed publisher watermark, pending chunk, completeness, native cache watermark and scope-only response |
| Legacy clients | Accepted with `legacy_unknown`; their placements remain advisory |
| Restart/reconnect | Empty/unknown state, no durable replay or automatic complete recovery |

Register one source per independent worker/rank publisher stream, with unique
worker IDs. `RegisterSource(worker_id, source_epoch, previous_epoch, tiers)` is
idempotent for the current epoch and same tier set. Replacing an epoch clears the
union of old and new registered tiers. A stale epoch or mismatched CAS fails
without replacing the current source. Registration alone starts **unknown**;
it does not prove an empty cache or reconstruct events missed before subscribe.
After Indexer restart an expected previous epoch is absent, so replacement CAS
fails until the lifecycle authority explicitly reconciles that restart. Initial
registration without a previous epoch is an explicit authority assertion; it
still grants no complete coverage.

`ReconcileSource` requires the current Indexer instance epoch, source version,
source epoch, a bounded external
`evidence_id`, and `next_seq`. With `confirmed_empty=true`, the authority asserts
that **all registered tiers were empty at the exact publisher sequence cut**;
the backend clears its records and admits contiguous batches from that cut.
The cut must follow all observed batches. Ordinary `CLEAR_ALL_AT_TIER` events
cannot restore completeness after a gap. With `confirmed_empty=false`, the
source is invalidated and suspended. This supports an external worker registry's
death signal; there is no automatic heartbeat or liveness TTL.

Lost PUB events are not replayed. A missing or out-of-order chunk clears the
source domain, and processing can resume at a later chunk-zero boundary while
coverage remains incomplete. Older completed sequences are duplicate historical
replay only within the same registered epoch; they never recreate cleared
placements. Exact recent chunk payloads are fingerprinted for idempotence.
The active and last-completed batch each retain at most 1,024 fingerprints,
without unbounded event history. A very late replay outside that payload window
can only receive a duplicate receipt, never be re-applied.

Source limits default to 4,096 registered sources, 4,096 worker records and
8,192 retired epochs.
Library deployments may use `InMemoryKvIndexerBackend::with_source_limits`.
Retired epoch fences are never silently evicted; exhaustion refuses admission.
Placement bytes are still governed by the existing in-memory metadata deployment
budget, not a new persistent store.

The bridge optionally takes `KV_INDEXER_SOURCE_EPOCH` and
`KV_INDEXER_PREVIOUS_SOURCE_EPOCH`. These identify the controlled **location
publisher lifecycle**, independently of control-operation and feedback epochs.
The epoch must rotate when that worker publisher restarts. Bridge connection
registration uses the configured CAS; mismatch stops rather than replacing a
newer source. A reconnect invalidates coverage and does not call the empty-cache
recovery API. Epoch-less native streams remain compatible and unknown. For those
streams, reconnect/gap detection clears configured tiers before new reports, and
lower sequences are discarded. Silence, a malformed frame with no usable
sequence and worker death require an external invalidation signal; this build
cannot infer a complete live view from an idle PUB socket.

The optional fourth `KVEventBatch` array slot carries native
`cache_watermark={source_id,source_epoch,generation,version}`. The bridge retains
it separately from the ZMQ frame's publisher sequence. It becomes a source
watermark only after the complete batch commits. Native generation changes clear
old placements and mark coverage incomplete. An unexpected native cache source
epoch change clears/suspends the source until trusted reconciliation; epochs are
never selected by arrival time. Old three-slot batches continue to decode.

`MatchExternalKv` and `MatchExternalKvPrefix` return `observation.scope_only=true`
and `full_snapshot=false`; they describe only requested hashes. `complete` refers
to that source's registered tier domain, not every tier or the whole deployment.
A worker missing from metadata is unknown. A partial scope query is never proof
that the full cache was reconstructed. Consumers must check the epoch, required
tiers and completeness before promoting an observation. The existing routing
`PrefixIndex` API retains advisory matching; new orchestration can use
`GrpcPrefixIndex::match_prefix_observed` and
`confirmed_matches_for_tiers(required_tiers)`, or query the execution owner for a
scope-specific observation. Prefix results aggregate tiers without per-block
provenance, so that helper conservatively requires complete coverage of all
three indexer tiers; a partial-tier source uses diagnostic per-tier scope query
metadata instead. Metadata absent from an older server is unknown. Every query and source receipt
also carries a random `indexer_epoch`; compare it by equality when retaining
observations across calls. A response from a retired Indexer instance cannot
restore a newer instance's state. Reconciliation also CAS-checks this epoch, so
a stale reset receipt cannot exploit source-version reuse after restart.

There is no full snapshot RPC, persistent source authority, durable replay,
replication or multi-server failover. These conservative recovery rules reduce
false confirmations; they do not provide full high availability.

## In-memory data model

The server stores:

- block hash → token count and `(worker, tier) → component mask`
- worker → router-facing address, `WorkerCacheSpec`, and reverse holdings by tier
- block hash → cumulative hit count

Component masks are `FULL=1`, `SWA=2`, `MAMBA=4`, and `0` for legacy
whole-block events. `REPORT` replaces the component snapshot for one
`(worker, tier, block)` placement. `REVOKE` removes that placement, and
`CLEAR_ALL_AT_TIER` removes every placement for the worker at that tier.
Optional `session_id` is retained as latest report attribution per placement and
returned aligned with diagnostic hashes. It does not create session claims or
duplicate the physical report; it is not a history of every associated session.

## Build

```bash
cd sgl-router
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
  --policy cache_aware \
  --cache-prefix-provider indexer \
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

- `RegisterSource`: trusted source epoch/tier registration with previous-epoch CAS.
- `ReconcileSource`: explicit empty-cache recovery or source invalidation.
- `ApplyExternalKvBatch`: placement reports, revocations and clears with optional
  source epoch and bounded batch chunk coordinates.
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

- `KV_INDEXER_SOURCE_EPOCH`: optional trusted location-publisher lifecycle ID
- `KV_INDEXER_PREVIOUS_SOURCE_EPOCH`: explicit previous epoch CAS for replacement
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
