# SGL KV Indexer (in-memory build)

`sgl-kv-indexer` is an experimental metadata service for SGLang KV-cache
blocks. It records which worker and storage tier currently holds each
content-addressed block, allowing a router to query likely cache hits without
moving KV data itself.

The Indexer's storage is pluggable behind the `KvIndexerBackend` trait. This
build ships two backends: a process-local in-memory index (the default) and a
[Tair KVCache Manager](https://github.com/alibaba/tair-kvcache) (KVCM) backend
that delegates placement state to a KVCM MetaService. See
[The KVCM backend](#the-kvcm-backend).

The default in-memory build has no external storage dependency, but it is
soft-state: restarting the Indexer loses all placement metadata.

## Architecture

```text
SGLang worker ── ZMQ PUB ──> bridge ── gRPC ──> indexer ── backend
                                                        ├─ in-memory index
                                                        └─ KVCM MetaService
```

- SGLang publishes `BlockStored`, `BlockRemoved`, and `AllBlocksCleared` events.
- One `kv-indexer-bridge` follows each independent worker/rank event stream.
- One `kv-indexer-server` applies event batches and serves match queries; the
  selected backend stores the placements.
- For LoRA, the bridge translates worker block IDs into lookup keys derived
  from tokens, `lora_name`, and `cache_salt`. It tracks each external ID and tier
  independently so removing one adapter load does not revoke another.
- The in-memory backend keeps placement, worker metadata, reverse holdings,
  and hit counters in the server process behind a single read/write lock.

Each apply RPC is ordered and atomic within the in-memory backend, and its
queries see a consistent snapshot. The bridge splits larger worker event
batches into ordered RPCs of at most 16,384 hashes and 256 actions while
keeping per-hash metadata aligned; the KVCM backend may further split one RPC
into several `ReportEvent` calls under a single deadline. If a later RPC
fails, earlier chunks may already be applied; there is no rollback or replay.
There is no state sharing between Indexer servers.

## Operational contract

Run exactly one Indexer server for a deployment. Multiple bridge processes and
workers may report to it, but active-active Indexer servers have independent
state and must not be treated as replicas.

The in-memory backend has no sequence gate, incarnation fencing, replay
recovery, worker liveness TTL, or restart recovery:

- An Indexer restart starts with an empty index.
- A worker death is not detected; its last placements remain until revoked or
  until the Indexer restarts.
- Events published while a bridge is disconnected are not replayed.
- A publisher sequence gap is logged and otherwise ignored.
- Redelivered or reordered batches are applied again in arrival order.

Individual report, revoke, and clear mutations are idempotent. A future
Snapshot plus event-replay mechanism is required before production high
availability can rebuild state safely after restart or event loss.

The KVCM backend inherits the same bridge semantics (no replay, applies
unconditional). KVCM owns persistence and capacity eviction. Reporter liveness
is deliberately *not* used for reclaim: the backend heartbeats every worker it
has seen for its own lifetime, so an idle worker keeps its cache affinity, and
a dead worker's placements linger exactly like the in-memory backend's (the
Router's health intersection keeps them unroutable) until KVCM's capacity
eviction removes them.

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
`KV_INDEXER_BACKEND` selects the storage backend: `memory` (default) or
`kvcm` (see [The KVCM backend](#the-kvcm-backend)).

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

## The KVCM backend

`KV_INDEXER_BACKEND=kvcm` selects `KvcmIndexerBackend`, which stores placements
in a [Tair KVCache Manager](https://github.com/alibaba/tair-kvcache) (KVCM)
MetaService instead of the server process. The wire contract is the vendored
`proto/kvcm_meta_service.proto` (`kv_cache_manager.proto.meta.MetaService`).

```bash
KV_INDEXER_BACKEND=kvcm \
KV_INDEXER_LISTEN_ADDR=127.0.0.1:50051 \
KV_INDEXER_KVCM_ENDPOINT=http://kvcm:50052 \
KV_INDEXER_KVCM_INSTANCE_ID=my-model-deployment \
  cargo run --release --bin kv-indexer-server
```

Translation, both directions:

- `ApplyExternalKvBatch` → `ReportEvent`: the worker id becomes the KVCM
  reporter id, REPORT becomes `BLOCK_ADD` (one spec per resident component,
  legacy whole blocks report `full` alone), REVOKE becomes `BLOCK_DELETE`
  (every component name, a superset removal), and the first batch per worker
  prepends `NODE_REGISTER`. Heartbeats keep reporters alive so KVCM liveness
  does not reclaim idle-but-cached workers.
- `MatchExternalKvPrefix` → `GetHostCacheState` (`QT_PREFIX_MATCH`): per host,
  the longest contiguous prefix from block 0 — the same quantity the in-memory
  backend computes. Reporters map back to workers through the address restated
  on every apply batch; workers with no address cannot be routed and are
  excluded.

Configuration variables (all under `KV_INDEXER_KVCM_`):

- `ENDPOINT` (required): MetaService gRPC address, e.g. `http://10.0.0.1:50052`
- `INSTANCE_ID` (required): one KVCM instance per routed model deployment
- `STORAGE_TYPE`: event-report channel, `l1p5` (default) or `l2`, matching what
  the deployment's KVCM admin provisioned
- `REPORT_TIERS`: tiers whose events are forwarded, default `HBM` (see below)
- `AUTO_REGISTER` (default `0`) with `INSTANCE_GROUP`, `BLOCK_SIZE`,
  `MODEL_NAME`, `MODEL_DTYPE`: opt-in `RegisterInstance` before the first batch.
  It declares the `full,swa,mamba` spec names, and restates the block size and
  model deployment KVCM requires of every registration (a registration without
  a model name/dtype, or with a non-positive block size, is rejected
  `INVALID_ARGUMENT`).
- `QUERY_TIMEOUT_MS` (default 50, half the Router-side prefix deadline),
  `APPLY_TIMEOUT_MS` (default 5000, one budget shared by every chunk of a batch)
- `HEARTBEAT_INTERVAL_MS` (default 30000; `0` disables heartbeats)

Deployment prerequisites on the KVCM side: an event-report storage and an
instance group listing it as an `event_report_storage_candidates` entry, and a
registered instance — either by the workers' own KVCM connector (the common
case; leave `AUTO_REGISTER` off, since `RegisterInstance` fails with
`DUPLICATE_ENTITY` on any field mismatch) or by `AUTO_REGISTER=1` in
routing-only deployments, which then also has to carry the deployment's model
name, dtype and block size.

Semantics and limitations:

- Host-dram events are dropped by default: deployments running the KVCM
  connector already manage those placements through the data-plane write path,
  and double reporting would race eviction. `REPORT_TIERS=HBM,DRAM` widens the
  set for routing-only deployments. A report tier outside HBM/DRAM (`SSD`) is
  not servable by the prefix rules: legacy workers still report it, but every
  worker declaring a cache spec is excluded from answers (warned at startup).
- `CLEAR_ALL_AT_TIER` has no KVCM equivalent (a `BLOCK_SNAPSHOT` replaces every
  medium at once). It is skipped with a `WARN`; the worker keeps heartbeating,
  so its stale placements stay indexed until KVCM's capacity eviction removes
  them.
- `MatchExternalKv` (per-block placement) has no KVCM equivalent and answers
  `UNIMPLEMENTED`; `GetExternalKvHitCounts` stays empty. The prefix query is
  the only routing signal.
- The prefix answer is the legacy whole-block quantity: the longest contiguous
  run of held blocks. Workers whose `cache_spec` requires rules beyond that —
  an SWA window or a MAMBA boundary — or that report any block whose component
  detail does not confirm the `full` component (including a zero mask inside a
  component batch, which the bridge produces for mask-less blocks), are
  **excluded from the answer entirely** (fail closed, logged once per worker),
  so the answer stays a lower bound like the in-memory backend's instead of
  over-reporting. Legacy and FULL-only workers are unaffected.
- A worker id must not contain `#` (the KVCM location-id separator); the
  bridge dies on the resulting `INVALID_ARGUMENT`, which is the desired
  fail-fast for a misconfigured fleet.
- The instance the backend reports into must declare the `full`, `swa` and
  `mamba` spec names in its `location_spec_infos` — auto-registration does;
  a connector-registered instance does too if it registered those spec names.
  If it does not, every `BLOCK_ADD` fails `INVALID_ARGUMENT` and the bridge
  exits.
- The reporter→address map is process-local: after an indexer restart, prefix
  answers stay empty until each worker's next apply batch restates its
  address. KVCM's own placements survive; only the join is lost.
- `snapshot_required` is advisory: KVCM sets it while a reporter has no
  committed snapshot version, which is the normal state of a delta-only
  reporter — the request that writes its first delta creates that version — so
  the backend logs it once per reporter lifecycle and never sends a snapshot.
  It holds no placement state, so it could not rebuild one after KVCM-side
  cleanup anyway; until live blocks are re-reported the index under-reports
  (the safe direction).
- Reporter liveness is real and destructive: KVCM marks a reporter unavailable
  after its configured heartbeat timeout and then *deletes* that reporter's
  placements once the cleanup grace has passed, so a deployment whose KVCM
  liveness is tighter than `HEARTBEAT_INTERVAL_MS` loses cache affinity for
  idle workers. After such a cleanup the next plain report fails
  `NODE_NOT_REGISTERED`; the retry re-registers the reporter, but the reclaimed
  blocks stay gone until the worker re-reports them.
- A `NODE_REGISTER` is resent whenever a report fails with
  `NODE_NOT_REGISTERED` (a reporter KVCM forgot) or never succeeded, and each
  heartbeat re-registers in passing. `NODE_NOT_REGISTERED` itself is
  transient: the bridge retries, and the next report self-heals it.
- Permanent configuration faults (`INSTANCE_NOT_EXIST`, `UNSUPPORTED`,
  `REACH_MAX_ENTITY_CAPACITY`, duplicate registration with different fields)
  surface as `FAILED_PRECONDITION`, which the bridge treats as fatal;
  transient KVCM failures surface as `UNAVAILABLE` and the Router degrades to
  load-based routing. KVCM `SERVER_NOT_LEADER` (HA deployments) is
  `UNAVAILABLE` — this backend does not follow leader redirects.

## Tests

No external service is needed:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`tests/kvcm_contract.rs` runs the KVCM backend against an in-process mock
MetaService, including one end-to-end pass behind the real `KVIndexer` service.

## Live KVCM integration

`tests/kvcm_live.rs` runs the backend against a real Tair KVCache Manager. It is
`#[ignore]`d and gated on `KVCM_LIVE_ENDPOINT`, so a normal `cargo test` never
touches it.

`tests/kvcm_live/setup_kvcm.py` builds the manager's own process and
configuration: it starts `kv_cache_manager_bin`, provisions two event-report
storages (L1P5 and L2) and an instance group listing them, and registers an
instance declaring the `full,swa,mamba` spec names. Everything is one command:

```bash
# once, in the tair-kvcache checkout
bazelisk build //kv_cache_manager:kv_cache_manager_bin

# in experimental/sgl-router
python3 sgl-kv-indexer/tests/kvcm_live/setup_kvcm.py start --workdir /tmp/kvcm-live
source /tmp/kvcm-live/kvcm_live.env
cargo test -p sgl-kv-indexer --test kvcm_live -- --ignored
python3 sgl-kv-indexer/tests/kvcm_live/setup_kvcm.py stop --workdir /tmp/kvcm-live
```

`start` is idempotent (`config` re-applies provisioning, `status` reports the
process, `stop` kills it), and the workdir keeps the manager's registry, logs and
a `kvcm_live.env` with the endpoints and instance names the tests read. Pass
`--kvcm-repo` / `--binary` if the checkout or the binary moved.

The harness provisions deliberately tight liveness (2s heartbeat timeout, 2s
cleanup grace) so the reclamation tests finish in seconds; production values are
an order of magnitude larger. That is why every test states its own heartbeat
interval, and why a deployment should keep `HEARTBEAT_INTERVAL_MS` well below
whatever the KVCM admin configured.

The suite pins, among others: the `RegisterInstance` fields KVCM requires, the
`BLOCK_ADD`/`BLOCK_DELETE` spec-name and URI rules, `NODE_REGISTER` resend
idempotency, reporter liveness and the `NODE_NOT_REGISTERED` self-heal, the
reported-medium filter, `snapshot_required` semantics, and the raw
`p2p_host_count = 0` answer (`local`-only, `p2p_1_total_match == local`,
`p2p_1_fetch == 0`).
