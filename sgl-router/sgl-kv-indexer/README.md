# SGL KV Indexer (in-memory build)

`sgl-kv-indexer` is an experimental metadata service for SGLang KV-cache
blocks. It records which worker and storage tier currently holds each
content-addressed block, allowing a router to query likely cache hits without
moving KV data itself.

The Indexer's storage is pluggable behind the `KvIndexerBackend` trait. This
build ships two backends: a process-local in-memory index (the default) and a
**Tair KVCache backend** that delegates placement state to a
[Tair KVCache Manager](https://github.com/alibaba/tair-kvcache) (KVCM) `MetaService`.
See [The Tair KVCache backend](#the-tair-kvcache-backend).

The default in-memory build has no external storage dependency, but it is
soft-state: restarting the Indexer loses all placement metadata.

## Architecture

```text
SGLang worker ── ZMQ PUB ──> bridge ── gRPC ──> indexer ── backend
                                                        ├─ in-memory index
                                                        └─ Tair KVCache MetaService
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
keeping per-hash metadata aligned; the Tair KVCache backend may further split one RPC
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

The Tair KVCache backend inherits the same bridge semantics (no replay, applies
unconditional). Tair KVCache owns persistence and capacity eviction. Reporter liveness
is deliberately *not* used for reclaim: the backend heartbeats every worker it
has seen for its own lifetime, so an idle worker keeps its cache affinity, and
a dead worker's placements linger exactly like the in-memory backend's (the
Router's health intersection keeps them unroutable) until Tair KVCache's capacity
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
`KV_INDEXER_BACKEND` selects the storage backend: `memory` (default) or
`tair_kvcache_kvcm` (see [The Tair KVCache backend](#the-tair-kvcache-backend)).

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

## The Tair KVCache backend

`KV_INDEXER_BACKEND=tair_kvcache_kvcm` selects `TairKVCacheIndexerBackend`, which keeps
placements in a [Tair KVCache Manager](https://github.com/alibaba/tair-kvcache)
(KVCM) `MetaService` instead of the server process, over the vendored
`proto/tair_kvcache_meta_service.proto` (`kv_cache_manager.proto.meta.MetaService`).

### Prerequisites and setup

The backend is compiled into `kv-indexer-server`; a KVCM is required only under
`KV_INDEXER_BACKEND=tair_kvcache_kvcm`, built from `main` at `5246532b` or
later, the commit that added `GetHostCacheState`. An older manager answers
`UNIMPLEMENTED`.

Build and start a single-node manager
([deploy](https://github.com/alibaba/tair-kvcache/blob/main/docs/deploy/README.md)):

```bash
git clone https://github.com/alibaba/tair-kvcache
cd tair-kvcache
bazelisk build //kv_cache_manager:kv_cache_manager_bin
mkdir -p ~/kvcm
./bazel-bin/kv_cache_manager/kv_cache_manager_bin \
  --env=kvcm.service.rpc_port=50052 \
  --env=kvcm.service.http_port=50053 \
  --env=kvcm.service.admin_http_port=50054 \
  --env=kvcm.registry_storage.uri=local://$HOME/kvcm/registry.json
```

`rpc_port` is the gRPC port `ENDPOINT` points at, `http_port` serves the
`MetaService` HTTP endpoints used to verify below, and `admin_http_port` serves
provisioning. `local://` keeps the registry in one JSON file whose parent
directory must exist. A port already in use logs `bind port: ... Address already
in use` and the manager runs on without it.

Provision an event-report storage for each channel the backend reports on
(`ST_EVENT_REPORT_L1P5` for the default `STORAGE_TYPE=l1p5`,
`ST_EVENT_REPORT_L2` for `l2`):

```bash
curl -sS -X POST http://127.0.0.1:50054/api/addStorage -H 'Content-Type: application/json' -d '{
  "trace_id": "indexer-setup",
  "storage": {
    "global_unique_name": "indexer_event_report_l1p5",
    "storage_type": "ST_EVENT_REPORT_L1P5",
    "event_report": {},
    "check_storage_available_when_open": false
  }
}'
```

`event_report` is the oneof member that types the storage; without it the call
fails `INVALID_ARGUMENT {storage_spec}`. `check_storage_available_when_open` is
`false` because an event-report storage holds metadata and has no medium to open.

Then an instance group whose `event_report_storage_candidates` lists that
storage:

```bash
curl -sS -X POST http://127.0.0.1:50054/api/createInstanceGroup -H 'Content-Type: application/json' -d '{
  "trace_id": "indexer-setup",
  "instance_group": {
    "name": "my-deployment-group",
    "storage_candidates": ["data_plane_storage"],
    "event_report_storage_candidates": ["indexer_event_report_l1p5"],
    "global_quota_group_name": "default_quota_group",
    "cache_config": {
      "reclaim_strategy": {"storage_unique_name": "data_plane_storage"},
      "data_storage_strategy": "CPS_PREFER_3FS",
      "meta_indexer_config": {
        "max_key_count": 1000000,
        "mutex_shard_num": 16,
        "meta_storage_backend_config": {"storage_type": "local"}
      }
    },
    "version": 1
  }
}'
```

`storage_candidates` and `cache_config` describe the data-plane storage the group
shares with the workers' connector: use the one your workers write to. When the
group already exists, post the same body to `updateInstanceGroup` with
`current_version` from `getInstanceGroup` and `version` one higher.

Then the instance:

```bash
curl -sS -X POST http://127.0.0.1:50054/api/registerInstance -H 'Content-Type: application/json' -d '{
  "trace_id": "indexer-setup",
  "instance_group": "my-deployment-group",
  "instance_id": "my-model-deployment",
  "block_size": 128,
  "model_deployment": {"model_name": "my-model", "dtype": "FP16"},
  "location_spec_infos": [
    {"name": "full", "size": 0},
    {"name": "swa", "size": 0},
    {"name": "mamba", "size": 0}
  ],
  "default_query_type": "QT_PREFIX_MATCH"
}'
```

Skip this when the workers' connector registers the instance itself;
`INSTANCE_ID` names that instance. With `AUTO_REGISTER=1` the server registers it
instead.

Verify the three objects:

```bash
curl -sS http://127.0.0.1:50054/api/healthy
curl -sS -X POST http://127.0.0.1:50053/api/getInstanceInfo -H 'Content-Type: application/json' -d '{"trace_id":"verify","instance_id":"my-model-deployment"}'
curl -sS -X POST http://127.0.0.1:50053/api/getHostCacheState -H 'Content-Type: application/json' -d '{"trace_id":"verify","instance_id":"my-model-deployment","storage_type":"ST_EVENT_REPORT_L1P5","query_type":"QT_PREFIX_MATCH","block_cache_keys":[0]}'
```

The first answers `OK`, the second echoes the group, the block size and the
`full,swa,mamba` specs, and the third lists the reporter's `host_ip_port` once
the server has reported.

Then start the server against it:

```bash
KV_INDEXER_BACKEND=tair_kvcache_kvcm \
KV_INDEXER_LISTEN_ADDR=127.0.0.1:50051 \
KV_INDEXER_TAIR_KVCACHE_KVCM_ENDPOINT=http://127.0.0.1:50052 \
KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_ID=my-model-deployment \
  cargo run --release --bin kv-indexer-server
```

Both directions of the translation:

- `ApplyExternalKvBatch` -> `ReportEvent`: the worker id becomes the reporter
  id, REPORT becomes `BLOCK_ADD` (one spec per resident component; legacy whole
  blocks report `full` alone), REVOKE becomes `BLOCK_DELETE` (a superset
  removal), `CLEAR_ALL_AT_TIER` becomes an empty `BLOCK_SNAPSHOT` (see below),
  and the first batch per worker prepends `NODE_REGISTER`. Heartbeats keep
  reporters alive so liveness does not reclaim idle-but-cached workers.
- `MatchExternalKvPrefix` -> `GetHostCacheState` (`QT_PREFIX_MATCH`): per host,
  the longest contiguous prefix from block 0, the same quantity the in-memory
  backend computes. Reporters map back through the address restated on every
  apply batch; a worker with no address is excluded.

Configuration (all under `KV_INDEXER_TAIR_KVCACHE_KVCM_`):

- `ENDPOINT` and `INSTANCE_ID` (both required): the `MetaService` address and
  the instance (one per routed model deployment).
- `STORAGE_TYPE`: event-report channel the deployment provisioned, `l1p5`
  (default) or `l2`. `REPORT_TIERS` (default `HBM`) lists the forwarded tiers.
- `AUTO_REGISTER` (default `0`) with `INSTANCE_GROUP`, `BLOCK_SIZE`,
  `MODEL_NAME`, `MODEL_DTYPE`: opt-in `RegisterInstance` before the first thing
  this backend sends. It declares the `full,swa,mamba` spec names and restates
  the fields the manager requires; a registration must match an existing
  instance field for field, or it fails `DUPLICATE_ENTITY`.
- `QUERY_TIMEOUT_MS` (default 50, half the Router-side deadline) and
  `APPLY_TIMEOUT_MS` (default 5000, one budget per batch, registration included).
- `HEARTBEAT_INTERVAL_MS` (default 30000; `0` disables heartbeats). Keep it well
  below the manager's EventReport liveness `heartbeat_timeout_ms`, which defaults
  to 30s: with the default storage spec, set `HEARTBEAT_INTERVAL_MS=10000`.

Semantics and limitations:

- Host-dram events are dropped by default (the connector manages those
  placements through the data plane; double reporting would race eviction);
  `REPORT_TIERS=HBM,DRAM` widens the set for routing-only deployments. A tier
  outside HBM/DRAM (`SSD`) is not servable by the prefix rules: legacy workers
  still report it, but every worker declaring a cache spec is excluded from
  answers (warned at startup).
- `CLEAR_ALL_AT_TIER` is applied as an **empty `BLOCK_SNAPSHOT`**, the one
  primitive with clear semantics: it commits synchronously and fences the
  reporter's older locations behind strict visibility, so the prefix answer
  stops claiming them immediately. Its scope is the whole reporter × storage
  channel across **every medium** (no tier granularity); it travels alone (the
  manager rejects a snapshot mixed with deltas) and in its action order. It is
  also rate-limited per (instance, channel, reporter), 30s by default: a flush
  inside that window is dropped with a `WARN` (`flush dropped, placements stay`)
  rather than failing the batch, which would cost the deltas riding with it. The
  first flush switches a delta-only reporter to strict visibility (that is what
  hides the pre-clear blocks), so the indexer must be the only writer of that
  reporter's placements on that channel; a per-medium `CLEAR` remains a
  manager-side ask.
- The prefix answer is the legacy whole-block quantity, so a worker whose
  `cache_spec` needs rules beyond it (an SWA window, a MAMBA boundary) or that
  reports component detail not confirming `full` (including a zero mask inside a
  component batch, which the bridge produces for mask-less blocks) is
  **excluded entirely** (fail closed, logged once per worker): the answer stays
  a lower bound instead of over-reporting. `MatchExternalKv` has no equivalent
  and answers `UNIMPLEMENTED`; `GetExternalKvHitCounts` stays empty.
- A worker id must not contain `#` (the location-id separator) and the instance
  must declare the `full,swa,mamba` spec names: either mistake fails
  `INVALID_ARGUMENT` and the bridge exits, the desired fail-fast.
- The reporter->address map is process-local: after an indexer restart, prefix
  answers stay empty until the next batch restates an address. Placements
  survive; only the join is lost.
- `snapshot_required` is advisory (a reporter with no committed snapshot
  version, the normal state here), so the backend logs it once per lifecycle and
  never sends a snapshot; until live blocks are re-reported the index
  under-reports, which is the safe direction.
- Reporter liveness is destructive: a silent reporter is marked unavailable
  after the heartbeat timeout and its placements are *deleted* once the cleanup
  grace passes. With liveness tighter than `HEARTBEAT_INTERVAL_MS` the beats
  land on a tombstoned node, the blocks are swept, and prefix queries keep
  answering "nothing here" — a silently empty index, not an error. Check those
  two intervals together first when cache affinity looks broken. The next report
  after such a cleanup fails `NODE_NOT_REGISTERED` and the retry re-registers,
  but the reclaimed blocks stay gone until the worker re-reports them.
- A `NODE_REGISTER` is resent whenever a report fails `NODE_NOT_REGISTERED` or
  never succeeded (each heartbeat also re-registers in passing). Permanent
  configuration faults (`INSTANCE_NOT_EXIST`, `UNSUPPORTED`,
  `REACH_MAX_ENTITY_CAPACITY`, a mismatched registration) are
  `FAILED_PRECONDITION`, which the bridge treats as fatal; transient failures
  are `UNAVAILABLE` and the Router degrades to load-based routing.
  `SERVER_NOT_LEADER` (HA) is `UNAVAILABLE`: no leader redirects are followed.

## Tests

No external service is needed:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`tests/tair_kvcache_contract.rs` runs the Tair KVCache backend against an in-process mock
MetaService, including one end-to-end pass behind the real `KVIndexer` service.

## Live Tair KVCache integration

`tests/test_tair_kvcache_live.rs` runs the backend against a real Tair KVCache
Manager. It is `#[ignore]`d and gated on `TAIR_KVCACHE_LIVE_ENDPOINT`, so a
normal `cargo test` never touches it. CI runs it in the `tier-kv-indexer`
job of `pr-test-sgl-router.yml`, which builds the manager at a pinned
commit and executes the harness below; the other backend's tests need no such
job because the in-memory backend has no external dependency.

`tests/test_tair_kvcache_live/setup_tair_kvcache_manager.py` starts
`kv_cache_manager_bin`, provisions two event-report storages (L1P5 and L2) with
an instance group listing them, and registers an instance declaring the
`full,swa,mamba` spec names. It needs a Tair KVCache checkout: `--manager-repo`
points at it (or set `TAIR_KVCACHE_MANAGER_REPO`; an absolute `--binary` needs
neither):

```bash
# once, in the tair-kvcache checkout
bazelisk build //kv_cache_manager:kv_cache_manager_bin

# in sgl-router
python3 sgl-kv-indexer/tests/test_tair_kvcache_live/setup_tair_kvcache_manager.py start \
    --workdir /tmp/tair_kvcache-live --manager-repo <tair-kvcache checkout>
source /tmp/tair_kvcache-live/tair_kvcache_live.env
cargo test -p sgl-kv-indexer --test test_tair_kvcache_live -- --ignored
python3 sgl-kv-indexer/tests/test_tair_kvcache_live/setup_tair_kvcache_manager.py stop --workdir /tmp/tair_kvcache-live
```

With `--ignored`, a missing `TAIR_KVCACHE_LIVE_ENDPOINT` fails every test instead
of skipping, so `ok. N passed` always means the manager was really driven; an
endpoint that no longer answers fails fast on a connection probe naming it.
`start` is idempotent (`config` re-provisions, `status` reports the process,
`stop` kills it), the workdir holds the registry, the logs and the env file the
tests read, and `--manager-repo` / `--binary` relocate a moved checkout. The
harness provisions deliberately tight liveness (2s timeout, 2s grace) so the
reclamation tests finish in seconds, which is why every test states its own
heartbeat interval; production values are an order of magnitude larger. Besides
the clear, registration and liveness behavior above, the suite pins the
manager's acceptance rules for specs and URIs, `NODE_REGISTER` resend
idempotency, the reported-medium filter, `snapshot_required` semantics and the
raw `p2p_host_count = 0` answer.
