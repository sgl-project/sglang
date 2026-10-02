# Training Snapshot Producer

This package implements the tensor contract and producer-side data plane for
`maas_target_kv_v1`. It is under development. Runtime collection is opt-in through
`--training-capture-config`; see the implementation status document for verified
model/runtime combinations. The default configuration does not collect requests.

## Runtime Configuration

An example for an unquantized Qwen3-0.6B target with layers 0, 14 and 27:

```json
{
  "dataset_id": "dspark-qwen3",
  "model_id": "Qwen/Qwen3-0.6B",
  "producer_revision": "your-immutable-sglang-build-revision",
  "selected_layer_ids": [0, 14, 27],
  "catalog_endpoint": "http://catalog:8080/v1/training",
  "catalog_token_env": "CAPTURE_CATALOG_TOKEN",
  "journal_directory": "/var/lib/sglang/capture/worker-0",
  "store": {
    "local_hostname": "10.0.0.11:50052",
    "master_server_addr": "10.0.0.10:50051",
    "protocol": "tcp",
    "metadata_server": "P2PHANDSHAKE",
    "global_segment_size": 0,
    "local_buffer_size": 16777216,
    "rdma_devices": ""
  },
  "sample_ratio": 0.01,
  "max_sample_tokens": 8192,
  "max_inflight_samples": 4,
  "max_host_bytes": 536870912,
  "storage_chunk_tokens": 256
}
```

Synchronous and normal overlap scheduling are supported for ordinary AR and
DSpark verification. Ordinary AR can use TP/PP with DP=1;
the complete worker group must participate in capture startup. Real-model
numerical validation currently covers Qwen3-0.6B at TP2/PP1, TP1/PP2 and
TP2/PP2, including chunked prefill, prefix reuse, decode CUDA Graphs and TP
overlap scheduling. Other topologies and model families still require deployment
validation. PP uses SGLang's non-overlap pipeline scheduler.
DSpark capture supports TP and synchronous PP with DP=1. Pipeline execution
requires a static target-KV draft; hidden-input and confidence-scheduled drafts
remain limited to PP=1. Qwen3-0.6B with a synthetic target-KV draft is verified
at TP2/PP1, TP1/PP2 and TP2/PP2, including eager/graph execution, prefix reuse,
full acceptance, rejection and mixed acceptance lengths in a batch. Overlap
is verified at PP=1; pipeline serving remains synchronous.
Each rank writes its own captured KV heads to the Store. An independent reader
reconstructs the global tensors after producer exit; the snapshot path does not
gather KV payloads through its control group.
Capture also accepts confidence-scheduled `cap-accept` and `compact` verification.
Compact capture follows each request's actual row range, excludes graph padding,
and publishes only the committed target path. Budget-trimmed candidates remain
excluded even when their token IDs match a later output. The target-KV v1 draft
checkpoint still requires static verification; non-static execution needs the
existing hidden-input DSpark model with a confidence head.
Real Qwen3-0.6B capture is verified at TP1 and TP2 for both non-static modes,
with eager execution and graph/overlap. Cap-accept uses Triton target attention;
compact uses FA3 to exercise ragged graph replay and folded acceptance. Triton
currently falls back to eager for ragged target verification. These tests use
synthetic draft weights and synchronous source observations, so they establish
data correctness, not draft quality or latency/SLO acceptance.
The configuration is checked before weights load; unsupported DP/context
parallelism, non-DSpark speculative algorithms, non-Mooncake or optimistic PD,
mixed-chunk, LoRA, quantized or embedding execution is rejected. Model/pool
binding additionally requires a local safetensors target, local tokenizer
artifacts, standard unscaled RoPE, full attention, and dense unquantized NHD
BF16/FP16 KV. Initial codecs recognize Qwen3, Qwen2 and Llama implementations;
recognition is not a substitute for per-model numerical verification.

P/D capture requires compatible capture configuration on both roles and the
Mooncake disaggregation backend. D owns admission, the complete target-KV
snapshot and Store publication. P receives the fenced capture context before
its forward and transfers the first raw teacher row with the accepted first
token. D exports the received target prefix and later decode/verify data;
neither role reruns target prefill to reconstruct missing teacher data. Missing
or stale handoffs fail capture while serving continues.

AR P/D supports matching and asymmetric TP, subject to the global target
contract. Mooncake permits matching P/D PP sizes or reduction to D PP1; PP
expansion is unsupported. Static target-KV DSpark is verified in matching TP2,
matching TP1/PP2, reduced PP2-to-PP1 and matching TP2/PP2 deployments. P may omit
the KV-input draft because D reconstructs draft context from transferred target
KV. See the current [implementation status](../../../../mooncake-study/IMPLEMENTATION_STATUS.md)
for graph/backend, topology and transport evidence. These tests do not certify
unlisted deployment combinations, trained draft quality or serving SLOs.

Keep model files immutable during loading and serving. Weight/tokenizer file
digests and actual attention geometry, K norm, RoPE and output transforms bind
each sample. Optional `expected_weights_revision` and
`expected_tokenizer_revision` pin the computed artifact digests. Weight updates
or memory release disable capture and invalidate active attempts; start a new
producer with a newly bound identity after changing the model.

The Catalog endpoint must implement the protocol below. A separate Mooncake
data node owns storage when the producer uses `global_segment_size=0`; its
lifetime must exceed producer shutdown. A process-exclusive durable journal
directory is required for each publisher. In TP/PP, only the auxiliary owner
(TP0 of the final PP stage) opens the journal and publishes the global manifest.
Every KV owner writes its own local layer/head shards; all owners agree before
the auxiliary owner publishes READY. A separate Gloo group carries capture
control messages, while tensor payloads travel through Mooncake.

For multiple ranks on one host, set `store.local_hostname` to a bare address
such as `10.0.0.11`, without a port. The Mooncake SDK then allocates independent
ports for each client; sharing one explicit port would make rank startup fail.
Different nodes need their own reachable local addresses and journal paths;
these rank-local settings are excluded from common startup policy comparison.
Keep each serving instance's publisher journal directory exclusive.
RDMA uses the existing SDK's `protocol=rdma`
and `rdma_devices` settings and requires a separately validated NIC allocation.

Admission samples requests only once, excludes health checks and unsupported
request features, and requires capacity for prompt plus maximum response.
Background threads reserve/renew Catalog leases and publish snapshots. The
inference path never performs Catalog or Store network calls. Lack of a free
registered slot drops capture admission without delaying inference. A request
abort/retract fails the entire attempt; no partial READY sample is published.
Counters, Host slot states and the capture disable reason are exposed as
`training_capture` in SGLang's existing internal-state response.
These counters are rank-local. In PP, the READY publication counter belongs to
the auxiliary owner on the last stage; PP0 reporting zero READY does not imply
that no global snapshot was published.

### Operator Control

`POST /control_training_capture` accepts `{"action":"pause"}`, `resume`, or
`abort` on a server started with capture configured. It uses the same
`ADMIN_OPTIONAL` authentication policy as the other management endpoints:
an administrator key when configured, otherwise the API key when configured.
With neither key configured, the endpoint is open. Invalid actions and servers
without capture return HTTP 400.

`pause` stops new request selection and background reservation refill. Existing
captures, lease renewals and publication continue. `resume` removes this manual
pause; it does not clear permanent disable reasons, reset adaptive cooldowns,
change the sampling ratio, revive aborted captures, or start collection in the
middle of a request that was skipped while paused.

`abort` pauses admission and invalidates collecting contexts and distributed
tickets that have not bound. It leaves ordinary generation running. D2H and
Store actors retain buffer ownership until their existing completion/cleanup
protocol finishes. Snapshots already handed to a writer may still publish;
the action does not revoke READY samples or delete Store objects. Spare Host
arenas and existing leases remain allocated, and a reservation already in
flight can finish. This is not a request to unload the capture subsystem.

The response contains `success` and a `results` list of scheduler replies,
including each replying scheduler's capture `state`. Existing scheduler
communication forwards the command through TP/PP. The response is not an
all-rank drain barrier or a Store cleanup acknowledgement. In particular, the
ingress rank's counters cannot certify that the final PP stage has published.
Use per-rank metrics and the Catalog to verify final outcomes.

For P/D, control each endpoint separately. Pause D first to stop new complete
samples, leave P running while existing handoffs drain, then pause P. Resume P
before D. For immediate collection abort, send `abort` to both endpoints; P
fences its old teacher state even if a later `resume` precedes its handoff.
An ordinary request can finish successfully while its capture is discarded.

State exposes `admission_paused` separately from `disabled_reason`. Prometheus
exports `sglang:training_capture_admission_paused` and bounded `control_pause`,
`control_resume`, `control_abort`, `excluded_paused` event counters. The Grafana
dashboard distinguishes manual pause from failure disable. See the
[operator-control runbook](../../../../mooncake-study/experiments/CAPTURE_CONTROL.md)
for commands, validation scope and asynchronous rollback limits.
The [P/D control matrix](../../../../mooncake-study/experiments/PD_CAPTURE_CONTROL.md)
covers separate TP1/PP1 P/D HTTP endpoints with AR and static target-KV DSpark,
in eager and decode graph/overlap execution. It validates partial-prefill pause,
late teacher handoff after abort/resume, and active decode abort while ordinary
generation completes. It does not certify multi-GPU or cross-node control.

### Reservation Refill

The single-rank coordinator fills available Host slots with background Catalog
reservations. It checks renewals, expiry and admission state between successful
reservation calls, without a fixed delay after each success. Writer retirement
wakes this loop after returning a slot; recovery and shutdown also wake it.
When full or paused, it retains the 100 ms maintenance poll. Failed Catalog
admissions impose a separate 100 ms retry deadline that slot-release wakeups
cannot bypass; an enabled adaptive cooldown can delay admission further.

This changes neither the Host/device budgets nor request admission semantics.
An unavailable slot still skips capture, and a quarantined buffer stays
unavailable. Startup/recovery/refill do not guarantee that all slots are ready
when a request arrives. Distributed cohorts retain their separate all-rank
reservation loop. See the [refill experiment](../../../../mooncake-study/experiments/RESERVATION_REFILL.md)
for measured capture yield and serving cost.

### Adaptive Admission

Fixed sampling remains the default. Add an `adaptive` object to the capture
configuration to enable pressure feedback; `sample_ratio` remains the ceiling:

```json
{
  "sample_ratio": 0.01,
  "adaptive": {
    "interval_seconds": 1.0,
    "low_watermark": 0.25,
    "high_watermark": 0.75,
    "writer_stall_seconds": 10.0,
    "cooldown_seconds": 5.0
  }
}
```

Occupancy counts active/queued/writing reservations and quarantined slots,
divided by `max_inflight_samples`. Pre-reserved, available slots do not count as
busy. At or above the high watermark, the target probability halves at most once
per interval, down to 1% of the configured ceiling. At or below the low watermark,
it recovers by 10% of the ceiling per interval. Between the watermarks it holds.
All limits must be finite, time intervals must be positive, and watermarks must
satisfy `0 <= low < high <= 1`.

If the oldest queued/writing task exceeds `writer_stall_seconds`, new capture
admission pauses. Catalog admission/heartbeat failures and writer failures also
pause admission. Repeated faults or an ongoing stall extend the cooldown; they
do not repeatedly reduce the ratio within the same adjustment interval. A known
stall keeps the effective ratio at zero until a fresh pressure observation clears
it, even if a short cooldown expires between observations. New
Catalog reservations pause during cooldown, while existing lease renewals and
writer work continue. Healthy recovery never exceeds the configured ratio;
`sample_ratio=0` stays zero.

The inference path reads local state only. In-flight samples retain their Host
arenas and continue toward publication or fenced failure. Admission recovery
cannot release an uncertain transfer or reuse a quarantined slot. Existing hard
Host limits, unsupported-request exclusions and permanent disable reasons still
apply.

`training_capture.admission` exposes configured/target/effective ratios, the last
control reason, cooldown remaining, observed occupancy/writer age, and decrease,
recovery, failure and pause counters. `adaptive_sampled_out` counts requests that
fixed sampling would have selected but the controller excluded. Without the
optional latency configuration below, these are local pressure signals only.

### Scheduler Latency Protection

Optional `adaptive.latency` adds explicit scheduler-side budgets. Set budgets
from a capture-off baseline for the deployment's input distribution; the values
below illustrate configuration and are not established production thresholds:

```json
{
  "adaptive": {
    "latency": {
      "ttft_seconds": 0.5,
      "tpot_seconds": 0.03,
      "window_seconds": 30.0,
      "min_observations": 16,
      "max_observations": 2048,
      "percentile": 0.95,
      "recovery_fraction": 0.8
    }
  }
}
```

At least one budget is required. These are scheduler observations, independent
of `--enable-metrics` and of capture selection:

- TTFT: scheduler request creation to processing the first committed output.
  This includes scheduler queueing/chunked prefill, but excludes API tokenization,
  detokenization and client/network delivery.
- TPOT: elapsed time between processed output updates divided by the newly
  committed output-token count. This normalizes speculative multi-token updates;
  rejected drafts, pending tokens, duplicate callbacks and a stop-truncated tail
  do not inflate the count. It is an interval statistic, not per-request mean TPOT.
- Health checks and aborted results are excluded. Retraction does not erase
  previous output timing. Finished request state prevents duplicate observations.

Each enabled metric has a bounded recent-observation deque. At most
`max_observations` values from `window_seconds` are retained; nearest-rank
quantiles are assessed at most once per adaptive interval. A metric needs
`min_observations` before it can trigger a budget breach. Any ready metric above
its budget pauses new capture and applies the existing cooldown/decrease policy.
Generation, existing capture work and lease renewal continue.

Recovery requires every enabled metric to have enough fresh observations at or
below `budget * recovery_fraction`, followed by the normal cooldown and gradual
ratio recovery. Missing or expired data cannot clear an existing breach or
increase a reduced ratio. Unsampled requests continue supplying observations,
so a paused collector can recover without admitting new samples. No baseline
overhead is inferred automatically and no client-facing SLO is certified.

`admission.latency` reports warmup/healthy/breached/stale state, observed quantiles,
budgets, counts, observation age and whether fresh values permit recovery. It is
`null` when unconfigured. Prometheus exports bounded state, quantiles, budgets
and counts; observation age and invalid counts are available in `/server_info`.
The dashboard plots scheduler quantiles/budgets separately from serving histograms.

## Ownership

- `teacher.capture_teacher` reads the unpadded vocabulary before serving-side
  processors and returns independent top-128 IDs, raw values, and full-vocabulary
  LSE. All operations run on the calling CUDA stream. FP16/BF16/FP32 CUDA scores
  reuse the existing single-pass row-LSE kernel with FP32 accumulation; top-128
  selection remains `torch.topk`. CPU and other floating dtypes retain the Torch
  normalizer. Serving initializes the FP32 path during target-contract binding,
  before admission and before P/D's prefill early return. Warmup synchronizes
  only its startup stream; failures participate in the existing startup vote.
  See the [teacher LSE experiment](../../../../mooncake-study/experiments/TEACHER_LSE.md)
  for numerical and performance evidence.
- `SelectedLayerKVExporter` gathers the selected layers through logical-to-pool
  slot indices into independent temporaries and copies into pinned Host views.
  Enqueue before source-slot reuse. The caller must retain Host slots and record
  a completion event after the final copy. Layer IDs preserve contract order.
- `HostBufferPool` allocates fixed-capacity registered arenas once. Admission
  fails when all slots are held. A slot is reusable only after D2H and all Store
  calls finish. An uncertain transfer quarantines the slot.
- `teacher_d2h_batch_tokens` opts into request-owned compact teacher staging.
  Its default is one (direct D2H). Larger values require `max_device_bytes`,
  shared with KV staging across all local slots. The aux owner alone allocates
  these buffers; CPU P/D handoff rows remain on Host. Seal flushes pending rows
  on their producer stream before publication; abort discards unpublished tails.
- `prepare_snapshot` runs only after D2H completion. It retains the final partial
  chunk and describes request-owned views, including tokens and masks. It
  never aliases or modifies a shared HiCache page. Shape/coverage checks and
  byte digests do not certify tensor contents. `build_snapshot` adds full
  content validation for direct consumers.
- `SnapshotWriter` is synchronous and belongs on one background writer thread.
  It fully validates the snapshot before registering object identities with the
  Catalog, writing tensors, sealing, writing the READY manifest and publishing
  a metadata-only reference. Validation is mandatory even for prepared views.
  The capture lease must already exist; the caller owns admission, heartbeats
  and failure handling. The single-rank coordinator supplies a `check_current`
  callback, which rejects cancellation, local lease expiry or excessive capture
  age after validation and before registration. Catalog fencing still governs
  each subsequent operation and recovery.
- `PublicationJournal` persists exact manifest bytes before sealing. On restart,
  `recover()` confirms the fence, prepared state and each stored tensor digest,
  retries the same manifest,
  and republishes with the same idempotency identity. It needs no target forward
  or saved tensor payload. Unknown outcomes leave the journal entry intact.

The Store adapter must have one owner thread after initialization. Stop that
thread and complete outstanding CUDA copies before `close()`. Store failures
keep strong references to registered source/receive memory until client close.
Do not unregister quarantined arenas based on an exception alone. Closing a
producer does not authorize deletion of published Store objects.

`MooncakeSnapshotStore.put_registered_batch()` prevalidates every source and
rejects duplicate keys before any network operation. When the SDK provides both
`batch_is_exist` and `batch_put_from`, each snapshot/owner's tensor payloads use
one existence query and one write for missing keys. Existing keys are read back
and checked for exact size and digest before any missing key is submitted.
The same mandatory hard-pin/replica configuration applies to the batch.

Every per-object write result must be an integer zero. Known failed entries
quarantine their enclosing registered arenas; an exception, malformed result or
missing status quarantines all submitted sources. No transport-error fallback
retries an uncertain batch. SDKs lacking either optional batch API use the
existing checked single-object path. Manifest writes remain separate and last,
and Catalog WRITTEN/seal/publication cannot advance after a payload failure.
The [batch-write experiment](../../../../mooncake-study/experiments/BATCH_STORE_WRITES.md)
records the native SDK checks and serving measurements.

## Global Target Identity

`bind_rank_target_contract` inspects one loaded target rank at startup. Supply
the serving TP/PP rank and world sizes, plus the DP replica ID. The model's PP
group and local layer interval must agree with the actual KV pool. The native
QKV projection must use ordinary matching query/KV TP groups. Every selected
local layer reports its global geometry, actual logical head range, RoPE and K
normalization; local K/V buffer shapes and dtypes are checked before export.
Other PP stages' placeholder layers are never accessed. A stage without selected
layers still checks its first local projection's TP placement.

The returned `RankTargetContract` contains metadata only and can be serialized
with `canonical_bytes`, then decoded with `msgspec.json.decode(...,
type=RankTargetContract)`. Each rank hashes its local model/tokenizer artifacts
using the existing identity rules. Expected immutable revisions are enforced
when supplied. The teacher fingerprint still includes the resolved model
configuration, dtype and output transform; no target forward is performed.

Collect exactly one record per serving TP/PP rank, including inactive payload
replicas, then call `assemble_target_contract(records, tp_size=..., pp_size=...,
dp_rank=..., aux_tp_rank=...)`. It returns `(teacher, global_kv, layout)` only
when artifact identity, vocabulary/output semantics, selected-layer order,
storage parameters, PP intervals and replicated layer semantics agree. Actual
head ranges must match the canonical layout even on non-owning replicas.
The result can directly configure the partitioned buffers and snapshot APIs.

The existing `bind_target_contract` and capture/DSpark callers use this path for
a complete single-rank target. Passing a sharded target to that single-rank API
fails instead of treating local heads as the complete model. These identity
helpers perform synchronous inspection. The caller must bind immutable artifacts
corresponding to the loaded model; live weight updates remain outside this path.

### Startup Exchange

`coordinate_target_startup(group=..., build_local=..., tp_size=..., pp_size=...,
dp_rank=..., aux_tp_rank=..., timeout_seconds=120)` exchanges identity records
through the existing Gloo CPU group. Its members must be one complete TP/PP
replica in PP-major, TP-minor order. Every serving rank participates, including
ranks with no Store payload ownership. The callback performs local configuration
loading and target binding before any Store or Catalog resource is created.

| Phase | Exchange | Failure Handling |
| --- | --- | --- |
| Binding | Fixed-size version/status/length headers | A local callback, serialization or size failure is reported before any payload collective |
| Allocation | A second status vote | All ranks confirm bounded CPU send/receive allocation before transferring metadata |
| Metadata | Padded CPU byte buffers containing strict JSON | Decode, group-origin, topology and global identity checks run independently on every rank |
| Agreement | Status and SHA-256 of the resulting teacher/KV/layout | No rank returns a contract until all validators succeed and all resulting digests match |
| Confirmation | Acknowledgement of the agreement vote | A late validator cannot accept a completed old collective after peers have already timed out |

Startup wire version 2 uses eight int64 control words: version, error flag,
payload length, four SHA-256 words and phase ID. Phase disagreement is rejected.
The wire payload is metadata-only JSON, not a pickled Python object. Each record
is limited to 1MiB; padded receive buffers across the group are limited to 64MiB
per receiver. Small control buffers are reserved before local binding and reused
for failure votes. Exception messages and tensor payloads are not transmitted.
`CaptureStartupError` reports the failed phase and the group ranks whose vote
failed. For shared metadata validation errors, every validator may report failure;
these rank IDs are not necessarily the origin of the invalid record.

Each collective wait is bounded by `timeout_seconds` (120 seconds by default),
subject also to the process group's transport timeout. A `transport` failure
requires worker/group teardown; never retry on that group. A callback stuck
before entering a collective, failure to reserve the initial small control
buffers, or a failure before process-group initialization still requires the
serving supervisor's watchdog. All members must enter the same startup invocation
order with a compatible protocol and a valid Gloo group.

The serving worker now supplies its existing world CPU group and actual rank
coordinates to `CaptureCoordinator.create`. Enabled capture loads configuration
inside the binding callback; configuration errors therefore participate in the
startup vote. Disabled capture returns before invoking the protocol. For snapshot
publishers, the factory prepares single-rank or cohort collection and coordinates
resource readiness across distributed ranks.
Distributed admission, failure propagation, descriptor exchange and owner-local
publication use the cohort path described below. P/D prefill uses the same global
identity but only participates in the fenced first-teacher handoff; D owns the
snapshot resources. DP/context parallelism remains unsupported.

### Resource Readiness

`coordinate_resource_startup(group=..., build_policy=..., prepare_local=...,
timeout_seconds=120)` first agrees on a bounded policy digest, then prepares
passive resources on every rank. Local policy construction failures are voted;
differing policies fail before any resource callback. `CaptureConfig.startup_policy`
includes request admission, sample limits, lease policy, Catalog endpoint and
shared Store settings. Only journal paths, Host/device byte budgets and local
Store address/buffer/segment/device settings may differ. The serving caller also
includes teacher, KV, layout, capture mode and overlap mode in the agreement.
Policy values and credentials are not exchanged, only SHA-256 words.

`CaptureResources.prepare(config=..., kv=..., partition=..., source_pool=...)`
creates the local exporter, Catalog client, Store client and registered Host pool.
It allocates only the partition's canonical KV heads. The aux owner also allocates
aux/manifest buffers and locks the publication journal. An aux-only owner needs
no source KV exporter; an inactive rank allocates neither a Store client nor a
Host pool, but still participates in every readiness vote. Preparation does not
issue Catalog calls, start capture threads or write Store objects.

After preparation, ranks exchange status and acknowledge the completed vote.
The extra acknowledgement matters because a timed-out Gloo `all_gather` can
still complete for a late participant. On failure, successfully prepared ranks
close their resources. Cooperative cleanup errors are voted as `cleanup`; after
transport failure there is no further collective. Callbacks must clean their
own partial failures. Store shutdown is the barrier before registered storage
can lose its references. A failed Store close retains the adapter and buffers
until a successful explicit close or process teardown. Failed resource close
also retains the owning resource bundle and journal lock.

The serving coordinator prepares its writer, lease and metrics threads behind
an activation event. Only after readiness confirmation does `activate()` release
them to perform recovery and admission. A thread creation/start failure closes
the prepared resources and wakes/stops any threads already waiting. Single-rank
serving uses this path on its existing CPU group; the resource API is also tested
with independent TP/PP-style rank partitions and real TCP Store clients.

This is startup coordination, not a durable transaction or a guarantee against
a process dying immediately after confirmation. Resource callbacks, transport
close calls and earlier worker initialization still require the serving
supervisor's watchdog. A transport-failed group must be torn down. Distributed
serving activation remains gated until the request coordinator is connected.

## Catalog Producer API

### Partitioned Writes

The writer also exposes the transport/publication boundary needed by a future
TP/PP capture coordinator. These methods do not enable distributed inference
capture; the current serving capability gates remain in force.

1. Obtain a validated global target contract and layout with
   `assemble_target_contract`. For an already established global KV contract,
   `plan_capture_layout(global_kv, tp_size=..., pp_layer_ranges=...,
   aux_tp_rank=..., dp_rank=...)` also constructs the ownership plan directly.
   The explicit half-open PP ranges must be contiguous from layer zero and cover
   every selected layer. `global_kv` describes logical heads before TP sharding.
   The designated rank on the last PP stage owns aux tensors and the manifest.
2. Pass the rank's `CapturePartition` to `HostBufferPool` and, when it owns KV,
   `SelectedLayerKVExporter.from_pool`. Each owner allocates and exports only
   its local selected layers and head count. Source pool tensors already have
   the rank-local head axis; no tensor gather is performed.
3. After local D2H completion, call `prepare_snapshot_partition` with the same
   `SnapshotMetadata` on every owner. KV-only owners must also supply their
   committed `token_ids` ledger; the aux owner may use its completed token buffer.
   It returns local registered tensor views and metadata-only
   `PreparedSnapshotPartition`. Each partition includes a `token_ids_sha256` that
   must agree with every other owner and with the aux `token_ids` descriptor.
   Equal lengths and valid KV ranges are insufficient to establish one sequence.
   The coordinator collects these descriptions and calls
   `assemble_snapshot(metadata, parts, layout=layout)`.
   Assembly checks exact expected owners, metadata identity, canonical head
   ranges, complete coverage and consistent final-token validity. Every owner
   then receives the same immutable full manifest and capture lease.
4. Each owner calls `write_partition(manifest, local_tensors, lease, owner_id=...)`.
   It validates complete manifest coverage and exactly that owner's payloads,
   sends REGISTERED, completes registered-buffer puts, and waits for WRITTEN ACK.
   The aux owner also checks token/teacher alignment, masks and vocabulary values.
5. The designated aux coordinator calls
   `publish_partitions(manifest, receipts, manifest_buffer, lease)` with exactly
   one `OwnerWriteReceipt` per declared owner. Capture ID, fence, owner and
   manifest digest must all match. It then registers the manifest and uses the
   existing journal, seal and manifest-last publication path.

For ordinary dense TP, heads are either split evenly or replicated evenly.
When TP exceeds the global KV head count, the first rank in each replica group
is the canonical owner, matching the native QKV weight loader's head placement.
An inactive partition is explicit (`active=False`); it is excluded from the
manifest's owners and must skip payload allocation/export. Passing it to a Host
pool fails, while `partition=None` retains the full single-owner behavior.
Inactive ranks must still participate in the future distributed control and
collective protocol. An aux-only owner allocates no KV staging, may allocate
teacher staging when `teacher_d2h_batch_tokens > 1`, and must
skip construction of a KV exporter. Non-aux owners allocate no aux or manifest
capacity. Host and device budgets are enforced independently per rank.

### Owner-Local Requests

Pass the same partition to `RequestCaptureContext(..., partition=...)` after
request admission. It validates the Host slot's local tensor set and KV head
counts before use. All active owners retain the request's committed token ledger;
only the aux owner writes token, position, mask and teacher payloads. Inactive
partitions cannot create a request context.

KV owners advance through `export_kv`, preserving the existing D2H events,
bounded staging and source-slot reuse rules. An aux-only owner calls
`record_kv_progress(end=...)` with the computed prefix from its target forward;
that API cannot bypass KV export on a KV owner. Only the aux owner may call
`record_positions` or `record_teacher[_range]`. Every owner observes and commits
the same accepted target path in order; a commit cannot exceed its known KV
prefix or capacity. Aux commits additionally require matching teacher rows.

Trim lookahead/verify suffixes with `trim_terminal_prefix`, then call `seal` on
every owner. Sealing flushes any staged KV tail, validates local coverage and
forms the sequence metadata; only aux writes masks and final token payloads.
In the background writer, `prepare_partition(**metadata)` waits for CUDA
completion and returns the local descriptors/views, binding the committed token
ledger to `token_ids_sha256`. The caller supplies identical global metadata on
every owner. The aux token buffer must still match its own ledger, and assembly
requires the same sequence digest from every owner. The digest is an internal
partition field, not a new public manifest field; all producers must use the
matching preparation/assembly interface.

`snapshot()` remains the fully validated, unpartitioned context API, with a
sealed-state check after content validation. The single-rank background writer
uses `prepare_snapshot()` followed by the writer's mandatory validation, avoiding
a duplicate full payload scan. Partitioned contexts must use coordinated
assembly; their preparation path is unchanged. `abort` can invalidate collecting or
sealed contexts. Preparation rechecks cancellation after copy waits and descriptor
construction before returning a result. This does not revoke already
prepared descriptors, Store writes or published samples. The request coordinator
must discard pending preparations and enforce the Catalog fence on cancellation.

These request contexts are exercised with canonical TP shards, an aux-only PP
stage, source reuse, incremental decode and accepted verify prefixes. Runtime
distribution of accepted tokens, admission decisions and aborts is still pending;
constructing a partitioned context alone does not enable distributed serving.

Preparation and assembly are background work after CUDA completion. Their views
do not extend a buffer lease: each owner retains its Host slot until its Store
writes complete, and the coordinator retains the manifest buffer until
publication completes. Existing uncertain-transfer quarantine rules still apply.
These APIs do not exchange per-sample descriptors between processes. Connecting
resource readiness to distributed request admission, sample metadata transport,
failure coordination and scheduler wiring remains required.
The planner covers ordinary dense TP/PP, not context parallelism or sparse KV.

Receipts contain metadata only. They are trusted producer acknowledgements,
not retention leases or signed Catalog attestations. The Catalog must verify
every exact WRITTEN descriptor and the current fence at seal. Missing owners,
duplicate receipts or receipts for a different manifest never reach publication.
After journaling, recovery requires no original owner buffers. Before that
point, an incomplete capture can fail; caller/Catalog failure and expiry handling
must reclaim its registered objects. Replica lifetime must outlive producer
processes, and a completed owner write does not authorize Store object deletion.

### Collective Reservations

`CaptureCohortAllocator` reserves one complete set of owner-local Host slots
before a serving request is bound. Construct it after global identity/resource
startup, using the agreed config, teacher, KV spec, layout and local
`CaptureResources`. Its Gloo group must be dedicated to background capture
control, in the layout's PP-major/TP-minor order. The default world group is
rejected. Every rank, including inactive partitions, calls `reserve()` in the
same sequence; the call is synchronous and must stay off the inference thread.

The protocol first agrees on the common capture policy. Each active owner then
acquires a local slot. Missing capacity returns `None` on every rank, after
releasing all slots acquired in that round, without a Catalog begin. If every
owner has capacity, only the aux owner calls begin with the complete owner set
and summed registered Host bytes. All ranks receive and validate one typed
`CaptureLease`, agree on its identity/fence, and confirm readiness before return.
Round and phase IDs reject divergent control sequences. CPU control buffers are
allocated once; each rank's lease payload is bounded to 4096 bytes.

The returned `CaptureCohort` carries the shared lease, local slot (`None` for an
inactive partition), summed byte reservation and local monotonic deadline/renewal
time. Each local clock is anchored before the capacity vote, before Catalog begin
can run. Delays consume the available lease interval conservatively. Validation
and final confirmation both reject a lease already due for renewal. Callers must
still check these times before binding or writing: readiness is not a guarantee
against later expiry or process failure.

On failure, only locally reserved, unused slots are released, and only the aux
owner reports a known matching lease as failed. A lost begin response or a
response with changed identity cannot safely be failed using untrusted lease
credentials; the Catalog must expire that unused reservation. No CUDA/Store
transfer has begun at this point. Cleanup failure is reported to every peer.
Transport or control-sequence failure poisons the allocator; stop capture and
tear down its control group instead of retrying it.

`renew(cohort)` renews the agreed identity/fence through the aux Catalog client,
then repeats lease validation and readiness on all ranks. A failed renewal never
releases the slot. `fail(cohort)` reports a drained capture's terminal failure
and requires a confirmed FAILED response. `synchronize(build_local)` exchanges
bounded CPU registry frames after all ranks agree on the registry ledger and
frame shape. Frame construction and validation errors are voted; a final
acknowledgement precedes any state-dependent control operation. These operations
use the same serialized, versioned control sequence as reservation.

### Background Cohort Lifecycle

`CaptureCohortService` owns the allocator after startup. Construct it alongside
passive resources, validate construction on all ranks, then call `start()` on
each rank. Its background thread fills a bounded registry, maintains leases,
propagates failure and retires drained cohorts. A newly reserved cohort is not
claimable until another registry exchange confirms every rank installed it.
No HTTP request or collective executes from the request-facing methods:

| Method | Ownership Contract |
| --- | --- |
| `claim(request_sha256)` | Rank zero returns a fresh `CaptureTicket`, or `None` immediately when no prepared cohort is available. |
| `bind(ticket, request_sha256)` | Each participating actor binds once to its local cohort; stale fences, mismatched request identity and expired/failed cohorts cannot bind. |
| `status(handle)` | Read the current renewed lease and local invalidation reason under the service lock. |
| `cancel(ticket, reason)` | Cancel before binding, or invalidate an existing request without relinquishing its buffers. |
| `fail(handle, reason)` | Report local failure; a bound handle still requires `finish`. |
| `finish(handle, outcome, transfer_complete)` | Relinquish local actor ownership after CUDA/Store work resolves. KV owners report `stored` or `failed`; aux reports `published` or `failed`. |
| `close(timeout)` | Request collective shutdown and wait boundedly. A false result prohibits resource/group teardown while workers or uncertain transfers remain. |

Tickets carry the capture ID, fence and SHA-256 of the request contract. The
runtime must compute that digest from stable request identity and the prompt /
sampling contract, then carry the ticket through existing request propagation.
The service checks agreement but does not construct that request digest or
authenticate arbitrary producer processes. Handles are process-local; use the
service methods to inspect/update them and never serialize their tensors or
mutate their fields from request code.

Cancellation and lease expiry close admission immediately when observed. Slot
recycling additionally requires every rank to have already voted invalid and
every bound actor to acknowledge completion. This extra round prevents a bind
that occurred after a control snapshot from losing its slot during peer
cancellation. Published cohorts require all active owners to have bound and
all bound actors, including inactive metadata-only ranks, to finish. Inactive
ranks without a handle still participate in control but have nothing to drain.
Completed publication wins over a concurrent cancellation; it is never failed.

The aux actor must resolve any ambiguous publish through its durable journal
before reporting a final outcome. A renewal failure cannot prove publication
failed. While an invalid capture is still draining, renewal may continue to
protect its lease; an unsuccessful renewal disables further renewal of that
cohort. A control failure stops admission and retains bound buffers until local
completion. `transfer_complete=False` permanently quarantines the slot, and the
service retains the resources even if its caller drops its reference. Normal
close does not destroy the dedicated group, close Store, synchronize CUDA, or
unregister memory. The supervisor owns those actions after a successful close;
uncertain transfers require explicit transport/device teardown or process exit.
An aux owner can report `published` with `transfer_complete=False` after recovery
using another manifest buffer. Publication wins, while the original Host slot
remains quarantined and prevents a normal service close. A `stored` receipt still
requires proven transfer completion.

The scheduler request hooks below can carry these tickets, and the metadata
exchange below coordinates completed owner partitions. Distributed coordinator
construction, accepted tokens and global teacher logits still need integration
before opening the distributed serving gates. In particular, an all-PP
collective inside `before_forward` would deadlock
against PP proxy receive/send ordering. All service collectives remain on its
dedicated group.

### Request Ticket Routing

`CaptureRequestRouter` adapts a ready cohort service to the existing request
transport. Install it as `coordinator.request_router` before scheduler receiver
construction. `SchedulerRequestReceiver` invokes `prepare()` only on the first
PP stage's TP/CP ingress rank, after input blocking and before TP broadcast.
It selects eligible bounded text requests once, consumes a ready ticket locally,
and appends a bounded metadata field to `TokenizedGenerateReqInput`. The existing
TP broadcast and PP request send/receive carry that field. Control requests and
batch wrappers preserve their ordinary routing. A missing ready cohort skips
capture without retrying from downstream ranks or waiting for Catalog.

The version-1 wire ticket contains a fresh request-incarnation nonce and the
cohort ticket. Its request hash covers request ID, prompt tokens, every sampling
parameter field, token types, reasoning mode and cache salt. Stop-token sets
are canonicalized; mapping insertion order does not change the hash. The nonce
separates repeated requests that reuse the same ID and input. The wire budget
is 2048 bytes and the request identity budget is 1 MiB. The ingress overwrites
client-supplied capture metadata. Excluded inputs include sessions, multimodal
or embedding inputs, LoRA, custom positions/processors/parameters, health checks,
requests marked `no_logs`, and requests outside the configured sample capacity.
Adaptive selection requires an explicit admission-ratio callback.

The normal scheduler request handler calls `attach(incoming, req)` to validate
the original ingress contract without acquiring local actor ownership. Only a
selected request gets a private copy of its sampling parameters. Thus scheduler
clamping of `max_new_tokens`/`min_new_tokens` and later local normalization cannot
modify the ingress object sent to another PP stage. Appending the optional field
preserves decoding of older array IPC messages that omit it.

The distributed coordinator must call `bind(req)` before the first capture
forward. It returns a process-local route containing the cohort handle and a
separate `execution_sha256` for the effective request after normalization. The
original ingress identity remains the service's ticket/fence binding. The
execution hash must join descriptor agreement before publication; merely
checking the ingress hash does not prove that ranks executed identical effective
requests. Rebinding, advanced/retracted/aborted requests and unavailable cohorts
fail capture without rejecting inference.

Queue rejection or priority eviction cancels the removed request's ticket.
Waiting timeout, explicit queued/running/chunked abort and grammar rejection
also invalidate its route. These callbacks never finish a bound actor's
transfers; the coordinator/writer remains responsible for `service.finish()`.
Unbound captures are reclaimed by the background invalidation protocol.

The current single-rank `CaptureCoordinator` leaves `request_router=None` and
keeps its established first-forward admission. Distributed construction still
rejects unsupported topology. The hooks and real TP/PP request transport are
tested independently with a cohort service; this is not yet a distributed model
forward, teacher-logit or Store-publication workflow.

### Owner Metadata And Receipts

The cohort service exposes a writer-side protocol for completed partitioned
snapshots. Its control thread chooses each exchange from the registry state
agreed by every rank; submission and polling never execute a collective. Run
descriptor preparation, submission, polling and Store I/O in writer actors,
after `RequestCaptureContext.wait_for_copies()` / `prepare_partition()` has
confirmed D2H completion. No tensor payload enters the control group.

| Method | Writer Contract |
| --- | --- |
| `submit_snapshot(handle, execution_sha256=..., prepared=..., metadata=...)` | Every rank submits once. Active owners supply their prepared partition; only the aux owner supplies `SnapshotMetadata`. Inactive ranks submit the effective execution digest with both optional fields unset. |
| `get_manifest(handle)` | Returns `None` until all descriptors validate and every rank acknowledges the same manifest digest. Returns a newly decoded view once ready. |
| `submit_receipt(handle, receipt)` | Each active owner submits once, after `SnapshotWriter.write_partition()` completes. Receipt identity, fence, owner and full manifest SHA-256 must match. |
| `get_receipts(handle)` | Returns `None` until all active owner receipts agree, then a fresh tuple for aux `SnapshotWriter.publish_partitions()`. |

The descriptor exchange binds the original ingress request digest, the effective
execution digest, the capture identity/fence and the canonical owner layout.
Inactive ranks also vote on execution identity. Aux metadata must match the
reserved dataset/sample/generation, teacher, KV contract and contract ID.
`assemble_snapshot()` additionally verifies complete logical head/token coverage,
metadata hashes, identical committed token ledgers, KV validity and the sole aux
payload. A single timestamp supplied by aux makes the manifest identical across
ranks. Every rank votes its final manifest hash before writers can retrieve it.
Submission freezes bytes; subsequent caller mutations cannot change the vote.

Control protocol version 3 first votes metadata kind, capacity and payload
lengths, then allocates explicit CPU buffers. Per-rank offers and agreed results
are bounded by `manifest_buffer_bytes + 4096`; the padded send/receive tensor
arena is limited to 64 MiB. The manifest must also fit `manifest_buffer_bytes`.
Encoder, allocation, parser and validator failures are voted before advancing,
and a final vote checks the conservative local lease deadline. Transport failure
poisons the allocator. Metadata rejection invalidates the capture without
releasing any actor-owned transfer storage.

KV owners may `finish(..., outcome="stored")` after submitting their local
receipt; their frozen metadata remains available to the background protocol.
Inactive actors finish after manifest agreement. Aux must collect all receipts,
publish through `SnapshotWriter` and resolve any ambiguous journal outcome
before reporting `published`. For handles using this exchange, successful finish
rejects missing manifests, local receipts or (for aux) the agreed receipt set.
These acknowledgements are trusted producer statements; the Catalog still must
validate every WRITTEN descriptor at seal and enforce retention/fencing.

The four-process Gloo/HTTP Catalog fixture validates failure ordering. A real
TCP Mooncake test also writes all partitions, exchanges receipts, publishes from
aux and reads the entire sample after the producer processes exit. These are
storage/control tests with synthetic tensors, not distributed model inference.
The serving coordinator and global teacher/token delivery remain to be connected.

### Completed-Context Writer

`CohortSnapshotWriter(service)` owns the Store actor for one rank. Construct it
after resources and the service are ready, call `start()`, and wait for
`stats()["ready"]` before handing it completed actors. Aux startup replays its
existing publication journal before becoming ready; a recovery error keeps it
unready. Inactive ranks have a writer for metadata/handle completion but no Store.

`submit(handle, context=..., metadata=..., execution_sha256=...)` transfers a
sealed partitioned `RequestCaptureContext` and frozen `SnapshotMetadata` to the
writer. Every bound rank submits, with no context/metadata on inactive ranks.
The inference thread must stop touching the context after a successful submit.
An exception leaves ownership with the caller, which must still drain/finish it.
Submissions are bounded by `max_inflight_samples` and reject duplicate handles.
`failure_reason=...` submits an aborted actor; its outstanding copies are still
waited before acknowledging failure. A failure with no context means no copy was
ever enqueued, such as an admission failure before context creation.

One Store thread advances each queued actor through copies, manifest agreement,
owner writes, receipt agreement and aux publication. Waiting for peer descriptors
or receipts does not block processing another queued request. This matters when
different PP/TP ranks finish requests in different orders. The worker never
reads a serving request or calls a collective; it uses the cohort service's
submission/polling APIs and only writer-owned completed Host views.

Once aux attempts publication, an exception enters recovery rather than failing
the capture. `SnapshotWriter.recover_partitions()` validates the exact frozen
manifest/receipt set and each Store object, then retries the same identity using
a fresh registered manifest buffer. It covers lost seal/publish responses and
an exception after journal unlink; journal absence is not proof of failure.
The original publication lease is retained across retries, including its fence.
Recovery buffer allocation honors the Store receive/quarantine budget. Uncertain
original arenas stay quarantined even if recovery confirms publication.

`close(timeout)` stops admission, fails/drains unattempted actors and continues
resolving attempted publications. A false result retains contexts and forbids
resource teardown. True only proves that this writer stopped: the supervisor
must also close the cohort service, resolve any quarantined transport/device
storage, and then close resources and destroy the dedicated control group.
Capture expiry, cancellation or shutdown cannot turn an ambiguous publish into
a failed sample. Permanently rejected fences remain pending for reconciliation.

This actor is exercised with real request contexts, four Gloo producer processes
and TCP Mooncake, including opposite request completion orders and independent
readers after producers exit. It is ready for distributed coordinator ownership;
the serving factory and global teacher/accepted-token delivery remain gated.

### HTTP Contract

`HTTPCaptureCatalog` implements the producer half of the proposed SpecForge
Catalog API. It does not implement the Catalog server, distributor, consumer
leases, checkpoints, or GC. The endpoint is the API base URL. Requests include
`X-Training-Capture-Protocol: 1` and an optional Bearer credential supplied at
construction. Internal traffic ignores external-network proxy settings.

| POST Route | Request | Required Response |
| --- | --- | --- |
| `/captures:begin` | Immutable dataset/sample/generation identity, contract/owner metadata, reserved budget, idempotency key | `CaptureLease` |
| `/captures/{id}/heartbeat` | Capture ID and fence | Renewed `CaptureLease`, same identity/fence |
| `/captures/{id}/objects` | Fence, `phase=REGISTERED` or `WRITTEN`, descriptors, idempotency key | Same phase and sorted `accepted_object_ids` |
| `/captures/{id}/seal` | Fence, owner, sequence, total tensor bytes, manifest descriptor, exact `manifest_base64`, idempotency key | `state=PREPARED` or already `READY`, matching `manifest_sha256` |
| `/samples:publish` | Fence, dataset/sample/generation, contract, manifest key/digest/byte length, idempotency key | `state=AVAILABLE`, `publication_id`, `catalog_cursor` |
| `/captures/{id}/fail` | Fence, reason, idempotency key | Catalog failure receipt |

A `CaptureLease` contains `capture_id`, `fencing_token`, `dataset_id`,
`sample_id`, `generation_id`, `expires_in_seconds`, and `renew_after_seconds`.
TTL values are relative to the Catalog response, not shared machine clocks.
The producer should conservatively compute local deadlines from request start.

The Catalog must validate namespace, budget, owner coverage, descriptors,
checksums and the full decoded manifest at seal. Every retry must check the
current fence/state, including previously successful seals; a deleted sample
must not become writable again. Publishing atomically records the publication
and durable outbox event. Published captures cannot be reclaimed by the
unfinished-capture expiry path. Consumer retention and GC remain Catalog-owned.

The writer pre-registers the manifest alongside tensor identities. The manifest
descriptor has `object_id=manifest`, `kind=manifest`, key, byte length, digest
and owner. Its write receipt follows the manifest put. Catalog metadata stores
the exact manifest bytes to support recovery across ambiguous seal responses.

Only metadata enters the journal and HTTP API. Tokens, logits and KV bytes are
raw Store objects. Journal files should live on a durable volume; one process
owns a directory through a filesystem lock. A rejected fence leaves its entry
for Catalog/operator reconciliation and never triggers unconditional deletion.

## Prometheus Metrics

With both `--training-capture-config` and `--enable-metrics`, the existing
`/metrics` endpoint exports the following `sglang:training_capture_` families.
No capture metrics or monitoring thread are created when either flag is absent.
The producer inherits the scheduler's model/rank and configured extra labels.
Request IDs, dataset contents, object keys, exception text and per-sample identity
are never metric labels. Event/action/state/kind/metric/stage labels have bounded value sets.

| Suffix | Type / Extra Label | Meaning |
| --- | --- | --- |
| `events_total` | Counter / `event` | Admission, exclusions, forwards and writer lifecycle events |
| `admission_adjustments_total` | Counter / `action` | Decreases, recoveries, failures and pause transitions |
| `sample_ratio` | Gauge / `kind` | Configured ceiling, adaptive target and effective admission probability |
| `reservations` | Gauge / `state` | Available, active, queued, writing and pending-publication reservations |
| `host_slots` | Gauge / `state` | Free, filling and quarantined registered Host slots |
| `host_allocated_bytes` | Gauge | Allocated Host arena capacity, including manifest buffers |
| `occupied_fraction` | Gauge | Busy reservations plus quarantined slots divided by configured capacity |
| `queue_depth` | Gauge | Work awaiting background processing |
| `writer_age_seconds` | Gauge | Age of oldest queued, writing or pending-publication work |
| `stage_calls_total` | Counter / `stage` | Completed background stage attempts, including failures/retries |
| `stage_failures_total` | Counter / `stage` | Stage attempts that raised an exception |
| `stage_seconds_total` | Counter / `stage` | Cumulative wall time of completed attempts |
| `stage_max_seconds` | Gauge / `stage` | Lifetime maximum completed attempt, not a quantile |
| `adaptive_enabled` | Gauge | Whether adaptive admission is configured |
| `disabled` | Gauge | Capture disabled by a failure or shutdown, distinct from adaptive cooldown |
| `admission_paused` | Gauge | Operator pause/abort blocks new capture, independently of failures |
| `cooldown_seconds` | Gauge | Remaining adaptive cooldown |
| `latency_control_enabled` | Gauge | Whether scheduler latency protection is configured |
| `latency_blocked` | Gauge | New capture paused by a latency breach or missing recovery evidence |
| `latency_recovery_ready` | Gauge | All enabled metrics have enough fresh values below the recovery threshold |
| `latency_state` | Gauge / `state` | One-hot disabled, warming, healthy, breached or stale state |
| `latency_percentile` | Gauge | Configured quantile, NaN when latency control is disabled |
| `scheduler_latency_seconds` | Gauge / `metric`, `kind` | TTFT/TPOT observed quantile and budget; missing or unconfigured values are NaN |
| `latency_window_observations` | Gauge / `metric` | Number of TTFT/TPOT observations in the bounded window |
| `latency_observations_total` | Counter / `metric` | Valid TTFT/TPOT observations, including requests excluded from capture |
| `metrics_update_timestamp_seconds` | Gauge | Unix time of last successful metrics update |

A dedicated background thread snapshots CPU state once per second. It keeps
running while Catalog RPCs or Store writes block, requires no GPU synchronization
and performs no Prometheus work on the inference thread. Export failures are
logged and retried without changing capture ownership or disabling serving.
Gauges use the existing server's multiprocess `mostrecent` convention; monitor
their freshness together with Prometheus `up`, especially after worker failures.

Counters aggregate observed lifecycle events, not mutually exclusive outcomes:
`adaptive_sampled_out` is a subset of `sampled_out`; `capture_failed` and
`writer_failed` can refer to the same request. `ready` counts direct successful
writer publications in this process, excluding journal recovery and consumer
acknowledgements. `snapshot_built` means descriptors and a manifest were prepared;
it does not imply successful content validation or publication. Do not infer
exact dataset completeness from a rate ratio.
`writer_age_seconds` includes queue waiting, CUDA completion, serialization and
Catalog work; it is not RDMA latency. `host_slots{state="filling"}` includes
spare leases, whereas `occupied_fraction` excludes them. Host bytes report
allocated capacity, not payload transfer volume. These metrics also work with
fixed sampling (`adaptive` absent).

`stage_timings` in the existing producer status reports the same cumulative
`calls`, `errors`, `seconds` and `max_seconds`, even when Prometheus is disabled.
The fixed stages are `queue_wait`, `copy_wait`, `snapshot_build`, `validation`,
`catalog_register`, `store_payload`, `catalog_written`, `journal_save`,
`catalog_seal`, `store_manifest`, `catalog_publish`, `journal_complete` and
`recovery_read`. State uses constant memory; no samples or request IDs are kept.
Measurements run in background writers and add no CUDA events/synchronization.
Pending operations appear in `writer_age_seconds`; their stage duration is
recorded only when they finish or raise. A zero count means no completed attempt.
`snapshot_build` includes descriptor preparation and source hashing, while
`validation` includes full writer validation and the optional current-request
check. Direct checked snapshot APIs still validate before returning. See the
[publication validation experiment](../../../../mooncake-study/experiments/PUBLICATION_VALIDATION.md)
for the single-rank scan consolidation and its measured limits.

These are host wall times including scheduling/GIL waits. `copy_wait` measures
remaining completion wait, not total D2H time; `store_payload` includes adapter
validation, existence checks and retry readback, not just wire transfer.
Journal timing includes the existing durability operations. `catalog_written`
counts both payload and manifest receipts, while recovery may repeat stages.
Each distributed rank reports local work; peer wait/collectives and uninstrumented
metadata work are not included. Do not add these values across ranks or include
queue wait to estimate end-to-end latency or GPU cost. The benchmark reports
per-phase count/time deltas excluding warmup, with lifetime maxima retained only
in the raw status snapshots. The dashboard displays mean completed-attempt time
and failure rate; full runtime dashboard acceptance remains a deployment gate.

The [monitoring example](../../../../examples/monitoring/README.md) provisions
a training-capture Grafana dashboard alongside serving metrics. Scheduler
TTFT/TPOT feedback requires `adaptive.latency`; service-specific alerts, measured
capture overhead and rollout thresholds still require deployment validation.

## Verification

```bash
PYTHONPATH=python python3 -m pytest test/registered/unit/training_capture -v
PYTHONPATH=python python3 -m pytest test/registered/storage/test_training_snapshot_mooncake.py -v -s
PYTHONPATH=python python3 test/registered/storage/test_training_capture_runtime.py --model-path /models/Qwen3-0.6B -v
```

The first command includes CUDA ownership tests and compares canonical TP head
ownership against the actual native QKV loader at TP sizes 1, 2, 4 and 8. It
also checks rank-local budgets, PP layer selection, aux-only/inactive ranks and
metadata assembly failures. Identity tests serialize rank records and reconstruct
the same global teacher/KV contract from TP4/PP3 and a single-rank fixture, while
rejecting replica, artifact and stage disagreements. The second requires the
Mooncake SDK and `mooncake_master`. It starts an isolated master, writes through
`SnapshotWriter` and a registered arena, and reads/validates every object from a
different process over TCP. Its Catalog is a test double, not a real SpecForge
service. The two-owner case prepares descriptors through the production layout,
pool and assembly APIs before independent processes publish their local tensors.
Every process it starts is cleaned up by the test.

The third command adds actual SGLang prefill/decode, a prefix hit, a one-token
response and a serving logit bias. A test-only observer records each selected
layer's attention inputs and raw logits, independently of the exporter and KV
pool. Separate Store readback must match K/V and saved logits exactly, include
valid top-128 IDs, and match full-vocabulary LSE within 1e-5. The test verifies
readback after producer exit, then runs the same requests on normal serving with
decode CUDA graphs and compares every captured tensor. This currently covers
ordinary and padded graph batches; current mode coverage is listed in the
implementation status document. A controlled-writer experiment enables
adaptive admission with normal overlap: generation continues with a spare Host
slot while new capture is paused, the original snapshot publishes after release,
and a later request is captured after recovery. Both snapshots are read from the
real Store and validated. The adaptive server also enables Prometheus, scrapes
the HTTP multiprocess endpoint during the writer stall and after recovery, and
checks ratios, event counters, reservation resets and quarantine state.
Another ordinary overlap server enables latency protection. A test-only result
processing delay causes new capture to pause while generation and existing
publication continue. Expired observations retain the pause; fresh unsampled
requests restore admission after cooldown. All four responses agree, both
captured samples survive producer exit, and the HTTP metrics include all four
requests. Production code contains no delay injection hook.
This is a functional test, not a performance benchmark.

A Transformers reference additionally checks teacher logits/LSE and reports KV
errors. BF16 intermediate KV is sensitive to the target implementation; the
optional `--assert-hf-kv` cross-engine gate currently fails for this fixture.
Its original tolerances are unchanged. See `mooncake-study/IMPLEMENTATION_STATUS.md`
and `mooncake-study/experiments/diagnose_qwen3_kv.py` for the reference-only
reproduction. Exact preservation of online tensors is the capture gate; DSpark
training/serving parity remains a separate work package.

Target recomputation exists only in development tests; the producer/reader API
never uses it. These tests use a Catalog double and do not validate production
consumer retention. The environment lock and resident-worker invocation are in
`mooncake-study/experiments/README.md`.

The design schema remains in
`mooncake-study/training-data-contract/manifest.schema.json`. Produced manifests
are checked against that schema as well as cross-object and tensor-content
constraints. Strict SDK checks distinguish zero-status writes from byte-count
reads and reject clients without hard pin support.
