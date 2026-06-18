# PD Flip Active Decode Migration Design

## Context

SGLang already has a PD flip state machine in
`python/sglang/srt/disaggregation/flip_state_machine.py`, plus scheduler wiring
that can pause admission, expose `/server_info` state, and wait for external
prepare and commit acknowledgements. The current implementation is intentionally
conservative: a D-to-P flip requires the decode worker to drain to idle before
the flip can proceed.

The missing piece for the KV pre-transfer experiment is active decode request
migration. When a decode worker is selected to become a prefill worker, we want
to move its in-flight decode requests to another decode worker by transferring
their already-computed KV cache before the source worker finishes draining.

This first version is an internal demo path. It proves that active request KV can
move from one decode worker to another and that the target decode worker can
continue scheduling the migrated request. It does not preserve the original
client HTTP or gRPC stream.

## Goals

- Support D-to-P flip preparation by migrating active decode requests from a
  source decode worker to a target decode worker.
- Reuse the existing PD KV transfer abstractions, metadata buffers, request
  preallocation, polling, and backend implementations where possible.
- Expose enough migration state through `/server_info` for the experiment script
  to drive and debug the flow on remote servers.
- Keep the first version isolated behind PD flip / migration controls so normal
  PD serving behavior is unchanged.

## Non-Goals

- No transparent client stream handoff in the first version.
- No router-level replay or automatic request redirection in the first version.
- No support for P-to-D active migration in this design.
- No new KV transfer backend protocol unless existing sender and receiver
  abstractions prove insufficient during implementation.

## Recommended Approach

Use the existing prefill-to-decode transfer lifecycle as a decode-to-decode
migration channel:

- The source decode worker creates a migration sender over its current KV pool.
- The target decode worker creates normal decode-side receivers and preallocates
  destination KV slots.
- The target sends destination metadata to the source.
- The source sends the source KV pages for each migrated request.
- The target polls the transfer queue, commits request state, and places the
  request into its waiting queue for normal decode scheduling.

The source decode worker temporarily acts like the "prefill sender" side of the
existing protocol, but the data being sent is the active decode KV state rather
than freshly computed prefill KV.

## Components

### Migration Session

Add a small scheduler-owned migration session object for D-to-P source workers.
It tracks:

- `session_id`
- target decode URL or bootstrap address
- request manifests
- per-request transfer status: `pending`, `preallocating`, `transferring`,
  `transferred`, `failed`, `released`
- aggregate counters for `/server_info`
- terminal error messages

The session is created only on decode workers during `FlipState.PREPARING`.

### Request Manifest

Each migrated request needs a compact manifest that can recreate enough target
state to continue decode scheduling:

- `rid`
- `origin_input_ids`
- `output_ids`
- sampling parameters needed by `Req`
- `bootstrap_room`
- `priority`, `routing_key`, and `extra_key`
- `return_logprob` and compatible logprob settings for demo coverage
- `kv_committed_len`, derived from the request if available, otherwise from the
  current prompt and generated-token lengths
- optional metadata for trace and debugging

The target constructs a `Req` from the manifest and marks it as migration-backed
so it does not expect normal prefill output metadata to supply the whole request
history.

### Source Decode Migration Sender

Add a helper that mirrors the relevant parts of `PrefillBootstrapQueue` but uses
the source decode worker's KV pool and req-to-token mapping:

- build a transfer manager against the source decode KV buffers
- create one sender per migration request
- expose source bootstrap information to the target receiver path
- send page indices from `req_to_token_pool.req_to_token[req_pool_idx, :kv_len]`
- include state indices for Mamba, SWA, and DSA using the same conventions as
  existing prefill send code
- poll sender completion and cleanup resources

This should avoid changing backend-specific sender implementations.

### Target Decode Migration Receiver

Add a target-side helper that reuses `DecodePreallocQueue` and
`DecodeTransferQueue` as much as possible:

- create `DecodeRequest` objects from migration manifests
- initialize receivers with the source migration bootstrap address
- preallocate destination KV slots for `kv_committed_len`
- send destination metadata back to the source sender
- poll transfers through the existing transfer queue
- after success, move the reconstructed requests into `waiting_queue`

Target requests then enter the existing `get_new_prebuilt_batch` path and resume
normal decode scheduling.

### Control Plane

Add experimental HTTP control endpoints backed by tokenizer-manager
communicators for the script:

- `POST /pd_flip/migration/source/start`: create a source migration session and
  return source bootstrap information plus request manifests
- `POST /pd_flip/migration/target/prepare`: create target receivers and begin
  target-side preallocation for the supplied manifests
- `GET /pd_flip/migration/status`: return source or target migration status for
  the local worker
- `POST /pd_flip/migration/source/finish`: release source-side migrated requests
  after the target reports success
- `POST /pd_flip/migration/abort`: abort a local source or target migration
  session

These endpoints are experimental and are not stable public API.

`scripts/playground/disaggregation/pd_flip_experiment.py` gains a
`--migration-target-url` option. In demo mode it:

1. Triggers or observes a D-to-P flip on the source decode worker.
2. Starts migration once the source enters `preparing`.
3. Prepares the target decode worker with the source manifests.
4. Polls source and target migration status.
5. Sends prepare ack once source-side migrated requests are released or no
   active requests remain.

## State Machine Integration

For D-to-P:

- `SAFE -> PREPARING` remains driven by the evaluator.
- In `PREPARING`, source decode pauses new admission.
- If active request migration is enabled and a target is configured,
  `prepare_pd_flip` waits for the migration session to finish instead of waiting
  only for natural drain-to-idle.
- Once migrated requests are released locally and all other queues are empty,
  the existing external prepare ack can advance the state to `FLIPPING`.
- Commit still requires external orchestration in this first version.

For P-to-D:

- Behavior stays unchanged.

Status should distinguish this path from drain-only behavior:

- `active_request_migration_strategy = "decode_to_decode_kv_transfer"` when a
  D-to-P migration session is active.
- `migration_enabled`, `migration_state`, `migration_pending_reqs`,
  `migration_transferred_reqs`, `migration_failed_reqs`, and last error fields
  are exposed under the existing `pd_flip` object.

## Data Flow

1. Source decode enters `PREPARING`.
2. Experiment script chooses a target decode worker.
3. Source snapshots migration candidates at a scheduler loop boundary and pauses
   those requests from further local decode scheduling.
4. Source creates migration senders and returns request manifests.
5. Target reconstructs requests, preallocates KV, creates receivers, and sends
   metadata to the source.
6. Source sends KV pages and state pages.
7. Target transfer queue commits the received KV and enqueues requests for
   normal decode scheduling.
8. Source releases migrated local request state.
9. The experiment script observes completion and sends the existing prepare ack.

## Error Handling

- If target preparation fails before source releases a request, the source keeps
  the local request and resumes local decode by default.
- If transfer fails after source has released local KV, the target marks the
  migrated request failed and exposes the error through `/server_info`.
- Source and target cleanup must clear sender/receiver state, metadata buffer
  indices, and KV allocations.
- Abort requests should abort any active migration sender or receiver for the
  matching request id.
- The demo script should time out with a clear source/target status summary.

## Testing

Unit tests:

- flip status reports the decode-to-decode migration strategy when enabled
- source migration session selects only active decode requests
- source migration waits before prepare ack while requests are pending
- target migration manifests create preallocated decode requests
- failure paths release metadata and KV allocations

Integration-style tests with fake transfer:

- one source decode request is exported, prepared on target, transferred, and
  admitted into the target waiting queue
- failed fake sender or receiver is reported in migration status
- D-to-P `prepare_pd_flip` can advance after migrated requests are released

Manual remote demo:

- run one prefill worker and at least two decode workers
- start router in PD mode
- issue a long generation request to the source decode worker
- trigger D-to-P flip with `pd_flip_experiment.py --migration-target-url`
- verify target decode reports transferred requests and continues scheduling
- verify source decode can advance past prepare without waiting for natural
  completion of the migrated request

## Rollout

The implementation stays experimental and opt-in:

- gated by `--enable-pd-flip-state-machine`
- additionally gated by a migration control flag or by the experiment endpoint
- normal router selection continues to skip workers in `preparing` or `flipping`
- no production behavior changes unless the experiment script invokes migration
