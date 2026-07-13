# PD Flip Waiting Queue Migration Design

Date: 2026-07-08

## Goal

Extend the PD flip D->P KV migration path so decode requests parked in
`Scheduler.waiting_queue` can be migrated when they already hold stable KV
state. The experiment output must show the added waiting-queue stages on the
full-link latency diagram.

## Scope

This change migrates only waiting requests that can safely use the existing KV
transfer protocol:

- `req_pool_idx` is present.
- `kv_committed_len` is positive.
- `output_ids` is non-empty, so the target can resume decode from a generated
  token boundary.

Waiting requests that do not satisfy these checks are not migrated by this
patch. They remain observable through skip counters and skip reasons, and they
can still block source idle until the normal scheduler drains them.

## Architecture

The source migration start path will scan both `running_batch.reqs` and
`waiting_queue`. It will build the existing manifest shape for eligible
requests and add queue metadata such as `pd_flip_source_queue`.

Eligible waiting requests are frozen by removing them from the source
`waiting_queue` after all source entries are successfully initialized. If the
session aborts, the original waiting requests are restored to the source queue
in their original order. If the session commits, the source-side waiting
requests are released as migrated KV state without using the running-request
`FINISH_MIGRATED()` completion path.

The target side keeps the current behavior: migrated requests are received into
`transferred_held`, then `commit` adopts them by appending reconstructed
requests to the target `waiting_queue`.

## Data Flow

1. Source receives `source/start`.
2. Source scans `running_batch.reqs`.
3. Source scans `waiting_queue`.
4. Source classifies waiting requests as eligible or skipped.
5. Source builds manifests for running plus eligible waiting requests.
6. Source initializes KV sender entries for all manifests.
7. Source freezes eligible waiting requests by removing them from
   `waiting_queue`.
8. Target prepares receivers and receives KV blocks.
9. Target holds requests in `transferred_held`.
10. On commit, target appends requests to target `waiting_queue`.
11. Source finish releases source KV and metadata.
12. On abort, source restores frozen waiting requests.

## Observability

The migration status payload will include:

- `waiting_reqs`
- `waiting_manifest_count`
- `waiting_skipped_count`
- `waiting_skipped`
- per-entry `source_queue`
- waiting scan/build/freeze/restore/release timings

The full-link diagram will add labels for waiting-queue scanning, waiting
manifest creation, waiting freeze, target held adoption, and source waiting
release.

## Error Handling

If source entry initialization fails, no waiting request is removed from the
source `waiting_queue`.

If the source session aborts after freezing waiting requests, each frozen
request is restored once, preserving the original queue order as much as
possible.

If the target session aborts before adoption, the existing target release path
continues to release target-side KV state.

## Testing

Add scheduler-level unit coverage for:

- Eligible waiting requests are included in source manifests.
- Ineligible waiting requests are skipped with reasons.
- Waiting requests are frozen only after source entry initialization succeeds.
- Abort restores frozen waiting requests.
- Source finish releases waiting entries without setting `FINISH_MIGRATED()`.

## Spec Self-Review

- No placeholder requirements are left.
- The target behavior intentionally reuses the existing target adoption path.
- The patch excludes no-KV, prealloc, transfer, offload, and grammar queue
  requests from migration; those states require separate protocols.
