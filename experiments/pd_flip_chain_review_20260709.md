# PD Flip Chain Review 2026-07-09

## Raw Bundles

- `experiments/pd_flip_waiting_queue_20260709_112341_fullfix.tar.gz`
  - Shows full source/start migration after metadata-buffer fix.
  - Source manifest: 6 requests total, 2 running + 4 waiting, `waiting_skipped_count=0`.
  - Outcome: failed safely because running requests advanced after the migration snapshot.

- `experiments/pd_flip_waiting_queue_20260709_114412_orderfix.tar.gz`
  - Re-run after controller order fix with cluster reused.
  - Replay completed 40/40 requests with 0 request errors.
  - Outcome: still failed safely on running-request advancement, but target did not commit before source finish.

## Verified Fixes

- Removed the old partial-migration cap: `PD_FLIP_MIGRATION_MAX_REQS` is unbounded by default.
- Source now refuses partial live waiting migration instead of silently skipping eligible requests.
- Waiting queue requests with committed KV are included in migration manifests.
- Target chosen explicitly for migration stays drained before flip, avoiding normal traffic consuming its req pool.
- PD metadata buffers now have a 1024-slot floor for runtime role switch migration.
- Controller now fails fast on source/target migration failure instead of waiting for timeout.
- Controller now finishes source before committing target, so stale source snapshots cannot be adopted by target first.

## Current Blocking Issue

The remaining hard blocker is running-request consistency while the source continues decoding during KV transfer.

Observed failure:

```text
running requests advanced after migration snapshot; delta KV transfer is required before source release
```

In the fullfix run, two running requests advanced during transfer:

```text
2728 -> 2770
2587 -> 2770
```

In the orderfix run, two running requests advanced during transfer:

```text
2663 -> 2770
2643 -> 2770
```

This is expected with the current one-shot migration: source snapshots KV at source/start, then running decode continues producing more committed KV and output tokens. Without a delta transfer/update phase, target has an older request state.

## Stage Timing From Fullfix Run

| Stage | Next stage | Duration |
| --- | --- | ---: |
| router_source_drained | source_admission_paused | 15.856 s |
| source_admission_paused | source_migration_started | 13.918 s |
| source_migration_started | target_migration_prepared | 0.023 s |
| target_migration_prepared | kv_transfer_first_progress | 0.816 s |
| kv_transfer_first_progress | kv_transfer_complete | 1.376 s |
| kv_transfer_complete | migration_abort_or_failed | 0.445 s |

## Required Next Design

To satisfy "source keeps working during transfer", we need a delta/chasing phase:

1. Initial snapshot transfer moves running + waiting committed KV.
2. Before source release, source compares current `kv_committed_len/output_ids` against snapshot.
3. For advanced running requests, source sends delta KV pages and updated request metadata/output ids to target.
4. Repeat until the source state is stable, then finish source.
5. Commit target only after source finish succeeds.

If we do not implement delta, the only safe alternative is to freeze running requests at snapshot time, which makes the experiment pass but does not match the desired "source continues working during transfer" behavior.
