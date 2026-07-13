# PD Flip Migration Link Measurement Plan

Goal: measure every observable step in one PD flip migration attempt, from SLO risk through router drain, source admission pause, KV source/target migration, commit or abort, cleanup, and request impact.

Collected raw files:
- `migration_events.jsonl`: sidecar polling raw events.
- `migration_timeline.csv` / `migration_timeline.jsonl`: first-observed chain stages.
- `migration_stage_durations.csv`: time from each observed stage to the next stage.
- `migration_status_samples.csv`: per-node `/pd_flip/migration/status` samples, including pending/transferred/failed/released/held counts and index debug.
- `router_worker_samples.csv`: router role/drain/load view.
- `worker_pd_flip_samples.csv`: worker state-machine view from `/server_info`.
- `controller_actions.csv`: controller HTTP actions parsed from the monitor log.
- `controller_state_trace.csv`: controller high-level state trace.
- `request_impact_by_stage.csv`: request metrics labeled as before, overlaps, during, or after migration.

Interpretation:
- A successful chain should show `router_source_drained`, `source_admission_paused`, `source_migration_started`, `target_migration_prepared`, `kv_transfer_first_progress`, `kv_transfer_complete`, `source_role_committed`, and `cleanup_router_undrain`.
- If `source_migration_started` appears without `kv_transfer_first_progress`, KV transfer did not visibly progress.
- If `migration_abort_or_failed` appears, inspect `migration_status_samples.csv` and `controller_actions.csv` around the same timestamps for the failing node, `last_error`, pending request count, and index debug.
