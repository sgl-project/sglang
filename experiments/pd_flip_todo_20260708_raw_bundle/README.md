# PD flip todo raw bundle, 2026-07-08

This bundle contains the 40-request trace runs requested for the PD flip todo.

Trace shape for all included runs:
- 40 requests, arrival interval 0.5s.
- 20 short prompts at 1000 chars and 20 long prompts at 10000 chars.
- Source code changes include the new `observe_source_quiesce` stage, bounded `post_migration_idle_assertion`, explicit migration target selection, and flattened `worker_load_samples.csv`.

## Runs

### `pd_flip_1p3d_to_2p2d_20260708_124626`

Successful control-chain run.

- Initial roles: node0=prefill, node1/node2/node3=decode.
- Forced source: node2.
- Explicit migration target: node3, kept router-draining so it only receives KV migration traffic.
- Controller result: success, `pd flip committed after two-phase migration`.
- Requests: 40 completed, 0 errors.
- SLO: TTFT attainment 0.05, TPOT avg/p95 attainment 1.0, all attainment 0.05.
- Important latency points:
  - observe source quiesce: 15.006s, final residual=0.
  - controller migration section: 31.33ms.
  - target prepare: 2.68ms.
  - commit target: 2.10ms.
  - finish source: 4.61ms.
  - post-migration idle assertion: 1.58ms.
  - runtime role switch: 2.25ms.

Because residual reached 0 during the observation/quiesce window, this run validates the "wait source idle is covered by observation" flow but does not contain actual KV payload transfer.

### `pd_flip_1p3d_to_2p2d_20260708_121443`

KV-transfer stress run.

- Same 1P3D->2P2D setup.
- Controller result: failed at post-migration idle assertion.
- Requests: 38 completed, 2 HTTP 500 errors.
- It does contain real KV transfer:
  - observe source quiesce: 15.020s, final residual=6.
  - scanned running=2, waiting=4, skipped=2.
  - built 4 manifests, including 2 waiting-origin manifests.
  - KV transfer to target held: 1.912s.
  - controller migration section: 2.271s.
  - commit target: 8.65ms.
  - finish source: 13.36ms.
  - post-migration idle assertion: 2.009s timeout.

Interpretation: `PD_FLIP_MIGRATION_MAX_REQS=4` migrated 4 requests, but source still had 2 residual requests, so the bounded post-idle check failed and cleanup/abort ran.

### `pd_flip_2p2d_to_3p1d_20260708_123441`

2P2D->3P1D comparison run.

- Initial roles: node0/node1=prefill, node2/node3=decode.
- Requests: 40 completed, 0 errors.
- Controller result: two-phase D->P timed out.
- Observation found residual=2, but node3 also had active/waiting user traffic by the time it was selected as target, so this run is useful as a target-reschedule failure case rather than a clean success case.

## Key files inside each run

- `suite_manifest.json`: experiment parameters.
- `reschedule_stitching_manifest.json`: present in 1P3D runs; documents router/reschedule/KV stitching assumptions.
- `trace_*/trace_requests.jsonl` and `.csv`: constructed trace.
- `<run>/<mode>/request_metrics.jsonl`: TTFT/TPOT raw metrics per request.
- `<run>/<mode>/trace_slo_ledger.jsonl`: SLO ledger.
- `<run>/<mode>/summary.json`: SLO summary for that mode.
- `<run>/migration_link/migration_events.jsonl`: raw sidecar polling events.
- `<run>/migration_link/worker_load_samples.csv`: flattened running/waiting/transfer queue samples.
- `<run>/migration_link/controller_actions.csv`: controller action timings.
- `<run>/migration_link/migration_status_samples.csv`: source/target migration status samples.
- `pd_state_machine_full_chain_latency.csv` and `.svg`: full-chain latency table and diagram.
