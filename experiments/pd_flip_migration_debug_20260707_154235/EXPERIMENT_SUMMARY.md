# PD Flip Migration Debug Experiment

Run directory: `experiments/pd_flip_migration_debug_20260707_154235`

## Setup

- Nodes: cloud099/node0 prefill, cloud100/node1 prefill, cloud101/node2 decode, cloud102/node3 decode.
- Mode: SGLang with PD flip state machine enabled.
- Trace: 80 mixed short/medium/long requests, 1 second arrival interval.
- Raw trace files: `trace_requests.jsonl`, `trace_requests.csv`.

## Migration Chain

Outcome: committed.

Source: node2 (`http://192.168.0.39:32000`) flipped from decode to prefill.
Target: node3 (`http://192.168.0.41:32000`) prepared as migration target.

Observed chain:

1. `router_source_drained`
2. `source_admission_paused`
3. `source_migration_started`
4. `target_migration_prepared`
5. `kv_transfer_first_progress`
6. `kv_transfer_complete`
7. `source_role_committed`
8. `cleanup_router_undrain`

End-to-end observed chain time: 1.200905 s from router drain to router undrain.
KV migration visible time: 0.396014 s from source start to KV complete.

See:

- `migration_link/migration_timeline.csv`
- `migration_link/migration_stage_durations.csv`
- `migration_link/migration_status_samples.csv`
- `migration_link/router_worker_samples.csv`
- `migration_link/worker_pd_flip_samples.csv`
- `migration_link/controller_actions.csv`
- `migration_link/controller_state_trace.csv`

## Request Impact

- Total requests: 80
- Completed: 79
- Error: 1
- TTFT SLO met: 49/80 = 61.25%
- TPOT average SLO met: 59/80 = 73.75%
- TPOT interval SLO met: 11187/11212 = 99.7770%
- All SLO met: 49/80 = 61.25%
- Migration phase labels: before 6, overlaps 9, after 65.

Important outlier:

- `trace-0013` overlapped the migration window, received 146 tokens, met TTFT and average TPOT, then timed out after the 900s replay socket timeout.

See:

- `state_machine/request_metrics.jsonl`
- `state_machine/ttft.csv`
- `state_machine/tpot.csv`
- `state_machine/tpot_tokens.csv`
- `state_machine/slo_summary.csv`
- `state_machine/errors.jsonl`
- `migration_link/request_impact_by_stage.csv`

## Raw Logs

- `migration_events.jsonl`: 25,285 sidecar polling events.
- `state_machine_monitor.log`: controller monitor output.
- `state_machine_replay.log`: replay summary.
- `logs/router.log`
- `logs/node0_worker.log`
- `logs/node1_worker.log`
- `logs/node2_worker.log`
- `logs/node3_worker.log`

