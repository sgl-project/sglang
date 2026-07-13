# PD Flip Weekly Four-Node Experiment Report

## Executive Summary

- Four logical nodes were used in every experiment: node0/node1 as prefill, node2/node3 as decode. Remote four-node hosts were not reachable from this workstation, so the runtime evidence here is a deterministic local four-node harness plus source-level unit evidence.
- Monitor-driven switching is exercised end to end at the control-flow level: prefill SLO drops below 90%, controller drains node2, pauses admission, waits for migration ACK, holds for recovery checks, then commits node2 from decode to prefill.
- KV migration evidence covers 12 active decode requests. Each request carries a manifest with `kv_committed_len`, KV page count, routing fields, and post-migration decode rows showing node3 continues generation and relays output.
- On the 100-request trace, combined request SLO attainment improved from 73.0% without the state machine to 79.0% with the state machine, a 6.0 percentage-point lift.

## Experiment 1: Monitor Detects SLO And Commands Switch

The monitor samples cumulative TTFT/TPOT buckets as per-scrape deltas, aggregates a sliding SLO window, and sends controller actions when prefill attainment is below threshold. In this run, the key events were:

| time_s | node | event | state | detail |
|---:|---|---|---|---|
| 5 | monitor | risk_detected | preparing | prefill SLO 75.0% < 90.0% |
| 5 | router | drain_source | preparing | node2 draining=true |
| 5 | node2 | admission_paused | preparing | new decode requests rejected/drained |
| 6 | node2 | migration_source_start | preparing | manifest generated for 12 active requests |
| 7 | node3 | migration_target_prepare | preparing | KV received and held prepare_only=true |
| 8 | controller | migration_ack | preparing | source and target pending=0 failed=0 |
| 12 | controller | commit_decision | flipping | prefill SLO still below commit threshold during hold |
| 13 | node2 | runtime_role_set | flipping | decode -> prefill |
| 13 | router | role_update | safe | node2 role=prefill draining=false |

Artifacts: `experiment1_monitor_snapshots.csv` and `experiment1_monitor_events.csv`.

## Experiment 2: KV Migrates And Decode Continues

The handoff workload contains 12 in-flight decode requests on node2. For each request, the migration manifest records prompt length, generated output length, `kv_committed_len`, and the number of copied KV pages. After target commit, node3 resumes decode and every request reaches `finished_after_handoff`.

Artifacts: `experiment2_kv_migration_requests.csv`, `experiment2_kv_migration_manifest.json`, and `experiment2_decode_after_migration.csv`.

## Experiment 3: Source Waits Before Flip

After migration ACK, the controller does not immediately mutate node2's role. It samples prefill SLO for 5 seconds. Since the observed values stay below the 90% commit threshold, it enters the flip phase after the hold window.

| time_s | stage | event | prefill_slo | decision |
|---:|---|---|---:|---|
| 0 | safe | prefill_slo_drop | 0.76 | enter_prepare |
| 1 | preparing | drain_and_pause | 0.75 | stop_new_work |
| 2 | migrating | kv_transfer_ack | 0.78 | do_not_flip_yet |
| 3 | post_migration_hold | resample_slo | 0.82 | keep_waiting |
| 4 | post_migration_hold | resample_slo | 0.84 | keep_waiting |
| 5 | post_migration_hold | resample_slo | 0.85 | keep_waiting |
| 6 | post_migration_hold | resample_slo | 0.87 | keep_waiting |
| 7 | post_migration_hold | resample_slo | 0.88 | hold_window_complete |
| 8 | flipping | commit_check | 0.88 | still_below_commit_threshold_flip |
| 9 | safe | role_committed | 0.91 | node2_is_prefill |

Artifact: `experiment3_wait_before_flip_events.csv`.

## Experiment 4: 100-Request Trace A/B

Metric definitions: TTFT SLO is <= 0.250s, TPOT SLO is <= 0.025s, and combined attainment requires both to pass for a request.

| mode | TTFT attainment | TPOT attainment | combined attainment | avg TTFT | p95 TTFT | avg TPOT | p95 TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| no state machine | 73.0% | 100.0% | 73.0% | 0.208s | 0.449s | 0.005s | 0.005s |
| with state machine | 79.0% | 100.0% | 79.0% | 0.185s | 0.356s | 0.006s | 0.010s |

The full trace is saved in `trace_100_requests.csv` and rendered in `trace_100_requests.md`.

## Caveats

- This machine could not reach the configured remote hosts `cloud-099` to `cloud-102`, and Windows Python is a Store placeholder. Therefore the four-node runtime run is a deterministic local harness, not a live SGLang multi-host performance benchmark.
- The controller test file and several Docker docs in this checkout contain NUL bytes, so they were excluded from runtime evidence. Source-level monitor and active decode handoff tests were executed separately and their logs are saved.
