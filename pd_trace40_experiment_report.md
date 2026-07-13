# 40-request PD flip experiment report

## Configuration

- Artifact root: `/home/tiancij/pd-artifacts/20260713-095054-trace40`
- Model: Qwen3-8B
- Topology: node0 prefill; node1/node2/node3 decode
- Migration: node2 decode -> node3 decode, then intended node2 decode -> prefill
- Request count: 40 (20 long and 20 short, strictly interleaved)
- Input size: long 10,000 characters; short 1,000 characters
- Final measurement run: burst arrival, 4,096 output-token budget per request
- Per-request constructed SLOs (source trace had no SLO fields):
  - long: TTFT <= 8.0 s, TPOT <= 35 ms
  - short: TTFT <= 3.0 s, TPOT <= 25 ms
- Initial migration ratio: 0.5; selection policy: first N requests
- Sidecar sampling interval: 50 ms

## Result

The basic P->D path passed before this experiment (HTTP 200). In the final
40-request run, all 40 requests completed without workload errors. Node2 had
13 concurrent decode requests before the controller started. At migration
start the controller observed 6 eligible running requests and selected the
first 3 (ratio 0.5).

The D->P workflow did not reach role commit. The target entered
`partial_prefix_stitch`, but all three selected requests failed during
target HiCache restore. The source reported
`KVTransferError: Failed due to an unknown reason from another rank`; the
target reported `migration target HiCache restore failed`. The controller
then aborted and safely restored source admission and router admission. Node2
remained decode.

## Measured stage timings

Controller HTTP action latency:

| Stage | Seconds |
|---|---:|
| Router drain source | 0.000328 |
| Pause source admission | 0.007249 |
| Observe source quiesce/status | 0.015645 |
| Build/start source migration (6 manifests) | 0.027034 |
| Target prepare + HiCache restore attempt | 0.062935 |
| Wait/check source migration | 0.087720 |
| Wait/check target migration | 0.032365 |
| Target abort request | 0.023550 |
| Source abort | 0.006186 |
| Resume source admission | 0.005977 |
| Router undrain source | 0.000308 |

50 ms sidecar timeline (polling timestamps):

| Transition | Seconds |
|---|---:|
| Admission paused -> source migration started | 0.013388 |
| Source migration started -> migration failed | 0.085893 |
| Failure observed -> router source drained observation | 0.021248 |
| Router drained observation -> cleanup undrain | 0.139424 |

Because failure occurred in initial target restore, these stages were not
reached and therefore have no valid timing: observation window, delta
migration, source finish, target activation, runtime role switch, router role
commit.

## Workload SLO result

- Requests completed: 40/40
- Request errors: 0
- Run elapsed time: 31.426 s
- TTFT attainment: 39/40 = 97.5%
- Average-TPOT attainment: 39/40 = 97.5%
- P95-TPOT attainment: 39/40 = 97.5%
- TPOT interval attainment: 81,812 / 81,856 = 99.946%
- Both SLOs met: 39/40

## Raw evidence

- Interleaved trace: `trace/trace_interleaved_burst.jsonl`
- Per-request metrics: `workload/run5/request_metrics.jsonl`
- Per-token TPOT data: `workload/run5/tpot_tokens.csv`
- Request-level SLO ledger: `workload/run5/trace_slo_ledger.jsonl`
- Controller raw output: `controller/controller_run5.log`
- Controller journal: `controller/pd_flip_session_run5.json`
- 50 ms raw samples: `metrics/migration_events_run5.jsonl`
- Worker/router/Mooncake logs: `logs/`
- Derived CSV/JSONL: `report/summary/`

Secrets in collected controller and service logs were replaced with
`[REDACTED]`.

## Diagnosis and next action

The experiment validates request construction, per-request SLO accounting,
P->D routing, source drain/admission control, fixed-ratio first-N selection,
manifest construction, target prepare, failure detection, and safe rollback.
It does not validate successful dual-source stitching or D->P commit.

The next debugging target is target-side HiCache restore for
`partial_prefix_stitch`: compare the three selected manifests' P/H/C0/C1
layout and Mooncake object availability, then rerun the same trace after the
restore failure is fixed.
