# PD flip waiting_queue full-chain observation

Run: `pd_flip_waiting_queue_20260708_102000`

## Result

- Controller result: success, `pd flip committed after two-phase migration`
- Total D->P flip time: 12.741526 s
- Controller migration window: 0.537262 s
- Migrated manifests: 2 total = 1 running request + 1 source `waiting_queue` request
- Source waiting scan: 2 waiting requests observed, 1 migrated, 1 skipped by `max_reqs`
- Replay result: 60 requests, 58 completed, 2 socket timeout errors

## Key stage latencies

| Step | Stage | Latency |
|---:|---|---:|
| 1 | Router marks source draining | 0.28 ms |
| 2 | Pause source admission | 4.86 ms |
| 3 | Scan running + waiting | 0.02 ms |
| 4 | Build manifests | 0.09 ms |
| 5 | Freeze source waiting_queue | 0.01 ms |
| 6 | Target prepare receiver | 8.44 ms |
| 7 | KV transfer to target held | 112.92 ms |
| 8 | Enter transferred_held queue | 0.00 ms |
| 9 | Target commit/adopt | 2.77 ms |
| 10 | Source finish/release | 3.98 ms |
| 11 | Wait source idle | 12.188 s |
| 12 | Set runtime role decode -> prefill | 2.27 ms |
| 13 | Refresh router role | 0.34 ms |
| 14 | Resume admission | 1.88 ms |
| 15 | Router undrain | 0.27 ms |

## Waiting_queue-specific timings

- Waiting request source freeze: 0.01 ms for 1 request
- Waiting request source KV send -> transferred: about 102.70 ms
- Waiting request target transferring -> held: about 112.82 ms
- Waiting request held -> target adopted: about 409.14 ms
- Waiting request source transferred -> source release: about 420.58 ms

## SLO replay summary

- TTFT attainment: 1 / 60 = 1.67%
- TPOT avg attainment: 59 / 60 = 98.33%
- TPOT p95 attainment: 59 / 60 = 98.33%
- TPOT interval attainment: 100.00%
- Error requests: `trace-0053`, `trace-0054`, both `socket.timeout`

## Files to read first

- `pd_state_machine_full_chain_latency.svg`: full PD state-machine chain diagram with latency labels.
- `pd_state_machine_full_chain_latency.csv`: same latency data as a table.
- `01_waiting_queue_two_phase/controller.log`: authoritative controller actions and per-entry timing_debug.
- `01_waiting_queue_two_phase/waiting_queue_state_machine/{ttft.csv,tpot.csv,slo_attainment.csv,request_metrics.jsonl}`: request-level raw SLO data.
- `01_waiting_queue_two_phase/migration_link/*.csv`: sampled link/status raw measurements.

Note: `migration_link_summary.json` mislabels the experiment-prep drain of node3 as an aborted/failed migration stage. The authoritative migration result is the controller log and the generated full-chain latency CSV/SVG.
