# Cohort Routing Metrics

Distributed capture chooses requests in `CaptureRequestRouter`, before local
coordinators bind the same ticket on each rank. Those decisions were present in
`/server_info` but absent from Prometheus, whose lifecycle exporter only read
coordinator counters. The pre-fix regression expects eight routing exclusions
and receives no metric sample.

## Metric Contract

`sglang:training_capture_routing_events_total{event=...}` exports the existing
router state from the background metrics thread. It adds no request-path
Prometheus call, collective, GPU synchronization or sampling decision.

| Events | Ownership |
| --- | --- |
| `considered`, `excluded`, `sampled_out`, `backpressure`, `selected`, `selection_failed` | Ingress only |
| `attached`, `attachment_failed`, `bound`, `binding_failed`, `cancelled` | Local ticket lifecycle on each rank |
| `other` | Bounded aggregation of unknown names; never an arbitrary label |

`excluded` includes health checks and unsupported requests. `sampled_out`
includes the effective sampling probability, readiness gate and operator pause;
it does not identify adaptive pressure alone. `backpressure` means selection
could not claim a ticket. Bindings across ranks are the same request, not
additional samples. Catalog remains the authority for complete dataset counts.
Existing lifecycle counters retain their meaning. Producers without a cohort
router have no samples in the routing family. Repeated refreshes are idempotent.

The dashboard separates ingress decisions from ticket events and preserves
TP/PP labels on the lifecycle panel. Thus several rank-local admissions cannot
appear as an unlabelled total next to the publisher's READY count.

## Reproduction

Use the pinned CUDA/Mooncake environment and local Qwen3-0.6B weights.

```bash
python -m unittest discover -s test/registered/unit/training_capture -p 'test_*.py' -v
python test/registered/storage/test_training_capture_cohort_backpressure.py -v -f
```

The second command uses two GPUs, running TP2 and PP2 separately. It compares
every rank's twelve routing series to its nonce-tagged status during a manifest
stall, checks exactly two ingress selections and eight skipped requests, and
requires two bindings per rank. Continued serving, recovery and post-exit
Store content checks remain part of the same test.

For the full dashboard check, follow [the monitoring runbook](CAPTURE_MONITORING.md)
and add `--tp-size 2` to `verify_capture_monitoring.py`. The driver compares
TP0/PP0 status to that rank's metrics while retaining all rank series in raw
scrapes and Grafana. It waits for admission before initial traffic and after
resume. Use an isolated Prometheus history and Grafana data directory.
The browser verifier requires actual data in all seventeen panels for a cohort
run. A single-rank run must show no data in its two inapplicable routing panels;
query errors remain failures in either mode.

## Evidence

On 2026-10-03, all 287 unit methods and both TP2/PP2 backpressure methods passed.
The latter complete 22 requests and validate six post-exit snapshots. The TP2
monitoring run completes 144 requests, with 48 during pause and 96 complete
post-exit snapshots / 59,857,920 payload bytes. Its 36 observed batches cover all
three phases equally.

Grafana passes 42 expression/filter checks and renders all seventeen panels in
three views: 1440x1000 desktop with all/selected instances, and 390x844 mobile.
The 198 datasource responses contain finite values; 54 screenshots and plot-pixel
checks are retained. An initial browser launch lacked the existing sysroot's
`lib` directory in `LD_LIBRARY_PATH`; adding it resolves `libdbus-1.so.3` without
changing the dashboard or runtime sources.

The [machine-readable report](routing-metrics.json) retains job IDs, failed
baseline/fixture attempts, final runtime counts, source identities, Prometheus
query checks, browser viewports, screenshots and cleanup evidence. The first
distributed monitoring attempt exposed the old fixture's one-producer
assumption; the corrected comparison selects the rank matching `/server_info`.

The loopback relay forwards unmodified metrics using existing pod exec access.
It is a test transport, not production service discovery. Qwen3-0.6B BF16,
Triton, TCP Store and the HTTP test Catalog establish correctness of this
monitoring path. They do not establish serving SLOs, production retention,
cross-node monitoring or draft-model quality.
