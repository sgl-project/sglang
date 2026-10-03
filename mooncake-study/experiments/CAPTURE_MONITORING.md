# Capture Monitoring Runtime Verification

This experiment exercises actual SGLang capture, a real Mooncake TCP Store,
Prometheus and the provisioned Grafana dashboard. It uses the existing test
Catalog and a single H100 with Qwen3-0.6B. It validates monitoring correctness;
it does not establish production latency thresholds, Kubernetes service
discovery, cross-node monitoring or production Catalog behavior.

## Changes

The current dashboard additionally exports cohort routing decisions and local
ticket events, and preserves TP/PP labels on the lifecycle panel. See
[cohort routing verification](ROUTING_METRICS.md) for the later distributed run;
the original fifteen-panel results below remain evidence for their recorded
source version.

- `device_allocated_bytes` and `device_limit_bytes` describe the full capture
  device arena, including teacher staging and HiCache pointer/position tables.
  Existing `kv_staging_*` gauges remain aliases with corrected descriptions.
- `kv_export_enqueued_bytes_total{destination="host|device"}` exports the existing
  optional HiCache byte counters through the background metrics thread. It is
  zero for the default Torch exporter. It counts submitted copy work, including
  lookahead and later-aborted captures, not completed transfers, published
  dataset bytes or RDMA bandwidth.
- Two panels show capture device allocation/budget and HiCache enqueue rate.
  The existing thirteen panels and default exporter are unchanged.

## Reproduction

Use the pinned H100 runtime in `h100-runtime-lock.json`. Start Prometheus before
the producer so it records the entire capture/pause/resume sequence. Configure
a two-second scrape interval and a target reachable at the producer's
`/metrics` endpoint. Configure the Grafana Prometheus datasource with the same
two-second minimum interval. Provision the repository's dashboard JSON through
the existing monitoring example. The default verifier addresses are
Prometheus `127.0.0.1:19090`, Grafana `127.0.0.1:13000`, datasource UID
`capture-prometheus`, producer target `127.0.0.1:18081`.

Run in an isolated output directory on the GPU host:

```bash
PYTHONPATH=python:test/registered/unit/training_capture \
  python mooncake-study/experiments/verify_capture_monitoring.py \
  --model-path /models/Qwen3-0.6B \
  --source-revision "$(git rev-parse HEAD)" \
  --output-dir /tmp/capture-monitoring-runtime \
--port 18081 --hold-seconds 180
```

Add `--tp-size 2` to exercise the cohort router with two CUDA devices. The
driver waits for admission at startup and resume, and also compares routing
counters with the HTTP metrics endpoint. Status comparisons select TP0/PP0;
the metrics endpoint and charts retain all ranks. The separate backpressure
tests also verify every rank's routing series against its local state. Single-rank runs
have no cohort router, so the browser verifier requires the two routing panels
to show no data; a cohort run requires actual data in both panels.

The driver requires the experiment helpers and local Mooncake master used by
`benchmark_training_capture.py`. It starts a fresh Store, test Catalog and
producer, uses sixteen-token prompts and thirty-two-token replies at batch
size four, and alternates streaming/non-streaming requests. The three equal
time phases are capture, operator pause and operator resume. Each request must
finish; READY count must equal the requests in capture-enabled phases. Every
iteration compares the real `/metrics` gauges/counters with producer status.
After stopping SGLang, a separate Store client reads every published manifest
and tensor, checks hashes/schema and exact request membership. The driver
retains `ready.json`, `timeline.jsonl`, `.prom` scrapes and `report.json` on
failure as well as success.

The 60-second TTFT/TPOT control budgets are deliberately loose fixture values
to exercise the protection metrics without suppressing capture. They are not
proposed production SLOs. This driver validates Store snapshot integrity, not
an independent numerical comparison against online KV/logits; the unchanged
exporter has separate numerical/content runtime coverage in `KV_HICACHE.md`.

After the driver completes, keep Prometheus history and Grafana running. With
Node and Playwright installed, run on the monitoring host:

```bash
node mooncake-study/experiments/verify_capture_dashboard.cjs \
  --runtime-dir /tmp/capture-monitoring-runtime \
  --output-dir /tmp/capture-monitoring-dashboard \
  --grafana http://127.0.0.1:13000 \
  --prometheus http://127.0.0.1:19090 \
  --datasource capture-prometheus \
  --model capture-monitoring-qwen3 \
  --instance 127.0.0.1:18081
```

`NODE_PATH` may point to a separate Playwright installation; browser binaries
can use `PLAYWRIGHT_BROWSERS_PATH`. The runtime directory must be readable on
the monitoring host. The verifier requires a completed runtime report and an
identical dashboard SHA256. Use a fresh output directory for every attempt.
If the monitoring hosts use authentication, supply it through a locally
configured authenticated proxy; this fixture's verifier assumes local access.

It verifies the provisioned panel definitions, Prometheus scrape continuity,
all panel expressions with both all-instance and selected-instance
filters, and a negative instance filter. It independently checks the pause
gauge, stable READY/byte counts while paused, and increasing publication/export
counts before and after the pause. Browser checks load all current panels at
desktop and mobile sizes, require nonempty Grafana datasource responses for
applicable panels,
inspect plot pixels and dimensions, and save per-panel screenshots. A missing
streaming latency series is a failure, not a successful empty chart. Inactive
writer stages may have NaN means; each expression must still return finite
data for the exercised stages.

## Local Stack and Evidence

The recorded run uses Grafana OSS 13.2.3, Prometheus 3.15.0, Playwright 1.56.1
and Chromium 141.0.7390.37. Download archives were checked against their
official SHA256 values:

| Archive | SHA256 |
| --- | --- |
| Grafana 13.2.3 Linux amd64 | `6107ad27016296aac38e0d7ffa8753ab540b5541ad27e94790f771289d733235` |
| Prometheus 3.15.0 Linux amd64 | `2a542df32eac02ee17b9d844fb2aa1de00dafa5476579ba8a3ba862e9d572ea0` |

The cluster denies `pods/portforward`. The local Prometheus target therefore
uses a loopback HTTP relay that reads the producer's actual `/metrics` via the
existing authorized pod `exec` path. It forwards the response bytes without
rewriting metrics and returns HTTP 502 when the producer is absent. That relay
is only a correctness fixture; its RPC overhead is unsuitable for interpreting
service performance. No RBAC permission was changed. Native deployment scrape
transport and authentication remain separate acceptance work.

The first browser probes failed because the environment inherited the invalid
BCP47 locale `en-US@posix`. Explicit Playwright `locale: 'en-US'` fixes the
fixture without changing the dashboard. Grafana's anonymous Viewer issues one
401 request to `/api/user/stars`; the verifier records and narrowly permits
that known non-query endpoint. Other HTTP failures and datasource errors fail
verification.

The first traffic run failed in its post-exit validator because it iterated
Catalog dictionary keys instead of publication records. The corrected run
uses `.values()` and strengthens exact request membership and paused READY
count checks. A subsequent non-streaming run passed runtime checks but could
not cover the serving inter-token latency histogram. The final fixture adds
streaming traffic. All attempts and their source identities are retained in
the evidence; successful later runs do not replace earlier failures.

Browser acceptance also retains two verifier failures: an early page-title
check before Grafana finished initializing, and an incorrect assumption that
`$__rate_interval` must be expanded in the browser. The final verifier waits for
the dashboard title and checks the datasource response's `executedQueryString`
for fully expanded backend queries.

## Results

The [machine-readable evidence](capture-monitoring.json) records frozen sources,
job outcomes, tool versions, artifact hashes and the exact dashboard time range.

| Check | Result |
| --- | --- |
| Metrics unit methods | 11 passed, 0.061 seconds |
| Corrected non-streaming runtime | 160 requests, 104 post-exit snapshots, 64,846,080 payload bytes |
| Mixed streaming runtime | 156 requests, 96 post-exit snapshots, 59,857,920 payload bytes |
| Paused phase in mixed run | 60 successful requests, zero new captures; READY stays 48 |
| Prometheus | 19 expressions with all and selected filters: 38 checks; continuous successful scrapes within the measured interval |
| Grafana browser | 15 panels and 19 distinct datasource expressions in each of 3 views; 190 responses including refreshes |
| Browser viewports | Desktop 1440x1000, mobile 390x844; 45 panel screenshots plus 3 viewport screenshots |

The final runtime reports 1,717,248 allocated capture device bytes against a
16,777,216-byte budget. HiCache enqueues 18,874,368 Host bytes and 37,748,736
device-staging bytes. The paused phase holds both publication and enqueue
counters constant, with progress observed before and after the pause.

Raw reports, `.prom` scrapes, Prometheus query results, screenshots, fixture
configuration and logs are archived under
`/gpfs/user/fuxuanwei/mooncake-lab-archive/capture-monitoring-20261003`.
Its manifest hashes 369 artifacts. Private writer journals are not part of this
archive; the driver completed independent Store reads before Store shutdown.
The default exporter remains Torch. Production SLOs, alerting, broader workloads
and multi-node rollout/rollback acceptance remain open.
