<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
SPDX-License-Identifier: Apache-2.0
-->

# sgl-router (experimental) monitoring

Grafana dashboard for the experimental router's Prometheus metrics, exposed
on `/metrics` (text/plain, version 0.0.4) on the router's serving port
(default `30000`).

## Files

- `grafana-dashboard.json` — importable Grafana dashboard, **SGLang Router
  (experimental)** (uid `sgl-router-experimental`).

## Metrics covered

The dashboard graphs every family the router emits:

| Metric | Type | What it shows |
|---|---|---|
| `sgl_router_requests_total` | Counter | **Edge intake** — every request received at the router HTTP boundary, by `route`, `method`, counted before worker dispatch (true intake) |
| `sgl_router_responses_total` | Counter | **Edge responses** — every response returned, by `route`, `method`, `status_code` (incl. early-exit 400/413/503). `requests_total - responses_total` = received-but-not-answered |
| `sgl_router_worker_requests_total` | Counter | Per-worker **dispatches** by `worker_url`, `model_id`, `mode`, `outcome` (recorded after dispatch; blind to pre-dispatch drops). See [Dispatch outcomes](#dispatch-outcomes) |
| `sgl_router_request_duration_seconds` | Histogram | End-to-end request latency by `model_id` |
| `sgl_router_ttft_seconds` | Histogram | Time to first token (streaming) by `model_id` |
| `sgl_router_stream_outcome_total` | Counter | Streaming outcomes by `worker_url`, `model_id`, and `outcome` (`ok`, `stream_error_event`, `upstream_error`, or `client_disconnect`). Counts committed 2xx streams only — non-2xx responses are counted by status in `responses_total` |
| `sgl_router_active_load` | Gauge | Per-worker prefill-token / decode-block load |
| `sgl_router_workers` | Gauge | Registered worker count by `mode` |
| `sgl_router_worker_health` | Gauge | Per-worker health (1=breaker admits, 0=open) |
| `sgl_router_worker_cb_state` | Gauge | Per-worker circuit breaker state (0=closed, 1=open, 2=half_open) |
| `sgl_router_worker_inflight_requests` | Gauge | In-flight requests per worker |
| `sgl_router_stale_requests_total` | Counter | Stale-request cancellations |
| `sgl_router_decode_affinity_total` | Counter | PD decode-affinity outcomes |
| `sgl_router_sticky_total` | Counter | Sticky-session selection outcomes |

The legacy `sgl_router_overlap_blocks` metric was removed with the
`cache_aware_zmq` policy and has no direct replacement. Remove queries, alerts,
and dashboard panels that depend on this metric before upgrading.

The `sgl_router_workers` / `sgl_router_worker_*` gauges are sampled from the
live worker registry on every scrape, so a removed worker stops emitting
series immediately rather than leaving a stale value.

## Prometheus scrape config

Point Prometheus at the router's `/metrics` endpoint:

```yaml
scrape_configs:
  - job_name: sgl-router
    metrics_path: /metrics
    static_configs:
      - targets:
          - '127.0.0.1:30000'   # router host:port
```

## Import into Grafana

1. **Dashboards → New → Import**.
2. Upload `grafana-dashboard.json` (or paste its contents).
3. When prompted, select your Prometheus data source for the `Datasource`
   variable. The dashboard uses a templated data source, so it imports into
   any Grafana without editing the JSON.

The top bar exposes `model_id` and `worker_url` template variables (both
default to *All*) to scope the panels.

## Regenerating

The JSON is generated programmatically to keep the ~20 panels consistent. If
the metric surface changes, update the generator and overwrite the JSON
rather than hand-editing — hand-edits drift from the panel conventions.

## Dispatch outcomes

`sgl_router_worker_requests_total{outcome}` is derived from the status the
client saw, not from whether the router's internal dispatch returned `Ok` — a
worker error the router forwards is a successful *proxy* operation and a failed
*request*.

| `outcome` | Source | Counts as a worker fault? |
|---|---|---|
| `success` | 2xx | no |
| `client_error` | 4xx except 429 | no — the caller sent something invalid |
| `backpressure` | 429, 503 | no — responsive but at capacity |
| `error` | 5xx except 503, plus transport failures, timeouts and incomplete bodies | **yes** |
| `cancelled` | the router's own stale-request deadline | no |

`error` is the only bucket that means *this worker failed*, which is why the
Error-ratio panel uses it alone. The split matters during an incident: a
saturated fleet answering with its own queue-full 503s registers as
`backpressure`, and the circuit breaker likewise declines to open on those
statuses — so the two agree, and the error ratio keeps pointing at genuine
faults instead of pegging at 100% exactly when it is being read.

A hung worker surfaces as `error` (the router's upstream timeout), *not* as
`cancelled`. Only the stale-request deadline produces `cancelled`;
`sgl_router_stale_requests_total{outcome="expired"}` counts the same events.

## Access log

The router emits one `http_request` event per request from a single middleware,
so requests that never reach a handler (a body-limit 413, an unrouted 404, a
panic-500) are logged too. Fields: `pod_id`, `request_id`, `method`, `path`,
`status`, `outcome`, `worker`, `model`, `stream`, `latency_ms`.

`worker` and `model` are empty when the request was rejected before dispatch or
hit a route that does not dispatch — that is normal, not a gap. Successful infra
polls (`/healthz`, `/readyz`, `/metrics`) log at DEBUG so they do not bury real
traffic; a *failing* probe keeps the INFO line. For a stream the line is written
when the response head is ready, so `status=200` there does not mean the stream
finished — `sgl_router_stream_outcome_total` carries that.
