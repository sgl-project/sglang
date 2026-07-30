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

## Selected metrics reference

The dashboard graphs the high-signal request, routing, tree, and bootstrap
families. This table is not an exhaustive list of everything exposed on
`/metrics`; it also lists related drill-down metrics intended for ad-hoc
PromQL rather than a dedicated panel:

| Metric | Type | What it shows |
|---|---|---|
| `sgl_router_requests_total` | Counter | **Edge intake** — every request received at the router HTTP boundary, by `route`, `method`, counted before worker dispatch (true intake). Ad hoc only; no dedicated panel |
| `sgl_router_responses_total` | Counter | **Edge responses** — every response returned, by `route`, `method`, `status_code` (incl. early-exit 400/413/503). `requests_total - responses_total` = received-but-not-answered |
| `sgl_router_worker_requests_total` | Counter | Per-worker **dispatches** by `worker_url`, `model_id`, `mode`, `outcome` (recorded after dispatch; blind to pre-dispatch drops) |
| `sgl_router_stream_outcome_total` | Counter | **End-of-stream truth** for 2xx streaming responses by `worker_url`, `model_id`, `outcome` (`ok` / `inband_error` / `upstream_error` / `client_disconnect`). Headers-time counters record a committed 200 as success even when the engine later fails in-band; this is where that failure shows up |
| `sgl_router_request_duration_seconds` | Histogram | End-to-end request latency by `model_id` |
| `sgl_router_ttft_seconds` | Histogram | Time to first token (streaming) by `model_id` |
| `sgl_router_itl_seconds` | Histogram | Inter-token latency (gap between successive upstream chunks, 2xx streaming) by `model_id`; bucket edges match the engine's `sglang:inter_token_latency_seconds` |
| `sgl_router_overlap_blocks` | Histogram | Cache-aware-zmq overlap blocks by `model_id` |
| `sgl_router_cache_aware_query_blocks_total` | Counter | Query blocks considered by cache-aware policy tree lookups, by `model_id` and `decision`. Shared denominator for the two numerators below; the dashboard requires it to be `> 0` so a no-lookup window remains a gap rather than a false 0% match. Emitted only for selections that reached the hash lookup and placed a worker, so it does not join with `sgl_router_cache_aware_decisions_total` on `decision` |
| `sgl_router_matched_overlap_blocks_total` | Counter | Overlap blocks the BEST-matching worker in the fleet held, by `model_id` and `decision`. Counter form of `sgl_router_overlap_blocks_sum`. A ceiling, not an outcome — it is metered before the load gate decides where the request goes, so dividing it by query blocks reads structurally above the engine's real hit rate |
| `sgl_router_selected_overlap_blocks_total` | Counter | Overlap blocks the worker actually SELECTED held, by `model_id` and `decision`. This is the numerator to compare against the engine's `sglang:cached_tokens_total / sglang:prompt_tokens_total`; `matched - selected` per decision is the locality each routing rule gave up. Reads 0 for a worker publishing no KV events, so a ratio *below* the engine's indicts the subscriber fleet rather than eviction |
| `sgl_router_cache_aware_decisions_total` | Counter | Terminal cache-aware policy-evaluation outcomes by `model_id`. The lookup-health panel derives coverage from the lookup-terminal labels and exposes early exits such as `hash_config_unknown`. These are not request counts: admission races can evaluate repeatedly, while parked hand-offs skip evaluation |
| `sgl_router_kv_tree_nodes` | Gauge | Cache-aware KV tree size per router replica; a restart-time cliff exposes a cold or incomplete tree |
| `sgl_router_kv_bootstrap_settled` | Gauge | Whether initial peer bootstrap has settled; compare with tree size because a timed-out empty bootstrap also settles |
| `sgl_router_kv_peers` | Gauge | Number of sibling router replicas visible to peer bootstrap. Ad hoc only; no dedicated panel |
| `sgl_router_kv_bootstrap_state` | Gauge | Per-worker/rank bootstrap state: 0=pending, 1=recovered, 2=failed. Ad hoc only; no dedicated panel |
| `sgl_router_kv_peer_snapshot_total` | Counter | Peer snapshot fetch outcomes. Ad hoc only; no dedicated panel |
| `sgl_router_kv_bootstrap_rank_total` | Counter | Final bootstrap outcomes aggregated across engine ranks, labeled only by `outcome`. Ad hoc only; no dedicated panel |
| `sgl_router_active_load` | Gauge | Per-worker prefill-token / decode-block load |
| `sgl_router_workers` | Gauge | Registered worker count by `mode` |
| `sgl_router_worker_health` | Gauge | Per-worker health (1=breaker admits, 0=open) |
| `sgl_router_worker_cb_state` | Gauge | Per-worker circuit breaker state (0=closed, 1=open, 2=half_open) |
| `sgl_router_worker_inflight_requests` | Gauge | In-flight requests per worker |
| `sgl_router_stale_requests_total` | Counter | Stale-request cancellations |
| `sgl_router_decode_affinity_total` | Counter | PD decode-affinity outcomes |
| `sgl_router_sticky_total` | Counter | Sticky-session selection outcomes |
| `sgl_router_mm_affinity_total` | Counter | Multimodal-affinity outcomes by `outcome` (`hit` / `assigned` / `unkeyed` / `unavailable`). Healthy image traffic is mostly `hit`; sustained `unkeyed` means a media shape gets no affinity and routes cold every turn |

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

The dashboard uses Prometheus's `instance` target label as the router-replica
identity, so this static configuration works without Kubernetes-specific
relabeling.

## Import into Grafana

1. **Dashboards → New → Import**.
2. Upload `grafana-dashboard.json` (or paste its contents).
3. When prompted, select your Prometheus data source for the `Datasource`
   variable. The dashboard uses a templated data source, so it imports into
   any Grafana without editing the JSON.

The top bar exposes `model_id` and `worker_url` template variables (both
default to *All*) to scope the panels.

## Editing

`grafana-dashboard.json` is the checked-in dashboard source. Keep panel IDs
unique, preserve the templated data source, and validate the file with `jq`
after editing.
