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

The dashboard graphs the high-signal request, admission, routing, tree, and
bootstrap families. This table is not an exhaustive list of everything
exposed on `/metrics`; it also lists related drill-down metrics intended for
ad-hoc PromQL rather than a dedicated panel:

| Metric | Type | What it shows |
|---|---|---|
| `sgl_router_requests_total` | Counter | **Edge intake** — every request received at the router HTTP boundary, by `route`, `method`, counted before worker dispatch (true intake). Graphed against `worker_requests_total{worker_url!=""}` on **Edge intake vs dispatched**, where the gap is everything shed before a worker was chosen |
| `sgl_router_responses_total` | Counter | **Edge responses** — every response returned, by `route`, `method`, `status_code` (incl. early-exit 400/413/503). `requests_total - responses_total` = received-but-not-answered |
| `sgl_router_worker_requests_total` | Counter | Per-worker **dispatches** by `worker_url`, `model_id`, `mode`, `outcome`. Written for every non-infra request at response time, not only for routed ones — a request rejected before a worker was chosen is recorded with an empty `worker_url`. Select `worker_url!=""` for true dispatches; the empty bucket is the pre-dispatch drops |
| `sgl_router_stream_outcome_total` | Counter | **End-of-stream truth** for 2xx streaming responses by `worker_url`, `model_id`, `outcome` (`ok` / `inband_error` / `upstream_error` / `client_disconnect`). Headers-time counters record a committed 200 as success even when the engine later fails in-band; this is where that failure shows up |
| `sgl_router_request_duration_seconds` | Histogram | End-to-end request latency by `model_id` |
| `sgl_router_ttft_seconds` | Histogram | Time to first token (streaming) by `model_id` |
| `sgl_router_itl_seconds` | Histogram | Inter-token latency (gap between successive upstream chunks, 2xx streaming) by `model_id`; bucket edges match the engine's `sglang:inter_token_latency_seconds` |
| `sgl_router_overlap_blocks` | Histogram | Cache-aware-zmq overlap blocks by `model_id` |
| `sgl_router_cache_aware_query_blocks_total` | Counter | Query blocks considered by cache-aware policy tree lookups, by `model_id` and `decision`. Shared denominator for the two numerators below; the dashboard requires it to be `> 0` so a no-lookup window remains a gap rather than a false 0% match. Emitted only for selections that reached the hash lookup and placed a worker, so it does not join with `sgl_router_cache_aware_decisions_total` on `decision` |
| `sgl_router_matched_overlap_blocks_total` | Counter | Overlap blocks the BEST-matching worker in the fleet held, by `model_id` and `decision`. Counter form of `sgl_router_overlap_blocks_sum`. A ceiling, not an outcome — it is metered before the load gate decides where the request goes, so dividing it by query blocks reads structurally above the engine's real hit rate. **Locality loss by decision** graphs `matched - selected` per decision: the prefix each routing rule gave up |
| `sgl_router_selected_overlap_blocks_total` | Counter | Overlap blocks the worker actually SELECTED held, by `model_id` and `decision`. This is the numerator to compare against the engine's `sglang:cached_tokens_total / sglang:prompt_tokens_total`; `matched - selected` per decision is the locality each routing rule gave up. Reads 0 for a worker publishing no KV events, so a ratio *below* the engine's indicts the subscriber fleet rather than eviction |
| `sgl_router_cache_aware_decisions_total` | Counter | Terminal cache-aware policy-evaluation outcomes by `model_id`. The lookup-health panel derives coverage from the lookup-terminal labels and exposes early exits such as `hash_config_unknown`. These are not request counts: admission races can evaluate repeatedly, while parked hand-offs skip evaluation |
| `sgl_router_ttft_overhead_seconds` | Histogram | Router-internal TTFT overhead (tokenize + admission wait + request build) before dispatch, by `model_id`. A sub-term of `sgl_router_ttft_seconds`; the dashboard graphs their ratio, so a rising share indicts the router rather than engine prefill |
| `sgl_router_queued_requests` | Gauge | Requests parked in the admission queue right now, per router replica (unlabeled; replicas separate on `instance`). Depth on one replica alone is a routing imbalance, not engine slowness |
| `sgl_router_admission_wait_seconds` | Histogram | Time parked in the admission queue before dispatch, by `model_id`. Already lost to TTFT by the time it is observed |
| `sgl_router_backpressure_rejected_total` | Counter | Requests shed at admission because the queue was full, by `model_id`. These never reach a worker, so every per-worker panel is blind to them |
| `sgl_router_kv_tree_nodes` | Gauge | Cache-aware KV tree size per router replica; a restart-time cliff exposes a cold or incomplete tree |
| `sgl_router_kv_bootstrap_settled` | Gauge | Whether initial peer bootstrap has settled; compare with tree size because a timed-out empty bootstrap also settles |
| `sgl_router_kv_fleet_bimodal` | Gauge | 1 while the fleet carries both KV-block hashing modes, so every selection dual-hashes. Expected across a rolling update; a stuck 1 means a worker never finished the switch |
| `sgl_router_kv_peers` | Gauge | Number of sibling router replicas visible to peer bootstrap. Zero on a fresh replica means it rebuilds its tree from live events alone |
| `sgl_router_kv_bootstrap_state` | Gauge | Per-worker/rank bootstrap state: 0=pending, 1=recovered, 2=failed. Ad hoc only; no dedicated panel |
| `sgl_router_kv_peer_snapshot_total` | Counter | Peer snapshot fetch outcomes. Counts FETCHES, so it does not divide into the rank counter below — one accepted fetch can settle several ranks |
| `sgl_router_kv_bootstrap_rank_total` | Counter | Final bootstrap outcomes aggregated across engine ranks, labeled only by `outcome`. Counts RANKS, the fetch counter above counts fetches |
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

### Optional deployment-identity labels

Three of the template variables read target labels the router does not export
itself — a scrape config has to attach them:

| Label | Meaning |
|---|---|
| `serving_endpoint` | The logical endpoint whose traffic these router pods serve. Several deployments can serve one endpoint (a migration, a multi-cluster spread), and this is what makes them add up as one thing |
| `cluster` | The cluster a pod runs in |
| `namespace` | The deployment a pod belongs to — one namespace per deployment, so this is the per-deployment drilldown |

Under Kubernetes service discovery these come from pod labels, e.g.:

```yaml
global:
  external_labels:
    cluster: my-cluster        # per-Prometheus, not per-pod

scrape_configs:
  - job_name: sgl-router
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_serving_endpoint]
        target_label: serving_endpoint
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
```

`cluster` has no pod-level source — it identifies the Prometheus, so it comes
from `external_labels` (or whatever your remote-write agent stamps) rather
than from relabeling. Omit it and the `cluster` variable stays empty and the
**Request rate by cluster / namespace** legends read as `" / namespace"`.

None of them is required. When a label is absent the matching variable lists
no values and stays on *All*, which is `.*` and therefore matches series that
never carried the label — so every panel keeps working on a plain static
scrape. The reverse is also true and worth knowing before you go looking for
missing history: series scraped before the labels were attached do not match a
*specific* `endpoint` selection, so a freshly labeled fleet starts with an
empty history under that selection and fills forward.

## Import into Grafana

1. **Dashboards → New → Import**.
2. Upload `grafana-dashboard.json` (or paste its contents).
3. When prompted, select your Prometheus data source for the `Datasource`
   variable. The dashboard uses a templated data source, so it imports into
   any Grafana without editing the JSON.

## Scoping the panels

The top bar is a cascade, each variable multi-select and defaulting to *All*:

```
endpoint (slug)  →  cluster  →  namespace  →  model_id  →  worker_url
```

Each one lists only values that exist under the selections to its left, so
`namespace` offers the deployments of the endpoint you picked rather than every
namespace in the estate.

Read it two ways. Leave `namespace` on *All* and the board is the endpoint's
whole serving fleet, summed across every deployment behind it — the default,
and the only view that stays honest while traffic moves between deployments.
Pick a single `cluster` / `namespace` pair and the same panels narrow to that
one deployment. **Request rate by cluster / namespace** shows the split itself,
so an uneven or shifting one is visible without touching the variables.

`instance` is not a variable: per-replica panels break replicas out on the
legend instead, because a router replica is rarely worth isolating for a whole
board.

## Editing

`grafana-dashboard.json` is the checked-in dashboard source. Keep panel IDs
unique, preserve the templated data source, and validate the file with `jq`
after editing. Every panel over a `sgl_router_*` series carries the full
`{serving_endpoint,cluster,namespace}` scope in its selector — a new panel
that omits it silently ignores the top bar and reads the whole estate.
