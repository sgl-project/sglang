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

Families the router emits, all of them graphed:

| Metric | Type | What it shows |
|---|---|---|
| `sgl_router_requests_total` | Counter | **Edge intake** — every request received at the router HTTP boundary, by `route`, `method`, counted before worker dispatch (true intake) |
| `sgl_router_responses_total` | Counter | **Edge responses** — every response returned, by `route`, `method`, `status_code` (incl. early-exit 400/413/503). `requests_total - responses_total` = received-but-not-answered |
| `sgl_router_worker_requests_total` | Counter | Per-worker **dispatches** by `worker_url`, `model_id`, `mode`, `outcome` (recorded after dispatch; blind to pre-dispatch drops) |
| `sgl_router_request_duration_seconds` | Histogram | End-to-end request latency by `model_id` |
| `sgl_router_ttft_seconds` | Histogram | Time to first token (streaming) by `model_id` |
| `sgl_router_active_load` | Gauge | Per-worker prefill-token / decode-block load |
| `sgl_router_workers` | Gauge | Registered worker count by `mode` |
| `sgl_router_worker_health` | Gauge | Per-worker health (1=breaker admits, 0=open) |
| `sgl_router_worker_cb_state` | Gauge | Per-worker circuit breaker state (0=closed, 1=open, 2=half_open) |
| `sgl_router_worker_inflight_requests` | Gauge | In-flight requests per worker |
| `sgl_router_stale_requests_total` | Counter | Stale-request cancellations |
| `sgl_router_decode_affinity_total` | Counter | PD decode-affinity outcomes |
| `sgl_router_sticky_total` | Counter | Sticky-session selection outcomes |
| `sgl_router_kv_events_total` | Counter | KV-cache events the pump consumed, by `event` and storage `medium` |
| `sgl_router_kv_event_blocks_total` | Counter | Block hashes those events carried, by `event` and `medium` |
| `sgl_router_kv_tree_blocks` | Gauge | Blocks the tree attributes to a `worker_url` / `dp_rank`, by storage `tier` |
| `sgl_router_kv_block_size` | Gauge | Tokens per block hash, as established from the fleet (0 until a worker reports) |
| `sgl_router_kv_event_batches_lost_total` | Counter | KV-event batches dropped in transit, from gaps in each publisher's sequence |
| `sgl_router_kv_tree_accounting_errors_total` | Counter | Occupancy-bookkeeping contradictions, by `reason`. Always 0 on a correct tree |
| `sgl_router_kv_tree_maintained` | Gauge | 1 when this router maintains its own KV tree, 0 under an external Indexer |
| `sgl_router_cache_aware_decisions_total` | Counter | Terminal outcome of each cache-aware selection, by `model_id` and `decision` |
| `sgl_router_cache_aware_query_blocks_total` | Counter | Blocks the request was looked up on — the locality denominator, by `model_id` and `decision` |
| `sgl_router_matched_overlap_blocks_total` | Counter | Blocks the fleet's BEST holder has — the ceiling, by `model_id` and `decision` |
| `sgl_router_selected_overlap_blocks_total` | Counter | Blocks the CHOSEN worker has — the router's prediction of the engine's hit rate |
| `sgl_router_overlap_blocks` | Histogram | Distribution behind the matched counter, by `model_id` |
| `sgl_router_selected_owner_tier_total` | Counter | Storage `tier` the chosen worker holds the matched prefix on, by `model_id` |
| `sgl_router_zero_match_block0_total` | Counter | Zero-overlap selections by whether block 0 is in the tree (`presence`), by `model_id` |

`sgl_router_overlap_blocks` is back after its removal with the
`cache_aware_zmq` policy, and its meaning is narrower than the one old queries
assume: it is the FLEET-BEST overlap, the ceiling a router with no load
constraints could reach, not what the chosen worker holds. Queries that read
its `_sum` as a hit rate were already reading the ceiling; point them at
`sgl_router_selected_overlap_blocks_total` over
`sgl_router_cache_aware_query_blocks_total` instead.

## Reading cache locality against the engine

The three block counters decompose prefix reuse into terms that subtract. The
ratio comparable to the engine's
`sglang:cached_tokens_total / sglang:prompt_tokens_total` is
**`selected / query`**, not `matched / query`: the latter is the fleet-wide
best and reads structurally high, because it meters the deepest prefix anyone
holds even on selections that then routed elsewhere. `matched - selected` is
locality the routing decision gave up, attributable to a decision bucket
because all three share the `(model_id, decision)` key.

The residual against the engine's own number is not one-directional, and the
direction is the diagnosis:

| observation | conclusion |
|---|---|
| `selected/query` fell, `matched/query` flat | routing is diverting off the prefix owner |
| both fell | the tree or the indexer is losing state |
| router ratios flat, engine hit rate fell | engine-side eviction, not routing |
| `selected/query` **below** the engine's rate | a worker serves traffic while publishing no KV events — a failed `/server_info` probe or a page-size disagreement |
| `zero_match_block0_total{presence="absent"}` rising | engine-side publish gap, not a router linkage fault |
| `zero_match_block0_total{presence="in_tree"}` rising | router-side linkage: the hash is carried but unreachable from the root |

Block counts convert to the engine's token units by multiplying by
`sgl_router_kv_block_size`. The query-block denominator rounds a partial
trailing block up to a whole one, so per request it can overstate the engine's
token count by up to one block less a token — averaging half a block on
uniformly distributed lengths, which is where the two denominators agree in
aggregate.

Ratios are **not** additive across models: keep `model_id` on any locality or
coverage panel and collapse it only for rates. The per-decision counters are
evaluations, not requests, which is why their panels use `evals/s` rather than
`ops` and are deliberately not stacked.

The `sgl_router_workers` / `sgl_router_worker_*` gauges are sampled from the
live worker registry on every scrape, so a removed worker stops emitting
series immediately rather than leaving a stale value. The `sgl_router_kv_*`
series are pulled from the KV-event index the same way.

`sgl_router_kv_tree_blocks * sgl_router_kv_block_size` for one worker and
tier, divided by that pod's own occupancy of the tier (device:
`sglang_kv_used_tokens + sglang_kv_evictable_tokens`; host:
`sglang_hicache_host_used_tokens`; `tp_rank="0"`), is the tree's coverage of
that tier. Scope both sides to the same deployment before dividing — block
size and fleet membership both vary between them, and an unscoped ratio
divides one fleet's tree by another's occupancy.

Read it as: about 1, the tree mirrors the engine; about 0, the engine holds a
tier routing cannot see; **above 1, the tree holds tiers a worker has already
released** — check `sgl_router_kv_event_batches_lost_total`, because a tagged
removal clears only its own tier and a lost batch strands the rest.

`sgl_router_kv_events_total` renders every `(event, medium)` cell including
zeros, so a `CPU_PINNED` row pinned at 0 on a hierarchical-cache fleet is
visible rather than absent. A nonzero `block_stored/unknown` row is the
upgrade signal: the engine is publishing a storage tier this build cannot
rank, so the tree drops those stores rather than filing them under a guess. Comparing `sgl_router_kv_event_blocks_total` for
`block_stored/CPU_PINNED` against the engine's `sglang_hicache_backup_tokens_total`
needs `sum without(pool)` on the engine side, and the two are not equal
anyway: the engine also evicts device blocks it never backed up.

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

The JSON is generated programmatically to keep the ~35 panels consistent. If
the metric surface changes, update the generator and overwrite the JSON
rather than hand-editing — hand-edits drift from the panel conventions.
