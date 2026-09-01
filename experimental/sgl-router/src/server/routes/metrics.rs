// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `/metrics` endpoint — Prometheus 0.0.4 exposition.
//!
//! Returns the live snapshot of [`crate::server::metrics::MetricsRegistry`].
//! Plain-text body; charset is utf-8. We deliberately don't gate this on
//! readiness — scrapers should be able to read the metrics surface even
//! while the router is warming up so the "router started but no workers
//! discovered" failure mode is observable.

use crate::discovery::WorkerMode;
use crate::server::app_context::AppContext;
use crate::server::metrics::WorkerSnapshot;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use std::sync::Arc;

/// Content-Type per Prometheus exposition format spec.
const PROMETHEUS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

pub async fn metrics(State(ctx): State<Arc<AppContext>>) -> impl IntoResponse {
    // Sample the live registry into a snapshot for the worker gauges. These
    // are pull-on-scrape (not pushed) so removed workers stop emitting series
    // immediately; see `MetricsRegistry::render_with_workers`.
    let workers: Vec<WorkerSnapshot> = ctx
        .registry
        .all()
        .into_iter()
        .map(|w| {
            // One lock acquisition for both health + state so the two gauges
            // can't report a torn (self-contradictory) pair for one scrape.
            let cb = w.breaker.snapshot();
            WorkerSnapshot {
                worker_url: w.url.clone(),
                mode: match w.mode() {
                    WorkerMode::Plain => "plain",
                    WorkerMode::Prefill => "prefill",
                    WorkerMode::Decode => "decode",
                },
                healthy: cb.admit,
                cb_state: cb.state_code,
                // Saturating rather than `as i64`: a guard-accounting
                // underflow would wrap usize and render as a nonsensical
                // negative gauge; clamp to a large positive ceiling instead.
                inflight: i64::try_from(w.active_load()).unwrap_or(i64::MAX),
            }
        })
        .collect();
    let mut body = ctx.metrics.render_with_workers(&workers);
    // Per-worker ITL gauge, sampled from the shared table at scrape time (same
    // pull-on-scrape model as the worker gauges above; a worker with no fresh
    // ITL sample emits no series).
    let itl_samples = ctx.itl.snapshot_fresh(std::time::Instant::now());
    body.push_str(&ctx.metrics.render_worker_itl(&itl_samples));
    // Cache-aware bootstrap, same pull-on-scrape model. Emitted only when the
    // KV index exists, so a router without cache-aware-zmq shows no series.
    if let Some(index) = ctx.kv_index() {
        body.push_str(&render_kv_bootstrap(index));
    }
    (
        StatusCode::OK,
        [(CONTENT_TYPE, PROMETHEUS_CONTENT_TYPE)],
        body,
    )
}

/// Render the cache-aware bootstrap series.
///
/// `sgl_router_kv_tree_nodes` makes a cold or leaking tree visible, and the
/// bootstrap gauges make "this replica is serving cache-blind" alertable.
/// Without them both conditions are only inferable from a hit-rate dip.
fn render_kv_bootstrap(index: &Arc<crate::policies::kv_events::KvEventIndex>) -> String {
    use crate::server::metrics::escape_label;

    let mut out = String::new();
    out.push_str("# HELP sgl_router_kv_tree_nodes Nodes in the cache-aware KV hash tree.\n");
    out.push_str("# TYPE sgl_router_kv_tree_nodes gauge\n");
    out.push_str(&format!(
        "sgl_router_kv_tree_nodes {}\n",
        index.tree().node_count()
    ));
    out.push_str(&render_kv_tiers(index));

    let tracker = index.bootstrap();
    out.push_str(
        "# HELP sgl_router_kv_bootstrap_settled 1 when initial peer bootstrap has settled.\n",
    );
    out.push_str("# TYPE sgl_router_kv_bootstrap_settled gauge\n");
    out.push_str(&format!(
        "sgl_router_kv_bootstrap_settled {}\n",
        u8::from(tracker.settled())
    ));

    out.push_str("# HELP sgl_router_kv_peers Sibling replicas available to bootstrap from.\n");
    out.push_str("# TYPE sgl_router_kv_peers gauge\n");
    out.push_str(&format!("sgl_router_kv_peers {}\n", index.peers().len()));

    // The bimodal window doubles selection cost (every query is hashed and
    // walked twice) and typically spans a rolling update, so it needs a time
    // series, not just the transition log line.
    out.push_str(
        "# HELP sgl_router_kv_fleet_bimodal 1 while the fleet carries both KV-block hashing modes (selection dual-hashes).\n",
    );
    out.push_str("# TYPE sgl_router_kv_fleet_bimodal gauge\n");
    out.push_str(&format!(
        "sgl_router_kv_fleet_bimodal {}\n",
        u8::from(index.block_size_oracle().is_bimodal())
    ));

    let states = tracker.states();
    if !states.is_empty() {
        out.push_str(
            "# HELP sgl_router_kv_bootstrap_state Per-rank bootstrap state \
             (0=pending, 1=recovered, 2=failed).\n",
        );
        out.push_str("# TYPE sgl_router_kv_bootstrap_state gauge\n");
        for (id, state) in states {
            out.push_str(&format!(
                "sgl_router_kv_bootstrap_state{{worker=\"{}\",dp_rank=\"{}\"}} {}\n",
                escape_label(&id.url),
                id.dp_rank,
                state.as_metric(),
            ));
        }
    }

    // Two counters, deliberately not one: this counts FETCHES against peers,
    // the next counts RANKS. One accepted fetch can settle several ranks and one
    // rank can outlive many rejected fetches, so a shared counter would be
    // divisible by nothing.
    let peer_outcomes = tracker.peer_outcome_counts();
    if !peer_outcomes.is_empty() {
        out.push_str(
            "# HELP sgl_router_kv_peer_snapshot_total Peer snapshot fetches by outcome.\n",
        );
        out.push_str("# TYPE sgl_router_kv_peer_snapshot_total counter\n");
        for (outcome, count) in peer_outcomes {
            out.push_str(&format!(
                "sgl_router_kv_peer_snapshot_total{{outcome=\"{outcome}\"}} {count}\n",
            ));
        }
    }

    // Recorded once per rank, at the point its verdict is final — so `warm`
    // lags the state gauge by the splice proof, and the labels sum to the number
    // of ranks that finished bootstrapping.
    let rank_outcomes = tracker.rank_outcome_counts();
    if !rank_outcomes.is_empty() {
        out.push_str(
            "# HELP sgl_router_kv_bootstrap_rank_total Ranks by final bootstrap outcome.\n",
        );
        out.push_str("# TYPE sgl_router_kv_bootstrap_rank_total counter\n");
        for (outcome, count) in rank_outcomes {
            out.push_str(&format!(
                "sgl_router_kv_bootstrap_rank_total{{outcome=\"{outcome}\"}} {count}\n",
            ));
        }
    }
    out
}

/// Render the storage-tier series: what the tree holds per worker and tier,
/// the block size to convert it to tokens, and the tagged event stream it
/// consumed. Same pull-on-scrape model as the bootstrap gauges.
///
/// These exist to make a router-vs-engine tier mismatch a number instead of
/// an inference. `sgl_router_kv_tree_blocks * sgl_router_kv_block_size`
/// for a worker and tier, divided by that pod's own occupancy of the tier
/// (device: `sglang_kv_used_tokens + sglang_kv_evictable_tokens`; host:
/// `sglang_hicache_host_used_tokens`; `tp_rank="0"`), is the tree's coverage
/// of the tier. About 1 means the tree mirrors the engine; about 0 means the
/// engine holds a tier that routing cannot see; a missing series means the
/// worker publishes nothing. The event counters show whether the tagged
/// stream that should feed the tree is arriving at all.
fn render_kv_tiers(index: &Arc<crate::policies::kv_events::KvEventIndex>) -> String {
    use crate::policies::kv_events::Tiers;
    use crate::server::metrics::escape_label;

    let mut out = String::new();

    // 0 until the first worker reports, and the dashboard multiplies by it, so
    // a coverage panel reads 0 rather than NaN on a fleet that has not
    // registered yet.
    let block_size = index.block_size_oracle().get().unwrap_or(0);
    out.push_str(
        "# HELP sgl_router_kv_block_size Tokens per KV block hash, as established from the fleet (0 until a worker reports). Multiply sgl_router_kv_tree_blocks by this to compare with the engine's token gauges.\n",
    );
    out.push_str("# TYPE sgl_router_kv_block_size gauge\n");
    out.push_str(&format!("sgl_router_kv_block_size {block_size}\n"));

    // Every tier is emitted per carrier, zeros included: a host row at 0 next
    // to a device row in the millions is the mismatch signature, and an
    // absent series cannot be told from a tier the tree never tracked.
    out.push_str(
        "# HELP sgl_router_kv_tree_blocks Blocks the cache-aware tree attributes to a worker rank, by the storage tier the worker holds them on (a block held on device and host counts under both). Times sgl_router_kv_block_size, and divided by the engine's own occupancy of that tier for the same pod (device: sglang_kv_used_tokens + sglang_kv_evictable_tokens; host: sglang_hicache_host_used_tokens; tp_rank=\"0\"), this is the tree's coverage of the tier: ~1 mirrors the engine, ~0 means the engine holds a tier routing cannot see.\n",
    );
    out.push_str("# TYPE sgl_router_kv_tree_blocks gauge\n");
    for (id, counts) in index.tree().tier_occupancy() {
        for (slot, (_, tier)) in Tiers::SLOTS.iter().enumerate() {
            out.push_str(&format!(
                "sgl_router_kv_tree_blocks{{worker_url=\"{}\",dp_rank=\"{}\",tier=\"{}\"}} {}\n",
                escape_label(&id.url),
                id.dp_rank,
                tier,
                counts[slot],
            ));
        }
    }

    let rows = index.event_tally().snapshot();
    out.push_str(
        "# HELP sgl_router_kv_events_total KV-cache events applied to the tree, by kind and the storage medium tag they carried (untagged = no medium field; unknown = a medium this build does not recognise). On a hierarchical-cache fleet block_stored/CPU_PINNED runs at about the block_removed/GPU rate; a CPU_PINNED row pinned at 0 with hicache enabled means the tier stream is not reaching the router.\n",
    );
    out.push_str("# TYPE sgl_router_kv_events_total counter\n");
    for r in &rows {
        out.push_str(&format!(
            "sgl_router_kv_events_total{{event=\"{}\",medium=\"{}\"}} {}\n",
            r.event, r.medium, r.events,
        ));
    }
    out.push_str(
        "# HELP sgl_router_kv_event_blocks_total Block hashes carried by the KV-cache events applied to the tree, by kind and storage medium tag. Times sgl_router_kv_block_size this is comparable to the engine's sglang_hicache_backup_tokens_total (block_stored/CPU_PINNED) and its device eviction volume (block_removed/GPU).\n",
    );
    out.push_str("# TYPE sgl_router_kv_event_blocks_total counter\n");
    for r in &rows {
        out.push_str(&format!(
            "sgl_router_kv_event_blocks_total{{event=\"{}\",medium=\"{}\"}} {}\n",
            r.event, r.medium, r.blocks,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::metrics::{RequestOutcome, WorkerModeLabel};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// The tier series are what the coverage dashboard joins on, so their
    /// names and label keys are contract: per-worker blocks by tier with
    /// zeros emitted, the block size to convert them, and every
    /// (event, medium) cell of the tally.
    #[tokio::test]
    async fn kv_tier_series_render_per_worker_and_per_medium() {
        use crate::policies::kv_events::{EventKind, KvEventIndex, KvWorkerId, Tiers};

        let index = KvEventIndex::new();
        index.block_size_oracle().try_set(64).unwrap();
        let w = KvWorkerId::new("http://w0:30000".into(), 0);
        index
            .tree()
            .insert_tiered(&w, None, &[1, 2, 3], Tiers::DEVICE);
        index.tree().insert_tiered(&w, None, &[1, 2], Tiers::HOST);
        index
            .event_tally()
            .record(EventKind::BlockStored, Some("CPU_PINNED"), 2);

        let out = render_kv_tiers(&index);
        for want in [
            "sgl_router_kv_block_size 64\n",
            r#"sgl_router_kv_tree_blocks{worker_url="http://w0:30000",dp_rank="0",tier="device"} 3"#,
            r#"sgl_router_kv_tree_blocks{worker_url="http://w0:30000",dp_rank="0",tier="host"} 2"#,
            r#"sgl_router_kv_tree_blocks{worker_url="http://w0:30000",dp_rank="0",tier="storage"} 0"#,
            r#"sgl_router_kv_events_total{event="block_stored",medium="CPU_PINNED"} 1"#,
            r#"sgl_router_kv_event_blocks_total{event="block_stored",medium="CPU_PINNED"} 2"#,
            r#"sgl_router_kv_events_total{event="block_removed",medium="GPU"} 0"#,
        ] {
            assert!(out.contains(want), "missing {want:?}; got:\n{out}");
        }
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_text() {
        let ctx = Arc::new(AppContext::stub());
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let content_type = res
            .headers()
            .get(CONTENT_TYPE)
            .expect("content-type header")
            .to_str()
            .unwrap()
            .to_owned();
        assert!(
            content_type.starts_with("text/plain"),
            "expected text/plain, got {content_type}",
        );
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        // Every metric family should at least carry its HELP/TYPE lines.
        assert!(body.contains("# TYPE sgl_router_worker_requests_total counter"));
        assert!(body.contains("# TYPE sgl_router_overlap_blocks histogram"));
        assert!(body.contains("# TYPE sgl_router_active_load gauge"));
    }

    #[tokio::test]
    async fn metrics_endpoint_reflects_recorded_counters() {
        let ctx = Arc::new(AppContext::stub());
        ctx.metrics.record_worker_request(
            "http://w-test:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(
            body.contains(r#"worker_url="http://w-test:30000""#),
            "metrics did not include the recorded worker_url; got:\n{body}",
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_samples_worker_gauges_from_registry() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

        let ctx = Arc::new(AppContext::stub());
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("p0".into()),
                url: "http://p0:30000".into(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
            })
            .unwrap();
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        // Pool size reflects the registered prefill worker, and the per-worker
        // gauges are sampled (fresh breaker => healthy, closed, 0 inflight).
        assert!(
            body.contains(r#"sgl_router_workers{mode="prefill"} 1"#),
            "got:\n{body}"
        );
        assert!(body.contains(r#"sgl_router_worker_health{worker_url="http://p0:30000"} 1"#));
        assert!(body.contains(r#"sgl_router_worker_cb_state{worker_url="http://p0:30000"} 0"#));
        assert!(
            body.contains(r#"sgl_router_worker_inflight_requests{worker_url="http://p0:30000"} 0"#)
        );
    }
}
