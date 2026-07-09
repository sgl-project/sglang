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
    (
        StatusCode::OK,
        [(CONTENT_TYPE, PROMETHEUS_CONTENT_TYPE)],
        body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::metrics::{RequestOutcome, WorkerModeLabel};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

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
