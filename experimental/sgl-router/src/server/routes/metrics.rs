// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `/metrics` endpoint — Prometheus 0.0.4 exposition.
//!
//! Returns the live snapshot of [`crate::server::metrics::MetricsRegistry`].
//! Plain-text body; charset is utf-8. We deliberately don't gate this on
//! readiness — scrapers should be able to read the metrics surface even
//! while the router is warming up so the "router started but no workers
//! discovered" failure mode is observable.

use crate::server::app_context::AppContext;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use std::sync::Arc;

/// Content-Type per Prometheus exposition format spec.
const PROMETHEUS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

pub async fn metrics(State(ctx): State<Arc<AppContext>>) -> impl IntoResponse {
    let body = ctx.metrics.render();
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
        assert!(body.contains("# TYPE sgl_router_requests_total counter"));
        assert!(body.contains("# TYPE sgl_router_overlap_blocks histogram"));
        assert!(body.contains("# TYPE sgl_router_active_load gauge"));
    }

    #[tokio::test]
    async fn metrics_endpoint_reflects_recorded_counters() {
        let ctx = Arc::new(AppContext::stub());
        ctx.metrics.record_request(
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
}
