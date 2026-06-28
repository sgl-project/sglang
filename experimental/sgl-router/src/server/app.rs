// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::metrics::{
    outcome_from_status, MetricsRegistry, RequestLogContext, RequestOutcome, WorkerModeLabel,
};
use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;
use tower_http::catch_panic::CatchPanicLayer;

/// Middleware: log 413 PAYLOAD_TOO_LARGE responses with the request method
/// and URI so an operator investigating "client X gets 413s" has a
/// server-side breadcrumb. The 413 is produced by axum's `DefaultBodyLimit`
/// layer BEFORE the handler runs, so without this we would have no record
/// of which request was rejected.
async fn log_413(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let resp = next.run(req).await;
    if resp.status() == StatusCode::PAYLOAD_TOO_LARGE {
        tracing::warn!(
            %method,
            %uri,
            "request rejected with 413 PAYLOAD_TOO_LARGE (body exceeded route limit)",
        );
    }
    resp
}

/// Infra endpoints excluded from the access log (logged at DEBUG instead) and
/// from `worker_requests_total`. They are polled constantly — Prometheus scrapes
/// `/metrics`, the kubelet hits `/healthz` + `/readyz` every few seconds — so
/// logging them at INFO would bury real API traffic and counting them in
/// `worker_requests_total` would swamp the by-outcome view with probe successes. They
/// are still counted in `responses_total{route,method,status_code}`.
fn is_infra_path(path: &str) -> bool {
    matches!(path, "/healthz" | "/readyz" | "/metrics")
}

/// Outermost middleware: the single edge-counting + access-log site.
///
/// Runs for EVERY request — all routes, plus responses produced before any
/// handler runs (a 413 from the body-limit layer; a 400 from the body extractor
/// when a client drops the connection mid-upload; a `CatchPanicLayer` 500) and
/// handler short-circuits that return via `?` (a 503 admission shed, a 400
/// body-validation, a 404 model-not-found).
///
/// At ENTRY (before the handler runs) it counts `requests_total{route,method}` —
/// true intake, so a request parked/shed/cancelled/dropped before producing a
/// response is still counted. `route` is the matched [`MatchedPath`] template
/// (bounded cardinality), `unmatched` for a 404. At EXIT it:
///   * counts `responses_total{route,method,status_code}` (every response, incl.
///     infra); `requests_total - responses_total` is the received-but-not-answered
///     gap;
///   * for non-infra paths, counts
///     `worker_requests_total{worker_url,model_id,mode,outcome}` — reading the
///     per-worker labels a routed handler attached via [`RequestLogContext`], or
///     an empty `worker_url` when the request was rejected before routing; and
///   * emits one access-log line (INFO; DEBUG for infra).
///
/// Handlers therefore no longer log or count requests themselves: a single site
/// means no outcome is missed and none is double-counted.
async fn access_log_and_record(
    State(metrics): State<Arc<MetricsRegistry>>,
    req: Request,
    next: Next,
) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    // Matched route template (e.g. `/v1/chat/completions`), not the raw URI, so
    // `requests_total` / `responses_total` stay low-cardinality; `unmatched` for a 404.
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".to_owned());
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_string();
    let start = std::time::Instant::now();

    // ENTRY: true intake — counted before the handler can park/shed/drop it, so
    // a never-answered request still shows up (intake - responses = the gap).
    metrics.record_ingress(&route, method.as_str());

    let resp = next.run(req).await;

    let status = resp.status();
    let latency_ms = start.elapsed().as_millis() as u64;
    // Edge responses count ALL routes (incl. infra) so the intake/response
    // population matches; filter infra by `route` in PromQL.
    metrics.record_response(&route, method.as_str(), status.as_u16());

    if is_infra_path(&path) {
        tracing::debug!(
            method = %method,
            path = %path,
            status = status.as_u16(),
            latency_ms,
            "http_request",
        );
        return resp;
    }

    let outcome = outcome_from_status(status.as_u16());
    let outcome_str = match outcome {
        RequestOutcome::Success => "success",
        RequestOutcome::Error => "error",
        RequestOutcome::Cancelled => "cancelled",
    };
    // Per-worker labels are present only when a handler routed the request and
    // attached them; pre-routing rejections record an empty worker_url.
    let ctx = resp.extensions().get::<RequestLogContext>();
    let worker = ctx.map(|c| c.worker_url.as_str()).unwrap_or("");
    let model = ctx.map(|c| c.model_id.as_str()).unwrap_or("");
    let mode = ctx.map(|c| c.mode).unwrap_or(WorkerModeLabel::Plain);

    metrics.record_worker_request(worker, model, mode, outcome);
    tracing::info!(
        request_id = %request_id,
        method = %method,
        path = %path,
        route = %route,
        status = status.as_u16(),
        outcome = outcome_str,
        worker = %worker,
        model = %model,
        latency_ms,
        "http_request",
    );
    resp
}

pub fn build_router(ctx: Arc<AppContext>) -> Router {
    Router::new()
        .route("/healthz", get(crate::server::routes::health::healthz))
        .route("/readyz", get(crate::server::routes::health::readyz))
        .route("/metrics", get(crate::server::routes::metrics::metrics))
        .route(
            "/v1/models",
            get(crate::server::routes::models::list_models),
        )
        .route(
            "/v1/tokenize",
            post(crate::server::routes::tokenize::tokenize),
        )
        .route(
            "/v1/detokenize",
            post(crate::server::routes::tokenize::detokenize),
        )
        .route(
            "/v1/chat/completions",
            post(crate::server::routes::chat::chat_completions)
                .layer(DefaultBodyLimit::max(MAX_CHAT_BODY_BYTES))
                .layer(middleware::from_fn(log_413)),
        )
        .route(
            "/flush_cache",
            post(crate::server::routes::cache::flush_cache),
        )
        // Convert a handler panic into a 500 response. hyper otherwise catches
        // the panic and drops the connection WITHOUT a Response, so the failure
        // never reaches the `access_log_and_record` middleware below and is
        // invisible to metrics and logs. Positioned INNER relative to that
        // middleware (added before it, so it sits closer to the handlers) so the
        // synthesized 500 is observed and counted.
        .layer(CatchPanicLayer::new())
        // Outermost layer: the single access-log + metric site. Logs every
        // request and counts `requests_total{route,method}` (intake),
        // `responses_total{route,method,status_code}`, and
        // `worker_requests_total{...,outcome}`, so error short-circuits (admission
        // 503s, 413s, client-dropped-upload 400s, …) and synthesized panic-500s
        // are logged and counted alongside successes.
        .layer(middleware::from_fn_with_state(
            Arc::clone(&ctx.metrics),
            access_log_and_record,
        ))
        .with_state(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// A handler panic must become a 500 that the `access_log_and_record`
    /// middleware still observes. hyper catches a handler panic and drops the
    /// connection WITHOUT producing a Response, so without `CatchPanicLayer`
    /// the failure is invisible to the response-code metric. `CatchPanicLayer`
    /// synthesizes a 500; the metrics layer must be OUTER (applied after) so it
    /// counts that synthesized 500.
    ///
    /// This composes the SAME two layers in the SAME order as `build_router`
    /// (metrics outer, catch-panic inner) over a panicking route — the real
    /// `build_router` has no panicking route to exercise, so a minimal Router
    /// pins the ordering contract directly.
    #[tokio::test]
    async fn handler_panic_becomes_500_and_is_counted() {
        let metrics = MetricsRegistry::new();
        let app = Router::new()
            .route(
                "/boom",
                get(|| async {
                    panic!("handler exploded");
                    #[allow(unreachable_code)]
                    StatusCode::OK
                }),
            )
            // Inner: convert a handler panic into a 500 response.
            .layer(CatchPanicLayer::new())
            // Outer: count the final status — must observe the synthesized 500.
            .layer(middleware::from_fn_with_state(
                Arc::clone(&metrics),
                access_log_and_record,
            ));

        let req = Request::builder()
            .method("GET")
            .uri("/boom")
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(
            res.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "a handler panic must surface as 500, not a dropped connection",
        );
        assert!(
            metrics.render().contains(
                r#"sgl_router_responses_total{route="/boom",method="GET",status_code="500"} 1"#
            ),
            "the metrics middleware must observe and count the panic-500; got:\n{}",
            metrics.render(),
        );
    }

    /// The point of counting intake at ENTRY: a request that never produces a
    /// response — client disconnect / cancellation drops the in-flight handler
    /// future at its `.await` — is still counted in `requests_total`, while
    /// `responses_total` stays empty. `requests_total - responses_total` is the
    /// received-but-not-answered gap that exit-only counting can never show. A
    /// handler that sleeps far past our abandon timeout models the dropped future.
    #[tokio::test]
    async fn intake_counted_at_entry_even_when_request_never_completes() {
        let metrics = MetricsRegistry::new();
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(|| async {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    StatusCode::OK
                }),
            )
            .layer(middleware::from_fn_with_state(
                Arc::clone(&metrics),
                access_log_and_record,
            ));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .body(Body::empty())
            .unwrap();
        // Abandon before the handler completes — drops the middleware future at
        // its `.await`, exactly like a client disconnect mid-request.
        let abandoned =
            tokio::time::timeout(std::time::Duration::from_millis(50), app.oneshot(req)).await;
        assert!(abandoned.is_err(), "request must not complete within 50ms");

        let m = metrics.render();
        assert!(
            m.contains(
                r#"sgl_router_requests_total{route="/v1/chat/completions",method="POST"} 1"#
            ),
            "intake must be counted at entry even when the request never completes; got:\n{m}",
        );
        assert!(
            !m.contains(r#"sgl_router_responses_total{route="/v1/chat/completions""#),
            "a never-answered request must NOT appear in responses_total; got:\n{m}",
        );
    }
}
