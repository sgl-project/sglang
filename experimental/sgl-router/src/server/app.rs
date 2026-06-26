// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::metrics::MetricsRegistry;
use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;
use axum::extract::{DefaultBodyLimit, Request, State};
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

/// Middleware: record the final HTTP status of every response into
/// `sgl_router_responses_total{status_code}`. Applied globally as the outermost
/// layer so it observes the status of EVERY response — including errors produced
/// before a handler runs (a 413 from the body-limit layer) and handler
/// short-circuits that return via `?` before reaching their own bookkeeping
/// (a 503 from the admission gate, or any other `ApiError`). Recording here is
/// why handlers no longer count the status themselves: a single site means no
/// outcome is missed and none is double-counted.
async fn record_response_status(
    State(metrics): State<Arc<MetricsRegistry>>,
    req: Request,
    next: Next,
) -> Response {
    let resp = next.run(req).await;
    metrics.record_response(resp.status().as_u16());
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
        // never reaches the `record_response_status` middleware below and is
        // invisible to metrics and logs. Positioned INNER relative to that
        // middleware (added before it, so it sits closer to the handlers) so the
        // synthesized 500 is observed and counted.
        .layer(CatchPanicLayer::new())
        // Outermost layer: count the final status of every response, so error
        // short-circuits (admission 503s, 413s, …) and synthesized panic-500s
        // land in `sgl_router_responses_total` alongside successes.
        .layer(middleware::from_fn_with_state(
            Arc::clone(&ctx.metrics),
            record_response_status,
        ))
        .with_state(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// A handler panic must become a 500 that the `record_response_status`
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
                record_response_status,
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
            metrics
                .render()
                .contains(r#"sgl_router_responses_total{status_code="500"} 1"#),
            "the metrics middleware must observe and count the panic-500; got:\n{}",
            metrics.render(),
        );
    }
}
