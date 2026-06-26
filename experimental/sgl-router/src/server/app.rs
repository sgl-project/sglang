// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;

/// Middleware: count every request at the router HTTP edge. Increments
/// `sgl_router_requests_total{route,method}` at ENTRY — before routing into the
/// handler, before any worker pick or admission parking — so it counts the true
/// intake, including requests that later stall, get shed, or are cancelled
/// before a worker is ever dispatched to. On the way out it records
/// `sgl_router_responses_total{route,method,status_code}` for every response,
/// so early-exit outcomes (400 validation, 413 body-limit, 503 shed) are
/// counted too. `requests_total - responses_total` is then the set of requests
/// the router received but never answered — the silent-overload gap the
/// per-worker `worker_requests_total` (recorded only after dispatch) can't see.
///
/// `route` is the matched route template (e.g. `/v1/chat/completions`), not the
/// raw URI, so an unmatched/garbage path can't blow up label cardinality.
async fn count_requests(State(ctx): State<Arc<AppContext>>, req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_owned();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".to_owned());
    ctx.metrics.record_ingress(&route, &method);
    let resp = next.run(req).await;
    ctx.metrics
        .record_response(&route, &method, resp.status().as_u16());
    resp
}

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
        // Global edge counter — runs after routing (so MatchedPath is set) for
        // every route. from_fn_with_state carries its own ctx handle, so this
        // is independent of the router's `.with_state` below.
        .layer(middleware::from_fn_with_state(ctx.clone(), count_requests))
        .with_state(ctx)
}
