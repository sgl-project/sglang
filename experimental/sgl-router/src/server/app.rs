// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::metrics::{outcome_from_status, RequestLogContext};
use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use std::sync::{Arc, OnceLock};
use tower_http::catch_panic::CatchPanicLayer;

/// Infra endpoints logged at DEBUG rather than INFO. They are polled constantly
/// — Prometheus scrapes `/metrics`, the kubelet hits `/healthz` + `/readyz`
/// every few seconds — so logging them at INFO would bury real API traffic. They
/// are still counted in both edge counters; filter them by `route` in PromQL.
fn is_infra_path(path: &str) -> bool {
    matches!(path, "/healthz" | "/readyz" | "/metrics")
}

/// Collapse an HTTP method to a bounded allow-list for use as a metric label.
/// `http::Method` accepts any RFC-7230 extension token, and this middleware runs
/// before axum's method-router rejects an unknown verb with 405 — so the raw
/// method on `requests_total` / `responses_total` would be caller-controlled
/// unbounded-cardinality input. Unknown verbs collapse to `other`. The access
/// log below keeps the real method.
fn normalize_method(method: &axum::http::Method) -> &'static str {
    use axum::http::Method;
    match *method {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::DELETE => "DELETE",
        Method::PATCH => "PATCH",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        Method::TRACE => "TRACE",
        Method::CONNECT => "CONNECT",
        _ => "other",
    }
}

/// Router pod identity stamped on every access-log line, so a multi-replica
/// router fleet's aggregated logs show which pod handled each request. Resolved
/// once, lazily, from the environment: `POD_NAME` (a downward-API env var an
/// operator opts into) wins, else `HOSTNAME` (Kubernetes defaults a pod's
/// hostname to its `metadata.name`, which the runtime exposes here), else
/// `"unknown"` (running outside a container with neither set).
static POD_ID: OnceLock<String> = OnceLock::new();

fn pod_id() -> &'static str {
    POD_ID.get_or_init(|| {
        std::env::var("POD_NAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "unknown".to_string())
    })
}

/// Outermost middleware: the single access-log and edge-counter site.
///
/// Edge counters: `requests_total{route,method}` at entry (true intake, incl.
/// requests parked/shed/cancelled before dispatch), `responses_total{...,
/// status_code}` on exit (incl. early-exit 400/413/503). Their difference =
/// received-but-not-answered, invisible to post-dispatch `worker_requests_total`.
/// `route` is the matched template (not the raw URI) and `method` a known-verb
/// allow-list, so neither label's cardinality is caller-controlled.
///
/// The access log runs at the same site, which is what makes it complete: it
/// covers responses produced before any handler runs (a 413 from the body-limit
/// layer, a 400 from the body extractor when a client drops the connection
/// mid-upload, a `CatchPanicLayer` 500) and handler short-circuits that return
/// via `?` (a body-validation 400, a model-not-found 404) — none of which can
/// reach a handler's own logging. Routed requests carry a [`RequestLogContext`]
/// naming the worker and model; pre-routing rejections log those fields empty.
async fn access_log_and_record(
    State(ctx): State<Arc<AppContext>>,
    req: Request,
    next: Next,
) -> Response {
    let method = req.method().clone();
    let method_label = normalize_method(&method);
    let path = req.uri().path().to_owned();
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
        .to_owned();
    let start = std::time::Instant::now();

    ctx.metrics.record_ingress(&route, method_label);
    let resp = next.run(req).await;
    let status = resp.status();
    let latency_ms = start.elapsed().as_millis() as u64;
    ctx.metrics
        .record_response(&route, method_label, status.as_u16());

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

    // Per-worker fields are present only when a handler routed the request and
    // attached them; a pre-routing rejection logs them empty.
    let log_ctx = resp.extensions().get::<RequestLogContext>();
    tracing::info!(
        pod_id = %pod_id(),
        request_id = %request_id,
        method = %method,
        path = %path,
        status = status.as_u16(),
        outcome = outcome_from_status(status.as_u16()).as_str(),
        worker = log_ctx.map(|c| c.worker_url.as_str()).unwrap_or(""),
        model = log_ctx.map(|c| c.model_id.as_str()).unwrap_or(""),
        stream = log_ctx.is_some_and(|c| c.streaming),
        latency_ms,
        "http_request",
    );
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
        // Convert a handler panic into a 500 response. hyper otherwise catches
        // the panic and drops the connection WITHOUT a Response, so the failure
        // never reaches the `access_log_and_record` middleware below and is
        // invisible to both the edge counters and the access log. Positioned
        // INNER relative to that middleware (added before it, so it sits closer
        // to the handlers) so the synthesized 500 is observed and counted.
        .layer(CatchPanicLayer::new())
        // After routing, so MatchedPath is set for every route.
        .layer(middleware::from_fn_with_state(
            ctx.clone(),
            access_log_and_record,
        ))
        .with_state(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use axum::response::IntoResponse;
    use std::sync::Mutex;
    use tower::ServiceExt;
    use tracing_subscriber::fmt::MakeWriter;

    /// Capture the `tracing` output of one test into a buffer.
    #[derive(Clone)]
    struct VecWriter(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for VecWriter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for VecWriter {
        type Writer = VecWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Install a buffer-backed subscriber for the current thread. The returned
    /// guard must stay alive for the duration of the capture.
    fn capture_logs() -> (Arc<Mutex<Vec<u8>>>, tracing::subscriber::DefaultGuard) {
        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(VecWriter(buf.clone()))
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        (buf, guard)
    }

    fn captured(buf: &Arc<Mutex<Vec<u8>>>) -> String {
        String::from_utf8(buf.lock().unwrap().clone()).unwrap()
    }

    /// Metric labels must stay bounded: standard verbs pass through, but an
    /// arbitrary RFC-7230 extension token (which reaches this middleware before
    /// axum's 405) collapses to `other` so it can't explode label cardinality.
    #[test]
    fn normalize_method_collapses_unknown_verbs() {
        use axum::http::Method;
        assert_eq!(normalize_method(&Method::GET), "GET");
        assert_eq!(normalize_method(&Method::POST), "POST");
        let exotic = Method::from_bytes(b"BREW").unwrap();
        assert_eq!(
            normalize_method(&exotic),
            "other",
            "an unknown verb must collapse to `other`, not mint a new label series",
        );
    }

    /// An unrouted path must collapse to `route="unmatched"` and never put the
    /// raw URI in a metric label — otherwise any caller could mint unbounded
    /// label series by walking made-up paths.
    #[tokio::test]
    async fn unmatched_route_does_not_leak_the_raw_uri_into_metrics() {
        let ctx = Arc::new(AppContext::stub());
        let req = Request::builder()
            .method("GET")
            .uri("/not/a/real/route-9f3c")
            .body(Body::empty())
            .unwrap();
        let res = build_router(Arc::clone(&ctx)).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);

        let m = ctx.metrics.render();
        assert!(
            m.contains(r#"sgl_router_requests_total{route="unmatched",method="GET"} 1"#),
            "an unrouted request must be counted under route=\"unmatched\": {m}",
        );
        assert!(
            !m.contains("route-9f3c"),
            "the raw URI must never reach a metric label: {m}",
        );
    }

    /// A handler panic must become a 500 that the `access_log_and_record`
    /// middleware still observes. hyper catches a handler panic and drops the
    /// connection WITHOUT producing a Response, so without `CatchPanicLayer`
    /// the failure is invisible to the edge counters and the access log.
    /// `CatchPanicLayer` synthesizes a 500; the middleware must be OUTER
    /// (applied after) so it counts and logs that synthesized 500.
    ///
    /// This composes the SAME two layers in the SAME order as `build_router`
    /// (middleware outer, catch-panic inner) over a panicking route — the real
    /// `build_router` has no panicking route to exercise, so a minimal Router
    /// pins the ordering contract directly.
    #[tokio::test]
    async fn handler_panic_becomes_500_and_is_counted() {
        let ctx = Arc::new(AppContext::stub());
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
            // Outer: count and log the final status — must observe the 500.
            .layer(middleware::from_fn_with_state(
                Arc::clone(&ctx),
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
        let m = ctx.metrics.render();
        assert!(
            m.contains(r#"status_code="500""#),
            "the middleware must observe and count the panic-500; got:\n{m}",
        );
    }

    /// A request rejected BEFORE any handler runs — here an unrouted path, which
    /// axum 404s with no handler involved at all — must still produce an access
    /// log line. This is the gap a per-handler log cannot close: the response
    /// exists, but no handler ever saw the request. The same site covers the
    /// body-limit 413, the extractor 400 from a client that drops mid-upload,
    /// and the `?` short-circuits inside a handler.
    #[tokio::test]
    async fn request_that_never_reaches_a_handler_is_still_logged() {
        let (buf, _guard) = capture_logs();
        let ctx = Arc::new(AppContext::stub());
        let req = Request::builder()
            .method("GET")
            .uri("/nope")
            .header("x-request-id", "rid-unrouted")
            .body(Body::empty())
            .unwrap();
        let res = build_router(ctx).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);

        let logs = captured(&buf);
        assert!(
            logs.contains("http_request") && logs.contains("rid-unrouted"),
            "every request must be logged by the middleware; captured:\n{logs}",
        );
        assert!(
            logs.contains("status=404") && logs.contains("outcome=\"error\""),
            "the access log must carry the final status and its outcome; captured:\n{logs}",
        );
    }

    /// A routed request carries the worker and model it was dispatched to, via
    /// the [`RequestLogContext`] its handler attaches to the response — so the
    /// one central log line can still answer "which engine served this?". The
    /// middleware cannot see that itself; without the extension the fields would
    /// be blank on every line.
    #[tokio::test]
    async fn routed_response_context_names_the_worker_in_the_access_log() {
        let (buf, _guard) = capture_logs();
        let ctx = Arc::new(AppContext::stub());
        let app = Router::new()
            .route(
                "/routed",
                get(|| async {
                    let mut resp = StatusCode::OK.into_response();
                    resp.extensions_mut().insert(RequestLogContext {
                        worker_url: "http://worker-a:30000".into(),
                        model_id: "tiny".into(),
                        streaming: false,
                    });
                    resp
                }),
            )
            .layer(middleware::from_fn_with_state(
                Arc::clone(&ctx),
                access_log_and_record,
            ));

        let req = Request::builder()
            .method("GET")
            .uri("/routed")
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let logs = captured(&buf);
        assert!(
            logs.contains("worker=\"http://worker-a:30000\"") && logs.contains("model=\"tiny\""),
            "a routed request must be logged with its worker and model; captured:\n{logs}",
        );
        assert!(
            logs.contains("outcome=\"success\""),
            "a 200 must log outcome=success; captured:\n{logs}",
        );
    }

    /// Infra probes (`/healthz`, `/readyz`, `/metrics`) are polled every few
    /// seconds by the kubelet and Prometheus. They log at DEBUG, so a default
    /// INFO subscriber sees nothing — otherwise probe traffic buries real API
    /// traffic. They are still counted in both edge counters.
    #[tokio::test]
    async fn infra_probe_is_counted_but_not_logged_at_info() {
        let (buf, _guard) = capture_logs();
        let ctx = Arc::new(AppContext::stub());
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .body(Body::empty())
            .unwrap();
        let res = build_router(Arc::clone(&ctx)).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        assert!(
            !captured(&buf).contains("http_request"),
            "infra probes must not reach the INFO access log; captured:\n{}",
            captured(&buf),
        );
        let m = ctx.metrics.render();
        assert!(
            m.contains(r#"sgl_router_requests_total{route="/healthz",method="GET"} 1"#)
                && m.contains(
                    r#"sgl_router_responses_total{route="/healthz",method="GET",status_code="200"} 1"#
                ),
            "infra probes must still be counted at the edge: {m}",
        );
    }
}
