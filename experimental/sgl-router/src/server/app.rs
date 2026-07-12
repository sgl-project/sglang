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
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
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

/// Collapse an HTTP method to a bounded allow-list for use as a metric label.
/// `http::Method` accepts any RFC-7230 extension token, and this middleware runs
/// before axum's method-router rejects an unknown verb with 405 — so the raw
/// method on `requests_total` / `responses_total` would be attacker-influenced
/// unbounded-cardinality input. Unknown verbs collapse to `other`.
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

/// Router pod identity stamped on every request-status access-log line, so a
/// multi-replica router fleet's aggregated logs show which pod handled each
/// request. Resolved once, lazily, from the environment: `POD_NAME` (a
/// downward-API env var an operator opts into) wins, else `HOSTNAME`
/// (Kubernetes defaults a pod's hostname to its `metadata.name`, which the
/// runtime exposes here), else `"unknown"` (running outside a container with
/// neither set).
static POD_ID: OnceLock<String> = OnceLock::new();

fn pod_id() -> &'static str {
    POD_ID.get_or_init(|| {
        std::env::var("POD_NAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "unknown".to_string())
    })
}

/// Coarse point in the request lifecycle a handler has reached, as observed
/// by the cancelled-request fallback log line (see `AccessLogFallbackGuard`).
/// Distinct from [`crate::server::error::ApiError::stage`] (which attributes a
/// completed error *response*): this attributes an *abandoned* request that
/// never produced one, so the fallback line can tell "the caller hung up
/// while queued for admission" apart from "the caller hung up while the
/// engine was still working" without an engine-side join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum RequestPhase {
    /// Before admission: parsing / validating the request, or not yet
    /// reached the admission gate. The default — a handler that never
    /// advances the phase attributes any cancellation here.
    Ingress,
    /// Parked in (or about to enter) the admission wait queue.
    Queue,
    /// Past admission, waiting on the selected engine.
    Dispatch,
}

impl RequestPhase {
    fn as_str(self) -> &'static str {
        match self {
            RequestPhase::Ingress => "ingress",
            RequestPhase::Queue => "queue",
            RequestPhase::Dispatch => "dispatch",
        }
    }
}

/// Shared, request-scoped cell holding the current [`RequestPhase`]. Inserted
/// into `req.extensions_mut()` by `access_log_and_record` and read by the
/// `AccessLogFallbackGuard`'s `Drop`; the chat handler holds a clone (via the
/// `Extension` extractor) and advances it as the request moves through
/// admission. Only meaningful pre-headers: the fallback guard disarms once
/// the handler returns a response, and a mid-stream drop after that point is
/// the SSE pump's own outcome to record, not this cell's.
///
/// `Relaxed` ordering is enough even though the write (handler) and the read
/// (the guard's `Drop`) can run on different executor threads: safety here
/// comes from the runtime's task handoff, not from the ordering. The guard's
/// `Drop` only runs after the handler's future has been dropped, and dropping
/// a future establishes a happens-before edge over everything the future did
/// while it was still alive (including any `set()` call) — so the read is
/// already ordered-after every write without needing `Acquire`/`Release`.
pub(crate) struct RequestPhaseCell(AtomicU8);

impl Default for RequestPhaseCell {
    fn default() -> Self {
        Self(AtomicU8::new(RequestPhase::Ingress as u8))
    }
}

impl RequestPhaseCell {
    pub(crate) fn set(&self, phase: RequestPhase) {
        self.0.store(phase as u8, Ordering::Relaxed);
    }

    pub(crate) fn get(&self) -> RequestPhase {
        match self.0.load(Ordering::Relaxed) {
            v if v == RequestPhase::Ingress as u8 => RequestPhase::Ingress,
            v if v == RequestPhase::Queue as u8 => RequestPhase::Queue,
            v if v == RequestPhase::Dispatch as u8 => RequestPhase::Dispatch,
            // The only writer is `set()`, which only ever stores one of the
            // three discriminants above — a fourth value here would mean the
            // cell was corrupted or written through some other path, not that
            // a real 4th `RequestPhase` variant was added (that case is
            // exhaustively handled by the arms above).
            v => unreachable!("RequestPhaseCell holds invalid discriminant {v}"),
        }
    }
}

/// Guards `access_log_and_record`'s normal access-log line, which is only
/// reached after `next.run(req).await` resolves. If the client disconnects
/// while that await is still pending, axum drops that function's future
/// there -- a dropped `Future` does not run its remaining code, so without
/// this guard the request's `rid`/method/path would never appear anywhere in
/// the router's own logs even though the request was genuinely received and
/// dispatched (only `record_ingress`, below, would reflect it, as an
/// unexplained gap against `responses_total`). Armed before the await,
/// disarmed once the normal log line is about to fire -- but never for a
/// request on an infra path (`/healthz`, `/readyz`, `/metrics`), which are
/// polled constantly and would otherwise turn routine probe timeouts into
/// WARN-level "client disconnected" noise indistinguishable from a real API
/// client giving up; see the `is_infra_path` check in `access_log_and_record`.
/// A cancelled non-infra request still gets exactly one log line via `Drop`.
struct AccessLogFallbackGuard {
    request_id: String,
    method_label: &'static str,
    path: String,
    route: String,
    start: std::time::Instant,
    is_infra: bool,
    armed: bool,
    phase: Arc<RequestPhaseCell>,
}

impl AccessLogFallbackGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for AccessLogFallbackGuard {
    fn drop(&mut self) {
        if !self.armed || self.is_infra {
            return;
        }
        tracing::warn!(
            pod_id = %pod_id(),
            request_id = %self.request_id,
            method = %self.method_label,
            path = %self.path,
            route = %self.route,
            latency_ms = self.start.elapsed().as_millis() as u64,
            outcome = "cancelled",
            stage = %self.phase.get().as_str(),
            "http_request: client disconnected before a response was produced",
        );
    }
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
    mut req: Request,
    next: Next,
) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    // Metric labels must stay low-cardinality. `route` is the matched route
    // template (`/v1/chat/completions`, …), not the raw URI — `unmatched` for a
    // 404. `method` is collapsed to a known-verb allow-list (`normalize_method`):
    // an arbitrary RFC-7230 extension token reaches this middleware before axum's
    // method-router returns 405, so the raw method is untrusted unbounded input.
    // The access log below keeps the real method.
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".to_owned());
    let method_label = normalize_method(&method);
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_string();
    let start = std::time::Instant::now();

    // ENTRY: true intake — counted before the handler can park/shed/drop it, so
    // a never-answered request still shows up (intake - responses = the gap).
    metrics.record_ingress(&route, method_label);

    // Handed to the handler via `req.extensions_mut()` (advanced as the
    // request moves through admission) and to the fallback guard (read only
    // if the handler's future is dropped before it responds) — see
    // `RequestPhaseCell`'s doc comment.
    let phase = Arc::new(RequestPhaseCell::default());
    req.extensions_mut().insert(Arc::clone(&phase));

    let mut log_guard = AccessLogFallbackGuard {
        request_id: request_id.clone(),
        method_label,
        path: path.clone(),
        route: route.clone(),
        start,
        is_infra: is_infra_path(&path),
        armed: true,
        phase,
    };

    let resp = next.run(req).await;
    // Past this point the function is fully synchronous (no more `.await`),
    // so it will run to completion — safe to disarm the fallback now.
    log_guard.disarm();

    let status = resp.status();
    let latency_ms = start.elapsed().as_millis() as u64;
    // Edge responses count ALL routes (incl. infra) so the intake/response
    // population matches; filter infra by `route` in PromQL.
    metrics.record_response(&route, method_label, status.as_u16());

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
        pod_id = %pod_id(),
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
    let router = Router::new()
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
        );

    // Debug-only CPU flamegraph endpoint — compiled in only with `--features
    // profiling` (see Cargo.toml and `routes::pprof`'s module doc), never
    // present in the normal production image.
    #[cfg(feature = "profiling")]
    let router = router.route(
        "/debug/pprof/profile",
        get(crate::server::routes::pprof::profile),
    );

    router
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

    /// Ensure a real (non-no-op) global default `tracing` subscriber exists for
    /// the whole test binary. `tracing`'s per-callsite interest is computed once,
    /// process-wide, on first hit, and cached; a callsite whose first-ever hit
    /// happens on a thread with no active subscriber (the no-op default) can get
    /// cached as a hard "never interested" that no *later* per-thread
    /// `tracing::subscriber::set_default` override can undo. Any test that can
    /// reach `access_log_and_record`'s abandoned-request path -- which is any
    /// test using the abandon-via-timeout pattern below, not just the ones that
    /// assert on log content -- must call this first, or it can race a sibling
    /// test for which thread hits `AccessLogFallbackGuard`'s `tracing::warn!`
    /// callsite first and poison it for the rest of the process.
    fn ensure_global_tracing_default() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let _ = tracing::subscriber::set_global_default(
                tracing_subscriber::fmt()
                    .with_writer(std::io::sink)
                    .finish(),
            );
        });
    }

    /// The point of counting intake at ENTRY: a request that never produces a
    /// response — client disconnect / cancellation drops the in-flight handler
    /// future at its `.await` — is still counted in `requests_total`, while
    /// `responses_total` stays empty. `requests_total - responses_total` is the
    /// received-but-not-answered gap that exit-only counting can never show. A
    /// handler that sleeps far past our abandon timeout models the dropped future.
    #[tokio::test]
    async fn intake_counted_at_entry_even_when_request_never_completes() {
        ensure_global_tracing_default();
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

    /// Companion to the metrics-side test above: a cancelled request must also
    /// leave a log line, not just a metrics gap. Without `AccessLogFallbackGuard`,
    /// the normal `tracing::info!` call is unreachable here because it sits after
    /// the `.await` this test drops the future at -- this request's `rid` would
    /// never appear anywhere in the router's own logs.
    #[derive(Clone)]
    struct VecWriter(Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for VecWriter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for VecWriter {
        type Writer = VecWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Installs a thread-local `tracing` subscriber (via `set_default`, not the
    /// global default `ensure_global_tracing_default` installs) that captures
    /// every log line into an in-memory buffer, so a test can assert on the
    /// exact content the middleware emitted.
    fn capture_logs() -> (
        Arc<std::sync::Mutex<Vec<u8>>>,
        tracing::subscriber::DefaultGuard,
    ) {
        let buf = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(VecWriter(buf.clone()))
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        (buf, guard)
    }

    #[tokio::test]
    async fn cancelled_request_still_emits_one_log_line() {
        ensure_global_tracing_default();
        let (buf, _guard) = capture_logs();

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
            .header("x-request-id", "test-cancelled-rid")
            .body(Body::empty())
            .unwrap();
        let abandoned =
            tokio::time::timeout(std::time::Duration::from_millis(50), app.oneshot(req)).await;
        assert!(abandoned.is_err(), "request must not complete within 50ms");
        // `timeout` returning Err only means the outer await gave up; the
        // abandoned future's drop glue (and thus this guard's Drop) still has
        // to actually run on the runtime, and exactly when that happens
        // relative to this point is not guaranteed, so poll with a generous
        // ceiling instead of asserting on one fixed wait.
        let mut logs = String::new();
        for _ in 0..20 {
            logs = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
            if logs.contains("test-cancelled-rid") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            logs.contains("test-cancelled-rid"),
            "a cancelled request's rid must still be logged; captured:\n{logs}"
        );
        assert!(
            logs.contains(r#"outcome="cancelled""#) || logs.contains("outcome=cancelled"),
            "the fallback log line must mark the outcome as cancelled; captured:\n{logs}"
        );
        // The stub handler above never advances the phase past its default —
        // the fallback line must attribute the cancellation to `ingress`.
        assert!(
            logs.contains(r#"stage="ingress""#) || logs.contains("stage=ingress"),
            "a handler that never advances the phase must attribute the \
             cancellation to `ingress`; captured:\n{logs}"
        );
    }

    /// Companion to the above: a handler that DOES advance the phase (as the
    /// chat handler does immediately before `ctx.admission.acquire(...)`) must
    /// have that phase reflected on the fallback line — validates the full
    /// extension plumbing (middleware inserts the cell, handler reads and
    /// advances it, guard reads it back on `Drop`) with a stub handler,
    /// without standing up a real admission queue. A real park-and-cancel
    /// through the actual admission queue is covered separately by the chat
    /// integration suite.
    #[tokio::test]
    async fn cancelled_request_after_queue_phase_reports_stage_queue() {
        use axum::extract::Extension;

        ensure_global_tracing_default();
        let (buf, _guard) = capture_logs();

        let metrics = MetricsRegistry::new();
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(|phase: Extension<Arc<RequestPhaseCell>>| async move {
                    phase.set(RequestPhase::Queue);
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
            .header("x-request-id", "test-queue-cancelled-rid")
            .body(Body::empty())
            .unwrap();
        let abandoned =
            tokio::time::timeout(std::time::Duration::from_millis(50), app.oneshot(req)).await;
        assert!(abandoned.is_err(), "request must not complete within 50ms");
        let mut logs = String::new();
        for _ in 0..20 {
            logs = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
            if logs.contains("test-queue-cancelled-rid") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            logs.contains("test-queue-cancelled-rid"),
            "a cancelled request's rid must still be logged; captured:\n{logs}"
        );
        assert!(
            logs.contains(r#"stage="queue""#) || logs.contains("stage=queue"),
            "a handler that advanced to Queue before being cancelled must \
             report stage=queue on the fallback line; captured:\n{logs}"
        );
    }

    /// Mirror of the `Queue` case above for `Dispatch` — the chat handler's
    /// third and last phase, set once a worker has been selected. Together
    /// the two stub tests round-trip all three `RequestPhase` values through
    /// the fallback line (a cancellation that never advances the phase is
    /// covered by `cancelled_request_still_emits_one_log_line`, above).
    #[tokio::test]
    async fn cancelled_request_after_dispatch_phase_reports_stage_dispatch() {
        use axum::extract::Extension;

        ensure_global_tracing_default();
        let (buf, _guard) = capture_logs();

        let metrics = MetricsRegistry::new();
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(|phase: Extension<Arc<RequestPhaseCell>>| async move {
                    phase.set(RequestPhase::Dispatch);
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
            .header("x-request-id", "test-dispatch-cancelled-rid")
            .body(Body::empty())
            .unwrap();
        let abandoned =
            tokio::time::timeout(std::time::Duration::from_millis(50), app.oneshot(req)).await;
        assert!(abandoned.is_err(), "request must not complete within 50ms");
        let mut logs = String::new();
        for _ in 0..20 {
            logs = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
            if logs.contains("test-dispatch-cancelled-rid") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            logs.contains("test-dispatch-cancelled-rid"),
            "a cancelled request's rid must still be logged; captured:\n{logs}"
        );
        assert!(
            logs.contains(r#"stage="dispatch""#) || logs.contains("stage=dispatch"),
            "a handler that advanced to Dispatch before being cancelled must \
             report stage=dispatch on the fallback line; captured:\n{logs}"
        );
    }

    /// A request that completes normally must disarm `AccessLogFallbackGuard`
    /// before the guard's `Drop` can fire, so it gets exactly one log line (the
    /// normal one), never a second "cancelled" line from the fallback.
    #[tokio::test]
    async fn normal_completion_does_not_also_log_cancelled() {
        ensure_global_tracing_default();
        let (buf, _guard) = capture_logs();

        let metrics = MetricsRegistry::new();
        let app = Router::new()
            .route("/v1/chat/completions", post(|| async { StatusCode::OK }))
            .layer(middleware::from_fn_with_state(
                Arc::clone(&metrics),
                access_log_and_record,
            ));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("x-request-id", "test-normal-rid")
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let logs = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(
            logs.contains("test-normal-rid"),
            "a completed request must still be logged; captured:\n{logs}"
        );
        assert!(
            !logs.contains("outcome=\"cancelled\"") && !logs.contains("outcome=cancelled"),
            "a request that completed normally must not ALSO get the fallback \
             cancelled line; captured:\n{logs}"
        );
    }

    /// The fallback guard must stay silent for a cancelled infra-path request
    /// (`/healthz` and friends are polled continuously; a client — the probe —
    /// giving up mid-poll is routine, not something worth a WARN-level "client
    /// disconnected" line indistinguishable from a real API client giving up).
    #[tokio::test]
    async fn cancelled_infra_path_does_not_warn() {
        ensure_global_tracing_default();
        let (buf, _guard) = capture_logs();

        let metrics = MetricsRegistry::new();
        let app = Router::new()
            .route(
                "/healthz",
                get(|| async {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    StatusCode::OK
                }),
            )
            .layer(middleware::from_fn_with_state(
                Arc::clone(&metrics),
                access_log_and_record,
            ));

        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("x-request-id", "test-infra-cancelled-rid")
            .body(Body::empty())
            .unwrap();
        let abandoned =
            tokio::time::timeout(std::time::Duration::from_millis(50), app.oneshot(req)).await;
        assert!(abandoned.is_err(), "request must not complete within 50ms");

        // Give the dropped future's Drop glue a generous window to run, same
        // as `cancelled_request_still_emits_one_log_line` — but here we expect
        // it to run and produce NOTHING, so there is no log-content poll loop
        // to break out of early; the sleep is the entire wait.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let logs = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(
            !logs.contains("test-infra-cancelled-rid"),
            "a cancelled infra-path request must not produce a WARN fallback \
             line; captured:\n{logs}"
        );
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
}
