// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP proxy — forwards requests to the upstream SGLang worker.

pub mod sse;

use crate::health::circuit_breaker::CircuitBreaker;
use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use anyhow::Context;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use reqwest::{Client, Url};
use std::sync::Arc;
use std::time::Duration;

/// Parse a worker URL emitted by discovery.  On failure, trip the worker's
/// circuit breaker so the malformed worker drops out of subsequent
/// `healthy_workers_for(...)` selection, then surface the error as
/// `ApiError::WorkerMisconfigured`.
fn parse_worker_url(worker_url: &str, breaker: &CircuitBreaker) -> Result<Url, ApiError> {
    Url::parse(worker_url).map_err(|e| {
        breaker.record_failure();
        ApiError::WorkerMisconfigured {
            worker: worker_url.to_string(),
            source: anyhow::Error::new(e).context("parse worker URL"),
        }
    })
}

/// How an upstream HTTP response status should affect the worker's circuit
/// breaker. Each variant maps to one `CircuitBreaker` call at the dispatch
/// sites (`forward_json_to` / `forward_streaming_to`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BreakerOutcome {
    /// Healthy completion → `record_success`: reset the failure streak and
    /// close the breaker.
    Success,
    /// A real fault (5xx other than backpressure) → `record_failure`: count
    /// toward opening.
    Failure,
    /// Backpressure (the worker is responsive but at capacity) →
    /// `record_backpressure`: never opens the breaker and, while Closed, leaves
    /// an in-progress failure streak intact — but still resolves a half-open
    /// probe so a recovered-but-busy worker isn't wedged shut.
    Neutral,
}

/// Classify an upstream status for circuit-breaker accounting.
///
/// A backpressure status — `503 Service Unavailable` or `429 Too Many
/// Requests` — is the worker signalling "responsive but at capacity", not a
/// fault. Counting it as a breaker failure is actively harmful: a saturated
/// worker trips the breaker on its own queue-full 503s, and with a single
/// worker the router then sheds *every* request for the whole cool-down —
/// including after the engine has drained and gone idle. So backpressure is
/// [`Neutral`](BreakerOutcome::Neutral) (see [`CircuitBreaker::record_backpressure`]
/// for its exact effect per breaker state). Genuine 5xx faults (500 / 502 /
/// 504 / …) still count as failures, and transport errors / timeouts /
/// mid-body drops are recorded as failures at the call sites.
///
/// Tradeoff: because 503 never opens the breaker, a worker stuck returning 503
/// indefinitely (a wedged engine, not transient load) is NOT detected here —
/// HTTP status alone can't distinguish "busy" from "broken-and-saying-503", and
/// counting it caused the worse fleet-wide false-shed above. Detecting a
/// chronically-backpressuring worker is left to higher-level signals.
fn breaker_outcome(status: reqwest::StatusCode) -> BreakerOutcome {
    use reqwest::StatusCode;
    match status {
        StatusCode::SERVICE_UNAVAILABLE | StatusCode::TOO_MANY_REQUESTS => BreakerOutcome::Neutral,
        s if s.is_server_error() => BreakerOutcome::Failure,
        _ => BreakerOutcome::Success,
    }
}

/// Idle (between-bytes) timeout for streaming upstream responses. A stream that
/// delivers no bytes for this long is treated as hung and aborted, releasing the
/// admission / active-load guards it holds. Distinct from `request_timeout` (the
/// total budget, which streaming deliberately skips so long generations can run):
/// this fires only on a *stall*, not on slow-but-progressing generation. Without
/// it, a half-open upstream (e.g. a worker killed mid-stream) pins the SSE pump
/// and leaks the per-worker in-flight slot forever.
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug)]
pub struct Proxy {
    pub client: Client,
    /// Wall-clock timeout applied to non-streaming upstream requests. Streaming
    /// requests deliberately do not use this (long generations are valid).
    pub request_timeout: Duration,
}

impl Proxy {
    /// Build a proxy. `request_timeout` is the per-request wall-clock budget for
    /// non-streaming forwards. Connect timeout is hard-coded to 5 s — even a
    /// streaming request fails fast at TCP setup if the worker is unreachable.
    pub fn new(request_timeout: Duration) -> Result<Self, anyhow::Error> {
        let client = Client::builder()
            .pool_max_idle_per_host(64)
            .connect_timeout(Duration::from_secs(5))
            .build()
            .context("build reqwest client")?;
        Ok(Self {
            client,
            request_timeout,
        })
    }

    /// Classify a reqwest error into the right `ApiError` variant, given an
    /// explicit worker URL. Called from the breaker-gated `forward_*_to`
    /// methods, which carry per-request worker URLs (not a single proxy-level
    /// URL).
    ///
    /// Walks the full source chain to detect timeouts, because reqwest wraps
    /// hyper which wraps `std::io::Error` — a top-level `is_timeout()` check
    /// misses both the wrapped reqwest timeout and the `io::ErrorKind::TimedOut`
    /// cases.
    fn classify_reqwest_error_for(worker: Url, e: reqwest::Error, path: &str) -> ApiError {
        let source = anyhow::Error::new(e).context(format!("worker {worker}: post {path}"));
        let is_timeout = source.chain().any(|c| {
            c.downcast_ref::<reqwest::Error>()
                .is_some_and(|r| r.is_timeout())
        }) || source.chain().any(|c| {
            c.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::TimedOut)
        });
        if is_timeout {
            ApiError::UpstreamTimeout { worker }
        } else {
            ApiError::UpstreamUnreachable { worker, source }
        }
    }

    /// Breaker-gated JSON POST: checks `breaker.allow()` first, records
    /// success/failure based on response status, and returns
    /// `ApiError::BreakerOpen` immediately when the breaker is Open.
    ///
    /// `worker_url` is the discovery-emitted worker URL string. It's parsed
    /// to [`reqwest::Url`] internally so we can use [`Url::join`] for clean
    /// path concatenation (no double-slash) and pass a typed URL to the
    /// split error variants (`UpstreamUnreachable` / `UpstreamTimeout` /
    /// `UpstreamStatus`).
    pub async fn forward_json_to(
        &self,
        worker_url: &str,
        breaker: &CircuitBreaker,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .timeout(self.request_timeout);
        let resp = req.send().await.map_err(|e| {
            breaker.record_failure();
            Self::classify_reqwest_error_for(worker_url.clone(), e, path)
        })?;
        let status = resp.status();
        // Defer breaker recording until after the body completes — a
        // worker that returns 2xx headers and then drops mid-body is
        // still failing the request, and crediting it as healthy lets
        // a misbehaving worker stay eligible. For 5xx the early bail is
        // safe (no body to consume meaningfully), but we still wait
        // until after the read attempt to record exactly once.
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                // Walk the full source chain (`{:#}`) like the connect-error
                // handler in `classify_reqwest_error_for` — a mid-body drop's
                // real cause (incomplete message, connection reset) lives in the
                // wrapped source, not the outer reqwest error.
                let cause = anyhow::Error::new(e);
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = %format_args!("{cause:#}"),
                    "upstream dropped connection mid-body",
                );
                breaker.record_failure();
                return Err(ApiError::UpstreamStatus { status });
            }
        };
        match breaker_outcome(status) {
            BreakerOutcome::Failure => breaker.record_failure(),
            BreakerOutcome::Success => breaker.record_success(),
            // Backpressure (503/429): the engine is healthy but busy. This never
            // opens the breaker and (in Closed) leaves the failure streak
            // intact, but it DOES resolve a half-open probe so a recovered
            // worker that answers a probe with 503 isn't wedged shut.
            BreakerOutcome::Neutral => breaker.record_backpressure(),
        }
        let mut out = Response::new(Body::from(bytes));
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    /// Breaker-gated streaming POST: checks `breaker.allow()` first, records
    /// success/failure, and returns `ApiError::BreakerOpen` when Open.
    ///
    /// `stream_guards` — when `Some`, the value is threaded into the SSE
    /// pump task and held for the entire body lifetime (headers → last byte
    /// / client disconnect).  The proxy does not inspect the boxed value; it
    /// relies entirely on `Drop` semantics, so callers typically pack
    /// `(LoadGuard, ActiveLoadGuard)` here. This keeps both the per-worker
    /// `active_requests` counter and the per-request active-load entry alive
    /// for the full streaming lifetime — without which a long-running SSE
    /// response would under-report load.
    // Each parameter is a distinct, required input to a single upstream
    // forward (target, breaker, path, headers, body, plus the two
    // streaming-lifetime callbacks). Bundling them into a struct purely to
    // satisfy the arg-count heuristic would add indirection without clarity.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_streaming_to(
        &self,
        worker_url: &str,
        breaker: &Arc<CircuitBreaker>,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
        stream_guards: Option<Box<dyn Send + 'static>>,
        on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");
        let resp = req.send().await.map_err(|e| {
            breaker.record_failure();
            Self::classify_reqwest_error_for(worker_url.clone(), e, path)
        })?;
        let status = resp.status();
        let upstream_ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        let content_type = if status.is_success() {
            "text/event-stream".to_string()
        } else {
            upstream_ct
        };
        // Breaker recording is deferred to the pump's completion hook so
        // an upstream that returns 2xx headers and then drops mid-stream
        // is recorded as a failure. For a genuine 5xx fault we record_failure
        // up front and skip the pump hook (the body we surface is the
        // error response — its stream completing is not a worker win). For a
        // backpressure status (503/429) we record_backpressure up front and
        // skip the hook: a busy-but-healthy engine's queue-full responses can't
        // open the breaker, but a half-open probe answered with 503 is still
        // resolved rather than wedged (see `breaker_outcome` / `record_backpressure`).
        let on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>> =
            match breaker_outcome(status) {
                BreakerOutcome::Failure => {
                    breaker.record_failure();
                    None
                }
                BreakerOutcome::Neutral => {
                    breaker.record_backpressure();
                    None
                }
                BreakerOutcome::Success => {
                    let breaker_for_hook = Arc::clone(breaker);
                    Some(Box::new(move |ok| {
                        if ok {
                            breaker_for_hook.record_success();
                        } else {
                            breaker_for_hook.record_failure();
                        }
                    }))
                }
            };
        // Only record TTFT for successful streams — a 4xx/5xx error body
        // streaming back is not a generated token, so drop the hook for
        // non-2xx responses.
        let first_byte_hook = if status.is_success() {
            on_first_byte
        } else {
            None
        };
        let body = sse::bytes_stream_to_body(
            sse::idle_timeout_stream(resp.bytes_stream(), STREAM_IDLE_TIMEOUT),
            stream_guards,
            on_complete,
            first_byte_hook,
        );
        let mut out = Response::new(body);
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_str(&content_type)
                .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use axum::routing::post;
    use axum::Router;
    use reqwest::StatusCode;
    use std::num::NonZeroU32;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    #[tokio::test]
    async fn new_returns_result_not_panic() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        assert_eq!(p.request_timeout, Duration::from_secs(5));
    }

    #[test]
    fn breaker_outcome_treats_backpressure_as_neutral() {
        // Backpressure: healthy but busy — must not touch the breaker.
        assert_eq!(
            breaker_outcome(StatusCode::SERVICE_UNAVAILABLE),
            BreakerOutcome::Neutral,
        );
        assert_eq!(
            breaker_outcome(StatusCode::TOO_MANY_REQUESTS),
            BreakerOutcome::Neutral,
        );
        // Genuine faults: still failures.
        for s in [
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::GATEWAY_TIMEOUT,
        ] {
            assert_eq!(breaker_outcome(s), BreakerOutcome::Failure, "{s}");
        }
        // Non-5xx (incl. 4xx client errors): treated as success.
        for s in [
            StatusCode::OK,
            StatusCode::BAD_REQUEST,
            StatusCode::NOT_FOUND,
        ] {
            assert_eq!(breaker_outcome(s), BreakerOutcome::Success, "{s}");
        }
    }

    /// A fake upstream that answers every POST with a fixed status + tiny body.
    async fn spawn_status_worker(status: u16) -> (String, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let code = StatusCode::from_u16(status).unwrap();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || async move { (code, "{\"error\":\"x\"}") }),
        );
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), tx)
    }

    /// The bug this fixes: a saturated engine returning its own queue-full 503s
    /// must NOT trip the router's circuit breaker. Dispatch well past the
    /// default threshold (3) and assert the breaker stays Closed and admitting.
    #[tokio::test]
    async fn engine_503_does_not_trip_breaker() {
        let (url, _shutdown) = spawn_status_worker(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = CircuitBreaker::new(); // default threshold = 3
        let headers = HeaderMap::new();

        for i in 0..6 {
            let resp = proxy
                .forward_json_to(
                    &url,
                    &breaker,
                    "/v1/chat/completions",
                    &headers,
                    Bytes::from_static(b"{}"),
                )
                .await
                .expect("dispatch should reach the worker (breaker must stay closed)");
            assert_eq!(
                resp.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "iter {i}: client must still see the engine's 503",
            );
            assert_eq!(
                breaker.snapshot().state_code,
                0,
                "iter {i}: 503 backpressure must leave the breaker Closed",
            );
        }
        assert!(
            breaker.would_allow(),
            "breaker must keep admitting after a burst of engine 503s",
        );
    }

    /// Contrast / regression guard: a genuine 5xx fault (500) MUST still trip the
    /// breaker after the threshold, so the backpressure carve-out didn't disable
    /// fault detection.
    #[tokio::test]
    async fn engine_500_still_trips_breaker() {
        let (url, _shutdown) = spawn_status_worker(500).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = CircuitBreaker::new(); // default threshold = 3
        let headers = HeaderMap::new();

        for _ in 0..3 {
            let _ = proxy
                .forward_json_to(
                    &url,
                    &breaker,
                    "/v1/chat/completions",
                    &headers,
                    Bytes::from_static(b"{}"),
                )
                .await;
        }
        assert_eq!(
            breaker.snapshot().state_code,
            1,
            "three 500s must open the breaker (fault detection still works)",
        );
    }

    /// End-to-end wedge guard: a breaker that opened on real faults, then has
    /// its half-open probe answered with a 503, must RECOVER — not stay shut
    /// out forever. Exercises the `Neutral => record_backpressure` wiring in
    /// `forward_json_to` through the half-open path.
    #[tokio::test]
    async fn engine_503_recovers_a_half_open_breaker() {
        let (url, _shutdown) = spawn_status_worker(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        // threshold=1 so one prior fault opens it; tiny cooldown so the probe
        // is admitted almost immediately.
        let breaker = CircuitBreaker::with_config(CircuitBreakerConfig {
            threshold: NonZeroU32::new(1).unwrap(),
            cool_down: Duration::from_millis(50),
        });
        let headers = HeaderMap::new();

        // Simulate a prior genuine fault (e.g. a 500 / timeout) that tripped it.
        breaker.record_failure();
        assert_eq!(breaker.snapshot().state_code, 1, "breaker should be Open");

        // Let the cooldown elapse so the next dispatch claims the half-open probe.
        tokio::time::sleep(Duration::from_millis(80)).await;

        let resp = proxy
            .forward_json_to(
                &url,
                &breaker,
                "/v1/chat/completions",
                &headers,
                Bytes::from_static(b"{}"),
            )
            .await
            .expect("the half-open probe must be admitted and reach the worker");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            breaker.snapshot().state_code,
            0,
            "a 503 answer to the probe must close the breaker, not wedge it half-open",
        );
        assert!(
            breaker.would_allow(),
            "worker must admit traffic again after recovering from the probe",
        );
    }

    /// A worker's own response is forwarded with its status VERBATIM and carries
    /// NO `x-router-error-code` header. The absence of that header is exactly how
    /// a gateway tells "the engine said this" from "the router said this" — a
    /// worker 2xx, 4xx, and 5xx (incl. a complete 500, distinct from a synthesized
    /// 502 mid-body drop) all pass straight through, unannotated.
    #[tokio::test]
    async fn forwarded_worker_response_is_verbatim_with_no_router_error_code() {
        for status in [200u16, 400, 500, 503] {
            let (url, _shutdown) = spawn_status_worker(status).await;
            let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
            let breaker = CircuitBreaker::new();
            let resp = proxy
                .forward_json_to(
                    &url,
                    &breaker,
                    "/v1/chat/completions",
                    &HeaderMap::new(),
                    Bytes::from_static(b"{}"),
                )
                .await
                .expect("dispatch should reach the worker");
            assert_eq!(
                resp.status().as_u16(),
                status,
                "worker status {status} must be forwarded verbatim",
            );
            assert!(
                resp.headers().get("x-router-error-code").is_none(),
                "a forwarded worker response must NOT carry x-router-error-code (status {status})",
            );
            assert!(
                resp.headers().get("x-router-upstream-status").is_none(),
                "a forwarded worker response must NOT carry x-router-upstream-status (status {status})",
            );
        }
    }

    /// Streaming path parity: the engine's 503 on the streaming arm must also
    /// leave the breaker untouched (no up-front failure, no completion hook).
    #[tokio::test]
    async fn engine_503_does_not_trip_breaker_streaming() {
        use http_body_util::BodyExt;

        let (url, _shutdown) = spawn_status_worker(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let headers = HeaderMap::new();

        for i in 0..6 {
            let resp = proxy
                .forward_streaming_to(
                    &url,
                    &breaker,
                    "/v1/chat/completions",
                    &headers,
                    Bytes::from_static(b"{}"),
                    None,
                    None,
                )
                .await
                .expect("streaming dispatch should reach the worker");
            assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE, "iter {i}");
            // Drain the body so the pump task runs to completion (would fire any
            // completion hook). For a 503 there is none, but draining proves it.
            let _ = resp.into_body().collect().await;
            assert_eq!(
                breaker.snapshot().state_code,
                0,
                "iter {i}: streaming 503 must leave the breaker Closed",
            );
        }
    }
}
