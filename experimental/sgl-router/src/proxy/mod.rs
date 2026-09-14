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
/// breaker, at the dispatch sites (`forward_json_to` / `forward_streaming_to`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BreakerOutcome {
    /// The worker was responsive — a 2xx, or a 4xx it answered cleanly (a
    /// client's bad request says nothing about worker health). The non-streaming
    /// arm records success immediately; the streaming arm defers to the pump's
    /// completion hook, which records success or failure by
    /// [`sse::StreamEnd::transport_ok`], since a 2xx head can still be followed
    /// by a body that never completes.
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
/// mid-body drops are recorded as failures at the call sites — as are a
/// malformed discovery URL (`parse_worker_url`) and a stream whose body dies
/// after a 2xx head.
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

/// How long an abort POST may take before we give up. The client is already
/// gone, so this only bounds how long the fire-and-forget task lingers; a slow
/// abort must never wedge a worker's connection pool.
const ABORT_TIMEOUT: Duration = Duration::from_secs(5);

/// Tell an engine to stop generating a request whose client has disconnected,
/// by `POST`ing `/abort_request {rid, abort_all:false}`. The engine's scheduler
/// cancels every in-flight request whose `rid` starts with this one.
///
/// `auth` replays the request's own `Authorization` header. SGLang marks
/// `/abort_request` `ADMIN_OPTIONAL`, so an engine started with `--api-key`
/// rejects an unauthenticated abort with 401 — silently turning the whole
/// feature off on exactly the deployments that secure their engines. The
/// generate request carried this header upstream already (see
/// [`should_forward_request_header`]), so replaying it is the same credential
/// the worker just accepted.
///
/// Best-effort by construction: the client is already gone, so there is no one
/// to surface an error to, and a missed abort wastes engine compute but is not
/// a correctness fault. Failures are logged, not propagated. Not circuit-breaker
/// gated — an abort is a courtesy to the engine, never counted against a worker.
/// "Logged" means visibly: a refused abort (401 from an admin-key engine, 404
/// from a mis-derived URL, 422 from a payload the engine no longer accepts)
/// warns like a transport failure does, because it fails the same way — the
/// engine keeps generating.
async fn send_abort(client: &Client, abort_url: &str, rid: &str, auth: Option<&HeaderValue>) {
    let body = serde_json::json!({ "rid": rid, "abort_all": false });
    let mut req = client.post(abort_url).json(&body).timeout(ABORT_TIMEOUT);
    if let Some(auth) = auth {
        req = req.header(reqwest::header::AUTHORIZATION, auth);
    }
    match req.send().await {
        Ok(resp) if resp.status().is_success() => tracing::debug!(
            abort_url,
            rid,
            status = %resp.status(),
            "told engine to abort request after client disconnect",
        ),
        Ok(resp) => tracing::warn!(
            abort_url,
            rid,
            status = %resp.status(),
            "engine refused the abort; it may still be generating this request",
        ),
        Err(e) => tracing::warn!(
            abort_url,
            rid,
            error = %e,
            "failed to send abort to engine after client disconnect",
        ),
    }
}

/// The `Authorization` header to replay on an abort for a request carrying
/// `headers`, if any. Split out so both guard constructors agree on it.
fn abort_auth(headers: &HeaderMap) -> Option<HeaderValue> {
    headers
        .get(reqwest::header::AUTHORIZATION.as_str())
        .cloned()
}

/// Whether the engine may still be generating, given how the SSE pump reported
/// the stream ended.
///
/// The one shape that proves the engine stopped on its own is a stream drained
/// to its clean end with the client still attached. A `client_disconnect` means
/// the reader went away mid-generation. A transport failure (which
/// [`Proxy::forward_streaming_to`] folds a pump panic into) means the router
/// lost the connection to the engine, which is no evidence at all about what
/// the engine is doing — and is precisely when it may still be producing tokens
/// for a reader that no longer exists.
///
/// `saw_error_event` is deliberately not consulted: an engine reporting its own
/// failure as an SSE `data: {"error"…}` event then closes the stream cleanly,
/// and that clean close is what says it is done.
///
/// A redundant abort costs one POST the engine answers by finding no such rid;
/// a missed one costs a whole generation, so the doubtful cases abort.
fn engine_may_still_be_generating(end: sse::StreamEnd) -> bool {
    end.client_disconnect || !end.transport_ok
}

/// Drop guard that aborts an in-flight engine request when the client goes
/// away. Spawns [`send_abort`] from its `Drop` when (and only when) it is still
/// "armed" at drop time.
///
/// This covers the **non-streaming** forward and the window before a streaming
/// forward has a response: armed until [`disarm`](Self::disarm) is called. The
/// handler disarms it once a complete response is in hand, so a drop while
/// armed means the handler future was cancelled (client disconnect) or the
/// stale-request janitor fired — both cases where the engine may still be
/// working and should be told to stop. Once a stream is established the pump's
/// own completion report takes over, via
/// [`engine_may_still_be_generating`].
pub(crate) struct AbortOnDrop {
    client: Client,
    abort_url: String,
    rid: String,
    /// The request's own `Authorization` header, replayed on the abort — see
    /// [`send_abort`].
    auth: Option<HeaderValue>,
    armed: bool,
}

impl AbortOnDrop {
    fn new(client: Client, abort_url: String, rid: String, auth: Option<HeaderValue>) -> Self {
        Self {
            client,
            abort_url,
            rid,
            auth,
            armed: true,
        }
    }

    /// Mark the request as completed so the guard does NOT abort on drop. Call
    /// once a full response has been received from the engine.
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let client = self.client.clone();
        let abort_url = std::mem::take(&mut self.abort_url);
        let rid = std::mem::take(&mut self.rid);
        let auth = self.auth.take();
        // `Drop` is sync; the POST is async and fire-and-forget. We need a
        // runtime handle to spawn it — present on every normal drop (the
        // handler / SSE pump run on the tokio runtime). It is absent only when
        // the runtime itself is tearing down, in which case the process is
        // exiting and there is no point chasing an abort.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(
                    async move { send_abort(&client, &abort_url, &rid, auth.as_ref()).await },
                );
            }
            Err(_) => tracing::debug!(
                %abort_url,
                %rid,
                "no tokio runtime at drop (shutdown); skipping engine abort",
            ),
        }
    }
}

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

    /// Build an [`AbortOnDrop`] guard for a **non-streaming** forward to
    /// `worker_url`. Hold it across the forward; disarm it once a complete
    /// response is in hand. If it instead drops while armed — the handler
    /// future was cancelled by a client disconnect, or the stale-request
    /// janitor fired — it `POST`s `/abort_request` so the engine stops
    /// generating a reply no one will read.
    ///
    /// `rid` must be the request id the router injected into the forwarded body
    /// (so the engine's request carries it). `headers` are the client's, so the
    /// abort can replay their `Authorization` — see [`send_abort`]. Returns
    /// `None` only if `worker_url` can't be parsed — in which case the forward
    /// itself fails the same way and the absent guard is moot. Streaming
    /// forwards get their guard internally via
    /// [`forward_streaming_to`](Self::forward_streaming_to)'s `abort_rid`.
    pub(crate) fn abort_guard_for(
        &self,
        worker_url: &str,
        rid: &str,
        headers: &HeaderMap,
    ) -> Option<AbortOnDrop> {
        let abort_url = Url::parse(worker_url).ok()?.join("/abort_request").ok()?;
        Some(AbortOnDrop::new(
            self.client.clone(),
            abort_url.to_string(),
            rid.to_string(),
            abort_auth(headers),
        ))
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

    /// Breaker-gated JSON POST: checks `breaker.allow()` first, classifies the
    /// response status through [`breaker_outcome`] (success / failure /
    /// backpressure), and returns `ApiError::BreakerOpen` immediately when the
    /// breaker is Open.
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

    /// Breaker-gated streaming POST: checks `breaker.allow()` first, classifies
    /// the response status through [`breaker_outcome`], and returns
    /// `ApiError::BreakerOpen` when Open.
    ///
    /// `stream_guards` — when `Some`, the value is threaded into the SSE
    /// pump task and held for the entire body lifetime (headers → last byte
    /// / client disconnect).  The proxy does not inspect the boxed value; it
    /// relies entirely on `Drop` semantics, so callers typically pack
    /// `(LoadGuard, ActiveLoadGuard)` here. This keeps both the per-worker
    /// `active_requests` counter and the per-request active-load entry alive
    /// for the full streaming lifetime — without which a long-running SSE
    /// response would under-report load.
    ///
    /// `abort_rid` — when `Some`, the request id the router injected into the
    /// forwarded body. On a successful stream this arms a client-disconnect
    /// abort: if the pump reports anything other than the engine reaching its
    /// own clean end, the engine is told to stop generating this rid. `None`
    /// disables it (callers that don't track a rid).
    // Each parameter is a distinct, required input to a single upstream
    // forward (target, breaker, path, headers, body, plus the
    // streaming-lifetime callbacks and the abort rid). Bundling them into a
    // struct purely to satisfy the arg-count heuristic would add indirection
    // without clarity.
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
        on_stream_end: Option<Box<dyn FnOnce(sse::StreamEnd) + Send + 'static>>,
        abort_rid: Option<&str>,
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
        // resolved rather than wedged (see `breaker_outcome` /
        // `record_backpressure`).
        let caller_end_hook = if status.is_success() {
            on_stream_end
        } else {
            None
        };
        // Client-disconnect abort, armed off the same completion report. A
        // non-2xx stream is the engine's own error body — it isn't generating,
        // so it is never abortable.
        let abort_on_end = match abort_rid {
            Some(rid) if status.is_success() => worker_url.join("/abort_request").ok().map(|url| {
                (
                    self.client.clone(),
                    url.to_string(),
                    rid.to_string(),
                    abort_auth(headers),
                )
            }),
            _ => None,
        };
        let on_complete: Option<Box<dyn FnOnce(sse::StreamEnd) + Send + 'static>> =
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
                    Some(Box::new(move |end| {
                        if end.transport_ok {
                            breaker_for_hook.record_success();
                        } else {
                            breaker_for_hook.record_failure();
                        }
                        if engine_may_still_be_generating(end) {
                            if let Some((client, abort_url, rid, auth)) = abort_on_end {
                                // The hook runs inside the pump task, so a
                                // runtime is always present; the POST is
                                // fire-and-forget so the pump task can finish
                                // now.
                                tokio::spawn(async move {
                                    send_abort(&client, &abort_url, &rid, auth.as_ref()).await
                                });
                            }
                        }
                        if let Some(hook) = caller_end_hook {
                            hook(end);
                        }
                    }))
                }
            };
        // Only record TTFT for successful streams; error-body chunks are not
        // generated tokens.
        let first_byte_hook = if status.is_success() {
            on_first_byte
        } else {
            None
        };
        let body = sse::bytes_stream_to_body(
            resp.bytes_stream(),
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
    use crate::health::circuit_breaker::CircuitBreakerConfig;
    use axum::extract::State;
    use axum::routing::post;
    use axum::{Json, Router};
    use futures::StreamExt;
    use reqwest::StatusCode;
    use serde_json::Value;
    use std::num::NonZeroU32;
    use std::sync::Mutex;
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

    // ---- AbortOnDrop / send_abort -----------------------------------------

    /// Every `/abort_request` POST appends its parsed JSON body here, so tests
    /// can assert on `rid` / `abort_all` without racing a single "last body"
    /// slot.
    type AbortLog = Arc<Mutex<Vec<Value>>>;

    async fn abort_request_handler(
        State(log): State<AbortLog>,
        Json(body): Json<Value>,
    ) -> StatusCode {
        log.lock().unwrap().push(body);
        StatusCode::OK
    }

    async fn failing_abort_handler(
        State(log): State<AbortLog>,
        Json(body): Json<Value>,
    ) -> StatusCode {
        log.lock().unwrap().push(body);
        StatusCode::INTERNAL_SERVER_ERROR
    }

    /// Serve `app` on an ephemeral port; the returned sender shuts it down.
    async fn serve(app: Router) -> (String, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
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

    /// A saturated engine's own queue-full 503s must not trip the router's
    /// circuit breaker. Dispatch far past any plausible failure threshold and
    /// assert the breaker stays Closed and admitting.
    #[tokio::test]
    async fn engine_503_does_not_trip_breaker() {
        let (url, _abort_log, _shutdown) = spawn_status_worker(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = CircuitBreaker::new();
        let headers = HeaderMap::new();

        for i in 0..50 {
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

    /// Contrast guard, so the backpressure carve-out cannot disable fault
    /// detection: a genuine 5xx fault (500) MUST still open the breaker. Loops
    /// on `would_allow()` rather than a fixed count so the test stays correct if
    /// the default `CircuitBreakerConfig` threshold changes.
    #[tokio::test]
    async fn engine_500_still_trips_breaker() {
        let (url, _abort_log, _shutdown) = spawn_status_worker(500).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = CircuitBreaker::new();
        let headers = HeaderMap::new();

        for _ in 0..50 {
            if !breaker.would_allow() {
                break;
            }
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
            "a run of 500s must open the breaker (fault detection still works)",
        );
    }

    /// End-to-end wedge guard: a breaker that opened on real faults, then has
    /// its half-open probe answered with a 503, must RECOVER — not stay shut
    /// out forever. Exercises the `Neutral => record_backpressure` wiring in
    /// `forward_json_to` through the half-open path.
    #[tokio::test]
    async fn engine_503_recovers_a_half_open_breaker() {
        let (url, _abort_log, _shutdown) = spawn_status_worker(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        // threshold=1 so one prior fault opens it; a short cooldown so the probe
        // is admitted quickly. The wait below is an order of magnitude longer
        // than the cooldown rather than a thin margin, since this test needs a
        // real socket and so cannot pause the clock.
        let breaker = CircuitBreaker::with_config(CircuitBreakerConfig {
            threshold: NonZeroU32::new(1).unwrap(),
            cool_down: Duration::from_millis(20),
        });
        let headers = HeaderMap::new();

        // Simulate a prior genuine fault (e.g. a 500 / timeout) that tripped it.
        breaker.record_failure();
        assert_eq!(breaker.snapshot().state_code, 1, "breaker should be Open");

        // Let the cooldown elapse so the next dispatch claims the half-open probe.
        tokio::time::sleep(Duration::from_millis(200)).await;

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

    /// Streaming path parity: the engine's 503 on the streaming arm must also
    /// leave the breaker untouched (no up-front failure, no completion hook).
    #[tokio::test]
    async fn engine_503_does_not_trip_breaker_streaming() {
        use http_body_util::BodyExt;

        let (url, _abort_log, _shutdown) = spawn_status_worker(503).await;
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

    /// An SSE body that emits `chunks` with `delay` between each, like a real
    /// engine generating tokens.
    async fn stream_chat(chunks: Vec<&'static str>, delay: Duration) -> Response<Body> {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        tokio::spawn(async move {
            for c in chunks {
                tokio::time::sleep(delay).await;
                if tx.send(Ok(Bytes::from(c))).await.is_err() {
                    break;
                }
            }
        });
        let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
        let mut r = Response::new(body);
        *r.status_mut() = StatusCode::OK;
        r.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("text/event-stream"),
        );
        r
    }

    /// A fake upstream whose `/v1/chat/completions` sleeps `delay` before
    /// answering 200 (an engine still generating a unary response), and whose
    /// `/abort_request` records every POSTed body.
    async fn spawn_hanging_worker(delay: Duration) -> (String, AbortLog, oneshot::Sender<()>) {
        let log: AbortLog = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move || async move {
                    tokio::time::sleep(delay).await;
                    (StatusCode::OK, "{}")
                }),
            )
            .route("/abort_request", post(abort_request_handler))
            .with_state(log.clone());
        let (url, tx) = serve(app).await;
        (url, log, tx)
    }

    /// A fake upstream that streams `chunks` and records aborts. `abort_status`
    /// picks whether `/abort_request` answers 200 or 500 — the 500 variant
    /// proves a failing abort never reaches the circuit breaker.
    async fn spawn_streaming_worker(
        chunks: Vec<&'static str>,
        delay: Duration,
        abort_ok: bool,
    ) -> (String, AbortLog, oneshot::Sender<()>) {
        let log: AbortLog = Arc::new(Mutex::new(Vec::new()));
        let abort_route = if abort_ok {
            post(abort_request_handler)
        } else {
            post(failing_abort_handler)
        };
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move || stream_chat(chunks.clone(), delay)),
            )
            .route("/abort_request", abort_route)
            .with_state(log.clone());
        let (url, tx) = serve(app).await;
        (url, log, tx)
    }

    /// A fake upstream answering a fixed non-2xx status, recording aborts — so
    /// a test can assert a non-2xx response never triggers one, rather than
    /// just hoping a stray POST went nowhere.
    async fn spawn_status_worker(status: u16) -> (String, AbortLog, oneshot::Sender<()>) {
        let log: AbortLog = Arc::new(Mutex::new(Vec::new()));
        let code = StatusCode::from_u16(status).unwrap();
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move || async move { (code, "{\"error\":\"x\"}") }),
            )
            .route("/abort_request", post(abort_request_handler))
            .with_state(log.clone());
        let (url, tx) = serve(app).await;
        (url, log, tx)
    }

    /// A fake upstream that records the `Authorization` header of every
    /// `/abort_request` POST (empty string when absent), so a test can prove
    /// the abort carries the same credential the generate request did.
    /// Also serves a slow SSE `/v1/chat/completions`, so the same worker can
    /// exercise the streaming guard's auth path as well as the unary one.
    async fn spawn_auth_recording_worker() -> (String, Arc<Mutex<Vec<String>>>, oneshot::Sender<()>)
    {
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let state = Arc::clone(&seen);
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move || {
                    stream_chat(
                        vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
                        Duration::from_millis(50),
                    )
                }),
            )
            .route(
                "/abort_request",
                post(move |headers: HeaderMap| async move {
                    state.lock().unwrap().push(
                        headers
                            .get("authorization")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("")
                            .to_string(),
                    );
                    StatusCode::OK
                }),
            );
        let (url, tx) = serve(app).await;
        (url, seen, tx)
    }

    /// Poll `seen` until it holds an entry or `timeout` elapses.
    async fn wait_for_auth(seen: &Arc<Mutex<Vec<String>>>, timeout: Duration) {
        let deadline = std::time::Instant::now() + timeout;
        while seen.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    fn bearer(token: &'static str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_static(token),
        );
        headers
    }

    /// SGLang marks `/abort_request` `ADMIN_OPTIONAL`, so an engine started
    /// with `--api-key` 401s an unauthenticated abort and keeps generating.
    /// The guard must replay the request's own `Authorization` — the same
    /// credential the worker already accepted on the generate call.
    #[tokio::test]
    async fn abort_replays_the_request_authorization_header() {
        let (url, seen, _shutdown) = spawn_auth_recording_worker().await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        {
            let _guard = proxy
                .abort_guard_for(&url, "rid-authed", &bearer("Bearer sk-test"))
                .expect("a well-formed worker URL must yield a guard");
        }
        wait_for_auth(&seen, Duration::from_secs(2)).await;
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            ["Bearer sk-test"],
            "the abort must carry the request's Authorization, or an api-key \
             engine silently rejects it and keeps generating",
        );
    }

    /// The streaming guard builds its own `AbortOnDrop` inside
    /// `forward_streaming_to`, on a separate code path from `abort_guard_for` —
    /// so it needs its own proof that the credential is replayed.
    #[tokio::test]
    async fn streaming_abort_replays_the_request_authorization_header() {
        let (url, seen, _shutdown) = spawn_auth_recording_worker().await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let resp = proxy
            .forward_streaming_to(
                &url,
                &breaker,
                "/v1/chat/completions",
                &bearer("Bearer sk-stream"),
                Bytes::from_static(b"{}"),
                None,
                None,
                None,
                Some("rid-stream-authed"),
            )
            .await
            .expect("streaming dispatch should reach the worker");

        let mut data_stream = resp.into_body().into_data_stream();
        assert!(data_stream.next().await.is_some());
        drop(data_stream);

        wait_for_auth(&seen, Duration::from_secs(2)).await;
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            ["Bearer sk-stream"],
            "the streaming disconnect abort must carry the request's Authorization",
        );
    }

    /// Poll `log` until it holds at least `count` entries or `timeout` elapses.
    async fn wait_for_aborts(log: &AbortLog, count: usize, timeout: Duration) {
        let deadline = std::time::Instant::now() + timeout;
        while log.lock().unwrap().len() < count && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Unary guard: dropped while still armed (no `disarm()` call) must POST
    /// exactly one abort — the handler-cancelled-by-client-disconnect case.
    #[tokio::test]
    async fn abort_on_drop_unary_fires_when_dropped_armed() {
        let (url, abort_log, _shutdown) = spawn_hanging_worker(Duration::from_secs(10)).await;
        {
            let _guard = AbortOnDrop::new(
                Client::new(),
                format!("{url}/abort_request"),
                "test-rid-1".into(),
                None,
            );
        }
        wait_for_aborts(&abort_log, 1, Duration::from_secs(2)).await;
        let log = abort_log.lock().unwrap();
        assert_eq!(
            log.len(),
            1,
            "an armed guard dropped without disarm must POST exactly one abort"
        );
        assert_eq!(log[0]["rid"], "test-rid-1");
        assert_eq!(log[0]["abort_all"], false);
    }

    /// Unary guard: `disarm()` before drop (a complete response was received)
    /// must suppress the abort entirely.
    #[tokio::test]
    async fn abort_on_drop_unary_does_not_fire_when_disarmed() {
        let (url, abort_log, _shutdown) = spawn_hanging_worker(Duration::from_secs(10)).await;
        {
            let mut guard = AbortOnDrop::new(
                Client::new(),
                format!("{url}/abort_request"),
                "test-rid-2".into(),
                None,
            );
            guard.disarm();
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            abort_log.lock().unwrap().is_empty(),
            "a disarmed guard must never abort"
        );
    }

    /// The streaming arm's decision table. Only a stream drained to its clean
    /// end with the client still attached proves the engine stopped on its own;
    /// every other shape leaves it possibly still generating for nobody.
    #[test]
    fn engine_may_still_be_generating_only_spares_a_clean_attached_end() {
        let clean = sse::StreamEnd {
            transport_ok: true,
            saw_error_event: false,
            client_disconnect: false,
        };
        assert!(
            !engine_may_still_be_generating(clean),
            "a clean end with the client attached means the engine finished"
        );
        assert!(
            !engine_may_still_be_generating(sse::StreamEnd {
                saw_error_event: true,
                ..clean
            }),
            "an engine that reported its own error then closed cleanly is done — \
             aborting it would be a spurious abort on every failed generation"
        );
        assert!(
            engine_may_still_be_generating(sse::StreamEnd {
                client_disconnect: true,
                ..clean
            }),
            "the client went away mid-generation"
        );
        assert!(
            engine_may_still_be_generating(sse::StreamEnd {
                transport_ok: false,
                ..clean
            }),
            "losing the router-to-engine connection says nothing about the engine"
        );
    }

    /// `send_abort` posts `{rid, abort_all:false}` to the given URL.
    #[tokio::test]
    async fn send_abort_posts_rid_and_abort_all_false() {
        let (url, abort_log, _shutdown) = spawn_hanging_worker(Duration::from_secs(10)).await;
        send_abort(
            &Client::new(),
            &format!("{url}/abort_request"),
            "direct-rid",
            None,
        )
        .await;
        let log = abort_log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0]["rid"], "direct-rid");
        assert_eq!(log[0]["abort_all"], false);
    }

    /// `send_abort` is best-effort: an unreachable abort URL (connection
    /// refused) must be swallowed — logged, not panicked or propagated.
    #[tokio::test]
    async fn send_abort_swallows_unreachable_url_without_panicking() {
        // Port 1 is privileged / never listened on in CI sandboxes — refused
        // promptly. No assertion beyond "this does not panic or hang".
        send_abort(
            &Client::new(),
            "http://127.0.0.1:1/abort_request",
            "rid-x",
            None,
        )
        .await;
    }

    /// `abort_guard_for` returns `None` for a worker URL that cannot be parsed
    /// — matching the moot forward failure (the dispatch itself would fail the
    /// same way, so there is nothing to abort).
    #[tokio::test]
    async fn abort_guard_for_returns_none_for_unparsable_url() {
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        assert!(
            proxy
                .abort_guard_for("not a valid url", "rid", &HeaderMap::new())
                .is_none(),
            "an unparsable worker URL must yield no guard"
        );
    }

    /// `abort_guard_for` joins `worker_url` + `/abort_request` correctly: a
    /// guard built through it (not constructed directly) must still reach the
    /// right endpoint when dropped armed.
    #[tokio::test]
    async fn abort_guard_for_builds_a_working_guard() {
        let (url, abort_log, _shutdown) = spawn_hanging_worker(Duration::from_secs(10)).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        {
            let _guard = proxy
                .abort_guard_for(&url, "rid-via-proxy", &HeaderMap::new())
                .expect("a well-formed worker URL must yield a guard");
        }
        wait_for_aborts(&abort_log, 1, Duration::from_secs(2)).await;
        let log = abort_log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0]["rid"], "rid-via-proxy");
    }

    // ---- forward_streaming_to abort wiring --------------------------------

    /// End-to-end through `forward_streaming_to`: a client that disconnects
    /// before the upstream stream reaches its terminal item must trigger an
    /// abort POST carrying the `abort_rid` the caller supplied.
    #[tokio::test]
    async fn forward_streaming_to_aborts_on_client_disconnect() {
        let (url, abort_log, _shutdown) = spawn_streaming_worker(
            vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
            Duration::from_millis(50),
            true,
        )
        .await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let resp = proxy
            .forward_streaming_to(
                &url,
                &breaker,
                "/v1/chat/completions",
                &HeaderMap::new(),
                Bytes::from_static(b"{}"),
                None,
                None,
                None,
                Some("stream-rid-1"),
            )
            .await
            .expect("streaming dispatch should reach the worker");

        // Read exactly one chunk, then drop the body — simulating the client
        // going away before the engine finishes streaming.
        let mut data_stream = resp.into_body().into_data_stream();
        assert!(
            data_stream.next().await.is_some(),
            "expected at least one chunk before drop"
        );
        drop(data_stream);

        wait_for_aborts(&abort_log, 1, Duration::from_secs(2)).await;
        let log = abort_log.lock().unwrap();
        assert_eq!(
            log.len(),
            1,
            "client disconnect mid-stream must trigger exactly one abort"
        );
        assert_eq!(log[0]["rid"], "stream-rid-1");
        assert_eq!(log[0]["abort_all"], false);
    }

    /// Contrast: a stream drained to its normal completion must NEVER abort —
    /// the engine finished on its own, so telling it to stop would be a
    /// spurious abort that pollutes engine abort metrics.
    #[tokio::test]
    async fn forward_streaming_to_does_not_abort_on_normal_completion() {
        use http_body_util::BodyExt;

        let (url, abort_log, _shutdown) = spawn_streaming_worker(
            vec!["data: a\n\n", "data: b\n\n"],
            Duration::from_millis(10),
            true,
        )
        .await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let resp = proxy
            .forward_streaming_to(
                &url,
                &breaker,
                "/v1/chat/completions",
                &HeaderMap::new(),
                Bytes::from_static(b"{}"),
                None,
                None,
                None,
                Some("stream-rid-2"),
            )
            .await
            .expect("streaming dispatch should reach the worker");

        let _ = resp.into_body().collect().await;
        // Give the pump's completion hook a moment to run.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            abort_log.lock().unwrap().is_empty(),
            "a stream drained to completion must never trigger an abort"
        );
    }

    /// A non-2xx upstream response is the engine's own error body (it is not
    /// generating), so it must never be abortable even if the client
    /// disconnects while reading it.
    #[tokio::test]
    async fn forward_streaming_to_does_not_abort_non_2xx_response() {
        let (url, abort_log, _shutdown) = spawn_status_worker(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let resp = proxy
            .forward_streaming_to(
                &url,
                &breaker,
                "/v1/chat/completions",
                &HeaderMap::new(),
                Bytes::from_static(b"{}"),
                None,
                None,
                None,
                Some("stream-rid-3"),
            )
            .await
            .expect("streaming dispatch should reach the worker");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Drop the body without draining it — the same "client disconnect"
        // shape as the positive abort test — to prove the `status.is_success()`
        // gate, and not merely an absent disconnect, is what suppresses the
        // abort here.
        let mut data_stream = resp.into_body().into_data_stream();
        let _ = data_stream.next().await;
        drop(data_stream);

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            abort_log.lock().unwrap().is_empty(),
            "a non-2xx response must never trigger an abort, even on client disconnect"
        );
    }

    /// `send_abort` must never affect the worker's circuit breaker — an abort
    /// is a courtesy to the engine, never counted against a worker (see
    /// `send_abort`'s doc comment). `send_abort` has no breaker parameter at
    /// all today, so this cannot currently fail; it is a regression guard
    /// against a future refactor that routes the abort POST through a
    /// breaker-gated path.
    ///
    /// Threshold is 1 so a leaked `record_failure()` would open the breaker off
    /// a single failure. Note this is NOT a race-proof guard: each iteration's
    /// own chat request succeeds and disconnects from a healthy stream, which
    /// records a breaker SUCCESS synchronously in the pump's completion hook —
    /// the moment `AbortOnDrop` drops and spawns `send_abort`. A regressed
    /// `send_abort` recording a failure would do so only after a network
    /// round-trip, so the synchronous success is likely to win that race. Treat
    /// this as a behavioral pin; the API shape is what carries the guarantee.
    #[tokio::test]
    async fn send_abort_failure_does_not_trip_circuit_breaker() {
        let (url, abort_log, _shutdown) = spawn_streaming_worker(
            vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
            Duration::from_millis(50),
            false,
        )
        .await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
            threshold: NonZeroU32::new(1).unwrap(),
            cool_down: Duration::from_secs(30),
        }));

        for i in 0..3 {
            let resp = proxy
                .forward_streaming_to(
                    &url,
                    &breaker,
                    "/v1/chat/completions",
                    &HeaderMap::new(),
                    Bytes::from_static(b"{}"),
                    None,
                    None,
                    None,
                    Some(&format!("breaker-test-rid-{i}")),
                )
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "iter {i}: dispatch must reach the worker (breaker must stay closed): {e}"
                    )
                });
            assert_eq!(resp.status(), StatusCode::OK, "iter {i}");

            let mut data_stream = resp.into_body().into_data_stream();
            assert!(
                data_stream.next().await.is_some(),
                "iter {i}: expected at least one chunk before drop"
            );
            drop(data_stream);
        }

        wait_for_aborts(&abort_log, 3, Duration::from_secs(2)).await;
        assert_eq!(
            abort_log.lock().unwrap().len(),
            3,
            "all 3 disconnects must have attempted an abort, even though each one 500s"
        );
        assert!(
            breaker.would_allow(),
            "3 failed (500) /abort_request POSTs must NOT trip the breaker"
        );
    }
}
