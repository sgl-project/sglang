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

#[derive(Debug)]
pub struct Proxy {
    pub client: Client,
    /// Wall-clock timeout applied to non-streaming upstream requests. Streaming
    /// requests deliberately do not use this (long generations are valid).
    pub request_timeout: Duration,
}

/// Which timeout — if any — a forward error's source chain describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimeoutKind {
    /// The KERNEL gave up on the socket: an `io::ErrorKind::TimedOut` that the
    /// OS itself raised (`ETIMEDOUT`). Not caused by any budget we set.
    Socket,
    /// One of OUR budgets elapsed — `.timeout(request_timeout)` or
    /// `connect_timeout`. Actionable by changing that budget.
    Budget,
    /// Not a timeout at all.
    None,
}

/// Classify the timeout in an error chain: OS-raised socket give-up vs. a
/// budget of ours running out.
///
/// Two traps make this fiddlier than it looks, and both are load-bearing:
///
/// 1. **Ask the `io` question first.** `reqwest::Error::is_timeout()` walks
///    the chain itself and answers `true` for BOTH reqwest's own
///    (crate-private) timeout marker and a wrapped `io::ErrorKind::TimedOut`,
///    so asking it first collapses the two cases unconditionally.
/// 2. **`io::ErrorKind::TimedOut` alone does NOT mean the kernel timed out.**
///    `connect_timeout` — our own budget — is implemented in hyper-util as
///    `io::Error::new(ErrorKind::TimedOut, tokio::time::error::Elapsed)`
///    (`src/client/legacy/connect/http.rs`), so a plain `kind()` test files
///    every connect-budget expiry under `Socket`. The discriminator is
///    `raw_os_error()`: a genuine `ETIMEDOUT` carries an OS errno, a
///    synthesized one wraps `Elapsed` and has none.
///
/// Unrecognized shapes fall through to `Budget`/`None` rather than `Socket`,
/// so an unexpected error can only degrade to the pre-split behaviour — it can
/// never invent a spurious "the network broke" verdict.
///
/// Pure and `&anyhow::Error`-shaped so both invariants can be pinned by unit
/// tests without provoking a real kernel `ETIMEDOUT`.
fn timeout_kind(source: &anyhow::Error) -> TimeoutKind {
    if source.chain().any(|c| {
        c.downcast_ref::<std::io::Error>().is_some_and(|io| {
            io.kind() == std::io::ErrorKind::TimedOut && io.raw_os_error().is_some()
        })
    }) {
        return TimeoutKind::Socket;
    }
    if source.chain().any(|c| {
        c.downcast_ref::<reqwest::Error>()
            .is_some_and(|r| r.is_timeout())
    }) {
        return TimeoutKind::Budget;
    }
    TimeoutKind::None
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
    /// Walks the full source chain, because reqwest wraps hyper which wraps
    /// `std::io::Error`, and splits the two timeouts that chain can carry:
    ///
    /// * `UpstreamTimeout` — OUR budget (`.timeout(request_timeout)` /
    ///   `connect_timeout`) expired. Actionable by changing the budget.
    /// * `UpstreamSocketTimeout` — the SOCKET gave up under us (kernel
    ///   `ETIMEDOUT` surfacing as `io::ErrorKind::TimedOut`). A broken network
    ///   path; no budget change helps.
    ///
    /// The split itself lives in [`timeout_kind`] — see there for the two
    /// non-obvious traps (chain-walk order, and why `io::ErrorKind::TimedOut`
    /// on its own does not mean the kernel timed out). Note that
    /// `reqwest::Error::is_timeout()` already answers `true` for a wrapped
    /// `io::TimedOut`, so an `is_timeout() || <io::TimedOut in chain>` test
    /// has an unreachable second arm and files every socket give-up under
    /// `upstream_timeout`.
    fn classify_reqwest_error_for(worker: Url, e: reqwest::Error, path: &str) -> ApiError {
        let source = anyhow::Error::new(e).context(format!("worker {worker}: post {path}"));
        match timeout_kind(&source) {
            // `source` moves into the variant so the concrete OS error survives
            // to `into_response`, which logs it once — same shape as the
            // `None` arm below. Logging here too would double every line of
            // the loudest failure mode this split exists to diagnose.
            TimeoutKind::Socket => ApiError::UpstreamSocketTimeout { worker, source },
            TimeoutKind::Budget => ApiError::UpstreamTimeout { worker },
            TimeoutKind::None => ApiError::UpstreamUnreachable { worker, source },
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
    /// `UpstreamSocketTimeout` / `UpstreamStatus`).
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
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = ?e,
                    "upstream dropped connection mid-body",
                );
                breaker.record_failure();
                return Err(ApiError::UpstreamStatus { status });
            }
        };
        if status.is_server_error() {
            breaker.record_failure();
        } else {
            breaker.record_success();
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
        // is recorded as a failure. For 5xx headers we record_failure
        // up front and skip the pump hook (the body we surface is the
        // error response — its stream completing is not a worker win).
        let on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>> =
            if status.is_server_error() {
                breaker.record_failure();
                None
            } else {
                let breaker_for_hook = Arc::clone(breaker);
                Some(Box::new(move |ok| {
                    if ok {
                        breaker_for_hook.record_success();
                    } else {
                        breaker_for_hook.record_failure();
                    }
                }))
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
    use std::time::Duration;

    #[tokio::test]
    async fn new_returns_result_not_panic() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        assert_eq!(p.request_timeout, Duration::from_secs(5));
    }

    /// Errno for a kernel `ETIMEDOUT`, which is what separates a real socket
    /// give-up from a timer we started ourselves. Linux 110, macOS/BSD 60;
    /// other platforms would fail the tests' precondition asserts loudly
    /// rather than pass vacuously.
    const ETIMEDOUT: i32 = if cfg!(target_os = "linux") { 110 } else { 60 };

    /// Build a REAL `reqwest::Error` whose source chain carries a kernel-shaped
    /// `ETIMEDOUT`, by failing the connector rather than waiting on a socket.
    ///
    /// A hand-built `anyhow` chain cannot test this: with no `reqwest::Error`
    /// in it, the reqwest branch matches nothing and BOTH branch orders return
    /// the same answer. Only a genuine reqwest error — whose `is_timeout()`
    /// answers `true` through its io arm — can discriminate the ordering.
    async fn reqwest_error_with_injected_io(kind_err: std::io::Error) -> reqwest::Error {
        use std::task::{Context, Poll};
        type BoxErr = Box<dyn std::error::Error + Send + Sync>;

        #[derive(Clone)]
        struct FailWith<S>(S, Arc<std::io::Error>);
        impl<S, Req> tower::Service<Req> for FailWith<S>
        where
            S: tower::Service<Req, Error = BoxErr>,
        {
            type Response = S::Response;
            type Error = S::Error;
            type Future = futures::future::Ready<Result<S::Response, S::Error>>;
            fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }
            fn call(&mut self, _: Req) -> Self::Future {
                let e = match self.1.raw_os_error() {
                    Some(code) => std::io::Error::from_raw_os_error(code),
                    None => std::io::Error::new(self.1.kind(), "synthesized, no errno"),
                };
                futures::future::ready(Err(Box::new(e) as BoxErr))
            }
        }

        let injected = Arc::new(kind_err);
        let client = Client::builder()
            .connector_layer(tower::layer::layer_fn(move |s| {
                FailWith(s, Arc::clone(&injected))
            }))
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        client
            .post("http://127.0.0.1:1/v1/chat/completions")
            .send()
            .await
            .expect_err("the injected connector always fails")
    }

    /// A kernel `ETIMEDOUT` must classify as `Socket` even though reqwest's own
    /// `is_timeout()` also answers `true` for it.
    ///
    /// THIS is the test that pins the branch ORDER. Reverse the two arms in
    /// `timeout_kind` and this fails with `Budget` — reproducing the original
    /// bug where every socket give-up was reported as `upstream_timeout`.
    #[tokio::test]
    async fn kernel_etimedout_classifies_socket_despite_reqwest_is_timeout() {
        let e = reqwest_error_with_injected_io(std::io::Error::from_raw_os_error(ETIMEDOUT)).await;
        // Precondition: this is exactly the shape reqwest collapses. If this
        // ever stops holding, the ordering trap is gone and so is this test's
        // reason to exist.
        assert!(
            e.is_timeout(),
            "precondition: reqwest must report an io-TimedOut chain as a timeout"
        );

        let worker = Url::parse("http://127.0.0.1:1/").unwrap();
        let source = anyhow::Error::new(e).context("worker http://127.0.0.1:1/: post /p");
        assert_eq!(
            timeout_kind(&source),
            TimeoutKind::Socket,
            "a kernel ETIMEDOUT under a reqwest::Error must be Socket, not Budget"
        );

        // And the classifier must actually construct the new variant — without
        // this, the whole split can be a silent no-op end to end.
        let e = reqwest_error_with_injected_io(std::io::Error::from_raw_os_error(ETIMEDOUT)).await;
        assert!(
            matches!(
                Proxy::classify_reqwest_error_for(worker, e, "/v1/chat/completions"),
                ApiError::UpstreamSocketTimeout { .. }
            ),
            "classify_reqwest_error_for must map a kernel ETIMEDOUT to UpstreamSocketTimeout"
        );
    }

    /// Our OWN `connect_timeout` must NOT be reported as a broken network path.
    ///
    /// hyper-util implements it as
    /// `io::Error::new(ErrorKind::TimedOut, tokio::time::error::Elapsed)`
    /// (`src/client/legacy/connect/http.rs`), i.e. `ErrorKind::TimedOut` with
    /// NO errno — so a `kind()`-only test files every connect-budget expiry
    /// under `Socket` and tells the operator "the network broke" when the fix
    /// is a bigger budget. Drop the `raw_os_error().is_some()` guard and this
    /// fails.
    #[tokio::test]
    async fn our_own_connect_timeout_shape_is_not_a_socket_timeout() {
        let e = reqwest_error_with_injected_io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "deadline has elapsed",
        ))
        .await;
        assert!(e.is_timeout(), "precondition: still a reqwest timeout");

        let source = anyhow::Error::new(e).context("worker http://127.0.0.1:1/: post /p");
        assert_eq!(
            timeout_kind(&source),
            TimeoutKind::Budget,
            "an errno-less TimedOut is one of OUR timers, not the kernel's"
        );

        // End-to-end: the Budget arm of the classifier's match must construct
        // `UpstreamTimeout` — pins the arm mapping, not just the predicate.
        let e = reqwest_error_with_injected_io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "deadline has elapsed",
        ))
        .await;
        let worker = Url::parse("http://127.0.0.1:1/").unwrap();
        assert!(
            matches!(
                Proxy::classify_reqwest_error_for(worker, e, "/v1/chat/completions"),
                ApiError::UpstreamTimeout { .. }
            ),
            "classify_reqwest_error_for must map an errno-less TimedOut to UpstreamTimeout"
        );
    }

    /// A genuine `.timeout(request_timeout)` expiry — reqwest's crate-private
    /// timeout marker with NO io error in the chain at all — must classify as
    /// `Budget`, end to end. This is the most common production timeout shape;
    /// a "simplification" of the second arm to require an io error would
    /// silently reclassify every request-budget expiry as `UpstreamUnreachable`.
    #[tokio::test]
    async fn request_budget_expiry_shape_is_a_budget_timeout() {
        use std::task::{Context, Poll};
        type BoxErr = Box<dyn std::error::Error + Send + Sync>;

        // A connector that never answers; reqwest's own request timeout is
        // what fires.
        #[derive(Clone)]
        struct PendForever<S>(S);
        impl<S, Req> tower::Service<Req> for PendForever<S>
        where
            S: tower::Service<Req, Error = BoxErr>,
        {
            type Response = S::Response;
            type Error = S::Error;
            type Future = futures::future::Pending<Result<S::Response, S::Error>>;
            fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }
            fn call(&mut self, _: Req) -> Self::Future {
                futures::future::pending()
            }
        }

        let client = Client::builder()
            .connector_layer(tower::layer::layer_fn(PendForever))
            .timeout(Duration::from_millis(50))
            .build()
            .unwrap();
        let e = client
            .post("http://127.0.0.1:1/v1/chat/completions")
            .send()
            .await
            .expect_err("the request budget must elapse under a pending connector");
        assert!(
            e.is_timeout(),
            "precondition: reqwest reports its own expiry"
        );

        let worker = Url::parse("http://127.0.0.1:1/").unwrap();
        let source = anyhow::Error::new(e).context("worker http://127.0.0.1:1/: post /p");
        assert_eq!(timeout_kind(&source), TimeoutKind::Budget);

        let e = client
            .post("http://127.0.0.1:1/v1/chat/completions")
            .send()
            .await
            .expect_err("the request budget must elapse under a pending connector");
        assert!(
            matches!(
                Proxy::classify_reqwest_error_for(worker, e, "/v1/chat/completions"),
                ApiError::UpstreamTimeout { .. }
            ),
            "classify_reqwest_error_for must map a request-budget expiry to UpstreamTimeout"
        );
    }

    /// A chain with no timeout at all stays `None` → `UpstreamUnreachable`.
    /// Guards against a socket check so loose it swallows connection-refused.
    #[test]
    fn timeout_kind_reports_none_for_a_non_timeout_chain() {
        let err = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "connection refused",
        ))
        .context("worker http://w:1/: post /v1/chat/completions");
        assert_eq!(timeout_kind(&err), TimeoutKind::None);

        // Non-io, non-reqwest chains are not timeouts either.
        assert_eq!(
            timeout_kind(&anyhow::anyhow!("some unrelated failure")),
            TimeoutKind::None
        );
    }

    /// End-to-end pin for the `None` arm: a non-timeout failure must come out
    /// as `UpstreamUnreachable` (and keeps its `source` chain).
    #[tokio::test]
    async fn non_timeout_failure_classifies_unreachable() {
        let e = reqwest_error_with_injected_io(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "connection refused",
        ))
        .await;
        assert!(!e.is_timeout(), "precondition: not any kind of timeout");

        let worker = Url::parse("http://127.0.0.1:1/").unwrap();
        assert!(
            matches!(
                Proxy::classify_reqwest_error_for(worker, e, "/v1/chat/completions"),
                ApiError::UpstreamUnreachable { .. }
            ),
            "classify_reqwest_error_for must map a non-timeout failure to UpstreamUnreachable"
        );
    }
}
