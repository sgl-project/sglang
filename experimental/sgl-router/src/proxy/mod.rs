// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP proxy — forwards requests to the upstream SGLang worker.

pub mod sse;

use crate::health::circuit_breaker::CircuitBreaker;
use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use crate::workers::WireProtocol;
use anyhow::Context;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use reqwest::{Client, Url};
use std::sync::Arc;
use std::sync::OnceLock;
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
    /// The single forwarding client, paired with the wire protocol it was
    /// built for. Stored as one cell so the protocol the proxy reports and
    /// the protocol the live client actually speaks can never drift: both
    /// are read from this tuple. The fleet is homogeneous — every worker
    /// speaks the same protocol — so one client suffices and a per-request
    /// selector is unnecessary. Installed by [`Self::set_protocol`] when the
    /// manager introspects a worker; if a forward somehow precedes
    /// resolution, [`Self::client`] falls back to the always-safe HTTP/1.1
    /// default.
    client: OnceLock<(WireProtocol, Client)>,
    /// Wall-clock timeout applied to non-streaming upstream requests. Streaming
    /// requests deliberately do not use this (long generations are valid).
    pub request_timeout: Duration,
}

/// Build a forwarding client for `protocol`, sharing pool/connect tuning
/// across protocols. The h2c variant pins HTTP/2 prior knowledge so it
/// speaks cleartext h2c to Granian engines (no ALPN on plaintext); the
/// HTTP/1.1 variant is the reqwest default and is safe against any engine.
fn build_client(protocol: WireProtocol) -> Result<Client, anyhow::Error> {
    let builder = Client::builder()
        .pool_max_idle_per_host(64)
        .connect_timeout(Duration::from_secs(5));
    match protocol {
        WireProtocol::Http1 => builder,
        WireProtocol::H2c => builder.http2_prior_knowledge(),
    }
    .build()
    .context("build reqwest client")
}

impl Proxy {
    /// Build a proxy. `request_timeout` is the per-request wall-clock budget for
    /// non-streaming forwards. Connect timeout is hard-coded to 5 s — even a
    /// streaming request fails fast at TCP setup if the worker is unreachable.
    ///
    /// The forwarding client is not built here: its wire protocol is resolved
    /// later from worker introspection (see [`Self::set_protocol`]) and the
    /// client is built then.
    pub fn new(request_timeout: Duration) -> Result<Self, anyhow::Error> {
        Ok(Self {
            client: OnceLock::new(),
            request_timeout,
        })
    }

    /// Install the fleet's forwarding client, built for the wire protocol the
    /// worker manager resolved from a worker's `/server_info`. The first call
    /// wins: the fleet is assumed homogeneous, so later workers report the
    /// same protocol and re-setting is a no-op. A later worker reporting a
    /// *different* protocol is logged (it cannot be honored — there is one
    /// client) but otherwise ignored. A client-construction failure is logged
    /// here, at registration time, rather than panicking on the first forward.
    ///
    /// The manager calls this before the worker becomes routable, so the
    /// client is in place before [`Self::client`] is first read.
    pub fn set_protocol(&self, protocol: WireProtocol) {
        if let Some((current, _)) = self.client.get() {
            // Already resolved (the common case: every subsequent worker
            // re-reports the fleet protocol). Only a genuine divergence is
            // worth a log.
            if *current != protocol {
                tracing::warn!(
                    fleet_protocol = ?current,
                    reported = ?protocol,
                    "worker reported a wire protocol that differs from the fleet's resolved \
                     protocol; the router uses a single forwarding client and assumes a \
                     homogeneous fleet, so the reported protocol is ignored",
                );
            }
            return;
        }
        match build_client(protocol) {
            // `set` can still race a concurrent first registration; the loser's
            // freshly built client is simply dropped (first-write-wins).
            Ok(client) => {
                let _ = self.client.set((protocol, client));
            }
            Err(e) => tracing::error!(
                error = %e,
                ?protocol,
                "failed to build the forwarding client; will fall back to the HTTP/1.1 default",
            ),
        }
    }

    /// The fleet's resolved wire protocol, or the HTTP/1.1 default if no worker
    /// has been introspected yet. Read from the same cell as [`Self::client`],
    /// so it always reflects the protocol the live client actually speaks.
    pub fn protocol(&self) -> WireProtocol {
        self.client.get().map(|(p, _)| *p).unwrap_or_default()
    }

    /// The single forwarding client (built for the resolved fleet protocol, or
    /// the always-safe HTTP/1.1 default if none was resolved). Also used for
    /// side-channel admin traffic (e.g. `/flush_cache`): under the homogeneous
    /// fleet assumption every engine accepts the resolved protocol, so one
    /// client serves both the request hot path and admin traffic.
    pub fn client(&self) -> &Client {
        &self
            .client
            .get_or_init(|| {
                (
                    WireProtocol::Http1,
                    build_client(WireProtocol::Http1)
                        .expect("HTTP/1.1 reqwest client construction is infallible"),
                )
            })
            .1
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
        let mut req = self.client().post(url.clone()).body(body);
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
        let mut req = self.client().post(url.clone()).body(body);
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

    #[tokio::test]
    async fn protocol_defaults_to_http1_until_resolved() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        assert_eq!(p.protocol(), WireProtocol::Http1);
    }

    #[tokio::test]
    async fn set_protocol_is_first_write_wins() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        p.set_protocol(WireProtocol::H2c);
        assert_eq!(p.protocol(), WireProtocol::H2c);
        // Homogeneous-fleet assumption: a later, differing report is ignored
        // (the single client is already committed to the first protocol). The
        // h2c-on-the-wire behavior itself is covered by tests/proxy/h2c_forward.rs.
        p.set_protocol(WireProtocol::Http1);
        assert_eq!(p.protocol(), WireProtocol::H2c);
    }

    /// `protocol()` and `client()` read the same cell, so the reported
    /// protocol can never drift from the client actually in use. In
    /// particular, once a forward (or `/flush_cache`) builds the client via
    /// `client()` ahead of resolution — which locks in the HTTP/1.1 default —
    /// a later `set_protocol(H2c)` is a no-op AND `protocol()` keeps reporting
    /// the truth (Http1), rather than claiming H2c while the live client
    /// speaks HTTP/1.1. The manager orders `set_protocol` before a worker is
    /// routable so this fallback never engages in production; this test pins
    /// the invariant that the two can't disagree if it ever did.
    #[tokio::test]
    async fn client_built_before_resolution_keeps_protocol_consistent() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        // Force the client to build before any protocol is resolved.
        let _ = p.client();
        assert_eq!(p.protocol(), WireProtocol::Http1);
        // A late resolution cannot rebuild the committed client, and crucially
        // cannot make protocol() lie about it.
        p.set_protocol(WireProtocol::H2c);
        assert_eq!(p.protocol(), WireProtocol::Http1);
    }
}
