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
use std::time::Duration;

#[derive(Debug)]
pub struct Proxy {
    /// Pre-parsed worker URL. Downstream paths are built via [`Url::join`]
    /// (see [`Self::worker_path`]), which handles trailing slashes per the
    /// URL spec — `http://x:30000/` joined with `/v1/tokenize` yields
    /// `http://x:30000/v1/tokenize`, no double slash.
    pub worker_url: Url,
    pub client: Client,
    /// Wall-clock timeout applied to non-streaming upstream requests. Streaming
    /// requests deliberately do not use this (long generations are valid).
    pub request_timeout: Duration,
}

impl Proxy {
    /// Build a proxy. `request_timeout` is the per-request wall-clock budget for
    /// non-streaming forwards. Connect timeout is hard-coded to 5 s — even a
    /// streaming request fails fast at TCP setup if the worker is unreachable.
    pub fn new(worker_url: Url, request_timeout: Duration) -> Result<Self, anyhow::Error> {
        let client = Client::builder()
            .pool_max_idle_per_host(64)
            .connect_timeout(Duration::from_secs(5))
            .build()
            .context("build reqwest client")?;
        Ok(Self {
            worker_url,
            client,
            request_timeout,
        })
    }

    /// Build a full upstream URL for an absolute path (e.g. `/v1/tokenize`).
    /// Uses [`Url::join`] semantics: an absolute path replaces the base
    /// path, so a base URL with or without a trailing `/` produces the same
    /// result. Errors only on malformed `path` inputs; we surface them as
    /// `ApiError::Internal` since callers pass static path literals.
    fn worker_path(&self, path: &str) -> Result<Url, ApiError> {
        self.worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })
    }

    /// Forward a JSON POST to the worker and return the buffered response.
    /// The worker's status code is preserved; content-type is set to
    /// application/json.
    pub async fn forward_json(
        &self,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        let url = self.worker_path(path)?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .timeout(self.request_timeout);
        let resp = req
            .send()
            .await
            .map_err(|e| self.classify_reqwest_error(e, path))?;
        let status = resp.status();
        // Parity with forward_streaming: log non-2xx upstream responses so a
        // 5xx spike on non-streaming requests is visible server-side, not
        // streaming-only.
        if !status.is_success() {
            tracing::warn!(
                upstream = %url,
                path = path,
                status = %status,
                "upstream returned non-2xx on non-streaming request",
            );
        }
        // Headers + status were already received; a mid-body read failure
        // here is NOT "unreachable" (the upstream demonstrably replied). We
        // surface it as `UpstreamStatus { status }` with a server-side log
        // so operators can see the worker started a response and dropped.
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = ?e,
                    "upstream dropped connection mid-body",
                );
                return Err(ApiError::UpstreamStatus { status });
            }
        };
        let mut out = Response::new(Body::from(bytes));
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    /// Classify a reqwest error into the right `ApiError` variant.
    ///
    /// Walks the full source chain to detect timeouts, because reqwest wraps
    /// hyper which wraps `std::io::Error` — a top-level `is_timeout()` check
    /// misses both the wrapped reqwest timeout and the `io::ErrorKind::TimedOut`
    /// cases.
    fn classify_reqwest_error(&self, e: reqwest::Error, path: &str) -> ApiError {
        Self::classify_reqwest_error_for(self.worker_url.clone(), e, path)
    }

    /// Classify a reqwest error into the right `ApiError` variant, given an
    /// explicit worker URL. Used by both the legacy methods (which thread
    /// `self.worker_url`) and the M2 breaker-gated `_to` methods (which take
    /// per-request worker URLs).
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

    /// One-shot health check against the configured worker. Returns `Ok(())`
    /// if the worker responded with any 2xx or 3xx within the timeout;
    /// `Err` otherwise. Result is informational — callers log and continue.
    pub async fn probe_health(&self, timeout: Duration) -> Result<(), String> {
        let url = self
            .worker_url
            .join("/health")
            .map_err(|e| format!("worker {}: join /health failed: {e}", self.worker_url))?;
        match tokio::time::timeout(timeout, self.client.get(url.clone()).send()).await {
            Ok(Ok(resp)) if resp.status().is_success() || resp.status().is_redirection() => Ok(()),
            Ok(Ok(resp)) => Err(format!("worker {} returned status {}", url, resp.status())),
            Ok(Err(e)) => Err(format!("worker {} unreachable: {e}", url)),
            Err(_) => Err(format!(
                "worker {} probe timed out after {:?}",
                url, timeout
            )),
        }
    }

    /// Breaker-gated JSON POST: checks `breaker.allow()` first, records
    /// success/failure based on response status, and returns
    /// `ApiError::ServiceUnavailable` immediately when the breaker is Open.
    ///
    /// `worker_url` is the typed `reqwest::Url` per M1's typed-URL contract;
    /// joining with `path` uses [`Url::join`] which handles trailing-slash
    /// normalization cleanly. Transport failures map to the split
    /// `UpstreamUnreachable` / `UpstreamTimeout` / `UpstreamStatus` variants
    /// from M1, NOT a flat `UpstreamWorker(String)`.
    pub async fn forward_json_to(
        &self,
        worker_url: &Url,
        breaker: &CircuitBreaker,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::ServiceUnavailable(
                "worker circuit breaker open".into(),
            ));
        }
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
        if status.is_server_error() {
            breaker.record_failure();
        } else {
            breaker.record_success();
        }
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = ?e,
                    "upstream dropped connection mid-body",
                );
                return Err(ApiError::UpstreamStatus { status });
            }
        };
        let mut out = Response::new(Body::from(bytes));
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    /// Breaker-gated streaming POST: checks `breaker.allow()` first, records
    /// success/failure, and returns `ApiError::ServiceUnavailable` when Open.
    pub async fn forward_streaming_to(
        &self,
        worker_url: &Url,
        breaker: &CircuitBreaker,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::ServiceUnavailable(
                "worker circuit breaker open".into(),
            ));
        }
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
        if status.is_server_error() {
            breaker.record_failure();
        } else {
            breaker.record_success();
        }
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
        let body = sse::bytes_stream_to_body(resp.bytes_stream());
        let mut out = Response::new(body);
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_str(&content_type)
                .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
        );
        Ok(out)
    }

    /// Forward a JSON POST and stream the SSE response back unmodified.
    /// Adds `accept: text/event-stream` to the outbound request.
    ///
    /// Note: streaming requests deliberately do NOT set a wall-clock
    /// `.timeout(...)` — long generations are valid and clients drive
    /// cancellation by dropping the response stream. The client-level
    /// `connect_timeout` configured in [`Self::new`] still applies, so a
    /// worker that never accepts TCP fails fast at request initiation.
    pub async fn forward_streaming(
        &self,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        let url = self.worker_path(path)?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");
        let resp = req
            .send()
            .await
            .map_err(|e| self.classify_reqwest_error(e, path))?;
        let status = resp.status();
        if !status.is_success() {
            tracing::warn!(
                upstream = %url,
                path = path,
                status = %status,
                "upstream returned non-2xx on streaming request",
            );
        }
        // Capture content-type BEFORE consuming resp via bytes_stream().
        // If the upstream has a content-type header but its value is not
        // ASCII-printable (non-conforming worker, mojibake, etc.), we fall
        // back to application/json — but we log the case so this isn't a
        // silent corruption. Conformant workers (axum, tonic, fastapi)
        // never hit this fallback.
        let upstream_ct = match resp.headers().get(reqwest::header::CONTENT_TYPE) {
            Some(v) => match v.to_str() {
                Ok(s) => s.to_string(),
                Err(_) => {
                    tracing::warn!(
                        upstream = %url,
                        "upstream content-type header is not ASCII-printable; \
                         defaulting to application/json",
                    );
                    "application/json".to_string()
                }
            },
            None => "application/json".to_string(),
        };
        let content_type = if status.is_success() {
            "text/event-stream".to_string()
        } else {
            upstream_ct
        };
        let body = sse::bytes_stream_to_body(resp.bytes_stream());
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
    async fn probe_health_succeeds_against_a_listening_server() {
        use axum::routing::get;
        use axum::Router;
        let app = Router::new().route("/health", get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await;
        });

        let p = Proxy::new(
            Url::parse(&format!("http://{addr}")).unwrap(),
            Duration::from_secs(5),
        )
        .unwrap();
        let res = p.probe_health(Duration::from_secs(2)).await;
        assert!(res.is_ok(), "expected probe to succeed, got: {res:?}");
        let _ = tx.send(());
    }

    #[tokio::test]
    async fn new_returns_result_not_panic() {
        // Smoke: Proxy::new returns Result<Self>; the happy path works.
        let p = Proxy::new(
            Url::parse("http://127.0.0.1:1").unwrap(),
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(p.worker_url.as_str(), "http://127.0.0.1:1/");
    }

    #[tokio::test]
    async fn probe_health_fails_against_dead_server() {
        // Bind a port to get a free one, then drop it so connections are refused.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let p = Proxy::new(
            Url::parse(&format!("http://{addr}")).unwrap(),
            Duration::from_secs(5),
        )
        .unwrap();
        let res = p.probe_health(Duration::from_millis(500)).await;
        let err = res.expect_err("expected probe to fail against a closed port");
        assert!(
            err.contains("unreachable") || err.contains("timed out") || err.contains("returned"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn worker_path_handles_trailing_slash_and_absolute_path() {
        // Same base, two forms (trailing or not), absolute path: same result.
        // This pins the "no double-slash" property end-to-end.
        let p_with = Proxy::new(
            Url::parse("http://x:30000/").unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        let p_without = Proxy::new(
            Url::parse("http://x:30000").unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            p_with.worker_path("/v1/tokenize").unwrap().as_str(),
            "http://x:30000/v1/tokenize"
        );
        assert_eq!(
            p_without.worker_path("/v1/tokenize").unwrap().as_str(),
            "http://x:30000/v1/tokenize"
        );
    }
}
