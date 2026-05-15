// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP proxy — forwards requests to the upstream SGLang worker.

pub mod sse;

use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use bytes::Bytes;
use reqwest::Client;

pub struct Proxy {
    pub worker_url: String,
    pub client: Client,
}

impl Proxy {
    pub fn new(worker_url: String) -> Self {
        let client = Client::builder()
            .pool_max_idle_per_host(64)
            .build()
            .expect("reqwest::Client::builder");
        Self { worker_url, client }
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
        let url = format!("{}{}", self.worker_url, path);
        let mut req = self.client.post(&url).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req.header("content-type", "application/json");
        let resp = req
            .send()
            .await
            .map_err(|e| ApiError::UpstreamWorker(format!("{e}")))?;
        let status =
            StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ApiError::UpstreamWorker(format!("read body: {e}")))?;
        let mut out = Response::new(Body::from(bytes));
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    /// One-shot health check against the configured worker. Returns `Ok(())`
    /// if the worker responded with any 2xx or 3xx within the timeout;
    /// `Err` otherwise. Result is informational — callers log and continue.
    pub async fn probe_health(&self, timeout: std::time::Duration) -> Result<(), String> {
        let url = format!("{}/health", self.worker_url);
        match tokio::time::timeout(timeout, self.client.get(&url).send()).await {
            Ok(Ok(resp)) if resp.status().is_success() || resp.status().is_redirection() => Ok(()),
            Ok(Ok(resp)) => Err(format!("worker {} returned status {}", url, resp.status())),
            Ok(Err(e)) => Err(format!("worker {} unreachable: {e}", url)),
            Err(_) => Err(format!(
                "worker {} probe timed out after {:?}",
                url, timeout
            )),
        }
    }

    /// Forward a JSON POST and stream the SSE response back unmodified.
    /// Adds `accept: text/event-stream` to the outbound request.
    pub async fn forward_streaming(
        &self,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        let url = format!("{}{}", self.worker_url, path);
        let mut req = self.client.post(&url).body(body);
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
            .map_err(|e| ApiError::UpstreamWorker(format!("{e}")))?;
        let status =
            StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        // Capture content-type BEFORE consuming resp via bytes_stream().
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

        let p = Proxy::new(format!("http://{addr}"));
        let res = p.probe_health(Duration::from_secs(2)).await;
        assert!(res.is_ok(), "expected probe to succeed, got: {res:?}");
        let _ = tx.send(());
    }

    #[tokio::test]
    async fn probe_health_fails_against_dead_server() {
        // Bind a port to get a free one, then drop it so connections are refused.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let p = Proxy::new(format!("http://{addr}"));
        let res = p.probe_health(Duration::from_millis(500)).await;
        let err = res.expect_err("expected probe to fail against a closed port");
        assert!(
            err.contains("unreachable") || err.contains("timed out") || err.contains("returned"),
            "unexpected error: {err}",
        );
    }
}
