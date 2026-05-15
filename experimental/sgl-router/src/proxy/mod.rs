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
