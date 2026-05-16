// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use thiserror::Error;

pub const X_ROUTER_ERROR_CODE: HeaderName = HeaderName::from_static("x-router-error-code");

// Body-size limits land in M6 with the tower_http::limit layer.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// Could not reach the upstream worker (connect refused, DNS, TLS, request
    /// build error). `source` captures the full anyhow chain for server-side
    /// logging; clients see a generic message.
    #[error("upstream unreachable: worker {worker}")]
    UpstreamUnreachable {
        worker: String,
        #[source]
        source: anyhow::Error,
    },

    /// Upstream worker replied with a non-2xx status code that the router
    /// itself surfaced (vs. forwarding the worker's body). Currently unused in
    /// the proxy path (we forward the worker's body verbatim), reserved for
    /// future use by structured retries / circuit-breakers.
    #[error("upstream returned status {status}")]
    UpstreamStatus { status: StatusCode },

    /// Wall-clock timeout exceeded while waiting for the upstream worker's
    /// response (per-request `request_timeout`).
    #[error("upstream timed out: worker {worker}")]
    UpstreamTimeout { worker: String },

    #[error("internal: {0}")]
    Internal(#[from] anyhow::Error),
}

impl ApiError {
    fn status_and_code(&self) -> (StatusCode, &'static str) {
        match self {
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            ApiError::ModelNotFound(_) => (StatusCode::NOT_FOUND, "model_not_found"),
            ApiError::UpstreamUnreachable { .. } => {
                (StatusCode::BAD_GATEWAY, "upstream_unreachable")
            }
            ApiError::UpstreamStatus { .. } => (StatusCode::BAD_GATEWAY, "upstream_status"),
            ApiError::UpstreamTimeout { .. } => (StatusCode::BAD_GATEWAY, "upstream_timeout"),
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    #[serde(rename = "type")]
    typ: &'static str,
    code: &'a str,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = self.status_and_code();
        let typ = match status.as_u16() {
            400..=499 => "invalid_request_error",
            _ => "server_error",
        };
        // Pick a client-facing message that NEVER leaks worker URLs or raw
        // source chains; full structured details are logged server-side.
        let message = match &self {
            ApiError::Internal(e) => {
                // `{:#}` prints the anyhow chain (top error + sources) — `?e`
                // would only show the outermost message.
                tracing::error!("internal error serving request: {e:#}");
                "internal error".to_string()
            }
            ApiError::UpstreamUnreachable { worker, source } => {
                tracing::warn!(
                    upstream = %worker,
                    error = %format_args!("{source:#}"),
                    "upstream worker unreachable",
                );
                "upstream unavailable".to_string()
            }
            ApiError::UpstreamStatus { status } => {
                tracing::warn!(
                    upstream_status = %status,
                    "upstream returned an error status",
                );
                "upstream returned an error status".to_string()
            }
            ApiError::UpstreamTimeout { worker } => {
                tracing::warn!(upstream = %worker, "upstream request timed out");
                "upstream request timed out".to_string()
            }
            ApiError::BadRequest(_) | ApiError::ModelNotFound(_) => self.to_string(),
        };
        let mut resp = (
            status,
            Json(ErrorEnvelope {
                error: ErrorBody { typ, code, message },
            }),
        )
            .into_response();
        resp.headers_mut()
            .insert(X_ROUTER_ERROR_CODE, HeaderValue::from_static(code));
        resp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;
    use serde::Deserialize;

    fn collect_body(resp: Response) -> String {
        let bytes = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { BodyExt::collect(resp.into_body()).await.unwrap().to_bytes() });
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Pin the exact JSON envelope shape that clients see. Renaming any of
    /// these fields (or removing one) breaks every downstream consumer
    /// silently, so we deserialize into a fixed struct rather than
    /// regex-matching the rendered JSON.
    #[derive(Deserialize)]
    struct ErrEnv {
        error: ErrField,
    }

    #[derive(Deserialize)]
    struct ErrField {
        #[serde(rename = "type")]
        typ: String,
        code: String,
        message: String,
    }

    fn parse_envelope(resp: Response) -> (StatusCode, Option<String>, ErrEnv) {
        let status = resp.status();
        let code_header = resp
            .headers()
            .get("x-router-error-code")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let body_str = collect_body(resp);
        let env: ErrEnv = serde_json::from_str(&body_str)
            .unwrap_or_else(|e| panic!("envelope did not match expected shape: {e}: {body_str}"));
        (status, code_header, env)
    }

    #[test]
    fn upstream_unreachable_envelope_has_code_and_no_leak() {
        let worker = "http://10.0.0.42:30000";
        let secret = "TLS_HANDSHAKE_FAILED at /etc/secret_ca.pem";
        let err = ApiError::UpstreamUnreachable {
            worker: worker.into(),
            source: anyhow::anyhow!("{secret}"),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_unreachable"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_unreachable\""), "{body}");
        assert!(body.contains("\"type\":\"server_error\""), "{body}");
        assert!(
            !body.contains(worker) && !body.contains(secret),
            "client body must NOT leak worker URL or reqwest source chain; got: {body}",
        );
    }

    #[test]
    fn upstream_status_envelope_has_code() {
        let err = ApiError::UpstreamStatus {
            status: StatusCode::INTERNAL_SERVER_ERROR,
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_status"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_status\""), "{body}");
    }

    #[test]
    fn upstream_timeout_envelope_has_code_and_no_leak() {
        let worker = "http://10.0.0.42:30000";
        let err = ApiError::UpstreamTimeout {
            worker: worker.into(),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_timeout"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_timeout\""), "{body}");
        assert!(
            !body.contains(worker),
            "client body must NOT leak worker URL; got: {body}",
        );
    }

    #[test]
    fn bad_request_envelope_has_expected_shape() {
        let msg = "invalid_request: body must be an object";
        let err = ApiError::BadRequest(msg.into());
        let resp = err.into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(code_header.as_deref(), Some("bad_request"));
        assert_eq!(env.error.code, "bad_request");
        assert_eq!(env.error.typ, "invalid_request_error");
        assert!(
            !env.error.message.is_empty(),
            "message must not be empty: {:?}",
            env.error.message,
        );
        // Sanity: variants must not collide on the canonical code.
        assert_ne!(env.error.code, "internal_error");
        assert_ne!(env.error.code, "model_not_found");
    }

    #[test]
    fn model_not_found_envelope_has_expected_shape() {
        let err = ApiError::ModelNotFound("ghost-7b".into());
        let resp = err.into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(code_header.as_deref(), Some("model_not_found"));
        assert_eq!(env.error.code, "model_not_found");
        assert_eq!(env.error.typ, "invalid_request_error");
        assert!(
            !env.error.message.is_empty(),
            "message must not be empty: {:?}",
            env.error.message,
        );
        // Sanity: variants must not collide on the canonical code.
        assert_ne!(env.error.code, "internal_error");
        assert_ne!(env.error.code, "bad_request");
    }

    #[test]
    fn internal_error_response_sanitizes_anyhow_chain() {
        let secret_msg = "internal /opt/secret/credential.json missing";
        let err = ApiError::Internal(anyhow::anyhow!("{secret_msg}"));
        let resp = err.into_response();
        let body_str = collect_body(resp);
        // Generic to client:
        assert!(
            body_str.contains("\"code\":\"internal_error\""),
            "body: {body_str}"
        );
        assert!(
            body_str.contains("\"type\":\"server_error\""),
            "body: {body_str}"
        );
        // No leak of the original anyhow message:
        assert!(
            !body_str.contains(secret_msg),
            "ApiError::Internal must not leak anyhow chain to client; got: {body_str}"
        );
    }
}
