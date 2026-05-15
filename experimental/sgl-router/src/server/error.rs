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

    #[error("upstream worker error: {0}")]
    UpstreamWorker(String),

    #[error("internal: {0}")]
    Internal(#[from] anyhow::Error),
}

impl ApiError {
    fn status_and_code(&self) -> (StatusCode, &'static str) {
        match self {
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            ApiError::ModelNotFound(_) => (StatusCode::NOT_FOUND, "model_not_found"),
            ApiError::UpstreamWorker(_) => (StatusCode::BAD_GATEWAY, "upstream_error"),
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
        // For Internal: log full chain server-side, return generic message to client.
        let message = match &self {
            ApiError::Internal(e) => {
                // `{:#}` prints the anyhow chain (top error + sources) — `?e`
                // would only show the outermost message.
                tracing::error!("internal error serving request: {e:#}");
                "internal error".to_string()
            }
            _ => self.to_string(),
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

    #[test]
    fn internal_error_response_sanitizes_anyhow_chain() {
        use http_body_util::BodyExt;
        let secret_msg = "internal /opt/secret/credential.json missing";
        let err = ApiError::Internal(anyhow::anyhow!("{secret_msg}"));
        let resp = err.into_response();
        let bytes = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { BodyExt::collect(resp.into_body()).await.unwrap().to_bytes() });
        let body_str = String::from_utf8_lossy(&bytes);
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
