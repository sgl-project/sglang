// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Paths sgl-model-gateway served for features this router does not provide.
//! They answer 501 naming the gateway, rather than a bare 404, so a client
//! moved from the gateway learns why the call fails.

use crate::server::error::ApiError;
use axum::http::Uri;
use axum::response::{IntoResponse, Response};

/// Responses and conversations API, parser endpoints, the worker and tokenizer
/// management APIs, WASM modules, mesh HA, and the aggregated load and engine
/// metrics endpoints.
pub(crate) const GATEWAY_ONLY_PATHS: &[&str] = &[
    "/v1/responses",
    "/v1/responses/{*rest}",
    "/v1/conversations",
    "/v1/conversations/{*rest}",
    "/parse/{*rest}",
    "/workers",
    "/workers/{*rest}",
    "/v1/tokenizers",
    "/v1/tokenizers/{*rest}",
    "/wasm",
    "/wasm/{*rest}",
    "/ha/{*rest}",
    "/v1/loads",
    "/get_loads",
    "/engine_metrics",
];

pub async fn gateway_only(uri: Uri) -> Response {
    ApiError::GatewayOnlyEndpoint {
        path: uri.path().to_string(),
    }
    .into_response()
}

#[cfg(test)]
mod tests {
    use crate::server::app_context::AppContext;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use std::sync::Arc;
    use tower::ServiceExt;

    #[tokio::test]
    async fn gateway_only_paths_answer_501_naming_the_gateway() {
        for (i, pattern) in super::GATEWAY_ONLY_PATHS.iter().enumerate() {
            let uri = pattern.replace("{*rest}", "x/y");
            let method = ["GET", "POST", "PUT", "DELETE"][i % 4];
            let req = Request::builder()
                .method(method)
                .uri(&uri)
                .body(Body::empty())
                .unwrap();
            let res = crate::server::app::build_router(Arc::new(AppContext::stub()))
                .oneshot(req)
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::NOT_IMPLEMENTED, "{method} {uri}");
            let body = res.into_body().collect().await.unwrap().to_bytes();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["error"]["code"], "gateway_only_endpoint", "{uri}");
            assert_eq!(
                body["error"]["message"],
                format!("{uri} is an sgl-model-gateway endpoint that sgl-router does not provide"),
            );
        }
    }
}
