// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::tokenizer::adapter;
use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizeRequest {
    pub model: String,
    pub prompt: String,
}

#[derive(Serialize)]
#[cfg_attr(test, derive(Deserialize))]
pub struct TokenizeResponse {
    pub model: String,
    pub tokens: Vec<u32>,
    pub count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetokenizeRequest {
    pub model: String,
    pub tokens: Vec<u32>,
    #[serde(default)]
    pub skip_special_tokens: bool,
}

#[derive(Serialize)]
#[cfg_attr(test, derive(Deserialize))]
pub struct DetokenizeResponse {
    pub model: String,
    pub text: String,
}

pub async fn tokenize(
    State(ctx): State<Arc<AppContext>>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    let tok = ctx
        .tokenizers
        .get(&req.model)
        .ok_or_else(|| ApiError::ModelNotFound(req.model.clone()))?;
    // Structured log on failure so an operator can correlate
    // "every encode for model X errors" against the route, model id, and
    // prompt size. The generic anyhow-chain log in ApiError::Internal still
    // fires from IntoResponse — duplication is intentional: the route-level
    // line carries `model` / `prompt_len`, the IntoResponse line carries
    // the full anyhow chain.
    let ids = adapter::encode(&tok, &req.prompt).map_err(|e| {
        tracing::error!(
            route = "/v1/tokenize",
            model = %req.model,
            prompt_len = req.prompt.len(),
            error = ?e,
            "tokenize.encode failed",
        );
        ApiError::Internal(e)
    })?;
    Ok(Json(TokenizeResponse {
        model: req.model,
        count: ids.len(),
        tokens: ids,
    }))
}

pub async fn detokenize(
    State(ctx): State<Arc<AppContext>>,
    Json(req): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, ApiError> {
    let tok = ctx
        .tokenizers
        .get(&req.model)
        .ok_or_else(|| ApiError::ModelNotFound(req.model.clone()))?;
    let text =
        adapter::decode_complete(&tok, &req.tokens, req.skip_special_tokens).map_err(|e| {
            tracing::error!(
                route = "/v1/detokenize",
                model = %req.model,
                n_tokens = req.tokens.len(),
                skip_special = req.skip_special_tokens,
                error = ?e,
                "detokenize.decode_complete failed",
            );
            ApiError::Internal(e)
        })?;
    Ok(Json(DetokenizeResponse {
        model: req.model,
        text,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::config::PolicyKind;

    fn ctx_with_tiny() -> Arc<AppContext> {
        let cfg = crate::config::Config {
            server: crate::config::ServerConfig {
                host: "x".into(),
                port: 0,
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                policy: PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
            admission: crate::config::AdmissionConfig::default(),
        };
        let registry = crate::tokenizer::TokenizerRegistry::load_from_config(&cfg).unwrap();
        let proxy = Arc::new(
            crate::proxy::Proxy::new(std::time::Duration::from_secs(60)).expect("stub proxy"),
        );
        let worker_registry = Arc::new(crate::workers::WorkerRegistry::default());
        let policies = Arc::new(crate::policies::PolicyRegistry::default());
        Arc::new(AppContext::new(
            cfg,
            Arc::new(registry),
            proxy,
            worker_registry,
            policies,
        ))
    }

    #[tokio::test]
    async fn tokenize_round_trip() {
        let app = crate::server::app::build_router(ctx_with_tiny());
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny", "prompt": "hello world"
        }))
        .unwrap();
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/tokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let r: TokenizeResponse = serde_json::from_slice(&bytes).unwrap();
        assert!(r.count > 0);

        let body2 = serde_json::to_vec(&serde_json::json!({
            "model": "tiny", "tokens": r.tokens, "skip_special_tokens": true
        }))
        .unwrap();
        let res2 = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/detokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(body2))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res2.status(), StatusCode::OK);
        let bytes2 = res2.into_body().collect().await.unwrap().to_bytes();
        let r2: DetokenizeResponse = serde_json::from_slice(&bytes2).unwrap();
        assert_eq!(r2.text, "hello world");
    }

    #[tokio::test]
    async fn tokenize_request_does_not_advertise_add_special_tokens() {
        // Regression: TokenizeRequest must not have an `add_special_tokens` field.
        // Background: dynamo-tokenizers cannot honor it. Silently ignoring it
        // would be a footgun for clients that set it.
        let req: TokenizeRequest =
            serde_json::from_str(r#"{"model": "tiny", "prompt": "hi"}"#).unwrap();
        let _ = req; // compiles → schema is correct minus that field

        // If someone sets it anyway, serde should reject with deny_unknown_fields.
        let parsed: Result<TokenizeRequest, _> = serde_json::from_str(
            r#"{"model": "tiny", "prompt": "hi", "add_special_tokens": true}"#,
        );
        assert!(
            parsed.is_err(),
            "add_special_tokens should be rejected as unknown field"
        );
    }

    /// Ported from SMG tests/api/parser_endpoints_test.rs (parse_function_call_missing_fields):
    /// DetokenizeRequest has `deny_unknown_fields`; an extra field must yield 422
    /// Unprocessable Entity from axum's JSON extractor, not 200 with the field silently ignored.
    /// Gap: the existing `tokenize_request_does_not_advertise_add_special_tokens` test only
    /// exercises serde deserialization directly; this test exercises the HTTP layer.
    #[tokio::test]
    async fn detokenize_rejects_unknown_field() {
        let app = crate::server::app::build_router(ctx_with_tiny());
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "tokens": [15496, 995],
            "skip_special_tokens": false,
            "add_special_tokens": true   // unknown field — must be rejected
        }))
        .unwrap();
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/detokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "DetokenizeRequest must reject unknown fields via deny_unknown_fields"
        );
    }

    /// Gap: `tokenize_round_trip` tests only `skip_special_tokens: true`.
    /// When omitted, `#[serde(default)]` gives `false` — a different decode code-path.
    /// This covers the default (omitted) and explicit-false routes end-to-end via HTTP.
    #[tokio::test]
    async fn detokenize_skip_special_tokens_false_default() {
        let app = crate::server::app::build_router(ctx_with_tiny());

        // First tokenize to get IDs.
        let tok_body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny", "prompt": "hello world"
        }))
        .unwrap();
        let tok_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/tokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(tok_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tok_res.status(), StatusCode::OK);
        let tok_bytes = tok_res.into_body().collect().await.unwrap().to_bytes();
        let r: TokenizeResponse = serde_json::from_slice(&tok_bytes).unwrap();

        // Detokenize with skip_special_tokens omitted (defaults to false).
        let det_body_omitted = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "tokens": r.tokens
            // skip_special_tokens intentionally absent — must default to false
        }))
        .unwrap();
        let det_res_omitted = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/detokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(det_body_omitted))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(det_res_omitted.status(), StatusCode::OK);
        let det_bytes_omitted = det_res_omitted
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let d_omitted: DetokenizeResponse = serde_json::from_slice(&det_bytes_omitted).unwrap();
        assert_eq!(
            d_omitted.text, "hello world",
            "detokenize with skip_special_tokens omitted (default false) must round-trip"
        );

        // Also test explicit false — must be identical to omitted.
        let det_body_explicit = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "tokens": r.tokens,
            "skip_special_tokens": false
        }))
        .unwrap();
        let det_res_explicit = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/detokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(det_body_explicit))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(det_res_explicit.status(), StatusCode::OK);
        let det_bytes_explicit = det_res_explicit
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let d_explicit: DetokenizeResponse = serde_json::from_slice(&det_bytes_explicit).unwrap();
        assert_eq!(
            d_explicit.text, d_omitted.text,
            "explicit skip_special_tokens=false must produce same result as omitted"
        );
    }

    #[tokio::test]
    async fn unknown_model_404() {
        let app = crate::server::app::build_router(ctx_with_tiny());
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "nope", "prompt": "x"
        }))
        .unwrap();
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/tokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            res.headers().get("x-router-error-code").unwrap(),
            "model_not_found"
        );
    }
}
