//! Router implementations

use std::fmt::Debug;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::protocols::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    rerank::RerankRequest,
    responses::{ResponsesGetParams, ResponsesRequest},
};

pub mod conversations;
pub mod error;
pub mod factory;
pub mod grpc;
pub mod header_utils;
pub mod http;
pub mod mcp_utils;
pub mod mesh;
pub mod openai;
pub mod parse;
pub mod persistence_utils;
pub mod router_manager;
pub mod streaming_utils;
pub mod tokenize;

pub use factory::RouterFactory;
// Re-export HTTP routers for convenience
pub use http::{pd_router, pd_types, router};

/// Core trait for all router implementations
///
/// This trait provides a unified interface for routing requests,
/// regardless of whether it's a regular router or PD router.
#[async_trait]
pub trait RouterTrait: Send + Sync + Debug {
    /// Get a reference to self as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Route a health generate request
    async fn health_generate(&self, _req: Request<Body>) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not implemented",
        )
            .into_response()
    }

    /// Get server information
    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Server info not implemented").into_response()
    }

    /// Get available models
    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Get models not implemented").into_response()
    }

    /// Get model information
    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Get model info not implemented",
        )
            .into_response()
    }

    /// Route a generate request
    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Generate endpoint not implemented",
        )
            .into_response()
    }

    /// Route a chat completion request
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route a chat completion request, preserving top-level extension fields
    /// from the client's original JSON payload (e.g. engine extensions like
    /// `cache_salt`).
    ///
    /// The gateway deserializes chat requests into `ChatCompletionRequest`
    /// (openai-protocol 1.0.0), which drops unknown top-level fields; without
    /// `extensions` those fields would be silently lost when the request is
    /// re-serialized for the worker. Routers that forward the request body
    /// should merge `extensions` back in; routers without access to the raw
    /// payload (e.g. gRPC ingress) use the default implementation, which
    /// falls back to [`RouterTrait::route_chat`].
    async fn route_chat_with_extensions(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
        extensions: &serde_json::Map<String, serde_json::Value>,
    ) -> Response {
        let _ = extensions;
        self.route_chat(headers, body, model_id).await
    }

    /// Route a completion request
    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Completion endpoint not implemented",
        )
            .into_response()
    }

    /// Route a responses request
    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses endpoint not implemented",
        )
            .into_response()
    }

    /// Retrieve a stored/background response by id
    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Get response not implemented").into_response()
    }

    /// Cancel a background response by id
    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Cancel response not implemented",
        )
            .into_response()
    }

    /// Delete a response by id
    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses delete endpoint not implemented",
        )
            .into_response()
    }

    /// List input items of a response by id
    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses list input items endpoint not implemented",
        )
            .into_response()
    }

    /// Route embedding requests (OpenAI-compatible /v1/embeddings)
    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Embeddings not implemented").into_response()
    }

    /// Route classification requests (OpenAI-compatible /v1/classify)
    async fn route_classify(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ClassifyRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Classify not implemented").into_response()
    }

    /// Route rerank requests
    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Rerank not implemented").into_response()
    }

    /// Get router type name
    fn router_type(&self) -> &'static str;

    /// Check if this is a PD router
    fn is_pd_mode(&self) -> bool {
        self.router_type() == "pd"
    }
}

/// Extract the top-level fields of a raw chat-completions JSON payload that
/// the typed `ChatCompletionRequest` does not know about (e.g. engine
/// extensions like `cache_salt`).
///
/// `typed_before_normalize` is the typed request serialized BEFORE
/// `Normalizable::normalize()` runs: fields the typed request knows but drops
/// during normalization (e.g. deprecated `max_tokens`, migrated to
/// `max_completion_tokens`) are still present there, so they are correctly
/// NOT treated as extensions. Null-valued fields are skipped: JSON null is
/// semantically "absent" for optional fields, and forwarding an explicit null
/// for a field the gateway currently drops would change behavior.
///
/// The gateway pins openai-protocol 1.0.0, whose `ChatCompletionRequest` has
/// no flattened extension map, so unknown fields must be carried separately
/// and merged back when forwarding to the worker (see
/// https://github.com/sgl-project/sglang/issues/48793).
pub(crate) fn extract_unknown_request_fields(
    raw: &serde_json::Value,
    typed_before_normalize: &serde_json::Value,
) -> serde_json::Map<String, serde_json::Value> {
    let mut extensions = serde_json::Map::new();
    let Some(raw_obj) = raw.as_object() else {
        return extensions;
    };
    let known = typed_before_normalize
        .as_object()
        .map(|obj| obj.keys().collect::<std::collections::HashSet<_>>())
        .unwrap_or_default();
    for (key, value) in raw_obj {
        if !known.contains(key) && !value.is_null() {
            extensions.insert(key.clone(), value.clone());
        }
    }
    extensions
}

/// Merge preserved extension fields into a re-serialized typed request body.
///
/// Known fields (already validated and normalized in `typed`) always win on
/// key collision, so extensions can only add fields the typed request does
/// not carry.
pub(crate) fn merge_unknown_request_fields(
    mut typed: serde_json::Value,
    extensions: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Value {
    if let Some(obj) = typed.as_object_mut() {
        for (key, value) in extensions {
            obj.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    typed
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_extract_unknown_request_fields_preserves_cache_salt() {
        // Regression test for https://github.com/sgl-project/sglang/issues/48793
        let raw = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "cache_salt": "tenant-a",
            "extra_key": "k",
            "temperature": null,
        });
        // Serialized typed request BEFORE normalize(): known fields are
        // present (including deprecated `max_tokens`, which normalize()
        // migrates to `max_completion_tokens` and clears).
        let typed_before_normalize = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "logprobs": false,
            "max_tokens": 1,
            "stream": false,
        });

        let extensions = extract_unknown_request_fields(&raw, &typed_before_normalize);

        assert_eq!(
            extensions.get("cache_salt").and_then(|v| v.as_str()),
            Some("tenant-a"),
            "cache_salt must be preserved, got: {extensions:?}"
        );
        assert_eq!(
            extensions.get("extra_key").and_then(|v| v.as_str()),
            Some("k"),
            "other engine extension fields must also be preserved, got: {extensions:?}"
        );
        assert!(
            !extensions.contains_key("max_tokens"),
            "deprecated max_tokens is a known field and must not be re-added after normalize() clears it, got: {extensions:?}"
        );
        assert!(
            !extensions.contains_key("temperature"),
            "null-valued fields are semantically absent and must not be forwarded, got: {extensions:?}"
        );
        assert!(
            !extensions.contains_key("model"),
            "known fields must not be treated as extensions, got: {extensions:?}"
        );
    }

    #[test]
    fn test_merge_unknown_request_fields_keeps_known_fields() {
        let typed = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 1,
        });
        let mut extensions = serde_json::Map::new();
        extensions.insert(
            "cache_salt".to_string(),
            serde_json::Value::String("tenant-a".to_string()),
        );
        // A colliding key must never override the validated typed value.
        extensions.insert(
            "model".to_string(),
            serde_json::Value::String("other-model".to_string()),
        );

        let merged = merge_unknown_request_fields(typed, &extensions);

        assert_eq!(
            merged.get("cache_salt").and_then(|v| v.as_str()),
            Some("tenant-a")
        );
        assert_eq!(
            merged.get("model").and_then(|v| v.as_str()),
            Some("test-model"),
            "typed known fields must win over extensions, got: {merged}"
        );
        assert_eq!(
            merged.get("max_completion_tokens").and_then(|v| v.as_u64()),
            Some(1)
        );
    }
}
