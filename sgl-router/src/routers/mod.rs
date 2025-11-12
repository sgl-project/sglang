//! Router implementations

use std::fmt::Debug;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::Value;

use crate::protocols::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    rerank::RerankRequest,
    responses::{ResponsesGetParams, ResponsesRequest},
};

pub mod factory;
pub mod grpc;
pub mod header_utils;
pub mod http;
pub mod openai; // New refactored OpenAI router module
pub mod router_manager;

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
    async fn health_generate(&self, req: Request<Body>) -> Response;

    /// Get server information
    async fn get_server_info(&self, req: Request<Body>) -> Response;

    /// Get available models
    async fn get_models(&self, req: Request<Body>) -> Response;

    /// Get model information
    async fn get_model_info(&self, req: Request<Body>) -> Response;

    /// Route a generate request
    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route a chat completion request
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route a completion request
    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route a responses request
    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Retrieve a stored/background response by id
    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        params: &ResponsesGetParams,
    ) -> Response;

    /// Cancel a background response by id
    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response;

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
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route classification requests (OpenAI-compatible /v1/classify)
    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response;

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response;

    // Conversations API
    async fn create_conversation(&self, _headers: Option<&HeaderMap>, _body: &Value) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Conversations create endpoint not implemented",
        )
            .into_response()
    }

    async fn get_conversation(
        &self,
        _headers: Option<&HeaderMap>,
        _conversation_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Conversations get endpoint not implemented",
        )
            .into_response()
    }

    async fn update_conversation(
        &self,
        _headers: Option<&HeaderMap>,
        _conversation_id: &str,
        _body: &Value,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Conversations update endpoint not implemented",
        )
            .into_response()
    }

    async fn delete_conversation(
        &self,
        _headers: Option<&HeaderMap>,
        _conversation_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Conversations delete endpoint not implemented",
        )
            .into_response()
    }

    /// List items for a conversation
    async fn list_conversation_items(
        &self,
        _headers: Option<&HeaderMap>,
        _conversation_id: &str,
        _limit: Option<usize>,
        _order: Option<String>,
        _after: Option<String>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Conversation items list endpoint not implemented",
        )
            .into_response()
    }

    /// Create items in a conversation
    async fn create_conversation_items(
        &self,
        _headers: Option<&HeaderMap>,
        _conversation_id: &str,
        _body: &Value,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Conversation items create endpoint not implemented",
        )
            .into_response()
    }

    /// Get a single conversation item
    /// The `include` parameter is accepted but not yet implemented
    async fn get_conversation_item(
        &self,
        _headers: Option<&HeaderMap>,
        _conversation_id: &str,
        _item_id: &str,
        _include: Option<Vec<String>>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Conversation item get endpoint not implemented",
        )
            .into_response()
    }

    /// Delete a conversation item
    async fn delete_conversation_item(
        &self,
        _headers: Option<&HeaderMap>,
        _conversation_id: &str,
        _item_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Conversation item delete endpoint not implemented",
        )
            .into_response()
    }

    /// Get router type name
    fn router_type(&self) -> &'static str;

    /// Check if this is a PD router
    fn is_pd_mode(&self) -> bool {
        self.router_type() == "pd"
    }
}
