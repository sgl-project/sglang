//! OpenAI router - main coordinator that delegates to specialized modules

use std::{
    any::Any,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use dashmap::DashMap;
use serde_json::{json, Value};
use tracing::warn;

// Import from sibling modules
use super::conversations::{
    create_conversation, create_conversation_items, delete_conversation, delete_conversation_item,
    get_conversation, get_conversation_item, list_conversation_items, update_conversation,
};
use super::{
    context::{CachedEndpoint, RequestType, SharedComponents},
    pipeline::RequestPipeline,
    utils::apply_provider_headers,
};
use crate::{
    core::{CircuitBreaker, CircuitBreakerConfig as CoreCircuitBreakerConfig},
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseId, ResponseStorage},
    mcp::McpManager,
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{generate_id, ResponsesGetParams, ResponsesRequest},
    },
};

// ============================================================================
// OpenAIRouter Struct
// ============================================================================

/// Router for OpenAI backend
pub struct OpenAIRouter {
    /// HTTP client for upstream OpenAI-compatible API
    client: reqwest::Client,
    /// Multiple OpenAI-compatible API endpoints (OpenAI, xAI, etc.)
    worker_urls: Vec<String>,
    /// Model cache: model_id -> endpoint URL
    model_cache: Arc<DashMap<String, CachedEndpoint>>,
    /// Circuit breaker
    circuit_breaker: CircuitBreaker,
    /// Health status
    healthy: AtomicBool,
    /// Response storage for managing conversation history
    response_storage: Arc<dyn ResponseStorage>,
    /// Conversation storage backend
    conversation_storage: Arc<dyn ConversationStorage>,
    /// Conversation item storage backend
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    /// MCP manager (handles both static and dynamic servers)
    mcp_manager: Arc<McpManager>,
}

impl std::fmt::Debug for OpenAIRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIRouter")
            .field("worker_urls", &self.worker_urls)
            .field("healthy", &self.healthy)
            .finish()
    }
}

impl OpenAIRouter {
    /// Create a new OpenAI router
    pub async fn new(
        worker_urls: Vec<String>,
        ctx: &Arc<crate::app_context::AppContext>,
    ) -> Result<Self, String> {
        // Use HTTP client from AppContext
        let client = ctx.client.clone();

        // Normalize URLs (remove trailing slashes)
        let worker_urls: Vec<String> = worker_urls
            .into_iter()
            .map(|url| url.trim_end_matches('/').to_string())
            .collect();

        // Convert circuit breaker config from AppContext
        let cb = &ctx.router_config.circuit_breaker;
        let core_cb_config = CoreCircuitBreakerConfig {
            failure_threshold: cb.failure_threshold,
            success_threshold: cb.success_threshold,
            timeout_duration: Duration::from_secs(cb.timeout_duration_secs),
            window_duration: Duration::from_secs(cb.window_duration_secs),
        };

        let circuit_breaker = CircuitBreaker::with_config(core_cb_config);

        // Get MCP manager from AppContext (must be initialized)
        let mcp_manager = ctx
            .mcp_manager
            .get()
            .ok_or_else(|| "MCP manager not initialized in AppContext".to_string())?
            .clone();

        Ok(Self {
            client,
            worker_urls,
            model_cache: Arc::new(DashMap::new()),
            circuit_breaker,
            healthy: AtomicBool::new(true),
            response_storage: ctx.response_storage.clone(),
            conversation_storage: ctx.conversation_storage.clone(),
            conversation_item_storage: ctx.conversation_item_storage.clone(),
            mcp_manager,
        })
    }
}

// ============================================================================
// RouterTrait Implementation
// ============================================================================

#[async_trait::async_trait]
impl crate::routers::RouterTrait for OpenAIRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Check all endpoints in parallel - only healthy if ALL are healthy
        if self.worker_urls.is_empty() {
            return (StatusCode::SERVICE_UNAVAILABLE, "No endpoints configured").into_response();
        }

        let mut handles = vec![];
        for url in &self.worker_urls {
            let url = url.clone();
            let client = self.client.clone();

            let handle = tokio::spawn(async move {
                let probe_url = format!("{}/v1/models", url);
                match client
                    .get(&probe_url)
                    .timeout(Duration::from_secs(2))
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let code = resp.status();
                        // Treat success and auth-required as healthy (endpoint reachable)
                        if code.is_success() || code.as_u16() == 401 || code.as_u16() == 403 {
                            Ok(())
                        } else {
                            Err(format!("Endpoint {} returned status {}", url, code))
                        }
                    }
                    Err(e) => Err(format!("Endpoint {} error: {}", url, e)),
                }
            });

            handles.push(handle);
        }

        // Collect all results
        let mut errors = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(())) => (),
                Ok(Err(e)) => errors.push(e),
                Err(e) => errors.push(format!("Task join error: {}", e)),
            }
        }

        if errors.is_empty() {
            (StatusCode::OK, "OK").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Some endpoints unhealthy: {}", errors.join(", ")),
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        let info = json!({
            "router_type": "openai",
            "workers": self.worker_urls.len(),
            "worker_urls": &self.worker_urls
        });
        (StatusCode::OK, info.to_string()).into_response()
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Aggregate models from all endpoints
        if self.worker_urls.is_empty() {
            return (StatusCode::SERVICE_UNAVAILABLE, "No endpoints configured").into_response();
        }

        let headers = req.headers();
        let auth = headers
            .get("authorization")
            .or_else(|| headers.get("Authorization"));

        // Query all endpoints in parallel
        let mut handles = vec![];
        for url in &self.worker_urls {
            let url = url.clone();
            let client = self.client.clone();
            let auth = auth.cloned();

            let handle = tokio::spawn(async move {
                let models_url = format!("{}/v1/models", url);
                let req = client.get(&models_url);

                // Apply provider-specific headers (handles Anthropic, xAI, OpenAI, etc.)
                let req = apply_provider_headers(req, &url, auth.as_ref());

                match req.send().await {
                    Ok(res) => {
                        if res.status().is_success() {
                            match res.json::<Value>().await {
                                Ok(json) => Ok(json),
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to parse models response from '{}': {}",
                                        url,
                                        e
                                    );
                                    Err(())
                                }
                            }
                        } else {
                            tracing::warn!(
                                "Getting models from '{}' failed with status: {}",
                                url,
                                res.status()
                            );
                            Err(())
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Request to get models from '{}' failed: {}", url, e);
                        Err(())
                    }
                }
            });

            handles.push(handle);
        }

        // Collect all model lists
        let mut all_models = Vec::new();
        for handle in handles {
            if let Ok(Ok(json)) = handle.await {
                if let Some(data) = json.get("data").and_then(|v| v.as_array()) {
                    all_models.extend_from_slice(data);
                }
            }
        }

        // Return aggregated models
        let response_json = json!({
            "object": "list",
            "data": all_models
        });

        (StatusCode::OK, Json(response_json)).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        // Not directly supported without model param; return 501
        (
            StatusCode::NOT_IMPLEMENTED,
            "get_model_info not implemented for OpenAI router",
        )
            .into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Generate endpoint is SGLang-specific, not supported for OpenAI backend
        (
            StatusCode::NOT_IMPLEMENTED,
            "Generate endpoint not supported for OpenAI backend",
        )
            .into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Create shared components from router fields
        let components = Arc::new(SharedComponents {
            http_client: self.client.clone(),
            circuit_breaker: Arc::new(self.circuit_breaker.clone()),
            model_cache: self.model_cache.clone(),
            mcp_manager: self.mcp_manager.clone(),
            response_storage: self.response_storage.clone(),
            conversation_storage: self.conversation_storage.clone(),
            conversation_item_storage: self.conversation_item_storage.clone(),
            worker_urls: self.worker_urls.clone(),
        });

        // Execute pipeline
        let pipeline = RequestPipeline::new(self.worker_urls.clone());
        pipeline
            .execute(
                RequestType::Chat(Arc::new(body.clone())),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                components,
            )
            .await
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Completion endpoint not implemented for OpenAI backend
        (
            StatusCode::NOT_IMPLEMENTED,
            "Completion endpoint not implemented for OpenAI backend",
        )
            .into_response()
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Create shared components from router fields
        let components = Arc::new(SharedComponents {
            http_client: self.client.clone(),
            circuit_breaker: Arc::new(self.circuit_breaker.clone()),
            model_cache: self.model_cache.clone(),
            mcp_manager: self.mcp_manager.clone(),
            response_storage: self.response_storage.clone(),
            conversation_storage: self.conversation_storage.clone(),
            conversation_item_storage: self.conversation_item_storage.clone(),
            worker_urls: self.worker_urls.clone(),
        });

        // Execute pipeline
        let pipeline = RequestPipeline::new(self.worker_urls.clone());
        pipeline
            .execute(
                RequestType::Responses(Arc::new(body.clone())),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                components,
            )
            .await
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        let id = ResponseId::from(response_id);
        match self.response_storage.get_response(&id).await {
            Ok(Some(stored)) => {
                let mut response_json = stored.raw_response;
                if let Some(obj) = response_json.as_object_mut() {
                    obj.insert("id".to_string(), json!(id.0));
                }
                (StatusCode::OK, Json(response_json)).into_response()
            }
            Ok(None) => (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Response not found"})),
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to get response: {}", e) })),
            )
                .into_response(),
        }
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Cancel response not implemented for OpenAI router",
        )
            .into_response()
    }

    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
    ) -> Response {
        let resp_id = ResponseId::from(response_id);

        match self.response_storage.get_response(&resp_id).await {
            Ok(Some(stored)) => {
                // Extract items from input field (which is a JSON array)
                let items = match &stored.input {
                    Value::Array(arr) => arr.clone(),
                    _ => vec![],
                };

                // Generate IDs for items if they don't have them
                let items_with_ids: Vec<Value> = items
                    .into_iter()
                    .map(|mut item| {
                        if item.get("id").is_none() {
                            // Generate ID if not present using centralized utility
                            if let Some(obj) = item.as_object_mut() {
                                obj.insert("id".to_string(), json!(generate_id("msg")));
                            }
                        }
                        item
                    })
                    .collect();

                let response_body = json!({
                    "object": "list",
                    "data": items_with_ids,
                    "first_id": items_with_ids.first().and_then(|v| v.get("id").and_then(|i| i.as_str())),
                    "last_id": items_with_ids.last().and_then(|v| v.get("id").and_then(|i| i.as_str())),
                    "has_more": false
                });

                (StatusCode::OK, Json(response_body)).into_response()
            }
            Ok(None) => (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": {
                        "message": format!("No response found with id '{}'", response_id),
                        "type": "invalid_request_error",
                        "param": Value::Null,
                        "code": "not_found"
                    }
                })),
            )
                .into_response(),
            Err(e) => {
                warn!("Failed to retrieve input items for {}: {}", response_id, e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": {
                            "message": format!("Failed to retrieve input items: {}", e),
                            "type": "internal_error",
                            "param": Value::Null,
                            "code": "storage_error"
                        }
                    })),
                )
                    .into_response()
            }
        }
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Embeddings not supported").into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Rerank not supported").into_response()
    }

    async fn route_classify(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ClassifyRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Classify not supported").into_response()
    }

    async fn create_conversation(&self, _headers: Option<&HeaderMap>, body: &Value) -> Response {
        create_conversation(&self.conversation_storage, body.clone()).await
    }

    async fn get_conversation(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
    ) -> Response {
        get_conversation(&self.conversation_storage, conversation_id).await
    }

    async fn update_conversation(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
        body: &Value,
    ) -> Response {
        update_conversation(&self.conversation_storage, conversation_id, body.clone()).await
    }

    async fn delete_conversation(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
    ) -> Response {
        delete_conversation(&self.conversation_storage, conversation_id).await
    }

    fn router_type(&self) -> &'static str {
        "openai"
    }

    async fn list_conversation_items(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
        limit: Option<usize>,
        order: Option<String>,
        after: Option<String>,
    ) -> Response {
        let mut query_params = std::collections::HashMap::new();
        query_params.insert("limit".to_string(), limit.unwrap_or(100).to_string());
        if let Some(after_val) = after {
            if !after_val.is_empty() {
                query_params.insert("after".to_string(), after_val);
            }
        }
        if let Some(order_val) = order {
            query_params.insert("order".to_string(), order_val);
        }

        list_conversation_items(
            &self.conversation_storage,
            &self.conversation_item_storage,
            conversation_id,
            query_params,
        )
        .await
    }

    async fn create_conversation_items(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
        body: &Value,
    ) -> Response {
        create_conversation_items(
            &self.conversation_storage,
            &self.conversation_item_storage,
            conversation_id,
            body.clone(),
        )
        .await
    }

    async fn get_conversation_item(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
        item_id: &str,
        include: Option<Vec<String>>,
    ) -> Response {
        get_conversation_item(
            &self.conversation_storage,
            &self.conversation_item_storage,
            conversation_id,
            item_id,
            include,
        )
        .await
    }

    async fn delete_conversation_item(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
        item_id: &str,
    ) -> Response {
        delete_conversation_item(
            &self.conversation_storage,
            &self.conversation_item_storage,
            conversation_id,
            item_id,
        )
        .await
    }
}
