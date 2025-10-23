//! OpenAI router - main coordinator that delegates to specialized modules

use std::{
    any::Any,
    sync::{atomic::AtomicBool, Arc},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use dashmap::DashMap;
use futures_util::StreamExt;
use serde_json::{json, to_value, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;

// Import from sibling modules
use super::conversations::{
    create_conversation, create_conversation_items, delete_conversation, delete_conversation_item,
    get_conversation, get_conversation_item, list_conversation_items, persist_conversation_items,
    update_conversation,
};
use super::{
    mcp::{
        execute_tool_loop, mcp_manager_from_request_tools, prepare_mcp_payload_for_streaming,
        McpLoopConfig,
    },
    responses::{mask_tools_as_mcp, patch_streaming_response_json},
    streaming::handle_streaming_response,
    utils::{apply_provider_headers, extract_auth_header, probe_endpoint_for_model},
};
use crate::{
    config::CircuitBreakerConfig,
    core::{CircuitBreaker, CircuitBreakerConfig as CoreCircuitBreakerConfig},
    data_connector::{
        conversation_items::{ListParams, SortOrder},
        ConversationId, ResponseId, SharedConversationItemStorage, SharedConversationStorage,
        SharedResponseStorage,
    },
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{
            ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesGetParams,
            ResponsesRequest,
        },
    },
    routers::header_utils::apply_request_headers,
};

// ============================================================================
// OpenAIRouter Struct
// ============================================================================

/// Cached endpoint information
#[derive(Clone, Debug)]
struct CachedEndpoint {
    url: String,
    cached_at: Instant,
}

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
    response_storage: SharedResponseStorage,
    /// Conversation storage backend
    conversation_storage: SharedConversationStorage,
    /// Conversation item storage backend
    conversation_item_storage: SharedConversationItemStorage,
    /// Optional MCP manager (enabled via config presence)
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
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
    /// Maximum number of conversation items to attach as input when a conversation is provided
    const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;

    /// Model discovery cache TTL (1 hour)
    const MODEL_CACHE_TTL_SECS: u64 = 3600;

    /// Create a new OpenAI router
    pub async fn new(
        worker_urls: Vec<String>,
        circuit_breaker_config: Option<CircuitBreakerConfig>,
        response_storage: SharedResponseStorage,
        conversation_storage: SharedConversationStorage,
        conversation_item_storage: SharedConversationItemStorage,
    ) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        // Normalize URLs (remove trailing slashes)
        let worker_urls: Vec<String> = worker_urls
            .into_iter()
            .map(|url| url.trim_end_matches('/').to_string())
            .collect();

        // Convert circuit breaker config
        let core_cb_config = circuit_breaker_config
            .map(|cb| CoreCircuitBreakerConfig {
                failure_threshold: cb.failure_threshold,
                success_threshold: cb.success_threshold,
                timeout_duration: Duration::from_secs(cb.timeout_duration_secs),
                window_duration: Duration::from_secs(cb.window_duration_secs),
            })
            .unwrap_or_default();

        let circuit_breaker = CircuitBreaker::with_config(core_cb_config);

        // Optional MCP manager activation via env var path (config-driven gate)
        let mcp_manager = match std::env::var("SGLANG_MCP_CONFIG").ok() {
            Some(path) if !path.trim().is_empty() => {
                match crate::mcp::McpConfig::from_file(&path).await {
                    Ok(cfg) => match crate::mcp::McpClientManager::new(cfg).await {
                        Ok(mgr) => Some(Arc::new(mgr)),
                        Err(err) => {
                            warn!("Failed to initialize MCP manager: {}", err);
                            None
                        }
                    },
                    Err(err) => {
                        warn!("Failed to load MCP config from '{}': {}", path, err);
                        None
                    }
                }
            }
            _ => None,
        };

        Ok(Self {
            client,
            worker_urls,
            model_cache: Arc::new(DashMap::new()),
            circuit_breaker,
            healthy: AtomicBool::new(true),
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_manager,
        })
    }

    /// Discover which endpoint has the model
    async fn find_endpoint_for_model(
        &self,
        model_id: &str,
        auth_header: Option<&str>,
    ) -> Result<String, Response> {
        // Single endpoint - fast path
        if self.worker_urls.len() == 1 {
            return Ok(self.worker_urls[0].clone());
        }

        // Check cache
        if let Some(entry) = self.model_cache.get(model_id) {
            if entry.cached_at.elapsed() < Duration::from_secs(Self::MODEL_CACHE_TTL_SECS) {
                return Ok(entry.url.clone());
            }
        }

        // Probe all endpoints in parallel
        let mut handles = vec![];
        let model = model_id.to_string();
        let auth = auth_header.map(|s| s.to_string());

        for url in &self.worker_urls {
            let handle = tokio::spawn(probe_endpoint_for_model(
                self.client.clone(),
                url.clone(),
                model.clone(),
                auth.clone(),
            ));
            handles.push(handle);
        }

        // Return first successful endpoint
        for handle in handles {
            if let Ok(Ok(url)) = handle.await {
                // Cache it
                self.model_cache.insert(
                    model_id.to_string(),
                    CachedEndpoint {
                        url: url.clone(),
                        cached_at: Instant::now(),
                    },
                );
                return Ok(url);
            }
        }

        // Model not found on any endpoint
        Err((
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("Model '{}' not found on any endpoint", model_id),
                    "type": "model_not_found",
                }
            })),
        )
            .into_response())
    }

    /// Handle non-streaming response with optional MCP tool loop
    async fn handle_non_streaming_response(
        &self,
        url: String,
        headers: Option<&HeaderMap>,
        mut payload: Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> Response {
        // Check if MCP is active for this request
        let req_mcp_manager = if let Some(ref tools) = original_body.tools {
            mcp_manager_from_request_tools(tools.as_slice()).await
        } else {
            None
        };
        let active_mcp = req_mcp_manager.as_ref().or(self.mcp_manager.as_ref());

        let mut response_json: Value;

        // If MCP is active, execute tool loop
        if let Some(mcp) = active_mcp {
            let config = McpLoopConfig::default();

            // Transform MCP tools to function tools
            prepare_mcp_payload_for_streaming(&mut payload, mcp);

            match execute_tool_loop(
                &self.client,
                &url,
                headers,
                payload,
                original_body,
                mcp,
                &config,
            )
            .await
            {
                Ok(resp) => response_json = resp,
                Err(err) => {
                    self.circuit_breaker.record_failure();
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": {"message": err}})),
                    )
                        .into_response();
                }
            }
        } else {
            // No MCP - simple request

            let mut request_builder = self.client.post(&url).json(&payload);
            if let Some(h) = headers {
                request_builder = apply_request_headers(h, request_builder, true);
            }

            let response = match request_builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    tracing::error!(
                        url = %url,
                        error = %e,
                        "Failed to forward request to OpenAI"
                    );
                    return (
                        StatusCode::BAD_GATEWAY,
                        format!("Failed to forward request to OpenAI: {}", e),
                    )
                        .into_response();
                }
            };

            if !response.status().is_success() {
                self.circuit_breaker.record_failure();
                let status = StatusCode::from_u16(response.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let body = response.text().await.unwrap_or_default();
                return (status, body).into_response();
            }

            response_json = match response.json::<Value>().await {
                Ok(r) => r,
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to parse upstream response: {}", e),
                    )
                        .into_response();
                }
            };

            self.circuit_breaker.record_success();
        }

        // Patch response with metadata
        mask_tools_as_mcp(&mut response_json, original_body);
        patch_streaming_response_json(
            &mut response_json,
            original_body,
            original_previous_response_id.as_deref(),
        );

        // Always persist conversation items and response (even without conversation)
        if let Err(err) = persist_conversation_items(
            self.conversation_storage.clone(),
            self.conversation_item_storage.clone(),
            self.response_storage.clone(),
            &response_json,
            original_body,
        )
        .await
        {
            warn!("Failed to persist conversation items: {}", err);
        }

        (StatusCode::OK, Json(response_json)).into_response()
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
        _model_id: Option<&str>,
    ) -> Response {
        if !self.circuit_breaker.can_execute() {
            return (StatusCode::SERVICE_UNAVAILABLE, "Circuit breaker open").into_response();
        }

        // Extract auth header
        let auth = extract_auth_header(headers);

        // Find endpoint for model
        let base_url = match self
            .find_endpoint_for_model(body.model.as_str(), auth)
            .await
        {
            Ok(url) => url,
            Err(response) => return response,
        };

        // Serialize request body, removing SGLang-only fields
        let mut payload = match to_value(body) {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to serialize request: {}", e),
                )
                    .into_response();
            }
        };
        if let Some(obj) = payload.as_object_mut() {
            // Always remove SGLang-specific fields (unsupported by OpenAI)
            for key in [
                "top_k",
                "min_p",
                "min_tokens",
                "regex",
                "ebnf",
                "stop_token_ids",
                "no_stop_trim",
                "ignore_eos",
                "continue_final_message",
                "skip_special_tokens",
                "lora_path",
                "session_params",
                "separate_reasoning",
                "stream_reasoning",
                "chat_template_kwargs",
                "return_hidden_states",
                "repetition_penalty",
                "sampling_seed",
            ] {
                obj.remove(key);
            }

            // Remove logprobs if false (Gemini don't accept it)
            if obj.get("logprobs").and_then(|v| v.as_bool()) == Some(false) {
                obj.remove("logprobs");
            }
        }

        let url = format!("{}/v1/chat/completions", base_url);
        let mut req = self.client.post(&url).json(&payload);

        // Forward Authorization header if provided
        if let Some(h) = headers {
            if let Some(auth) = h.get("authorization").or_else(|| h.get("Authorization")) {
                req = req.header("Authorization", auth);
            }
        }

        // Accept SSE when stream=true
        if body.stream {
            req = req.header("Accept", "text/event-stream");
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                self.circuit_breaker.record_failure();
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Failed to contact upstream: {}", e),
                )
                    .into_response();
            }
        };

        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !body.stream {
            // Capture Content-Type before consuming response body
            let content_type = resp.headers().get(CONTENT_TYPE).cloned();
            match resp.bytes().await {
                Ok(body) => {
                    self.circuit_breaker.record_success();
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    if let Some(ct) = content_type {
                        response.headers_mut().insert(CONTENT_TYPE, ct);
                    }
                    response
                }
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read response: {}", e),
                    )
                        .into_response()
                }
            }
        } else {
            // Stream SSE bytes to client
            let stream = resp.bytes_stream();
            let (tx, rx) = mpsc::unbounded_channel();
            tokio::spawn(async move {
                let mut s = stream;
                while let Some(chunk) = s.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok(bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Stream error: {}", e)));
                            break;
                        }
                    }
                }
            });
            let mut response = Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
            *response.status_mut() = status;
            response
                .headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            response
        }
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
        // Extract auth header
        let auth = extract_auth_header(headers);

        // Find endpoint for model (use model_id if provided, otherwise use body.model)
        let model = model_id.unwrap_or(body.model.as_str());
        let base_url = match self.find_endpoint_for_model(model, auth).await {
            Ok(url) => url,
            Err(response) => return response,
        };

        let url = format!("{}/v1/responses", base_url);

        // Validate mutually exclusive params: previous_response_id and conversation
        // TODO: this validation logic should move the right place, also we need a proper error message module
        if body.previous_response_id.is_some() && body.conversation.is_some() {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": "Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.",
                        "type": "invalid_request_error",
                        "param": Value::Null,
                        "code": "mutually_exclusive_parameters"
                    }
                })),
            )
                .into_response();
        }

        // Clone the body for validation and logic, but we'll build payload differently
        let mut request_body = body.clone();
        if let Some(model) = model_id {
            request_body.model = model.to_string();
        }
        // Do not forward conversation field upstream; retain for local persistence only
        request_body.conversation = None;

        // Store the original previous_response_id for the response
        let original_previous_response_id = request_body.previous_response_id.clone();

        // Handle previous_response_id by loading prior context
        let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;
        if let Some(prev_id_str) = request_body.previous_response_id.clone() {
            let prev_id = ResponseId::from(prev_id_str.as_str());
            match self
                .response_storage
                .get_response_chain(&prev_id, None)
                .await
            {
                Ok(chain) => {
                    let mut items = Vec::new();
                    for stored in chain.responses.iter() {
                        // Convert input to conversation item
                        items.push(ResponseInputOutputItem::Message {
                            id: format!("msg_u_{}", stored.id.0.trim_start_matches("resp_")),
                            role: "user".to_string(),
                            content: vec![ResponseContentPart::InputText {
                                text: stored.input.clone(),
                            }],
                            status: Some("completed".to_string()),
                        });

                        // Convert output to conversation items directly from stored response
                        if let Some(output_arr) =
                            stored.raw_response.get("output").and_then(|v| v.as_array())
                        {
                            for item in output_arr {
                                if let Ok(output_item) =
                                    serde_json::from_value::<ResponseInputOutputItem>(item.clone())
                                {
                                    items.push(output_item);
                                }
                            }
                        }
                    }
                    conversation_items = Some(items);
                    request_body.previous_response_id = None;
                }
                Err(e) => {
                    warn!(
                        "Failed to load previous response chain for {}: {}",
                        prev_id_str, e
                    );
                }
            }
        }

        // Handle conversation by loading history
        if let Some(conv_id_str) = body.conversation.clone() {
            let conv_id = ConversationId::from(conv_id_str.as_str());

            // Verify conversation exists
            if let Ok(None) = self.conversation_storage.get_conversation(&conv_id).await {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({"error": "Conversation not found"})),
                )
                    .into_response();
            }

            // Load conversation history (ascending order for chronological context)
            let params = ListParams {
                limit: Self::MAX_CONVERSATION_HISTORY_ITEMS,
                order: SortOrder::Asc,
                after: None,
            };

            match self
                .conversation_item_storage
                .list_items(&conv_id, params)
                .await
            {
                Ok(stored_items) => {
                    let mut items: Vec<ResponseInputOutputItem> = Vec::new();
                    for item in stored_items.into_iter() {
                        // Only use message items for conversation context
                        // Skip non-message items (reasoning, function calls, etc.)
                        if item.item_type == "message" {
                            if let Ok(content_parts) =
                                serde_json::from_value::<Vec<ResponseContentPart>>(
                                    item.content.clone(),
                                )
                            {
                                items.push(ResponseInputOutputItem::Message {
                                    id: item.id.0.clone(),
                                    role: item.role.clone().unwrap_or_else(|| "user".to_string()),
                                    content: content_parts,
                                    status: item.status.clone(),
                                });
                            }
                        }
                    }

                    // Append current request
                    match &request_body.input {
                        ResponseInput::Text(text) => {
                            items.push(ResponseInputOutputItem::Message {
                                id: format!("msg_u_{}", conv_id.0),
                                role: "user".to_string(),
                                content: vec![ResponseContentPart::InputText {
                                    text: text.clone(),
                                }],
                                status: Some("completed".to_string()),
                            });
                        }
                        ResponseInput::Items(current_items) => {
                            items.extend_from_slice(current_items);
                        }
                    }

                    request_body.input = ResponseInput::Items(items);
                }
                Err(e) => {
                    warn!("Failed to load conversation history: {}", e);
                }
            }
        }

        // If we have conversation_items from previous_response_id, use them
        if let Some(mut items) = conversation_items {
            // Append current request
            match &request_body.input {
                ResponseInput::Text(text) => {
                    items.push(ResponseInputOutputItem::Message {
                        id: format!(
                            "msg_u_{}",
                            original_previous_response_id
                                .as_ref()
                                .unwrap_or(&"new".to_string())
                        ),
                        role: "user".to_string(),
                        content: vec![ResponseContentPart::InputText { text: text.clone() }],
                        status: Some("completed".to_string()),
                    });
                }
                ResponseInput::Items(current_items) => {
                    items.extend_from_slice(current_items);
                }
            }

            request_body.input = ResponseInput::Items(items);
        }

        // Always set store=false for upstream (we store internally)
        request_body.store = Some(false);

        // Convert to JSON and strip SGLang-specific fields
        let mut payload = match to_value(&request_body) {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to serialize request: {}", e),
                )
                    .into_response();
            }
        };

        // Remove SGLang-specific fields only
        if let Some(obj) = payload.as_object_mut() {
            // Remove SGLang-specific fields (not part of OpenAI API)
            for key in [
                "request_id",
                "priority",
                "top_k",
                "min_p",
                "min_tokens",
                "regex",
                "ebnf",
                "stop_token_ids",
                "no_stop_trim",
                "ignore_eos",
                "continue_final_message",
                "skip_special_tokens",
                "lora_path",
                "session_params",
                "separate_reasoning",
                "stream_reasoning",
                "chat_template_kwargs",
                "return_hidden_states",
                "repetition_penalty",
                "sampling_seed",
            ] {
                obj.remove(key);
            }
            // XAI (Grok models) requires special handling of input items
            // Check if model is a Grok model
            let is_grok_model = obj
                .get("model")
                .and_then(|v| v.as_str())
                .map(|m| m.starts_with("grok"))
                .unwrap_or(false);

            if is_grok_model {
                // XAI doesn't support the OPENAI item type input: https://platform.openai.com/docs/api-reference/responses/create#responses-create-input-input-item-list-item
                // To Achieve XAI compatibility, strip extra fields from input messages (id, status)
                // XAI doesn't support output_text as type for content with role of assistant
                // so normalize content types: output_text -> input_text
                if let Some(input_arr) = obj.get_mut("input").and_then(Value::as_array_mut) {
                    for item_obj in input_arr.iter_mut().filter_map(Value::as_object_mut) {
                        // Remove fields not universally supported
                        item_obj.remove("id");
                        item_obj.remove("status");

                        // Normalize content types to input_text (xAI compatibility)
                        if let Some(content_arr) =
                            item_obj.get_mut("content").and_then(Value::as_array_mut)
                        {
                            for content_obj in
                                content_arr.iter_mut().filter_map(Value::as_object_mut)
                            {
                                // Change output_text to input_text
                                if content_obj.get("type").and_then(Value::as_str)
                                    == Some("output_text")
                                {
                                    content_obj.insert(
                                        "type".to_string(),
                                        Value::String("input_text".to_string()),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Delegate to streaming or non-streaming handler
        if body.stream.unwrap_or(false) {
            handle_streaming_response(
                &self.client,
                &self.circuit_breaker,
                self.mcp_manager.as_ref(),
                self.response_storage.clone(),
                self.conversation_storage.clone(),
                self.conversation_item_storage.clone(),
                url,
                headers,
                payload,
                body,
                original_previous_response_id,
            )
            .await
        } else {
            self.handle_non_streaming_response(
                url,
                headers,
                payload,
                body,
                original_previous_response_id,
            )
            .await
        }
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
