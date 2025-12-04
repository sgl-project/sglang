//! OpenAI router - main coordinator that delegates to specialized modules

use std::{
    any::Any,
    collections::HashSet,
    sync::{atomic::AtomicBool, Arc},
};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::{future::join_all, StreamExt};
use once_cell::sync::Lazy;
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
        ensure_request_mcp_client, execute_tool_loop, prepare_mcp_payload_for_streaming,
        McpLoopConfig,
    },
    responses::{mask_tools_as_mcp, patch_streaming_response_json},
    streaming::handle_streaming_response,
    utils::{apply_provider_headers, extract_auth_header},
};
use crate::{
    app_context::AppContext,
    core::{ModelCard, RuntimeType, Worker, WorkerRegistry},
    data_connector::{
        ConversationId, ConversationItemStorage, ConversationStorage, ListParams, ResponseId,
        ResponseStorage, SortOrder,
    },
    mcp::McpManager,
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{
            generate_id, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponsesGetParams, ResponsesRequest,
        },
    },
};

// ============================================================================
// OpenAIRouter Struct
// ============================================================================

/// Fields specific to SGLang that should be stripped when forwarding to OpenAI-compatible endpoints
static SGLANG_FIELDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
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
    ])
});

/// Router for OpenAI backend
///
/// This router manages connections to OpenAI-compatible API endpoints (OpenAI, xAI, etc.)
/// using the Worker abstraction. Workers are registered via the external worker registration
/// workflow and stored in the WorkerRegistry.
pub struct OpenAIRouter {
    /// HTTP client for upstream OpenAI-compatible API
    client: reqwest::Client,
    /// Worker registry for model-based worker lookup
    worker_registry: Arc<WorkerRegistry>,
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
        let registry_stats = self.worker_registry.stats();
        f.debug_struct("OpenAIRouter")
            .field("registered_workers", &registry_stats.total_workers)
            .field("registered_models", &registry_stats.total_models)
            .field("healthy_workers", &registry_stats.healthy_workers)
            .field("healthy", &self.healthy)
            .finish()
    }
}

impl OpenAIRouter {
    /// Maximum number of conversation items to attach as input when a conversation is provided
    const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;

    /// Create a new OpenAI router
    ///
    /// Workers are registered separately via the external worker registration workflow.
    /// This router queries the WorkerRegistry to find workers that support requested models.
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Use HTTP client from AppContext
        let client = ctx.client.clone();

        // Get worker registry from AppContext
        let worker_registry = ctx.worker_registry.clone();

        // Get MCP manager from AppContext (must be initialized)
        let mcp_manager = ctx
            .mcp_manager
            .get()
            .ok_or_else(|| "MCP manager not initialized in AppContext".to_string())?
            .clone();

        Ok(Self {
            client,
            worker_registry,
            healthy: AtomicBool::new(true),
            response_storage: ctx.response_storage.clone(),
            conversation_storage: ctx.conversation_storage.clone(),
            conversation_item_storage: ctx.conversation_item_storage.clone(),
            mcp_manager,
        })
    }

    /// Refresh models for a single external worker by querying its /v1/models endpoint.
    ///
    /// Returns true if refresh succeeded and models were cached on the worker.
    async fn refresh_worker_models(
        &self,
        worker: &Arc<dyn Worker>,
        auth_header: Option<&HeaderValue>,
    ) -> bool {
        let url = format!("{}/v1/models", worker.url());

        // Build request to backend
        let mut backend_req = self.client.get(&url);
        if let Some(auth) = auth_header {
            backend_req = apply_provider_headers(backend_req, &url, Some(auth));
        }

        match backend_req.send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<Value>().await {
                    Ok(json_response) => {
                        if let Some(data) = json_response.get("data").and_then(|d| d.as_array()) {
                            let model_cards: Vec<ModelCard> = data
                                .iter()
                                .filter_map(|m| m.get("id").and_then(|id| id.as_str()))
                                .map(ModelCard::new)
                                .collect();

                            if !model_cards.is_empty() {
                                tracing::info!(
                                    "Model refresh: found {} models from {}",
                                    model_cards.len(),
                                    url
                                );
                                worker.set_models(model_cards);
                                return true;
                            }
                        }
                        false
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse models response: {}", e);
                        false
                    }
                }
            }
            Ok(response) => {
                tracing::debug!(
                    "Model refresh returned non-success status {} from {}",
                    response.status(),
                    url
                );
                false
            }
            Err(e) => {
                tracing::warn!("Failed to fetch models from backend: {}", e);
                false
            }
        }
    }

    /// Refresh models for ALL external workers in parallel.
    async fn refresh_external_models(&self, auth_header: Option<&HeaderValue>) {
        let external_workers = self.worker_registry.get_workers_filtered(
            None,
            None,
            None,
            Some(RuntimeType::External),
            true, // healthy_only
        );

        if external_workers.is_empty() {
            return;
        }

        tracing::debug!(
            "Refreshing models for {} external workers",
            external_workers.len()
        );

        // Refresh all workers in parallel
        let futures: Vec<_> = external_workers
            .iter()
            .map(|w| self.refresh_worker_models(w, auth_header))
            .collect();

        join_all(futures).await;
    }

    /// Select a worker for the given model using the WorkerRegistry.
    ///
    /// This method queries the registry for external workers (RuntimeType::External)
    /// that support the requested model. It checks:
    /// 1. Workers registered with matching model ID (including aliases via ModelCard)
    /// 2. Worker health status
    /// 3. Circuit breaker state
    ///
    /// If no worker is found with explicit model support, it will refresh models
    /// on all external workers in parallel, then retry the search.
    ///
    /// Returns an error response if no suitable worker is found.
    async fn select_worker_for_model(
        &self,
        model_id: &str,
        auth_header: Option<&HeaderValue>,
    ) -> Result<Arc<dyn Worker>, Box<Response>> {
        // Helper to find candidates for a model
        // Note: We get ALL external workers and filter by supports_model() because
        // wildcard workers (empty models) aren't in the model index but support any model
        let find_candidates = || {
            self.worker_registry
                .get_workers_filtered(
                    None, // Get all external workers, not just those in model index
                    None,
                    None,
                    Some(RuntimeType::External),
                    true, // healthy_only
                )
                .into_iter()
                .filter(|w| w.supports_model(model_id) && w.circuit_breaker().can_execute())
                .collect::<Vec<_>>()
        };

        // First try: find workers that already support this model
        let candidates = find_candidates();
        if !candidates.is_empty() {
            return Ok(candidates
                .into_iter()
                .min_by_key(|w| w.load())
                .expect("candidates is not empty"));
        }

        // No match found - refresh models on all external workers
        tracing::debug!(
            "No worker found for model '{}', refreshing external worker models",
            model_id
        );
        self.refresh_external_models(auth_header).await;

        // Second try: check if any worker now supports the model after refresh
        let candidates = find_candidates();
        if !candidates.is_empty() {
            return Ok(candidates
                .into_iter()
                .min_by_key(|w| w.load())
                .expect("candidates is not empty"));
        }

        Err(Box::new(
            (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": {
                        "message": format!("No worker available for model '{}'", model_id),
                        "type": "model_not_found",
                    }
                })),
            )
                .into_response(),
        ))
    }

    /// Handle non-streaming response with optional MCP tool loop
    async fn handle_non_streaming_response(
        &self,
        worker: &Arc<dyn Worker>,
        headers: Option<&HeaderMap>,
        mut payload: Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> Response {
        let url = format!("{}/v1/responses", worker.url());

        // Check if MCP is active for this request
        // Ensure dynamic client is created if needed
        if let Some(ref tools) = original_body.tools {
            ensure_request_mcp_client(&self.mcp_manager, tools.as_slice()).await;
        }

        // Use the tool loop if the manager has any tools available (static or dynamic).
        let active_mcp = if self.mcp_manager.list_tools().is_empty() {
            None
        } else {
            Some(&self.mcp_manager)
        };

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
                    worker.circuit_breaker().record_failure();
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

            // Apply provider-specific headers (handles Anthropic x-api-key, etc.)
            // Passthrough mode: user's auth header takes priority, worker's key is fallback
            let auth_header = extract_auth_header(headers, worker.api_key());
            request_builder = apply_provider_headers(request_builder, &url, auth_header.as_ref());

            let response = match request_builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    worker.circuit_breaker().record_failure();
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
                worker.circuit_breaker().record_failure();
                let status = StatusCode::from_u16(response.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let body = response.text().await.unwrap_or_default();
                return (status, body).into_response();
            }

            response_json = match response.json::<Value>().await {
                Ok(r) => r,
                Err(e) => {
                    worker.circuit_breaker().record_failure();
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to parse upstream response: {}", e),
                    )
                        .into_response();
                }
            };

            worker.circuit_breaker().record_success();
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
        // Check health of all external workers
        let external_workers: Vec<_> = self
            .worker_registry
            .get_all()
            .into_iter()
            .filter(|w| w.metadata().runtime_type == RuntimeType::External)
            .collect();

        if external_workers.is_empty() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "No external workers registered",
            )
                .into_response();
        }

        let mut healthy_count = 0;
        let mut unhealthy_workers = Vec::new();

        for worker in &external_workers {
            if worker.is_healthy() {
                healthy_count += 1;
            } else {
                unhealthy_workers.push(format!("{} ({})", worker.model_id(), worker.url()));
            }
        }

        if unhealthy_workers.is_empty() {
            (
                StatusCode::OK,
                format!("OK - {} workers healthy", healthy_count),
            )
                .into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "{}/{} workers unhealthy: {}",
                    unhealthy_workers.len(),
                    external_workers.len(),
                    unhealthy_workers.join(", ")
                ),
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        let stats = self.worker_registry.stats();
        let external_workers: Vec<_> = self
            .worker_registry
            .get_all()
            .into_iter()
            .filter(|w| w.metadata().runtime_type == RuntimeType::External)
            .collect();

        let worker_urls: Vec<String> = external_workers
            .iter()
            .map(|w| w.url().to_string())
            .collect();

        let info = json!({
            "router_type": "openai",
            "total_workers": stats.total_workers,
            "external_workers": external_workers.len(),
            "healthy_workers": stats.healthy_workers,
            "total_models": stats.total_models,
            "worker_urls": worker_urls
        });
        (StatusCode::OK, info.to_string()).into_response()
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Return models from all registered external workers
        let external_workers: Vec<_> = self
            .worker_registry
            .get_all()
            .into_iter()
            .filter(|w| w.metadata().runtime_type == RuntimeType::External)
            .collect();

        if external_workers.is_empty() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "No external workers registered",
            )
                .into_response();
        }

        // Refresh models for all external workers using user's auth header
        let auth_header = extract_auth_header(Some(req.headers()), &None);
        self.refresh_external_models(auth_header.as_ref()).await;

        // Collect models from all workers
        let mut all_models = Vec::new();
        let mut seen_models = HashSet::new();

        for worker in &external_workers {
            for model_card in worker.models() {
                let owned_by = model_card
                    .provider
                    .as_ref()
                    .map(|p| format!("{:?}", p).to_lowercase())
                    .unwrap_or_else(|| "unknown".to_string());

                // Add primary model ID
                if seen_models.insert(model_card.id.clone()) {
                    all_models.push(json!({
                        "id": &model_card.id,
                        "object": "model",
                        "created": 0,
                        "owned_by": &owned_by,
                        "aliases": model_card.aliases,
                        "model_type": format!("{:?}", model_card.model_type),
                    }));
                }

                // Add aliases as separate entries for compatibility
                for alias in &model_card.aliases {
                    if seen_models.insert(alias.clone()) {
                        all_models.push(json!({
                            "id": alias,
                            "object": "model",
                            "created": 0,
                            "owned_by": &owned_by,
                            "primary_model": &model_card.id,
                        }));
                    }
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
        // Extract auth header for passthrough mode
        let auth_header = extract_auth_header(headers, &None);

        // Select worker for model (discovery happens inside if needed)
        let worker = match self
            .select_worker_for_model(body.model.as_str(), auth_header.as_ref())
            .await
        {
            Ok(w) => w,
            Err(response) => return *response,
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
            obj.retain(|k, _| !SGLANG_FIELDS.contains(&k.as_str()));
            // Remove logprobs if false (Gemini don't accept it)
            if obj.get("logprobs").and_then(|v| v.as_bool()) == Some(false) {
                obj.remove("logprobs");
            }
        }

        let url = format!("{}/v1/chat/completions", worker.url());
        let mut req = self.client.post(&url).json(&payload);

        // Apply provider-specific headers (handles Anthropic x-api-key, etc.)
        // Passthrough mode: user's auth header takes priority, worker's key is fallback
        let auth_header = extract_auth_header(headers, worker.api_key());
        req = apply_provider_headers(req, &url, auth_header.as_ref());

        // Accept SSE when stream=true
        if body.stream {
            req = req.header("Accept", "text/event-stream");
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                worker.circuit_breaker().record_failure();
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
                    worker.circuit_breaker().record_success();
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    if let Some(ct) = content_type {
                        response.headers_mut().insert(CONTENT_TYPE, ct);
                    }
                    response
                }
                Err(e) => {
                    worker.circuit_breaker().record_failure();
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
        // Extract auth header for passthrough mode
        let auth_header = extract_auth_header(headers, &None);

        // Select worker for model (discovery happens inside if needed)
        let model = model_id.unwrap_or(body.model.as_str());
        let worker = match self
            .select_worker_for_model(model, auth_header.as_ref())
            .await
        {
            Ok(w) => w,
            Err(response) => return *response,
        };

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
                        // Convert input items from stored input (which is now a JSON array)
                        if let Some(input_arr) = stored.input.as_array() {
                            for item in input_arr {
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.clone(),
                                ) {
                                    Ok(input_item) => {
                                        items.push(input_item);
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Failed to deserialize stored input item: {}. Item: {}",
                                            e, item
                                        );
                                    }
                                }
                            }
                        }

                        // Convert output items from stored output (which is now a JSON array)
                        if let Some(output_arr) = stored.output.as_array() {
                            for item in output_arr {
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.clone(),
                                ) {
                                    Ok(output_item) => {
                                        items.push(output_item);
                                    }
                                    Err(e) => {
                                        warn!("Failed to deserialize stored output item: {}. Item: {}", e, item);
                                    }
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
                        // Include messages, function calls, and function call outputs
                        // Skip reasoning items as they're internal processing details
                        match item.item_type.as_str() {
                            "message" => {
                                match serde_json::from_value::<Vec<ResponseContentPart>>(
                                    item.content.clone(),
                                ) {
                                    Ok(content_parts) => {
                                        items.push(ResponseInputOutputItem::Message {
                                            id: item.id.0.clone(),
                                            role: item
                                                .role
                                                .clone()
                                                .unwrap_or_else(|| "user".to_string()),
                                            content: content_parts,
                                            status: item.status.clone(),
                                        });
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Failed to deserialize message content: {}",
                                            e
                                        );
                                    }
                                }
                            }
                            "function_call" => {
                                // The entire function_call item is stored in content field
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.content.clone(),
                                ) {
                                    Ok(func_call) => items.push(func_call),
                                    Err(e) => {
                                        tracing::error!(
                                            "Failed to deserialize function_call: {}",
                                            e
                                        );
                                    }
                                }
                            }
                            "function_call_output" => {
                                // The entire function_call_output item is stored in content field
                                tracing::debug!(
                                    "Loading function_call_output from DB - content: {}",
                                    serde_json::to_string_pretty(&item.content)
                                        .unwrap_or_else(|_| "failed to serialize".to_string())
                                );
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.content.clone(),
                                ) {
                                    Ok(func_output) => {
                                        tracing::debug!(
                                            "Successfully deserialized function_call_output"
                                        );
                                        items.push(func_output);
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Failed to deserialize function_call_output: {}",
                                            e
                                        );
                                    }
                                }
                            }
                            "reasoning" => {
                                // Skip reasoning items - they're internal processing details
                            }
                            _ => {
                                // Skip unknown item types
                                warn!("Unknown item type in conversation: {}", item.item_type);
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
                            // Process all item types, converting SimpleInputMessage to Message
                            for item in current_items.iter() {
                                let normalized =
                                    crate::protocols::responses::normalize_input_item(item);
                                items.push(normalized);
                            }
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
                    // Process all item types, converting SimpleInputMessage to Message
                    for item in current_items.iter() {
                        let normalized = crate::protocols::responses::normalize_input_item(item);
                        items.push(normalized);
                    }
                }
            }

            request_body.input = ResponseInput::Items(items);
        }

        // Always set store=false for upstream (we store internally)
        request_body.store = Some(false);
        // Filter out reasoning items from input - they're internal processing details
        if let ResponseInput::Items(ref mut items) = request_body.input {
            items.retain(|item| !matches!(item, ResponseInputOutputItem::Reasoning { .. }));
        }

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
            obj.retain(|k, _| !SGLANG_FIELDS.contains(&k.as_str()));
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
        let url = format!("{}/v1/responses", worker.url());
        if body.stream.unwrap_or(false) {
            handle_streaming_response(
                &self.client,
                worker.circuit_breaker(),
                Some(&self.mcp_manager),
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
                &worker,
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

    async fn route_classify(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ClassifyRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Classify not supported").into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Rerank not supported").into_response()
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

    fn router_type(&self) -> &'static str {
        "openai"
    }
}
