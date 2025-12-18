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
use serde_json::{json, to_value, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;

use super::{
    context::{
        ComponentRefs, PayloadState, RequestContext, ResponsesComponents, SharedComponents,
        WorkerSelection,
    },
    conversations::persist_conversation_items,
    mcp::{
        ensure_request_mcp_client, execute_tool_loop, prepare_mcp_payload_for_streaming,
        McpLoopConfig,
    },
    provider::ProviderRegistry,
    responses::{mask_tools_as_mcp, patch_streaming_response_json},
    streaming::handle_streaming_response,
};
use crate::{
    app_context::AppContext,
    core::{model_type::Endpoint, ModelCard, ProviderType, RuntimeType, Worker, WorkerRegistry},
    data_connector::{ConversationId, ListParams, ResponseId, SortOrder},
    protocols::{
        chat::ChatCompletionRequest,
        responses::{
            generate_id, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponsesGetParams, ResponsesRequest,
        },
    },
    routers::header_utils::{apply_provider_headers, extract_auth_header},
};

pub struct OpenAIRouter {
    worker_registry: Arc<WorkerRegistry>,
    provider_registry: ProviderRegistry,
    healthy: AtomicBool,
    shared_components: Arc<SharedComponents>,
    responses_components: Arc<ResponsesComponents>,
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

/// Error response helpers for consistent API error formatting
mod error_responses {
    use axum::{
        http::StatusCode,
        response::{IntoResponse, Response},
        Json,
    };
    use serde_json::json;

    pub fn bad_request(message: impl Into<String>) -> Response {
        (StatusCode::BAD_REQUEST, message.into()).into_response()
    }

    pub fn not_found(resource: &str, id: &str) -> Response {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("No {} found with id '{}'", resource, id),
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "not_found"
                }
            })),
        )
            .into_response()
    }

    pub fn internal_error(message: impl Into<String>) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": message.into(),
                    "type": "internal_error",
                    "param": null,
                    "code": "storage_error"
                }
            })),
        )
            .into_response()
    }

    pub fn service_unavailable(message: impl Into<String>) -> Response {
        (StatusCode::SERVICE_UNAVAILABLE, message.into()).into_response()
    }

    pub fn model_not_found(model: &str) -> Response {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("No worker available for model '{}'", model),
                    "type": "model_not_found",
                }
            })),
        )
            .into_response()
    }
}

impl OpenAIRouter {
    const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;

    /// Get all external workers from the registry
    fn external_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.worker_registry
            .get_all()
            .into_iter()
            .filter(|w| w.metadata().runtime_type == RuntimeType::External)
            .collect()
    }

    fn shared_components(&self) -> Arc<SharedComponents> {
        Arc::clone(&self.shared_components)
    }

    fn responses_components(&self) -> Arc<ResponsesComponents> {
        Arc::clone(&self.responses_components)
    }

    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        let worker_registry = ctx.worker_registry.clone();
        let mcp_manager = ctx
            .mcp_manager
            .get()
            .ok_or_else(|| "MCP manager not initialized in AppContext".to_string())?
            .clone();

        let shared_components = Arc::new(SharedComponents {
            client: ctx.client.clone(),
        });

        let responses_components = Arc::new(ResponsesComponents {
            shared: SharedComponents {
                client: ctx.client.clone(),
            },
            mcp_manager: mcp_manager.clone(),
            response_storage: ctx.response_storage.clone(),
            conversation_storage: ctx.conversation_storage.clone(),
            conversation_item_storage: ctx.conversation_item_storage.clone(),
        });

        Ok(Self {
            worker_registry,
            provider_registry: ProviderRegistry::new(),
            healthy: AtomicBool::new(true),
            shared_components,
            responses_components,
        })
    }

    fn get_provider_arc_for_worker(
        &self,
        worker: &dyn Worker,
        model_id: Option<&str>,
    ) -> Arc<dyn super::provider::Provider> {
        if let Some(model) = model_id {
            if let Some(pt) = worker.provider_for_model(model) {
                return self.provider_registry.get_arc(pt);
            }
            if let Some(pt) = ProviderType::from_model_name(model) {
                return self.provider_registry.get_arc(&pt);
            }
        }
        self.provider_registry.default_provider_arc()
    }

    async fn refresh_worker_models(
        &self,
        worker: &Arc<dyn Worker>,
        auth_header: Option<&HeaderValue>,
    ) -> bool {
        let url = format!("{}/v1/models", worker.url());
        let mut backend_req = self.shared_components.client.get(&url);
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

        let futures: Vec<_> = external_workers
            .iter()
            .map(|w| self.refresh_worker_models(w, auth_header))
            .collect();

        join_all(futures).await;
    }

    /// Find workers that can handle the given model and select the least loaded one
    fn find_best_worker_for_model(&self, model_id: &str) -> Option<Arc<dyn Worker>> {
        self.worker_registry
            .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
            .into_iter()
            .filter(|w| w.supports_model(model_id) && w.circuit_breaker().can_execute())
            .min_by_key(|w| w.load())
    }

    async fn select_worker_for_model(
        &self,
        model_id: &str,
        auth_header: Option<&HeaderValue>,
    ) -> Result<Arc<dyn Worker>, Response> {
        // Try to find a worker immediately
        if let Some(worker) = self.find_best_worker_for_model(model_id) {
            return Ok(worker);
        }

        // Refresh external models and try again
        tracing::debug!(
            "No worker found for model '{}', refreshing external worker models",
            model_id
        );
        self.refresh_external_models(auth_header).await;

        self.find_best_worker_for_model(model_id)
            .ok_or_else(|| error_responses::model_not_found(model_id))
    }

    /// Deserialize ResponseInputOutputItems from a JSON array value
    fn deserialize_items_from_array(array: &Value) -> Vec<ResponseInputOutputItem> {
        array
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        serde_json::from_value::<ResponseInputOutputItem>(item.clone())
                            .map_err(|e| warn!("Failed to deserialize item: {}. Item: {}", e, item))
                            .ok()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Append current request input to items list, creating a user message if needed
    fn append_current_input(
        items: &mut Vec<ResponseInputOutputItem>,
        input: &ResponseInput,
        id_suffix: &str,
    ) {
        match input {
            ResponseInput::Text(text) => {
                items.push(ResponseInputOutputItem::Message {
                    id: format!("msg_u_{}", id_suffix),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                });
            }
            ResponseInput::Items(current_items) => {
                for item in current_items {
                    items.push(crate::protocols::responses::normalize_input_item(item));
                }
            }
        }
    }

    async fn handle_non_streaming_response(&self, mut ctx: RequestContext) -> Response {
        let payload_state = ctx.take_payload().expect("Payload not prepared");
        let mut payload = payload_state.json;
        let url = payload_state.url;
        let previous_response_id = payload_state.previous_response_id;
        let original_body = ctx.responses_request();
        let worker = ctx.worker().expect("Worker not selected");
        let mcp_manager = ctx.components.mcp_manager().expect("MCP manager required");

        if let Some(ref tools) = original_body.tools {
            ensure_request_mcp_client(mcp_manager, tools.as_slice()).await;
        }

        let active_mcp = if mcp_manager.list_tools().is_empty() {
            None
        } else {
            Some(mcp_manager)
        };

        let mut response_json: Value;

        if let Some(mcp) = active_mcp {
            let config = McpLoopConfig::default();
            prepare_mcp_payload_for_streaming(&mut payload, mcp);

            match execute_tool_loop(
                ctx.components.client(),
                &url,
                ctx.headers(),
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
            let mut request_builder = ctx.components.client().post(&url).json(&payload);
            let auth_header = extract_auth_header(ctx.headers(), worker.api_key());
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

        mask_tools_as_mcp(&mut response_json, original_body);
        patch_streaming_response_json(
            &mut response_json,
            original_body,
            previous_response_id.as_deref(),
        );

        if let Err(err) = persist_conversation_items(
            ctx.components
                .conversation_storage()
                .expect("Conversation storage required")
                .clone(),
            ctx.components
                .conversation_item_storage()
                .expect("Conversation item storage required")
                .clone(),
            ctx.components
                .response_storage()
                .expect("Response storage required")
                .clone(),
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

#[async_trait::async_trait]
impl crate::routers::RouterTrait for OpenAIRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        let external_workers = self.external_workers();
        if external_workers.is_empty() {
            return error_responses::service_unavailable("No external workers registered");
        }

        let (healthy, unhealthy): (Vec<_>, Vec<_>) =
            external_workers.iter().partition(|w| w.is_healthy());

        if unhealthy.is_empty() {
            (
                StatusCode::OK,
                format!("OK - {} workers healthy", healthy.len()),
            )
                .into_response()
        } else {
            let unhealthy_info: Vec<_> = unhealthy
                .iter()
                .map(|w| format!("{} ({})", w.model_id(), w.url()))
                .collect();
            error_responses::service_unavailable(format!(
                "{}/{} workers unhealthy: {}",
                unhealthy.len(),
                external_workers.len(),
                unhealthy_info.join(", ")
            ))
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        let stats = self.worker_registry.stats();
        let external_workers = self.external_workers();
        let worker_urls: Vec<_> = external_workers.iter().map(|w| w.url()).collect();

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
        let external_workers = self.external_workers();
        if external_workers.is_empty() {
            return error_responses::service_unavailable("No external workers registered");
        }

        let auth_header = extract_auth_header(Some(req.headers()), &None);
        self.refresh_external_models(auth_header.as_ref()).await;

        let mut all_models = Vec::new();
        let mut seen_models = HashSet::new();

        for worker in &external_workers {
            for model_card in worker.models() {
                let owned_by = model_card
                    .provider
                    .as_ref()
                    .map(|p| format!("{:?}", p).to_lowercase())
                    .unwrap_or_else(|| "unknown".to_string());

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

        let response_json = json!({
            "object": "list",
            "data": all_models
        });

        (StatusCode::OK, Json(response_json)).into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let auth_header = extract_auth_header(headers, &None);

        let worker = match self
            .select_worker_for_model(body.model.as_str(), auth_header.as_ref())
            .await
        {
            Ok(w) => w,
            Err(response) => return response,
        };

        let mut payload = match to_value(body) {
            Ok(v) => v,
            Err(e) => {
                return error_responses::bad_request(format!("Failed to serialize request: {}", e))
            }
        };

        let provider = self.get_provider_arc_for_worker(worker.as_ref(), model_id);
        if let Err(e) = provider.transform_request(&mut payload, Endpoint::Chat) {
            return error_responses::bad_request(format!("Provider transform error: {}", e));
        }

        let mut ctx = RequestContext::for_chat(
            Arc::new(body.clone()),
            headers.cloned(),
            model_id.map(String::from),
            ComponentRefs::Shared(self.shared_components()),
        );

        ctx.state.worker = Some(WorkerSelection {
            worker: Arc::clone(&worker),
            provider,
        });

        let url = format!("{}/v1/chat/completions", worker.url());
        ctx.state.payload = Some(PayloadState {
            json: payload,
            url: url.clone(),
            previous_response_id: None,
        });

        let payload_ref = ctx.payload().expect("Payload not prepared");
        let mut req = ctx.components.client().post(&url).json(&payload_ref.json);
        let auth_header = extract_auth_header(ctx.headers(), worker.api_key());
        req = apply_provider_headers(req, &url, auth_header.as_ref());

        if ctx.is_streaming() {
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

        if !ctx.is_streaming() {
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

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let auth_header = extract_auth_header(headers, &None);

        let model = model_id.unwrap_or(body.model.as_str());
        let worker = match self
            .select_worker_for_model(model, auth_header.as_ref())
            .await
        {
            Ok(w) => w,
            Err(response) => return response,
        };

        let mut request_body = body.clone();
        if let Some(model) = model_id {
            request_body.model = model.to_string();
        }
        request_body.conversation = None;

        let original_previous_response_id = request_body.previous_response_id.clone();

        // Load items from previous response chain if specified
        let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;
        if let Some(prev_id_str) = request_body.previous_response_id.take() {
            let prev_id = ResponseId::from(prev_id_str.as_str());
            match self
                .responses_components
                .response_storage
                .get_response_chain(&prev_id, None)
                .await
            {
                Ok(chain) => {
                    let items: Vec<ResponseInputOutputItem> = chain
                        .responses
                        .iter()
                        .flat_map(|stored| {
                            Self::deserialize_items_from_array(&stored.input)
                                .into_iter()
                                .chain(Self::deserialize_items_from_array(&stored.output))
                        })
                        .collect();
                    conversation_items = Some(items);
                }
                Err(e) => {
                    warn!(
                        "Failed to load previous response chain for {}: {}",
                        prev_id_str, e
                    );
                }
            }
        }

        if let Some(conv_id_str) = body.conversation.clone() {
            let conv_id = ConversationId::from(conv_id_str.as_str());

            if let Ok(None) = self
                .responses_components
                .conversation_storage
                .get_conversation(&conv_id)
                .await
            {
                return error_responses::not_found("conversation", &conv_id.0);
            }

            let params = ListParams {
                limit: Self::MAX_CONVERSATION_HISTORY_ITEMS,
                order: SortOrder::Asc,
                after: None,
            };

            match self
                .responses_components
                .conversation_item_storage
                .list_items(&conv_id, params)
                .await
            {
                Ok(stored_items) => {
                    let mut items: Vec<ResponseInputOutputItem> = Vec::new();
                    for item in stored_items.into_iter() {
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
                            "reasoning" => {}
                            _ => {
                                warn!("Unknown item type in conversation: {}", item.item_type);
                            }
                        }
                    }

                    Self::append_current_input(&mut items, &request_body.input, &conv_id.0);
                    request_body.input = ResponseInput::Items(items);
                }
                Err(e) => {
                    warn!("Failed to load conversation history: {}", e);
                }
            }
        }

        // Apply previous response chain items if loaded
        if let Some(mut items) = conversation_items {
            let id_suffix = original_previous_response_id.as_deref().unwrap_or("new");
            Self::append_current_input(&mut items, &request_body.input, id_suffix);
            request_body.input = ResponseInput::Items(items);
        }

        request_body.store = Some(false);
        if let ResponseInput::Items(ref mut items) = request_body.input {
            items.retain(|item| !matches!(item, ResponseInputOutputItem::Reasoning { .. }));
        }

        let mut payload = match to_value(&request_body) {
            Ok(v) => v,
            Err(e) => {
                return error_responses::bad_request(format!("Failed to serialize request: {}", e))
            }
        };

        let provider = self.get_provider_arc_for_worker(worker.as_ref(), model_id);
        if let Err(e) = provider.transform_request(&mut payload, Endpoint::Responses) {
            return error_responses::bad_request(format!("Provider transform error: {}", e));
        }

        let mut ctx = RequestContext::for_responses(
            Arc::new(body.clone()),
            headers.cloned(),
            model_id.map(String::from),
            ComponentRefs::Responses(self.responses_components()),
        );

        ctx.state.worker = Some(WorkerSelection {
            worker: Arc::clone(&worker),
            provider: Arc::clone(&provider),
        });

        ctx.state.payload = Some(PayloadState {
            json: payload,
            url: format!("{}/v1/responses", worker.url()),
            previous_response_id: original_previous_response_id,
        });

        if ctx.is_streaming() {
            handle_streaming_response(ctx).await
        } else {
            self.handle_non_streaming_response(ctx).await
        }
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        let id = ResponseId::from(response_id);
        match self
            .responses_components
            .response_storage
            .get_response(&id)
            .await
        {
            Ok(Some(stored)) => {
                let mut response_json = stored.raw_response;
                if let Some(obj) = response_json.as_object_mut() {
                    obj.insert("id".to_string(), json!(id.0));
                }
                (StatusCode::OK, Json(response_json)).into_response()
            }
            Ok(None) => error_responses::not_found("response", response_id),
            Err(e) => error_responses::internal_error(format!("Failed to get response: {}", e)),
        }
    }

    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
    ) -> Response {
        let resp_id = ResponseId::from(response_id);

        match self
            .responses_components
            .response_storage
            .get_response(&resp_id)
            .await
        {
            Ok(Some(stored)) => {
                let items = stored.input.as_array().cloned().unwrap_or_default();

                let items_with_ids: Vec<Value> = items
                    .into_iter()
                    .map(|mut item| {
                        if item.get("id").is_none() {
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
            Ok(None) => error_responses::not_found("response", response_id),
            Err(e) => {
                warn!("Failed to retrieve input items for {}: {}", response_id, e);
                error_responses::internal_error(format!("Failed to retrieve input items: {}", e))
            }
        }
    }

    fn router_type(&self) -> &'static str {
        "openai"
    }
}
