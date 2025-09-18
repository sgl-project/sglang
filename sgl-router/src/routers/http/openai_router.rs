//! OpenAI router implementation

use crate::config::CircuitBreakerConfig;
use crate::core::{CircuitBreaker, CircuitBreakerConfig as CoreCircuitBreakerConfig};
use crate::data_connector::{ResponseId, SharedResponseStorage, StoredResponse};
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, ReasoningInfo,
    RerankRequest, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
    ResponseStatus, ResponseTextFormat, ResponsesGetParams, ResponsesRequest, ResponsesResponse,
    ResponsesUsage, TextFormatType, UsageInfo,
};
use crate::routers::header_utils::{apply_request_headers, preserve_response_headers};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
};
use futures_util::StreamExt;
use serde_json::{json, to_value, Value};
use std::{
    any::Any,
    convert::Infallible,
    sync::atomic::{AtomicBool, Ordering},
};
use tokio::sync::mpsc;
use tokio_stream::{iter, wrappers::ReceiverStream};
use tracing::{error, info, warn};
use uuid::Uuid;

/// Router for OpenAI backend
pub struct OpenAIRouter {
    /// HTTP client for upstream OpenAI-compatible API
    client: reqwest::Client,
    /// Base URL for identification (no trailing slash)
    base_url: String,
    /// Circuit breaker
    circuit_breaker: CircuitBreaker,
    /// Health status
    healthy: AtomicBool,
    /// Response storage for managing conversation history
    response_storage: SharedResponseStorage,
}

impl std::fmt::Debug for OpenAIRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIRouter")
            .field("base_url", &self.base_url)
            .field("healthy", &self.healthy)
            .finish()
    }
}

impl OpenAIRouter {
    /// Create a new OpenAI router
    pub async fn new(
        base_url: String,
        circuit_breaker_config: Option<CircuitBreakerConfig>,
        response_storage: SharedResponseStorage,
    ) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let base_url = base_url.trim_end_matches('/').to_string();

        // Convert circuit breaker config
        let core_cb_config = circuit_breaker_config
            .map(|cb| CoreCircuitBreakerConfig {
                failure_threshold: cb.failure_threshold,
                success_threshold: cb.success_threshold,
                timeout_duration: std::time::Duration::from_secs(cb.timeout_duration_secs),
                window_duration: std::time::Duration::from_secs(cb.window_duration_secs),
            })
            .unwrap_or_default();

        let circuit_breaker = CircuitBreaker::with_config(core_cb_config);

        Ok(Self {
            client,
            base_url,
            circuit_breaker,
            healthy: AtomicBool::new(true),
            response_storage,
        })
    }

    async fn handle_non_streaming_response(
        &self,
        url: String,
        headers: Option<&HeaderMap>,
        payload: Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> Response {
        let request_builder = self.client.post(&url).json(&payload);

        // Apply headers with filtering
        let request_builder = if let Some(headers) = headers {
            apply_request_headers(headers, request_builder, true)
        } else {
            request_builder
        };

        match request_builder.send().await {
            Ok(response) => {
                let status = response.status();

                if !status.is_success() {
                    let error_text = response
                        .text()
                        .await
                        .unwrap_or_else(|e| format!("Failed to get error body: {}", e));
                    return (status, error_text).into_response();
                }

                // Parse the response
                match response.json::<Value>().await {
                    Ok(openai_response_json) => {
                        // Build our complete response object
                        let complete_response = self.build_complete_response(
                            &openai_response_json,
                            original_body,
                            original_previous_response_id,
                        );

                        // Store the response internally
                        if let Err(e) = self
                            .store_response_internal(&complete_response, original_body)
                            .await
                        {
                            warn!("Failed to store response: {}", e);
                        }

                        // Return the complete response object
                        match serde_json::to_string(&complete_response) {
                            Ok(json_str) => (
                                StatusCode::OK,
                                [("content-type", "application/json")],
                                json_str,
                            )
                                .into_response(),
                            Err(e) => {
                                error!("Failed to serialize response: {}", e);
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    json!({"error": {"message": "Failed to serialize response", "type": "internal_error"}}).to_string(),
                                ).into_response()
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to parse OpenAI response: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to parse response: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                format!("Failed to forward request to OpenAI: {}", e),
            )
                .into_response(),
        }
    }

    async fn handle_streaming_response(
        &self,
        url: String,
        headers: Option<&HeaderMap>,
        payload: Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> Response {
        // Set up streaming request
        let mut request_builder = self.client.post(&url).json(&payload);

        // Apply headers
        if let Some(headers) = headers {
            request_builder = apply_request_headers(headers, request_builder, true);
        }

        // Make the request
        match request_builder.send().await {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    let error_text = response
                        .text()
                        .await
                        .unwrap_or_else(|e| format!("Failed to get error body: {}", e));
                    return (status, error_text).into_response();
                }

                // Set up channel for streaming
                let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(100);
                let response_storage = self.response_storage.clone();
                let body_clone = original_body.clone();
                let prev_response_id = original_previous_response_id.clone();

                // Spawn task to handle streaming response
                tokio::spawn(async move {
                    let mut accumulated_text = String::new();
                    let mut response_id = format!("resp_{}", Uuid::new_v4().simple());
                    let mut model_name = String::new();
                    let mut total_tokens = 0u32;

                    // Read the stream
                    let mut stream = response.bytes_stream();
                    let mut buffer = String::new();

                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(bytes) => {
                                buffer.push_str(&String::from_utf8_lossy(&bytes));

                                // Process complete SSE events
                                while let Some(event_end) = buffer.find("\n\n") {
                                    let event = buffer[..event_end].to_string();
                                    buffer = buffer[event_end + 2..].to_string();

                                    if let Some(data) = event.strip_prefix("data: ") {
                                        if data == "[DONE]" {
                                            // Send final response object
                                            let final_response =
                                                Self::build_streaming_final_response(
                                                    response_id.clone(),
                                                    model_name.clone(),
                                                    accumulated_text.clone(),
                                                    total_tokens,
                                                    &body_clone,
                                                    prev_response_id.clone(),
                                                );

                                            // Store the response
                                            Self::store_streaming_response(
                                                &response_storage,
                                                &final_response,
                                                &body_clone,
                                            )
                                            .await;

                                            // Send the final response
                                            let _ = tx
                                                .send(Ok(Event::default().data(
                                                    serde_json::to_string(&final_response)
                                                        .unwrap_or_default(),
                                                )))
                                                .await;
                                            let _ =
                                                tx.send(Ok(Event::default().data("[DONE]"))).await;
                                            break;
                                        }

                                        // Parse the delta
                                        if let Ok(delta_json) = serde_json::from_str::<Value>(data)
                                        {
                                            // Extract info from delta
                                            if let Some(id) =
                                                delta_json.get("id").and_then(|v| v.as_str())
                                            {
                                                response_id = id.to_string();
                                            }
                                            if let Some(model) =
                                                delta_json.get("model").and_then(|v| v.as_str())
                                            {
                                                model_name = model.to_string();
                                            }

                                            // Extract text delta
                                            if let Some(choices) =
                                                delta_json.get("choices").and_then(|v| v.as_array())
                                            {
                                                for choice in choices {
                                                    if let Some(delta) = choice.get("delta") {
                                                        if let Some(content) = delta
                                                            .get("content")
                                                            .and_then(|v| v.as_str())
                                                        {
                                                            accumulated_text.push_str(content);
                                                            total_tokens +=
                                                                content.split_whitespace().count()
                                                                    as u32; // Rough estimate

                                                            // Send incremental update
                                                            let incremental = json!({
                                                                "id": response_id,
                                                                "object": "response.delta",
                                                                "delta": {
                                                                    "content": content
                                                                }
                                                            });
                                                            let _ = tx
                                                                .send(Ok(Event::default()
                                                                    .data(incremental.to_string())))
                                                                .await;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Stream error: {}", e);
                                // Send error event
                                let error_event = json!({
                                    "error": {
                                        "message": format!("Stream error: {}", e),
                                        "type": "stream_error"
                                    }
                                });
                                let _ = tx
                                    .send(Ok(Event::default().data(error_event.to_string())))
                                    .await;
                                break;
                            }
                        }
                    }
                });

                // Return SSE stream
                let stream = ReceiverStream::new(rx);
                Sse::new(stream).into_response()
            }
            Err(e) => {
                error!("Failed to initiate streaming request: {}", e);
                (
                    StatusCode::BAD_GATEWAY,
                    format!("Failed to forward streaming request: {}", e),
                )
                    .into_response()
            }
        }
    }

    fn build_complete_response(
        &self,
        openai_response: &Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> ResponsesResponse {
        // Parse what we can from OpenAI response
        let id = openai_response
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or(&original_body.request_id)
            .to_string();

        let model = openai_response
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| original_body.model.as_deref().unwrap_or("gpt-4"))
            .to_string();

        let created_at = openai_response
            .get("created")
            .and_then(|v| v.as_i64())
            .unwrap_or_else(|| chrono::Utc::now().timestamp());

        // Parse output
        let mut output = Vec::new();
        if let Some(choices) = openai_response.get("choices").and_then(|v| v.as_array()) {
            for choice in choices {
                if let Some(message) = choice.get("message") {
                    let msg_id = format!("msg_{}", Uuid::new_v4().simple());
                    let content_text = message
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    output.push(ResponseOutputItem::Message {
                        id: msg_id,
                        role: "assistant".to_string(),
                        status: "completed".to_string(),
                        content: vec![ResponseContentPart::OutputText {
                            text: content_text,
                            annotations: vec![],
                            logprobs: None,
                        }],
                    });
                }
            }
        }

        // Parse usage
        let usage = openai_response.get("usage").map(|u| {
            let input_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let output_tokens = u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let total_tokens =
                u.get("total_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(input_tokens as u64 + output_tokens as u64) as u32;

            ResponsesUsage::Classic(UsageInfo {
                prompt_tokens: input_tokens,
                completion_tokens: output_tokens,
                total_tokens,
                reasoning_tokens: None,
                prompt_tokens_details: None,
            })
        });

        // Build complete response
        ResponsesResponse {
            id,
            object: "response".to_string(),
            created_at,
            status: ResponseStatus::Completed,
            error: None,
            incomplete_details: None,
            instructions: original_body.instructions.clone(),
            max_output_tokens: original_body.max_output_tokens,
            model,
            output,
            parallel_tool_calls: original_body.parallel_tool_calls,
            previous_response_id: original_previous_response_id,
            reasoning: original_body.reasoning.as_ref().map(|r| ReasoningInfo {
                effort: r.effort.as_ref().map(|e| format!("{:?}", e)),
                summary: None,
            }),
            store: original_body.store, // Return what the user requested
            temperature: original_body.temperature.or(Some(1.0)),
            text: Some(ResponseTextFormat {
                format: TextFormatType {
                    format_type: "text".to_string(),
                },
            }),
            tool_choice: match &original_body.tool_choice {
                crate::protocols::spec::ToolChoice::Value(v) => match v {
                    crate::protocols::spec::ToolChoiceValue::Auto => "auto".to_string(),
                    crate::protocols::spec::ToolChoiceValue::Required => "required".to_string(),
                    crate::protocols::spec::ToolChoiceValue::None => "none".to_string(),
                },
                crate::protocols::spec::ToolChoice::Function { .. } => "function".to_string(),
            },
            tools: original_body.tools.clone(),
            top_p: original_body.top_p.or(Some(1.0)),
            truncation: match &original_body.truncation {
                crate::protocols::spec::Truncation::Auto => Some("auto".to_string()),
                crate::protocols::spec::Truncation::Disabled => Some("disabled".to_string()),
            },
            usage,
            user: original_body.user.clone(),
            metadata: original_body.metadata.clone().unwrap_or_default(),
        }
    }

    fn build_streaming_final_response(
        response_id: String,
        model: String,
        text: String,
        estimated_tokens: u32,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> ResponsesResponse {
        let msg_id = format!("msg_{}", Uuid::new_v4().simple());

        ResponsesResponse {
            id: response_id,
            object: "response".to_string(),
            created_at: chrono::Utc::now().timestamp(),
            status: ResponseStatus::Completed,
            error: None,
            incomplete_details: None,
            instructions: original_body.instructions.clone(),
            max_output_tokens: original_body.max_output_tokens,
            model,
            output: vec![ResponseOutputItem::Message {
                id: msg_id,
                role: "assistant".to_string(),
                status: "completed".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text,
                    annotations: vec![],
                    logprobs: None,
                }],
            }],
            parallel_tool_calls: original_body.parallel_tool_calls,
            previous_response_id: original_previous_response_id,
            reasoning: original_body.reasoning.as_ref().map(|r| ReasoningInfo {
                effort: r.effort.as_ref().map(|e| format!("{:?}", e)),
                summary: None,
            }),
            store: original_body.store,
            temperature: original_body.temperature.or(Some(1.0)),
            text: Some(ResponseTextFormat {
                format: TextFormatType {
                    format_type: "text".to_string(),
                },
            }),
            tool_choice: "auto".to_string(),
            tools: original_body.tools.clone(),
            top_p: original_body.top_p.or(Some(1.0)),
            truncation: Some("disabled".to_string()),
            usage: Some(ResponsesUsage::Classic(UsageInfo {
                prompt_tokens: estimated_tokens / 2, // Rough estimate
                completion_tokens: estimated_tokens / 2,
                total_tokens: estimated_tokens,
                reasoning_tokens: None,
                prompt_tokens_details: None,
            })),
            user: original_body.user.clone(),
            metadata: original_body.metadata.clone().unwrap_or_default(),
        }
    }

    async fn store_response_internal(
        &self,
        response: &ResponsesResponse,
        original_body: &ResponsesRequest,
    ) -> Result<(), String> {
        let input_text = match &original_body.input {
            ResponseInput::Text(text) => text.clone(),
            ResponseInput::Items(_) => "complex input".to_string(),
        };

        let output_text = response
            .output
            .iter()
            .find_map(|item| {
                if let ResponseOutputItem::Message { content, .. } = item {
                    content.iter().find_map(|c| match c {
                        ResponseContentPart::OutputText { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let mut stored_response = StoredResponse::new(
            input_text,
            output_text,
            original_body
                .previous_response_id
                .as_ref()
                .map(|id| ResponseId::from_string(id.clone())),
        );

        stored_response.instructions = original_body.instructions.clone();
        stored_response.model = Some(response.model.clone());
        stored_response.user = original_body.user.clone();
        stored_response.metadata = original_body.metadata.clone().unwrap_or_default();
        stored_response.id = ResponseId::from_string(response.id.clone());
        stored_response.raw_response = serde_json::to_value(response).unwrap_or(Value::Null);

        self.response_storage
            .store_response(stored_response)
            .await
            .map_err(|e| format!("Failed to store response: {}", e))?;

        info!(response_id = %response.id, "Stored response locally");
        Ok(())
    }

    async fn store_streaming_response(
        response_storage: &SharedResponseStorage,
        response: &ResponsesResponse,
        original_body: &ResponsesRequest,
    ) {
        let input_text = match &original_body.input {
            ResponseInput::Text(text) => text.clone(),
            ResponseInput::Items(_) => "complex input".to_string(),
        };

        let output_text = response
            .output
            .iter()
            .find_map(|item| {
                if let ResponseOutputItem::Message { content, .. } = item {
                    content.iter().find_map(|c| match c {
                        ResponseContentPart::OutputText { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let mut stored_response = StoredResponse::new(
            input_text,
            output_text,
            original_body
                .previous_response_id
                .as_ref()
                .map(|id| ResponseId::from_string(id.clone())),
        );

        stored_response.instructions = original_body.instructions.clone();
        stored_response.model = Some(response.model.clone());
        stored_response.user = original_body.user.clone();
        stored_response.metadata = original_body.metadata.clone().unwrap_or_default();
        stored_response.id = ResponseId::from_string(response.id.clone());
        stored_response.raw_response = serde_json::to_value(response).unwrap_or(Value::Null);

        if let Err(e) = response_storage.store_response(stored_response).await {
            warn!(response_id = %response.id, error = %e, "Failed to store streaming response");
        } else {
            info!(response_id = %response.id, "Stored streaming response locally");
        }
    }
}

#[async_trait]
impl super::super::WorkerManagement for OpenAIRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        Err("Cannot add workers to OpenAI router".to_string())
    }

    fn remove_worker(&self, _worker_url: &str) {
        // No-op for OpenAI router
    }

    fn get_worker_urls(&self) -> Vec<String> {
        vec![self.base_url.clone()]
    }
}

#[async_trait]
impl super::super::RouterTrait for OpenAIRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        // Simple upstream probe: GET {base}/v1/models without auth
        let url = format!("{}/v1/models", self.base_url);
        match self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) => {
                let code = resp.status();
                // Treat success and auth-required as healthy (endpoint reachable)
                if code.is_success() || code.as_u16() == 401 || code.as_u16() == 403 {
                    (StatusCode::OK, "OK").into_response()
                } else {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("Upstream status: {}", code),
                    )
                        .into_response()
                }
            }
            Err(e) => (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Upstream error: {}", e),
            )
                .into_response(),
        }
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // For OpenAI, health_generate is the same as health
        self.health(_req).await
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        let info = json!({
            "router_type": "openai",
            "workers": 1,
            "base_url": &self.base_url
        });
        (StatusCode::OK, info.to_string()).into_response()
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Proxy to upstream /v1/models; forward Authorization header if provided
        let headers = req.headers();

        let mut upstream = self.client.get(format!("{}/v1/models", self.base_url));

        if let Some(auth) = headers
            .get("authorization")
            .or_else(|| headers.get("Authorization"))
        {
            upstream = upstream.header("Authorization", auth);
        }

        match upstream.send().await {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let content_type = res.headers().get(CONTENT_TYPE).cloned();
                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(axum::body::Body::from(body));
                        *response.status_mut() = status;
                        if let Some(ct) = content_type {
                            response.headers_mut().insert(CONTENT_TYPE, ct);
                        }
                        response
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read upstream response: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                format!("Failed to contact upstream: {}", e),
            )
                .into_response(),
        }
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
            ] {
                obj.remove(key);
            }
        }

        let url = format!("{}/v1/chat/completions", self.base_url);
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
                    let mut response = Response::new(axum::body::Body::from(body));
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
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
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
            let mut response = Response::new(Body::from_stream(
                tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
            ));
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
        let url = format!("{}/v1/responses", self.base_url);

        info!(
            requested_store = body.store,
            is_streaming = body.stream,
            "openai_responses_request"
        );

        // Clone the body and override model if needed
        let mut request_body = body.clone();
        if let Some(model) = model_id {
            request_body.model = Some(model.to_string());
        }

        // Store the original previous_response_id for the response
        let original_previous_response_id = request_body.previous_response_id.clone();

        // Handle previous_response_id by loading prior context
        let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;
        if let Some(prev_id_str) = request_body.previous_response_id.clone() {
            let prev_id = ResponseId::from_string(prev_id_str.clone());
            match self
                .response_storage
                .get_response_chain(&prev_id, None)
                .await
            {
                Ok(chain) => {
                    if !chain.responses.is_empty() {
                        let mut items = Vec::new();
                        for stored in chain.responses.iter() {
                            let trimmed_id = stored.id.0.trim_start_matches("resp_");
                            if !stored.input.is_empty() {
                                items.push(ResponseInputOutputItem::Message {
                                    id: format!("msg_u_{}", trimmed_id),
                                    role: "user".to_string(),
                                    status: Some("completed".to_string()),
                                    content: vec![ResponseContentPart::InputText {
                                        text: stored.input.clone(),
                                    }],
                                });
                            }
                            if !stored.output.is_empty() {
                                items.push(ResponseInputOutputItem::Message {
                                    id: format!("msg_a_{}", trimmed_id),
                                    role: "assistant".to_string(),
                                    status: Some("completed".to_string()),
                                    content: vec![ResponseContentPart::OutputText {
                                        text: stored.output.clone(),
                                        annotations: vec![],
                                        logprobs: None,
                                    }],
                                });
                            }
                        }
                        conversation_items = Some(items);
                    } else {
                        info!(previous_response_id = %prev_id_str, "previous chain empty");
                    }
                }
                Err(err) => {
                    warn!(previous_response_id = %prev_id_str, %err, "failed to fetch previous response chain");
                }
            }
            // Clear previous_response_id from request since we're converting to conversation
            request_body.previous_response_id = None;
        }

        if let Some(mut items) = conversation_items {
            match &request_body.input {
                ResponseInput::Text(text) => {
                    items.push(ResponseInputOutputItem::Message {
                        id: format!("msg_u_current_{}", items.len()),
                        role: "user".to_string(),
                        status: Some("completed".to_string()),
                        content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    });
                }
                ResponseInput::Items(existing) => {
                    items.extend(existing.clone());
                }
            }
            request_body.input = ResponseInput::Items(items);
        }

        // Always set store=false for OpenAI (we store internally)
        request_body.store = false;

        // Convert to JSON payload and strip SGLang-specific fields before forwarding
        let mut payload = match to_value(&request_body) {
            Ok(value) => value,
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to serialize responses request: {}", err),
                )
                    .into_response();
            }
        };
        if let Some(obj) = payload.as_object_mut() {
            for key in [
                "request_id",
                "priority",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "top_k",
                "min_p",
                "repetition_penalty",
            ] {
                obj.remove(key);
            }
        }

        // Check if streaming is requested
        if body.stream {
            // Handle streaming response
            self.handle_streaming_response(
                url,
                headers,
                payload,
                body,
                original_previous_response_id,
            )
            .await
        } else {
            // Handle non-streaming response
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
        params: &ResponsesGetParams,
    ) -> Response {
        let stored_id = ResponseId::from_string(response_id.to_string());
        if let Ok(Some(stored_response)) = self.response_storage.get_response(&stored_id).await {
            let stream_requested = params.stream.unwrap_or(false);
            let raw_value = stored_response.raw_response.clone();

            if !raw_value.is_null() {
                if stream_requested {
                    let event = Event::default().data(raw_value.to_string());
                    let done_event = Event::default().data("[DONE]");
                    let stream = iter(vec![
                        Ok::<Event, Infallible>(event),
                        Ok::<Event, Infallible>(done_event),
                    ]);
                    return Sse::new(stream).into_response();
                }

                return (
                    StatusCode::OK,
                    [("content-type", "application/json")],
                    raw_value.to_string(),
                )
                    .into_response();
            }

            let openai_response = ResponsesResponse {
                id: stored_response.id.0.clone(),
                object: "response".to_string(),
                created_at: stored_response.created_at.timestamp(),
                status: ResponseStatus::Completed,
                error: None,
                incomplete_details: None,
                instructions: stored_response.instructions.clone(),
                max_output_tokens: None,
                model: stored_response
                    .model
                    .unwrap_or_else(|| "gpt-4o".to_string()),
                output: vec![ResponseOutputItem::Message {
                    id: format!("msg_{}", stored_response.id.0),
                    role: "assistant".to_string(),
                    status: "completed".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: stored_response.output,
                        annotations: vec![],
                        logprobs: None,
                    }],
                }],
                parallel_tool_calls: true,
                previous_response_id: stored_response.previous_response_id.map(|id| id.0),
                reasoning: None,
                store: true,
                temperature: Some(1.0),
                text: Some(ResponseTextFormat {
                    format: TextFormatType {
                        format_type: "text".to_string(),
                    },
                }),
                tool_choice: "auto".to_string(),
                tools: vec![],
                top_p: Some(1.0),
                truncation: Some("disabled".to_string()),
                usage: None,
                user: stored_response.user.clone(),
                metadata: stored_response.metadata.clone(),
            };

            if stream_requested {
                if let Ok(value) = serde_json::to_value(&openai_response) {
                    let event = Event::default().data(value.to_string());
                    let done_event = Event::default().data("[DONE]");
                    let stream = iter(vec![
                        Ok::<Event, Infallible>(event),
                        Ok::<Event, Infallible>(done_event),
                    ]);
                    return Sse::new(stream).into_response();
                }
            }

            return (
                StatusCode::OK,
                [("content-type", "application/json")],
                serde_json::to_string(&openai_response).unwrap_or_else(|e| {
                    format!("{{\"error\": \"Failed to serialize response: {}\"}}", e)
                }),
            )
                .into_response();
        }

        (
            StatusCode::NOT_FOUND,
            format!(
                "Response with id '{}' not found in local storage",
                response_id
            ),
        )
            .into_response()
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        // Forward to OpenAI's cancel endpoint
        let url = format!("{}/v1/responses/{}/cancel", self.base_url, response_id);

        let request_builder = self.client.post(&url);

        // Apply headers with filtering (skip content headers for POST without body)
        let request_builder = if let Some(headers) = headers {
            apply_request_headers(headers, request_builder, true)
        } else {
            request_builder
        };

        match request_builder.send().await {
            Ok(response) => {
                let status = response.status();
                let headers = response.headers().clone();

                match response.text().await {
                    Ok(body_text) => {
                        let mut response = (status, body_text).into_response();
                        *response.headers_mut() = preserve_response_headers(&headers);
                        response
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read response body: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                format!("Failed to cancel response on OpenAI: {}", e),
            )
                .into_response(),
        }
    }

    async fn flush_cache(&self) -> Response {
        (
            StatusCode::FORBIDDEN,
            "flush_cache not supported for OpenAI router",
        )
            .into_response()
    }

    async fn get_worker_loads(&self) -> Response {
        (
            StatusCode::FORBIDDEN,
            "get_worker_loads not supported for OpenAI router",
        )
            .into_response()
    }

    fn router_type(&self) -> &'static str {
        "openai"
    }

    fn readiness(&self) -> Response {
        if self.healthy.load(Ordering::Acquire) && self.circuit_breaker.can_execute() {
            (StatusCode::OK, "Ready").into_response()
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "Not ready").into_response()
        }
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::FORBIDDEN,
            "Embeddings endpoint not supported for OpenAI backend",
        )
            .into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::FORBIDDEN,
            "Rerank endpoint not supported for OpenAI backend",
        )
            .into_response()
    }
}
