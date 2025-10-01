//! OpenAI router implementation

use crate::config::CircuitBreakerConfig;
use crate::core::{CircuitBreaker, CircuitBreakerConfig as CoreCircuitBreakerConfig};
use crate::data_connector::{ResponseId, SharedResponseStorage, StoredResponse};
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, RerankRequest,
    ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
    ResponseStatus, ResponseTextFormat, ResponseTool, ResponseToolType, ResponsesGetParams,
    ResponsesRequest, ResponsesResponse, TextFormatType,
};
use crate::routers::header_utils::{apply_request_headers, preserve_response_headers};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, to_value, Value};
use std::{
    any::Any,
    borrow::Cow,
    collections::HashMap,
    io,
    sync::{atomic::AtomicBool, Arc},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, info, warn};

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
    /// Optional MCP manager (enabled via config presence)
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
}

impl std::fmt::Debug for OpenAIRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIRouter")
            .field("base_url", &self.base_url)
            .field("healthy", &self.healthy)
            .finish()
    }
}

/// Configuration for MCP tool calling loops
#[derive(Debug, Clone)]
struct McpLoopConfig {
    /// Maximum iterations as safety limit (internal only, default: 10)
    /// Prevents infinite loops when max_tool_calls is not set
    max_iterations: usize,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self { max_iterations: 10 }
    }
}

/// State for tracking multi-turn tool calling loop
struct ToolLoopState {
    /// Current iteration number (starts at 0, increments with each tool call)
    iteration: usize,
    /// Total number of tool calls executed
    total_calls: usize,
    /// Conversation history (function_call and function_call_output items)
    conversation_history: Vec<Value>,
    /// Original user input (preserved for building resume payloads)
    original_input: ResponseInput,
}

impl ToolLoopState {
    fn new(original_input: ResponseInput) -> Self {
        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
        }
    }

    /// Record a tool call in the loop state
    fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
    ) {
        // Add function_call item to history
        let func_item = json!({
            "type": "function_call",
            "call_id": call_id,
            "name": tool_name,
            "arguments": args_json_str
        });
        self.conversation_history.push(func_item);

        // Add function_call_output item to history
        let output_item = json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_str
        });
        self.conversation_history.push(output_item);
    }
}

/// Helper that parses SSE frames from the OpenAI responses stream and
/// accumulates enough information to persist the final response locally.
struct StreamingResponseAccumulator {
    /// The initial `response.created` payload (if emitted).
    initial_response: Option<Value>,
    /// The final `response.completed` payload (if emitted).
    completed_response: Option<Value>,
    /// Collected output items keyed by the upstream output index, used when
    /// a final response payload is absent and we need to synthesize one.
    output_items: Vec<(usize, Value)>,
    /// Captured error payload (if the upstream stream fails midway).
    encountered_error: Option<Value>,
}

impl StreamingResponseAccumulator {
    fn new() -> Self {
        Self {
            initial_response: None,
            completed_response: None,
            output_items: Vec::new(),
            encountered_error: None,
        }
    }

    /// Feed the accumulator with the next SSE chunk.
    fn ingest_block(&mut self, block: &str) {
        if block.trim().is_empty() {
            return;
        }
        self.process_block(block);
    }

    /// Consume the accumulator and produce the best-effort final response value.
    fn into_final_response(mut self) -> Option<Value> {
        if self.completed_response.is_some() {
            return self.completed_response;
        }

        self.build_fallback_response()
    }

    fn encountered_error(&self) -> Option<&Value> {
        self.encountered_error.as_ref()
    }
    fn process_block(&mut self, block: &str) {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            return;
        }

        let mut event_name: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();

        for line in trimmed.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                event_name = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim_start().to_string());
            }
        }

        let data_payload = data_lines.join("\n");
        if data_payload.is_empty() {
            return;
        }

        self.handle_event(event_name.as_deref(), &data_payload);
    }

    fn handle_event(&mut self, event_name: Option<&str>, data_payload: &str) {
        let parsed: Value = match serde_json::from_str(data_payload) {
            Ok(value) => value,
            Err(err) => {
                warn!("Failed to parse streaming event JSON: {}", err);
                return;
            }
        };

        let event_type = event_name
            .map(|s| s.to_string())
            .or_else(|| {
                parsed
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_default();

        match event_type.as_str() {
            "response.created" => {
                if let Some(response) = parsed.get("response") {
                    self.initial_response = Some(response.clone());
                }
            }
            "response.completed" => {
                if let Some(response) = parsed.get("response") {
                    self.completed_response = Some(response.clone());
                }
            }
            "response.output_item.done" => {
                if let (Some(index), Some(item)) = (
                    parsed
                        .get("output_index")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                    parsed.get("item"),
                ) {
                    self.output_items.push((index, item.clone()));
                }
            }
            "response.error" => {
                self.encountered_error = Some(parsed);
            }
            _ => {}
        }
    }

    fn build_fallback_response(&mut self) -> Option<Value> {
        let mut response = self.initial_response.clone()?;

        if let Some(obj) = response.as_object_mut() {
            obj.insert("status".to_string(), Value::String("completed".to_string()));

            self.output_items.sort_by_key(|(index, _)| *index);
            let outputs: Vec<Value> = self
                .output_items
                .iter()
                .map(|(_, item)| item.clone())
                .collect();
            obj.insert("output".to_string(), Value::Array(outputs));
        }

        Some(response)
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
            base_url,
            circuit_breaker,
            healthy: AtomicBool::new(true),
            response_storage,
            mcp_manager,
        })
    }

    async fn handle_non_streaming_response(
        &self,
        url: String,
        headers: Option<&HeaderMap>,
        mut payload: Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> Response {
        // Request-scoped MCP: build from request tools if provided; otherwise fall back to router-level MCP
        let req_mcp_manager = Self::mcp_manager_from_request_tools(&original_body.tools).await;
        let active_mcp = req_mcp_manager.as_ref().or(self.mcp_manager.as_ref());

        // If the client requested MCP but we couldn't initialize it, fail early with a clear error
        let requested_mcp = original_body
            .tools
            .iter()
            .any(|t| matches!(t.r#type, ResponseToolType::Mcp));
        if requested_mcp && active_mcp.is_none() {
            return (
                StatusCode::BAD_GATEWAY,
                json!({
                    "error": {
                        "message": "MCP server unavailable or failed to initialize from request tools",
                        "type": "mcp_unavailable",
                        "param": "tools",
                    }
                })
                .to_string(),
            )
                .into_response();
        }

        // If MCP is active, mirror one function tool into the outgoing payload
        if let Some(mcp) = active_mcp {
            if let Some(obj) = payload.as_object_mut() {
                // Remove any non-function tools (e.g., custom "mcp" items) from outgoing payload
                if let Some(v) = obj.get_mut("tools") {
                    if let Some(arr) = v.as_array_mut() {
                        arr.retain(|item| {
                            item.get("type")
                                .and_then(|v| v.as_str())
                                .map(|s| s == "function")
                                .unwrap_or(false)
                        });
                        if arr.is_empty() {
                            obj.remove("tools");
                            obj.insert(
                                "tool_choice".to_string(),
                                Value::String("none".to_string()),
                            );
                        }
                    }
                }
                // Build function tools for all discovered MCP tools
                let mut tools_json = Vec::new();
                let tools = mcp.list_tools();
                for t in tools {
                    let parameters = t.parameters.clone().unwrap_or(serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }));
                    let tool = serde_json::json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": parameters
                    });
                    tools_json.push(tool);
                }
                if !tools_json.is_empty() {
                    obj.insert("tools".to_string(), Value::Array(tools_json));
                    // Ensure tool_choice auto to allow model planning
                    obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
                }
            }
        }
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
                    Ok(mut openai_response_json) => {
                        if let Some(prev_id) = original_previous_response_id {
                            if let Some(obj) = openai_response_json.as_object_mut() {
                                let should_insert = obj
                                    .get("previous_response_id")
                                    .map(|v| v.is_null())
                                    .unwrap_or(true);
                                if should_insert {
                                    obj.insert(
                                        "previous_response_id".to_string(),
                                        Value::String(prev_id),
                                    );
                                }
                            }
                        }

                        if let Some(obj) = openai_response_json.as_object_mut() {
                            if !obj.contains_key("instructions") {
                                if let Some(instructions) = &original_body.instructions {
                                    obj.insert(
                                        "instructions".to_string(),
                                        Value::String(instructions.clone()),
                                    );
                                }
                            }

                            if !obj.contains_key("metadata") {
                                if let Some(metadata) = &original_body.metadata {
                                    let metadata_map: serde_json::Map<String, Value> = metadata
                                        .iter()
                                        .map(|(k, v)| (k.clone(), v.clone()))
                                        .collect();
                                    obj.insert("metadata".to_string(), Value::Object(metadata_map));
                                }
                            }

                            // Reflect the client's requested store preference in the response body
                            obj.insert("store".to_string(), Value::Bool(original_body.store));
                        }

                        // If MCP is active and we detect a function call, enter the tool loop
                        let mut final_response_json = if let Some(mcp) = active_mcp {
                            if Self::extract_function_call(&openai_response_json).is_some() {
                                // Use the loop to handle potentially multiple tool calls
                                let loop_config = McpLoopConfig::default();
                                match self
                                    .execute_tool_loop(
                                        &url,
                                        headers,
                                        payload.clone(),
                                        original_body,
                                        mcp,
                                        &loop_config,
                                    )
                                    .await
                                {
                                    Ok(loop_result) => loop_result,
                                    Err(err) => {
                                        warn!("Tool loop failed: {}", err);
                                        let error_body = json!({
                                            "error": {
                                                "message": format!("Tool loop failed: {}", err),
                                                "type": "internal_error",
                                            }
                                        })
                                        .to_string();
                                        return (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            [("content-type", "application/json")],
                                            error_body,
                                        )
                                            .into_response();
                                    }
                                }
                            } else {
                                // No function call detected, use response as-is
                                openai_response_json
                            }
                        } else {
                            openai_response_json
                        };

                        // Mask tools back to MCP format for client
                        Self::mask_tools_as_mcp(&mut final_response_json, original_body);
                        if original_body.store {
                            if let Err(e) = self
                                .store_response_internal(&final_response_json, original_body)
                                .await
                            {
                                warn!("Failed to store response: {}", e);
                            }
                        }

                        match serde_json::to_string(&final_response_json) {
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
                                )
                                    .into_response()
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

    /// Build a request-scoped MCP manager from request tools, if present.
    async fn mcp_manager_from_request_tools(
        tools: &[ResponseTool],
    ) -> Option<Arc<crate::mcp::McpClientManager>> {
        let tool = tools
            .iter()
            .find(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some())?;
        let server_url = tool.server_url.as_ref()?.trim().to_string();
        if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
            warn!(
                "Ignoring MCP server_url with unsupported scheme: {}",
                server_url
            );
            return None;
        }
        let name = tool
            .server_label
            .clone()
            .unwrap_or_else(|| "request-mcp".to_string());
        let token = tool.authorization.clone();
        let transport = if server_url.contains("/sse") {
            crate::mcp::McpTransport::Sse {
                url: server_url,
                token,
            }
        } else {
            crate::mcp::McpTransport::Streamable {
                url: server_url,
                token,
            }
        };
        let cfg = crate::mcp::McpConfig {
            servers: vec![crate::mcp::McpServerConfig { name, transport }],
        };
        match crate::mcp::McpClientManager::new(cfg).await {
            Ok(mgr) => Some(Arc::new(mgr)),
            Err(err) => {
                warn!("Failed to initialize request-scoped MCP manager: {}", err);
                None
            }
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
        let mut request_builder = self.client.post(&url).json(&payload);

        if let Some(headers) = headers {
            request_builder = apply_request_headers(headers, request_builder, true);
        }

        request_builder = request_builder.header("Accept", "text/event-stream");

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(err) => {
                self.circuit_breaker.record_failure();
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("Failed to forward request to OpenAI: {}", err),
                )
                    .into_response();
            }
        };

        let status = response.status();
        let status_code =
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !status.is_success() {
            self.circuit_breaker.record_failure();
            let error_body = match response.text().await {
                Ok(body) => body,
                Err(err) => format!("Failed to read upstream error body: {}", err),
            };
            return (status_code, error_body).into_response();
        }

        self.circuit_breaker.record_success();

        let preserved_headers = preserve_response_headers(response.headers());
        let mut upstream_stream = response.bytes_stream();

        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        let should_store = original_body.store;
        let storage = self.response_storage.clone();
        let original_request = original_body.clone();
        let previous_response_id = original_previous_response_id.clone();

        tokio::spawn(async move {
            let mut accumulator = StreamingResponseAccumulator::new();
            let mut upstream_failed = false;
            let mut receiver_connected = true;
            let mut pending = String::new();

            while let Some(chunk_result) = upstream_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_text = match std::str::from_utf8(&chunk) {
                            Ok(text) => Cow::Borrowed(text),
                            Err(_) => Cow::Owned(String::from_utf8_lossy(&chunk).to_string()),
                        };

                        pending.push_str(&chunk_text.replace("\r\n", "\n"));

                        while let Some(pos) = pending.find("\n\n") {
                            let raw_block = pending[..pos].to_string();
                            pending.drain(..pos + 2);

                            if raw_block.trim().is_empty() {
                                continue;
                            }

                            let block_cow = if let Some(modified) = Self::rewrite_streaming_block(
                                raw_block.as_str(),
                                &original_request,
                                previous_response_id.as_deref(),
                            ) {
                                Cow::Owned(modified)
                            } else {
                                Cow::Borrowed(raw_block.as_str())
                            };

                            if should_store {
                                accumulator.ingest_block(block_cow.as_ref());
                            }

                            if receiver_connected {
                                let chunk_to_send = format!("{}\n\n", block_cow);
                                if tx.send(Ok(Bytes::from(chunk_to_send))).is_err() {
                                    receiver_connected = false;
                                }
                            }

                            if !receiver_connected && !should_store {
                                break;
                            }
                        }

                        if !receiver_connected && !should_store {
                            break;
                        }
                    }
                    Err(err) => {
                        upstream_failed = true;
                        let io_err = io::Error::other(err);
                        let _ = tx.send(Err(io_err));
                        break;
                    }
                }
            }

            if should_store && !upstream_failed {
                if !pending.trim().is_empty() {
                    accumulator.ingest_block(&pending);
                }
                let encountered_error = accumulator.encountered_error().cloned();
                if let Some(mut response_json) = accumulator.into_final_response() {
                    Self::patch_streaming_response_json(
                        &mut response_json,
                        &original_request,
                        previous_response_id.as_deref(),
                    );

                    if let Err(err) =
                        Self::store_response_impl(&storage, &response_json, &original_request).await
                    {
                        warn!("Failed to store streaming response: {}", err);
                    }
                } else if let Some(error_payload) = encountered_error {
                    warn!("Upstream streaming error payload: {}", error_payload);
                } else {
                    warn!("Streaming completed without a final response payload");
                }
            }
        });

        let body_stream = UnboundedReceiverStream::new(rx);
        let mut response = Response::new(Body::from_stream(body_stream));
        *response.status_mut() = status_code;

        let headers_mut = response.headers_mut();
        for (name, value) in preserved_headers.iter() {
            headers_mut.insert(name, value.clone());
        }

        if !headers_mut.contains_key(CONTENT_TYPE) {
            headers_mut.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        }

        response
    }

    async fn store_response_internal(
        &self,
        response_json: &Value,
        original_body: &ResponsesRequest,
    ) -> Result<(), String> {
        if !original_body.store {
            return Ok(());
        }

        match Self::store_response_impl(&self.response_storage, response_json, original_body).await
        {
            Ok(response_id) => {
                info!(response_id = %response_id.0, "Stored response locally");
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    async fn store_response_impl(
        response_storage: &SharedResponseStorage,
        response_json: &Value,
        original_body: &ResponsesRequest,
    ) -> Result<ResponseId, String> {
        let input_text = match &original_body.input {
            ResponseInput::Text(text) => text.clone(),
            ResponseInput::Items(_) => "complex input".to_string(),
        };

        let output_text = Self::extract_primary_output_text(response_json).unwrap_or_default();

        let mut stored_response = StoredResponse::new(input_text, output_text, None);

        stored_response.instructions = response_json
            .get("instructions")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| original_body.instructions.clone());

        stored_response.model = response_json
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| original_body.model.clone());

        stored_response.user = response_json
            .get("user")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| original_body.user.clone());

        stored_response.metadata = response_json
            .get("metadata")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_else(|| original_body.metadata.clone().unwrap_or_default());

        stored_response.previous_response_id = response_json
            .get("previous_response_id")
            .and_then(|v| v.as_str())
            .map(|s| ResponseId::from_string(s.to_string()))
            .or_else(|| {
                original_body
                    .previous_response_id
                    .as_ref()
                    .map(|id| ResponseId::from_string(id.clone()))
            });

        if let Some(id_str) = response_json.get("id").and_then(|v| v.as_str()) {
            stored_response.id = ResponseId::from_string(id_str.to_string());
        }

        stored_response.raw_response = response_json.clone();

        response_storage
            .store_response(stored_response)
            .await
            .map_err(|e| format!("Failed to store response: {}", e))
    }

    fn patch_streaming_response_json(
        response_json: &mut Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<&str>,
    ) {
        if let Some(obj) = response_json.as_object_mut() {
            if let Some(prev_id) = original_previous_response_id {
                let should_insert = obj
                    .get("previous_response_id")
                    .map(|v| v.is_null() || v.as_str().map(|s| s.is_empty()).unwrap_or(false))
                    .unwrap_or(true);
                if should_insert {
                    obj.insert(
                        "previous_response_id".to_string(),
                        Value::String(prev_id.to_string()),
                    );
                }
            }

            if !obj.contains_key("instructions")
                || obj
                    .get("instructions")
                    .map(|v| v.is_null())
                    .unwrap_or(false)
            {
                if let Some(instructions) = &original_body.instructions {
                    obj.insert(
                        "instructions".to_string(),
                        Value::String(instructions.clone()),
                    );
                }
            }

            if !obj.contains_key("metadata")
                || obj.get("metadata").map(|v| v.is_null()).unwrap_or(false)
            {
                if let Some(metadata) = &original_body.metadata {
                    let metadata_map: serde_json::Map<String, Value> = metadata
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    obj.insert("metadata".to_string(), Value::Object(metadata_map));
                }
            }

            obj.insert("store".to_string(), Value::Bool(original_body.store));

            if obj
                .get("model")
                .and_then(|v| v.as_str())
                .map(|s| s.is_empty())
                .unwrap_or(true)
            {
                if let Some(model) = &original_body.model {
                    obj.insert("model".to_string(), Value::String(model.clone()));
                }
            }

            if obj.get("user").map(|v| v.is_null()).unwrap_or(false) {
                if let Some(user) = &original_body.user {
                    obj.insert("user".to_string(), Value::String(user.clone()));
                }
            }
        }
    }

    fn rewrite_streaming_block(
        block: &str,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<&str>,
    ) -> Option<String> {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut data_lines: Vec<String> = Vec::new();

        for line in trimmed.lines() {
            if line.starts_with("data:") {
                data_lines.push(line.trim_start_matches("data:").trim_start().to_string());
            }
        }

        if data_lines.is_empty() {
            return None;
        }

        let payload = data_lines.join("\n");
        let mut parsed: Value = match serde_json::from_str(&payload) {
            Ok(value) => value,
            Err(err) => {
                warn!("Failed to parse streaming JSON payload: {}", err);
                return None;
            }
        };

        let event_type = parsed
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let should_patch = matches!(
            event_type,
            "response.created" | "response.in_progress" | "response.completed"
        );

        if !should_patch {
            return None;
        }

        let mut changed = false;
        if let Some(response_obj) = parsed.get_mut("response").and_then(|v| v.as_object_mut()) {
            let desired_store = Value::Bool(original_body.store);
            if response_obj.get("store") != Some(&desired_store) {
                response_obj.insert("store".to_string(), desired_store);
                changed = true;
            }

            if let Some(prev_id) = original_previous_response_id {
                let needs_previous = response_obj
                    .get("previous_response_id")
                    .map(|v| v.is_null() || v.as_str().map(|s| s.is_empty()).unwrap_or(false))
                    .unwrap_or(true);

                if needs_previous {
                    response_obj.insert(
                        "previous_response_id".to_string(),
                        Value::String(prev_id.to_string()),
                    );
                    changed = true;
                }
            }
        }

        if !changed {
            return None;
        }

        let new_payload = match serde_json::to_string(&parsed) {
            Ok(json) => json,
            Err(err) => {
                warn!("Failed to serialize modified streaming payload: {}", err);
                return None;
            }
        };

        let mut rebuilt_lines = Vec::new();
        let mut data_written = false;
        for line in trimmed.lines() {
            if line.starts_with("data:") {
                if !data_written {
                    rebuilt_lines.push(format!("data: {}", new_payload));
                    data_written = true;
                }
            } else {
                rebuilt_lines.push(line.to_string());
            }
        }

        if !data_written {
            rebuilt_lines.push(format!("data: {}", new_payload));
        }

        Some(rebuilt_lines.join("\n"))
    }
    fn extract_primary_output_text(response_json: &Value) -> Option<String> {
        if let Some(items) = response_json.get("output").and_then(|v| v.as_array()) {
            for item in items {
                if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                    for part in content {
                        if part
                            .get("type")
                            .and_then(|v| v.as_str())
                            .map(|t| t == "output_text")
                            .unwrap_or(false)
                        {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                return Some(text.to_string());
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

impl OpenAIRouter {
    fn extract_function_call(resp: &Value) -> Option<(String, String, String)> {
        let output = resp.get("output")?.as_array()?;
        for item in output {
            let obj = item.as_object()?;
            let t = obj.get("type")?.as_str()?;
            if t == "function_tool_call" || t == "function_call" {
                let call_id = obj
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        obj.get("id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })?;
                let name = obj.get("name")?.as_str()?.to_string();
                let arguments = obj.get("arguments")?.as_str()?.to_string();
                return Some((call_id, name, arguments));
            }
        }
        None
    }

    /// Replace returned tools with the original request's MCP tool block (if present) so
    /// external clients see MCP semantics rather than internal function tools.
    fn mask_tools_as_mcp(resp: &mut Value, original_body: &ResponsesRequest) {
        let mcp_tool = original_body
            .tools
            .iter()
            .find(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some());
        let Some(t) = mcp_tool else {
            return;
        };

        let mut m = serde_json::Map::new();
        m.insert("type".to_string(), Value::String("mcp".to_string()));
        if let Some(label) = &t.server_label {
            m.insert("server_label".to_string(), Value::String(label.clone()));
        }
        if let Some(url) = &t.server_url {
            m.insert("server_url".to_string(), Value::String(url.clone()));
        }
        if let Some(desc) = &t.server_description {
            m.insert(
                "server_description".to_string(),
                Value::String(desc.clone()),
            );
        }
        if let Some(req) = &t.require_approval {
            m.insert("require_approval".to_string(), Value::String(req.clone()));
        }
        if let Some(allowed) = &t.allowed_tools {
            m.insert(
                "allowed_tools".to_string(),
                Value::Array(allowed.iter().map(|s| Value::String(s.clone())).collect()),
            );
        }

        if let Some(obj) = resp.as_object_mut() {
            obj.insert("tools".to_string(), Value::Array(vec![Value::Object(m)]));
            obj.entry("tool_choice")
                .or_insert(Value::String("auto".to_string()));
        }
    }

    async fn execute_mcp_call(
        mcp_mgr: &Arc<crate::mcp::McpClientManager>,
        tool_name: &str,
        args_json_str: &str,
    ) -> Result<(String, String), String> {
        let args_value: Value =
            serde_json::from_str(args_json_str).map_err(|e| format!("parse tool args: {}", e))?;
        let args_obj = args_value.as_object().cloned();

        let server_name = mcp_mgr
            .get_tool(tool_name)
            .map(|t| t.server)
            .ok_or_else(|| format!("tool not found: {}", tool_name))?;

        let result = mcp_mgr
            .call_tool(tool_name, args_obj)
            .await
            .map_err(|e| format!("tool call failed: {}", e))?;

        let output_str = serde_json::to_string(&result)
            .map_err(|e| format!("Failed to serialize tool result: {}", e))?;
        Ok((server_name, output_str))
    }

    /// Build a resume payload with conversation history
    fn build_resume_payload(
        base_payload: &Value,
        conversation_history: &[Value],
        original_input: &ResponseInput,
        tools_json: &Value,
    ) -> Result<Value, String> {
        // Clone the base payload which already has cleaned fields
        let mut payload = base_payload.clone();

        let obj = payload
            .as_object_mut()
            .ok_or_else(|| "payload not an object".to_string())?;

        // Build input array: start with original user input
        let mut input_array = Vec::new();

        // Add original user message
        // For structured input, serialize the original input items
        match original_input {
            ResponseInput::Text(text) => {
                let user_item = json!({
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": text }]
                });
                input_array.push(user_item);
            }
            ResponseInput::Items(items) => {
                // Items are already structured ResponseInputOutputItem, convert to JSON
                if let Ok(items_value) = serde_json::to_value(items) {
                    if let Some(items_arr) = items_value.as_array() {
                        input_array.extend_from_slice(items_arr);
                    }
                }
            }
        }

        // Add all conversation history (function calls and outputs)
        input_array.extend_from_slice(conversation_history);

        obj.insert("input".to_string(), Value::Array(input_array));

        // Use the transformed tools (function tools, not MCP tools)
        if let Some(tools_arr) = tools_json.as_array() {
            if !tools_arr.is_empty() {
                obj.insert("tools".to_string(), tools_json.clone());
            }
        }

        // Ensure non-streaming and no store to upstream
        obj.insert("stream".to_string(), Value::Bool(false));
        obj.insert("store".to_string(), Value::Bool(false));

        // Note: SGLang-specific fields were already removed from base_payload
        // before it was passed to execute_tool_loop (see route_responses lines 1935-1946)

        Ok(payload)
    }

    /// Helper function to build mcp_call items from executed tool calls in conversation history
    fn build_executed_mcp_call_items(
        conversation_history: &[Value],
        server_label: &str,
    ) -> Vec<Value> {
        let mut mcp_call_items = Vec::new();

        for item in conversation_history {
            if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");

                // Find corresponding output
                let output_item = conversation_history.iter().find(|o| {
                    o.get("type").and_then(|t| t.as_str()) == Some("function_call_output")
                        && o.get("call_id").and_then(|c| c.as_str()) == Some(call_id)
                });

                let output_str = output_item
                    .and_then(|o| o.get("output").and_then(|v| v.as_str()))
                    .unwrap_or("{}");

                // Check if output contains error by parsing JSON
                let is_error = serde_json::from_str::<serde_json::Value>(output_str)
                    .map(|v| v.get("error").is_some())
                    .unwrap_or(false);

                let mcp_call_item = Self::build_mcp_call_item(
                    tool_name,
                    args,
                    output_str,
                    server_label,
                    !is_error,
                    if is_error {
                        Some("Tool execution failed")
                    } else {
                        None
                    },
                );
                mcp_call_items.push(mcp_call_item);
            }
        }

        mcp_call_items
    }

    /// Build an incomplete response when limits are exceeded
    fn build_incomplete_response(
        mut response: Value,
        state: ToolLoopState,
        reason: &str,
        active_mcp: &Arc<crate::mcp::McpClientManager>,
        original_body: &ResponsesRequest,
    ) -> Result<Value, String> {
        let obj = response
            .as_object_mut()
            .ok_or_else(|| "response not an object".to_string())?;

        // Set status to completed (not failed - partial success)
        obj.insert("status".to_string(), Value::String("completed".to_string()));

        // Set incomplete_details
        obj.insert(
            "incomplete_details".to_string(),
            json!({ "reason": reason }),
        );

        // Convert any function_call in output to mcp_call format
        if let Some(output_array) = obj.get_mut("output").and_then(|v| v.as_array_mut()) {
            let server_label = original_body
                .tools
                .iter()
                .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                .and_then(|t| t.server_label.as_deref())
                .unwrap_or("mcp");

            // Find any function_call items and convert them to mcp_call (incomplete)
            let mut mcp_call_items = Vec::new();
            for item in output_array.iter() {
                if item.get("type").and_then(|t| t.as_str()) == Some("function_tool_call") {
                    let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");

                    // Mark as incomplete - not executed
                    let mcp_call_item = Self::build_mcp_call_item(
                        tool_name,
                        args,
                        "", // No output - wasn't executed
                        server_label,
                        false, // Not successful
                        Some("Not executed - response stopped due to limit"),
                    );
                    mcp_call_items.push(mcp_call_item);
                }
            }

            // Add mcp_list_tools and executed mcp_call items at the beginning
            if state.total_calls > 0 || !mcp_call_items.is_empty() {
                let list_tools_item = Self::build_mcp_list_tools_item(active_mcp, server_label);
                output_array.insert(0, list_tools_item);

                // Add mcp_call items for executed calls using helper
                let executed_items =
                    Self::build_executed_mcp_call_items(&state.conversation_history, server_label);

                let mut insert_pos = 1;
                for item in executed_items {
                    output_array.insert(insert_pos, item);
                    insert_pos += 1;
                }

                // Add incomplete mcp_call items
                for item in mcp_call_items {
                    output_array.insert(insert_pos, item);
                    insert_pos += 1;
                }
            }
        }

        // Add warning to metadata
        if let Some(metadata_val) = obj.get_mut("metadata") {
            if let Some(metadata_obj) = metadata_val.as_object_mut() {
                if let Some(mcp_val) = metadata_obj.get_mut("mcp") {
                    if let Some(mcp_obj) = mcp_val.as_object_mut() {
                        mcp_obj.insert(
                            "truncation_warning".to_string(),
                            Value::String(format!(
                                "Loop terminated at {} iterations, {} total calls (reason: {})",
                                state.iteration, state.total_calls, reason
                            )),
                        );
                    }
                }
            }
        }

        Ok(response)
    }

    /// Execute the tool calling loop
    async fn execute_tool_loop(
        &self,
        url: &str,
        headers: Option<&HeaderMap>,
        initial_payload: Value,
        original_body: &ResponsesRequest,
        active_mcp: &Arc<crate::mcp::McpClientManager>,
        config: &McpLoopConfig,
    ) -> Result<Value, String> {
        let mut state = ToolLoopState::new(original_body.input.clone());

        // Get max_tool_calls from request (None means no user-specified limit)
        let max_tool_calls = original_body.max_tool_calls.map(|n| n as usize);

        // Keep initial_payload as base template (already has fields cleaned)
        let base_payload = initial_payload.clone();
        let tools_json = base_payload.get("tools").cloned().unwrap_or(json!([]));
        let mut current_payload = initial_payload;

        info!(
            "Starting tool loop: max_tool_calls={:?}, max_iterations={}",
            max_tool_calls, config.max_iterations
        );

        loop {
            // Make request to upstream
            let request_builder = self.client.post(url).json(&current_payload);
            let request_builder = if let Some(headers) = headers {
                apply_request_headers(headers, request_builder, true)
            } else {
                request_builder
            };

            let response = request_builder
                .send()
                .await
                .map_err(|e| format!("upstream request failed: {}", e))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(format!("upstream error {}: {}", status, body));
            }

            let mut response_json = response
                .json::<Value>()
                .await
                .map_err(|e| format!("parse response: {}", e))?;

            // Check for function call
            if let Some((call_id, tool_name, args_json_str)) =
                Self::extract_function_call(&response_json)
            {
                state.iteration += 1;
                state.total_calls += 1;

                info!(
                    "Tool loop iteration {}: calling {} (call_id: {})",
                    state.iteration, tool_name, call_id
                );

                // Check combined limit: use minimum of user's max_tool_calls (if set) and safety max_iterations
                let effective_limit = match max_tool_calls {
                    Some(user_max) => user_max.min(config.max_iterations),
                    None => config.max_iterations,
                };

                if state.total_calls > effective_limit {
                    if let Some(user_max) = max_tool_calls {
                        if state.total_calls > user_max {
                            warn!("Reached user-specified max_tool_calls limit: {}", user_max);
                        } else {
                            warn!(
                                "Reached safety max_iterations limit: {}",
                                config.max_iterations
                            );
                        }
                    } else {
                        warn!(
                            "Reached safety max_iterations limit: {}",
                            config.max_iterations
                        );
                    }

                    return Self::build_incomplete_response(
                        response_json,
                        state,
                        "max_tool_calls",
                        active_mcp,
                        original_body,
                    );
                }

                // Execute tool
                let call_result =
                    Self::execute_mcp_call(active_mcp, &tool_name, &args_json_str).await;

                let output_str = match call_result {
                    Ok((_, output)) => output,
                    Err(err) => {
                        warn!("Tool execution failed: {}", err);
                        // Return error as output, let model decide how to proceed
                        json!({ "error": err }).to_string()
                    }
                };

                // Record the call
                state.record_call(call_id, tool_name, args_json_str, output_str);

                // Build resume payload
                current_payload = Self::build_resume_payload(
                    &base_payload,
                    &state.conversation_history,
                    &state.original_input,
                    &tools_json,
                )?;
            } else {
                // No more tool calls, we're done
                info!(
                    "Tool loop completed: {} iterations, {} total calls",
                    state.iteration, state.total_calls
                );

                // Inject MCP output items if we executed any tools
                if state.total_calls > 0 {
                    let server_label = original_body
                        .tools
                        .iter()
                        .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                        .and_then(|t| t.server_label.as_deref())
                        .unwrap_or("mcp");

                    // Build mcp_list_tools item
                    let list_tools_item = Self::build_mcp_list_tools_item(active_mcp, server_label);

                    // Insert at beginning of output array
                    if let Some(output_array) = response_json
                        .get_mut("output")
                        .and_then(|v| v.as_array_mut())
                    {
                        output_array.insert(0, list_tools_item);

                        // Build mcp_call items using helper function
                        let mcp_call_items = Self::build_executed_mcp_call_items(
                            &state.conversation_history,
                            server_label,
                        );

                        // Insert mcp_call items after mcp_list_tools using mutable position
                        let mut insert_pos = 1;
                        for item in mcp_call_items {
                            output_array.insert(insert_pos, item);
                            insert_pos += 1;
                        }
                    }
                }

                return Ok(response_json);
            }
        }
    }

    /// Generate a unique ID for MCP output items (similar to OpenAI format)
    fn generate_mcp_id(prefix: &str) -> String {
        use rand::RngCore;
        let mut rng = rand::rng();
        let mut bytes = [0u8; 30];
        rng.fill_bytes(&mut bytes);
        let hex_string: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        format!("{}_{}", prefix, hex_string)
    }

    /// Build an mcp_list_tools output item
    fn build_mcp_list_tools_item(
        mcp: &Arc<crate::mcp::McpClientManager>,
        server_label: &str,
    ) -> Value {
        let tools = mcp.list_tools();
        let tools_json: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters.clone().unwrap_or_else(|| json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    })),
                    "annotations": {
                        "read_only": false
                    }
                })
            })
            .collect();

        json!({
            "id": Self::generate_mcp_id("mcpl"),
            "type": "mcp_list_tools",
            "server_label": server_label,
            "tools": tools_json
        })
    }

    /// Build an mcp_call output item
    fn build_mcp_call_item(
        tool_name: &str,
        arguments: &str,
        output: &str,
        server_label: &str,
        success: bool,
        error: Option<&str>,
    ) -> Value {
        json!({
            "id": Self::generate_mcp_id("mcp"),
            "type": "mcp_call",
            "status": if success { "completed" } else { "failed" },
            "approval_request_id": Value::Null,
            "arguments": arguments,
            "error": error,
            "name": tool_name,
            "output": output,
            "server_label": server_label
        })
    }
}

#[async_trait]
impl super::super::RouterTrait for OpenAIRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
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
                "sampling_seed",
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
                    return (
                        StatusCode::NOT_IMPLEMENTED,
                        "Streaming retrieval not yet implemented",
                    )
                        .into_response();
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
                return (
                    StatusCode::NOT_IMPLEMENTED,
                    "Streaming retrieval not yet implemented",
                )
                    .into_response();
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

    fn router_type(&self) -> &'static str {
        "openai"
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
