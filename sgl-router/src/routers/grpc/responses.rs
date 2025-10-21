//! gRPC Router `/v1/responses` endpoint implementation
//!
//! This module handles all responses-specific logic including:
//! - Request validation
//! - Conversation history and response chain loading
//! - Background mode execution
//! - Streaming support
//! - MCP tool loop wrapper (future)
//! - Response persistence

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    http::{self, header, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use serde_json::json;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};
use tracing::{debug, warn};
use uuid::Uuid;

use super::{
    context::SharedComponents, conversions, pipeline::RequestPipeline, router::BackgroundTaskInfo,
};
use crate::{
    data_connector::{
        ConversationId, ResponseId, SharedConversationItemStorage, SharedConversationStorage,
        SharedResponseStorage,
    },
    mcp::McpClientManager,
    protocols::{
        chat::{ChatCompletionResponse, ChatCompletionStreamResponse},
        common::{FunctionCallResponse, ToolCall},
        responses::{
            ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
            ResponseStatus, ResponseTool, ResponseToolType, ResponsesRequest, ResponsesResponse,
            ResponsesUsage,
        },
    },
};

// ============================================================================
// Main Request Handler
// ============================================================================

/// Main handler for POST /v1/responses
///
/// Validates request, determines execution mode (sync/async/streaming), and delegates
#[allow(clippy::too_many_arguments)]
pub async fn route_responses(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    background_tasks: Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>,
) -> Response {
    // 1. Validate mutually exclusive parameters
    if request.previous_response_id.is_some() && request.conversation.is_some() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({
                "error": {
                    "message": "Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.",
                    "type": "invalid_request_error",
                    "param": serde_json::Value::Null,
                    "code": "mutually_exclusive_parameters"
                }
            })),
        )
            .into_response();
    }

    // 2. Check for incompatible parameter combinations
    let is_streaming = request.stream.unwrap_or(false);
    let is_background = request.background.unwrap_or(false);

    if is_streaming && is_background {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({
                "error": {
                    "message": "Cannot use streaming with background mode. Please set either 'stream' or 'background' to false.",
                    "type": "invalid_request_error",
                    "param": serde_json::Value::Null,
                    "code": "incompatible_parameters"
                }
            })),
        )
            .into_response();
    }

    // 3. Route based on execution mode
    if is_streaming {
        route_responses_streaming(
            pipeline,
            request,
            headers,
            model_id,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
        )
        .await
    } else if is_background {
        route_responses_background(
            pipeline,
            request,
            headers,
            model_id,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            background_tasks,
        )
        .await
    } else {
        route_responses_sync(
            pipeline,
            request,
            headers,
            model_id,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            None, // No response_id for sync
            None, // No background_tasks for sync
        )
        .await
    }
}

// ============================================================================
// Synchronous Execution
// ============================================================================

/// Execute synchronous responses request
///
/// This is the core execution path that:
/// 1. Loads conversation history / response chain
/// 2. Converts to ChatCompletionRequest
/// 3. Executes chat pipeline
/// 4. Converts back to ResponsesResponse
/// 5. Persists to storage
#[allow(clippy::too_many_arguments)]
async fn route_responses_sync(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    response_id: Option<String>,
    background_tasks: Option<Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>>,
) -> Response {
    match route_responses_internal(
        pipeline,
        request,
        headers,
        model_id,
        components,
        response_storage,
        conversation_storage,
        conversation_item_storage,
        response_id,
        background_tasks,
    )
    .await
    {
        Ok(responses_response) => axum::Json(responses_response).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({
                "error": {
                    "message": e,
                    "type": "internal_error"
                }
            })),
        )
            .into_response(),
    }
}

/// Internal implementation that returns Result for background task compatibility
#[allow(clippy::too_many_arguments)]
async fn route_responses_internal(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    response_id: Option<String>,
    background_tasks: Option<Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>>,
) -> Result<ResponsesResponse, String> {
    // 1. Load conversation history and build modified request
    let modified_request = load_conversation_history(
        &request,
        &response_storage,
        &conversation_storage,
        &conversation_item_storage,
    )
    .await?;

    // 2. Check if request has MCP tools - if so, use tool loop
    let responses_response = if let Some(tools) = &request.tools {
        if let Some(mcp_manager) = create_mcp_manager_from_request(tools).await {
            debug!("MCP tools detected, using tool loop");

            // Execute with MCP tool loop
            execute_tool_loop(
                pipeline,
                modified_request,
                &request,
                headers,
                model_id,
                components,
                mcp_manager,
                response_id.clone(),
                background_tasks,
            )
            .await?
        } else {
            // No MCP manager, execute normally
            execute_without_mcp(
                pipeline,
                &modified_request,
                &request,
                headers,
                model_id,
                components,
                response_id.clone(),
                background_tasks,
            )
            .await?
        }
    } else {
        // No tools, execute normally
        execute_without_mcp(
            pipeline,
            &modified_request,
            &request,
            headers,
            model_id,
            components,
            response_id.clone(),
            background_tasks,
        )
        .await?
    };

    // 5. Persist response to storage if store=true
    if request.store.unwrap_or(true) {
        if let Ok(response_json) = serde_json::to_value(&responses_response) {
            if let Err(e) = crate::routers::openai::conversations::persist_conversation_items(
                conversation_storage,
                conversation_item_storage,
                response_storage,
                &response_json,
                &request,
            )
            .await
            {
                warn!("Failed to persist response: {}", e);
            }
        }
    }

    Ok(responses_response)
}

// ============================================================================
// Background Mode Execution
// ============================================================================

/// Execute responses request in background mode
#[allow(clippy::too_many_arguments)]
async fn route_responses_background(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    background_tasks: Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>,
) -> Response {
    // Generate response_id for background tracking
    let response_id = format!("resp_{}", Uuid::new_v4());

    // Get current timestamp
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Create queued response
    let queued_response = ResponsesResponse {
        id: response_id.clone(),
        object: "response".to_string(),
        created_at,
        status: ResponseStatus::Queued,
        error: None,
        incomplete_details: None,
        instructions: request.instructions.clone(),
        max_output_tokens: request.max_output_tokens,
        model: request
            .model
            .clone()
            .unwrap_or_else(|| "default".to_string()),
        output: Vec::new(),
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        previous_response_id: request.previous_response_id.clone(),
        reasoning: None,
        store: request.store.unwrap_or(true),
        temperature: request.temperature,
        text: None,
        tool_choice: "auto".to_string(),
        tools: request.tools.clone().unwrap_or_default(),
        top_p: request.top_p,
        truncation: None,
        usage: None,
        user: request.user.clone(),
        metadata: request.metadata.clone().unwrap_or_default(),
    };

    // Persist queued response to storage
    if let Ok(response_json) = serde_json::to_value(&queued_response) {
        if let Err(e) = crate::routers::openai::conversations::persist_conversation_items(
            conversation_storage.clone(),
            conversation_item_storage.clone(),
            response_storage.clone(),
            &response_json,
            &request,
        )
        .await
        {
            warn!("Failed to persist queued response: {}", e);
        }
    }

    // Spawn background task
    let pipeline = pipeline.clone();
    let request_clone = request.clone();
    let headers_clone = headers.clone();
    let model_id_clone = model_id.clone();
    let components_clone = components.clone();
    let response_storage_clone = response_storage.clone();
    let conversation_storage_clone = conversation_storage.clone();
    let conversation_item_storage_clone = conversation_item_storage.clone();
    let response_id_clone = response_id.clone();
    let background_tasks_clone = background_tasks.clone();

    let handle = tokio::task::spawn(async move {
        // Execute synchronously (set background=false to prevent recursion)
        let mut background_request = (*request_clone).clone();
        background_request.background = Some(false);

        match route_responses_internal(
            &pipeline,
            Arc::new(background_request),
            headers_clone,
            model_id_clone,
            components_clone,
            response_storage_clone,
            conversation_storage_clone,
            conversation_item_storage_clone,
            Some(response_id_clone.clone()),
            Some(background_tasks_clone.clone()),
        )
        .await
        {
            Ok(_) => {
                debug!(
                    "Background response {} completed successfully",
                    response_id_clone
                );
            }
            Err(e) => {
                warn!("Background response {} failed: {}", response_id_clone, e);
            }
        }

        // Clean up task handle when done
        background_tasks_clone
            .write()
            .await
            .remove(&response_id_clone);
    });

    // Store task info for cancellation support
    background_tasks.write().await.insert(
        response_id.clone(),
        BackgroundTaskInfo {
            handle,
            grpc_request_id: String::new(), // Will be populated by pipeline at DispatchMetadataStage
            client: Arc::new(RwLock::new(None)),
        },
    );

    // Return queued response immediately
    axum::Json(queued_response).into_response()
}

// ============================================================================
// Streaming Mode Execution
// ============================================================================

/// Execute streaming responses request
#[allow(clippy::too_many_arguments)]
async fn route_responses_streaming(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
) -> Response {
    // 1. Load conversation history
    let modified_request = match load_conversation_history(
        &request,
        &response_storage,
        &conversation_storage,
        &conversation_item_storage,
    )
    .await
    {
        Ok(req) => req,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(json!({
                    "error": {
                        "message": e,
                        "type": "invalid_request_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // 2. Check if request has MCP tools - if so, use streaming tool loop
    if let Some(tools) = &request.tools {
        if let Some(mcp_manager) = create_mcp_manager_from_request(tools).await {
            debug!("MCP tools detected in streaming mode, using streaming tool loop");

            return execute_tool_loop_streaming(
                pipeline,
                modified_request,
                &request,
                headers,
                model_id,
                components,
                mcp_manager,
                response_storage,
                conversation_storage,
                conversation_item_storage,
            )
            .await;
        }
    }

    // 3. Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = match conversions::responses_to_chat(&modified_request) {
        Ok(req) => Arc::new(req),
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(json!({
                    "error": {
                        "message": format!("Failed to convert request: {}", e),
                        "type": "invalid_request_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // 4. Execute chat pipeline and convert streaming format (no MCP tools)
    convert_chat_stream_to_responses_stream(
        pipeline,
        chat_request,
        headers,
        model_id,
        components,
        &request,
        response_storage,
        conversation_storage,
        conversation_item_storage,
    )
    .await
}

/// Convert chat streaming response to responses streaming format
///
/// This function:
/// 1. Gets chat SSE stream from pipeline
/// 2. Intercepts and parses each SSE event
/// 3. Converts ChatCompletionStreamResponse → ResponsesResponse delta
/// 4. Accumulates response state for final persistence
/// 5. Emits transformed SSE events in responses format
#[allow(clippy::too_many_arguments)]
async fn convert_chat_stream_to_responses_stream(
    pipeline: &RequestPipeline,
    chat_request: Arc<crate::protocols::chat::ChatCompletionRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    original_request: &ResponsesRequest,
    response_storage: SharedResponseStorage,
    _conversation_storage: SharedConversationStorage,
    _conversation_item_storage: SharedConversationItemStorage,
) -> Response {
    debug!("Converting chat SSE stream to responses SSE format");

    // Get chat streaming response
    let chat_response = pipeline
        .execute_chat(chat_request.clone(), headers, model_id, components)
        .await;

    // Extract body and headers from chat response
    let (parts, body) = chat_response.into_parts();

    // Create channel for transformed SSE events
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // Spawn background task to transform stream
    let original_request_clone = original_request.clone();
    let chat_request_clone = chat_request.clone();
    let response_storage_clone = response_storage.clone();
    let conversation_storage_clone = _conversation_storage.clone();
    let conversation_item_storage_clone = _conversation_item_storage.clone();

    tokio::spawn(async move {
        if let Err(e) = process_and_transform_sse_stream(
            body,
            original_request_clone,
            chat_request_clone,
            response_storage_clone,
            conversation_storage_clone,
            conversation_item_storage_clone,
            tx.clone(),
        )
        .await
        {
            warn!("Error transforming SSE stream: {}", e);
            let error_event = json!({
                "error": {
                    "message": e,
                    "type": "stream_error"
                }
            });
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", error_event))));
        }

        // Send final [DONE] event
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    // Build SSE response with transformed stream
    let stream = UnboundedReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    let mut response = Response::builder().status(parts.status).body(body).unwrap();

    // Copy headers from original chat response
    *response.headers_mut() = parts.headers;

    // Ensure SSE headers are set
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        header::CONNECTION,
        header::HeaderValue::from_static("keep-alive"),
    );

    response
}

/// Process chat SSE stream and transform to responses format
async fn process_and_transform_sse_stream(
    body: Body,
    original_request: ResponsesRequest,
    _chat_request: Arc<crate::protocols::chat::ChatCompletionRequest>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    // Create accumulator for final response
    let mut accumulator = StreamingResponseAccumulator::new(&original_request);

    // Convert body to data stream
    let mut stream = body.into_data_stream();

    // Process stream chunks (each chunk is a complete SSE event)
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;

        // Convert chunk to string
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        // Check for end of stream
        if event == "data: [DONE]" {
            break;
        }

        // Parse SSE event (format: "data: {...}\n\n" or "data: {...}")
        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();

            // Try to parse as ChatCompletionStreamResponse
            match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                Ok(chat_chunk) => {
                    // Update accumulator
                    accumulator.process_chunk(&chat_chunk);

                    // Convert to responses delta format
                    let responses_delta = convert_chat_chunk_to_responses_delta(&chat_chunk);

                    // Emit converted SSE event
                    let delta_json = serde_json::to_string(&responses_delta)
                        .map_err(|e| format!("Failed to serialize delta: {}", e))?;

                    if tx
                        .send(Ok(Bytes::from(format!("data: {}\n\n", delta_json))))
                        .is_err()
                    {
                        return Err("Client disconnected".to_string());
                    }
                }
                Err(_) => {
                    // Not a valid chat chunk - might be error event, pass through
                    debug!("Non-chunk SSE event, passing through: {}", event);
                    if tx.send(Ok(Bytes::from(format!("{}\n\n", event)))).is_err() {
                        return Err("Client disconnected".to_string());
                    }
                }
            }
        }
    }

    // Finalize and persist accumulated response
    if original_request.store.unwrap_or(true) {
        let final_response = accumulator.finalize();

        if let Ok(response_json) = serde_json::to_value(&final_response) {
            if let Err(e) = crate::routers::openai::conversations::persist_conversation_items(
                conversation_storage,
                conversation_item_storage,
                response_storage,
                &response_json,
                &original_request,
            )
            .await
            {
                warn!("Failed to persist streaming response: {}", e);
            } else {
                debug!("Persisted streaming response: {}", final_response.id);
            }
        }
    }

    Ok(())
}

/// Response accumulator for streaming responses
struct StreamingResponseAccumulator {
    // Response metadata
    response_id: String,
    model: String,
    created_at: i64,

    // Accumulated content
    content_buffer: String,
    reasoning_buffer: String,
    tool_calls: Vec<ResponseOutputItem>,

    // Completion state
    finish_reason: Option<String>,
    usage: Option<crate::protocols::common::Usage>,

    // Original request for final response construction
    original_request: ResponsesRequest,
}

impl StreamingResponseAccumulator {
    fn new(original_request: &ResponsesRequest) -> Self {
        Self {
            response_id: String::new(),
            model: String::new(),
            created_at: 0,
            content_buffer: String::new(),
            reasoning_buffer: String::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
            original_request: original_request.clone(),
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        // Initialize metadata on first chunk
        if self.response_id.is_empty() {
            self.response_id = chunk.id.clone();
            self.model = chunk.model.clone();
            self.created_at = chunk.created as i64;
        }

        // Process first choice (responses API doesn't support n>1)
        if let Some(choice) = chunk.choices.first() {
            // Accumulate content
            if let Some(content) = &choice.delta.content {
                self.content_buffer.push_str(content);
            }

            // Accumulate reasoning
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_buffer.push_str(reasoning);
            }

            // Process tool call deltas
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    // Use index directly (it's a u32, not Option<u32>)
                    let index = delta.index as usize;

                    // Ensure we have enough tool calls
                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(ResponseOutputItem::FunctionToolCall {
                            id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            output: None,
                            status: "in_progress".to_string(),
                        });
                    }

                    // Update the tool call at this index
                    if let ResponseOutputItem::FunctionToolCall {
                        id,
                        name,
                        arguments,
                        ..
                    } = &mut self.tool_calls[index]
                    {
                        if let Some(delta_id) = &delta.id {
                            id.push_str(delta_id);
                        }
                        if let Some(function) = &delta.function {
                            if let Some(delta_name) = &function.name {
                                name.push_str(delta_name);
                            }
                            if let Some(delta_args) = &function.arguments {
                                arguments.push_str(delta_args);
                            }
                        }
                    }
                }
            }

            // Update finish reason
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

        // Update usage
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ResponsesResponse {
        let mut output: Vec<ResponseOutputItem> = Vec::new();

        // Add message content if present
        if !self.content_buffer.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", self.response_id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: self.content_buffer,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            });
        }

        // Add reasoning if present
        if !self.reasoning_buffer.is_empty() {
            output.push(ResponseOutputItem::Reasoning {
                id: format!("reasoning_{}", self.response_id),
                summary: vec![],
                content: vec![
                    crate::protocols::responses::ResponseReasoningContent::ReasoningText {
                        text: self.reasoning_buffer,
                    },
                ],
                status: Some("completed".to_string()),
            });
        }

        // Add tool calls
        output.extend(self.tool_calls);

        // Determine final status
        let status = match self.finish_reason.as_deref() {
            Some("stop") | Some("length") => ResponseStatus::Completed,
            Some("tool_calls") => ResponseStatus::InProgress,
            Some("failed") | Some("error") => ResponseStatus::Failed,
            _ => ResponseStatus::Completed,
        };

        // Convert usage
        let usage = self.usage.as_ref().map(|u| {
            let usage_info = crate::protocols::common::UsageInfo {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                prompt_tokens_details: None,
            };
            ResponsesUsage::Classic(usage_info)
        });

        ResponsesResponse {
            id: self.response_id,
            object: "response".to_string(),
            created_at: self.created_at,
            status,
            error: None,
            incomplete_details: None,
            instructions: self.original_request.instructions.clone(),
            max_output_tokens: self.original_request.max_output_tokens,
            model: self.model,
            output,
            parallel_tool_calls: self.original_request.parallel_tool_calls.unwrap_or(true),
            previous_response_id: self.original_request.previous_response_id.clone(),
            reasoning: None,
            store: self.original_request.store.unwrap_or(true),
            temperature: self.original_request.temperature,
            text: None,
            tool_choice: "auto".to_string(),
            tools: self.original_request.tools.clone().unwrap_or_default(),
            top_p: self.original_request.top_p,
            truncation: None,
            usage,
            user: None,
            metadata: self.original_request.metadata.clone().unwrap_or_default(),
        }
    }
}

/// Convert ChatCompletionStreamResponse to ResponsesResponse delta format
fn convert_chat_chunk_to_responses_delta(
    chunk: &ChatCompletionStreamResponse,
) -> serde_json::Value {
    let mut delta = json!({
        "id": chunk.id,
        "object": "response.delta",
        "created_at": chunk.created,
        "model": chunk.model,
    });

    if let Some(choice) = chunk.choices.first() {
        let mut output_deltas = Vec::new();

        // Content delta
        if let Some(content) = &choice.delta.content {
            output_deltas.push(json!({
                "type": "message",
                "delta": {
                    "content": [{
                        "type": "text",
                        "text": content
                    }]
                }
            }));
        }

        // Reasoning delta
        if let Some(reasoning) = &choice.delta.reasoning_content {
            output_deltas.push(json!({
                "type": "reasoning",
                "delta": {
                    "content": [{
                        "type": "text",
                        "text": reasoning
                    }]
                }
            }));
        }

        // Tool call deltas
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tool_call in tool_calls {
                let mut tool_delta = json!({
                    "type": "function_tool_call",
                    "index": tool_call.index,
                });

                let mut delta_fields = serde_json::Map::new();
                if let Some(id) = &tool_call.id {
                    delta_fields.insert("id".to_string(), json!(id));
                }
                if let Some(function) = &tool_call.function {
                    if let Some(name) = &function.name {
                        delta_fields.insert("name".to_string(), json!(name));
                    }
                    if let Some(args) = &function.arguments {
                        delta_fields.insert("arguments".to_string(), json!(args));
                    }
                }

                tool_delta["delta"] = json!(delta_fields);
                output_deltas.push(tool_delta);
            }
        }

        if !output_deltas.is_empty() {
            delta["output"] = json!(output_deltas);
        }

        // Finish reason
        if let Some(reason) = &choice.finish_reason {
            delta["status"] = json!(match reason.as_str() {
                "stop" | "length" => "completed",
                "tool_calls" => "in_progress",
                _ => "completed",
            });
        }
    }

    // Usage delta
    if let Some(usage) = &chunk.usage {
        delta["usage"] = json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        });
    }

    delta
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Load conversation history and response chains, returning modified request
async fn load_conversation_history(
    request: &ResponsesRequest,
    response_storage: &SharedResponseStorage,
    conversation_storage: &SharedConversationStorage,
    conversation_item_storage: &SharedConversationItemStorage,
) -> Result<ResponsesRequest, String> {
    let mut modified_request = request.clone();
    let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;

    // Handle previous_response_id by loading response chain
    if let Some(ref prev_id_str) = modified_request.previous_response_id {
        let prev_id = ResponseId::from(prev_id_str.as_str());
        match response_storage.get_response_chain(&prev_id, None).await {
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

                    // Convert output to conversation items
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
                modified_request.previous_response_id = None;
            }
            Err(e) => {
                warn!(
                    "Failed to load previous response chain for {}: {}",
                    prev_id_str, e
                );
            }
        }
    }

    // Handle conversation by loading conversation history
    if let Some(ref conv_id_str) = request.conversation {
        let conv_id = ConversationId::from(conv_id_str.as_str());

        // Verify conversation exists
        if let Ok(None) = conversation_storage.get_conversation(&conv_id).await {
            return Err("Conversation not found".to_string());
        }

        // Load conversation history
        const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;
        let params = crate::data_connector::conversation_items::ListParams {
            limit: MAX_CONVERSATION_HISTORY_ITEMS,
            order: crate::data_connector::conversation_items::SortOrder::Asc,
            after: None,
        };

        match conversation_item_storage.list_items(&conv_id, params).await {
            Ok(stored_items) => {
                let mut items: Vec<ResponseInputOutputItem> = Vec::new();
                for item in stored_items.into_iter() {
                    if item.item_type == "message" {
                        if let Ok(content_parts) =
                            serde_json::from_value::<Vec<ResponseContentPart>>(item.content.clone())
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
                match &modified_request.input {
                    ResponseInput::Text(text) => {
                        items.push(ResponseInputOutputItem::Message {
                            id: format!("msg_u_{}", conv_id.0),
                            role: "user".to_string(),
                            content: vec![ResponseContentPart::InputText { text: text.clone() }],
                            status: Some("completed".to_string()),
                        });
                    }
                    ResponseInput::Items(current_items) => {
                        items.extend_from_slice(current_items);
                    }
                }

                modified_request.input = ResponseInput::Items(items);
            }
            Err(e) => {
                warn!("Failed to load conversation history: {}", e);
            }
        }
    }

    // If we have conversation_items from previous_response_id, merge them
    if let Some(mut items) = conversation_items {
        // Append current request
        match &modified_request.input {
            ResponseInput::Text(text) => {
                items.push(ResponseInputOutputItem::Message {
                    id: format!(
                        "msg_u_{}",
                        request
                            .previous_response_id
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

        modified_request.input = ResponseInput::Items(items);
    }

    Ok(modified_request)
}

// ============================================================================
// MCP Tool Support
// ============================================================================

/// Build a request-scoped MCP manager from request tools, if present
async fn create_mcp_manager_from_request(tools: &[ResponseTool]) -> Option<Arc<McpClientManager>> {
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

    match McpClientManager::new(cfg).await {
        Ok(mgr) => Some(Arc::new(mgr)),
        Err(err) => {
            warn!("Failed to initialize request-scoped MCP manager: {}", err);
            None
        }
    }
}

/// Extract function call from a chat completion response
/// Returns (call_id, tool_name, arguments_json_str) if found
fn extract_function_call_from_chat(
    response: &ChatCompletionResponse,
) -> Option<(String, String, String)> {
    // Check if response has choices with tool calls
    let choice = response.choices.first()?;
    let message = &choice.message;

    // Look for tool_calls in the message
    if let Some(tool_calls) = &message.tool_calls {
        if let Some(tool_call) = tool_calls.first() {
            return Some((
                tool_call.id.clone(),
                tool_call.function.name.clone(),
                tool_call
                    .function
                    .arguments
                    .clone()
                    .unwrap_or_else(|| "{}".to_string()),
            ));
        }
    }

    None
}

/// Execute an MCP tool call
async fn execute_mcp_call(
    mcp_mgr: &Arc<McpClientManager>,
    tool_name: &str,
    args_json_str: &str,
) -> Result<String, String> {
    let args_value: serde_json::Value =
        serde_json::from_str(args_json_str).map_err(|e| format!("parse tool args: {}", e))?;
    let args_obj = args_value.as_object().cloned();

    let _server_name = mcp_mgr
        .get_tool(tool_name)
        .map(|t| t.server)
        .ok_or_else(|| format!("tool not found: {}", tool_name))?;

    let result = mcp_mgr
        .call_tool(tool_name, args_obj)
        .await
        .map_err(|e| format!("tool call failed: {}", e))?;

    let output_str = serde_json::to_string(&result)
        .map_err(|e| format!("Failed to serialize tool result: {}", e))?;
    Ok(output_str)
}

/// State for tracking multi-turn tool calling loop
struct ToolLoopState {
    iteration: usize,
    total_calls: usize,
    conversation_history: Vec<ResponseInputOutputItem>,
    original_input: ResponseInput,
    mcp_call_items: Vec<ResponseOutputItem>,
    server_label: String,
}

impl ToolLoopState {
    fn new(original_input: ResponseInput, server_label: String) -> Self {
        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
            mcp_call_items: Vec::new(),
            server_label,
        }
    }

    fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
        success: bool,
        error: Option<String>,
    ) {
        // Add function_tool_call item with both arguments and output
        self.conversation_history
            .push(ResponseInputOutputItem::FunctionToolCall {
                id: call_id.clone(),
                name: tool_name.clone(),
                arguments: args_json_str.clone(),
                output: Some(output_str.clone()),
                status: Some("completed".to_string()),
            });

        // Add mcp_call output item for metadata
        let mcp_call = build_mcp_call_item(
            &tool_name,
            &args_json_str,
            &output_str,
            &self.server_label,
            success,
            error.as_deref(),
        );
        self.mcp_call_items.push(mcp_call);
    }
}

// ============================================================================
// MCP Metadata Builders
// ============================================================================

use crate::protocols::responses::McpToolInfo;

/// Generate unique ID for MCP items
fn generate_mcp_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4())
}

/// Build mcp_list_tools output item
fn build_mcp_list_tools_item(
    mcp: &Arc<McpClientManager>,
    server_label: &str,
) -> ResponseOutputItem {
    let tools = mcp.list_tools();
    let tools_info: Vec<McpToolInfo> = tools
        .iter()
        .map(|t| McpToolInfo {
            name: t.name.clone(),
            description: Some(t.description.clone()),
            input_schema: t.parameters.clone().unwrap_or_else(|| {
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                })
            }),
            annotations: Some(json!({
                "read_only": false
            })),
        })
        .collect();

    ResponseOutputItem::McpListTools {
        id: generate_mcp_id("mcpl"),
        server_label: server_label.to_string(),
        tools: tools_info,
    }
}

/// Build mcp_call output item
fn build_mcp_call_item(
    tool_name: &str,
    arguments: &str,
    output: &str,
    server_label: &str,
    success: bool,
    error: Option<&str>,
) -> ResponseOutputItem {
    ResponseOutputItem::McpCall {
        id: generate_mcp_id("mcp"),
        status: if success { "completed" } else { "failed" }.to_string(),
        approval_request_id: None,
        arguments: arguments.to_string(),
        error: error.map(|e| e.to_string()),
        name: tool_name.to_string(),
        output: output.to_string(),
        server_label: server_label.to_string(),
    }
}

/// Execute request without MCP tool loop (simple pipeline execution)
#[allow(clippy::too_many_arguments)]
async fn execute_without_mcp(
    pipeline: &RequestPipeline,
    modified_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_id: Option<String>,
    background_tasks: Option<Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>>,
) -> Result<ResponsesResponse, String> {
    // Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = conversions::responses_to_chat(modified_request)
        .map_err(|e| format!("Failed to convert request: {}", e))?;

    // Execute chat pipeline
    let chat_response = pipeline
        .execute_chat_for_responses(
            Arc::new(chat_request),
            headers,
            model_id,
            components,
            response_id,
            background_tasks,
        )
        .await
        .map_err(|e| format!("Pipeline execution failed: {}", e))?;

    // Convert ChatCompletionResponse → ResponsesResponse
    conversions::chat_to_responses(&chat_response, original_request)
        .map_err(|e| format!("Failed to convert to responses format: {}", e))
}

/// Execute the MCP tool calling loop
///
/// This wraps pipeline.execute_chat_for_responses() in a loop that:
/// 1. Executes the chat pipeline
/// 2. Checks if response has tool calls
/// 3. If yes, executes MCP tools and builds resume request
/// 4. Repeats until no more tool calls or limit reached
#[allow(clippy::too_many_arguments)]
async fn execute_tool_loop(
    pipeline: &RequestPipeline,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    mcp_manager: Arc<McpClientManager>,
    response_id: Option<String>,
    background_tasks: Option<Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>>,
) -> Result<ResponsesResponse, String> {
    // Get server label from original request tools
    let server_label = original_request
        .tools
        .as_ref()
        .and_then(|tools| {
            tools
                .iter()
                .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                .and_then(|t| t.server_label.clone())
        })
        .unwrap_or_else(|| "request-mcp".to_string());

    let mut state = ToolLoopState::new(original_request.input.clone(), server_label.clone());

    // Configuration: max iterations as safety limit
    const MAX_ITERATIONS: usize = 10;
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    debug!(
        "Starting MCP tool loop: server_label={}, max_tool_calls={:?}, max_iterations={}",
        server_label, max_tool_calls, MAX_ITERATIONS
    );

    loop {
        // Convert to chat request
        let chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {}", e))?;

        // Execute chat pipeline
        let chat_response = pipeline
            .execute_chat_for_responses(
                Arc::new(chat_request),
                headers.clone(),
                model_id.clone(),
                components.clone(),
                response_id.clone(),
                background_tasks.clone(),
            )
            .await
            .map_err(|e| format!("Pipeline execution failed: {}", e))?;

        // Check for function calls
        if let Some((call_id, tool_name, args_json_str)) =
            extract_function_call_from_chat(&chat_response)
        {
            state.iteration += 1;
            state.total_calls += 1;

            debug!(
                "Tool loop iteration {}: calling {} (call_id: {})",
                state.iteration, tool_name, call_id
            );

            // Check combined limit
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(MAX_ITERATIONS),
                None => MAX_ITERATIONS,
            };

            if state.total_calls > effective_limit {
                warn!(
                    "Reached tool call limit: {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls, max_tool_calls, MAX_ITERATIONS
                );

                // Convert chat response to responses format and mark as incomplete
                let mut responses_response =
                    conversions::chat_to_responses(&chat_response, original_request)
                        .map_err(|e| format!("Failed to convert to responses format: {}", e))?;

                // Mark as completed but with incomplete details
                responses_response.status = ResponseStatus::Completed;
                responses_response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                return Ok(responses_response);
            }

            // Execute the MCP tool
            let (output_str, success, error) =
                match execute_mcp_call(&mcp_manager, &tool_name, &args_json_str).await {
                    Ok(output) => (output, true, None),
                    Err(err) => {
                        warn!("Tool execution failed: {}", err);
                        let error_msg = err.clone();
                        // Return error as output, let model decide how to proceed
                        let error_json = json!({ "error": err }).to_string();
                        (error_json, false, Some(error_msg))
                    }
                };

            // Record the call in state (includes MCP metadata)
            state.record_call(
                call_id,
                tool_name,
                args_json_str,
                output_str,
                success,
                error,
            );

            // Build resume request with conversation history
            // Start with original input
            let mut input_items = match &state.original_input {
                ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
                    id: format!("msg_u_{}", state.iteration),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                }],
                ResponseInput::Items(items) => items.clone(),
            };

            // Append all conversation history (function calls and outputs)
            input_items.extend_from_slice(&state.conversation_history);

            // Build new request for next iteration
            current_request = ResponsesRequest {
                input: ResponseInput::Items(input_items),
                model: current_request.model.clone(),
                instructions: current_request.instructions.clone(),
                tools: current_request.tools.clone(),
                max_output_tokens: current_request.max_output_tokens,
                temperature: current_request.temperature,
                top_p: current_request.top_p,
                stream: Some(false), // Always non-streaming in tool loop
                store: Some(false),  // Don't store intermediate responses
                background: Some(false),
                max_tool_calls: current_request.max_tool_calls,
                tool_choice: current_request.tool_choice.clone(),
                parallel_tool_calls: current_request.parallel_tool_calls,
                previous_response_id: None,
                conversation: None,
                user: current_request.user.clone(),
                metadata: current_request.metadata.clone(),
                // Additional fields from ResponsesRequest
                include: current_request.include.clone(),
                reasoning: current_request.reasoning.clone(),
                service_tier: current_request.service_tier.clone(),
                top_logprobs: current_request.top_logprobs,
                truncation: current_request.truncation.clone(),
                request_id: None,
                priority: current_request.priority,
                frequency_penalty: current_request.frequency_penalty,
                presence_penalty: current_request.presence_penalty,
                stop: current_request.stop.clone(),
                top_k: current_request.top_k,
                min_p: current_request.min_p,
                repetition_penalty: current_request.repetition_penalty,
            };

            // Continue to next iteration
        } else {
            // No more tool calls, we're done
            debug!(
                "Tool loop completed: {} iterations, {} total calls",
                state.iteration, state.total_calls
            );

            // Convert final chat response to responses format
            let mut responses_response =
                conversions::chat_to_responses(&chat_response, original_request)
                    .map_err(|e| format!("Failed to convert to responses format: {}", e))?;

            // Inject MCP metadata into output
            if state.total_calls > 0 {
                // Prepend mcp_list_tools item
                let mcp_list_tools = build_mcp_list_tools_item(&mcp_manager, &server_label);
                responses_response.output.insert(0, mcp_list_tools);

                // Append all mcp_call items at the end
                responses_response.output.extend(state.mcp_call_items);

                debug!(
                    "Injected MCP metadata: 1 mcp_list_tools + {} mcp_call items",
                    state.total_calls
                );
            }

            return Ok(responses_response);
        }
    }
}

/// Execute MCP tool loop with streaming support
///
/// This streams each iteration's response to the client while accumulating
/// to check for tool calls. If tool calls are found, executes them and
/// continues with the next streaming iteration.
#[allow(clippy::too_many_arguments)]
async fn execute_tool_loop_streaming(
    pipeline: &RequestPipeline,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    mcp_manager: Arc<McpClientManager>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
) -> Response {
    // Get server label
    let server_label = original_request
        .tools
        .as_ref()
        .and_then(|tools| {
            tools
                .iter()
                .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                .and_then(|t| t.server_label.clone())
        })
        .unwrap_or_else(|| "request-mcp".to_string());

    // Create SSE channel for client
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // Clone data for background task
    let pipeline_clone = pipeline.clone();
    let original_request_clone = original_request.clone();

    // Spawn background task for tool loop
    tokio::spawn(async move {
        let result = execute_tool_loop_streaming_internal(
            &pipeline_clone,
            current_request,
            &original_request_clone,
            headers,
            model_id,
            components,
            mcp_manager,
            server_label,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            tx.clone(),
        )
        .await;

        if let Err(e) = result {
            warn!("Streaming tool loop error: {}", e);
            let error_event = json!({
                "error": {
                    "message": e,
                    "type": "tool_loop_error"
                }
            });
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", error_event))));
        }

        // Send [DONE]
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    // Build SSE response
    let stream = UnboundedReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(body)
        .unwrap();

    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        header::CONNECTION,
        header::HeaderValue::from_static("keep-alive"),
    );

    response
}

/// Internal streaming tool loop implementation
#[allow(clippy::too_many_arguments)]
async fn execute_tool_loop_streaming_internal(
    pipeline: &RequestPipeline,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    mcp_manager: Arc<McpClientManager>,
    server_label: String,
    _response_storage: SharedResponseStorage,
    _conversation_storage: SharedConversationStorage,
    _conversation_item_storage: SharedConversationItemStorage,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    const MAX_ITERATIONS: usize = 20;
    let mut state = ToolLoopState::new(original_request.input.clone(), server_label.clone());
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    loop {
        state.iteration += 1;
        if state.iteration > MAX_ITERATIONS {
            return Err(format!(
                "Tool loop exceeded maximum iterations ({})",
                MAX_ITERATIONS
            ));
        }

        debug!("Streaming MCP tool loop iteration {}", state.iteration);

        // Convert to chat request
        let chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {}", e))?;

        // Execute chat streaming
        let response = pipeline
            .execute_chat(
                Arc::new(chat_request),
                headers.clone(),
                model_id.clone(),
                components.clone(),
            )
            .await;

        // Forward stream to client while accumulating for tool call detection
        let accumulated_response =
            forward_and_accumulate_stream(response.into_body(), tx.clone()).await?;

        // Check for tool calls
        if let Some((call_id, tool_name, args_json_str)) =
            extract_function_call_from_chat(&accumulated_response)
        {
            state.total_calls += 1;

            debug!(
                "Tool loop iteration {}: calling {} (call_id: {})",
                state.iteration, tool_name, call_id
            );

            // Check combined limit
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(MAX_ITERATIONS),
                None => MAX_ITERATIONS,
            };

            if state.total_calls > effective_limit {
                warn!(
                    "Reached tool call limit: {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls, max_tool_calls, MAX_ITERATIONS
                );
                break;
            }

            // Execute the MCP tool
            let (output_str, success, error) =
                match execute_mcp_call(&mcp_manager, &tool_name, &args_json_str).await {
                    Ok(output) => (output, true, None),
                    Err(err) => {
                        warn!("Tool execution failed: {}", err);
                        let error_msg = err.clone();
                        let error_json = json!({ "error": err }).to_string();
                        (error_json, false, Some(error_msg))
                    }
                };

            // Record the call in state
            state.record_call(
                call_id,
                tool_name,
                args_json_str,
                output_str,
                success,
                error,
            );

            // Build next request with conversation history
            let mut input_items = match &state.original_input {
                ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
                    id: format!("msg_u_{}", state.iteration),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                }],
                ResponseInput::Items(items) => items.clone(),
            };

            // Append all conversation history
            input_items.extend_from_slice(&state.conversation_history);

            // Build new request for next iteration
            current_request = ResponsesRequest {
                input: ResponseInput::Items(input_items),
                model: current_request.model.clone(),
                instructions: current_request.instructions.clone(),
                tools: current_request.tools.clone(),
                max_output_tokens: current_request.max_output_tokens,
                temperature: current_request.temperature,
                top_p: current_request.top_p,
                stream: Some(true), // Keep streaming enabled
                store: Some(false), // Don't store intermediate responses
                background: Some(false),
                max_tool_calls: current_request.max_tool_calls,
                tool_choice: current_request.tool_choice.clone(),
                parallel_tool_calls: current_request.parallel_tool_calls,
                previous_response_id: None,
                conversation: None,
                user: current_request.user.clone(),
                metadata: current_request.metadata.clone(),
                include: current_request.include.clone(),
                reasoning: current_request.reasoning.clone(),
                service_tier: current_request.service_tier.clone(),
                top_logprobs: current_request.top_logprobs,
                truncation: current_request.truncation.clone(),
                request_id: None,
                priority: current_request.priority,
                frequency_penalty: current_request.frequency_penalty,
                presence_penalty: current_request.presence_penalty,
                stop: current_request.stop.clone(),
                top_k: current_request.top_k,
                min_p: current_request.min_p,
                repetition_penalty: current_request.repetition_penalty,
            };

            continue;
        }

        // No tool calls, emit MCP metadata and finish
        debug!("No tool calls found, ending streaming MCP loop");

        // Emit MCP metadata as SSE events
        if state.total_calls > 0 {
            // Emit mcp_list_tools event
            let list_tools_item = build_mcp_list_tools_item(&mcp_manager, &server_label);
            let event_json = serde_json::to_string(&list_tools_item)
                .map_err(|e| format!("Failed to serialize mcp_list_tools: {}", e))?;
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", event_json))));

            // Emit mcp_call events
            for call_item in &state.mcp_call_items {
                let event_json = serde_json::to_string(&call_item)
                    .map_err(|e| format!("Failed to serialize mcp_call: {}", e))?;
                let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", event_json))));
            }

            debug!(
                "Injected MCP metadata: 1 mcp_list_tools + {} mcp_call items",
                state.total_calls
            );
        }

        break;
    }

    Ok(())
}

/// Forward SSE stream to client while accumulating for tool call detection
async fn forward_and_accumulate_stream(
    body: Body,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<ChatCompletionResponse, String> {
    let mut accumulator = ChatResponseAccumulator::new();
    let mut stream = body.into_data_stream();

    while let Some(chunk_result) = futures_util::StreamExt::next(&mut stream).await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;

        // Forward chunk to client
        tx.send(Ok(chunk.clone()))
            .map_err(|e| format!("Failed to forward chunk: {}", e))?;

        // Parse and accumulate
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        if event == "data: [DONE]" {
            break;
        }

        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();
            if let Ok(chat_chunk) = serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                accumulator.process_chunk(&chat_chunk);
            }
        }
    }

    Ok(accumulator.finalize())
}

/// Accumulates chat streaming chunks into complete ChatCompletionResponse
struct ChatResponseAccumulator {
    id: String,
    model: String,
    content: String,
    tool_calls: std::collections::HashMap<usize, ToolCall>,
    finish_reason: Option<String>,
}

impl ChatResponseAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            content: String::new(),
            tool_calls: std::collections::HashMap::new(),
            finish_reason: None,
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        if !chunk.id.is_empty() {
            self.id = chunk.id.clone();
        }
        if !chunk.model.is_empty() {
            self.model = chunk.model.clone();
        }

        if let Some(choice) = chunk.choices.first() {
            // Accumulate content
            if let Some(content) = &choice.delta.content {
                self.content.push_str(content);
            }

            // Accumulate tool calls
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    let index = delta.index as usize;
                    let entry = self.tool_calls.entry(index).or_insert_with(|| ToolCall {
                        id: String::new(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: String::new(),
                            arguments: Some(String::new()),
                        },
                    });

                    if let Some(id) = &delta.id {
                        entry.id = id.clone();
                    }
                    if let Some(function) = &delta.function {
                        if let Some(name) = &function.name {
                            entry.function.name = name.clone();
                        }
                        if let Some(args) = &function.arguments {
                            if let Some(ref mut existing_args) = entry.function.arguments {
                                existing_args.push_str(args);
                            }
                        }
                    }
                }
            }

            // Capture finish reason
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }
    }

    fn finalize(self) -> ChatCompletionResponse {
        let mut tool_calls_vec: Vec<_> = self.tool_calls.into_iter().collect();
        tool_calls_vec.sort_by_key(|(index, _)| *index);
        let tool_calls: Vec<_> = tool_calls_vec.into_iter().map(|(_, call)| call).collect();

        ChatCompletionResponse {
            id: self.id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: self.model,
            choices: vec![crate::protocols::chat::ChatChoice {
                index: 0,
                message: crate::protocols::chat::ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: if self.content.is_empty() {
                        None
                    } else {
                        Some(self.content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    reasoning_content: None,
                },
                finish_reason: self.finish_reason,
                logprobs: None,
                matched_stop: None,
                hidden_states: None,
            }],
            usage: None,
            system_fingerprint: None,
        }
    }
}
