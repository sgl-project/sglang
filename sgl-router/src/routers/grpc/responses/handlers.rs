//! Handler functions for /v1/responses endpoints
//!
//! # Public API
//!
//! - `route_responses()` - POST /v1/responses (main entry point)
//! - `get_response_impl()` - GET /v1/responses/{response_id}
//! - `cancel_response_impl()` - POST /v1/responses/{response_id}/cancel
//!
//! # Architecture
//!
//! This module orchestrates all request handling for the /v1/responses endpoint.
//! It supports three execution modes:
//!
//! 1. **Synchronous** - Returns complete response immediately
//! 2. **Background** - Returns queued response, executes in background task
//! 3. **Streaming** - Returns SSE stream with real-time events
//!
//! # Request Flow
//!
//! ```text
//! route_responses()
//!   ├─► route_responses_sync()       → route_responses_internal()
//!   ├─► route_responses_background() → spawn(route_responses_internal())
//!   └─► route_responses_streaming()  → convert_chat_stream_to_responses_stream()
//!
//! route_responses_internal()
//!   ├─► load_conversation_history()
//!   ├─► execute_tool_loop() (if MCP tools)
//!   │   └─► pipeline.execute_chat_for_responses() [loop]
//!   └─► execute_without_mcp() (if no MCP tools)
//!       └─► pipeline.execute_chat_for_responses()
//! ```

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
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::{
    conversions,
    streaming::ResponseStreamEventEmitter,
    tool_loop::{execute_tool_loop, execute_tool_loop_streaming},
    types::BackgroundTaskInfo,
};
use crate::{
    data_connector::{
        ConversationId, ConversationItemStorage, ConversationStorage, ResponseId, ResponseStorage,
    },
    protocols::{
        chat::ChatCompletionStreamResponse,
        responses::{
            ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
            ResponseStatus, ResponsesRequest, ResponsesResponse, ResponsesUsage,
        },
    },
    routers::openai::{conversations::persist_conversation_items, mcp::ensure_request_mcp_client},
};

// ============================================================================
// Main Request Handler
// ============================================================================

/// Main handler for POST /v1/responses
///
/// Validates request, determines execution mode (sync/async/streaming), and delegates
pub async fn route_responses(
    ctx: &super::context::ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
    // 0. Fast worker validation (fail-fast before expensive operations)
    let requested_model: Option<&str> = model_id.as_deref().or(Some(request.model.as_str()));

    if let Some(model) = requested_model {
        // Check if any workers support this model
        let available_models = ctx.worker_registry.get_models();

        if !available_models.contains(&model.to_string()) {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(json!({
                    "error": {
                        "message": format!(
                            "No workers available for model '{}'. Available models: {}",
                            model,
                            available_models.join(", ")
                        ),
                        "type": "service_unavailable",
                        "param": "model",
                        "code": "no_available_workers"
                    }
                })),
            )
                .into_response();
        }
    }

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
        route_responses_streaming(ctx, request, headers, model_id).await
    } else if is_background {
        route_responses_background(ctx, request, headers, model_id).await
    } else {
        route_responses_sync(ctx, request, headers, model_id, None).await
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
async fn route_responses_sync(
    ctx: &super::context::ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Response {
    match route_responses_internal(ctx, request, headers, model_id, response_id).await {
        Ok(responses_response) => axum::Json(responses_response).into_response(),
        Err(response) => response, // Already a Response with proper status code
    }
}

/// Internal implementation that returns Result for background task compatibility
async fn route_responses_internal(
    ctx: &super::context::ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Result<ResponsesResponse, Response> {
    // 1. Load conversation history and build modified request
    let modified_request = load_conversation_history(ctx, &request).await?;

    // 2. Check if request has MCP tools - if so, use tool loop
    let responses_response = if let Some(tools) = &request.tools {
        // Ensure dynamic MCP client is registered for request-scoped tools
        if ensure_request_mcp_client(&ctx.mcp_manager, tools)
            .await
            .is_some()
        {
            debug!("MCP tools detected, using tool loop");

            // Execute with MCP tool loop
            execute_tool_loop(
                ctx,
                modified_request,
                &request,
                headers,
                model_id,
                response_id.clone(),
            )
            .await?
        } else {
            debug!("Failed to create MCP client from request tools");
            // Fall through to non-MCP execution
            execute_without_mcp(
                ctx,
                &modified_request,
                &request,
                headers,
                model_id,
                response_id.clone(),
            )
            .await?
        }
    } else {
        // No tools, execute normally
        execute_without_mcp(
            ctx,
            &modified_request,
            &request,
            headers,
            model_id,
            response_id.clone(),
        )
        .await?
    };

    // 5. Persist response to storage if store=true
    if request.store.unwrap_or(true) {
        if let Ok(response_json) = serde_json::to_value(&responses_response) {
            if let Err(e) = persist_conversation_items(
                ctx.conversation_storage.clone(),
                ctx.conversation_item_storage.clone(),
                ctx.response_storage.clone(),
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
    ctx: &super::context::ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
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
        model: request.model.clone(),
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
        if let Err(e) = persist_conversation_items(
            ctx.conversation_storage.clone(),
            ctx.conversation_item_storage.clone(),
            ctx.response_storage.clone(),
            &response_json,
            &request,
        )
        .await
        {
            warn!("Failed to persist queued response: {}", e);
        }
    }

    // Spawn background task
    let ctx_clone = ctx.clone();
    let request_clone = request.clone();
    let headers_clone = headers.clone();
    let model_id_clone = model_id.clone();
    let response_id_clone = response_id.clone();

    let handle = tokio::task::spawn(async move {
        // Execute synchronously (set background=false to prevent recursion)
        let mut background_request = (*request_clone).clone();
        background_request.background = Some(false);

        match route_responses_internal(
            &ctx_clone,
            Arc::new(background_request),
            headers_clone,
            model_id_clone,
            Some(response_id_clone.clone()),
        )
        .await
        {
            Ok(_) => {
                debug!(
                    "Background response {} completed successfully",
                    response_id_clone
                );
            }
            Err(response) => {
                warn!(
                    "Background response {} failed with status {}",
                    response_id_clone,
                    response.status()
                );
            }
        }

        // Clean up task handle when done
        ctx_clone
            .background_tasks
            .write()
            .await
            .remove(&response_id_clone);
    });

    // Store task info for cancellation support
    ctx.background_tasks.write().await.insert(
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
    ctx: &super::context::ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
    // 1. Load conversation history
    let modified_request = match load_conversation_history(ctx, &request).await {
        Ok(req) => req,
        Err(response) => return response, // Already a Response with proper status code
    };

    // 2. Check if request has MCP tools - if so, use streaming tool loop
    if let Some(tools) = &request.tools {
        // Ensure dynamic MCP client is registered for request-scoped tools
        if ensure_request_mcp_client(&ctx.mcp_manager, tools)
            .await
            .is_some()
        {
            debug!("MCP tools detected in streaming mode, using streaming tool loop");

            return execute_tool_loop_streaming(ctx, modified_request, &request, headers, model_id)
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
    convert_chat_stream_to_responses_stream(ctx, chat_request, headers, model_id, &request).await
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
    ctx: &super::context::ResponsesContext,
    chat_request: Arc<crate::protocols::chat::ChatCompletionRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    original_request: &ResponsesRequest,
) -> Response {
    debug!("Converting chat SSE stream to responses SSE format");

    // Get chat streaming response
    let chat_response = ctx
        .pipeline
        .execute_chat(
            chat_request.clone(),
            headers,
            model_id,
            ctx.components.clone(),
        )
        .await;

    // Extract body and headers from chat response
    let (parts, body) = chat_response.into_parts();

    // Create channel for transformed SSE events
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // Spawn background task to transform stream
    let original_request_clone = original_request.clone();
    let chat_request_clone = chat_request.clone();
    let response_storage = ctx.response_storage.clone();
    let conversation_storage = ctx.conversation_storage.clone();
    let conversation_item_storage = ctx.conversation_item_storage.clone();

    tokio::spawn(async move {
        if let Err(e) = process_and_transform_sse_stream(
            body,
            original_request_clone,
            chat_request_clone,
            response_storage,
            conversation_storage,
            conversation_item_storage,
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
    response_storage: Arc<dyn ResponseStorage>,
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    // Create accumulator for final response
    let mut accumulator = StreamingResponseAccumulator::new(&original_request);

    // Create event emitter for OpenAI-compatible streaming
    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = original_request.model.clone();
    let created_at = chrono::Utc::now().timestamp() as u64;
    let mut event_emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);

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

                    // Process chunk through event emitter (emits proper OpenAI events)
                    event_emitter.process_chunk(&chat_chunk, &tx)?;
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

    // Emit final response.completed event with accumulated usage
    let usage_json = accumulator.usage.as_ref().map(|u| {
        let mut usage_obj = json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens
        });

        // Include reasoning_tokens if present
        if let Some(details) = &u.completion_tokens_details {
            if let Some(reasoning_tokens) = details.reasoning_tokens {
                usage_obj["completion_tokens_details"] = json!({
                    "reasoning_tokens": reasoning_tokens
                });
            }
        }

        usage_obj
    });

    let completed_event = event_emitter.emit_completed(usage_json.as_ref());
    event_emitter.send_event(&completed_event, &tx)?;

    // Finalize and persist accumulated response
    if original_request.store.unwrap_or(true) {
        let final_response = accumulator.finalize();

        if let Ok(response_json) = serde_json::to_value(&final_response) {
            if let Err(e) = persist_conversation_items(
                conversation_storage.clone(),
                conversation_item_storage.clone(),
                response_storage.clone(),
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

// ============================================================================
// Helper Functions
// ============================================================================

/// Execute request without MCP tool loop (simple pipeline execution)
async fn execute_without_mcp(
    ctx: &super::context::ResponsesContext,
    modified_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Result<ResponsesResponse, Response> {
    use crate::routers::grpc::utils;

    // Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = conversions::responses_to_chat(modified_request)
        .map_err(|e| utils::bad_request_error(format!("Failed to convert request: {}", e)))?;

    // Execute chat pipeline (errors already have proper HTTP status codes)
    let chat_response = ctx
        .pipeline
        .execute_chat_for_responses(
            Arc::new(chat_request),
            headers,
            model_id,
            ctx.components.clone(),
            response_id.clone(),
            Some(ctx.background_tasks.clone()),
        )
        .await?; // Preserve the Response error as-is

    // Convert ChatCompletionResponse → ResponsesResponse
    conversions::chat_to_responses(&chat_response, original_request, response_id).map_err(|e| {
        utils::internal_error_message(format!("Failed to convert to responses format: {}", e))
    })
}

/// Load conversation history and response chains, returning modified request
async fn load_conversation_history(
    ctx: &super::context::ResponsesContext,
    request: &ResponsesRequest,
) -> Result<ResponsesRequest, Response> {
    let mut modified_request = request.clone();
    let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;

    // Handle previous_response_id by loading response chain
    if let Some(ref prev_id_str) = modified_request.previous_response_id {
        let prev_id = ResponseId::from(prev_id_str.as_str());
        match ctx
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
                            match serde_json::from_value::<ResponseInputOutputItem>(item.clone()) {
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
                            match serde_json::from_value::<ResponseInputOutputItem>(item.clone()) {
                                Ok(output_item) => {
                                    items.push(output_item);
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to deserialize stored output item: {}. Item: {}",
                                        e, item
                                    );
                                }
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

        // Auto-create conversation if it doesn't exist (OpenAI behavior)
        if let Ok(None) = ctx.conversation_storage.get_conversation(&conv_id).await {
            debug!(
                "Creating new conversation with user-provided ID: {}",
                conv_id_str
            );

            // Convert HashMap to JsonMap for metadata
            let metadata = request.metadata.as_ref().map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<serde_json::Map<String, serde_json::Value>>()
            });

            let new_conv = crate::data_connector::NewConversation {
                id: Some(conv_id.clone()), // Use user-provided conversation ID
                metadata,
            };
            ctx.conversation_storage
                .create_conversation(new_conv)
                .await
                .map_err(|e| {
                    crate::routers::grpc::utils::internal_error_message(format!(
                        "Failed to create conversation: {}",
                        e
                    ))
                })?;
        }

        // Load conversation history
        const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;
        let params = crate::data_connector::ListParams {
            limit: MAX_CONVERSATION_HISTORY_ITEMS,
            order: crate::data_connector::SortOrder::Asc,
            after: None,
        };

        match ctx
            .conversation_item_storage
            .list_items(&conv_id, params)
            .await
        {
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
                        // Process all item types, converting SimpleInputMessage to Message
                        for item in current_items.iter() {
                            let normalized =
                                crate::protocols::responses::normalize_input_item(item);
                            items.push(normalized);
                        }
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
                // Process all item types, converting SimpleInputMessage to Message
                for item in current_items.iter() {
                    let normalized = crate::protocols::responses::normalize_input_item(item);
                    items.push(normalized);
                }
            }
        }

        modified_request.input = ResponseInput::Items(items);
    }

    Ok(modified_request)
}

// ============================================================================
// GET Response Implementation
// ============================================================================

/// Implementation for GET /v1/responses/{response_id}
pub async fn get_response_impl(
    ctx: &super::context::ResponsesContext,
    response_id: &str,
) -> Response {
    let resp_id = ResponseId::from(response_id);

    // Retrieve response from storage
    match ctx.response_storage.get_response(&resp_id).await {
        Ok(Some(stored_response)) => axum::Json(stored_response.raw_response).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({
                "error": {
                    "message": format!("Response with id '{}' not found", response_id),
                    "type": "not_found_error",
                    "code": "response_not_found"
                }
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({
                "error": {
                    "message": format!("Failed to retrieve response: {}", e),
                    "type": "internal_error"
                }
            })),
        )
            .into_response(),
    }
}

// ============================================================================
// CANCEL Response Implementation
// ============================================================================

/// Implementation for POST /v1/responses/{response_id}/cancel
pub async fn cancel_response_impl(
    ctx: &super::context::ResponsesContext,
    response_id: &str,
) -> Response {
    let resp_id = ResponseId::from(response_id);

    // Retrieve response from storage to check if it exists and get current status
    match ctx.response_storage.get_response(&resp_id).await {
        Ok(Some(stored_response)) => {
            // Check current status - only queued or in_progress responses can be cancelled
            let current_status = stored_response
                .raw_response
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            match current_status {
                "queued" | "in_progress" => {
                    // Attempt to abort the background task
                    let mut tasks = ctx.background_tasks.write().await;
                    if let Some(task_info) = tasks.remove(response_id) {
                        // Abort the Rust task immediately
                        task_info.handle.abort();

                        // Abort the Python/scheduler request via gRPC (if client is available)
                        let client_opt = task_info.client.read().await;
                        if let Some(ref client) = *client_opt {
                            if let Err(e) = client
                                .abort_request(
                                    task_info.grpc_request_id.clone(),
                                    "User cancelled via API".to_string(),
                                )
                                .await
                            {
                                warn!(
                                    "Failed to abort Python request {}: {}",
                                    task_info.grpc_request_id, e
                                );
                            } else {
                                debug!(
                                    "Successfully aborted Python request: {}",
                                    task_info.grpc_request_id
                                );
                            }
                        } else {
                            debug!("Client not yet available for abort, request may not have started yet");
                        }

                        // Task was found and aborted
                        (
                            StatusCode::OK,
                            axum::Json(json!({
                                "id": response_id,
                                "status": "cancelled",
                                "message": "Background task has been cancelled"
                            })),
                        )
                            .into_response()
                    } else {
                        // Task handle not found but status is queued/in_progress
                        // This can happen if: (1) task crashed, or (2) storage persistence failed
                        error!(
                            "Response {} has status '{}' but task handle is missing. Task may have crashed or storage update failed.",
                            response_id, current_status
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            axum::Json(json!({
                                "error": {
                                    "message": "Internal error: background task completed but failed to update status in storage",
                                    "type": "internal_error",
                                    "code": "status_update_failed"
                                }
                            })),
                        )
                            .into_response()
                    }
                }
                "completed" => (
                    StatusCode::BAD_REQUEST,
                    axum::Json(json!({
                        "error": {
                            "message": "Cannot cancel completed response",
                            "type": "invalid_request_error",
                            "code": "response_already_completed"
                        }
                    })),
                )
                    .into_response(),
                "failed" => (
                    StatusCode::BAD_REQUEST,
                    axum::Json(json!({
                        "error": {
                            "message": "Cannot cancel failed response",
                            "type": "invalid_request_error",
                            "code": "response_already_failed"
                        }
                    })),
                )
                    .into_response(),
                "cancelled" => (
                    StatusCode::OK,
                    axum::Json(json!({
                        "id": response_id,
                        "status": "cancelled",
                        "message": "Response was already cancelled"
                    })),
                )
                    .into_response(),
                _ => {
                    // Unknown status
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        axum::Json(json!({
                            "error": {
                                "message": format!("Unknown response status: {}", current_status),
                                "type": "internal_error"
                            }
                        })),
                    )
                        .into_response()
                }
            }
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({
                "error": {
                    "message": format!("Response with id '{}' not found", response_id),
                    "type": "not_found_error",
                    "code": "response_not_found"
                }
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({
                "error": {
                    "message": format!("Failed to retrieve response: {}", e),
                    "type": "internal_error"
                }
            })),
        )
            .into_response(),
    }
}
