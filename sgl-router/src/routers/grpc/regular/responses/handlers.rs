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
//! It supports two execution modes:
//!
//! 1. **Synchronous** - Returns complete response immediately
//! 2. **Streaming** - Returns SSE stream with real-time events
//!
//! Note: Background mode is no longer supported. Requests with background=true
//! will be rejected with a 400 error.
//!
//! # Request Flow
//!
//! ```text
//! route_responses()
//!   ├─► route_responses_sync()       → route_responses_internal()
//!   └─► route_responses_streaming()  → convert_chat_stream_to_responses_stream()
//!
//! route_responses_internal()
//!   ├─► load_conversation_history()
//!   ├─► execute_tool_loop() (if MCP tools)
//!   │   └─► pipeline.execute_chat_for_responses() [loop]
//!   └─► execute_without_mcp() (if no MCP tools)
//!       └─► pipeline.execute_chat_for_responses()
//! ```

use std::sync::Arc;

use axum::{
    body::Body,
    http::{self, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::{
    conversions,
    tool_loop::{execute_tool_loop, execute_tool_loop_streaming},
};
use crate::{
    data_connector::{
        self, ConversationId, ConversationItemStorage, ConversationStorage, ResponseId,
        ResponseStorage,
    },
    protocols::{
        chat::{self, ChatCompletionStreamResponse},
        common::{self},
        responses::{
            self, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
            ResponseReasoningContent, ResponseStatus, ResponsesRequest, ResponsesResponse,
            ResponsesUsage,
        },
    },
    routers::grpc::{
        common::responses::{
            build_sse_response, ensure_mcp_connection, persist_response_if_needed,
            streaming::ResponseStreamEventEmitter,
        },
        error,
    },
};

/// Main handler for POST /v1/responses
///
/// Validates request, determines execution mode (sync/async/streaming), and delegates
pub async fn route_responses(
    ctx: &super::context::ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
    // 1. Reject background mode (no longer supported)
    let is_background = request.background.unwrap_or(false);
    if is_background {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({
                "error": {
                    "message": "Background mode is not supported. Please set 'background' to false or omit it.",
                    "type": "invalid_request_error",
                    "param": "background",
                    "code": "unsupported_parameter"
                }
            })),
        )
            .into_response();
    }

    // 2. Route based on execution mode
    let is_streaming = request.stream.unwrap_or(false);
    if is_streaming {
        route_responses_streaming(ctx, request, headers, model_id).await
    } else {
        // Generate response ID for synchronous execution
        let response_id = Some(format!("resp_{}", Uuid::new_v4()));
        route_responses_sync(ctx, request, headers, model_id, response_id).await
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

    // 2. Check MCP connection and get whether MCP tools are present
    let has_mcp_tools = ensure_mcp_connection(&ctx.mcp_manager, request.tools.as_deref()).await?;

    let responses_response = if has_mcp_tools {
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
        // No MCP tools - execute without MCP (may have function tools or no tools)
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
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &responses_response,
        &request,
    )
    .await;

    Ok(responses_response)
}

/// Execute streaming responses request
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

    // 2. Check MCP connection and get whether MCP tools are present
    let has_mcp_tools =
        match ensure_mcp_connection(&ctx.mcp_manager, request.tools.as_deref()).await {
            Ok(has_mcp) => has_mcp,
            Err(response) => return response,
        };

    if has_mcp_tools {
        debug!("MCP tools detected in streaming mode, using streaming tool loop");

        return execute_tool_loop_streaming(ctx, modified_request, &request, headers, model_id)
            .await;
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
async fn convert_chat_stream_to_responses_stream(
    ctx: &super::context::ResponsesContext,
    chat_request: Arc<chat::ChatCompletionRequest>,
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

    // Extract body from chat response
    let (_parts, body) = chat_response.into_parts();

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
    build_sse_response(rx)
}

/// Process chat SSE stream and transform to responses format
async fn process_and_transform_sse_stream(
    body: Body,
    original_request: ResponsesRequest,
    _chat_request: Arc<chat::ChatCompletionRequest>,
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
    event_emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events
    let event = event_emitter.emit_created();
    event_emitter
        .send_event(&event, &tx)
        .map_err(|_| "Failed to send response.created event".to_string())?;

    let event = event_emitter.emit_in_progress();
    event_emitter
        .send_event(&event, &tx)
        .map_err(|_| "Failed to send response.in_progress event".to_string())?;

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
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens
        });

        // Include reasoning_tokens if present
        if let Some(details) = &u.completion_tokens_details {
            if let Some(reasoning_tokens) = details.reasoning_tokens {
                usage_obj["output_tokens_details"] =
                    json!({ "reasoning_tokens": reasoning_tokens });
            }
        }

        usage_obj
    });

    let completed_event = event_emitter.emit_completed(usage_json.as_ref());
    event_emitter.send_event(&completed_event, &tx)?;

    // Finalize and persist accumulated response
    let final_response = accumulator.finalize();
    persist_response_if_needed(
        conversation_storage,
        conversation_item_storage,
        response_storage,
        &final_response,
        &original_request,
    )
    .await;

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
    usage: Option<common::Usage>,

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
                            call_id: String::new(),
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
                content: vec![ResponseReasoningContent::ReasoningText {
                    text: self.reasoning_buffer,
                }],
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
            let usage_info = common::UsageInfo {
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

        ResponsesResponse::builder(&self.response_id, &self.model)
            .copy_from_request(&self.original_request)
            .created_at(self.created_at)
            .status(status)
            .output(output)
            .maybe_usage(usage)
            .build()
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
    // Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = conversions::responses_to_chat(modified_request).map_err(|e| {
        error!(
            function = "execute_without_mcp",
            error = %e,
            "Failed to convert ResponsesRequest to ChatCompletionRequest"
        );
        error::bad_request(format!("Failed to convert request: {}", e))
    })?;

    // Execute chat pipeline (errors already have proper HTTP status codes)
    let chat_response = ctx
        .pipeline
        .execute_chat_for_responses(
            Arc::new(chat_request),
            headers,
            model_id,
            ctx.components.clone(),
        )
        .await?; // Preserve the Response error as-is

    // Convert ChatCompletionResponse → ResponsesResponse
    conversions::chat_to_responses(&chat_response, original_request, response_id).map_err(|e| {
        error!(
            function = "execute_without_mcp",
            error = %e,
            "Failed to convert ChatCompletionResponse to ResponsesResponse"
        );
        error::internal_error(format!("Failed to convert to responses format: {}", e))
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

        // Check if conversation exists - return error if not found
        let conversation = ctx
            .conversation_storage
            .get_conversation(&conv_id)
            .await
            .map_err(|e| {
                error!(
                    function = "load_conversation_history",
                    conversation_id = %conv_id_str,
                    error = %e,
                    "Failed to check conversation existence in storage"
                );
                error::internal_error(format!("Failed to check conversation: {}", e))
            })?;

        if conversation.is_none() {
            return Err(error::not_found(format!(
                "Conversation '{}' not found. Please create the conversation first using the conversations API.",
                conv_id_str
            )));
        }

        // Load conversation history
        const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;
        let params = data_connector::ListParams {
            limit: MAX_CONVERSATION_HISTORY_ITEMS,
            order: data_connector::SortOrder::Asc,
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
                            let normalized = responses::normalize_input_item(item);
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
                    let normalized = responses::normalize_input_item(item);
                    items.push(normalized);
                }
            }
        }

        modified_request.input = ResponseInput::Items(items);
    }

    Ok(modified_request)
}
