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
    collections::HashMap,
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
        common::{Function, FunctionCallResponse, Tool, ToolCall, ToolChoice, ToolChoiceValue},
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
    background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
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
    background_tasks: Option<Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>>,
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
    background_tasks: Option<Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>>,
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
    background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
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

    // Create event emitter for OpenAI-compatible streaming
    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = original_request
        .model
        .clone()
        .unwrap_or_else(|| "default".to_string());
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

// ============================================================================
// Output Item Tracking Types
// ============================================================================

/// Output item type tracking for proper event emission
#[derive(Debug, Clone)]
enum OutputItemType {
    Message,
    McpListTools,
    McpCall,
    Reasoning,
}

/// Status of an output item
#[derive(Debug, Clone, PartialEq)]
enum ItemStatus {
    InProgress,
    Completed,
}

/// State tracking for a single output item
#[derive(Debug, Clone)]
struct OutputItemState {
    output_index: usize,
    status: ItemStatus,
}

// ============================================================================
// Streaming Event Emitter
// ============================================================================

/// OpenAI-compatible event emitter for /v1/responses streaming
///
/// Manages state and sequence numbers to emit proper event types:
/// - response.created
/// - response.in_progress
/// - response.output_item.added
/// - response.content_part.added
/// - response.output_text.delta (multiple)
/// - response.output_text.done
/// - response.content_part.done
/// - response.output_item.done
/// - response.completed
/// - response.mcp_list_tools.in_progress
/// - response.mcp_list_tools.completed
/// - response.mcp_call.in_progress
/// - response.mcp_call_arguments.delta
/// - response.mcp_call_arguments.done
/// - response.mcp_call.completed
/// - response.mcp_call.failed
struct ResponseStreamEventEmitter {
    sequence_number: u64,
    response_id: String,
    model: String,
    created_at: u64,
    message_id: String,
    accumulated_text: String,
    has_emitted_created: bool,
    has_emitted_in_progress: bool,
    has_emitted_output_item_added: bool,
    has_emitted_content_part_added: bool,
    // MCP call tracking
    mcp_call_accumulated_args: HashMap<String, String>,
    // Output item tracking (NEW)
    output_items: Vec<OutputItemState>,
    next_output_index: usize,
    current_message_output_index: Option<usize>, // Tracks output_index of current message
    current_item_id: Option<String>,             // Tracks item_id of current item
}

impl ResponseStreamEventEmitter {
    fn new(response_id: String, model: String, created_at: u64) -> Self {
        let message_id = format!("msg_{}", Uuid::new_v4());

        Self {
            sequence_number: 0,
            response_id,
            model,
            created_at,
            message_id,
            accumulated_text: String::new(),
            has_emitted_created: false,
            has_emitted_in_progress: false,
            has_emitted_output_item_added: false,
            has_emitted_content_part_added: false,
            mcp_call_accumulated_args: HashMap::new(),
            output_items: Vec::new(),
            next_output_index: 0,
            current_message_output_index: None,
            current_item_id: None,
        }
    }

    fn next_sequence(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    fn emit_created(&mut self) -> serde_json::Value {
        self.has_emitted_created = true;
        json!({
            "type": "response.created",
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "in_progress",
                "model": self.model,
                "output": []
            }
        })
    }

    fn emit_in_progress(&mut self) -> serde_json::Value {
        self.has_emitted_in_progress = true;
        json!({
            "type": "response.in_progress",
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "in_progress"
            }
        })
    }

    fn emit_content_part_added(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.has_emitted_content_part_added = true;
        json!({
            "type": "response.content_part.added",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "text",
                "text": ""
            }
        })
    }

    fn emit_text_delta(
        &mut self,
        delta: &str,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.accumulated_text.push_str(delta);
        json!({
            "type": "response.output_text.delta",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta
        })
    }

    fn emit_text_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": "response.output_text.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "text": self.accumulated_text.clone()
        })
    }

    fn emit_content_part_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": "response.content_part.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "text",
                "text": self.accumulated_text.clone()
            }
        })
    }

    fn emit_completed(&mut self, usage: Option<&serde_json::Value>) -> serde_json::Value {
        let mut response = json!({
            "type": "response.completed",
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "completed",
                "model": self.model,
                "output": [{
                    "id": self.message_id.clone(),
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": self.accumulated_text.clone()
                    }]
                }]
            }
        });

        if let Some(usage_val) = usage {
            response["response"]["usage"] = usage_val.clone();
        }

        response
    }

    // ========================================================================
    // MCP Event Emission Methods
    // ========================================================================

    fn emit_mcp_list_tools_in_progress(&mut self, output_index: usize) -> serde_json::Value {
        json!({
            "type": "response.mcp_list_tools.in_progress",
            "sequence_number": self.next_sequence(),
            "output_index": output_index
        })
    }

    fn emit_mcp_list_tools_completed(
        &mut self,
        output_index: usize,
        tools: &[crate::mcp::ToolInfo],
    ) -> serde_json::Value {
        let tool_items: Vec<_> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters.clone().unwrap_or_else(|| json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    }))
                })
            })
            .collect();

        json!({
            "type": "response.mcp_list_tools.completed",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "tools": tool_items
        })
    }

    fn emit_mcp_call_in_progress(
        &mut self,
        output_index: usize,
        item_id: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.mcp_call.in_progress",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    fn emit_mcp_call_arguments_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
    ) -> serde_json::Value {
        // Accumulate arguments for this call
        self.mcp_call_accumulated_args
            .entry(item_id.to_string())
            .or_default()
            .push_str(delta);

        json!({
            "type": "response.mcp_call_arguments.delta",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta
        })
    }

    fn emit_mcp_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.mcp_call_arguments.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    fn emit_mcp_call_completed(&mut self, output_index: usize, item_id: &str) -> serde_json::Value {
        json!({
            "type": "response.mcp_call.completed",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    fn emit_mcp_call_failed(
        &mut self,
        output_index: usize,
        item_id: &str,
        error: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.mcp_call.failed",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "error": error
        })
    }

    // ========================================================================
    // Output Item Wrapper Events
    // ========================================================================

    /// Emit response.output_item.added event
    fn emit_output_item_added(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        json!({
            "type": "response.output_item.added",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Emit response.output_item.done event
    fn emit_output_item_done(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        json!({
            "type": "response.output_item.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Generate unique ID for item type
    fn generate_item_id(prefix: &str) -> String {
        format!("{}_{}", prefix, Uuid::new_v4().to_string().replace("-", ""))
    }

    /// Allocate next output index and track item
    fn allocate_output_index(&mut self, item_type: OutputItemType) -> (usize, String) {
        let index = self.next_output_index;
        self.next_output_index += 1;

        let id_prefix = match &item_type {
            OutputItemType::McpListTools => "mcpl",
            OutputItemType::McpCall => "mcp",
            OutputItemType::Message => "msg",
            OutputItemType::Reasoning => "rs",
        };

        let id = Self::generate_item_id(id_prefix);

        self.output_items.push(OutputItemState {
            output_index: index,
            status: ItemStatus::InProgress,
        });

        (index, id)
    }

    /// Mark output item as completed
    fn complete_output_item(&mut self, output_index: usize) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.status = ItemStatus::Completed;
        }
    }

    /// Emit reasoning item wrapper events (added + done)
    ///
    /// Reasoning items in OpenAI format are simple placeholders emitted between tool iterations.
    /// They don't have streaming content - just wrapper events with empty/null content.
    fn emit_reasoning_item(
        &mut self,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
        reasoning_content: Option<String>,
    ) -> Result<(), String> {
        // Allocate output index and generate ID
        let (output_index, item_id) = self.allocate_output_index(OutputItemType::Reasoning);

        // Build reasoning item structure
        let item = json!({
            "id": item_id,
            "type": "reasoning",
            "summary": [],
            "content": reasoning_content,
            "encrypted_content": null,
            "status": null
        });

        // Emit output_item.added
        let added_event = self.emit_output_item_added(output_index, &item);
        self.send_event(&added_event, tx)?;

        // Immediately emit output_item.done (no streaming for reasoning)
        let done_event = self.emit_output_item_done(output_index, &item);
        self.send_event(&done_event, tx)?;

        // Mark as completed
        self.complete_output_item(output_index);

        Ok(())
    }

    /// Process a chunk and emit appropriate events
    fn process_chunk(
        &mut self,
        chunk: &ChatCompletionStreamResponse,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        // Process content if present
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    // Allocate output_index and item_id for this message item (once per message)
                    if self.current_item_id.is_none() {
                        let (output_index, item_id) =
                            self.allocate_output_index(OutputItemType::Message);

                        // Build message item structure
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": []
                        });

                        // Emit output_item.added
                        let event = self.emit_output_item_added(output_index, &item);
                        self.send_event(&event, tx)?;
                        self.has_emitted_output_item_added = true;

                        // Store for subsequent events
                        self.current_item_id = Some(item_id);
                        self.current_message_output_index = Some(output_index);
                    }

                    let output_index = self.current_message_output_index.unwrap();
                    let item_id = self.current_item_id.clone().unwrap(); // Clone to avoid borrow checker issues
                    let content_index = 0; // Single content part for now

                    // Emit content_part.added before first delta
                    if !self.has_emitted_content_part_added {
                        let event =
                            self.emit_content_part_added(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                        self.has_emitted_content_part_added = true;
                    }

                    // Emit text delta
                    let event =
                        self.emit_text_delta(content, output_index, &item_id, content_index);
                    self.send_event(&event, tx)?;
                }
            }

            // Check for finish_reason to emit completion events
            if let Some(reason) = &choice.finish_reason {
                if reason == "stop" || reason == "length" {
                    let output_index = self.current_message_output_index.unwrap();
                    let item_id = self.current_item_id.clone().unwrap(); // Clone to avoid borrow checker issues
                    let content_index = 0;

                    // Emit closing events
                    if self.has_emitted_content_part_added {
                        let event = self.emit_text_done(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                        let event =
                            self.emit_content_part_done(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                    }

                    if self.has_emitted_output_item_added {
                        // Build complete message item for output_item.done
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "text",
                                "text": self.accumulated_text.clone()
                            }]
                        });
                        let event = self.emit_output_item_done(output_index, &item);
                        self.send_event(&event, tx)?;
                    }

                    // Mark item as completed
                    self.complete_output_item(output_index);

                    // Emit completed with usage if available
                    let usage = chunk.usage.as_ref().map(|u| {
                        json!({
                            "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "total_tokens": u.total_tokens
                        })
                    });
                    let event = self.emit_completed(usage.as_ref());
                    self.send_event(&event, tx)?;
                }
            }
        }

        Ok(())
    }

    fn send_event(
        &self,
        event: &serde_json::Value,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        let event_json = serde_json::to_string(event)
            .map_err(|e| format!("Failed to serialize event: {}", e))?;

        if tx
            .send(Ok(Bytes::from(format!("data: {}\n\n", event_json))))
            .is_err()
        {
            return Err("Client disconnected".to_string());
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert MCP tools to Chat API tool format
fn convert_mcp_tools_to_chat_tools(mcp_tools: &[crate::mcp::ToolInfo]) -> Vec<Tool> {
    mcp_tools
        .iter()
        .map(|tool_info| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: tool_info.name.clone(),
                description: Some(tool_info.description.clone()),
                parameters: tool_info.parameters.clone().unwrap_or_else(|| {
                    json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }),
                strict: None,
            },
        })
        .collect()
}

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

        // Auto-create conversation if it doesn't exist (OpenAI behavior)
        if let Ok(None) = conversation_storage.get_conversation(&conv_id).await {
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

            let new_conv = crate::data_connector::conversations::NewConversation {
                id: Some(conv_id.clone()), // Use user-provided conversation ID
                metadata,
            };
            conversation_storage
                .create_conversation(new_conv)
                .await
                .map_err(|e| format!("Failed to create conversation: {}", e))?;
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

/// Extract all tool calls from chat response (for parallel tool call support)
fn extract_all_tool_calls_from_chat(
    response: &ChatCompletionResponse,
) -> Vec<(String, String, String)> {
    // Check if response has choices with tool calls
    let Some(choice) = response.choices.first() else {
        return Vec::new();
    };
    let message = &choice.message;

    // Look for tool_calls in the message
    if let Some(tool_calls) = &message.tool_calls {
        tool_calls
            .iter()
            .map(|tool_call| {
                (
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    tool_call
                        .function
                        .arguments
                        .clone()
                        .unwrap_or_else(|| "{}".to_string()),
                )
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// Execute an MCP tool call
async fn execute_mcp_call(
    mcp_mgr: &Arc<McpClientManager>,
    tool_name: &str,
    args_json_str: &str,
) -> Result<String, String> {
    // Parse arguments JSON string to Value
    let mut args_value: serde_json::Value =
        serde_json::from_str::<serde_json::Value>(args_json_str)
            .map_err(|e| format!("parse tool args: {}", e))?;

    // Get tool info to access schema for type coercion
    let tool_info = mcp_mgr
        .get_tool(tool_name)
        .ok_or_else(|| format!("tool not found: {}", tool_name))?;

    // Coerce string numbers to actual numbers based on schema (LLMs often output numbers as strings)
    if let Some(params) = &tool_info.parameters {
        let properties = params.get("properties").and_then(|p| p.as_object());
        let args_obj = args_value.as_object_mut();

        if let (Some(props), Some(args)) = (properties, args_obj) {
            for (key, val) in args.iter_mut() {
                let should_be_number = props
                    .get(key)
                    .and_then(|s| s.get("type"))
                    .and_then(|t| t.as_str())
                    .is_some_and(|t| matches!(t, "number" | "integer"));

                if should_be_number {
                    if let Some(s) = val.as_str() {
                        if let Ok(num) = s.parse::<f64>() {
                            *val = json!(num);
                        }
                    }
                }
            }
        }
    }

    let args_obj = args_value.as_object().cloned();

    debug!(
        "Calling MCP tool '{}' with args: {}",
        tool_name, args_json_str
    );

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
    background_tasks: Option<Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>>,
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
    background_tasks: Option<Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>>,
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

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_tools = mcp_manager.list_tools();
    let chat_tools = convert_mcp_tools_to_chat_tools(&mcp_tools);
    debug!("Converted {} MCP tools to chat format", chat_tools.len());

    loop {
        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {}", e))?;

        // Add MCP tools to chat request so LLM knows about them
        chat_request.tools = Some(chat_tools.clone());
        chat_request.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

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

            debug!(
                "Tool loop iteration {}: found call to {} (call_id: {})",
                state.iteration, tool_name, call_id
            );

            // Check combined limit BEFORE executing
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(MAX_ITERATIONS),
                None => MAX_ITERATIONS,
            };

            if state.total_calls >= effective_limit {
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

            // Increment after check
            state.total_calls += 1;

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

            // Record the call in state
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
    const MAX_ITERATIONS: usize = 10;
    let mut state = ToolLoopState::new(original_request.input.clone(), server_label.clone());
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    // Create response event emitter
    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = current_request
        .model
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);

    // Emit initial response.created and response.in_progress events
    let event = emitter.emit_created();
    emitter.send_event(&event, &tx)?;
    let event = emitter.emit_in_progress();
    emitter.send_event(&event, &tx)?;

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_tools = mcp_manager.list_tools();
    let chat_tools = convert_mcp_tools_to_chat_tools(&mcp_tools);
    debug!(
        "Streaming: Converted {} MCP tools to chat format",
        chat_tools.len()
    );

    // Flag to track if mcp_list_tools has been emitted
    let mut mcp_list_tools_emitted = false;

    loop {
        state.iteration += 1;
        if state.iteration > MAX_ITERATIONS {
            return Err(format!(
                "Tool loop exceeded maximum iterations ({})",
                MAX_ITERATIONS
            ));
        }

        debug!("Streaming MCP tool loop iteration {}", state.iteration);

        // Emit mcp_list_tools as first output item (only once, on first iteration)
        if !mcp_list_tools_emitted {
            let (output_index, item_id) =
                emitter.allocate_output_index(OutputItemType::McpListTools);

            // Build tools list for item structure
            let tool_items: Vec<_> = mcp_tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters.clone().unwrap_or_else(|| json!({
                            "type": "object",
                            "properties": {},
                            "required": []
                        }))
                    })
                })
                .collect();

            // Build mcp_list_tools item
            let item = json!({
                "id": item_id,
                "type": "mcp_list_tools",
                "server_label": state.server_label,
                "status": "in_progress",
                "tools": []
            });

            // Emit output_item.added
            let event = emitter.emit_output_item_added(output_index, &item);
            emitter.send_event(&event, &tx)?;

            // Emit mcp_list_tools.in_progress
            let event = emitter.emit_mcp_list_tools_in_progress(output_index);
            emitter.send_event(&event, &tx)?;

            // Emit mcp_list_tools.completed
            let event = emitter.emit_mcp_list_tools_completed(output_index, &mcp_tools);
            emitter.send_event(&event, &tx)?;

            // Build complete item with tools
            let item_done = json!({
                "id": item_id,
                "type": "mcp_list_tools",
                "server_label": state.server_label,
                "status": "completed",
                "tools": tool_items
            });

            // Emit output_item.done
            let event = emitter.emit_output_item_done(output_index, &item_done);
            emitter.send_event(&event, &tx)?;

            emitter.complete_output_item(output_index);
            mcp_list_tools_emitted = true;
        }

        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {}", e))?;

        // Add MCP tools to chat request so LLM knows about them
        chat_request.tools = Some(chat_tools.clone());
        chat_request.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

        // Execute chat streaming
        let response = pipeline
            .execute_chat(
                Arc::new(chat_request),
                headers.clone(),
                model_id.clone(),
                components.clone(),
            )
            .await;

        // Convert chat stream to Responses API events while accumulating for tool call detection
        // Stream text naturally - it only appears on final iteration (tool iterations have empty content)
        let accumulated_response =
            convert_and_accumulate_stream(response.into_body(), &mut emitter, &tx).await?;

        // Check for tool calls (extract all of them for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&accumulated_response);

        if !tool_calls.is_empty() {
            debug!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Check combined limit
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(MAX_ITERATIONS),
                None => MAX_ITERATIONS,
            };

            if state.total_calls + tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    MAX_ITERATIONS
                );
                break;
            }

            // Process each tool call
            for (call_id, tool_name, args_json_str) in tool_calls {
                state.total_calls += 1;

                debug!(
                    "Executing tool call {}/{}: {} (call_id: {})",
                    state.total_calls, state.total_calls, tool_name, call_id
                );

                // Allocate output_index for this mcp_call item
                let (output_index, item_id) =
                    emitter.allocate_output_index(OutputItemType::McpCall);

                // Build initial mcp_call item
                let item = json!({
                    "id": item_id,
                    "type": "mcp_call",
                    "name": tool_name,
                    "server_label": state.server_label,
                    "status": "in_progress",
                    "arguments": ""
                });

                // Emit output_item.added
                let event = emitter.emit_output_item_added(output_index, &item);
                emitter.send_event(&event, &tx)?;

                // Emit mcp_call.in_progress
                let event = emitter.emit_mcp_call_in_progress(output_index, &item_id);
                emitter.send_event(&event, &tx)?;

                // Emit mcp_call_arguments.delta (simulate streaming by sending full arguments)
                let event =
                    emitter.emit_mcp_call_arguments_delta(output_index, &item_id, &args_json_str);
                emitter.send_event(&event, &tx)?;

                // Emit mcp_call_arguments.done
                let event =
                    emitter.emit_mcp_call_arguments_done(output_index, &item_id, &args_json_str);
                emitter.send_event(&event, &tx)?;

                // Execute the MCP tool
                let (output_str, success, error) =
                    match execute_mcp_call(&mcp_manager, &tool_name, &args_json_str).await {
                        Ok(output) => {
                            // Emit mcp_call.completed
                            let event = emitter.emit_mcp_call_completed(output_index, &item_id);
                            emitter.send_event(&event, &tx)?;

                            // Build complete item with output
                            let item_done = json!({
                                "id": item_id,
                                "type": "mcp_call",
                                "name": tool_name,
                                "server_label": state.server_label,
                                "status": "completed",
                                "arguments": args_json_str,
                                "output": output
                            });

                            // Emit output_item.done
                            let event = emitter.emit_output_item_done(output_index, &item_done);
                            emitter.send_event(&event, &tx)?;

                            emitter.complete_output_item(output_index);
                            (output, true, None)
                        }
                        Err(err) => {
                            warn!("Tool execution failed: {}", err);
                            // Emit mcp_call.failed
                            let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err);
                            emitter.send_event(&event, &tx)?;

                            // Build failed item
                            let item_done = json!({
                                "id": item_id,
                                "type": "mcp_call",
                                "name": tool_name,
                                "server_label": state.server_label,
                                "status": "failed",
                                "arguments": args_json_str,
                                "error": err
                            });

                            // Emit output_item.done
                            let event = emitter.emit_output_item_done(output_index, &item_done);
                            emitter.send_event(&event, &tx)?;

                            emitter.complete_output_item(output_index);
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
            }

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

        // No tool calls, this is the final response
        debug!("No tool calls found, ending streaming MCP loop");

        // Check for reasoning content
        let reasoning_content = accumulated_response
            .choices
            .first()
            .and_then(|c| c.message.reasoning_content.clone());

        // Emit reasoning item if present
        if let Some(reasoning) = reasoning_content {
            if !reasoning.is_empty() {
                emitter.emit_reasoning_item(&tx, Some(reasoning))?;
            }
        }

        // Text message events already emitted naturally by process_chunk during stream processing
        // (OpenAI router approach - text only appears on final iteration when no tool calls)

        // Emit final response.completed event
        let usage_json = accumulated_response.usage.as_ref().map(|u| {
            json!({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens
            })
        });
        let event = emitter.emit_completed(usage_json.as_ref());
        emitter.send_event(&event, &tx)?;

        break;
    }

    Ok(())
}

/// Convert chat stream to Responses API events while accumulating for tool call detection
async fn convert_and_accumulate_stream(
    body: Body,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<ChatCompletionResponse, String> {
    let mut accumulator = ChatResponseAccumulator::new();
    let mut stream = body.into_data_stream();

    while let Some(chunk_result) = futures_util::StreamExt::next(&mut stream).await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;

        // Parse chunk
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        if event == "data: [DONE]" {
            break;
        }

        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();
            if let Ok(chat_chunk) = serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                // Convert chat chunk to Responses API events and emit
                emitter.process_chunk(&chat_chunk, tx)?;

                // Accumulate for tool call detection
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
    tool_calls: HashMap<usize, ToolCall>,
    finish_reason: Option<String>,
}

impl ChatResponseAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            content: String::new(),
            tool_calls: HashMap::new(),
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
