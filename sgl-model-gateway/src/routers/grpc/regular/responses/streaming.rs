//! Streaming execution for Regular Responses API
//!
//! This module handles streaming request execution:
//! - `execute_tool_loop_streaming` - MCP tool loop with streaming
//! - `convert_chat_stream_to_responses_stream` - Non-MCP streaming conversion
//! - Streaming accumulators for response building

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    extract::ws::Message,
    http::{self, header, StatusCode},
    response::Response,
};
use bytes::Bytes;
use data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, trace, warn};
use uuid::Uuid;

use super::{
    common::{
        build_next_request, convert_mcp_tools_to_chat_tools, extract_all_tool_calls_from_chat,
        prepare_chat_tools_and_choice, ExtractedToolCall, ToolLoopState,
    },
    conversions,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    protocols::{
        chat::{
            ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
            ChatCompletionStreamResponse,
        },
        common::{FunctionCallResponse, ToolCall, Usage, UsageInfo},
        responses::{
            ResponseContentPart, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
            ResponsesRequest, ResponsesResponse, ResponsesUsage,
        },
    },
    routers::{
        grpc::common::responses::{
            build_sse_response, persist_response_if_needed,
            streaming::{
                OutputItemType, ResponseEventSink, ResponseStreamEventEmitter,
                SseResponseEventSink, WsResponseEventSink,
            },
            ResponsesContext,
        },
        mcp_utils::{extract_server_label, DEFAULT_MAX_ITERATIONS},
    },
};

/// Generate a unique item ID with the given prefix (e.g. `"fc"` → `"fc_a1b2c3..."`).
fn generate_item_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4().to_string().replace("-", ""))
}

// ============================================================================
// Shared SSE body parser — single implementation for both MCP and non-MCP paths
// ============================================================================

/// What a parsed SSE `data:` line contains.
enum SseDataRecord {
    /// A parsed `ChatCompletionStreamResponse` chunk.
    ChatChunk(ChatCompletionStreamResponse),
    /// A raw JSON payload that did not parse as a chat chunk (e.g. upstream error).
    RawJson(String),
    /// The `data: [DONE]` sentinel.
    Done,
}

/// Incrementally parse SSE records from an HTTP body stream.
///
/// Handles split and coalesced chunks using an internal pending buffer with
/// offset-based compaction (O(n) amortised).  Both the non-MCP and MCP
/// streaming paths use this so the SSE-byte-parsing logic lives in one place.
struct SseBodyParser {
    pending: String,
    offset: usize,
}

impl SseBodyParser {
    fn new() -> Self {
        Self {
            pending: String::new(),
            offset: 0,
        }
    }

    /// Append raw bytes from a body chunk.
    fn push(&mut self, chunk: &[u8]) {
        let text = String::from_utf8_lossy(chunk).replace("\r\n", "\n");
        self.pending.push_str(&text);
    }

    /// Yield the next complete SSE `data:` record, if one is available.
    fn next_record(&mut self) -> Option<SseDataRecord> {
        let rel_end = self.pending[self.offset..].find("\n\n")?;
        let record_end = self.offset + rel_end;

        // Extract and advance offset before any compaction so we don't
        // borrow `self.pending` across the mutation.
        let record = self.pending[self.offset..record_end].trim().to_string();
        self.offset = record_end + 2;

        // Compact when consumed prefix exceeds half the buffer.
        if self.offset > self.pending.len() / 2 {
            self.pending = self.pending[self.offset..].to_string();
            self.offset = 0;
        }

        if record.is_empty() {
            return self.next_record();
        }

        if record == "data: [DONE]" {
            return Some(SseDataRecord::Done);
        }

        let Some(json_str) = record.strip_prefix("data: ") else {
            return self.next_record();
        };
        let json_str = json_str.trim();

        match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
            Ok(chunk) => Some(SseDataRecord::ChatChunk(chunk)),
            Err(_) => Some(SseDataRecord::RawJson(json_str.to_string())),
        }
    }

    /// Flush any remaining partial data after the body stream ends.
    fn flush(&mut self) -> Option<SseDataRecord> {
        let remaining = self.pending[self.offset..].trim();
        if remaining.is_empty() || remaining == "data: [DONE]" {
            return None;
        }
        // Reset offset so we don't flush twice.
        self.offset = self.pending.len();

        let json_str = remaining.strip_prefix("data: ")?.trim();

        match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
            Ok(chunk) => Some(SseDataRecord::ChatChunk(chunk)),
            Err(_) => Some(SseDataRecord::RawJson(json_str.to_string())),
        }
    }
}

// ============================================================================
// Non-MCP Streaming Path
// ============================================================================

/// Convert chat streaming response to responses streaming format
///
/// This function:
/// 1. Gets chat SSE stream from pipeline
/// 2. Intercepts and parses each SSE event
/// 3. Converts ChatCompletionStreamResponse → ResponsesResponse delta
/// 4. Accumulates response state for final persistence
/// 5. Emits transformed SSE events in responses format
pub(super) async fn convert_chat_stream_to_responses_stream(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    original_request: &ResponsesRequest,
) -> Response {
    debug!("Converting chat SSE stream to responses SSE format");

    // Create channel for transformed SSE events
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    let sink = SseResponseEventSink::new(tx.clone());

    // Spawn background task to transform stream
    let original_request_clone = original_request.clone();
    let ctx_clone = ctx.clone();
    let chat_request_clone = chat_request.clone();

    tokio::spawn(async move {
        if let Err(e) = execute_non_mcp_stream_with_sink(
            &ctx_clone,
            chat_request_clone,
            original_request_clone,
            headers,
            model_id,
            &sink,
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
            let _ = sink.send_raw_json(&error_event.to_string());
        }

        let _ = sink.send_done();
    });

    // Build SSE response with transformed stream
    build_sse_response(rx)
}

pub(super) async fn execute_non_mcp_stream_with_sink(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    original_request: ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    sink: &impl ResponseEventSink,
) -> Result<ResponsesResponse, String> {
    let chat_response = ctx
        .pipeline
        .execute_chat(chat_request, headers, model_id, ctx.components.clone())
        .await;
    let (_parts, body) = chat_response.into_parts();

    process_and_transform_stream(
        body,
        original_request,
        ctx.response_storage.clone(),
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        sink,
    )
    .await
}

/// Process chat SSE stream and transform to responses format.
async fn process_and_transform_stream(
    body: Body,
    original_request: ResponsesRequest,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    sink: &impl ResponseEventSink,
) -> Result<ResponsesResponse, String> {
    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = original_request.model.clone();
    let created_at = chrono::Utc::now().timestamp() as u64;

    // Create accumulator for the persisted/final response. Seed it with the
    // same response metadata the WS/SSE event emitter exposes so storage,
    // cache, and retrieval all agree on the response id.
    let mut accumulator = StreamingResponseAccumulator::new(
        &original_request,
        response_id.clone(),
        model.clone(),
        created_at as i64,
    );

    // Create event emitter for OpenAI-compatible streaming
    let mut event_emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);
    event_emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events
    let event = event_emitter.emit_created();
    event_emitter
        .send_event(&event, sink)
        .map_err(|_| "Failed to send response.created event".to_string())?;

    let event = event_emitter.emit_in_progress();
    event_emitter
        .send_event(&event, sink)
        .map_err(|_| "Failed to send response.in_progress event".to_string())?;

    // Parse SSE records from the body stream using the shared parser.
    let mut stream = body.into_data_stream();
    let mut upstream_error_forwarded = false;
    let mut function_call_events = BTreeMap::new();
    let mut parser = SseBodyParser::new();

    'stream: while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;
        parser.push(&chunk);

        while let Some(record) = parser.next_record() {
            match process_non_mcp_sse_data_record(
                record,
                &mut accumulator,
                &mut event_emitter,
                sink,
                &mut function_call_events,
            )? {
                SseRecordOutcome::Continue => {}
                SseRecordOutcome::StopStream => break 'stream,
                SseRecordOutcome::UpstreamError => {
                    upstream_error_forwarded = true;
                    break 'stream;
                }
            }
        }
    }

    // Flush any remaining partial record after the body stream ends.
    if !upstream_error_forwarded {
        if let Some(record) = parser.flush() {
            if let SseRecordOutcome::UpstreamError = process_non_mcp_sse_data_record(
                record,
                &mut accumulator,
                &mut event_emitter,
                sink,
                &mut function_call_events,
            )? {
                upstream_error_forwarded = true;
            }
        }
    }

    if upstream_error_forwarded {
        debug!(
            "Upstream error payload was already forwarded; skipping synthetic response.completed"
        );
        return Ok(accumulator.finalize_with_status(ResponseStatus::Failed));
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

    let terminal_status = accumulator.response_status();
    let completed_event =
        event_emitter.emit_completed_with_status(terminal_status, usage_json.as_ref());
    event_emitter.send_event(&completed_event, sink)?;

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

    Ok(final_response)
}

enum SseRecordOutcome {
    Continue,
    StopStream,
    UpstreamError,
}

/// Process a pre-parsed SSE data record on the non-MCP streaming path.
fn process_non_mcp_sse_data_record(
    record: SseDataRecord,
    accumulator: &mut StreamingResponseAccumulator,
    event_emitter: &mut ResponseStreamEventEmitter,
    sink: &impl ResponseEventSink,
    function_call_events: &mut BTreeMap<usize, FunctionCallEventState>,
) -> Result<SseRecordOutcome, String> {
    match record {
        SseDataRecord::Done => Ok(SseRecordOutcome::StopStream),
        SseDataRecord::ChatChunk(chat_chunk) => {
            accumulator.process_chunk(&chat_chunk);

            let has_tool_call_delta = chat_chunk
                .choices
                .first()
                .and_then(|choice| choice.delta.tool_calls.as_ref())
                .is_some_and(|tool_calls| !tool_calls.is_empty());
            if has_tool_call_delta || !function_call_events.is_empty() {
                process_non_mcp_function_call_chunk(
                    &chat_chunk,
                    event_emitter,
                    sink,
                    function_call_events,
                )?;
            } else {
                event_emitter.process_chunk(&chat_chunk, sink)?;
            }

            Ok(SseRecordOutcome::Continue)
        }
        SseDataRecord::RawJson(json_str) => {
            debug!("Non-chunk SSE event, passing through: {}", json_str);
            sink.send_raw_json(&json_str)?;

            if is_upstream_error_payload(&json_str) {
                Ok(SseRecordOutcome::UpstreamError)
            } else {
                Ok(SseRecordOutcome::Continue)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct FunctionCallEventState {
    output_index: usize,
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    completed: bool,
}

fn process_non_mcp_function_call_chunk(
    chunk: &ChatCompletionStreamResponse,
    emitter: &mut ResponseStreamEventEmitter,
    sink: &impl ResponseEventSink,
    function_call_events: &mut BTreeMap<usize, FunctionCallEventState>,
) -> Result<(), String> {
    let Some(choice) = chunk.choices.first() else {
        return Ok(());
    };

    if let Some(tool_call_deltas) = &choice.delta.tool_calls {
        for delta in tool_call_deltas {
            let state = function_call_events
                .entry(delta.index as usize)
                .or_insert_with(|| {
                    let (output_index, item_id) =
                        emitter.allocate_output_index(OutputItemType::FunctionCall);

                    FunctionCallEventState {
                        output_index,
                        item_id,
                        call_id: String::new(),
                        name: String::new(),
                        arguments: String::new(),
                        completed: false,
                    }
                });

            if state.call_id.is_empty() {
                if let Some(delta_id) = &delta.id {
                    state.call_id = delta_id.clone();
                }
            } else if let Some(delta_id) = &delta.id {
                state.call_id.push_str(delta_id);
            }

            if let Some(function) = &delta.function {
                if let Some(delta_name) = &function.name {
                    state.name.push_str(delta_name);
                }

                if let Some(delta_args) = &function.arguments {
                    if !delta_args.is_empty() {
                        if state.arguments.is_empty() {
                            let item = json!({
                                "id": state.item_id,
                                "type": "function_call",
                                "call_id": state.call_id,
                                "name": state.name,
                                "status": "in_progress",
                                "arguments": ""
                            });
                            let event = emitter.emit_output_item_added(state.output_index, &item);
                            emitter.send_event(&event, sink)?;
                        }

                        state.arguments.push_str(delta_args);
                        let event = emitter.emit_function_call_arguments_delta(
                            state.output_index,
                            &state.item_id,
                            delta_args,
                        );
                        emitter.send_event(&event, sink)?;
                    }
                }
            }
        }
    }

    if choice.finish_reason.as_deref() == Some("tool_calls") {
        for state in function_call_events.values_mut() {
            if state.completed {
                continue;
            }

            let event = emitter.emit_function_call_arguments_done(
                state.output_index,
                &state.item_id,
                &state.arguments,
            );
            emitter.send_event(&event, sink)?;

            let item = json!({
                "id": state.item_id,
                "type": "function_call",
                "call_id": state.call_id,
                "name": state.name,
                "status": "completed",
                "arguments": state.arguments
            });
            let event = emitter.emit_output_item_done(state.output_index, &item);
            emitter.send_event(&event, sink)?;
            emitter.complete_output_item(state.output_index);
            state.completed = true;
        }
    }

    Ok(())
}

/// Response accumulator for streaming responses (non-MCP path)
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
    usage: Option<Usage>,

    // Original request for final response construction
    original_request: ResponsesRequest,
}

impl StreamingResponseAccumulator {
    fn new(
        original_request: &ResponsesRequest,
        response_id: String,
        model: String,
        created_at: i64,
    ) -> Self {
        Self {
            response_id,
            model,
            created_at,
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
            // Accumulate content only when no tool calls have been seen.
            // When tool calls are present, content is typically empty or
            // whitespace; including it would create a spurious Message
            // output item alongside the FunctionToolCall items.
            if let Some(content) = &choice.delta.content {
                if self.tool_calls.is_empty() {
                    self.content_buffer.push_str(content);
                }
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

                    // Ensure we have enough tool calls.  The item `id` is
                    // pre-generated so it remains stable across deltas.
                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(ResponseOutputItem::FunctionToolCall {
                            id: generate_item_id("fc"),
                            call_id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            output: None,
                            status: "in_progress".to_string(),
                        });
                    }

                    // Update the tool call at this index
                    if let ResponseOutputItem::FunctionToolCall {
                        call_id,
                        name,
                        arguments,
                        ..
                    } = &mut self.tool_calls[index]
                    {
                        // The tool call ID arrives on the first delta; assign
                        // once rather than appending on every chunk.
                        if let Some(delta_id) = &delta.id {
                            if call_id.is_empty() {
                                call_id.clone_from(delta_id);
                            }
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

    fn response_status(&self) -> ResponseStatus {
        match self.finish_reason.as_deref() {
            Some("stop") | Some("length") => ResponseStatus::Completed,
            Some("tool_calls") => ResponseStatus::InProgress,
            Some("failed") | Some("error") => ResponseStatus::Failed,
            _ => ResponseStatus::Completed,
        }
    }

    fn finalize(self) -> ResponsesResponse {
        let status = self.response_status();
        self.finalize_with_status(status)
    }

    fn finalize_with_status(self, status: ResponseStatus) -> ResponsesResponse {
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

        // Convert usage
        let usage = self.usage.as_ref().map(|u| {
            let usage_info = UsageInfo {
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

fn is_upstream_error_payload(payload: &str) -> bool {
    serde_json::from_str::<Value>(payload)
        .ok()
        .and_then(|value| value.get("error").cloned())
        .is_some()
}

// ============================================================================
// MCP Streaming Path
// ============================================================================

/// Execute MCP tool loop with streaming support
///
/// This streams each iteration's response to the client while accumulating
/// to check for tool calls. If tool calls are found, executes them and
/// continues with the next streaming iteration.
pub(super) async fn execute_tool_loop_streaming(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
    // Create SSE channel for client
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    let sink = SseResponseEventSink::new(tx.clone());

    // Clone data for background task
    let ctx_clone = ctx.clone();
    let original_request_clone = original_request.clone();

    // Spawn background task for tool loop
    tokio::spawn(async move {
        let result = execute_tool_loop_streaming_internal(
            &ctx_clone,
            current_request,
            &original_request_clone,
            headers,
            model_id,
            &sink,
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
            let _ = sink.send_raw_json(&error_event.to_string());
        }

        let _ = sink.send_done();
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

pub(super) async fn execute_tool_loop_streaming_with_sink(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    outbound_tx: mpsc::Sender<Message>,
) -> Result<ResponsesResponse, String> {
    let sink = WsResponseEventSink::new(outbound_tx);
    execute_tool_loop_streaming_internal(
        ctx,
        current_request,
        original_request,
        headers,
        model_id,
        &sink,
    )
    .await
}

/// Internal streaming tool loop implementation
async fn execute_tool_loop_streaming_internal(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    sink: &impl ResponseEventSink,
) -> Result<ResponsesResponse, String> {
    // Extract server label from original request tools
    let server_label = extract_server_label(original_request.tools.as_deref(), "request-mcp");

    let mut state = ToolLoopState::new(original_request.input.clone(), server_label.clone());
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    // Create response event emitter
    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = current_request.model.clone();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut emitter = ResponseStreamEventEmitter::new(response_id, model.clone(), created_at);
    emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events
    let event = emitter.emit_created();
    emitter.send_event(&event, sink)?;
    let event = emitter.emit_in_progress();
    emitter.send_event(&event, sink)?;

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_tools = ctx.mcp_manager.list_tools();
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&mcp_tools);
    trace!(
        "Streaming: Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    // Flag to track if mcp_list_tools has been emitted
    let mut mcp_list_tools_emitted = false;

    loop {
        state.iteration += 1;

        // Record tool loop iteration metric
        Metrics::record_mcp_tool_iteration(&model);

        if state.iteration > DEFAULT_MAX_ITERATIONS {
            return Err(format!(
                "Tool loop exceeded maximum iterations ({})",
                DEFAULT_MAX_ITERATIONS
            ));
        }

        trace!("Streaming MCP tool loop iteration {}", state.iteration);

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
                        "input_schema": Value::Object((*t.input_schema).clone())
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
            emitter.send_event(&event, sink)?;

            // Emit mcp_list_tools.in_progress
            let event = emitter.emit_mcp_list_tools_in_progress(output_index);
            emitter.send_event(&event, sink)?;

            // Emit mcp_list_tools.completed
            let event = emitter.emit_mcp_list_tools_completed(output_index, &mcp_tools);
            emitter.send_event(&event, sink)?;

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
            emitter.send_event(&event, sink)?;

            emitter.complete_output_item(output_index);
            mcp_list_tools_emitted = true;
        }

        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {}", e))?;

        // Prepare tools and tool_choice for this iteration (same logic as non-streaming)
        prepare_chat_tools_and_choice(&mut chat_request, &mcp_chat_tools, state.iteration);

        // Execute chat streaming
        let response = ctx
            .pipeline
            .execute_chat(
                Arc::new(chat_request),
                headers.clone(),
                model_id.clone(),
                ctx.components.clone(),
            )
            .await;

        // Convert chat stream to Responses API events while accumulating for tool call detection
        // Stream text naturally - it only appears on final iteration (tool iterations have empty content)
        let accumulated_response =
            convert_and_accumulate_stream(response.into_body(), &mut emitter, sink).await?;

        // Check for tool calls (extract all of them for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&accumulated_response);

        if !tool_calls.is_empty() {
            trace!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Separate MCP and function tool calls
            let mcp_tool_names: std::collections::HashSet<&str> =
                mcp_tools.iter().map(|t| t.name.as_ref()).collect();
            let (mcp_tool_calls, function_tool_calls): (Vec<ExtractedToolCall>, Vec<_>) =
                tool_calls
                    .into_iter()
                    .partition(|tc| mcp_tool_names.contains(tc.name.as_str()));

            trace!(
                "Separated tool calls: {} MCP, {} function",
                mcp_tool_calls.len(),
                function_tool_calls.len()
            );

            // Check combined limit (only count MCP tools since function tools will be returned)
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    DEFAULT_MAX_ITERATIONS
                );

                let mut final_response = emitter.finalize(accumulated_response.usage.clone());
                final_response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                let usage_json = accumulated_response.usage.as_ref().map(|u| {
                    json!({
                        "input_tokens": u.prompt_tokens,
                        "output_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens
                    })
                });
                let mut event = emitter.emit_completed(usage_json.as_ref());
                if let Some(response) = event.get_mut("response") {
                    response["incomplete_details"] = json!({ "reason": "max_tool_calls" });
                }
                emitter.send_event(&event, sink)?;

                persist_response_if_needed(
                    ctx.conversation_storage.clone(),
                    ctx.conversation_item_storage.clone(),
                    ctx.response_storage.clone(),
                    &final_response,
                    original_request,
                )
                .await;

                return Ok(final_response);
            }

            // Process each MCP tool call
            for tool_call in mcp_tool_calls {
                state.total_calls += 1;

                trace!(
                    "Executing tool call {}/{}: {} (call_id: {})",
                    state.total_calls,
                    state.total_calls,
                    tool_call.name,
                    tool_call.call_id
                );

                // Allocate output_index for this mcp_call item
                let (output_index, item_id) =
                    emitter.allocate_output_index(OutputItemType::McpCall);

                // Build initial mcp_call item
                let item = json!({
                    "id": item_id,
                    "type": "mcp_call",
                    "name": tool_call.name,
                    "server_label": state.server_label,
                    "status": "in_progress",
                    "arguments": ""
                });

                // Emit output_item.added
                let event = emitter.emit_output_item_added(output_index, &item);
                emitter.send_event(&event, sink)?;

                // Emit mcp_call.in_progress
                let event = emitter.emit_mcp_call_in_progress(output_index, &item_id);
                emitter.send_event(&event, sink)?;

                // Emit mcp_call_arguments.delta (simulate streaming by sending full arguments)
                let event = emitter.emit_mcp_call_arguments_delta(
                    output_index,
                    &item_id,
                    &tool_call.arguments,
                );
                emitter.send_event(&event, sink)?;

                // Emit mcp_call_arguments.done
                let event = emitter.emit_mcp_call_arguments_done(
                    output_index,
                    &item_id,
                    &tool_call.arguments,
                );
                emitter.send_event(&event, sink)?;

                // Execute the MCP tool - manager handles parsing and type coercion
                trace!(
                    "Calling MCP tool '{}' with args: {}",
                    tool_call.name,
                    tool_call.arguments
                );
                let tool_start = Instant::now();
                let (output_str, success, error) = match ctx
                    .mcp_manager
                    .call_tool(tool_call.name.as_str(), tool_call.arguments.as_str())
                    .await
                {
                    Ok(result) => match serde_json::to_string(&result) {
                        Ok(output) => {
                            // Emit mcp_call.completed
                            let event = emitter.emit_mcp_call_completed(output_index, &item_id);
                            emitter.send_event(&event, sink)?;

                            // Build complete item with output
                            let item_done = json!({
                                "id": item_id,
                                "type": "mcp_call",
                                "name": tool_call.name,
                                "server_label": state.server_label,
                                "status": "completed",
                                "arguments": tool_call.arguments,
                                "output": output
                            });

                            // Emit output_item.done
                            let event = emitter.emit_output_item_done(output_index, &item_done);
                            emitter.send_event(&event, sink)?;

                            emitter.complete_output_item(output_index);
                            (output, true, None)
                        }
                        Err(e) => {
                            let err = format!("Failed to serialize tool result: {}", e);
                            warn!("{}", err);
                            // Emit mcp_call.failed
                            let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err);
                            emitter.send_event(&event, sink)?;

                            // Build failed item
                            let item_done = json!({
                                "id": item_id,
                                "type": "mcp_call",
                                "name": tool_call.name,
                                "server_label": state.server_label,
                                "status": "failed",
                                "arguments": tool_call.arguments,
                                "error": &err
                            });

                            // Emit output_item.done
                            let event = emitter.emit_output_item_done(output_index, &item_done);
                            emitter.send_event(&event, sink)?;

                            emitter.complete_output_item(output_index);
                            let error_json = json!({ "error": &err }).to_string();
                            (error_json, false, Some(err))
                        }
                    },
                    Err(err) => {
                        let err_str = format!("tool call failed: {}", err);
                        warn!("Tool execution failed: {}", err_str);
                        // Emit mcp_call.failed
                        let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err_str);
                        emitter.send_event(&event, sink)?;

                        // Build failed item
                        let item_done = json!({
                            "id": item_id,
                            "type": "mcp_call",
                            "name": tool_call.name,
                            "server_label": state.server_label,
                            "status": "failed",
                            "arguments": tool_call.arguments,
                            "error": &err_str
                        });

                        // Emit output_item.done
                        let event = emitter.emit_output_item_done(output_index, &item_done);
                        emitter.send_event(&event, sink)?;

                        emitter.complete_output_item(output_index);
                        let error_json = json!({ "error": &err_str }).to_string();
                        (error_json, false, Some(err_str))
                    }
                };
                let tool_duration = tool_start.elapsed();

                // Record MCP tool metrics
                Metrics::record_mcp_tool_duration(&model, &tool_call.name, tool_duration);
                Metrics::record_mcp_tool_call(
                    &model,
                    &tool_call.name,
                    if success {
                        metrics_labels::RESULT_SUCCESS
                    } else {
                        metrics_labels::RESULT_ERROR
                    },
                );

                // Record the call in state
                state.record_call(
                    tool_call.call_id,
                    tool_call.name,
                    tool_call.arguments,
                    output_str,
                    success,
                    error,
                );
            }

            // If there are function tool calls, emit events and exit MCP loop
            if !function_tool_calls.is_empty() {
                trace!(
                    "Found {} function tool call(s) - emitting events and exiting MCP loop",
                    function_tool_calls.len()
                );

                // Emit function_tool_call events for each function tool
                for tool_call in function_tool_calls {
                    // Allocate output_index for this function_tool_call item
                    let (output_index, item_id) =
                        emitter.allocate_output_index(OutputItemType::FunctionCall);

                    // Build initial function_tool_call item
                    let item = json!({
                        "id": item_id,
                        "type": "function_tool_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "in_progress",
                        "arguments": ""
                    });

                    // Emit output_item.added
                    let event = emitter.emit_output_item_added(output_index, &item);
                    emitter.send_event(&event, sink)?;

                    // Emit function_call_arguments.delta
                    let event = emitter.emit_function_call_arguments_delta(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;

                    // Emit function_call_arguments.done
                    let event = emitter.emit_function_call_arguments_done(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;

                    // Build complete item
                    let item_complete = json!({
                        "id": item_id,
                        "type": "function_tool_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "completed",
                        "arguments": tool_call.arguments
                    });

                    // Emit output_item.done
                    let event = emitter.emit_output_item_done(output_index, &item_complete);
                    emitter.send_event(&event, sink)?;

                    emitter.complete_output_item(output_index);
                }

                let usage_json = accumulated_response.usage.as_ref().map(|u| {
                    json!({
                        "input_tokens": u.prompt_tokens,
                        "output_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens
                    })
                });
                let event = emitter
                    .emit_completed_with_status(ResponseStatus::InProgress, usage_json.as_ref());
                emitter.send_event(&event, sink)?;

                let final_response = emitter.finalize_with_status(
                    accumulated_response.usage.clone(),
                    ResponseStatus::InProgress,
                );
                persist_response_if_needed(
                    ctx.conversation_storage.clone(),
                    ctx.conversation_item_storage.clone(),
                    ctx.response_storage.clone(),
                    &final_response,
                    original_request,
                )
                .await;

                return Ok(final_response);
            }

            // Build next request with conversation history
            current_request = build_next_request(&state, &current_request);

            continue;
        }

        // No tool calls, this is the final response
        trace!("No tool calls found, ending streaming MCP loop");

        // Check for reasoning content
        let reasoning_content = accumulated_response
            .choices
            .first()
            .and_then(|c| c.message.reasoning_content.clone());

        // Emit reasoning item if present
        if let Some(reasoning) = reasoning_content {
            if !reasoning.is_empty() {
                emitter.emit_reasoning_item(sink, Some(reasoning))?;
            }
        }

        // Text message events already emitted naturally by process_chunk during stream processing
        // (OpenAI router approach - text only appears on final iteration when no tool calls)

        // Emit final response.completed event
        let usage_json = accumulated_response.usage.as_ref().map(|u| {
            json!({
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens
            })
        });
        let event = emitter.emit_completed(usage_json.as_ref());
        emitter.send_event(&event, sink)?;

        let final_response = emitter.finalize(accumulated_response.usage.clone());
        persist_response_if_needed(
            ctx.conversation_storage.clone(),
            ctx.conversation_item_storage.clone(),
            ctx.response_storage.clone(),
            &final_response,
            original_request,
        )
        .await;

        return Ok(final_response);
    }
}

/// Convert chat stream to Responses API events while accumulating for tool call detection.
///
/// Uses the shared `SseBodyParser` so the SSE-byte-parsing logic lives in one
/// place (removing the architectural violation flagged in RFC 0001 / T0).
async fn convert_and_accumulate_stream(
    body: Body,
    emitter: &mut ResponseStreamEventEmitter,
    sink: &impl ResponseEventSink,
) -> Result<ChatCompletionResponse, String> {
    let mut accumulator = ChatResponseAccumulator::new();
    let mut stream = body.into_data_stream();
    let mut parser = SseBodyParser::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;
        parser.push(&chunk);

        while let Some(record) = parser.next_record() {
            match record {
                SseDataRecord::Done => return Ok(accumulator.finalize()),
                SseDataRecord::ChatChunk(chat_chunk) => {
                    emitter.process_chunk(&chat_chunk, sink)?;
                    accumulator.process_chunk(&chat_chunk);
                }
                SseDataRecord::RawJson(_) => {
                    // MCP path ignores non-chat payloads (upstream errors are
                    // handled at the tool-loop level).
                }
            }
        }
    }

    // Flush any remaining partial record.
    if let Some(SseDataRecord::ChatChunk(chat_chunk)) = parser.flush() {
        emitter.process_chunk(&chat_chunk, sink)?;
        accumulator.process_chunk(&chat_chunk);
    }

    Ok(accumulator.finalize())
}

/// Accumulates chat streaming chunks into complete ChatCompletionResponse
struct ChatResponseAccumulator {
    id: String,
    model: String,
    content: String,
    reasoning_content: Option<String>,
    tool_calls: HashMap<usize, ToolCall>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

impl ChatResponseAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: HashMap::new(),
            finish_reason: None,
            usage: None,
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

            // Accumulate reasoning content
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_content
                    .get_or_insert_with(String::new)
                    .push_str(reasoning);
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

        // Update usage
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ChatCompletionResponse {
        let mut tool_calls_vec: Vec<_> = self.tool_calls.into_iter().collect();
        tool_calls_vec.sort_by_key(|(index, _)| *index);
        let tool_calls: Vec<_> = tool_calls_vec.into_iter().map(|(_, call)| call).collect();

        ChatCompletionResponse::builder(&self.id, &self.model)
            .choices(vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
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
                    reasoning_content: self.reasoning_content,
                },
                finish_reason: self.finish_reason,
                logprobs: None,
                matched_stop: None,
                hidden_states: None,
            }])
            .maybe_usage(self.usage)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::body::Body;
    use bytes::Bytes;
    use data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        ResponseId, ResponseStorage,
    };
    use futures_util::stream;
    use serde_json::Value;

    use super::*;
    use crate::protocols::responses::{ResponseInput, ServiceTier, Truncation};

    #[derive(Clone, Default)]
    struct RecordingSink {
        events: Arc<Mutex<Vec<Value>>>,
    }

    impl RecordingSink {
        fn events(&self) -> Vec<Value> {
            self.events.lock().unwrap().clone()
        }
    }

    impl ResponseEventSink for RecordingSink {
        fn send_event(&self, event: &Value) -> Result<(), String> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }

        fn send_raw_json(&self, payload: &str) -> Result<(), String> {
            let value = serde_json::from_str::<Value>(payload)
                .map_err(|err| format!("failed to parse test payload: {}", err))?;
            self.events.lock().unwrap().push(value);
            Ok(())
        }
    }

    fn test_responses_request() -> ResponsesRequest {
        ResponsesRequest {
            background: Some(false),
            include: None,
            input: ResponseInput::Text("trigger upstream error".to_string()),
            instructions: None,
            max_output_tokens: Some(64),
            max_tool_calls: None,
            metadata: None,
            model: "mock-model".to_string(),
            parallel_tool_calls: Some(true),
            previous_response_id: None,
            reasoning: None,
            service_tier: Some(ServiceTier::Auto),
            store: Some(true),
            stream: Some(true),
            temperature: Some(0.0),
            tool_choice: None,
            tools: None,
            top_logprobs: Some(0),
            top_p: None,
            truncation: Some(Truncation::Disabled),
            text: None,
            user: None,
            request_id: Some("resp_stream_upstream_error".to_string()),
            priority: 0,
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            stop: None,
            top_k: -1,
            min_p: 0.0,
            repetition_penalty: 1.0,
            conversation: None,
        }
    }

    #[tokio::test]
    async fn test_process_and_transform_stream_does_not_emit_completed_after_upstream_error() {
        let body = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(
                "data: {\"error\":{\"message\":\"upstream exploded\",\"type\":\"internal_error\"}}\n\n",
            )),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ]));
        let response_storage = Arc::new(MemoryResponseStorage::new());
        let conversation_storage = Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());
        let sink = RecordingSink::default();

        let final_response = process_and_transform_stream(
            body,
            test_responses_request(),
            response_storage.clone(),
            conversation_storage,
            conversation_item_storage,
            &sink,
        )
        .await
        .expect("stream processing should succeed after forwarding upstream error");

        let events = sink.events();
        let event_types: Vec<_> = events
            .iter()
            .filter_map(|event| event.get("type").and_then(|value| value.as_str()))
            .collect();

        assert_eq!(
            event_types,
            vec!["response.created", "response.in_progress"],
            "only the synthetic start events should be emitted before the upstream error",
        );
        assert!(
            events.iter().any(|event| event
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(|value| value.as_str())
                == Some("upstream exploded")),
            "the upstream error payload should be forwarded as-is",
        );
        assert_eq!(final_response.status, ResponseStatus::Failed);

        let response_id = ResponseId::from(final_response.id.as_str());
        assert!(
            response_storage
                .get_response(&response_id)
                .await
                .expect("storage lookup should succeed")
                .is_none(),
            "failed upstream streams should not be persisted",
        );
    }

    #[tokio::test]
    async fn test_process_and_transform_stream_emits_function_call_events_for_non_mcp_streams() {
        let body = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_test\",\"function\":{\"name\":\"get_weather\"},\"type\":\"function\"}]},\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"location\\\":\\\"Berlin\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\\n]\"},\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n\n",
            )),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ]));
        let response_storage = Arc::new(MemoryResponseStorage::new());
        let conversation_storage = Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());
        let sink = RecordingSink::default();

        let final_response = process_and_transform_stream(
            body,
            test_responses_request(),
            response_storage,
            conversation_storage,
            conversation_item_storage,
            &sink,
        )
        .await
        .expect("tool-call stream processing should succeed");

        let events = sink.events();
        let event_types: Vec<_> = events
            .iter()
            .filter_map(|event| event.get("type").and_then(|value| value.as_str()))
            .collect();

        assert_eq!(
            event_types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        );
        assert_eq!(events[2]["item"]["type"], "function_call");
        assert_eq!(events[2]["item"]["call_id"], "call_test");
        assert_eq!(events[3]["delta"], "{\"location\":\"Berlin\"}");
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["type"],
            "function_call"
        );
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["call_id"],
            "call_test"
        );
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["arguments"],
            "{\"location\":\"Berlin\"}"
        );
        assert_eq!(events.last().unwrap()["response"]["status"], "in_progress");

        assert_eq!(final_response.status, ResponseStatus::InProgress);
        assert_eq!(final_response.output.len(), 1);
        let serialized_output = serde_json::to_value(&final_response.output[0]).unwrap();
        assert_eq!(serialized_output["type"], "function_call");
        assert_eq!(serialized_output["call_id"], "call_test");
        assert_eq!(serialized_output["arguments"], "{\"location\":\"Berlin\"}");
    }

    #[tokio::test]
    async fn test_process_and_transform_stream_handles_coalesced_sse_records() {
        let body = Body::from_stream(stream::iter(vec![Ok::<_, std::io::Error>(
            Bytes::from(
                concat!(
                    "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_test\",\"function\":{\"name\":\"calculate\"},\"type\":\"function\"}]},\"finish_reason\":null}]}\n\n",
                    "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"expression\\\":\\\"42 * 17\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
                    "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\\n]\"},\"finish_reason\":null}]}\n\n",
                    "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":6,\"total_tokens\":14}}\n\n",
                    "data: [DONE]\n\n"
                ),
            ),
        )]));
        let response_storage = Arc::new(MemoryResponseStorage::new());
        let conversation_storage = Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());
        let sink = RecordingSink::default();

        let final_response = process_and_transform_stream(
            body,
            test_responses_request(),
            response_storage,
            conversation_storage,
            conversation_item_storage,
            &sink,
        )
        .await
        .expect("coalesced SSE record processing should succeed");

        let events = sink.events();
        let event_types: Vec<_> = events
            .iter()
            .filter_map(|event| event.get("type").and_then(|value| value.as_str()))
            .collect();

        assert_eq!(
            event_types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        );
        assert_eq!(events[3]["delta"], "{\"expression\":\"42 * 17\"}");
        assert_eq!(events.last().unwrap()["response"]["status"], "in_progress");
        assert_eq!(final_response.status, ResponseStatus::InProgress);
        assert_eq!(final_response.output.len(), 1);
        let serialized_output = serde_json::to_value(&final_response.output[0]).unwrap();
        assert_eq!(serialized_output["type"], "function_call");
        assert_eq!(serialized_output["call_id"], "call_test");
        assert_eq!(serialized_output["name"], "calculate");
        assert_eq!(
            serialized_output["arguments"],
            "{\"expression\":\"42 * 17\"}"
        );
    }

    // ------------------------------------------------------------------
    // T3: MCP path — coalesced / split SSE chunk handling
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_convert_and_accumulate_stream_handles_coalesced_sse_records() {
        // All SSE records arrive in a single body chunk.
        let body = Body::from_stream(stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(
            concat!(
                "data: {\"id\":\"chatcmpl_mcp\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n",
                "data: {\"id\":\"chatcmpl_mcp\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
                "data: {\"id\":\"chatcmpl_mcp\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
                "data: [DONE]\n\n"
            ),
        ))]));

        let sink = RecordingSink::default();
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_mcp_test".into(), "mock-model".into(), 1);
        let created = emitter.emit_created();
        emitter.send_event(&created, &sink).unwrap();
        let in_prog = emitter.emit_in_progress();
        emitter.send_event(&in_prog, &sink).unwrap();

        let result = convert_and_accumulate_stream(body, &mut emitter, &sink)
            .await
            .expect("coalesced MCP stream should succeed");

        assert_eq!(result.choices[0].message.content.as_deref(), Some("hello"));
        assert!(result.usage.is_some());

        let events = sink.events();
        let text_deltas: Vec<&str> = events
            .iter()
            .filter(|e| {
                e.get("type").and_then(|t| t.as_str()) == Some("response.output_text.delta")
            })
            .filter_map(|e| e.get("delta").and_then(|d| d.as_str()))
            .collect();
        assert_eq!(text_deltas, vec!["hel", "lo"]);
    }

    #[tokio::test]
    async fn test_convert_and_accumulate_stream_handles_split_sse_records() {
        // One SSE record is split across two body chunks.
        let first_half = "data: {\"id\":\"chatcmpl_split\",\"object\":\"chat.completion.chunk\",";
        let second_half = "\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"split\"},\"finish_reason\":null}]}\n\n\
                           data: {\"id\":\"chatcmpl_split\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                           data: [DONE]\n\n";

        let body = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(first_half)),
            Ok(Bytes::from(second_half)),
        ]));

        let sink = RecordingSink::default();
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_split_test".into(), "mock-model".into(), 1);
        let created = emitter.emit_created();
        emitter.send_event(&created, &sink).unwrap();
        let in_prog = emitter.emit_in_progress();
        emitter.send_event(&in_prog, &sink).unwrap();

        let result = convert_and_accumulate_stream(body, &mut emitter, &sink)
            .await
            .expect("split MCP stream should succeed");

        assert_eq!(result.choices[0].message.content.as_deref(), Some("split"));

        let events = sink.events();
        let text_deltas: Vec<&str> = events
            .iter()
            .filter(|e| {
                e.get("type").and_then(|t| t.as_str()) == Some("response.output_text.delta")
            })
            .filter_map(|e| e.get("delta").and_then(|d| d.as_str()))
            .collect();
        assert_eq!(text_deltas, vec!["split"]);
    }
}
