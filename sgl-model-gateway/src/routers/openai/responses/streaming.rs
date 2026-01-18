//! Streaming response handling for OpenAI-compatible responses
//!
//! This module handles all streaming-related functionality including:
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Streaming response accumulation for persistence
//! - Tool call detection and interception during streaming
//! - MCP tool execution loops within streaming responses
//! - Event transformation and output index remapping

use std::{borrow::Cow, io, sync::Arc};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;

use super::{
    accumulator::StreamingResponseAccumulator,
    common::{extract_output_index, get_event_type, parse_sse_block, ChunkProcessor},
    mcp::{
        build_mcp_list_tools_items_for_request, build_resume_payload, execute_streaming_tool_calls,
        inject_mcp_metadata_streaming, prepare_mcp_tools_as_functions, send_mcp_list_tools_events,
        ToolLoopState,
    },
    tool_handler::{StreamAction, StreamingToolHandler},
    utils::{mask_tools_as_mcp, patch_response_with_request_metadata, rewrite_streaming_block},
};
use crate::{
    protocols::{
        event_types::{
            is_function_call_type, is_response_event, FunctionCallEvent, ItemType, McpEvent,
            OutputItemEvent, ResponseEvent,
        },
        responses::{ResponseToolType, ResponsesRequest},
    },
    routers::{
        header_utils::{apply_request_headers, preserve_response_headers},
        mcp_utils::{
            build_allowed_tools_map, build_mcp_tool_lookup, build_server_label_map,
            decode_mcp_function_name, ensure_request_mcp_client, McpLoopConfig,
        },
        openai::context::{RequestContext, StreamingEventContext, StreamingRequest},
        persistence_utils::persist_conversation_items,
    },
};

// ============================================================================
// Event Transformation and Forwarding
// ============================================================================

/// Apply all transformations to event data in-place (rewrite + transform)
/// Optimized to parse JSON only once instead of multiple times
/// Returns true if any changes were made
pub(super) fn apply_event_transformations_inplace(
    parsed_data: &mut Value,
    ctx: &StreamingEventContext<'_>,
) -> bool {
    let mut changed = false;

    // 1. Apply rewrite_streaming_block logic (store, previous_response_id, tools masking)
    // Get event_type as owned String to avoid borrow conflict with mutable operations below
    let event_type = parsed_data
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let should_patch = is_response_event(event_type);
    // Need owned copy for the match below since we mutate parsed_data
    let event_type = event_type.to_string();

    if should_patch {
        if let Some(response_obj) = parsed_data
            .get_mut("response")
            .and_then(|v| v.as_object_mut())
        {
            let desired_store = Value::Bool(ctx.original_request.store.unwrap_or(false));
            if response_obj.get("store") != Some(&desired_store) {
                response_obj.insert("store".to_string(), desired_store);
                changed = true;
            }

            if let Some(prev_id) = ctx.previous_response_id {
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

            // Mask tools from function to MCP format (optimized without cloning)
            if response_obj.get("tools").is_some() {
                let requested_mcp = ctx
                    .original_request
                    .tools
                    .as_ref()
                    .map(|tools| {
                        tools
                            .iter()
                            .any(|t| matches!(t.r#type, ResponseToolType::Mcp))
                    })
                    .unwrap_or(false);

                if requested_mcp {
                    if let Some(mcp_tools) = build_mcp_tools_value(ctx.original_request) {
                        response_obj.insert("tools".to_string(), mcp_tools);
                        response_obj
                            .entry("tool_choice".to_string())
                            .or_insert(Value::String("auto".to_string()));
                        changed = true;
                    }
                }
            }
        }
    }

    // 2. Apply transform_streaming_event logic (function_call → mcp_call)
    match event_type.as_str() {
        OutputItemEvent::ADDED | OutputItemEvent::DONE => {
            if let Some(item) = parsed_data.get_mut("item") {
                if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                    if is_function_call_type(item_type) {
                        item["type"] = json!(ItemType::MCP_CALL);

                        if let Some(encoded_name) = item.get("name").and_then(|v| v.as_str()) {
                            if let Some((server_label, tool_name)) =
                                decode_mcp_function_name(encoded_name)
                            {
                                item["server_label"] = json!(server_label);
                                item["name"] = json!(tool_name);
                            } else {
                                item["server_label"] = json!(ctx.server_label);
                            }
                        } else {
                            item["server_label"] = json!(ctx.server_label);
                        }

                        // Transform ID from fc_* to mcp_*
                        if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                            if let Some(stripped) = id.strip_prefix("fc_") {
                                let new_id = format!("mcp_{}", stripped);
                                item["id"] = json!(new_id);
                            }
                        }

                        changed = true;
                    }
                }
            }
        }
        FunctionCallEvent::ARGUMENTS_DONE => {
            parsed_data["type"] = json!(McpEvent::CALL_ARGUMENTS_DONE);

            // Transform item_id from fc_* to mcp_*
            if let Some(item_id) = parsed_data.get("item_id").and_then(|v| v.as_str()) {
                if let Some(stripped) = item_id.strip_prefix("fc_") {
                    let new_id = format!("mcp_{}", stripped);
                    parsed_data["item_id"] = json!(new_id);
                }
            }

            changed = true;
        }
        _ => {}
    }

    changed
}

/// Helper to build MCP tools value
fn build_mcp_tools_value(original_body: &ResponsesRequest) -> Option<Value> {
    let tools = original_body.tools.as_ref()?;
    let tools_array: Vec<Value> = tools
        .iter()
        .filter(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some())
        .map(|mcp_tool| {
            json!({
                "type": "mcp",
                "server_label": mcp_tool.server_label,
                "server_url": mcp_tool.server_url,
                "server_description": mcp_tool.server_description,
                "require_approval": mcp_tool.require_approval,
                "allowed_tools": mcp_tool.allowed_tools,
                "headers": mcp_tool.headers,
            })
        })
        .collect();

    if tools_array.is_empty() {
        None
    } else {
        Some(Value::Array(tools_array))
    }
}

/// Send an SSE event to the client channel
/// Returns false if client disconnected
#[inline]
fn send_sse_event(
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    event_name: &str,
    data: &Value,
) -> bool {
    let block = format!("event: {}\ndata: {}\n\n", event_name, data);
    tx.send(Ok(Bytes::from(block))).is_ok()
}

/// Transform fc_* item IDs to mcp_* format
#[inline]
fn transform_fc_to_mcp_id(item_id: &str) -> String {
    item_id
        .strip_prefix("fc_")
        .map(|stripped| format!("mcp_{}", stripped))
        .unwrap_or_else(|| item_id.to_string())
}

/// Map function_call event names to mcp_call event names
#[inline]
fn map_event_name(event_name: &str) -> &str {
    match event_name {
        FunctionCallEvent::ARGUMENTS_DELTA => McpEvent::CALL_ARGUMENTS_DELTA,
        FunctionCallEvent::ARGUMENTS_DONE => McpEvent::CALL_ARGUMENTS_DONE,
        other => other,
    }
}

/// Send buffered function call arguments as a synthetic delta event.
/// Returns false if client disconnected.
fn send_buffered_arguments(
    parsed_data: &mut Value,
    handler: &StreamingToolHandler,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    sequence_number: &mut u64,
    mapped_output_index: &mut Option<usize>,
) -> bool {
    let Some(output_index) = extract_output_index(parsed_data) else {
        return true;
    };

    let assigned_index = handler
        .mapped_output_index(output_index)
        .unwrap_or(output_index);
    *mapped_output_index = Some(assigned_index);

    let Some(call) = handler
        .pending_calls
        .iter()
        .find(|c| c.output_index == output_index)
    else {
        return true;
    };

    let arguments_value = if call.arguments_buffer.is_empty() {
        "{}".to_string()
    } else {
        call.arguments_buffer.clone()
    };

    // Update the done event with full arguments
    parsed_data["arguments"] = Value::String(arguments_value.clone());

    // Transform item_id
    let item_id = parsed_data
        .get("item_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let mcp_item_id = transform_fc_to_mcp_id(item_id);

    // Build synthetic delta event
    let mut delta_event = json!({
        "type": McpEvent::CALL_ARGUMENTS_DELTA,
        "sequence_number": *sequence_number,
        "output_index": assigned_index,
        "item_id": mcp_item_id,
        "delta": arguments_value,
    });

    // Add obfuscation if present
    let obfuscation = call
        .last_obfuscation
        .as_ref()
        .map(|s| Value::String(s.clone()))
        .or_else(|| parsed_data.get("obfuscation").cloned());

    if let Some(obf) = obfuscation {
        if let Some(obj) = delta_event.as_object_mut() {
            obj.insert("obfuscation".to_string(), obf);
        }
    }

    if !send_sse_event(tx, McpEvent::CALL_ARGUMENTS_DELTA, &delta_event) {
        return false;
    }

    *sequence_number += 1;
    true
}

/// Forward and transform a streaming event to the client
/// Returns false if client disconnected
pub(super) fn forward_streaming_event(
    raw_block: &str,
    event_name: Option<&str>,
    data: &str,
    handler: &mut StreamingToolHandler,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ctx: &StreamingEventContext<'_>,
    sequence_number: &mut u64,
) -> bool {
    // Skip individual function_call_arguments.delta events - we'll send them as one
    if event_name == Some(FunctionCallEvent::ARGUMENTS_DELTA) {
        return true;
    }

    // Parse JSON data once
    let mut parsed_data: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => {
            let chunk = format!("{}\n\n", raw_block);
            return tx.send(Ok(Bytes::from(chunk))).is_ok();
        }
    };

    let event_type = get_event_type(event_name, &parsed_data);
    if event_type == ResponseEvent::COMPLETED {
        return true;
    }

    // Handle function_call_arguments.done - send buffered args first
    let mut mapped_output_index: Option<usize> = None;
    if event_name == Some(FunctionCallEvent::ARGUMENTS_DONE)
        && !send_buffered_arguments(
            &mut parsed_data,
            handler,
            tx,
            sequence_number,
            &mut mapped_output_index,
        )
    {
        return false;
    }

    // Remap output_index for sequential downstream indices
    if mapped_output_index.is_none() {
        if let Some(idx) = extract_output_index(&parsed_data) {
            mapped_output_index = handler.mapped_output_index(idx);
        }
    }
    if let Some(mapped) = mapped_output_index {
        parsed_data["output_index"] = json!(mapped);
    }

    // Apply transformations
    apply_event_transformations_inplace(&mut parsed_data, ctx);

    // Restore original response ID
    if let Some(response_obj) = parsed_data
        .get_mut("response")
        .and_then(|v| v.as_object_mut())
    {
        if let Some(original_id) = handler.original_response_id() {
            response_obj.insert("id".to_string(), Value::String(original_id.to_string()));
        }
    }

    // Update sequence number
    if parsed_data.get("sequence_number").is_some() {
        parsed_data["sequence_number"] = json!(*sequence_number);
        *sequence_number += 1;
    }

    // Serialize and send
    let final_data = match serde_json::to_string(&parsed_data) {
        Ok(s) => s,
        Err(_) => {
            let chunk = format!("{}\n\n", raw_block);
            return tx.send(Ok(Bytes::from(chunk))).is_ok();
        }
    };

    // Build SSE block with transformed event name
    let final_block = match event_name {
        Some(evt) => format!("event: {}\ndata: {}\n\n", map_event_name(evt), final_data),
        None => format!("data: {}\n\n", final_data),
    };

    if tx.send(Ok(Bytes::from(final_block))).is_err() {
        return false;
    }

    // After sending output_item.added for mcp_call, inject mcp_call.in_progress event
    if event_name == Some(OutputItemEvent::ADDED)
        && !maybe_inject_mcp_in_progress(&parsed_data, tx, sequence_number)
    {
        return false;
    }

    true
}

/// Inject mcp_call.in_progress event after an mcp_call item is added.
/// Returns false if client disconnected.
fn maybe_inject_mcp_in_progress(
    parsed_data: &Value,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    sequence_number: &mut u64,
) -> bool {
    let Some(item) = parsed_data.get("item") else {
        return true;
    };

    if item.get("type").and_then(|v| v.as_str()) != Some(ItemType::MCP_CALL) {
        return true;
    }

    let Some(item_id) = item.get("id").and_then(|v| v.as_str()) else {
        return true;
    };
    let Some(output_index) = parsed_data.get("output_index").and_then(|v| v.as_u64()) else {
        return true;
    };

    let event = json!({
        "type": McpEvent::CALL_IN_PROGRESS,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;

    send_sse_event(tx, McpEvent::CALL_IN_PROGRESS, &event)
}

/// Send final response.completed event to client
/// Returns false if client disconnected
pub(super) fn send_final_response_event(
    handler: &StreamingToolHandler,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    sequence_number: &mut u64,
    state: &ToolLoopState,
    active_mcp: Option<&Arc<crate::mcp::McpManager>>,
    ctx: &StreamingEventContext<'_>,
) -> bool {
    let mut final_response = match handler.snapshot_final_response() {
        Some(resp) => resp,
        None => {
            warn!("Final response snapshot unavailable; skipping synthetic completion event");
            return true;
        }
    };

    if let Some(original_id) = handler.original_response_id() {
        if let Some(obj) = final_response.as_object_mut() {
            obj.insert("id".to_string(), Value::String(original_id.to_string()));
        }
    }

    if let Some(mcp) = active_mcp {
        inject_mcp_metadata_streaming(
            &mut final_response,
            state,
            mcp,
            ctx.original_request.tools.as_deref(),
            ctx.server_keys,
        );
    }

    mask_tools_as_mcp(&mut final_response, ctx.original_request);
    patch_response_with_request_metadata(
        &mut final_response,
        ctx.original_request,
        ctx.previous_response_id,
    );

    if let Some(obj) = final_response.as_object_mut() {
        obj.insert("status".to_string(), Value::String("completed".to_string()));
    }

    let completed_payload = json!({
        "type": ResponseEvent::COMPLETED,
        "sequence_number": *sequence_number,
        "response": final_response
    });
    *sequence_number += 1;

    let completed_event = format!(
        "event: {}\ndata: {}\n\n",
        ResponseEvent::COMPLETED,
        completed_payload
    );
    tx.send(Ok(Bytes::from(completed_event))).is_ok()
}

// ============================================================================
// Main Streaming Handlers
// ============================================================================

/// Simple pass-through streaming without MCP interception
pub(super) async fn handle_simple_streaming_passthrough(
    client: &reqwest::Client,
    circuit_breaker: &crate::core::CircuitBreaker,
    headers: Option<&HeaderMap>,
    req: StreamingRequest,
) -> Response {
    let mut request_builder = client.post(&req.url).json(&req.payload);

    if let Some(headers) = headers {
        request_builder = apply_request_headers(headers, request_builder, true);
    }

    request_builder = request_builder.header("Accept", "text/event-stream");

    let response = match request_builder.send().await {
        Ok(resp) => resp,
        Err(err) => {
            circuit_breaker.record_failure();
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
        circuit_breaker.record_failure();
        let error_body = response
            .text()
            .await
            .unwrap_or_else(|err| format!("Failed to read upstream error body: {}", err));
        return (status_code, error_body).into_response();
    }

    circuit_breaker.record_success();

    let preserved_headers = preserve_response_headers(response.headers());
    let mut upstream_stream = response.bytes_stream();

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

    let should_store = req.original_body.store.unwrap_or(false);
    let original_request = req.original_body;
    let persist_needed = original_request.conversation.is_some();
    let previous_response_id = req.previous_response_id;
    let storage = req.storage;

    tokio::spawn(async move {
        let mut accumulator = StreamingResponseAccumulator::new();
        let mut upstream_failed = false;
        let mut receiver_connected = true;
        let mut chunk_processor = ChunkProcessor::new();

        while let Some(chunk_result) = upstream_stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    chunk_processor.push_chunk(&chunk);

                    while let Some(raw_block) = chunk_processor.next_block() {
                        let block_cow = match rewrite_streaming_block(
                            &raw_block,
                            &original_request,
                            previous_response_id.as_deref(),
                        ) {
                            Some(modified) => Cow::Owned(modified),
                            None => Cow::Borrowed(raw_block.as_str()),
                        };

                        if should_store || persist_needed {
                            accumulator.ingest_block(&block_cow);
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

        if (should_store || persist_needed) && !upstream_failed {
            if chunk_processor.has_remaining() {
                accumulator.ingest_block(&chunk_processor.take_remaining());
            }
            let encountered_error = accumulator.encountered_error().cloned();
            if let Some(mut response_json) = accumulator.into_final_response() {
                patch_response_with_request_metadata(
                    &mut response_json,
                    &original_request,
                    previous_response_id.as_deref(),
                );

                // Always persist conversation items and response (even without conversation)
                if let Err(err) = persist_conversation_items(
                    storage.conversation.clone(),
                    storage.conversation_item.clone(),
                    storage.response.clone(),
                    &response_json,
                    &original_request,
                )
                .await
                {
                    warn!("Failed to persist conversation items (stream): {}", err);
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

/// Handle streaming WITH MCP tool call interception and execution
pub(super) async fn handle_streaming_with_tool_interception(
    client: &reqwest::Client,
    headers: Option<&HeaderMap>,
    req: StreamingRequest,
    active_mcp: &Arc<crate::mcp::McpManager>,
    server_keys: Vec<String>,
) -> Response {
    // Transform MCP tools to function tools in payload
    let mut payload = req.payload;
    prepare_mcp_tools_as_functions(
        &mut payload,
        active_mcp,
        &server_keys,
        req.original_body.tools.as_deref(),
    );

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
    let should_store = req.original_body.store.unwrap_or(false);
    let original_request = req.original_body;
    let persist_needed = original_request.conversation.is_some();
    let previous_response_id = req.previous_response_id;
    let url = req.url;
    let storage = req.storage;

    let client_clone = client.clone();
    let url_clone = url.clone();
    let headers_opt = headers.cloned();
    let payload_clone = payload.clone();
    let active_mcp_clone = Arc::clone(active_mcp);
    let server_keys_clone = server_keys.clone();

    // Spawn the streaming loop task
    tokio::spawn(async move {
        let mut state = ToolLoopState::new(original_request.input.clone());
        let loop_config = McpLoopConfig {
            server_keys: server_keys_clone.clone(),
            ..McpLoopConfig::default()
        };
        let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);
        let tools_json = payload_clone.get("tools").cloned().unwrap_or(json!([]));
        let base_payload = payload_clone.clone();
        let mut current_payload = payload_clone;
        let mut mcp_list_tools_sent = false;
        let mut is_first_iteration = true;
        let mut sequence_number: u64 = 0;
        let mut next_output_index: usize = 0;
        let mut preserved_response_id: Option<String> = None;

        let server_label = original_request
            .tools
            .as_ref()
            .and_then(|tools| {
                tools
                    .iter()
                    .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                    .and_then(|t| t.server_label.as_deref())
            })
            .unwrap_or("mcp");

        let server_labels = build_server_label_map(original_request.tools.as_deref());
        let allowed_tools = build_allowed_tools_map(original_request.tools.as_deref());
        let tool_lookup = build_mcp_tool_lookup(
            &active_mcp_clone,
            &server_keys_clone,
            &server_labels,
            &allowed_tools,
        );

        let list_tools_items = build_mcp_list_tools_items_for_request(
            &active_mcp_clone,
            &server_keys_clone,
            original_request.tools.as_deref(),
        );
        let streaming_ctx = StreamingEventContext {
            server_label,
            original_request: &original_request,
            previous_response_id: previous_response_id.as_deref(),
            server_keys: &server_keys_clone,
        };

        loop {
            // Make streaming request
            let mut request_builder = client_clone.post(&url_clone).json(&current_payload);
            if let Some(ref h) = headers_opt {
                request_builder = apply_request_headers(h, request_builder, true);
            }
            request_builder = request_builder.header("Accept", "text/event-stream");

            let response = match request_builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    let error_event = format!(
                        "event: error\ndata: {{\"error\": {{\"message\": \"{}\"}}}}\n\n",
                        e
                    );
                    let _ = tx.send(Ok(Bytes::from(error_event)));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let error_event = format!("event: error\ndata: {{\"error\": {{\"message\": \"Upstream error {}: {}\"}}}}\n\n", status, body);
                let _ = tx.send(Ok(Bytes::from(error_event)));
                return;
            }

            // Stream events and check for tool calls
            let mut upstream_stream = response.bytes_stream();
            let mut handler = StreamingToolHandler::with_starting_index(next_output_index);
            if let Some(ref id) = preserved_response_id {
                handler.original_response_id = Some(id.clone());
            }
            let mut chunk_processor = ChunkProcessor::new();
            let mut tool_calls_detected = false;
            let mut seen_in_progress = false;

            while let Some(chunk_result) = upstream_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        chunk_processor.push_chunk(&chunk);

                        while let Some(raw_block) = chunk_processor.next_block() {
                            // Parse event
                            let (event_name, data) = parse_sse_block(&raw_block);

                            if data.is_empty() {
                                continue;
                            }

                            // Process through handler
                            let action = handler.process_event(event_name, data.as_ref());

                            match action {
                                StreamAction::Forward => {
                                    // Skip response.created and response.in_progress on subsequent iterations
                                    let should_skip = if !is_first_iteration {
                                        if let Ok(parsed) =
                                            serde_json::from_str::<Value>(data.as_ref())
                                        {
                                            matches!(
                                                parsed.get("type").and_then(|v| v.as_str()),
                                                Some(ResponseEvent::CREATED)
                                                    | Some(ResponseEvent::IN_PROGRESS)
                                            )
                                        } else {
                                            false
                                        }
                                    } else {
                                        false
                                    };

                                    if !should_skip {
                                        // Forward the event
                                        if !forward_streaming_event(
                                            &raw_block,
                                            event_name,
                                            data.as_ref(),
                                            &mut handler,
                                            &tx,
                                            &streaming_ctx,
                                            &mut sequence_number,
                                        ) {
                                            // Client disconnected
                                            return;
                                        }
                                    }

                                    // After forwarding response.in_progress, send mcp_list_tools events (once)
                                    if !seen_in_progress {
                                        if let Ok(parsed) =
                                            serde_json::from_str::<Value>(data.as_ref())
                                        {
                                            if parsed.get("type").and_then(|v| v.as_str())
                                                == Some(ResponseEvent::IN_PROGRESS)
                                            {
                                                seen_in_progress = true;
                                                if !mcp_list_tools_sent {
                                                    let mut list_tools_indexes =
                                                        Vec::with_capacity(list_tools_items.len());
                                                    for _ in 0..list_tools_items.len() {
                                                        list_tools_indexes.push(
                                                            handler
                                                                .allocate_synthetic_output_index(),
                                                        );
                                                    }

                                                    if !send_mcp_list_tools_events(
                                                        &tx,
                                                        &list_tools_items,
                                                        &list_tools_indexes,
                                                        &mut sequence_number,
                                                    ) {
                                                        // Client disconnected
                                                        return;
                                                    }
                                                    mcp_list_tools_sent = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                StreamAction::Buffer => {
                                    // Don't forward, just buffer
                                }
                                StreamAction::ExecuteTools => {
                                    if !forward_streaming_event(
                                        &raw_block,
                                        event_name,
                                        data.as_ref(),
                                        &mut handler,
                                        &tx,
                                        &streaming_ctx,
                                        &mut sequence_number,
                                    ) {
                                        // Client disconnected
                                        return;
                                    }
                                    tool_calls_detected = true;
                                    break; // Exit stream processing to execute tools
                                }
                            }
                        }

                        if tool_calls_detected {
                            break;
                        }
                    }
                    Err(e) => {
                        let error_event = format!("event: error\ndata: {{\"error\": {{\"message\": \"Stream error: {}\"}}}}\n\n", e);
                        let _ = tx.send(Ok(Bytes::from(error_event)));
                        return;
                    }
                }
            }

            next_output_index = handler.next_output_index();
            if let Some(id) = handler.original_response_id().map(|s| s.to_string()) {
                preserved_response_id = Some(id);
            }

            // If no tool calls, we're done - stream is complete
            if !tool_calls_detected {
                if !send_final_response_event(
                    &handler,
                    &tx,
                    &mut sequence_number,
                    &state,
                    Some(&active_mcp_clone),
                    &streaming_ctx,
                ) {
                    return;
                }

                let final_response_json = if should_store || persist_needed {
                    handler.accumulator.into_final_response()
                } else {
                    None
                };

                if let Some(mut response_json) = final_response_json {
                    if let Some(ref id) = preserved_response_id {
                        if let Some(obj) = response_json.as_object_mut() {
                            obj.insert("id".to_string(), Value::String(id.clone()));
                        }
                    }
                    mask_tools_as_mcp(&mut response_json, &original_request);
                    patch_response_with_request_metadata(
                        &mut response_json,
                        &original_request,
                        previous_response_id.as_deref(),
                    );

                    // Always persist conversation items and response (even without conversation)
                    if let Err(err) = persist_conversation_items(
                        storage.conversation.clone(),
                        storage.conversation_item.clone(),
                        storage.response.clone(),
                        &response_json,
                        &original_request,
                    )
                    .await
                    {
                        warn!(
                            "Failed to persist conversation items (stream + MCP): {}",
                            err
                        );
                    }
                }

                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                return;
            }

            // Execute tools
            let pending_calls = handler.take_pending_calls();

            // Check iteration limit
            state.iteration += 1;
            state.total_calls += pending_calls.len();

            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(loop_config.max_iterations),
                None => loop_config.max_iterations,
            };

            if state.total_calls > effective_limit {
                warn!(
                    "Reached tool call limit during streaming: {}",
                    effective_limit
                );
                let error_event = "event: error\ndata: {\"error\": {\"message\": \"Exceeded max_tool_calls limit\"}}\n\n".to_string();
                let _ = tx.send(Ok(Bytes::from(error_event)));
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                return;
            }

            // Execute all pending tool calls
            if !execute_streaming_tool_calls(
                pending_calls,
                &active_mcp_clone,
                &tx,
                &mut state,
                &tool_lookup,
                &mut sequence_number,
            )
            .await
            {
                // Client disconnected during tool execution
                return;
            }

            // Build resume payload
            match build_resume_payload(
                &base_payload,
                &state.conversation_history,
                &state.original_input,
                &tools_json,
                true, // is_streaming = true
            ) {
                Ok(resume_payload) => {
                    current_payload = resume_payload;
                    // Mark that we're no longer on the first iteration
                    is_first_iteration = false;
                    // Continue loop to make next streaming request
                }
                Err(e) => {
                    let error_event = format!("event: error\ndata: {{\"error\": {{\"message\": \"Failed to build resume payload: {}\"}}}}\n\n", e);
                    let _ = tx.send(Ok(Bytes::from(error_event)));
                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                    return;
                }
            }
        }
    });

    let body_stream = UnboundedReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
}

/// Main entry point for streaming responses
pub async fn handle_streaming_response(ctx: RequestContext) -> Response {
    let worker = ctx.worker().expect("Worker not selected").clone();
    let circuit_breaker = worker.circuit_breaker();
    let headers = ctx.headers().cloned();
    let original_body = ctx.responses_request();
    let mcp_manager = ctx.components.mcp_manager().expect("MCP manager required");

    let server_keys = match original_body.tools.as_ref() {
        Some(tools) => match ensure_request_mcp_client(mcp_manager, tools.as_slice()).await {
            Some((_manager, keys)) => keys,
            None => Vec::new(),
        },
        None => Vec::new(),
    };

    let active_mcp = if mcp_manager.list_tools_for_servers(&server_keys).is_empty() {
        None
    } else {
        Some(mcp_manager.clone())
    };

    let client = ctx.components.client().clone();
    let req = ctx.into_streaming_context();

    if active_mcp.is_none() {
        return handle_simple_streaming_passthrough(
            &client,
            circuit_breaker,
            headers.as_ref(),
            req,
        )
        .await;
    }

    let active_mcp = active_mcp.unwrap();

    // MCP is active - transform tools and set up interception
    handle_streaming_with_tool_interception(
        &client,
        headers.as_ref(),
        req,
        &active_mcp,
        server_keys,
    )
    .await
}
