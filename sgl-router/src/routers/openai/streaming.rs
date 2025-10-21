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

// Import from sibling modules
use super::conversations::persist_conversation_items;
use super::{
    mcp::{
        build_resume_payload, execute_streaming_tool_calls, inject_mcp_metadata_streaming,
        mcp_manager_from_request_tools, prepare_mcp_payload_for_streaming,
        send_mcp_list_tools_events, McpLoopConfig, ToolLoopState,
    },
    responses::{mask_tools_as_mcp, patch_streaming_response_json, rewrite_streaming_block},
    utils::{event_types, FunctionCallInProgress, OutputIndexMapper, StreamAction},
};
use crate::{
    data_connector::{
        SharedConversationItemStorage, SharedConversationStorage, SharedResponseStorage,
    },
    protocols::responses::{ResponseToolType, ResponsesRequest},
    routers::header_utils::{apply_request_headers, preserve_response_headers},
};

// ============================================================================
// Streaming Response Accumulator
// ============================================================================

/// Helper that parses SSE frames from the OpenAI responses stream and
/// accumulates enough information to persist the final response locally.
pub(super) struct StreamingResponseAccumulator {
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
    pub fn new() -> Self {
        Self {
            initial_response: None,
            completed_response: None,
            output_items: Vec::new(),
            encountered_error: None,
        }
    }

    /// Feed the accumulator with the next SSE chunk.
    pub fn ingest_block(&mut self, block: &str) {
        if block.trim().is_empty() {
            return;
        }
        self.process_block(block);
    }

    /// Consume the accumulator and produce the best-effort final response value.
    pub fn into_final_response(mut self) -> Option<Value> {
        if self.completed_response.is_some() {
            return self.completed_response;
        }

        self.build_fallback_response()
    }

    pub fn encountered_error(&self) -> Option<&Value> {
        self.encountered_error.as_ref()
    }

    pub fn original_response_id(&self) -> Option<&str> {
        self.initial_response
            .as_ref()
            .and_then(|response| response.get("id"))
            .and_then(|id| id.as_str())
    }

    pub fn snapshot_final_response(&self) -> Option<Value> {
        if let Some(resp) = &self.completed_response {
            return Some(resp.clone());
        }
        self.build_fallback_response_snapshot()
    }

    fn build_fallback_response_snapshot(&self) -> Option<Value> {
        let mut response = self.initial_response.clone()?;

        if let Some(obj) = response.as_object_mut() {
            obj.insert("status".to_string(), Value::String("completed".to_string()));

            let mut output_items = self.output_items.clone();
            output_items.sort_by_key(|(index, _)| *index);
            let outputs: Vec<Value> = output_items.into_iter().map(|(_, item)| item).collect();
            obj.insert("output".to_string(), Value::Array(outputs));
        }

        Some(response)
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
            event_types::RESPONSE_CREATED => {
                if self.initial_response.is_none() {
                    if let Some(response) = parsed.get("response") {
                        self.initial_response = Some(response.clone());
                    }
                }
            }
            event_types::RESPONSE_COMPLETED => {
                if let Some(response) = parsed.get("response") {
                    self.completed_response = Some(response.clone());
                }
            }
            event_types::OUTPUT_ITEM_DONE => {
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

// ============================================================================
// Streaming Tool Handler
// ============================================================================

/// Handles streaming responses with MCP tool call interception
pub(super) struct StreamingToolHandler {
    /// Accumulator for response persistence
    pub accumulator: StreamingResponseAccumulator,
    /// Function calls being built from deltas
    pub pending_calls: Vec<FunctionCallInProgress>,
    /// Track if we're currently in a function call
    in_function_call: bool,
    /// Manage output_index remapping so they increment per item
    output_index_mapper: OutputIndexMapper,
    /// Original response id captured from the first response.created event
    pub original_response_id: Option<String>,
}

impl StreamingToolHandler {
    pub fn with_starting_index(start: usize) -> Self {
        Self {
            accumulator: StreamingResponseAccumulator::new(),
            pending_calls: Vec::new(),
            in_function_call: false,
            output_index_mapper: OutputIndexMapper::with_start(start),
            original_response_id: None,
        }
    }

    pub fn ensure_output_index(&mut self, upstream_index: usize) -> usize {
        self.output_index_mapper.ensure_mapping(upstream_index)
    }

    pub fn mapped_output_index(&self, upstream_index: usize) -> Option<usize> {
        self.output_index_mapper.lookup(upstream_index)
    }

    pub fn allocate_synthetic_output_index(&mut self) -> usize {
        self.output_index_mapper.allocate_synthetic()
    }

    pub fn next_output_index(&self) -> usize {
        self.output_index_mapper.next_index()
    }

    pub fn original_response_id(&self) -> Option<&str> {
        self.original_response_id
            .as_deref()
            .or_else(|| self.accumulator.original_response_id())
    }

    pub fn snapshot_final_response(&self) -> Option<Value> {
        self.accumulator.snapshot_final_response()
    }

    /// Process an SSE event and determine what action to take
    pub fn process_event(&mut self, event_name: Option<&str>, data: &str) -> StreamAction {
        // Always feed to accumulator for storage
        self.accumulator.ingest_block(&format!(
            "{}data: {}",
            event_name
                .map(|n| format!("event: {}\n", n))
                .unwrap_or_default(),
            data
        ));

        let parsed: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return StreamAction::Forward,
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
            event_types::RESPONSE_CREATED => {
                if self.original_response_id.is_none() {
                    if let Some(response_obj) = parsed.get("response").and_then(|v| v.as_object()) {
                        if let Some(id) = response_obj.get("id").and_then(|v| v.as_str()) {
                            self.original_response_id = Some(id.to_string());
                        }
                    }
                }
                StreamAction::Forward
            }
            event_types::RESPONSE_COMPLETED => StreamAction::Forward,
            event_types::OUTPUT_ITEM_ADDED => {
                if let Some(idx) = parsed.get("output_index").and_then(|v| v.as_u64()) {
                    self.ensure_output_index(idx as usize);
                }

                // Check if this is a function_call item being added
                if let Some(item) = parsed.get("item") {
                    if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                        if item_type == event_types::ITEM_TYPE_FUNCTION_CALL
                            || item_type == event_types::ITEM_TYPE_FUNCTION_TOOL_CALL
                        {
                            match parsed.get("output_index").and_then(|v| v.as_u64()) {
                                Some(idx) => {
                                    let output_index = idx as usize;
                                    let assigned_index = self.ensure_output_index(output_index);
                                    let call_id =
                                        item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                                    let name =
                                        item.get("name").and_then(|v| v.as_str()).unwrap_or("");

                                    // Create or update the function call
                                    let call = self.get_or_create_call(output_index, item);
                                    call.call_id = call_id.to_string();
                                    call.name = name.to_string();
                                    call.assigned_output_index = Some(assigned_index);

                                    self.in_function_call = true;
                                }
                                None => {
                                    warn!(
                                        "Missing output_index in function_call added event, \
                                         forwarding without processing for tool execution"
                                    );
                                }
                            }
                        }
                    }
                }
                StreamAction::Forward
            }
            event_types::FUNCTION_CALL_ARGUMENTS_DELTA => {
                // Accumulate arguments for the function call
                if let Some(output_index) = parsed
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                {
                    let assigned_index = self.ensure_output_index(output_index);
                    if let Some(delta) = parsed.get("delta").and_then(|v| v.as_str()) {
                        if let Some(call) = self
                            .pending_calls
                            .iter_mut()
                            .find(|c| c.output_index == output_index)
                        {
                            call.arguments_buffer.push_str(delta);
                            if let Some(obfuscation) =
                                parsed.get("obfuscation").and_then(|v| v.as_str())
                            {
                                call.last_obfuscation = Some(obfuscation.to_string());
                            }
                            if call.assigned_output_index.is_none() {
                                call.assigned_output_index = Some(assigned_index);
                            }
                        }
                    }
                }
                StreamAction::Forward
            }
            event_types::FUNCTION_CALL_ARGUMENTS_DONE => {
                // Function call arguments complete - check if ready to execute
                if let Some(output_index) = parsed
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                {
                    let assigned_index = self.ensure_output_index(output_index);
                    if let Some(call) = self
                        .pending_calls
                        .iter_mut()
                        .find(|c| c.output_index == output_index)
                    {
                        if call.assigned_output_index.is_none() {
                            call.assigned_output_index = Some(assigned_index);
                        }
                    }
                }

                if self.has_complete_calls() {
                    StreamAction::ExecuteTools
                } else {
                    StreamAction::Forward
                }
            }
            event_types::OUTPUT_ITEM_DELTA => self.process_output_delta(&parsed),
            event_types::OUTPUT_ITEM_DONE => {
                // Check if we have complete function calls ready to execute
                if let Some(output_index) = parsed
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                {
                    self.ensure_output_index(output_index);
                }

                if self.has_complete_calls() {
                    StreamAction::ExecuteTools
                } else {
                    StreamAction::Forward
                }
            }
            _ => StreamAction::Forward,
        }
    }

    /// Process output delta events to detect and accumulate function calls
    fn process_output_delta(&mut self, event: &Value) -> StreamAction {
        let output_index = event
            .get("output_index")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let assigned_index = self.ensure_output_index(output_index);

        let delta = match event.get("delta") {
            Some(d) => d,
            None => return StreamAction::Forward,
        };

        // Check if this is a function call delta
        let item_type = delta.get("type").and_then(|v| v.as_str());

        if item_type == Some(event_types::ITEM_TYPE_FUNCTION_TOOL_CALL)
            || item_type == Some(event_types::ITEM_TYPE_FUNCTION_CALL)
        {
            self.in_function_call = true;

            // Get or create function call for this output index
            let call = self.get_or_create_call(output_index, delta);
            call.assigned_output_index = Some(assigned_index);

            // Accumulate call_id if present
            if let Some(call_id) = delta.get("call_id").and_then(|v| v.as_str()) {
                call.call_id = call_id.to_string();
            }

            // Accumulate name if present
            if let Some(name) = delta.get("name").and_then(|v| v.as_str()) {
                call.name.push_str(name);
            }

            // Accumulate arguments if present
            if let Some(args) = delta.get("arguments").and_then(|v| v.as_str()) {
                call.arguments_buffer.push_str(args);
            }

            if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
                call.last_obfuscation = Some(obfuscation.to_string());
            }

            // Buffer this event, don't forward to client
            return StreamAction::Buffer;
        }

        // Forward non-function-call events
        StreamAction::Forward
    }

    fn get_or_create_call(
        &mut self,
        output_index: usize,
        delta: &Value,
    ) -> &mut FunctionCallInProgress {
        // Find existing call for this output index
        if let Some(pos) = self
            .pending_calls
            .iter()
            .position(|c| c.output_index == output_index)
        {
            return &mut self.pending_calls[pos];
        }

        // Create new call
        let call_id = delta
            .get("call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut call = FunctionCallInProgress::new(call_id, output_index);
        if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
            call.last_obfuscation = Some(obfuscation.to_string());
        }

        self.pending_calls.push(call);
        self.pending_calls
            .last_mut()
            .expect("Just pushed to pending_calls, must have at least one element")
    }

    fn has_complete_calls(&self) -> bool {
        !self.pending_calls.is_empty() && self.pending_calls.iter().all(|c| c.is_complete())
    }

    pub fn take_pending_calls(&mut self) -> Vec<FunctionCallInProgress> {
        std::mem::take(&mut self.pending_calls)
    }
}

// ============================================================================
// SSE Parsing
// ============================================================================

/// Parse an SSE block into event name and data
///
/// Returns borrowed strings when possible to avoid allocations in hot paths.
/// Only allocates when multiple data lines need to be joined.
pub(super) fn parse_sse_block(block: &str) -> (Option<&str>, Cow<'_, str>) {
    let mut event_name: Option<&str> = None;
    let mut data_lines: Vec<&str> = Vec::new();

    for line in block.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start());
        }
    }

    let data = if data_lines.len() == 1 {
        Cow::Borrowed(data_lines[0])
    } else {
        Cow::Owned(data_lines.join("\n"))
    };

    (event_name, data)
}

// ============================================================================
// Event Transformation and Forwarding
// ============================================================================

/// Apply all transformations to event data in-place (rewrite + transform)
/// Optimized to parse JSON only once instead of multiple times
/// Returns true if any changes were made
pub(super) fn apply_event_transformations_inplace(
    parsed_data: &mut Value,
    server_label: &str,
    original_request: &ResponsesRequest,
    previous_response_id: Option<&str>,
) -> bool {
    let mut changed = false;

    // 1. Apply rewrite_streaming_block logic (store, previous_response_id, tools masking)
    let event_type = parsed_data
        .get("type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_default();

    let should_patch = matches!(
        event_type.as_str(),
        event_types::RESPONSE_CREATED
            | event_types::RESPONSE_IN_PROGRESS
            | event_types::RESPONSE_COMPLETED
    );

    if should_patch {
        if let Some(response_obj) = parsed_data
            .get_mut("response")
            .and_then(|v| v.as_object_mut())
        {
            let desired_store = Value::Bool(original_request.store.unwrap_or(false));
            if response_obj.get("store") != Some(&desired_store) {
                response_obj.insert("store".to_string(), desired_store);
                changed = true;
            }

            if let Some(prev_id) = previous_response_id {
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
                let requested_mcp = original_request
                    .tools
                    .as_ref()
                    .map(|tools| {
                        tools
                            .iter()
                            .any(|t| matches!(t.r#type, ResponseToolType::Mcp))
                    })
                    .unwrap_or(false);

                if requested_mcp {
                    if let Some(mcp_tools) = build_mcp_tools_value(original_request) {
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
        event_types::OUTPUT_ITEM_ADDED | event_types::OUTPUT_ITEM_DONE => {
            if let Some(item) = parsed_data.get_mut("item") {
                if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                    if item_type == event_types::ITEM_TYPE_FUNCTION_CALL
                        || item_type == event_types::ITEM_TYPE_FUNCTION_TOOL_CALL
                    {
                        item["type"] = json!(event_types::ITEM_TYPE_MCP_CALL);
                        item["server_label"] = json!(server_label);

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
        event_types::FUNCTION_CALL_ARGUMENTS_DONE => {
            parsed_data["type"] = json!(event_types::MCP_CALL_ARGUMENTS_DONE);

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
    let mcp_tool = tools
        .iter()
        .find(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some())?;

    let tools_array = vec![json!({
        "type": "mcp",
        "server_label": mcp_tool.server_label,
        "server_url": mcp_tool.server_url
    })];

    Some(Value::Array(tools_array))
}

/// Forward and transform a streaming event to the client
/// Returns false if client disconnected
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_streaming_event(
    raw_block: &str,
    event_name: Option<&str>,
    data: &str,
    handler: &mut StreamingToolHandler,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    server_label: &str,
    original_request: &ResponsesRequest,
    previous_response_id: Option<&str>,
    sequence_number: &mut u64,
) -> bool {
    // Skip individual function_call_arguments.delta events - we'll send them as one
    if event_name == Some(event_types::FUNCTION_CALL_ARGUMENTS_DELTA) {
        return true;
    }

    // Parse JSON data once (optimized!)
    let mut parsed_data: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => {
            // If parsing fails, forward raw block as-is
            let chunk_to_send = format!("{}\n\n", raw_block);
            return tx.send(Ok(Bytes::from(chunk_to_send))).is_ok();
        }
    };

    let event_type = event_name
        .or_else(|| parsed_data.get("type").and_then(|v| v.as_str()))
        .unwrap_or("");

    if event_type == event_types::RESPONSE_COMPLETED {
        return true;
    }

    // Check if this is function_call_arguments.done - need to send buffered args first
    let mut mapped_output_index: Option<usize> = None;

    if event_name == Some(event_types::FUNCTION_CALL_ARGUMENTS_DONE) {
        if let Some(output_index) = parsed_data
            .get("output_index")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
        {
            let assigned_index = handler
                .mapped_output_index(output_index)
                .unwrap_or(output_index);
            mapped_output_index = Some(assigned_index);

            if let Some(call) = handler
                .pending_calls
                .iter()
                .find(|c| c.output_index == output_index)
            {
                let arguments_value = if call.arguments_buffer.is_empty() {
                    "{}".to_string()
                } else {
                    call.arguments_buffer.clone()
                };

                // Make sure the done event carries full arguments
                parsed_data["arguments"] = Value::String(arguments_value.clone());

                // Get item_id and transform it
                let item_id = parsed_data
                    .get("item_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let mcp_item_id = if let Some(stripped) = item_id.strip_prefix("fc_") {
                    format!("mcp_{}", stripped)
                } else {
                    item_id.to_string()
                };

                // Emit a synthetic MCP arguments delta event before the done event
                let mut delta_event = json!({
                    "type": event_types::MCP_CALL_ARGUMENTS_DELTA,
                    "sequence_number": *sequence_number,
                    "output_index": assigned_index,
                    "item_id": mcp_item_id,
                    "delta": arguments_value,
                });

                if let Some(obfuscation) = call.last_obfuscation.as_ref() {
                    if let Some(obj) = delta_event.as_object_mut() {
                        obj.insert(
                            "obfuscation".to_string(),
                            Value::String(obfuscation.clone()),
                        );
                    }
                } else if let Some(obfuscation) = parsed_data.get("obfuscation").cloned() {
                    if let Some(obj) = delta_event.as_object_mut() {
                        obj.insert("obfuscation".to_string(), obfuscation);
                    }
                }

                let delta_block = format!(
                    "event: {}\ndata: {}\n\n",
                    event_types::MCP_CALL_ARGUMENTS_DELTA,
                    delta_event
                );
                if tx.send(Ok(Bytes::from(delta_block))).is_err() {
                    return false;
                }

                *sequence_number += 1;
            }
        }
    }

    // Remap output_index (if present) so downstream sees sequential indices
    if mapped_output_index.is_none() {
        if let Some(output_index) = parsed_data
            .get("output_index")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
        {
            mapped_output_index = handler.mapped_output_index(output_index);
        }
    }

    if let Some(mapped) = mapped_output_index {
        parsed_data["output_index"] = json!(mapped);
    }

    // Apply all transformations in-place (single parse/serialize!)
    apply_event_transformations_inplace(
        &mut parsed_data,
        server_label,
        original_request,
        previous_response_id,
    );

    if let Some(response_obj) = parsed_data
        .get_mut("response")
        .and_then(|v| v.as_object_mut())
    {
        if let Some(original_id) = handler.original_response_id() {
            response_obj.insert("id".to_string(), Value::String(original_id.to_string()));
        }
    }

    // Update sequence number if present in the event
    if parsed_data.get("sequence_number").is_some() {
        parsed_data["sequence_number"] = json!(*sequence_number);
        *sequence_number += 1;
    }

    // Serialize once
    let final_data = match serde_json::to_string(&parsed_data) {
        Ok(s) => s,
        Err(_) => {
            // Serialization failed, forward original
            let chunk_to_send = format!("{}\n\n", raw_block);
            return tx.send(Ok(Bytes::from(chunk_to_send))).is_ok();
        }
    };

    // Rebuild SSE block with potentially transformed event name
    let mut final_block = String::new();
    if let Some(evt) = event_name {
        // Update event name for function_call_arguments events
        if evt == event_types::FUNCTION_CALL_ARGUMENTS_DELTA {
            final_block.push_str(&format!(
                "event: {}\n",
                event_types::MCP_CALL_ARGUMENTS_DELTA
            ));
        } else if evt == event_types::FUNCTION_CALL_ARGUMENTS_DONE {
            final_block.push_str(&format!(
                "event: {}\n",
                event_types::MCP_CALL_ARGUMENTS_DONE
            ));
        } else {
            final_block.push_str(&format!("event: {}\n", evt));
        }
    }
    final_block.push_str(&format!("data: {}", final_data));

    let chunk_to_send = format!("{}\n\n", final_block);
    if tx.send(Ok(Bytes::from(chunk_to_send))).is_err() {
        return false;
    }

    // After sending output_item.added for mcp_call, inject mcp_call.in_progress event
    if event_name == Some(event_types::OUTPUT_ITEM_ADDED) {
        if let Some(item) = parsed_data.get("item") {
            if item.get("type").and_then(|v| v.as_str()) == Some(event_types::ITEM_TYPE_MCP_CALL) {
                // Already transformed to mcp_call
                if let (Some(item_id), Some(output_index)) = (
                    item.get("id").and_then(|v| v.as_str()),
                    parsed_data.get("output_index").and_then(|v| v.as_u64()),
                ) {
                    let in_progress_event = json!({
                        "type": event_types::MCP_CALL_IN_PROGRESS,
                        "sequence_number": *sequence_number,
                        "output_index": output_index,
                        "item_id": item_id
                    });
                    *sequence_number += 1;
                    let in_progress_block = format!(
                        "event: {}\ndata: {}\n\n",
                        event_types::MCP_CALL_IN_PROGRESS,
                        in_progress_event
                    );
                    if tx.send(Ok(Bytes::from(in_progress_block))).is_err() {
                        return false;
                    }
                }
            }
        }
    }

    true
}

/// Send final response.completed event to client
/// Returns false if client disconnected
#[allow(clippy::too_many_arguments)]
pub(super) fn send_final_response_event(
    handler: &StreamingToolHandler,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    sequence_number: &mut u64,
    state: &ToolLoopState,
    active_mcp: Option<&Arc<crate::mcp::McpClientManager>>,
    original_request: &ResponsesRequest,
    previous_response_id: Option<&str>,
    server_label: &str,
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
        inject_mcp_metadata_streaming(&mut final_response, state, mcp, server_label);
    }

    mask_tools_as_mcp(&mut final_response, original_request);
    patch_streaming_response_json(&mut final_response, original_request, previous_response_id);

    if let Some(obj) = final_response.as_object_mut() {
        obj.insert("status".to_string(), Value::String("completed".to_string()));
    }

    let completed_payload = json!({
        "type": event_types::RESPONSE_COMPLETED,
        "sequence_number": *sequence_number,
        "response": final_response
    });
    *sequence_number += 1;

    let completed_event = format!(
        "event: {}\ndata: {}\n\n",
        event_types::RESPONSE_COMPLETED,
        completed_payload
    );
    tx.send(Ok(Bytes::from(completed_event))).is_ok()
}

// ============================================================================
// Main Streaming Handlers
// ============================================================================

/// Simple pass-through streaming without MCP interception
#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_simple_streaming_passthrough(
    client: &reqwest::Client,
    circuit_breaker: &crate::core::CircuitBreaker,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    url: String,
    headers: Option<&HeaderMap>,
    payload: Value,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<String>,
) -> Response {
    let mut request_builder = client.post(&url).json(&payload);

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
        let error_body = match response.text().await {
            Ok(body) => body,
            Err(err) => format!("Failed to read upstream error body: {}", err),
        };
        return (status_code, error_body).into_response();
    }

    circuit_breaker.record_success();

    let preserved_headers = preserve_response_headers(response.headers());
    let mut upstream_stream = response.bytes_stream();

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

    let should_store = original_body.store.unwrap_or(false);
    let original_request = original_body.clone();
    let persist_needed = original_request.conversation.is_some();
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

                        let block_cow = if let Some(modified) = rewrite_streaming_block(
                            raw_block.as_str(),
                            &original_request,
                            previous_response_id.as_deref(),
                        ) {
                            Cow::Owned(modified)
                        } else {
                            Cow::Borrowed(raw_block.as_str())
                        };

                        if should_store || persist_needed {
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

        if (should_store || persist_needed) && !upstream_failed {
            if !pending.trim().is_empty() {
                accumulator.ingest_block(&pending);
            }
            let encountered_error = accumulator.encountered_error().cloned();
            if let Some(mut response_json) = accumulator.into_final_response() {
                patch_streaming_response_json(
                    &mut response_json,
                    &original_request,
                    previous_response_id.as_deref(),
                );

                // Always persist conversation items and response (even without conversation)
                if let Err(err) = persist_conversation_items(
                    conversation_storage.clone(),
                    conversation_item_storage.clone(),
                    response_storage.clone(),
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
#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_streaming_with_tool_interception(
    client: &reqwest::Client,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    url: String,
    headers: Option<&HeaderMap>,
    mut payload: Value,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<String>,
    active_mcp: &Arc<crate::mcp::McpClientManager>,
) -> Response {
    // Transform MCP tools to function tools in payload
    prepare_mcp_payload_for_streaming(&mut payload, active_mcp);

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
    let should_store = original_body.store.unwrap_or(false);
    let original_request = original_body.clone();
    let persist_needed = original_request.conversation.is_some();
    let previous_response_id = original_previous_response_id.clone();

    let client_clone = client.clone();
    let url_clone = url.clone();
    let headers_opt = headers.cloned();
    let payload_clone = payload.clone();
    let active_mcp_clone = Arc::clone(active_mcp);

    // Spawn the streaming loop task
    tokio::spawn(async move {
        let mut state = ToolLoopState::new(original_request.input.clone());
        let loop_config = McpLoopConfig::default();
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
            let mut pending = String::new();
            let mut tool_calls_detected = false;
            let mut seen_in_progress = false;

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
                                                Some(event_types::RESPONSE_CREATED)
                                                    | Some(event_types::RESPONSE_IN_PROGRESS)
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
                                            server_label,
                                            &original_request,
                                            previous_response_id.as_deref(),
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
                                                == Some(event_types::RESPONSE_IN_PROGRESS)
                                            {
                                                seen_in_progress = true;
                                                if !mcp_list_tools_sent {
                                                    let list_tools_index =
                                                        handler.allocate_synthetic_output_index();
                                                    if !send_mcp_list_tools_events(
                                                        &tx,
                                                        &active_mcp_clone,
                                                        server_label,
                                                        list_tools_index,
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
                                        server_label,
                                        &original_request,
                                        previous_response_id.as_deref(),
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
                    &original_request,
                    previous_response_id.as_deref(),
                    server_label,
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
                    inject_mcp_metadata_streaming(
                        &mut response_json,
                        &state,
                        &active_mcp_clone,
                        server_label,
                    );

                    mask_tools_as_mcp(&mut response_json, &original_request);
                    patch_streaming_response_json(
                        &mut response_json,
                        &original_request,
                        previous_response_id.as_deref(),
                    );

                    // Always persist conversation items and response (even without conversation)
                    if let Err(err) = persist_conversation_items(
                        conversation_storage.clone(),
                        conversation_item_storage.clone(),
                        response_storage.clone(),
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
                server_label,
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

/// Main entry point for handling streaming responses
/// Delegates to simple passthrough or MCP tool interception based on configuration
#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_streaming_response(
    client: &reqwest::Client,
    circuit_breaker: &crate::core::CircuitBreaker,
    mcp_manager: Option<&Arc<crate::mcp::McpClientManager>>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    url: String,
    headers: Option<&HeaderMap>,
    payload: Value,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<String>,
) -> Response {
    // Check if MCP is active for this request
    let req_mcp_manager = if let Some(ref tools) = original_body.tools {
        mcp_manager_from_request_tools(tools.as_slice()).await
    } else {
        None
    };
    let active_mcp = req_mcp_manager.as_ref().or(mcp_manager);

    // If no MCP is active, use simple pass-through streaming
    if active_mcp.is_none() {
        return handle_simple_streaming_passthrough(
            client,
            circuit_breaker,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            url,
            headers,
            payload,
            original_body,
            original_previous_response_id,
        )
        .await;
    }

    let active_mcp = active_mcp.unwrap();

    // MCP is active - transform tools and set up interception
    handle_streaming_with_tool_interception(
        client,
        response_storage,
        conversation_storage,
        conversation_item_storage,
        url,
        headers,
        payload,
        original_body,
        original_previous_response_id,
        active_mcp,
    )
    .await
}
