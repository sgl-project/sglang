//! MCP (Model Context Protocol) Integration Module
//!
//! This module contains all MCP-related functionality for the OpenAI router:
//! - Tool loop state management for multi-turn tool calling
//! - MCP tool execution and result handling
//! - Output item builders for MCP-specific response formats
//! - SSE event generation for streaming MCP operations
//! - Payload transformation for MCP tool interception
//! - Metadata injection for MCP operations

use std::{io, sync::Arc};

use axum::http::HeaderMap;
use bytes::Bytes;
use serde_json::{json, to_value, Value};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::{
    mcp,
    protocols::{
        event_types::{is_function_call_type, ItemType, McpEvent, OutputItemEvent},
        responses::{generate_id, ResponseInput, ResponseTool, ResponseToolType, ResponsesRequest},
    },
    routers::header_utils::apply_request_headers,
};

// ============================================================================
// Configuration and State Types
// ============================================================================

/// Configuration for MCP tool calling loops
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct McpLoopConfig {
    /// Maximum iterations as safety limit (internal only, default: 10)
    /// Prevents infinite loops when max_tool_calls is not set
    pub max_iterations: usize,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self { max_iterations: 10 }
    }
}

/// State for tracking multi-turn tool calling loop
pub(crate) struct ToolLoopState {
    /// Current iteration number (starts at 0, increments with each tool call)
    pub iteration: usize,
    /// Total number of tool calls executed
    pub total_calls: usize,
    /// Conversation history (function_call and function_call_output items)
    pub conversation_history: Vec<Value>,
    /// Original user input (preserved for building resume payloads)
    pub original_input: ResponseInput,
}

impl ToolLoopState {
    pub fn new(original_input: ResponseInput) -> Self {
        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
        }
    }

    /// Record a tool call in the loop state
    pub fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
    ) {
        // Add function_call item to history
        let func_item = json!({
            "type": ItemType::FUNCTION_CALL,
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

/// Represents a function call being accumulated across delta events
#[derive(Debug, Clone)]
pub(crate) struct FunctionCallInProgress {
    pub call_id: String,
    pub name: String,
    pub arguments_buffer: String,
    pub output_index: usize,
    pub last_obfuscation: Option<String>,
    pub assigned_output_index: Option<usize>,
}

impl FunctionCallInProgress {
    pub fn new(call_id: String, output_index: usize) -> Self {
        Self {
            call_id,
            name: String::new(),
            arguments_buffer: String::new(),
            output_index,
            last_obfuscation: None,
            assigned_output_index: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        // A tool call is complete if it has a name
        !self.name.is_empty()
    }

    pub fn effective_output_index(&self) -> usize {
        self.assigned_output_index.unwrap_or(self.output_index)
    }
}

// ============================================================================
// MCP Manager Integration
// ============================================================================

/// Ensure a dynamic MCP client exists for request-scoped tools.
///
/// This function parses request tools to extract MCP server configuration,
/// then ensures a dynamic client exists in the McpManager via `get_or_create_client()`.
/// The McpManager itself is returned (cloned Arc) for convenience, though the main
/// purpose is the side effect of registering the dynamic client.
///
/// Returns Some(manager) if a dynamic MCP tool was found and client was created/retrieved,
/// None if no MCP tools were found or connection failed.
pub async fn ensure_request_mcp_client(
    mcp_manager: &Arc<mcp::McpManager>,
    tools: &[ResponseTool],
) -> Option<Arc<mcp::McpManager>> {
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
        mcp::McpTransport::Sse {
            url: server_url.clone(),
            token,
        }
    } else {
        mcp::McpTransport::Streamable {
            url: server_url.clone(),
            token,
        }
    };

    // Create server config
    let server_config = mcp::McpServerConfig {
        name,
        transport,
        proxy: None,
        required: false,
    };

    // Use McpManager to get or create dynamic client
    match mcp_manager.get_or_create_client(server_config).await {
        Ok(_client) => Some(mcp_manager.clone()),
        Err(err) => {
            warn!("Failed to get/create MCP connection: {}", err);
            None
        }
    }
}

// ============================================================================
// Tool Execution
// ============================================================================

/// Execute detected tool calls and send completion events to client
/// Returns false if client disconnected during execution
pub(super) async fn execute_streaming_tool_calls(
    pending_calls: Vec<FunctionCallInProgress>,
    active_mcp: &Arc<mcp::McpManager>,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    state: &mut ToolLoopState,
    server_label: &str,
    sequence_number: &mut u64,
) -> bool {
    // Execute all pending tool calls (sequential, as PR3 is skipped)
    for call in pending_calls {
        // Skip if name is empty (invalid call)
        if call.name.is_empty() {
            warn!(
                "Skipping incomplete tool call: name is empty, args_len={}",
                call.arguments_buffer.len()
            );
            continue;
        }

        info!(
            "Executing tool call during streaming: {} ({})",
            call.name, call.call_id
        );

        // Use empty JSON object if arguments_buffer is empty
        let args_str = if call.arguments_buffer.is_empty() {
            "{}"
        } else {
            &call.arguments_buffer
        };

        // Call tool directly - manager handles parsing and type coercion
        debug!("Calling MCP tool '{}' with args: {}", call.name, args_str);
        let call_result = active_mcp.call_tool(&call.name, args_str).await;
        let (output_str, success, error_msg) = match call_result {
            Ok(result) => match serde_json::to_string(&result) {
                Ok(output) => (output, true, None),
                Err(e) => {
                    let err = format!("Failed to serialize tool result: {}", e);
                    warn!("{}", err);
                    (json!({ "error": &err }).to_string(), false, Some(err))
                }
            },
            Err(err) => {
                let err_str = format!("tool call failed: {}", err);
                warn!("Tool execution failed during streaming: {}", err_str);
                (
                    json!({ "error": &err_str }).to_string(),
                    false,
                    Some(err_str),
                )
            }
        };

        // Send mcp_call completion event to client
        if !send_mcp_call_completion_events_with_error(
            tx,
            &call,
            &output_str,
            server_label,
            success,
            error_msg.as_deref(),
            sequence_number,
        ) {
            // Client disconnected, no point continuing tool execution
            return false;
        }

        // Record the call
        state.record_call(call.call_id, call.name, call.arguments_buffer, output_str);
    }
    true
}

// ============================================================================
// Payload Transformation
// ============================================================================

/// Transform payload to replace MCP tools with function tools for streaming
pub(super) fn prepare_mcp_payload_for_streaming(
    payload: &mut Value,
    active_mcp: &Arc<mcp::McpManager>,
) {
    if let Some(obj) = payload.as_object_mut() {
        // Remove any non-function tools from outgoing payload
        if let Some(v) = obj.get_mut("tools") {
            if let Some(arr) = v.as_array_mut() {
                arr.retain(|item| {
                    item.get("type")
                        .and_then(|v| v.as_str())
                        .map(|s| s == ItemType::FUNCTION)
                        .unwrap_or(false)
                });
            }
        }

        // Build function tools for all discovered MCP tools
        let mut tools_json = Vec::new();
        let tools = active_mcp.list_tools();
        for t in tools {
            let parameters = Value::Object((*t.input_schema).clone());
            let tool = serde_json::json!({
                "type": ItemType::FUNCTION,
                "name": t.name,
                "description": t.description,
                "parameters": parameters
            });
            tools_json.push(tool);
        }
        if !tools_json.is_empty() {
            obj.insert("tools".to_string(), Value::Array(tools_json));
            obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
        }
    }
}

/// Build a resume payload with conversation history
pub(super) fn build_resume_payload(
    base_payload: &Value,
    conversation_history: &[Value],
    original_input: &ResponseInput,
    tools_json: &Value,
    is_streaming: bool,
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
            // Items are ResponseInputOutputItem (including SimpleInputMessage), convert to JSON
            if let Ok(items_value) = to_value(items) {
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

    // Set streaming mode based on caller's context
    obj.insert("stream".to_string(), Value::Bool(is_streaming));
    obj.insert("store".to_string(), Value::Bool(false));

    // Note: SGLang-specific fields were already removed from base_payload
    // before it was passed to execute_tool_loop (see route_responses lines 1935-1946)

    Ok(payload)
}

// ============================================================================
// SSE Event Senders
// ============================================================================

/// Send mcp_list_tools events to client at the start of streaming
/// Returns false if client disconnected
pub(super) fn send_mcp_list_tools_events(
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    mcp: &Arc<mcp::McpManager>,
    server_label: &str,
    output_index: usize,
    sequence_number: &mut u64,
) -> bool {
    let tools_item_full = build_mcp_list_tools_item(mcp, server_label);
    let item_id = tools_item_full
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Create empty tools version for the initial added event
    let mut tools_item_empty = tools_item_full.clone();
    if let Some(obj) = tools_item_empty.as_object_mut() {
        obj.insert("tools".to_string(), json!([]));
    }

    // Event 1: response.output_item.added with empty tools
    let event1_payload = json!({
        "type": OutputItemEvent::ADDED,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item": tools_item_empty
    });
    *sequence_number += 1;
    let event1 = format!(
        "event: {}\ndata: {}\n\n",
        OutputItemEvent::ADDED,
        event1_payload
    );
    if tx.send(Ok(Bytes::from(event1))).is_err() {
        return false; // Client disconnected
    }

    // Event 2: response.mcp_list_tools.in_progress
    let event2_payload = json!({
        "type": McpEvent::LIST_TOOLS_IN_PROGRESS,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;
    let event2 = format!(
        "event: {}\ndata: {}\n\n",
        McpEvent::LIST_TOOLS_IN_PROGRESS,
        event2_payload
    );
    if tx.send(Ok(Bytes::from(event2))).is_err() {
        return false;
    }

    // Event 3: response.mcp_list_tools.completed
    let event3_payload = json!({
        "type": McpEvent::LIST_TOOLS_COMPLETED,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;
    let event3 = format!(
        "event: {}\ndata: {}\n\n",
        McpEvent::LIST_TOOLS_COMPLETED,
        event3_payload
    );
    if tx.send(Ok(Bytes::from(event3))).is_err() {
        return false;
    }

    // Event 4: response.output_item.done with full tools list
    let event4_payload = json!({
        "type": OutputItemEvent::DONE,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item": tools_item_full
    });
    *sequence_number += 1;
    let event4 = format!(
        "event: {}\ndata: {}\n\n",
        OutputItemEvent::DONE,
        event4_payload
    );
    tx.send(Ok(Bytes::from(event4))).is_ok()
}

/// Send mcp_call completion events after tool execution
/// Returns false if client disconnected
pub(super) fn send_mcp_call_completion_events_with_error(
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    call: &FunctionCallInProgress,
    output: &str,
    server_label: &str,
    success: bool,
    error_msg: Option<&str>,
    sequence_number: &mut u64,
) -> bool {
    let effective_output_index = call.effective_output_index();

    // Build mcp_call item (reuse existing function)
    let mcp_call_item = build_mcp_call_item(
        &call.name,
        &call.arguments_buffer,
        output,
        server_label,
        success,
        error_msg,
    );

    // Get the mcp_call item_id
    let item_id = mcp_call_item
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Event 1: response.mcp_call.completed
    let completed_payload = json!({
        "type": McpEvent::CALL_COMPLETED,
        "sequence_number": *sequence_number,
        "output_index": effective_output_index,
        "item_id": item_id
    });
    *sequence_number += 1;

    let completed_event = format!(
        "event: {}\ndata: {}\n\n",
        McpEvent::CALL_COMPLETED,
        completed_payload
    );
    if tx.send(Ok(Bytes::from(completed_event))).is_err() {
        return false;
    }

    // Event 2: response.output_item.done (with completed mcp_call)
    let done_payload = json!({
        "type": OutputItemEvent::DONE,
        "sequence_number": *sequence_number,
        "output_index": effective_output_index,
        "item": mcp_call_item
    });
    *sequence_number += 1;

    let done_event = format!(
        "event: {}\ndata: {}\n\n",
        OutputItemEvent::DONE,
        done_payload
    );
    tx.send(Ok(Bytes::from(done_event))).is_ok()
}

// ============================================================================
// Metadata Injection
// ============================================================================

/// Inject MCP metadata into a streaming response
pub(super) fn inject_mcp_metadata_streaming(
    response: &mut Value,
    state: &ToolLoopState,
    mcp: &Arc<mcp::McpManager>,
    server_label: &str,
) {
    if let Some(output_array) = response.get_mut("output").and_then(|v| v.as_array_mut()) {
        output_array.retain(|item| {
            item.get("type").and_then(|t| t.as_str()) != Some(ItemType::MCP_LIST_TOOLS)
        });

        let list_tools_item = build_mcp_list_tools_item(mcp, server_label);
        output_array.insert(0, list_tools_item);

        let mcp_call_items =
            build_executed_mcp_call_items(&state.conversation_history, server_label);
        let mut insert_pos = 1;
        for item in mcp_call_items {
            output_array.insert(insert_pos, item);
            insert_pos += 1;
        }
    } else if let Some(obj) = response.as_object_mut() {
        let mut output_items = Vec::new();
        output_items.push(build_mcp_list_tools_item(mcp, server_label));
        output_items.extend(build_executed_mcp_call_items(
            &state.conversation_history,
            server_label,
        ));
        obj.insert("output".to_string(), Value::Array(output_items));
    }
}

// ============================================================================
// Tool Loop Execution
// ============================================================================

/// Execute the tool calling loop
pub(super) async fn execute_tool_loop(
    client: &reqwest::Client,
    url: &str,
    headers: Option<&HeaderMap>,
    initial_payload: Value,
    original_body: &ResponsesRequest,
    active_mcp: &Arc<mcp::McpManager>,
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
        let request_builder = client.post(url).json(&current_payload);
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
        if let Some((call_id, tool_name, args_json_str)) = extract_function_call(&response_json) {
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

                return build_incomplete_response(
                    response_json,
                    state,
                    "max_tool_calls",
                    active_mcp,
                    original_body,
                );
            }

            // Execute tool - manager handles parsing and type coercion
            debug!(
                "Calling MCP tool '{}' with args: {}",
                tool_name, args_json_str
            );
            let call_result = active_mcp
                .call_tool(&tool_name, args_json_str.as_str())
                .await;

            let output_str = match call_result {
                Ok(result) => match serde_json::to_string(&result) {
                    Ok(output) => output,
                    Err(e) => {
                        warn!("Failed to serialize tool result: {}", e);
                        json!({ "error": format!("Serialization error: {}", e) }).to_string()
                    }
                },
                Err(err) => {
                    warn!("Tool execution failed: {}", err);
                    // Return error as output, let model decide how to proceed
                    json!({ "error": format!("tool call failed: {}", err) }).to_string()
                }
            };

            // Record the call
            state.record_call(call_id, tool_name, args_json_str, output_str);

            // Build resume payload
            current_payload = build_resume_payload(
                &base_payload,
                &state.conversation_history,
                &state.original_input,
                &tools_json,
                false, // is_streaming = false (non-streaming tool loop)
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
                    .as_ref()
                    .and_then(|tools| {
                        tools
                            .iter()
                            .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                            .and_then(|t| t.server_label.as_deref())
                    })
                    .unwrap_or("mcp");

                // Build mcp_list_tools item
                let list_tools_item = build_mcp_list_tools_item(active_mcp, server_label);

                // Insert at beginning of output array
                if let Some(output_array) = response_json
                    .get_mut("output")
                    .and_then(|v| v.as_array_mut())
                {
                    output_array.insert(0, list_tools_item);

                    // Build mcp_call items using helper function
                    let mcp_call_items =
                        build_executed_mcp_call_items(&state.conversation_history, server_label);

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

/// Build an incomplete response when limits are exceeded
pub(super) fn build_incomplete_response(
    mut response: Value,
    state: ToolLoopState,
    reason: &str,
    active_mcp: &Arc<mcp::McpManager>,
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
            .as_ref()
            .and_then(|tools| {
                tools
                    .iter()
                    .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                    .and_then(|t| t.server_label.as_deref())
            })
            .unwrap_or("mcp");

        // Find any function_call items and convert them to mcp_call (incomplete)
        let mut mcp_call_items = Vec::new();
        for item in output_array.iter() {
            let item_type = item.get("type").and_then(|t| t.as_str());
            if item_type.is_some_and(is_function_call_type) {
                let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");

                // Mark as incomplete - not executed
                let mcp_call_item = build_mcp_call_item(
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
            let list_tools_item = build_mcp_list_tools_item(active_mcp, server_label);
            output_array.insert(0, list_tools_item);

            // Add mcp_call items for executed calls using helper
            let executed_items =
                build_executed_mcp_call_items(&state.conversation_history, server_label);

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

// ============================================================================
// Output Item Builders
// ============================================================================

/// Build an mcp_list_tools output item
pub(super) fn build_mcp_list_tools_item(mcp: &Arc<mcp::McpManager>, server_label: &str) -> Value {
    let tools = mcp.list_tools();
    let tools_json: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": Value::Object((*t.input_schema).clone()),
                "annotations": {
                    "read_only": false
                }
            })
        })
        .collect();

    json!({
        "id": generate_id("mcpl"),
        "type": ItemType::MCP_LIST_TOOLS,
        "server_label": server_label,
        "tools": tools_json
    })
}

/// Build an mcp_call output item
pub(super) fn build_mcp_call_item(
    tool_name: &str,
    arguments: &str,
    output: &str,
    server_label: &str,
    success: bool,
    error: Option<&str>,
) -> Value {
    json!({
        "id": generate_id("mcp"),
        "type": ItemType::MCP_CALL,
        "status": if success { "completed" } else { "failed" },
        "approval_request_id": Value::Null,
        "arguments": arguments,
        "error": error,
        "name": tool_name,
        "output": output,
        "server_label": server_label
    })
}

/// Helper function to build mcp_call items from executed tool calls in conversation history
pub(super) fn build_executed_mcp_call_items(
    conversation_history: &[Value],
    server_label: &str,
) -> Vec<Value> {
    let mut mcp_call_items = Vec::new();

    for item in conversation_history {
        if item.get("type").and_then(|t| t.as_str()) == Some(ItemType::FUNCTION_CALL) {
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
            let is_error = serde_json::from_str::<Value>(output_str)
                .map(|v| v.get("error").is_some())
                .unwrap_or(false);

            let mcp_call_item = build_mcp_call_item(
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

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract function call from a response
pub(super) fn extract_function_call(resp: &Value) -> Option<(String, String, String)> {
    let output = resp.get("output")?.as_array()?;
    for item in output {
        let obj = item.as_object()?;
        let t = obj.get("type")?.as_str()?;
        if is_function_call_type(t) {
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
