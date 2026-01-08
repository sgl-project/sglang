//! Shared helpers and state tracking for Harmony Responses

use axum::response::Response;
use serde_json::{from_value, json, to_string, Value};
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::execution::ToolResult;
use crate::{
    data_connector::ResponseId,
    mcp,
    protocols::{
        common::{ToolCall, ToolChoice, ToolChoiceValue},
        responses::{
            McpToolInfo, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponseOutputItem, ResponseReasoningContent, ResponseTool, ResponseToolType,
            ResponsesRequest, ResponsesResponse, StringOrContentParts,
        },
    },
    routers::{error, grpc::common::responses::ResponsesContext},
};

/// Record of a single MCP tool call execution
///
/// Stores metadata needed to build mcp_call output items for Responses API format
#[derive(Debug, Clone)]
pub(super) struct McpCallRecord {
    /// Tool call ID (stored for potential future use, currently generate new IDs)
    #[allow(dead_code)]
    pub call_id: String,
    /// Tool name
    pub tool_name: String,
    /// JSON-encoded arguments
    pub arguments: String,
    /// JSON-encoded output/result
    pub output: String,
    /// Whether execution succeeded
    pub success: bool,
    /// Error message if execution failed
    pub error: Option<String>,
}

/// Tracking structure for MCP tool calls across iterations
///
/// Accumulates all MCP tool call metadata during multi-turn conversation
/// so we can build proper mcp_list_tools and mcp_call output items.
#[derive(Debug, Clone)]
pub(super) struct McpCallTracking {
    /// MCP server label (e.g., "sglang-mcp")
    pub server_label: String,
    /// All tool call records across all iterations
    pub tool_calls: Vec<McpCallRecord>,
}

impl McpCallTracking {
    pub fn new(server_label: String) -> Self {
        Self {
            server_label,
            tool_calls: Vec::new(),
        }
    }

    pub fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        arguments: String,
        output: String,
        success: bool,
        error: Option<String>,
    ) {
        self.tool_calls.push(McpCallRecord {
            call_id,
            tool_name,
            arguments,
            output,
            success,
            error,
        });
    }

    pub fn total_calls(&self) -> usize {
        self.tool_calls.len()
    }
}

/// Build a HashSet of MCP tool names for O(1) lookup
///
/// Creates a HashSet containing the names of all MCP tools in the request,
/// allowing for efficient O(1) lookups when partitioning tool calls.
pub(super) fn build_mcp_tool_names_set(
    request_tools: &[ResponseTool],
) -> std::collections::HashSet<&str> {
    request_tools
        .iter()
        .filter(|t| t.r#type == ResponseToolType::Mcp)
        .filter_map(|t| t.function.as_ref().map(|f| f.name.as_str()))
        .collect()
}

/// Build next request with tool results appended to history
///
/// Constructs a new ResponsesRequest with:
/// 1. Original input items (preserved)
/// 2. Assistant message with analysis (reasoning) + partial_text + tool_calls
/// 3. Tool result messages for each tool execution
pub(super) fn build_next_request_with_tools(
    mut request: ResponsesRequest,
    tool_calls: Vec<ToolCall>,
    tool_results: Vec<ToolResult>,
    analysis: Option<String>, // Analysis channel content (becomes reasoning content)
    partial_text: String,     // Final channel content (becomes message content)
) -> Result<ResponsesRequest, Box<Response>> {
    // Get current input items (or empty vec if Text variant)
    let mut items = match request.input {
        ResponseInput::Items(items) => items,
        ResponseInput::Text(text) => {
            // Convert text to items format
            vec![ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String(text),
                role: "user".to_string(),
                r#type: None,
            }]
        }
    };

    // Build assistant response item with reasoning + content + tool calls
    // This represents what the model generated in this iteration
    let assistant_id = format!("msg_{}", Uuid::new_v4());

    // Add reasoning if present (from analysis channel)
    if let Some(analysis_text) = analysis {
        items.push(ResponseInputOutputItem::Reasoning {
            id: format!("reasoning_{}", assistant_id),
            summary: vec![],
            content: vec![ResponseReasoningContent::ReasoningText {
                text: analysis_text,
            }],
            status: Some("completed".to_string()),
        });
    }

    // Add message content if present (from final channel)
    if !partial_text.is_empty() {
        items.push(ResponseInputOutputItem::Message {
            id: assistant_id.clone(),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: partial_text,
                annotations: vec![],
                logprobs: None,
            }],
            status: Some("completed".to_string()),
        });
    }

    // Add function tool calls (from commentary channel)
    for tool_call in tool_calls {
        items.push(ResponseInputOutputItem::FunctionToolCall {
            id: tool_call.id.clone(),
            call_id: tool_call.id.clone(),
            name: tool_call.function.name.clone(),
            arguments: tool_call
                .function
                .arguments
                .unwrap_or_else(|| "{}".to_string()),
            output: None, // Output will be added next
            status: Some("in_progress".to_string()),
        });
    }

    // Add tool results
    for tool_result in tool_results {
        // Serialize tool output to string
        let output_str = to_string(&tool_result.output).unwrap_or_else(|e| {
            format!("{{\"error\": \"Failed to serialize tool output: {}\"}}", e)
        });

        // Update the corresponding tool call with output and completed status
        // Find and update the matching FunctionToolCall
        if let Some(ResponseInputOutputItem::FunctionToolCall {
            output,
            status,
            ..
        }) = items
            .iter_mut()
            .find(|item| matches!(item, ResponseInputOutputItem::FunctionToolCall { call_id, .. } if call_id == &tool_result.call_id))
        {
            *output = Some(output_str);
            *status = if tool_result.is_error {
                Some("failed".to_string())
            } else {
                Some("completed".to_string())
            };
        }
    }

    // Update request with new items
    request.input = ResponseInput::Items(items);

    // Switch tool_choice to "auto" for subsequent iterations
    // This prevents infinite loops when original tool_choice was "required" or specific function
    // After receiving tool results, the model should be free to decide whether to call more tools or finish
    request.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

    Ok(request)
}

/// Inject MCP metadata into final response
///
/// Adds mcp_list_tools and mcp_call output items to the response output array.
/// Following non-Harmony pipeline pattern:
/// 1. Prepend mcp_list_tools at the beginning
/// 2. Append all mcp_call items at the end
pub(super) fn inject_mcp_metadata(
    response: &mut ResponsesResponse,
    tracking: &McpCallTracking,
    mcp_tools: &[mcp::Tool],
) {
    // Build mcp_list_tools item
    let tools = mcp_tools;
    let tools_info: Vec<McpToolInfo> = tools
        .iter()
        .map(|t| McpToolInfo {
            name: t.name.to_string(),
            description: t.description.as_ref().map(|d| d.to_string()),
            input_schema: Value::Object((*t.input_schema).clone()),
            annotations: Some(json!({
                "read_only": false
            })),
        })
        .collect();

    let mcp_list_tools = ResponseOutputItem::McpListTools {
        id: format!("mcpl_{}", Uuid::new_v4()),
        server_label: tracking.server_label.clone(),
        tools: tools_info,
    };

    // Build mcp_call items for each tracked call
    let mcp_call_items: Vec<ResponseOutputItem> = tracking
        .tool_calls
        .iter()
        .map(|record| ResponseOutputItem::McpCall {
            id: format!("mcp_{}", Uuid::new_v4()),
            status: if record.success {
                "completed"
            } else {
                "failed"
            }
            .to_string(),
            approval_request_id: None,
            arguments: record.arguments.clone(),
            error: record.error.clone(),
            name: record.tool_name.clone(),
            output: record.output.clone(),
            server_label: tracking.server_label.clone(),
        })
        .collect();

    // Inject into response output:
    // 1. Prepend mcp_list_tools at the beginning
    response.output.insert(0, mcp_list_tools);

    // 2. Append all mcp_call items at the end
    response.output.extend(mcp_call_items);
}

/// Load previous conversation messages from storage
///
/// If the request has `previous_response_id`, loads the response chain from storage
/// and prepends the conversation history to the request input items.
pub(super) async fn load_previous_messages(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
) -> Result<ResponsesRequest, Response> {
    let Some(ref prev_id_str) = request.previous_response_id else {
        // No previous_response_id, return request as-is
        return Ok(request);
    };

    let prev_id = ResponseId::from(prev_id_str.as_str());

    // Load response chain from storage
    let chain = ctx
        .response_storage
        .get_response_chain(&prev_id, None)
        .await
        .map_err(|e| {
            error!(
                function = "load_previous_messages",
                prev_id = %prev_id_str,
                error = %e,
                "Failed to load previous response chain from storage"
            );
            error::internal_error(
                "load_previous_response_chain_failed",
                format!(
                    "Failed to load previous response chain for {}: {}",
                    prev_id_str, e
                ),
            )
        })?;

    // Build conversation history from stored responses
    let mut history_items = Vec::new();

    // Helper to deserialize and collect items from a JSON array
    let deserialize_items = |arr: &Value, item_type: &str| -> Vec<ResponseInputOutputItem> {
        arr.as_array()
            .into_iter()
            .flat_map(|items| items.iter())
            .filter_map(|item| {
                from_value::<ResponseInputOutputItem>(item.clone())
                    .map_err(|e| {
                        warn!(
                            "Failed to deserialize stored {} item: {}. Item: {}",
                            item_type, e, item
                        );
                    })
                    .ok()
            })
            .collect()
    };

    for stored in chain.responses.iter() {
        history_items.extend(deserialize_items(&stored.input, "input"));
        history_items.extend(deserialize_items(&stored.output, "output"));
    }

    debug!(
        previous_response_id = %prev_id_str,
        history_items_count = history_items.len(),
        "Loaded conversation history from previous response"
    );

    // Build modified request with history prepended
    let mut modified_request = request;

    // Convert current input to items format
    let all_items = match modified_request.input {
        ResponseInput::Items(items) => {
            // Prepend history to existing items
            let mut combined = history_items;
            combined.extend(items);
            combined
        }
        ResponseInput::Text(text) => {
            // Convert text to item and prepend history
            history_items.push(ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String(text),
                role: "user".to_string(),
                r#type: None,
            });
            history_items
        }
    };

    // Update request with combined items and clear previous_response_id
    modified_request.input = ResponseInput::Items(all_items);
    modified_request.previous_response_id = None;

    Ok(modified_request)
}
