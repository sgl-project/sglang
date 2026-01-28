//! MCP tool execution logic for Harmony Responses

use std::{sync::Arc, time::Instant};

use axum::response::Response;
use serde_json::{from_str, json, to_string, to_value, Value};
use tracing::{debug, error, warn};

use super::common::McpCallTracking;
use crate::{
    mcp::{self, McpManager},
    observability::metrics::{metrics_labels, Metrics},
    protocols::{
        common::{Function, ToolCall},
        responses::{ResponseTool, ResponseToolType},
    },
    routers::error,
};

/// Tool execution result
///
/// Contains the result of executing a single MCP tool.
pub(crate) struct ToolResult {
    /// Tool call ID (for matching with request)
    pub call_id: String,

    /// Tool name
    #[allow(dead_code)] // Kept for documentation and future use
    pub tool_name: String,

    /// Tool output (JSON value)
    pub output: Value,

    /// Whether this is an error result
    pub is_error: bool,
}

/// Execute MCP tools and collect results
///
/// Executes each tool call sequentially via the MCP manager.
/// Tool execution errors are returned as error results to the model
/// (allows model to handle gracefully).
///
/// Vector of tool results (one per tool call)
pub(super) async fn execute_mcp_tools(
    mcp_manager: &Arc<McpManager>,
    tool_calls: &[ToolCall],
    tracking: &mut McpCallTracking,
    model_id: &str,
) -> Result<Vec<ToolResult>, Response> {
    let mut results = Vec::new();

    for tool_call in tool_calls {
        debug!(
            tool_name = %tool_call.function.name,
            call_id = %tool_call.id,
            "Executing MCP tool"
        );

        // Parse tool arguments from JSON string
        let args_str = tool_call.function.arguments.as_deref().unwrap_or("{}");
        let args: Value = from_str(args_str).map_err(|e| {
            error!(
                function = "execute_mcp_tools",
                tool_name = %tool_call.function.name,
                call_id = %tool_call.id,
                error = %e,
                "Failed to parse tool arguments JSON"
            );
            error::internal_error(
                "invalid_tool_args",
                format!(
                    "Invalid tool arguments JSON for tool '{}': {}",
                    tool_call.function.name, e
                ),
            )
        })?;

        // Execute tool via MCP manager
        let args_map = if let Value::Object(map) = args {
            Some(map)
        } else {
            None
        };

        let tool_start = Instant::now();
        let tool_result = mcp_manager
            .call_tool(&tool_call.function.name, args_map)
            .await;
        let tool_duration = tool_start.elapsed();

        match tool_result {
            Ok(mcp_result) => {
                debug!(
                    tool_name = %tool_call.function.name,
                    call_id = %tool_call.id,
                    "Tool execution succeeded"
                );

                // Extract content from MCP result
                let output = if let Some(content) = mcp_result.content.first() {
                    // Serialize the entire content item
                    to_value(content)
                        .unwrap_or_else(|_| json!({"error": "Failed to serialize tool result"}))
                } else {
                    json!({"result": "success"})
                };

                let is_error = mcp_result.is_error.unwrap_or(false);
                let output_str = to_string(&output)
                    .unwrap_or_else(|_| r#"{"error": "Failed to serialize output"}"#.to_string());

                // Record this call in tracking
                tracking.record_call(
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    args_str.to_string(),
                    output_str.clone(),
                    !is_error,
                    if is_error {
                        Some(output_str.clone())
                    } else {
                        None
                    },
                );

                // Record MCP tool metrics
                Metrics::record_mcp_tool_duration(
                    model_id,
                    &tool_call.function.name,
                    tool_duration,
                );
                Metrics::record_mcp_tool_call(
                    model_id,
                    &tool_call.function.name,
                    if is_error {
                        metrics_labels::RESULT_ERROR
                    } else {
                        metrics_labels::RESULT_SUCCESS
                    },
                );

                results.push(ToolResult {
                    call_id: tool_call.id.clone(),
                    tool_name: tool_call.function.name.clone(),
                    output,
                    is_error,
                });
            }
            Err(e) => {
                warn!(
                    tool_name = %tool_call.function.name,
                    call_id = %tool_call.id,
                    error = %e,
                    "Tool execution failed"
                );

                let error_msg = format!("Tool execution failed: {}", e);
                let error_output = json!({
                    "error": error_msg.clone()
                });
                let error_output_str = to_string(&error_output)
                    .unwrap_or_else(|_| format!(r#"{{"error": "{}"}}"#, error_msg));

                // Record failed call in tracking
                tracking.record_call(
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    args_str.to_string(),
                    error_output_str.clone(),
                    false,
                    Some(error_msg),
                );

                // Record MCP tool metrics
                Metrics::record_mcp_tool_duration(
                    model_id,
                    &tool_call.function.name,
                    tool_duration,
                );
                Metrics::record_mcp_tool_call(
                    model_id,
                    &tool_call.function.name,
                    metrics_labels::RESULT_ERROR,
                );

                // Return error result to model (let it handle gracefully)
                results.push(ToolResult {
                    call_id: tool_call.id.clone(),
                    tool_name: tool_call.function.name.clone(),
                    output: error_output,
                    is_error: true,
                });
            }
        }
    }

    Ok(results)
}

/// Convert MCP tools to Responses API tool format
///
/// Converts MCP Tool entries (from rmcp SDK) to ResponseTool format so the model
/// knows about available MCP tools when making tool calls.
pub(crate) fn convert_mcp_tools_to_response_tools(mcp_tools: &[mcp::Tool]) -> Vec<ResponseTool> {
    mcp_tools
        .iter()
        .map(|tool_info| ResponseTool {
            r#type: ResponseToolType::Mcp,
            function: Some(Function {
                name: tool_info.name.to_string(),
                description: tool_info.description.as_ref().map(|d| d.to_string()),
                parameters: Value::Object((*tool_info.input_schema).clone()),
                strict: None,
            }),
            server_url: None, // MCP tools from inventory don't have individual server URLs
            authorization: None,
            server_label: None,
            server_description: tool_info.description.as_ref().map(|d| d.to_string()),
            require_approval: None,
            allowed_tools: None,
        })
        .collect()
}
