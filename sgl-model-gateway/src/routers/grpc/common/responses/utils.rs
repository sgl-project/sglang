//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::response::Response;
use serde_json::{json, to_value, Value};
use tracing::{debug, error, warn};

use crate::{
    core::WorkerRegistry,
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    mcp::McpManager,
    protocols::{
        common::Tool,
        responses::{
            generate_id, McpToolInfo, ResponseOutputItem, ResponseTool, ResponseToolType,
            ResponsesRequest, ResponsesResponse,
        },
    },
    routers::{conversations::persist_conversation_items, error, mcp::ensure_request_mcp_client},
};

/// Ensure MCP connection succeeds if MCP tools are declared
///
/// Checks if request declares MCP tools, and if so, validates that
/// the MCP client can be created and connected.
pub async fn ensure_mcp_connection(
    mcp_manager: &Arc<McpManager>,
    tools: Option<&[ResponseTool]>,
) -> Result<bool, Response> {
    let has_mcp_tools = tools
        .map(|t| {
            t.iter()
                .any(|tool| matches!(tool.r#type, ResponseToolType::Mcp))
        })
        .unwrap_or(false);

    if has_mcp_tools {
        if let Some(tools) = tools {
            if ensure_request_mcp_client(mcp_manager, tools)
                .await
                .is_none()
            {
                error!(
                    function = "ensure_mcp_connection",
                    "Failed to connect to MCP server"
                );
                return Err(error::failed_dependency(
                    "connect_mcp_server_failed",
                    "Failed to connect to MCP server. Check server_url and authorization.",
                ));
            }
        }
    }

    Ok(has_mcp_tools)
}

/// Validate that workers are available for the requested model
pub fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    let available_models = worker_registry.get_models();

    if !available_models.contains(&model.to_string()) {
        return Some(error::service_unavailable(
            "no_available_workers",
            format!(
                "No workers available for model '{}'. Available models: {}",
                model,
                available_models.join(", ")
            ),
        ));
    }

    None
}

/// Extract function tools (and optionally MCP tools) from ResponseTools
///
/// This utility consolidates the logic for extracting tools with schemas from ResponseTools.
/// It's used by both Harmony and Regular routers for different purposes:
///
/// - **Harmony router**: Extracts both Function and MCP tools (with `include_mcp: true`)
///   because MCP schemas are populated by convert_mcp_tools_to_response_tools() before the
///   pipeline runs. These tools are used to generate structural constraints in the
///   Harmony preparation stage.
///
/// - **Regular router**: Extracts only Function tools (with `include_mcp: false`) during
///   the initial conversion from ResponsesRequest to ChatCompletionRequest. MCP tools
///   are merged later by the tool loop before being sent to the chat pipeline, where
///   tool_choice constraints are generated for ALL tools (function + MCP combined).
pub fn extract_tools_from_response_tools(
    response_tools: Option<&[ResponseTool]>,
    include_mcp: bool,
) -> Vec<Tool> {
    let Some(tools) = response_tools else {
        return Vec::new();
    };

    tools
        .iter()
        .filter_map(|rt| {
            match rt.r#type {
                // Function tools: Schema in request
                ResponseToolType::Function => rt.function.as_ref().map(|f| Tool {
                    tool_type: "function".to_string(),
                    function: f.clone(),
                }),
                // MCP tools: Schema populated by convert_mcp_tools_to_response_tools()
                // Only include if requested (Harmony case)
                ResponseToolType::Mcp if include_mcp => rt.function.as_ref().map(|f| Tool {
                    tool_type: "function".to_string(),
                    function: f.clone(),
                }),
                // Hosted tools: No schema available, skip
                _ => None,
            }
        })
        .collect()
}

/// Persist response to storage if store=true
///
/// Common helper function to avoid duplication across sync and streaming paths
/// in both harmony and regular responses implementations.
pub async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
) {
    if !original_request.store.unwrap_or(true) {
        return;
    }

    if let Ok(response_json) = to_value(response) {
        if let Err(e) = persist_conversation_items(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response_json,
            original_request,
        )
        .await
        {
            warn!("Failed to persist response: {}", e);
        } else {
            debug!("Persisted response: {}", response.id);
        }
    }
}

// ============================================================================
// MCP Helper Functions
// ============================================================================

/// Build mcp_list_tools output item
pub fn build_mcp_list_tools_item(mcp: &McpManager, server_label: &str) -> ResponseOutputItem {
    let tools = mcp.list_tools();
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

    ResponseOutputItem::McpListTools {
        id: generate_id("mcpl"),
        server_label: server_label.to_string(),
        tools: tools_info,
    }
}

/// Build mcp_call output item
pub fn build_mcp_call_item(
    tool_name: &str,
    arguments: &str,
    output: &str,
    server_label: &str,
    success: bool,
    error: Option<&str>,
) -> ResponseOutputItem {
    ResponseOutputItem::McpCall {
        id: generate_id("mcp"),
        status: if success { "completed" } else { "failed" }.to_string(),
        approval_request_id: None,
        arguments: arguments.to_string(),
        error: error.map(|e| e.to_string()),
        name: tool_name.to_string(),
        output: output.to_string(),
        server_label: server_label.to_string(),
    }
}
