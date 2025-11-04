//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::response::Response;

use crate::{
    mcp::McpManager,
    protocols::responses::{ResponseTool, ResponseToolType},
    routers::{grpc::error, openai::mcp::ensure_request_mcp_client},
};

/// Ensure MCP connection succeeds if MCP tools are declared
///
/// Checks if request declares MCP tools, and if so, validates that
/// the MCP client can be created and connected.
///
/// # Returns
///
/// * `Ok(true)` - MCP tools present and connection succeeded
/// * `Ok(false)` - No MCP tools declared
/// * `Err(Response)` - MCP tools declared but connection failed (424 error)
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
                return Err(error::failed_dependency(
                    "Failed to connect to MCP server. Check server_url and authorization.",
                ));
            }
        }
    }

    Ok(has_mcp_tools)
}
