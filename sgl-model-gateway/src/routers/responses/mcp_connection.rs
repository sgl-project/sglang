//! MCP connection utilities for Responses API
//!
//! This module provides functions to establish MCP client connections
//! from request-level tool configurations.

use std::sync::Arc;

use tracing::warn;

use crate::{
    mcp::{McpManager, McpServerConfig, McpTransport},
    protocols::responses::{ResponseTool, ResponseToolType},
};

/// Ensure MCP client is connected for request-level MCP tools.
///
/// This function extracts MCP server configuration from request tools (server_url, authorization)
/// and ensures a client connection is established via the connection pool.
///
/// Returns `Some(manager)` if a dynamic MCP tool was found and client was created/retrieved,
/// `None` if no MCP tools with server_url were found or connection failed.
pub async fn ensure_request_mcp_client(
    mcp_manager: &Arc<McpManager>,
    tools: &[ResponseTool],
) -> Option<Arc<McpManager>> {
    // Find an MCP tool with a server_url
    let tool = tools
        .iter()
        .find(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some())?;

    let server_url = tool.server_url.as_ref()?.trim().to_string();

    // Validate URL scheme
    if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
        warn!(
            "Ignoring MCP server_url with unsupported scheme: {}",
            server_url
        );
        return None;
    }

    // Extract server label and auth token
    let name = tool
        .server_label
        .clone()
        .unwrap_or_else(|| "request-mcp".to_string());
    let token = tool.authorization.clone();

    // Determine transport type based on URL pattern
    let transport = if server_url.contains("/sse") {
        McpTransport::Sse {
            url: server_url.clone(),
            token,
        }
    } else {
        McpTransport::Streamable {
            url: server_url.clone(),
            token,
        }
    };

    // Create server config
    let server_config = McpServerConfig {
        name,
        transport,
        proxy: None,
        required: false,
    };

    // Use get_or_create_client to establish connection
    match mcp_manager.get_or_create_client(server_config).await {
        Ok(_client) => Some(Arc::clone(mcp_manager)),
        Err(err) => {
            warn!("Failed to get/create MCP connection: {}", err);
            None
        }
    }
}
