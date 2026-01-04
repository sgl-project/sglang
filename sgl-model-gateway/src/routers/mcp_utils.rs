//! Shared MCP utilities for routers.
//!
//! This module provides shared MCP-related functionality that can be
//! used across different router implementations (OpenAI, gRPC regular, gRPC harmony).

use std::sync::Arc;

use tracing::warn;

use crate::{
    mcp::{McpManager, McpServerConfig, McpTransport},
    protocols::responses::{ResponseTool, ResponseToolType},
};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum tool loop iterations (safety limit).
///
/// Used as fallback when user doesn't specify `max_tool_calls`.
/// All routers use this same value.
pub const DEFAULT_MAX_ITERATIONS: usize = 10;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for MCP tool calling loops.
///
/// Provides a common structure for loop configuration across routers.
#[derive(Debug, Clone)]
pub struct McpLoopConfig {
    /// Maximum iterations as safety limit (default: DEFAULT_MAX_ITERATIONS).
    /// Prevents infinite loops when max_tool_calls is not set by user.
    pub max_iterations: usize,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract MCP server label from request tools.
///
/// Searches for the first MCP tool in the tools array and returns its server_label.
/// Falls back to a default value if no MCP tool with server_label is found.
pub fn extract_server_label(tools: Option<&[ResponseTool]>, default_label: &str) -> String {
    tools
        .and_then(|tools| {
            tools.iter().find_map(|tool| {
                if matches!(tool.r#type, ResponseToolType::Mcp) {
                    tool.server_label.clone()
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| default_label.to_string())
}

// ============================================================================
// MCP Connection
// ============================================================================

/// Ensure MCP client is connected for request-level MCP tools.
///
/// This function extracts MCP server configuration from request tools (server_url, authorization)
/// and ensures a client connection is established via the connection pool.
///
/// Returns `Some(())` if a dynamic MCP tool was found and client was created/retrieved,
/// `None` if no MCP tools with server_url were found or connection failed.
pub async fn ensure_request_mcp_client(
    mcp_manager: &Arc<McpManager>,
    tools: &[ResponseTool],
) -> Option<()> {
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
        Ok(_client) => Some(()),
        Err(err) => {
            warn!("Failed to get/create MCP connection: {}", err);
            None
        }
    }
}
