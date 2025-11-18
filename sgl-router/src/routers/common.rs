//! Common utilities shared across router implementations

use std::{collections::HashMap, sync::Arc};

use serde_json::Value;
use tracing::warn;

use crate::{
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    mcp::{self, McpManager},
    protocols::responses::{ResponseTool, ResponseToolType, ResponsesRequest},
};

/// Extract dynamic MCP servers from request tools
///
/// Parses tools to find MCP servers with URLs and labels, deduplicating by server_label.
///
/// # Arguments
/// * `tools` - Request tools to extract MCP servers from
///
/// # Returns
/// Vec of (server_label, server_url) tuples, deduplicated by server_label
pub fn extract_dynamic_mcp_servers(tools: &[ResponseTool]) -> Vec<(String, String)> {
    tools
        .iter()
        .filter_map(|t| {
            if matches!(t.r#type, ResponseToolType::Mcp) {
                let url = t.server_url.as_ref()?.trim().to_string();

                if !(url.starts_with("http://") || url.starts_with("https://")) {
                    warn!("Ignoring MCP server_url with unsupported scheme: {}", url);
                    return None;
                }

                let label = t.server_label.clone().unwrap_or_else(|| {
                    format!(
                        "mcp_{}",
                        url.chars()
                            .filter(|c| c.is_alphanumeric())
                            .take(8)
                            .collect::<String>()
                    )
                });
                Some((label, url))
            } else {
                None
            }
        })
        .collect::<HashMap<String, String>>() // Dedupe by label
        .into_iter()
        .collect()
}

/// Ensure a dynamic MCP client exists for request-scoped tools
///
/// This function parses request tools to extract MCP server configuration,
/// then ensures a dynamic client exists in the McpManager via `get_or_create_client()`.
/// The McpManager itself is returned (cloned Arc) for convenience, though the main
/// purpose is the side effect of registering the dynamic client.
///
/// Returns Some(manager) if a dynamic MCP tool was found and client was created/retrieved,
/// None if no MCP tools were found or connection failed.
pub async fn ensure_request_mcp_client(
    mcp_manager: &Arc<McpManager>,
    tools: &[ResponseTool],
) -> Option<Arc<McpManager>> {
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

/// Persist conversation items (delegates to openai::conversations module)
///
/// This is a convenience wrapper that delegates to the internal implementation
/// in the openai::conversations module.
pub async fn persist_conversation_items(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> Result<(), String> {
    crate::routers::openai::conversations::persist_conversation_items(
        conversation_storage,
        item_storage,
        response_storage,
        response_json,
        original_body,
    )
    .await
}
