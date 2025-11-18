//! Common utilities shared across router implementations

use std::collections::HashMap;

use tracing::warn;

use crate::protocols::responses::{ResponseTool, ResponseToolType};

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
