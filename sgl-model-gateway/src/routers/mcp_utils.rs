//! Shared MCP utilities for routers.
//!
//! This module provides shared MCP-related functionality that can be
//! used across different router implementations (OpenAI, gRPC regular, gRPC harmony).

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use serde_json::Value;
use tracing::warn;

use crate::{
    mcp::{self, McpManager, McpServerConfig, McpTransport},
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
    /// Server keys for filtering MCP tools.
    /// Contains keys for dynamic servers that were connected for this request.
    pub server_keys: Vec<String>,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            server_keys: Vec::new(),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build a mapping of server_key -> server_label for request-scoped MCP tools.
pub fn build_server_label_map(tools: Option<&[ResponseTool]>) -> HashMap<String, String> {
    let mut labels = HashMap::new();

    if let Some(tools) = tools {
        for tool in tools {
            if tool.r#type != ResponseToolType::Mcp {
                continue;
            }

            let Some(server_url) = tool.server_url.as_ref().map(|s| s.trim().to_string()) else {
                continue;
            };

            // Validate URL scheme
            if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
                continue;
            }

            let label = tool
                .server_label
                .as_ref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| server_url.clone());

            let transport = if server_url.contains("/sse") {
                McpTransport::Sse {
                    url: server_url.clone(),
                    token: tool.authorization.clone(),
                    headers: tool.headers.clone(),
                }
            } else {
                McpTransport::Streamable {
                    url: server_url.clone(),
                    token: tool.authorization.clone(),
                    headers: tool.headers.clone(),
                }
            };

            let server_config = McpServerConfig {
                name: label.clone(),
                transport,
                proxy: None,
                required: false,
            };

            let server_key = McpManager::server_key(&server_config);
            labels.insert(server_key, label);
        }
    }

    labels
}

/// Build a mapping of server_label -> allowed tool names (None means no filter).
pub fn build_allowed_tools_map(
    tools: Option<&[ResponseTool]>,
) -> HashMap<String, Option<HashSet<String>>> {
    let mut allowed = HashMap::new();

    if let Some(tools) = tools {
        for tool in tools {
            if tool.r#type != ResponseToolType::Mcp {
                continue;
            }

            let Some(label) = tool
                .server_label
                .as_ref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            else {
                continue;
            };

            let value = tool.allowed_tools.as_ref().map(|names| {
                names
                    .iter()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect::<HashSet<String>>()
            });

            allowed.insert(label.to_string(), value);
        }
    }

    allowed
}

pub fn resolve_server_label(server_key: &str, server_labels: &HashMap<String, String>) -> String {
    if let Some(label) = server_labels.get(server_key) {
        return label.clone();
    }

    if let Some((url, _)) = server_key.split_once('#') {
        return url.to_string();
    }

    server_key.to_string()
}

pub fn filter_tools_for_server(
    tools: &[mcp::Tool],
    server_label: &str,
    allowed_tools: &HashMap<String, Option<HashSet<String>>>,
) -> Vec<mcp::Tool> {
    let allow_set = allowed_tools.get(server_label);

    if allow_set.is_some_and(|set| set.as_ref().is_some_and(|s| s.is_empty())) {
        return Vec::new();
    }

    if let Some(Some(allowed)) = allow_set {
        tools
            .iter()
            .filter(|tool| allowed.contains(tool.name.as_ref()))
            .cloned()
            .collect()
    } else {
        tools.to_vec()
    }
}

// ============================================================================
// MCP Tool Name Encoding
// ============================================================================

/// Encode an MCP function name with server label prefix.
///
/// Format: `mcp__{server_label}__{tool_name}`
///
/// This encoding allows distinguishing tools from different MCP servers
/// that may have the same tool name.
pub fn encode_mcp_function_name(server_label: &str, tool_name: &str) -> String {
    format!("mcp__{}__{}", server_label, tool_name)
}

/// Decode an MCP function name to extract server label and tool name.
///
/// Returns `Some((server_label, tool_name))` if the name matches the encoding format,
/// `None` otherwise.
pub fn decode_mcp_function_name(name: &str) -> Option<(String, String)> {
    let rest = name.strip_prefix("mcp__")?;
    let (label, tool_name) = rest.split_once("__")?;
    if label.is_empty() || tool_name.is_empty() {
        return None;
    }
    Some((label.to_string(), tool_name.to_string()))
}

// ============================================================================
// MCP Tool Lookup
// ============================================================================

/// Lookup structure for MCP tool routing.
///
/// Maps encoded tool names to their server keys and schemas,
/// enabling correct routing of tool calls to the right MCP server.
#[derive(Debug, Clone, Default)]
pub struct McpToolLookup {
    /// Maps encoded tool name -> server key
    pub tool_servers: HashMap<String, String>,
    /// Maps encoded tool name -> original tool name
    pub tool_names: HashMap<String, String>,
    /// Maps encoded tool name -> input schema
    pub tool_schemas: HashMap<String, Value>,
}

/// List tools grouped by server.
///
/// Returns a vector of (server_key, tools) pairs, maintaining the order
/// of server_keys where possible.
pub fn list_tools_by_server(
    mcp: &McpManager,
    server_keys: &[String],
) -> Vec<(String, Vec<mcp::Tool>)> {
    let mut tools_by_server: HashMap<String, Vec<mcp::Tool>> = HashMap::new();
    let server_key_set: HashSet<&str> = server_keys.iter().map(|s| s.as_str()).collect();

    for (_tool_name, server_key, tool) in mcp.inventory().list_tools() {
        if mcp.is_static_server(&server_key) || server_key_set.contains(server_key.as_str()) {
            tools_by_server.entry(server_key).or_default().push(tool);
        }
    }

    // Maintain order: first the requested server_keys, then any remaining
    let mut ordered = Vec::new();
    for server_key in server_keys {
        if let Some(tools) = tools_by_server.remove(server_key) {
            ordered.push((server_key.clone(), tools));
        }
    }

    // Add any remaining servers (static servers not in server_keys)
    if !tools_by_server.is_empty() {
        let mut remaining: Vec<(String, Vec<mcp::Tool>)> = tools_by_server.into_iter().collect();
        remaining.sort_by(|a, b| a.0.cmp(&b.0));
        ordered.extend(remaining);
    }

    ordered
}

/// Build McpToolLookup from MCP manager and server configuration.
pub fn build_mcp_tool_lookup(
    mcp: &McpManager,
    server_keys: &[String],
    server_labels: &HashMap<String, String>,
    allowed_tools: &HashMap<String, Option<HashSet<String>>>,
) -> McpToolLookup {
    let mut lookup = McpToolLookup::default();

    for (server_key, tools) in list_tools_by_server(mcp, server_keys) {
        let server_label = resolve_server_label(&server_key, server_labels);
        let filtered_tools = filter_tools_for_server(&tools, &server_label, allowed_tools);

        for tool in filtered_tools {
            let tool_name = tool.name.as_ref();
            let encoded_name = encode_mcp_function_name(&server_label, tool_name);
            lookup
                .tool_servers
                .insert(encoded_name.clone(), server_key.clone());
            lookup
                .tool_names
                .insert(encoded_name.clone(), tool_name.to_string());
            lookup
                .tool_schemas
                .insert(encoded_name, Value::Object((*tool.input_schema).clone()));
        }
    }

    lookup
}

// ============================================================================
// MCP Connection
// ============================================================================

/// Ensure MCP clients are connected for all request-level MCP tools.
///
/// This function extracts MCP server configurations from ALL request tools (server_url, authorization)
/// and ensures client connections are established via the connection pool.
///
/// Returns `Some((manager, server_keys))` if MCP tools were found and clients created,
/// `None` if no MCP tools with server_url were found.
pub async fn ensure_request_mcp_client(
    mcp_manager: &Arc<McpManager>,
    tools: &[ResponseTool],
) -> Option<(Arc<McpManager>, Vec<String>)> {
    let mut server_keys = Vec::new();
    let mut has_mcp_tools = false;

    // Process all MCP tools
    for tool in tools {
        if matches!(tool.r#type, ResponseToolType::Mcp) && tool.server_url.is_some() {
            has_mcp_tools = true;
            let Some(server_url) = tool.server_url.as_ref().map(|s| s.trim().to_string()) else {
                continue;
            };

            // Validate URL scheme
            if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
                warn!(
                    "Ignoring MCP server_url with unsupported scheme: {}",
                    server_url
                );
                continue;
            }

            // Extract server label and auth token
            let name = tool
                .server_label
                .clone()
                .unwrap_or_else(|| "request-mcp".to_string());
            let token = tool.authorization.clone();
            let headers = tool.headers.clone();

            // Determine transport type based on URL pattern
            let transport = if server_url.contains("/sse") {
                McpTransport::Sse {
                    url: server_url.clone(),
                    token,
                    headers,
                }
            } else {
                McpTransport::Streamable {
                    url: server_url.clone(),
                    token,
                    headers,
                }
            };

            // Create server config
            let server_config = McpServerConfig {
                name,
                transport,
                proxy: None,
                required: false,
            };

            // Get the server key for tracking
            let server_key = McpManager::server_key(&server_config);

            // Use get_or_create_client to establish connection
            match mcp_manager.get_or_create_client(server_config).await {
                Ok(_client) => {
                    // Track this server for filtering
                    if !server_keys.contains(&server_key) {
                        server_keys.push(server_key);
                    }
                }
                Err(err) => {
                    warn!(
                        "Failed to get/create MCP connection for {}: {}",
                        server_key, err
                    );
                    // Continue processing other tools
                }
            }
        }
    }

    if has_mcp_tools && !server_keys.is_empty() {
        Some((mcp_manager.clone(), server_keys))
    } else {
        None
    }
}
