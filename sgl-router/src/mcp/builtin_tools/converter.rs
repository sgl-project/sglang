//! Built-in tool to MCP tool conversion
//!
//! Converts built-in tool types to MCP tool definitions by:
//! 1. Resolving built-in type to static MCP server via label
//! 2. Listing ALL tools from that server
//! 3. Giving each tool a synthetic name for tracking

use std::sync::Arc;

use tracing::debug;

use super::types::BuiltinToolType;
use crate::mcp::{config::Tool as McpTool, manager::McpManager};

/// Converts built-in tools to MCP tool definitions
pub struct BuiltinToolConverter;

impl BuiltinToolConverter {
    /// Convert built-in tools to MCP tools
    ///
    /// # Arguments
    /// * `builtin_types` - Detected built-in tool types
    /// * `mcp_manager` - MCP manager for server resolution
    ///
    /// # Returns
    /// Vec of (server_name, MCP tool definition with synthetic name)
    ///
    /// # Important
    /// - Only resolves to STATIC servers configured in mcp.yaml
    /// - Labels are FIXED per tool type (no customization)
    /// - ALL tools from the server are made available
    /// - Each tool gets a synthetic name: `{type}_builtin__{original_name}`
    ///
    /// # Example
    /// ```ignore
    /// let builtin_types = vec![BuiltinToolType::WebSearch];
    /// let mcp_tools = BuiltinToolConverter::convert_to_mcp(&builtin_types, &mcp_manager).await?;
    /// // Returns: vec![("brave-search", McpTool { name: "web_search_builtin__search", ...})]
    /// ```
    pub async fn convert_to_mcp(
        builtin_types: &[BuiltinToolType],
        mcp_manager: &Arc<McpManager>,
    ) -> Result<Vec<(String, McpTool)>, String> {
        let mut mcp_tools = Vec::new();

        for tool_type in builtin_types {
            // Use the FIXED label for this tool type
            let label = tool_type.fixed_label();
            debug!(
                "Resolving built-in tool type {:?} to MCP server using label '{}'",
                tool_type, label
            );

            // Resolve MCP server by fixed label (STATIC SERVERS ONLY)
            let server_name = mcp_manager.get_server_by_label(label).ok_or_else(|| {
                format!(
                    "No MCP server configured with label '{}'. \
                        Please add a server with label='{}' in mcp config yaml file.",
                    label, label
                )
            })?;

            debug!(
                "Found MCP server '{}' for label '{}', listing available tools",
                server_name, label
            );

            // Get ALL tools from MCP server
            let server_tools = mcp_manager
                .list_tools_for_server(&server_name)
                .await
                .map_err(|e| {
                    format!("Failed to list tools from server '{}': {}", server_name, e)
                })?;

            debug!(
                "Retrieved {} tool(s) from MCP server '{}' for built-in type {:?}",
                server_tools.len(),
                server_name,
                tool_type
            );

            // Convert ALL tools to use synthetic names
            let converted_tools = Self::convert_server_tools(&server_tools, tool_type);

            debug!(
                "Converted {} tool(s) with synthetic names for built-in type {:?}",
                converted_tools.len(),
                tool_type
            );

            // Add all converted tools with their server name
            for tool in converted_tools {
                mcp_tools.push((server_name.clone(), tool));
            }
        }

        Ok(mcp_tools)
    }

    /// Convert all tools from MCP server to built-in tools with synthetic names
    ///
    /// Each tool from the server gets a synthetic name that includes:
    /// - The built-in type prefix (e.g., "web_search_builtin")
    /// - The original tool name
    ///
    /// Format: `{builtin_type}_builtin__{original_name}`
    /// Example: `web_search_builtin__search`, `web_search_builtin__local_search`
    ///
    /// # Arguments
    /// * `server_tools` - Tools from the MCP server
    /// * `builtin_type` - Built-in tool type (for naming)
    ///
    /// # Returns
    /// Vec of MCP tools with synthetic names
    fn convert_server_tools(
        server_tools: &[McpTool],
        builtin_type: &BuiltinToolType,
    ) -> Vec<McpTool> {
        // Convert ALL tools from the MCP server
        // Each tool gets a synthetic name to track it as a built-in tool
        server_tools
            .iter()
            .map(|tool| {
                // Generate synthetic name: web_search_builtin__original_name
                let synthetic_name =
                    format!("{}_builtin__{}", builtin_type.fixed_label(), tool.name);

                McpTool {
                    name: std::borrow::Cow::Owned(synthetic_name),
                    title: tool.title.clone(),
                    description: tool.description.clone(),
                    input_schema: tool.input_schema.clone(),
                    output_schema: tool.output_schema.clone(),
                    annotations: tool.annotations.clone(),
                    icons: tool.icons.clone(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_convert_server_tools_web_search() {
        use serde_json::Map;

        let schema1 = Map::new();
        let schema2 = Map::new();

        let server_tools = vec![
            McpTool {
                name: std::borrow::Cow::Borrowed("search"),
                title: None,
                description: Some(std::borrow::Cow::Borrowed("Search the web")),
                input_schema: Arc::new(schema1),
                output_schema: None,
                annotations: None,
                icons: None,
            },
            McpTool {
                name: std::borrow::Cow::Borrowed("local_search"),
                title: None,
                description: Some(std::borrow::Cow::Borrowed("Search local businesses")),
                input_schema: Arc::new(schema2),
                output_schema: None,
                annotations: None,
                icons: None,
            },
        ];

        let converted =
            BuiltinToolConverter::convert_server_tools(&server_tools, &BuiltinToolType::WebSearch);

        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].name, "web_search_builtin__search");
        assert_eq!(converted[1].name, "web_search_builtin__local_search");
    }

    #[test]
    fn test_convert_server_tools_preserves_metadata() {
        use serde_json::Map;

        let mut schema = Map::new();
        let mut props = Map::new();
        let mut query_prop = Map::new();
        query_prop.insert("type".to_string(), serde_json::json!("string"));
        props.insert("query".to_string(), serde_json::json!(query_prop));
        schema.insert("properties".to_string(), serde_json::json!(props));

        let server_tools = vec![McpTool {
            name: std::borrow::Cow::Borrowed("search"),
            title: None,
            description: Some(std::borrow::Cow::Borrowed("Search the web")),
            input_schema: Arc::new(schema),
            output_schema: None,
            annotations: None,
            icons: None,
        }];

        let converted =
            BuiltinToolConverter::convert_server_tools(&server_tools, &BuiltinToolType::WebSearch);

        assert_eq!(converted.len(), 1);
        assert_eq!(
            converted[0].description,
            Some(std::borrow::Cow::Borrowed("Search the web"))
        );
        assert!(converted[0].input_schema.contains_key("properties"));
    }
}
