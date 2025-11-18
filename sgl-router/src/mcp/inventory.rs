//! MCP tool inventory.
//!
//! Thread-safe cache for MCP tools with formatted name keys to prevent name collisions.

use dashmap::DashMap;

use crate::mcp::config::Tool;

/// Cached tool entry with metadata
///
/// Represents a tool with its associated server information.
/// The formatted name (`server_label__tool_name`) is used as the key in the inventory.
#[derive(Clone, Debug)]
pub struct CachedTool {
    /// Server label (identifier for the MCP server)
    pub server_label: String,
    /// Tool name (as provided by the MCP server)
    pub tool_name: String,
    /// Tool definition with schema
    pub tool: Tool,
    /// Server URL (where the tool is hosted)
    pub server_url: String,
}

impl CachedTool {
    /// Create a new cached tool entry
    pub fn new(server_label: String, tool_name: String, tool: Tool, server_url: String) -> Self {
        Self {
            server_label,
            tool_name,
            tool,
            server_url,
        }
    }

    /// Get the formatted name for this tool
    ///
    /// Returns: `server_label__tool_name`
    pub fn formatted_name(&self) -> String {
        format_tool_name(&self.server_label, &self.tool_name)
    }
}

/// Separator for formatted tool names: `server_label__tool_name`
pub const TOOL_NAME_SEPARATOR: &str = "__";

/// Format tool name as `server_label__tool_name`
///
/// This creates a globally unique tool identifier by combining the server label
/// and tool name with a double underscore separator.
///
/// # Examples
/// ```
/// use sglang_router_rs::mcp::inventory::format_tool_name;
/// let formatted = format_tool_name("my-server", "get_weather");
/// assert_eq!(formatted, "my-server__get_weather");
/// ```
pub fn format_tool_name(server_label: &str, tool_name: &str) -> String {
    format!("{}{}{}", server_label, TOOL_NAME_SEPARATOR, tool_name)
}

/// Tool inventory with formatted name keys
///
/// Provides thread-safe caching of MCP tools using formatted names as keys.
/// Formatted names follow the pattern `server_label__tool_name` to prevent name collisions
/// across multiple MCP servers and enable per-request dynamic tool management.
///
/// Architecture:
/// - **Static tools**: One shared inventory for tools from config (persists for server lifetime)
/// - **Dynamic tools**: Per-request inventory for tools from request MCP servers (request lifetime only)
///
/// Key format: `server_label__tool_name` (e.g., "brave-search__web_search")
pub struct ToolInventory {
    /// Map of formatted_name -> CachedTool
    /// Key format: "server_label__tool_name"
    tools: DashMap<String, CachedTool>,
}

impl ToolInventory {
    /// Create a new tool inventory
    pub fn new() -> Self {
        Self {
            tools: DashMap::new(),
        }
    }

    /// Get a cached tool by its formatted name
    ///
    /// # Arguments
    /// * `formatted_name` - The formatted name in format "server_label__tool_name"
    ///
    /// # Returns
    /// A cloned `CachedTool` if found
    pub fn get_tool(&self, formatted_name: &str) -> Option<CachedTool> {
        self.tools
            .get(formatted_name)
            .map(|entry| entry.value().clone())
    }

    /// Check if a tool exists by its formatted name
    pub fn has_tool(&self, formatted_name: &str) -> bool {
        self.tools.contains_key(formatted_name)
    }

    /// Insert a tool into the inventory
    ///
    /// Automatically creates the CachedTool and uses formatted name as key.
    ///
    /// # Arguments
    /// * `server_label` - Server label (identifier for the MCP server)
    /// * `tool_name` - Tool name (as provided by the MCP server)
    /// * `tool` - Tool definition with schema
    /// * `server_url` - Server URL (where the tool is hosted)
    pub fn insert_tool(
        &self,
        server_label: String,
        tool_name: String,
        tool: Tool,
        server_url: String,
    ) {
        let cached_tool = CachedTool::new(server_label, tool_name, tool, server_url);
        let formatted_name = cached_tool.formatted_name();
        self.tools.insert(formatted_name, cached_tool);
    }

    /// List all cached tools
    ///
    /// Returns a vector of all CachedTool entries
    pub fn list_tools(&self) -> Vec<CachedTool> {
        self.tools
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Clear all tools for a specific server
    ///
    /// Used when LRU evicts a client or when cleaning up after a request.
    ///
    /// # Arguments
    /// * `server_prefix` - The server label prefix (e.g., "server1" will clear "server1__*")
    pub fn clear_server_tools(&self, server_prefix: &str) {
        let prefix_with_sep = format!("{}__", server_prefix);
        self.tools
            .retain(|formatted_name, _| !formatted_name.starts_with(&prefix_with_sep));
    }

    /// Get count of cached tools
    pub fn count(&self) -> usize {
        self.tools.len()
    }

    /// Clear all cached tools
    pub fn clear_all(&self) {
        self.tools.clear();
    }

    /// Merge another inventory into this one
    ///
    /// Used to combine static and dynamic tools for request processing.
    /// Entries from `other` will overwrite entries with the same key.
    pub fn merge(&self, other: &ToolInventory) {
        for entry in other.tools.iter() {
            let (formatted_name, cached_tool) = entry.pair();
            self.tools
                .insert(formatted_name.clone(), cached_tool.clone());
        }
    }
}

impl Default for ToolInventory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use super::*;

    // Helper to format tool names for tests
    fn format_name(server_label: &str, tool_name: &str) -> String {
        format_tool_name(server_label, tool_name)
    }

    // Helper to create a test tool
    fn create_test_tool(name: &str) -> Tool {
        let schema_obj = serde_json::json!({
            "type": "object",
            "properties": {}
        });

        let schema_map = if let serde_json::Value::Object(m) = schema_obj {
            m
        } else {
            serde_json::Map::new()
        };

        Tool {
            name: Cow::Owned(name.to_string()),
            title: None,
            description: Some(Cow::Owned(format!("Test tool: {}", name))),
            input_schema: Arc::new(schema_map),
            output_schema: None,
            annotations: None,
            icons: None,
        }
    }

    #[test]
    fn test_tool_insert_and_get() {
        let inventory = ToolInventory::new();
        let tool = create_test_tool("test_tool");
        let formatted = format_name("server1", "test_tool");

        inventory.insert_tool(
            "server1".to_string(),
            "test_tool".to_string(),
            tool.clone(),
            "http://server1:8000".to_string(),
        );

        let result = inventory.get_tool(&formatted);
        assert!(result.is_some());

        let cached_tool = result.unwrap();
        assert_eq!(cached_tool.tool.name, "test_tool");
        assert_eq!(cached_tool.server_url, "http://server1:8000");
        assert_eq!(cached_tool.server_label, "server1");
        assert_eq!(cached_tool.tool_name, "test_tool");
    }

    #[test]
    fn test_has_tool() {
        let inventory = ToolInventory::new();
        let tool = create_test_tool("check_tool");
        let formatted = format_name("server1", "check_tool");

        assert!(!inventory.has_tool(&formatted));

        inventory.insert_tool(
            "server1".to_string(),
            "check_tool".to_string(),
            tool,
            "http://server1:8000".to_string(),
        );

        assert!(inventory.has_tool(&formatted));
    }

    #[test]
    fn test_composite_key_prevents_collisions() {
        let inventory = ToolInventory::new();
        let tool1 = create_test_tool("weather");
        let tool2 = create_test_tool("weather");

        // Same tool name, different servers - should not collide
        inventory.insert_tool(
            "server1".to_string(),
            "weather".to_string(),
            tool1,
            "http://server1:8000".to_string(),
        );
        inventory.insert_tool(
            "server2".to_string(),
            "weather".to_string(),
            tool2,
            "http://server2:8000".to_string(),
        );

        let result1 = inventory.get_tool(&format_name("server1", "weather"));
        let result2 = inventory.get_tool(&format_name("server2", "weather"));

        assert!(result1.is_some());
        assert!(result2.is_some());

        let cached1 = result1.unwrap();
        let cached2 = result2.unwrap();

        assert_eq!(cached1.server_url, "http://server1:8000");
        assert_eq!(cached2.server_url, "http://server2:8000");
    }

    #[test]
    fn test_list_tools() {
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "server1".to_string(),
            "tool1".to_string(),
            create_test_tool("tool1"),
            "http://server1:8000".to_string(),
        );
        inventory.insert_tool(
            "server1".to_string(),
            "tool2".to_string(),
            create_test_tool("tool2"),
            "http://server1:8000".to_string(),
        );
        inventory.insert_tool(
            "server2".to_string(),
            "tool3".to_string(),
            create_test_tool("tool3"),
            "http://server2:8000".to_string(),
        );

        let tools = inventory.list_tools();
        assert_eq!(tools.len(), 3);

        // Verify all entries have correct structure
        for cached_tool in tools {
            assert!(!cached_tool.server_label.is_empty());
            assert!(!cached_tool.tool_name.is_empty());
            assert_eq!(cached_tool.tool.name, cached_tool.tool_name);
            assert!(cached_tool.server_url.starts_with("http://"));
        }
    }

    #[test]
    fn test_clear_server_tools() {
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "server1".to_string(),
            "tool1".to_string(),
            create_test_tool("tool1"),
            "http://server1:8000".to_string(),
        );
        inventory.insert_tool(
            "server1".to_string(),
            "tool2".to_string(),
            create_test_tool("tool2"),
            "http://server1:8000".to_string(),
        );
        inventory.insert_tool(
            "server2".to_string(),
            "tool3".to_string(),
            create_test_tool("tool3"),
            "http://server2:8000".to_string(),
        );

        assert_eq!(inventory.count(), 3);

        inventory.clear_server_tools("server1");

        assert_eq!(inventory.count(), 1);
        assert!(!inventory.has_tool(&format_name("server1", "tool1")));
        assert!(!inventory.has_tool(&format_name("server1", "tool2")));
        assert!(inventory.has_tool(&format_name("server2", "tool3")));
    }

    #[test]
    fn test_merge_inventories() {
        let static_inventory = ToolInventory::new();
        let dynamic_inventory = ToolInventory::new();

        // Add static tools
        static_inventory.insert_tool(
            "static-server".to_string(),
            "tool1".to_string(),
            create_test_tool("tool1"),
            "http://static:8000".to_string(),
        );

        // Add dynamic tools
        dynamic_inventory.insert_tool(
            "dynamic-server".to_string(),
            "tool2".to_string(),
            create_test_tool("tool2"),
            "http://dynamic:8000".to_string(),
        );

        // Merge dynamic into static
        static_inventory.merge(&dynamic_inventory);

        // Should have both tools
        assert_eq!(static_inventory.count(), 2);
        assert!(static_inventory.has_tool(&format_name("static-server", "tool1")));
        assert!(static_inventory.has_tool(&format_name("dynamic-server", "tool2")));
    }

    #[test]
    fn test_merge_overwrites_duplicates() {
        let inventory1 = ToolInventory::new();
        let inventory2 = ToolInventory::new();

        // Add same key to both
        inventory1.insert_tool(
            "server1".to_string(),
            "tool1".to_string(),
            create_test_tool("tool1"),
            "http://old-url:8000".to_string(),
        );

        inventory2.insert_tool(
            "server1".to_string(),
            "tool1".to_string(),
            create_test_tool("tool1"),
            "http://new-url:8000".to_string(),
        );

        // Merge inventory2 into inventory1
        inventory1.merge(&inventory2);

        // Should have new URL
        let cached_tool = inventory1
            .get_tool(&format_name("server1", "tool1"))
            .unwrap();
        assert_eq!(cached_tool.server_url, "http://new-url:8000");
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        use std::sync::Arc;

        let inventory = Arc::new(ToolInventory::new());

        // Spawn multiple tasks that insert tools concurrently
        let mut handles = vec![];
        for i in 0..10 {
            let inv = Arc::clone(&inventory);
            let handle = tokio::spawn(async move {
                let tool = create_test_tool(&format!("tool_{}", i));
                inv.insert_tool(
                    format!("server_{}", i % 3),
                    format!("tool_{}", i),
                    tool,
                    format!("http://server{}:8000", i % 3),
                );
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Should have 10 tools
        assert_eq!(inventory.count(), 10);
    }

    #[test]
    fn test_clear_all() {
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "server1".to_string(),
            "tool1".to_string(),
            create_test_tool("tool1"),
            "http://server1:8000".to_string(),
        );
        inventory.insert_tool(
            "server2".to_string(),
            "tool2".to_string(),
            create_test_tool("tool2"),
            "http://server2:8000".to_string(),
        );

        assert_eq!(inventory.count(), 2);

        inventory.clear_all();

        assert_eq!(inventory.count(), 0);
    }

    #[test]
    fn test_format_tool_name() {
        let formatted = format_tool_name("my-server", "get_weather");
        assert_eq!(formatted, "my-server__get_weather");

        let formatted = format_tool_name("my_server", "get_weather_info");
        assert_eq!(formatted, "my_server__get_weather_info");
    }

    #[test]
    fn test_cached_tool_formatted_name() {
        let tool = create_test_tool("weather");
        let cached = CachedTool::new(
            "server1".to_string(),
            "weather".to_string(),
            tool,
            "http://server1:8000".to_string(),
        );

        assert_eq!(cached.formatted_name(), "server1__weather");
        assert_eq!(
            cached.formatted_name(),
            format_tool_name("server1", "weather")
        );
    }

    #[test]
    fn test_tool_name_separator() {
        assert_eq!(TOOL_NAME_SEPARATOR, "__");

        // Verify separator is used correctly in formatting
        let formatted = format_tool_name("server", "tool");
        assert!(formatted.contains(TOOL_NAME_SEPARATOR));
    }
}
