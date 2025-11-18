//! MCP tool inventory.
//!
//! Thread-safe cache for MCP tools with composite keys to prevent name collisions.

use dashmap::DashMap;

use crate::mcp::config::Tool;

/// Tool inventory with composite key support
///
/// Provides thread-safe caching of MCP tools using composite keys (server_label, tool_name).
/// This prevents name collisions across multiple MCP servers and enables per-request
/// dynamic tool management.
///
/// Architecture:
/// - **Static tools**: One shared inventory for tools from config (persists for server lifetime)
/// - **Dynamic tools**: Per-request inventory for tools from request MCP servers (request lifetime only)
pub struct ToolInventory {
    /// Map of (server_label, tool_name) -> (Tool, server_url)
    tools: DashMap<(String, String), (Tool, String)>,
}

impl ToolInventory {
    /// Create a new tool inventory
    pub fn new() -> Self {
        Self {
            tools: DashMap::new(),
        }
    }

    pub fn get_tool(&self, server_label: &str, tool_name: &str) -> Option<(Tool, String)> {
        self.tools
            .get(&(server_label.to_string(), tool_name.to_string()))
            .map(|entry| entry.value().clone())
    }

    pub fn has_tool(&self, server_label: &str, tool_name: &str) -> bool {
        self.tools
            .contains_key(&(server_label.to_string(), tool_name.to_string()))
    }

    pub fn insert_tool(
        &self,
        server_label: String,
        tool_name: String,
        tool: Tool,
        server_url: String,
    ) {
        self.tools
            .insert((server_label, tool_name), (tool, server_url));
    }

    pub fn list_tools(&self) -> Vec<(String, String, Tool, String)> {
        self.tools
            .iter()
            .map(|entry| {
                let ((server_label, tool_name), (tool, server_url)) = entry.pair();
                (
                    server_label.clone(),
                    tool_name.clone(),
                    tool.clone(),
                    server_url.clone(),
                )
            })
            .collect()
    }

    /// Clear all tools for a specific server
    ///
    /// Used when LRU evicts a client or when cleaning up after a request.
    pub fn clear_server_tools(&self, server_label: &str) {
        self.tools.retain(|(label, _), _| label != server_label);
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
            let ((server_label, tool_name), (tool, server_url)) = entry.pair();
            self.tools.insert(
                (server_label.clone(), tool_name.clone()),
                (tool.clone(), server_url.clone()),
            );
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

        inventory.insert_tool(
            "server1".to_string(),
            "test_tool".to_string(),
            tool.clone(),
            "http://server1:8000".to_string(),
        );

        let result = inventory.get_tool("server1", "test_tool");
        assert!(result.is_some());

        let (retrieved_tool, server_url) = result.unwrap();
        assert_eq!(retrieved_tool.name, "test_tool");
        assert_eq!(server_url, "http://server1:8000");
    }

    #[test]
    fn test_has_tool() {
        let inventory = ToolInventory::new();
        let tool = create_test_tool("check_tool");

        assert!(!inventory.has_tool("server1", "check_tool"));

        inventory.insert_tool(
            "server1".to_string(),
            "check_tool".to_string(),
            tool,
            "http://server1:8000".to_string(),
        );

        assert!(inventory.has_tool("server1", "check_tool"));
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

        let result1 = inventory.get_tool("server1", "weather");
        let result2 = inventory.get_tool("server2", "weather");

        assert!(result1.is_some());
        assert!(result2.is_some());

        let (_, url1) = result1.unwrap();
        let (_, url2) = result2.unwrap();

        assert_eq!(url1, "http://server1:8000");
        assert_eq!(url2, "http://server2:8000");
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
        for (server_label, tool_name, tool, server_url) in tools {
            assert!(!server_label.is_empty());
            assert!(!tool_name.is_empty());
            assert_eq!(tool.name, tool_name);
            assert!(server_url.starts_with("http://"));
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
        assert!(!inventory.has_tool("server1", "tool1"));
        assert!(!inventory.has_tool("server1", "tool2"));
        assert!(inventory.has_tool("server2", "tool3"));
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
        assert!(static_inventory.has_tool("static-server", "tool1"));
        assert!(static_inventory.has_tool("dynamic-server", "tool2"));
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
        let (_, url) = inventory1.get_tool("server1", "tool1").unwrap();
        assert_eq!(url, "http://new-url:8000");
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
}
