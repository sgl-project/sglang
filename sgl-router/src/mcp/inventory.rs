//! MCP tool, prompt, and resource inventory.
//!
//! Thread-safe cache for MCP capabilities across all connected servers.

use dashmap::DashMap;

use crate::mcp::config::Tool;

/// Cached tool with metadata
#[derive(Clone)]
pub struct CachedTool {
    pub server_label: String,
    pub server_url: String,
    pub tool: Tool,
}

/// Tool inventory with periodic refresh
///
/// Provides thread-safe caching of MCP tools.
/// Entries are refreshed periodically by background tasks.
pub struct ToolInventory {
    /// Map of server_label__tool_name -> cached tool
    tools: DashMap<String, CachedTool>,
}

impl ToolInventory {
    /// Create a new tool inventory
    pub fn new() -> Self {
        Self {
            tools: DashMap::new(),
        }
    }
}

impl Default for ToolInventory {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolInventory {
    // ============================================================================
    // Tool Methods
    // ============================================================================

    /// Get a tool if it exists
    ///
    /// Accepts qualified tool name in the format `server_label__tool_name`
    /// Returns (server_label, server_url, tool)
    pub fn get_tool(&self, qualified_tool_name: &str) -> Option<(String, String, Tool)> {
        self.tools.get(qualified_tool_name).map(|entry| {
            (
                entry.server_label.clone(),
                entry.server_url.clone(),
                entry.tool.clone(),
            )
        })
    }

    /// Check if tool exists
    pub fn has_tool(&self, qualified_tool_name: &str) -> bool {
        self.tools.contains_key(qualified_tool_name)
    }

    /// Insert or update a tool
    ///
    /// Constructs key as server_label__tool_name
    pub fn insert_tool(
        &self,
        tool_name: String,
        server_label: String,
        server_url: String,
        tool: Tool,
    ) {
        let qualified_name = format!("{}__{}", server_label, tool_name);
        self.tools.insert(
            qualified_name,
            CachedTool {
                server_label,
                server_url,
                tool,
            },
        );
    }

    /// Get all tools
    ///
    /// Returns Vec of (qualified_tool_name, server_label, Tool) where qualified_tool_name is server_label__tool_name
    pub fn list_tools(&self) -> Vec<(String, String, Tool)> {
        self.tools
            .iter()
            .map(|entry| {
                let (qualified_name, cached) = entry.pair();
                (
                    qualified_name.clone(),
                    cached.server_label.clone(),
                    cached.tool.clone(),
                )
            })
            .collect()
    }

    // ============================================================================
    // Server Management Methods
    // ============================================================================

    /// Clear all cached tools for a specific server
    ///
    /// Matches on server_label
    pub fn clear_server_tools(&self, server_label: &str) {
        self.tools
            .retain(|_, cached| cached.server_label != server_label);
    }

    /// Get count of cached tools
    pub fn count(&self) -> usize {
        self.tools.len()
    }

    /// Clear all cached tools
    pub fn clear_all(&self) {
        self.tools.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::config::Tool;

    // Helper to create a test tool
    fn create_test_tool(name: &str) -> Tool {
        use std::{borrow::Cow, sync::Arc};

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
            "test_tool".to_string(),
            "server1".to_string(),
            "url1".to_string(),
            tool.clone(),
        );

        let result = inventory.get_tool("server1__test_tool");
        assert!(result.is_some());

        let (server_name, server_url, retrieved_tool) = result.unwrap();
        assert_eq!(server_name, "server1");
        assert_eq!(server_url, "url1");
        assert_eq!(retrieved_tool.name, "test_tool");
    }

    #[test]
    fn test_has_tool() {
        let inventory = ToolInventory::new();
        let tool = create_test_tool("check_tool");

        assert!(!inventory.has_tool("server1__check_tool"));

        inventory.insert_tool(
            "check_tool".to_string(),
            "server1".to_string(),
            "url1".to_string(),
            tool,
        );

        assert!(inventory.has_tool("server1__check_tool"));
    }

    #[test]
    fn test_list_tools() {
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            "url1".to_string(),
            create_test_tool("tool1"),
        );
        inventory.insert_tool(
            "tool2".to_string(),
            "server1".to_string(),
            "url1".to_string(),
            create_test_tool("tool2"),
        );
        inventory.insert_tool(
            "tool3".to_string(),
            "server2".to_string(),
            "url2".to_string(),
            create_test_tool("tool3"),
        );

        let tools = inventory.list_tools();
        assert_eq!(tools.len(), 3);

        // Check that qualified names are returned
        let qualified_names: Vec<_> = tools.iter().map(|(name, _, _)| name.as_str()).collect();
        assert!(qualified_names.contains(&"server1__tool1"));
        assert!(qualified_names.contains(&"server1__tool2"));
        assert!(qualified_names.contains(&"server2__tool3"));
    }

    #[test]
    fn test_clear_server_tools() {
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            "url1".to_string(),
            create_test_tool("tool1"),
        );
        inventory.insert_tool(
            "tool2".to_string(),
            "server2".to_string(),
            "url2".to_string(),
            create_test_tool("tool2"),
        );

        assert_eq!(inventory.list_tools().len(), 2);

        inventory.clear_server_tools("server1");

        let tools = inventory.list_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].0, "server2__tool2");
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
                    format!("tool_{}", i),
                    format!("server_{}", i % 3),
                    format!("url_{}", i % 3),
                    tool,
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
            "tool1".to_string(),
            "server1".to_string(),
            "url1".to_string(),
            create_test_tool("tool1"),
        );

        assert_eq!(inventory.count(), 1);

        inventory.clear_all();

        assert_eq!(inventory.count(), 0);
    }
}
