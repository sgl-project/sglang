//! MCP tool, prompt, and resource inventory.
//!
//! Thread-safe cache for MCP capabilities across all connected servers.

use dashmap::DashMap;

use crate::mcp::config::{Prompt, RawResource, Tool};

/// Cached tool with metadata
#[derive(Clone)]
pub(crate) struct CachedTool {
    pub server_name: String,
    pub tool: Tool,
}

/// Cached prompt with metadata
#[derive(Clone)]
pub(crate) struct CachedPrompt {
    pub server_name: String,
    pub prompt: Prompt,
}

/// Cached resource with metadata
#[derive(Clone)]
pub(crate) struct CachedResource {
    pub server_name: String,
    pub resource: RawResource,
}

/// Tool inventory with periodic refresh
///
/// Provides thread-safe caching of MCP tools, prompts, and resources.
/// Entries are refreshed periodically by background tasks.
pub struct ToolInventory {
    /// Map of (server_name, tool_name) -> cached tool
    tools: DashMap<(String, String), CachedTool>,

    /// Map of prompt_name -> cached prompt
    prompts: DashMap<String, CachedPrompt>,

    /// Map of resource_uri -> cached resource
    resources: DashMap<String, CachedResource>,
}

impl ToolInventory {
    /// Create a new tool inventory
    pub fn new() -> Self {
        Self {
            tools: DashMap::new(),
            prompts: DashMap::new(),
            resources: DashMap::new(),
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
    pub fn get_tool(&self, tool_name: &str) -> Option<(String, Tool)> {
        self.tools
            .iter()
            .find(|entry| entry.key().1 == tool_name)
            .map(|entry| {
                (
                    entry.value().server_name.clone(),
                    entry.value().tool.clone(),
                )
            })
    }

    /// Check if tool exists
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tools.iter().any(|entry| entry.key().1 == tool_name)
    }

    /// Insert or update a tool
    pub fn insert_tool(&self, tool_name: String, server_name: String, tool: Tool) {
        let key = (server_name.clone(), tool_name);
        self.tools.insert(key, CachedTool { server_name, tool });
    }

    /// Get all tools
    pub fn list_tools(&self) -> Vec<(String, String, Tool)> {
        self.tools
            .iter()
            .map(|entry| {
                let (server_name, tool_name) = entry.key();
                (
                    tool_name.clone(),
                    server_name.clone(),
                    entry.value().tool.clone(),
                )
            })
            .collect()
    }

    // ============================================================================
    // Prompt Methods
    // ============================================================================

    /// Get a prompt if it exists
    pub fn get_prompt(&self, prompt_name: &str) -> Option<(String, Prompt)> {
        self.prompts
            .get(prompt_name)
            .map(|entry| (entry.server_name.clone(), entry.prompt.clone()))
    }

    /// Check if prompt exists
    pub fn has_prompt(&self, prompt_name: &str) -> bool {
        self.prompts.contains_key(prompt_name)
    }

    /// Insert or update a prompt
    pub fn insert_prompt(&self, prompt_name: String, server_name: String, prompt: Prompt) {
        self.prompts.insert(
            prompt_name,
            CachedPrompt {
                server_name,
                prompt,
            },
        );
    }

    /// Get all prompts
    pub fn list_prompts(&self) -> Vec<(String, String, Prompt)> {
        self.prompts
            .iter()
            .map(|entry| {
                let (name, cached) = entry.pair();
                (
                    name.clone(),
                    cached.server_name.clone(),
                    cached.prompt.clone(),
                )
            })
            .collect()
    }

    // ============================================================================
    // Resource Methods
    // ============================================================================

    /// Get a resource if it exists
    pub fn get_resource(&self, resource_uri: &str) -> Option<(String, RawResource)> {
        self.resources
            .get(resource_uri)
            .map(|entry| (entry.server_name.clone(), entry.resource.clone()))
    }

    /// Check if resource exists
    pub fn has_resource(&self, resource_uri: &str) -> bool {
        self.resources.contains_key(resource_uri)
    }

    /// Insert or update a resource
    pub fn insert_resource(
        &self,
        resource_uri: String,
        server_name: String,
        resource: RawResource,
    ) {
        self.resources.insert(
            resource_uri,
            CachedResource {
                server_name,
                resource,
            },
        );
    }

    /// Get all resources
    pub fn list_resources(&self) -> Vec<(String, String, RawResource)> {
        self.resources
            .iter()
            .map(|entry| {
                let (uri, cached) = entry.pair();
                (
                    uri.clone(),
                    cached.server_name.clone(),
                    cached.resource.clone(),
                )
            })
            .collect()
    }

    // ============================================================================
    // Server Management Methods
    // ============================================================================

    /// Clear all cached items for a specific server (called when LRU evicts client)
    pub fn clear_server_tools(&self, server_name: &str) {
        self.tools.retain(|key, _| key.0 != server_name);
        self.prompts
            .retain(|_, cached| cached.server_name != server_name);
        self.resources
            .retain(|_, cached| cached.server_name != server_name);
    }

    /// Get count of cached items
    pub fn counts(&self) -> (usize, usize, usize) {
        (self.tools.len(), self.prompts.len(), self.resources.len())
    }

    /// Clear all cached items
    pub fn clear_all(&self) {
        self.tools.clear();
        self.prompts.clear();
        self.resources.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::config::{Prompt, RawResource, Tool};

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

    // Helper to create a test prompt
    fn create_test_prompt(name: &str) -> Prompt {
        Prompt {
            name: name.to_string(),
            title: None,
            description: Some(format!("Test prompt: {}", name)),
            arguments: None,
            icons: None,
        }
    }

    // Helper to create a test resource
    fn create_test_resource(uri: &str) -> RawResource {
        RawResource {
            uri: uri.to_string(),
            name: uri.to_string(),
            title: None,
            description: Some(format!("Test resource: {}", uri)),
            mime_type: Some("text/plain".to_string()),
            size: None,
            icons: None,
        }
    }

    #[test]
    fn test_tool_insert_and_get() {
        let inventory = ToolInventory::new();
        let tool = create_test_tool("test_tool");

        inventory.insert_tool("test_tool".to_string(), "server1".to_string(), tool.clone());

        let result = inventory.get_tool("test_tool");
        assert!(result.is_some());

        let (server_name, retrieved_tool) = result.unwrap();
        assert_eq!(server_name, "server1");
        assert_eq!(retrieved_tool.name, "test_tool");
    }

    #[test]
    fn test_has_tool() {
        let inventory = ToolInventory::new();
        let tool = create_test_tool("check_tool");

        assert!(!inventory.has_tool("check_tool"));

        inventory.insert_tool("check_tool".to_string(), "server1".to_string(), tool);

        assert!(inventory.has_tool("check_tool"));
    }

    #[test]
    fn test_list_tools() {
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
        inventory.insert_tool(
            "tool2".to_string(),
            "server1".to_string(),
            create_test_tool("tool2"),
        );
        inventory.insert_tool(
            "tool3".to_string(),
            "server2".to_string(),
            create_test_tool("tool3"),
        );

        let tools = inventory.list_tools();
        assert_eq!(tools.len(), 3);
    }

    #[test]
    fn test_list_tools_with_duplicates() {
        use std::collections::HashSet;

        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
        inventory.insert_tool(
            "tool1".to_string(),
            "server2".to_string(),
            create_test_tool("tool1"),
        );

        let tools = inventory.list_tools();
        assert_eq!(tools.len(), 2);

        let servers: HashSet<String> = tools.into_iter().map(|(_, server, _)| server).collect();
        assert_eq!(
            servers,
            HashSet::from(["server1".to_string(), "server2".to_string()])
        );
    }

    #[test]
    fn test_clear_server_tools() {
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
        inventory.insert_tool(
            "tool2".to_string(),
            "server2".to_string(),
            create_test_tool("tool2"),
        );

        assert_eq!(inventory.list_tools().len(), 2);

        inventory.clear_server_tools("server1");

        let tools = inventory.list_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].0, "tool2");
    }

    #[test]
    fn test_prompt_operations() {
        let inventory = ToolInventory::new();
        let prompt = create_test_prompt("test_prompt");

        inventory.insert_prompt(
            "test_prompt".to_string(),
            "server1".to_string(),
            prompt.clone(),
        );

        assert!(inventory.has_prompt("test_prompt"));

        let result = inventory.get_prompt("test_prompt");
        assert!(result.is_some());

        let (server_name, retrieved_prompt) = result.unwrap();
        assert_eq!(server_name, "server1");
        assert_eq!(retrieved_prompt.name, "test_prompt");
    }

    #[test]
    fn test_resource_operations() {
        let inventory = ToolInventory::new();
        let resource = create_test_resource("file:///test.txt");

        inventory.insert_resource(
            "file:///test.txt".to_string(),
            "server1".to_string(),
            resource.clone(),
        );

        assert!(inventory.has_resource("file:///test.txt"));

        let result = inventory.get_resource("file:///test.txt");
        assert!(result.is_some());

        let (server_name, retrieved_resource) = result.unwrap();
        assert_eq!(server_name, "server1");
        assert_eq!(retrieved_resource.uri, "file:///test.txt");
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
                inv.insert_tool(format!("tool_{}", i), format!("server_{}", i % 3), tool);
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Should have 10 tools
        let (tools, _, _) = inventory.counts();
        assert_eq!(tools, 10);
    }

    #[test]
    fn test_clear_all() {
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
        inventory.insert_prompt(
            "prompt1".to_string(),
            "server1".to_string(),
            create_test_prompt("prompt1"),
        );
        inventory.insert_resource(
            "res1".to_string(),
            "server1".to_string(),
            create_test_resource("res1"),
        );

        let (tools, prompts, resources) = inventory.counts();
        assert_eq!(tools, 1);
        assert_eq!(prompts, 1);
        assert_eq!(resources, 1);

        inventory.clear_all();

        let (tools, prompts, resources) = inventory.counts();
        assert_eq!(tools, 0);
        assert_eq!(prompts, 0);
        assert_eq!(resources, 0);
    }
}
