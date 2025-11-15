//! MCP tool, prompt, and resource inventory.
//!
//! Thread-safe cache for MCP capabilities across all connected servers.

use dashmap::DashMap;

use crate::mcp::config::{Prompt, RawResource, Tool};

/// Cached tool with metadata
#[derive(Clone)]
pub struct CachedTool {
    pub server_name: String,
    pub tool: Tool,
}

/// Cached prompt with metadata
#[derive(Clone)]
pub struct CachedPrompt {
    pub server_name: String,
    pub prompt: Prompt,
}

/// Cached resource with metadata
#[derive(Clone)]
pub struct CachedResource {
    pub server_name: String,
    pub resource: RawResource,
}

/// Tool inventory with periodic refresh
///
/// Provides thread-safe caching of MCP tools, prompts, and resources.
/// Entries are refreshed periodically by background tasks.
///
/// Only caches tools from STATIC MCP servers (defined in mcp.yaml config).
/// Dynamic MCP clients (from request server_url) should list tools directly
/// from the client instead of using the inventory.
pub struct ToolInventory {
    /// Static tools: Map of (server_name, tool_name) -> cached tool
    /// Only for MCP servers defined in config
    static_tools: DashMap<(String, String), CachedTool>,

    /// Map of prompt_name -> cached prompt
    prompts: DashMap<String, CachedPrompt>,

    /// Map of resource_uri -> cached resource
    resources: DashMap<String, CachedResource>,
}

impl ToolInventory {
    /// Create a new tool inventory
    pub fn new() -> Self {
        Self {
            static_tools: DashMap::new(),
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

    /// Insert a static tool (from config servers)
    pub fn insert_static_tool(&self, server_name: String, tool_name: String, tool: Tool) {
        self.static_tools.insert(
            (server_name.clone(), tool_name),
            CachedTool { server_name, tool },
        );
    }

    /// Get a tool from static inventory by server name and tool name (O(1))
    pub fn get_tool(&self, server_name: &str, tool_name: &str) -> Option<Tool> {
        self.static_tools
            .get(&(server_name.to_string(), tool_name.to_string()))
            .map(|cached| cached.tool.clone())
    }

    /// Find a tool by name across all servers (O(N))
    /// Returns (server_name, tool). If multiple servers have the same tool, returns first match.
    /// Prefer get_tool() if you know the server name.
    pub fn find_tool_by_name(&self, tool_name: &str) -> Option<(String, Tool)> {
        self.static_tools
            .iter()
            .find(|entry| entry.key().1 == tool_name)
            .map(|entry| (entry.value().server_name.clone(), entry.value().tool.clone()))
    }

    /// Check if tool exists in static inventory (searches all servers)
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.find_tool_by_name(tool_name).is_some()
    }

    /// List all static tools
    pub fn list_tools(&self) -> Vec<(String, String, Tool)> {
        let mut tools = Vec::new();

        for entry in self.static_tools.iter() {
            let ((server_name, tool_name), cached) = entry.pair();
            tools.push((tool_name.clone(), server_name.clone(), cached.tool.clone()));
        }

        tools
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
        self.static_tools.retain(|key, _| key.0 != server_name);
        self.prompts
            .retain(|_, cached| cached.server_name != server_name);
        self.resources
            .retain(|_, cached| cached.server_name != server_name);
    }

    /// Get count of cached items (static tools, prompts, resources)
    pub fn counts(&self) -> (usize, usize, usize) {
        (
            self.static_tools.len(),
            self.prompts.len(),
            self.resources.len(),
        )
    }

    /// Clear all cached items
    pub fn clear_all(&self) {
        self.static_tools.clear();
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
    fn test_static_tool_insert_and_get() {
        let inventory = ToolInventory::new();
        let tool = create_test_tool("test_tool");

        inventory.insert_static_tool("server1".to_string(), "test_tool".to_string(), tool.clone());

        // Test O(1) get_tool with server name
        let result = inventory.get_tool("server1", "test_tool");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "test_tool");

        // Test O(N) find_tool_by_name
        let result = inventory.find_tool_by_name("test_tool");
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

        inventory.insert_static_tool("server1".to_string(), "check_tool".to_string(), tool);

        assert!(inventory.has_tool("check_tool"));
    }

    #[test]
    fn test_list_tools() {
        let inventory = ToolInventory::new();

        inventory.insert_static_tool(
            "server1".to_string(),
            "tool1".to_string(),
            create_test_tool("tool1"),
        );
        inventory.insert_static_tool(
            "server1".to_string(),
            "tool2".to_string(),
            create_test_tool("tool2"),
        );

        // List all static tools
        let static_tools = inventory.list_tools();
        assert_eq!(static_tools.len(), 2);
    }

    #[test]
    fn test_clear_server_tools() {
        let inventory = ToolInventory::new();

        inventory.insert_static_tool(
            "server1".to_string(),
            "tool1".to_string(),
            create_test_tool("tool1"),
        );
        inventory.insert_static_tool(
            "server2".to_string(),
            "tool2".to_string(),
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
                inv.insert_static_tool(format!("server_{}", i % 3), format!("tool_{}", i), tool);
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

        inventory.insert_static_tool(
            "server1".to_string(),
            "tool1".to_string(),
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
