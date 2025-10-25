// MCP Tool Inventory with TTL-based Caching
//
// This module provides TTL-based caching for MCP tools, prompts, and resources.
// Tools are cached with timestamps and automatically expire after the configured TTL.
// Background refresh tasks can proactively update the inventory.

use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::mcp::config::{PromptInfo, ResourceInfo, ToolInfo};

/// Cached tool with metadata
#[derive(Clone)]
pub struct CachedTool {
    pub server_name: String,
    pub tool: ToolInfo,
    pub cached_at: Instant,
}

/// Cached prompt with metadata
#[derive(Clone)]
pub struct CachedPrompt {
    pub server_name: String,
    pub prompt: PromptInfo,
    pub cached_at: Instant,
}

/// Cached resource with metadata
#[derive(Clone)]
pub struct CachedResource {
    pub server_name: String,
    pub resource: ResourceInfo,
    pub cached_at: Instant,
}

/// Tool inventory with TTL-based caching
///
/// Provides thread-safe caching of MCP tools, prompts, and resources with automatic expiration.
/// Entries are timestamped and can be queried with TTL validation.
pub struct ToolInventory {
    /// Map of tool_name -> cached tool
    tools: DashMap<String, CachedTool>,

    /// Map of prompt_name -> cached prompt
    prompts: DashMap<String, CachedPrompt>,

    /// Map of resource_uri -> cached resource
    resources: DashMap<String, CachedResource>,

    /// Tool cache TTL
    tool_ttl: Duration,

    /// Last refresh time per server
    server_refresh_times: DashMap<String, Instant>,
}

impl ToolInventory {
    /// Create a new tool inventory with the specified TTL
    pub fn new(tool_ttl: Duration) -> Self {
        Self {
            tools: DashMap::new(),
            prompts: DashMap::new(),
            resources: DashMap::new(),
            tool_ttl,
            server_refresh_times: DashMap::new(),
        }
    }

    // ============================================================================
    // Tool Methods
    // ============================================================================

    /// Get a tool if it exists and is fresh (within TTL)
    ///
    /// Returns None if the tool doesn't exist or has expired.
    pub fn get_tool(&self, tool_name: &str) -> Option<(String, ToolInfo)> {
        self.tools.get(tool_name).and_then(|entry| {
            let cached = entry.value();

            // Check if still fresh
            if cached.cached_at.elapsed() < self.tool_ttl {
                Some((cached.server_name.clone(), cached.tool.clone()))
            } else {
                // Expired - will be removed by cleanup
                None
            }
        })
    }

    /// Check if tool exists (regardless of TTL)
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tools.contains_key(tool_name)
    }

    /// Insert or update a tool
    pub fn insert_tool(&self, tool_name: String, server_name: String, tool: ToolInfo) {
        self.tools.insert(
            tool_name,
            CachedTool {
                server_name,
                tool,
                cached_at: Instant::now(),
            },
        );
    }

    /// Get all tools (fresh only)
    pub fn list_tools(&self) -> Vec<(String, String, ToolInfo)> {
        let now = Instant::now();
        self.tools
            .iter()
            .filter_map(|entry| {
                let (name, cached) = entry.pair();
                if now.duration_since(cached.cached_at) < self.tool_ttl {
                    Some((
                        name.clone(),
                        cached.server_name.clone(),
                        cached.tool.clone(),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    // ============================================================================
    // Prompt Methods
    // ============================================================================

    /// Get a prompt if it exists and is fresh (within TTL)
    pub fn get_prompt(&self, prompt_name: &str) -> Option<(String, PromptInfo)> {
        self.prompts.get(prompt_name).and_then(|entry| {
            let cached = entry.value();

            // Check if still fresh
            if cached.cached_at.elapsed() < self.tool_ttl {
                Some((cached.server_name.clone(), cached.prompt.clone()))
            } else {
                None
            }
        })
    }

    /// Check if prompt exists (regardless of TTL)
    pub fn has_prompt(&self, prompt_name: &str) -> bool {
        self.prompts.contains_key(prompt_name)
    }

    /// Insert or update a prompt
    pub fn insert_prompt(&self, prompt_name: String, server_name: String, prompt: PromptInfo) {
        self.prompts.insert(
            prompt_name,
            CachedPrompt {
                server_name,
                prompt,
                cached_at: Instant::now(),
            },
        );
    }

    /// Get all prompts (fresh only)
    pub fn list_prompts(&self) -> Vec<(String, String, PromptInfo)> {
        let now = Instant::now();
        self.prompts
            .iter()
            .filter_map(|entry| {
                let (name, cached) = entry.pair();
                if now.duration_since(cached.cached_at) < self.tool_ttl {
                    Some((
                        name.clone(),
                        cached.server_name.clone(),
                        cached.prompt.clone(),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    // ============================================================================
    // Resource Methods
    // ============================================================================

    /// Get a resource if it exists and is fresh (within TTL)
    pub fn get_resource(&self, resource_uri: &str) -> Option<(String, ResourceInfo)> {
        self.resources.get(resource_uri).and_then(|entry| {
            let cached = entry.value();

            // Check if still fresh
            if cached.cached_at.elapsed() < self.tool_ttl {
                Some((cached.server_name.clone(), cached.resource.clone()))
            } else {
                None
            }
        })
    }

    /// Check if resource exists (regardless of TTL)
    pub fn has_resource(&self, resource_uri: &str) -> bool {
        self.resources.contains_key(resource_uri)
    }

    /// Insert or update a resource
    pub fn insert_resource(
        &self,
        resource_uri: String,
        server_name: String,
        resource: ResourceInfo,
    ) {
        self.resources.insert(
            resource_uri,
            CachedResource {
                server_name,
                resource,
                cached_at: Instant::now(),
            },
        );
    }

    /// Get all resources (fresh only)
    pub fn list_resources(&self) -> Vec<(String, String, ResourceInfo)> {
        let now = Instant::now();
        self.resources
            .iter()
            .filter_map(|entry| {
                let (uri, cached) = entry.pair();
                if now.duration_since(cached.cached_at) < self.tool_ttl {
                    Some((
                        uri.clone(),
                        cached.server_name.clone(),
                        cached.resource.clone(),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    // ============================================================================
    // Server Management Methods
    // ============================================================================

    /// Clear all cached items for a specific server (before refresh)
    pub fn clear_server_tools(&self, server_name: &str) {
        self.tools
            .retain(|_, cached| cached.server_name != server_name);
        self.prompts
            .retain(|_, cached| cached.server_name != server_name);
        self.resources
            .retain(|_, cached| cached.server_name != server_name);
    }

    /// Mark server as refreshed
    pub fn mark_refreshed(&self, server_name: &str) {
        self.server_refresh_times
            .insert(server_name.to_string(), Instant::now());
    }

    /// Check if server needs refresh based on refresh interval
    pub fn needs_refresh(&self, server_name: &str, refresh_interval: Duration) -> bool {
        self.server_refresh_times
            .get(server_name)
            .map(|t| t.elapsed() > refresh_interval)
            .unwrap_or(true) // Never refreshed = needs refresh
    }

    /// Get last refresh time for a server
    pub fn last_refresh(&self, server_name: &str) -> Option<Instant> {
        self.server_refresh_times
            .get(server_name)
            .map(|t| *t.value())
    }

    // ============================================================================
    // Cleanup Methods
    // ============================================================================

    /// Cleanup expired entries
    ///
    /// Removes all tools, prompts, and resources that have exceeded their TTL.
    /// Should be called periodically by a background task.
    pub fn cleanup_expired(&self) {
        let now = Instant::now();

        // Remove expired tools
        self.tools
            .retain(|_, cached| now.duration_since(cached.cached_at) < self.tool_ttl);

        // Remove expired prompts
        self.prompts
            .retain(|_, cached| now.duration_since(cached.cached_at) < self.tool_ttl);

        // Remove expired resources
        self.resources
            .retain(|_, cached| now.duration_since(cached.cached_at) < self.tool_ttl);
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
        self.server_refresh_times.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a test tool
    fn create_test_tool(name: &str) -> ToolInfo {
        ToolInfo {
            name: name.to_string(),
            description: format!("Test tool: {}", name),
            server: "test_server".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {}
            })),
        }
    }

    // Helper to create a test prompt
    fn create_test_prompt(name: &str) -> PromptInfo {
        PromptInfo {
            name: name.to_string(),
            description: Some(format!("Test prompt: {}", name)),
            server: "test_server".to_string(),
            arguments: None,
        }
    }

    // Helper to create a test resource
    fn create_test_resource(uri: &str) -> ResourceInfo {
        ResourceInfo {
            uri: uri.to_string(),
            name: uri.to_string(),
            description: Some(format!("Test resource: {}", uri)),
            mime_type: Some("text/plain".to_string()),
            server: "test_server".to_string(),
        }
    }

    #[test]
    fn test_tool_insert_and_get() {
        let inventory = ToolInventory::new(Duration::from_secs(60));
        let tool = create_test_tool("test_tool");

        inventory.insert_tool("test_tool".to_string(), "server1".to_string(), tool.clone());

        let result = inventory.get_tool("test_tool");
        assert!(result.is_some());

        let (server_name, retrieved_tool) = result.unwrap();
        assert_eq!(server_name, "server1");
        assert_eq!(retrieved_tool.name, "test_tool");
    }

    #[test]
    fn test_tool_expiration() {
        let inventory = ToolInventory::new(Duration::from_millis(100));
        let tool = create_test_tool("expiring_tool");

        inventory.insert_tool(
            "expiring_tool".to_string(),
            "server1".to_string(),
            tool.clone(),
        );

        // Should be available immediately
        assert!(inventory.get_tool("expiring_tool").is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Should be expired now
        assert!(inventory.get_tool("expiring_tool").is_none());
    }

    #[test]
    fn test_has_tool() {
        let inventory = ToolInventory::new(Duration::from_secs(60));
        let tool = create_test_tool("check_tool");

        assert!(!inventory.has_tool("check_tool"));

        inventory.insert_tool("check_tool".to_string(), "server1".to_string(), tool);

        assert!(inventory.has_tool("check_tool"));
    }

    #[test]
    fn test_list_tools() {
        let inventory = ToolInventory::new(Duration::from_secs(60));

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
    fn test_list_tools_filters_expired() {
        let inventory = ToolInventory::new(Duration::from_millis(100));

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );

        // Should have 1 tool
        assert_eq!(inventory.list_tools().len(), 1);

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Should have 0 tools (filtered out)
        assert_eq!(inventory.list_tools().len(), 0);
    }

    #[test]
    fn test_clear_server_tools() {
        let inventory = ToolInventory::new(Duration::from_secs(60));

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
    fn test_server_refresh_tracking() {
        let inventory = ToolInventory::new(Duration::from_secs(60));

        // Never refreshed
        assert!(inventory.needs_refresh("server1", Duration::from_secs(10)));

        // Mark as refreshed
        inventory.mark_refreshed("server1");

        // Should not need refresh immediately
        assert!(!inventory.needs_refresh("server1", Duration::from_secs(10)));

        // Wait and check again
        std::thread::sleep(Duration::from_millis(100));
        assert!(inventory.needs_refresh("server1", Duration::from_millis(50)));
    }

    #[test]
    fn test_cleanup_expired() {
        let inventory = ToolInventory::new(Duration::from_millis(100));

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

        let (tools, _, _) = inventory.counts();
        assert_eq!(tools, 2);

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Cleanup expired entries
        inventory.cleanup_expired();

        let (tools, _, _) = inventory.counts();
        assert_eq!(tools, 0);
    }

    #[test]
    fn test_prompt_operations() {
        let inventory = ToolInventory::new(Duration::from_secs(60));
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
        let inventory = ToolInventory::new(Duration::from_secs(60));
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

        let inventory = Arc::new(ToolInventory::new(Duration::from_secs(60)));

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
        let inventory = ToolInventory::new(Duration::from_secs(60));

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

        inventory.mark_refreshed("server1");

        let (tools, prompts, resources) = inventory.counts();
        assert_eq!(tools, 1);
        assert_eq!(prompts, 1);
        assert_eq!(resources, 1);

        inventory.clear_all();

        let (tools, prompts, resources) = inventory.counts();
        assert_eq!(tools, 0);
        assert_eq!(prompts, 0);
        assert_eq!(resources, 0);
        assert!(inventory.last_refresh("server1").is_none());
    }
}
