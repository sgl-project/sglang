// MCP Manager - Hybrid manager for static and dynamic MCP servers
//
// This module provides a unified interface for managing both:
// 1. Static servers (from config file, registered via workflow at startup)
// 2. Dynamic servers (per-request tools, managed via connection pool)
//
// Architecture:
// - Static servers are stored in a DashMap and never removed during runtime
// - Dynamic servers are managed by the connection pool with TTL-based cleanup
// - Shared ToolInventory provides TTL-based caching for all servers

use std::{sync::Arc, time::Duration};

use dashmap::DashMap;

use crate::mcp::{
    client_manager::McpClientManager,
    connection_pool::McpConnectionPool,
    error::McpResult,
    inventory::ToolInventory,
    McpTransport::{Sse, Stdio, Streamable},
};

/// Unified MCP manager handling both static and dynamic servers
///
/// This manager provides a single interface for accessing MCP servers regardless of
/// whether they're static (from config) or dynamic (from per-request tools).
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────┐
/// │             McpManager                          │
/// ├─────────────────────────────────────────────────┤
/// │                                                 │
/// │  ┌──────────────────┐  ┌──────────────────┐   │
/// │  │ Static Servers   │  │ Connection Pool  │   │
/// │  │ (DashMap)        │  │ (Dynamic)        │   │
/// │  │ - From config    │  │ - Per-request    │   │
/// │  │ - Permanent      │  │ - TTL cleanup    │   │
/// │  └──────────────────┘  └──────────────────┘   │
/// │                                                 │
/// │  ┌─────────────────────────────────────────┐   │
/// │  │  Shared Tool Inventory (TTL cache)      │   │
/// │  └─────────────────────────────────────────┘   │
/// │                                                 │
/// └─────────────────────────────────────────────────┘
/// ```
pub struct McpManager {
    /// Static servers registered at startup (from config file)
    /// Registered via workflow, never removed during runtime
    static_servers: Arc<DashMap<String, Arc<McpClientManager>>>,

    /// Dynamic servers from per-request tools
    /// Managed by connection pool with TTL and cleanup
    connection_pool: Arc<McpConnectionPool>,

    /// Tool inventory with TTL refresh
    /// Shared across both static and dynamic servers
    inventory: Arc<ToolInventory>,
}

impl std::fmt::Debug for McpManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpManager")
            .field("static_server_count", &self.static_servers.len())
            .field("connection_pool_size", &self.connection_pool.len())
            .field("inventory_counts", &self.inventory.counts())
            .finish()
    }
}

impl McpManager {
    /// Create a new MCP manager
    ///
    /// # Arguments
    /// * `tool_ttl` - TTL for cached tools (e.g., 300 seconds for 5 minutes)
    /// * `pool_idle_ttl` - TTL for idle connections in pool (e.g., 300 seconds)
    /// * `pool_max_connections` - Maximum number of pooled connections (e.g., 100)
    pub fn new(tool_ttl: Duration, pool_idle_ttl: Duration, pool_max_connections: usize) -> Self {
        Self {
            static_servers: Arc::new(DashMap::new()),
            connection_pool: Arc::new(McpConnectionPool::with_config(
                pool_idle_ttl,
                pool_max_connections,
            )),
            inventory: Arc::new(ToolInventory::new(tool_ttl)),
        }
    }

    /// Create a new MCP manager with default settings
    ///
    /// Default settings:
    /// - tool_ttl: 300 seconds (5 minutes)
    /// - pool_idle_ttl: 300 seconds (5 minutes)
    /// - pool_max_connections: 100
    pub fn with_defaults() -> Self {
        Self::new(Duration::from_secs(300), Duration::from_secs(300), 100)
    }

    /// Get MCP client for a static server by name
    ///
    /// This method only checks static servers (from config file).
    /// For dynamic servers (per-request tools), use `get_or_create_dynamic_client()`.
    ///
    /// # Arguments
    /// * `server_name` - The server name from config
    ///
    /// # Returns
    /// Arc to the MCP client manager if found, None otherwise
    pub async fn get_client(&self, server_name: &str) -> Option<Arc<McpClientManager>> {
        // Check if this is a static server (from config)
        self.static_servers
            .get(server_name)
            .map(|e| Arc::clone(e.value()))
    }

    /// Register a static server (called by workflow)
    ///
    /// This method registers a static MCP server that was configured in the config file
    /// and connected via the workflow system. Static servers are never removed during runtime.
    ///
    /// # Arguments
    /// * `name` - Unique server name (from config)
    /// * `client` - Connected MCP client manager
    pub fn register_static_server(&self, name: String, client: Arc<McpClientManager>) {
        self.static_servers.insert(name, client);
    }

    /// Get static server by name
    ///
    /// # Arguments
    /// * `name` - Server name from config
    ///
    /// # Returns
    /// Arc to the MCP client manager if found, None otherwise
    pub fn get_static_server(&self, name: &str) -> Option<Arc<McpClientManager>> {
        self.static_servers.get(name).map(|e| Arc::clone(e.value()))
    }

    /// List all static server names
    ///
    /// # Returns
    /// Vector of server names (from config)
    pub fn list_static_servers(&self) -> Vec<String> {
        self.static_servers
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    /// Get or create a dynamic MCP client using the connection pool
    ///
    /// This method is used for per-request dynamic servers (from tools field in API requests).
    /// It uses the connection pool to reuse connections and avoid 70-650ms overhead.
    ///
    /// # Arguments
    /// * `server_config` - Configuration for the dynamic MCP server
    ///
    /// # Returns
    /// Arc to the MCP client manager on success, error otherwise
    pub async fn get_or_create_dynamic_client(
        &self,
        server_config: crate::mcp::McpServerConfig,
    ) -> McpResult<Arc<McpClientManager>> {
        // Extract server URL from transport for cache key
        let server_url = match &server_config.transport {
            Streamable { url, .. } => url.clone(),
            Sse { url, .. } => url.clone(),
            Stdio { command, .. } => command.clone(),
        };

        // Use connection pool to get or create the client
        self.connection_pool
            .get_or_create(&server_url, server_config)
            .await
    }

    /// Get the shared tool inventory
    ///
    /// The inventory provides TTL-based caching for tools, prompts, and resources
    /// across both static and dynamic servers.
    ///
    /// # Returns
    /// Arc to the shared ToolInventory
    pub fn inventory(&self) -> Arc<ToolInventory> {
        Arc::clone(&self.inventory)
    }

    /// Get the connection pool
    ///
    /// Primarily for internal use and testing. External code should use `get_client()`.
    ///
    /// # Returns
    /// Arc to the connection pool
    pub fn connection_pool(&self) -> Arc<McpConnectionPool> {
        Arc::clone(&self.connection_pool)
    }

    /// Get statistics about the manager
    pub fn stats(&self) -> McpManagerStats {
        let (tools, prompts, resources) = self.inventory.counts();
        McpManagerStats {
            static_server_count: self.static_servers.len(),
            pool_stats: self.connection_pool.stats(),
            tool_count: tools,
            prompt_count: prompts,
            resource_count: resources,
        }
    }

    /// Refresh inventory for a static server
    ///
    /// This method refreshes the tool inventory for a specific static server by calling
    /// the underlying client's refresh method. The refreshed tools are stored in the
    /// shared inventory with TTL-based caching.
    ///
    /// # Arguments
    /// * `server_name` - The name of the static server to refresh
    ///
    /// # Returns
    /// Ok(()) on success, error if server not found or refresh fails
    pub async fn refresh_static_server_inventory(&self, server_name: &str) -> McpResult<()> {
        let client = self
            .get_static_server(server_name)
            .ok_or_else(|| crate::mcp::McpError::ServerNotFound(server_name.to_string()))?;

        client.refresh_server_inventory(server_name).await
    }

    /// Start background refresh for all static servers
    ///
    /// This method spawns a SINGLE background task that periodically refreshes the inventory
    /// for ALL registered static MCP servers. This is more efficient than spawning one task
    /// per server.
    ///
    /// # Arguments
    /// * `refresh_interval` - How often to refresh (e.g., Duration::from_secs(300) for 5 minutes)
    ///
    /// # Returns
    /// Join handle for the spawned background task
    pub fn spawn_background_refresh_all(
        self: Arc<Self>,
        refresh_interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(refresh_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                let server_names = self.list_static_servers();

                if !server_names.is_empty() {
                    tracing::debug!(
                        "Background refresh: Refreshing {} static server(s)",
                        server_names.len()
                    );

                    for server_name in server_names {
                        if let Err(e) = self.refresh_static_server_inventory(&server_name).await {
                            tracing::warn!(
                                "Background refresh failed for '{}': {}",
                                server_name,
                                e
                            );
                        }
                    }

                    tracing::debug!("Background refresh: Completed refresh cycle");
                }
            }
        })
    }
}

/// Statistics about the MCP manager
#[derive(Debug, Clone)]
pub struct McpManagerStats {
    /// Number of static servers registered
    pub static_server_count: usize,
    /// Connection pool statistics
    pub pool_stats: crate::mcp::connection_pool::PoolStats,
    /// Number of cached tools
    pub tool_count: usize,
    /// Number of cached prompts
    pub prompt_count: usize,
    /// Number of cached resources
    pub resource_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = McpManager::with_defaults();

        // Should start with no static servers
        assert_eq!(manager.list_static_servers().len(), 0);

        // Should have empty inventory
        let (tools, prompts, resources) = manager.inventory.counts();
        assert_eq!(tools, 0);
        assert_eq!(prompts, 0);
        assert_eq!(resources, 0);
    }

    #[test]
    fn test_manager_custom_ttl() {
        let manager = McpManager::new(
            Duration::from_secs(600), // 10 minutes tool TTL
            Duration::from_secs(900), // 15 minutes pool idle TTL
            200,                      // 200 max connections
        );

        assert_eq!(manager.list_static_servers().len(), 0);
    }

    #[test]
    fn test_stats() {
        let manager = McpManager::with_defaults();

        let stats = manager.stats();
        assert_eq!(stats.static_server_count, 0);
        assert_eq!(stats.tool_count, 0);
        assert_eq!(stats.prompt_count, 0);
        assert_eq!(stats.resource_count, 0);
    }

    #[tokio::test]
    async fn test_get_client_not_found() {
        let manager = McpManager::with_defaults();

        // Should return None for non-existent server
        let result = manager.get_client("nonexistent").await;
        assert!(result.is_none());
    }

    #[test]
    fn test_list_static_servers_empty() {
        let manager = McpManager::with_defaults();

        let servers = manager.list_static_servers();
        assert!(servers.is_empty());
    }

    #[test]
    fn test_get_static_server_not_found() {
        let manager = McpManager::with_defaults();

        let result = manager.get_static_server("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_inventory_access() {
        let manager = McpManager::with_defaults();

        // Should be able to access inventory
        let inventory = manager.inventory();
        assert_eq!(inventory.counts(), (0, 0, 0));
    }

    #[test]
    fn test_connection_pool_access() {
        let manager = McpManager::with_defaults();

        // Should be able to access connection pool
        let pool = manager.connection_pool();
        assert_eq!(pool.len(), 0);
    }
}
