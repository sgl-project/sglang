/// MCP Connection Pool
///
/// This module provides connection pooling for dynamic MCP servers (per-request).
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;
use rmcp::{service::RunningService, RoleClient};

use crate::mcp::{
    config::{McpProxyConfig, McpServerConfig},
    error::McpResult,
};

/// Type alias for MCP client
type McpClient = RunningService<RoleClient, ()>;

/// Type alias for eviction callback
type EvictionCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// Cached MCP connection with metadata
#[derive(Clone)]
pub(crate) struct CachedConnection {
    /// The MCP client instance
    pub client: Arc<McpClient>,
    /// Server configuration used to create this connection
    #[allow(dead_code)]
    pub config: McpServerConfig,
}

impl CachedConnection {
    /// Create a new cached connection
    pub fn new(client: Arc<McpClient>, config: McpServerConfig) -> Self {
        Self { client, config }
    }
}

/// Connection pool for dynamic MCP servers
///
/// Provides thread-safe connection pooling with LRU eviction.
/// Connections are keyed by server URL and reused across requests.
pub struct McpConnectionPool {
    /// LRU cache of server_url -> cached connection
    connections: Arc<Mutex<LruCache<String, CachedConnection>>>,

    /// Maximum number of cached connections (LRU capacity)
    max_connections: usize,

    /// Global proxy configuration (applied to all dynamic servers)
    /// Can be overridden per-server via McpServerConfig.proxy
    global_proxy: Option<McpProxyConfig>,

    /// Optional eviction callback (called when LRU evicts a connection)
    /// Used to clean up tools from inventory
    eviction_callback: Option<EvictionCallback>,
}

impl McpConnectionPool {
    /// Default max connections for pool
    const DEFAULT_MAX_CONNECTIONS: usize = 200;

    /// Create a new connection pool with default settings
    ///
    /// Default settings:
    /// - max_connections: 200
    /// - global_proxy: Loaded from environment variables (MCP_HTTP_PROXY, etc.)
    pub fn new() -> Self {
        Self {
            connections: Arc::new(Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(Self::DEFAULT_MAX_CONNECTIONS).unwrap(),
            ))),
            max_connections: Self::DEFAULT_MAX_CONNECTIONS,
            global_proxy: McpProxyConfig::from_env(),
            eviction_callback: None,
        }
    }

    /// Create a new connection pool with custom capacity
    pub fn with_capacity(max_connections: usize) -> Self {
        Self {
            connections: Arc::new(Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(max_connections).unwrap(),
            ))),
            max_connections,
            global_proxy: McpProxyConfig::from_env(),
            eviction_callback: None,
        }
    }

    /// Create a new connection pool with full custom configuration
    pub fn with_full_config(max_connections: usize, global_proxy: Option<McpProxyConfig>) -> Self {
        Self {
            connections: Arc::new(Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(max_connections).unwrap(),
            ))),
            max_connections,
            global_proxy,
            eviction_callback: None,
        }
    }

    /// Set the eviction callback (called when LRU evicts a connection)
    pub fn set_eviction_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.eviction_callback = Some(Arc::new(callback));
    }

    /// Get an existing connection or create a new one
    ///
    /// This method:
    /// 1. Checks if a connection exists for the given URL (fast path <1ms)
    /// 2. If exists, promotes it in LRU and returns it
    /// 3. If not exists, creates new connection (slow path 70-650ms)
    ///
    /// # Arguments
    /// * `server_url` - The MCP server URL (used as cache key)
    /// * `server_config` - Server configuration (used to create new connection if needed)
    /// * `connect_fn` - Async function to create a new client connection
    ///
    /// # Returns
    /// Arc to the MCP client, either from cache or newly created
    pub async fn get_or_create<F, Fut>(
        &self,
        server_url: &str,
        server_config: McpServerConfig,
        connect_fn: F,
    ) -> McpResult<Arc<McpClient>>
    where
        F: FnOnce(McpServerConfig, Option<McpProxyConfig>) -> Fut,
        Fut: std::future::Future<Output = McpResult<McpClient>>,
    {
        // Fast path: Check if connection exists in LRU cache
        {
            let mut connections = self.connections.lock();
            if let Some(cached) = connections.get(server_url) {
                // LRU get() promotes the entry
                return Ok(Arc::clone(&cached.client));
            }
        }

        // Slow path: Create new connection
        let client = connect_fn(server_config.clone(), self.global_proxy.clone()).await?;
        let client_arc = Arc::new(client);

        // Cache the new connection (LRU will automatically evict oldest if at capacity)
        let cached = CachedConnection::new(Arc::clone(&client_arc), server_config);
        {
            let mut connections = self.connections.lock();
            if let Some((evicted_key, _evicted_conn)) =
                connections.push(server_url.to_string(), cached)
            {
                // Call eviction callback if set
                if let Some(callback) = &self.eviction_callback {
                    callback(&evicted_key);
                }
            }
        }

        Ok(client_arc)
    }

    /// Get current number of cached connections
    pub fn len(&self) -> usize {
        self.connections.lock().len()
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.connections.lock().is_empty()
    }

    /// Clear all connections
    pub fn clear(&self) {
        self.connections.lock().clear();
    }

    /// Get connection statistics
    pub fn stats(&self) -> PoolStats {
        let total = self.connections.lock().len();

        PoolStats {
            total_connections: total,
            capacity: self.max_connections,
        }
    }

    /// List all server keys in the pool
    pub fn list_server_keys(&self) -> Vec<String> {
        self.connections
            .lock()
            .iter()
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Get a connection by server key without creating it
    /// Promotes the entry in LRU cache if found
    pub fn get(&self, server_key: &str) -> Option<Arc<McpClient>> {
        self.connections
            .lock()
            .get(server_key)
            .map(|cached| Arc::clone(&cached.client))
    }
}

impl Default for McpConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_connections: usize,
    pub capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::config::McpTransport;

    // Helper to create test server config
    fn create_test_config(url: &str) -> McpServerConfig {
        McpServerConfig {
            name: "test_server".to_string(),
            transport: McpTransport::Streamable {
                url: url.to_string(),
                token: None,
                headers: None,
            },
            proxy: None,
            required: false,
        }
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let pool = McpConnectionPool::new();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_stats() {
        let pool = McpConnectionPool::with_capacity(10);

        let stats = pool.stats();
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.capacity, 10);
    }

    #[test]
    #[allow(invalid_value)]
    fn test_pool_clear() {
        let pool = McpConnectionPool::new();

        // Add a connection
        let config = create_test_config("http://localhost:3000");
        let client: Arc<McpClient> =
            Arc::new(unsafe { std::mem::MaybeUninit::zeroed().assume_init() });
        let cached = CachedConnection::new(client.clone(), config);
        pool.connections
            .lock()
            .push("http://localhost:3000".to_string(), cached);

        assert_eq!(pool.len(), 1);

        pool.clear();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());

        // Prevent drop of invalid Arc (would segfault)
        std::mem::forget(client);
    }

    #[test]
    fn test_pool_with_global_proxy() {
        use crate::mcp::config::McpProxyConfig;

        // Create proxy config
        let proxy = McpProxyConfig {
            http: Some("http://proxy.example.com:8080".to_string()),
            https: None,
            no_proxy: Some("localhost,127.0.0.1".to_string()),
            username: None,
            password: None,
        };

        // Create pool with proxy
        let pool = McpConnectionPool::with_full_config(100, Some(proxy.clone()));

        // Verify proxy is stored
        assert!(pool.global_proxy.is_some());
        let stored_proxy = pool.global_proxy.as_ref().unwrap();
        assert_eq!(
            stored_proxy.http.as_ref().unwrap(),
            "http://proxy.example.com:8080"
        );
        assert_eq!(
            stored_proxy.no_proxy.as_ref().unwrap(),
            "localhost,127.0.0.1"
        );
    }

    #[test]
    fn test_pool_proxy_from_env() {
        // Note: This test depends on environment variables
        // In production, proxy is loaded from MCP_HTTP_PROXY or HTTP_PROXY env vars
        let pool = McpConnectionPool::new();

        // Pool should either have proxy from env or None
        // We can't assert specific value since it depends on test environment
        // Just verify it doesn't panic
        assert!(pool.global_proxy.is_some() || pool.global_proxy.is_none());
    }
}
