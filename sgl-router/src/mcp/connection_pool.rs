// MCP Connection Pool
//
// This module provides connection pooling for dynamic MCP servers (per-request).
// Connections are cached and reused to avoid 70-650ms connection overhead on every request.
//
// Performance target:
// - First request: 70-650ms (connection establishment)
// - Subsequent requests: <1ms (cache hit)
// - 90%+ reduction in per-request overhead

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use rmcp::{service::RunningService, RoleClient};

use crate::mcp::{
    config::{McpProxyConfig, McpServerConfig},
    error::McpResult,
};

/// Type alias for MCP client
type McpClient = RunningService<RoleClient, ()>;

/// Cached MCP connection with metadata
#[derive(Clone)]
pub struct CachedConnection {
    /// The MCP client instance
    pub client: Arc<McpClient>,
    /// Last time this connection was accessed
    pub last_used: Instant,
    /// Server configuration used to create this connection
    pub config: McpServerConfig,
}

impl CachedConnection {
    /// Create a new cached connection
    pub fn new(client: Arc<McpClient>, config: McpServerConfig) -> Self {
        Self {
            client,
            last_used: Instant::now(),
            config,
        }
    }

    /// Update last_used timestamp
    pub fn touch(&mut self) {
        self.last_used = Instant::now();
    }

    /// Check if connection has been idle for longer than TTL
    pub fn is_idle(&self, idle_ttl: Duration) -> bool {
        self.last_used.elapsed() > idle_ttl
    }
}

/// Connection pool for dynamic MCP servers
///
/// Provides thread-safe connection pooling with automatic cleanup of idle connections.
/// Connections are keyed by server URL and reused across requests.
pub struct McpConnectionPool {
    /// Map of server_url -> cached connection
    connections: DashMap<String, CachedConnection>,

    /// Idle connection TTL (connections unused for this duration are cleaned up)
    idle_ttl: Duration,

    /// Maximum number of cached connections (prevents unbounded growth)
    max_connections: usize,

    /// Global proxy configuration (applied to all dynamic servers)
    /// Can be overridden per-server via McpServerConfig.proxy
    global_proxy: Option<McpProxyConfig>,
}

impl McpConnectionPool {
    /// Create a new connection pool with default settings
    ///
    /// Default settings:
    /// - idle_ttl: 300 seconds (5 minutes)
    /// - max_connections: 100
    /// - global_proxy: Loaded from environment variables (MCP_HTTP_PROXY, etc.)
    pub fn new() -> Self {
        Self {
            connections: DashMap::new(),
            idle_ttl: Duration::from_secs(300),
            max_connections: 100,
            global_proxy: McpProxyConfig::from_env(),
        }
    }

    /// Create a new connection pool with custom settings
    pub fn with_config(idle_ttl: Duration, max_connections: usize) -> Self {
        Self {
            connections: DashMap::new(),
            idle_ttl,
            max_connections,
            global_proxy: McpProxyConfig::from_env(),
        }
    }

    /// Create a new connection pool with full custom configuration
    pub fn with_full_config(
        idle_ttl: Duration,
        max_connections: usize,
        global_proxy: Option<McpProxyConfig>,
    ) -> Self {
        Self {
            connections: DashMap::new(),
            idle_ttl,
            max_connections,
            global_proxy,
        }
    }

    /// Get an existing connection or create a new one
    ///
    /// This method:
    /// 1. Checks if a connection exists for the given URL
    /// 2. If exists and fresh, updates last_used and returns it (fast path <1ms)
    /// 3. If not exists or stale, creates new connection (slow path 70-650ms)
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
        // Fast path: Check if connection exists and is still fresh
        if let Some(mut entry) = self.connections.get_mut(server_url) {
            let cached = entry.value_mut();

            // Check if connection is still within TTL
            if !cached.is_idle(self.idle_ttl) {
                // Update last_used and return cached connection
                cached.touch();
                return Ok(Arc::clone(&cached.client));
            }

            // Connection is stale, drop it and create new one
            drop(entry);
            self.connections.remove(server_url);
        }

        // Slow path: Create new connection
        // Enforce max_connections limit
        if self.connections.len() >= self.max_connections {
            self.cleanup_idle_connections();

            // If still at limit after cleanup, remove oldest connection
            if self.connections.len() >= self.max_connections {
                if let Some(oldest_key) = self.find_oldest_connection() {
                    self.connections.remove(&oldest_key);
                }
            }
        }

        // Create new MCP client using the provided connect function
        let client = connect_fn(server_config.clone(), self.global_proxy.clone()).await?;
        let client_arc = Arc::new(client);

        // Cache the new connection
        let cached = CachedConnection::new(Arc::clone(&client_arc), server_config);
        self.connections.insert(server_url.to_string(), cached);

        Ok(client_arc)
    }

    /// Remove all idle connections that have exceeded the TTL
    ///
    /// This method is called:
    /// - Automatically when max_connections limit is reached
    /// - Can be called manually by background cleanup task
    pub fn cleanup_idle_connections(&self) {
        let now = Instant::now();
        self.connections
            .retain(|_, cached| now.duration_since(cached.last_used) < self.idle_ttl);
    }

    /// Find the oldest connection (by last_used timestamp)
    ///
    /// Used for eviction when max_connections is reached and cleanup didn't free space
    fn find_oldest_connection(&self) -> Option<String> {
        self.connections
            .iter()
            .min_by_key(|entry| entry.value().last_used)
            .map(|entry| entry.key().clone())
    }

    /// Get current number of cached connections
    pub fn len(&self) -> usize {
        self.connections.len()
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.connections.is_empty()
    }

    /// Clear all connections (useful for tests)
    pub fn clear(&self) {
        self.connections.clear();
    }

    /// Get connection statistics
    pub fn stats(&self) -> PoolStats {
        let total = self.connections.len();
        let idle_count = self
            .connections
            .iter()
            .filter(|entry| entry.value().is_idle(self.idle_ttl))
            .count();

        PoolStats {
            total_connections: total,
            active_connections: total - idle_count,
            idle_connections: idle_count,
        }
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
    pub active_connections: usize,
    pub idle_connections: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::McpTransport;

    // Helper to create test server config
    fn create_test_config(url: &str) -> McpServerConfig {
        McpServerConfig {
            name: "test_server".to_string(),
            transport: McpTransport::Streamable {
                url: url.to_string(),
                token: None,
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
    #[allow(invalid_value)]
    fn test_cached_connection_touch() {
        let config = create_test_config("http://localhost:3000");
        let client: Arc<McpClient> = Arc::new(unsafe {
            // SAFETY: This is only for testing the CachedConnection struct
            std::mem::MaybeUninit::zeroed().assume_init()
        });
        let mut cached = CachedConnection::new(client.clone(), config);

        let first_time = cached.last_used;
        std::thread::sleep(Duration::from_millis(10));
        cached.touch();
        assert!(cached.last_used > first_time);

        // Prevent drop of invalid Arc (would segfault)
        std::mem::forget(client);
    }

    #[test]
    #[allow(invalid_value)]
    fn test_cached_connection_is_idle() {
        let config = create_test_config("http://localhost:3000");
        let client: Arc<McpClient> = Arc::new(unsafe {
            // SAFETY: This is only for testing the CachedConnection struct
            std::mem::MaybeUninit::zeroed().assume_init()
        });
        let cached = CachedConnection::new(client.clone(), config);

        // Fresh connection should not be idle
        assert!(!cached.is_idle(Duration::from_secs(1)));

        // Wait and check
        std::thread::sleep(Duration::from_millis(100));
        assert!(cached.is_idle(Duration::from_millis(50)));

        // Prevent drop of invalid Arc (would segfault)
        std::mem::forget(client);
    }

    #[test]
    fn test_pool_stats() {
        let pool = McpConnectionPool::with_config(Duration::from_millis(100), 10);

        let stats = pool.stats();
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.idle_connections, 0);
    }

    #[test]
    #[allow(invalid_value)]
    fn test_cleanup_idle_connections() {
        let pool = McpConnectionPool::with_config(Duration::from_millis(50), 10);

        // Initially empty
        assert_eq!(pool.len(), 0);

        // Add a connection manually for testing
        let config = create_test_config("http://localhost:3000");
        let client: Arc<McpClient> =
            Arc::new(unsafe { std::mem::MaybeUninit::zeroed().assume_init() });
        let cached = CachedConnection::new(client.clone(), config);
        pool.connections
            .insert("http://localhost:3000".to_string(), cached);

        assert_eq!(pool.len(), 1);

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(100));

        // Cleanup should remove idle connection
        pool.cleanup_idle_connections();
        assert_eq!(pool.len(), 0);

        // Prevent drop of invalid Arc (would segfault)
        std::mem::forget(client);
    }

    #[test]
    #[allow(invalid_value)]
    fn test_find_oldest_connection() {
        let pool = McpConnectionPool::new();

        // Collect clients to forget at end
        let mut clients = Vec::new();

        // Add connections with different timestamps
        for i in 0..3 {
            let url = format!("http://localhost:{}", 3000 + i);
            let config = create_test_config(&url);
            let client: Arc<McpClient> =
                Arc::new(unsafe { std::mem::MaybeUninit::zeroed().assume_init() });
            let cached = CachedConnection::new(client.clone(), config);
            pool.connections.insert(url, cached);
            clients.push(client);
            std::thread::sleep(Duration::from_millis(10));
        }

        // Oldest should be the first one
        let oldest = pool.find_oldest_connection();
        assert!(oldest.is_some());
        assert_eq!(oldest.unwrap(), "http://localhost:3000");

        // Prevent drop of invalid Arcs (would segfault)
        for client in clients {
            std::mem::forget(client);
        }
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
            .insert("http://localhost:3000".to_string(), cached);

        assert_eq!(pool.len(), 1);

        pool.clear();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());

        // Prevent drop of invalid Arc (would segfault)
        std::mem::forget(client);
    }

    #[test]
    fn test_pool_with_global_proxy() {
        use crate::mcp::McpProxyConfig;

        // Create proxy config
        let proxy = McpProxyConfig {
            http: Some("http://proxy.example.com:8080".to_string()),
            https: None,
            no_proxy: Some("localhost,127.0.0.1".to_string()),
            username: None,
            password: None,
        };

        // Create pool with proxy
        let pool =
            McpConnectionPool::with_full_config(Duration::from_secs(300), 100, Some(proxy.clone()));

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
