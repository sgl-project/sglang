//! MCP client management and orchestration.
//!
//! Manages both static MCP servers (from config) and dynamic MCP servers (from requests).
//! Static clients are never evicted; dynamic clients use LRU eviction via connection pool.

use std::{borrow::Cow, sync::Arc, time::Duration};

use backoff::ExponentialBackoffBuilder;
use dashmap::DashMap;
use rmcp::{
    model::{CallToolRequestParam, CallToolResult},
    service::RunningService,
    transport::{
        sse_client::SseClientConfig, streamable_http_client::StreamableHttpClientTransportConfig,
        ConfigureCommandExt, SseClientTransport, StreamableHttpClientTransport, TokioChildProcess,
    },
    RoleClient, ServiceExt,
};
use tracing::{debug, error, info, warn};

use crate::mcp::{
    config::{McpConfig, McpProxyConfig, McpServerConfig, McpTransport},
    connection_pool::McpConnectionPool,
    error::{McpError, McpResult},
    inventory::{format_tool_name, CachedTool, ToolInventory},
    tool_args::ToolArgs,
};

/// Type alias for MCP client
type McpClient = RunningService<RoleClient, ()>;

pub struct McpManager {
    static_clients: Arc<DashMap<String, Arc<McpClient>>>,
    inventory: Arc<ToolInventory>,
    connection_pool: Arc<McpConnectionPool>,
    _config: McpConfig,
}

impl McpManager {
    const MAX_DYNAMIC_CLIENTS: usize = 200;

    pub async fn new(config: McpConfig, pool_max_connections: usize) -> McpResult<Self> {
        let inventory = Arc::new(ToolInventory::new());

        let mut connection_pool =
            McpConnectionPool::with_full_config(pool_max_connections, config.proxy.clone());

        let inventory_clone = Arc::clone(&inventory);
        connection_pool.set_eviction_callback(move |server_key: &str| {
            debug!(
                "LRU evicted dynamic server '{}' - clearing tools from inventory",
                server_key
            );
            inventory_clone.clear_server_tools(server_key);
        });

        let connection_pool = Arc::new(connection_pool);

        // Create storage for static clients
        let static_clients = Arc::new(DashMap::new());

        // Get global proxy config for all servers
        let global_proxy = config.proxy.as_ref();

        // Connect to all static servers from config
        for server_config in &config.servers {
            let server_url = Self::server_key(server_config);
            match Self::connect_server(server_config, global_proxy).await {
                Ok(client) => {
                    let client_arc = Arc::new(client);
                    // Load inventory for this server (use URL as server_label for consistency)
                    Self::load_server_inventory(&inventory, &server_url, &client_arc).await;
                    static_clients.insert(server_url.clone(), client_arc);
                    info!(
                        "Connected to static server '{}' (URL: {})",
                        server_config.name, server_url
                    );
                }
                Err(e) => {
                    error!(
                        "Failed to connect to static server '{}' (URL: {}): {}",
                        server_config.name, server_url, e
                    );
                }
            }
        }

        if static_clients.is_empty() {
            warn!("No static MCP servers connected");
        }

        Ok(Self {
            static_clients,
            inventory,
            connection_pool,
            _config: config,
        })
    }

    pub async fn with_defaults(config: McpConfig) -> McpResult<Self> {
        Self::new(config, Self::MAX_DYNAMIC_CLIENTS).await
    }

    /// Get a client by server URL
    ///
    /// Checks both static clients (from config) and dynamic clients (from connection pool).
    /// Both use URL-based caching for consistency.
    pub async fn get_client(&self, server_url: &str) -> Option<Arc<McpClient>> {
        // Check static clients first (by URL)
        if let Some(client) = self.static_clients.get(server_url) {
            return Some(Arc::clone(client.value()));
        }
        // Fall back to connection pool (also by URL)
        self.connection_pool.get(server_url)
    }

    pub async fn get_or_create_client(
        &self,
        server_config: McpServerConfig,
    ) -> McpResult<Arc<McpClient>> {
        let server_key = Self::server_key(&server_config);

        // Check static clients by URL
        if let Some(client) = self.static_clients.get(&server_key) {
            return Ok(Arc::clone(client.value()));
        }

        // Not in static clients, use connection pool
        let client = self
            .connection_pool
            .get_or_create(
                &server_key,
                server_config,
                |config, global_proxy| async move {
                    Self::connect_server(&config, global_proxy.as_ref()).await
                },
            )
            .await?;
        Ok(client)
    }

    pub fn list_static_servers(&self) -> Vec<String> {
        self.static_clients
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    pub fn is_static_server(&self, server_url: &str) -> bool {
        self.static_clients.contains_key(server_url)
    }

    pub fn register_static_server(&self, server_url: String, client: Arc<McpClient>) {
        self.static_clients.insert(server_url.clone(), client);
        info!("Registered static MCP server at URL: {}", server_url);
    }

    /// List all available tools from all servers (static only)
    pub fn list_tools(&self) -> Vec<CachedTool> {
        self.inventory.list_tools()
    }

    /// Build a per-request tool inventory
    ///
    /// Creates a new ToolInventory that combines:
    /// - Static tools from config
    /// - Dynamic tools from request MCP servers
    ///
    /// The returned inventory exists only for the request lifetime.
    ///
    /// # Arguments
    /// * `dynamic_servers` - List of (server_label, server_url) pairs from request
    pub async fn build_request_inventory(
        &self,
        dynamic_servers: &[(String, String)],
    ) -> ToolInventory {
        let request_inventory = ToolInventory::new();

        // Merge static tools from config
        request_inventory.merge(&self.inventory);

        // Add dynamic tools from request servers
        for (server_label, server_url) in dynamic_servers {
            if let Some(client) = self.get_client(server_url).await {
                match client.peer().list_all_tools().await {
                    Ok(tools) => {
                        debug!(
                            "Discovered {} tools from dynamic server '{}' ({})",
                            tools.len(),
                            server_label,
                            server_url
                        );
                        for tool in tools {
                            let tool_name = tool.name.to_string();
                            request_inventory.insert_tool(
                                server_label.clone(),
                                tool_name,
                                tool,
                                server_url.clone(),
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to list tools from dynamic server '{}' ({}): {:?}",
                            server_label, server_url, e
                        );
                    }
                }
            } else {
                warn!(
                    "No client found for dynamic server '{}' at URL '{}'",
                    server_label, server_url
                );
            }
        }

        request_inventory
    }

    /// Call a tool by server_label and tool_name from an inventory
    ///
    /// This is the preferred method for calling tools with composite keys.
    /// Use this when you have a per-request inventory that includes both static and dynamic tools.
    ///
    /// # Arguments
    /// * `inventory` - ToolInventory to look up the tool (can be static or per-request)
    /// * `server_label` - Server label identifying which server owns the tool
    /// * `tool_name` - Name of the tool to call
    /// * `args` - Tool arguments (JSON string or Map)
    pub async fn call_tool_from_inventory(
        &self,
        inventory: &ToolInventory,
        server_label: &str,
        tool_name: &str,
        args: impl Into<ToolArgs>,
    ) -> McpResult<CallToolResult> {
        // Get tool info and server URL from inventory
        let formatted_name = format_tool_name(server_label, tool_name);
        let cached_tool = inventory
            .get_tool(&formatted_name)
            .ok_or_else(|| McpError::ToolNotFound(format!("{}::{}", server_label, tool_name)))?;

        let tool_info = cached_tool.tool;
        let server_url = cached_tool.server_url;

        // Convert args with type coercion based on schema
        let tool_schema = Some(serde_json::Value::Object((*tool_info.input_schema).clone()));
        let args_map = args
            .into()
            .into_map(tool_schema.as_ref())
            .map_err(McpError::InvalidArguments)?;

        // Get client for that server
        let client = self
            .get_client(&server_url)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_url.clone()))?;

        // Call the tool
        let request = CallToolRequestParam {
            name: Cow::Owned(tool_name.to_string()),
            arguments: args_map,
        };

        client
            .call_tool(request)
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to call tool: {}", e)))
    }

    /// Refresh inventory for a specific server by URL
    pub async fn refresh_server_inventory(&self, server_url: &str) -> McpResult<()> {
        let client = self
            .get_client(server_url)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_url.to_string()))?;

        info!("Refreshing inventory for server: {}", server_url);
        self.load_server_inventory_internal(server_url, &client)
            .await;
        Ok(())
    }

    /// Start background refresh for ALL servers (static + dynamic)
    /// Refreshes every 10-15 minutes to keep tool inventory up-to-date
    pub fn spawn_background_refresh_all(
        self: Arc<Self>,
        refresh_interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(refresh_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                // Get all static server keys
                // Note: Dynamic clients in the connection pool are refreshed on-demand
                // when they are accessed via get_or_create_client()
                let server_keys: Vec<String> = self
                    .static_clients
                    .iter()
                    .map(|e| e.key().clone())
                    .collect();

                if !server_keys.is_empty() {
                    debug!(
                        "Background refresh: Refreshing {} static server(s)",
                        server_keys.len()
                    );

                    for server_key in server_keys {
                        if let Err(e) = self.refresh_server_inventory(&server_key).await {
                            warn!("Background refresh failed for '{}': {}", server_key, e);
                        }
                    }

                    debug!("Background refresh: Completed refresh cycle");
                }
            }
        })
    }

    /// List all connected servers (static + dynamic)
    pub fn list_servers(&self) -> Vec<String> {
        let mut servers = Vec::new();

        // Add static servers
        servers.extend(self.static_clients.iter().map(|e| e.key().clone()));

        // Add dynamic servers from connection pool
        servers.extend(self.connection_pool.list_server_keys());

        servers
    }

    /// Disconnect from all servers (for cleanup)
    pub async fn shutdown(&self) {
        // Shutdown static servers
        let static_keys: Vec<String> = self
            .static_clients
            .iter()
            .map(|e| e.key().clone())
            .collect();
        for name in static_keys {
            if let Some((_key, client)) = self.static_clients.remove(&name) {
                // Try to unwrap Arc to call cancel
                match Arc::try_unwrap(client) {
                    Ok(client) => {
                        if let Err(e) = client.cancel().await {
                            warn!("Error disconnecting from static server '{}': {}", name, e);
                        }
                    }
                    Err(_) => {
                        warn!(
                            "Could not shutdown static server '{}': client still in use",
                            name
                        );
                    }
                }
            }
        }

        // Clear dynamic clients from connection pool
        // The pool will handle cleanup on drop
        self.connection_pool.clear();
    }

    /// Get statistics about the manager
    pub fn stats(&self) -> McpManagerStats {
        McpManagerStats {
            static_server_count: self.static_clients.len(),
            pool_stats: self.connection_pool.stats(),
            tool_count: self.inventory.count(),
        }
    }

    /// Get the shared tool inventory
    pub fn inventory(&self) -> Arc<ToolInventory> {
        Arc::clone(&self.inventory)
    }

    /// Get the connection pool
    pub fn connection_pool(&self) -> Arc<McpConnectionPool> {
        Arc::clone(&self.connection_pool)
    }

    // ========================================================================
    // Internal Helper Methods
    // ========================================================================

    /// Static helper for loading inventory (for new())
    /// Discover and cache tools for a connected static server
    ///
    /// This method is public to allow workflow-based inventory loading.
    /// For static servers, server_label = server_url (URL from transport config).
    pub async fn load_server_inventory(
        inventory: &Arc<ToolInventory>,
        server_url: &str,
        client: &Arc<McpClient>,
    ) {
        // List tools from the server
        match client.peer().list_all_tools().await {
            Ok(tools) => {
                info!("Discovered {} tools from '{}'", tools.len(), server_url);
                for tool in tools {
                    let tool_name = tool.name.to_string();
                    // For static servers: server_label = server_url, server_url = server_url
                    inventory.insert_tool(
                        server_url.to_string(),
                        tool_name,
                        tool,
                        server_url.to_string(),
                    );
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_url, e),
        }
    }

    /// Discover and cache tools for a connected server (internal wrapper)
    async fn load_server_inventory_internal(&self, server_url: &str, client: &McpClient) {
        // List tools from the server
        match client.peer().list_all_tools().await {
            Ok(tools) => {
                info!("Discovered {} tools from '{}'", tools.len(), server_url);
                for tool in tools {
                    let tool_name = tool.name.to_string();
                    // For static servers: server_label = server_url, server_url = server_url
                    self.inventory.insert_tool(
                        server_url.to_string(),
                        tool_name,
                        tool,
                        server_url.to_string(),
                    );
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_url, e),
        }
    }

    // ========================================================================
    // Connection Logic (from client_manager.rs)
    // ========================================================================

    /// Connect to an MCP server
    ///
    /// This method is public to allow workflow-based server registration at runtime.
    /// It handles connection with automatic retry for network-based transports (SSE/Streamable).
    pub async fn connect_server(
        config: &McpServerConfig,
        global_proxy: Option<&McpProxyConfig>,
    ) -> McpResult<McpClient> {
        let needs_retry = matches!(
            &config.transport,
            McpTransport::Sse { .. } | McpTransport::Streamable { .. }
        );
        if needs_retry {
            Self::connect_server_with_retry(config, global_proxy).await
        } else {
            Self::connect_server_impl(config, global_proxy).await
        }
    }

    /// Connect with exponential backoff retry for remote servers
    async fn connect_server_with_retry(
        config: &McpServerConfig,
        global_proxy: Option<&McpProxyConfig>,
    ) -> McpResult<McpClient> {
        let backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_secs(1))
            .with_max_interval(Duration::from_secs(30))
            .with_max_elapsed_time(Some(Duration::from_secs(30)))
            .build();

        backoff::future::retry(backoff, || async {
            match Self::connect_server_impl(config, global_proxy).await {
                Ok(client) => Ok(client),
                Err(e) => {
                    if Self::is_permanent_error(&e) {
                        error!(
                            "Permanent error connecting to '{}': {} - not retrying",
                            config.name, e
                        );
                        Err(backoff::Error::permanent(e))
                    } else {
                        warn!("Failed to connect to '{}', retrying: {}", config.name, e);
                        Err(backoff::Error::transient(e))
                    }
                }
            }
        })
        .await
    }

    /// Determine if an error is permanent (should not retry) or transient
    fn is_permanent_error(error: &McpError) -> bool {
        match error {
            McpError::Config(_) => true,
            McpError::Auth(_) => true,
            McpError::ServerNotFound(_) => true,
            McpError::Transport(_) => true,
            McpError::ConnectionFailed(msg) => {
                msg.contains("initialize")
                    || msg.contains("connection closed")
                    || msg.contains("connection refused")
                    || msg.contains("invalid URL")
                    || msg.contains("not found")
            }
            _ => false,
        }
    }

    /// Internal implementation of server connection (stdio/sse/streamable)
    async fn connect_server_impl(
        config: &McpServerConfig,
        global_proxy: Option<&McpProxyConfig>,
    ) -> McpResult<McpClient> {
        info!(
            "Connecting to MCP server '{}' via {:?}",
            config.name, config.transport
        );

        match &config.transport {
            McpTransport::Stdio {
                command,
                args,
                envs,
            } => {
                let transport = TokioChildProcess::new(
                    tokio::process::Command::new(command).configure(|cmd| {
                        cmd.args(args)
                            .envs(envs.iter())
                            .stderr(std::process::Stdio::inherit());
                    }),
                )
                .map_err(|e| McpError::Transport(format!("create stdio transport: {}", e)))?;

                let client = ().serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize stdio client: {}", e))
                })?;

                info!("Connected to stdio server '{}'", config.name);
                Ok(client)
            }

            McpTransport::Sse { url, token } => {
                // Resolve proxy configuration
                let proxy_config = crate::mcp::proxy::resolve_proxy_config(config, global_proxy);

                // Create HTTP client with proxy support
                let client = if token.is_some() {
                    let mut builder = reqwest::Client::builder()
                        .timeout(Duration::from_secs(30))
                        .connect_timeout(Duration::from_secs(10));

                    // Apply proxy configuration using proxy.rs helper
                    if let Some(proxy_cfg) = proxy_config {
                        builder = crate::mcp::proxy::apply_proxy_to_builder(builder, proxy_cfg)?;
                    }

                    // Add Authorization header
                    builder = builder.default_headers({
                        let mut headers = reqwest::header::HeaderMap::new();
                        headers.insert(
                            reqwest::header::AUTHORIZATION,
                            format!("Bearer {}", token.as_ref().unwrap())
                                .parse()
                                .map_err(|e| McpError::Transport(format!("auth token: {}", e)))?,
                        );
                        headers
                    });

                    builder
                        .build()
                        .map_err(|e| McpError::Transport(format!("build HTTP client: {}", e)))?
                } else {
                    crate::mcp::proxy::create_http_client(proxy_config)?
                };

                let cfg = SseClientConfig {
                    sse_endpoint: url.clone().into(),
                    ..Default::default()
                };

                let transport = SseClientTransport::start_with_client(client, cfg)
                    .await
                    .map_err(|e| McpError::Transport(format!("create SSE transport: {}", e)))?;

                let client = ().serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize SSE client: {}", e))
                })?;

                info!("Connected to SSE server '{}' at {}", config.name, url);
                Ok(client)
            }

            McpTransport::Streamable { url, token } => {
                // Note: Streamable transport doesn't support proxy yet
                let _proxy_config = crate::mcp::proxy::resolve_proxy_config(config, global_proxy);
                if _proxy_config.is_some() {
                    warn!(
                        "Proxy configuration detected but not supported for Streamable transport on server '{}'",
                        config.name
                    );
                }

                let transport = if let Some(tok) = token {
                    let mut cfg = StreamableHttpClientTransportConfig::with_uri(url.as_str());
                    cfg.auth_header = Some(format!("Bearer {}", tok));
                    StreamableHttpClientTransport::from_config(cfg)
                } else {
                    StreamableHttpClientTransport::from_uri(url.as_str())
                };

                let client = ().serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize streamable client: {}", e))
                })?;

                info!(
                    "Connected to streamable HTTP server '{}' at {}",
                    config.name, url
                );
                Ok(client)
            }
        }
    }

    /// Generate a unique key for a server config
    ///
    /// This extracts the URL from the transport configuration to use as a unique identifier.
    /// For stdio transports, the command path is used as the key.
    /// This method is public to allow workflows to compute server URLs from configs.
    pub fn server_key(config: &McpServerConfig) -> String {
        // Extract URL from transport or use name
        match &config.transport {
            McpTransport::Streamable { url, .. } => url.clone(),
            McpTransport::Sse { url, .. } => url.clone(),
            McpTransport::Stdio { command, .. } => command.clone(),
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_manager_creation() {
        let config = McpConfig {
            servers: vec![],
            pool: Default::default(),
            proxy: None,
            warmup: vec![],
            inventory: Default::default(),
        };

        let manager = McpManager::new(config, 100).await.unwrap();
        assert_eq!(manager.list_static_servers().len(), 0);
    }
}
