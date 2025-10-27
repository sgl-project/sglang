//! Refactored MCP Manager - Single flat structure for all MCP operations
//!
//! This replaces the previous hierarchy:
//! - McpManager (wrapper for static/dynamic distinction)
//! - McpClientManager (manages multiple clients)
//! - McpClient (actual client)
//!
//! New flat structure:
//! - McpManager (single component handling all MCP concerns)
//! - McpClient (actual client to one server)

use std::{borrow::Cow, sync::Arc, time::Duration};

use backoff::ExponentialBackoffBuilder;
use dashmap::DashMap;
use rmcp::{
    model::{
        CallToolRequestParam, CallToolResult, GetPromptRequestParam, GetPromptResult,
        ReadResourceRequestParam, ReadResourceResult, SubscribeRequestParam,
        UnsubscribeRequestParam,
    },
    service::RunningService,
    transport::{
        sse_client::SseClientConfig, streamable_http_client::StreamableHttpClientTransportConfig,
        ConfigureCommandExt, SseClientTransport, StreamableHttpClientTransport, TokioChildProcess,
    },
    RoleClient, ServiceExt,
};
use serde_json::Map;
use tracing::{debug, error, info, warn};

use crate::mcp::{
    config::{
        McpConfig, McpProxyConfig, McpServerConfig, McpTransport, PromptInfo, ResourceInfo,
        ToolInfo,
    },
    connection_pool::McpConnectionPool,
    error::{McpError, McpResult},
    inventory::ToolInventory,
    tool_args::ToolArgs,
};

/// Type alias for MCP client
type McpClient = RunningService<RoleClient, ()>;

/// Unified MCP Manager - handles all MCP operations
///
/// This single component manages:
/// - Client connections (both static and dynamic)
/// - Tool inventory and caching
/// - Connection pooling
/// - Background refresh
/// - Tool/prompt/resource operations
pub struct McpManager {
    /// All MCP clients (static + dynamic)
    /// Key: server_name for static, server_url for dynamic
    /// Using DashMap for concurrent access
    clients: Arc<DashMap<String, Arc<McpClient>>>,

    /// Track which servers are static (from config)
    /// Using DashMap for thread-safe mutation during workflow registration
    static_servers: Arc<DashMap<String, ()>>,

    /// Shared tool inventory with TTL and caching
    inventory: Arc<ToolInventory>,

    /// Connection pool for dynamic servers (TTL-based cleanup)
    connection_pool: Arc<McpConnectionPool>,

    /// Original config for static servers (kept for potential future use)
    _config: McpConfig,
}

impl McpManager {
    /// Create a new MCP manager with custom TTLs
    pub async fn new(
        config: McpConfig,
        tool_ttl: Duration,
        pool_idle_ttl: Duration,
        pool_max_connections: usize,
    ) -> McpResult<Self> {
        // Create shared inventory
        let inventory = Arc::new(ToolInventory::new(tool_ttl));

        // Create connection pool
        let connection_pool = Arc::new(McpConnectionPool::with_config(
            pool_idle_ttl,
            pool_max_connections,
        ));

        // Create manager structure
        let clients = Arc::new(DashMap::new());
        let static_servers = Arc::new(DashMap::new());

        // Get global proxy config for all servers
        let global_proxy = config.proxy.as_ref();

        // Connect to all static servers from config
        for server_config in &config.servers {
            static_servers.insert(server_config.name.clone(), ());

            match Self::connect_server(server_config, global_proxy).await {
                Ok(client) => {
                    let client_arc = Arc::new(client);
                    // Load inventory for this server
                    Self::load_server_inventory(&inventory, &server_config.name, &client_arc).await;
                    clients.insert(server_config.name.clone(), client_arc);
                    info!("Connected to static server '{}'", server_config.name);
                }
                Err(e) => {
                    error!(
                        "Failed to connect to static server '{}': {}",
                        server_config.name, e
                    );
                }
            }
        }

        if static_servers.is_empty() || clients.is_empty() {
            warn!("No static MCP servers connected");
        }

        Ok(Self {
            clients,
            static_servers,
            inventory,
            connection_pool,
            _config: config,
        })
    }

    /// Create with default settings (300s TTL, 300s idle, 100 max connections)
    pub async fn with_defaults(config: McpConfig) -> McpResult<Self> {
        Self::new(
            config,
            Duration::from_secs(300),
            Duration::from_secs(300),
            100,
        )
        .await
    }

    // ========================================================================
    // Client Management
    // ========================================================================

    /// Get a client by server name (static or dynamic)
    pub async fn get_client(&self, server_name: &str) -> Option<Arc<McpClient>> {
        self.clients.get(server_name).map(|e| Arc::clone(e.value()))
    }

    /// Get or create a dynamic client from server config
    pub async fn get_or_create_client(
        &self,
        server_config: McpServerConfig,
    ) -> McpResult<Arc<McpClient>> {
        // Check if client already exists
        let server_key = Self::server_key(&server_config);

        if let Some(client) = self.clients.get(&server_key) {
            return Ok(Arc::clone(client.value()));
        }

        // Client doesn't exist, create new one via connection pool
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

        // Store in clients map
        self.clients.insert(server_key, Arc::clone(&client));

        Ok(client)
    }

    /// List all static server names
    pub fn list_static_servers(&self) -> Vec<String> {
        self.static_servers
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    /// Check if a server is static
    pub fn is_static_server(&self, server_name: &str) -> bool {
        self.static_servers.contains_key(server_name)
    }

    /// Register a static server (called by workflow system)
    ///
    /// This method registers a static MCP server that was configured and connected
    /// via the workflow system. Static servers are never removed during runtime.
    ///
    /// # Arguments
    /// * `name` - Unique server name (from config)
    /// * `client` - Connected MCP client
    pub fn register_static_server(&self, name: String, client: Arc<McpClient>) {
        // Insert into clients map
        self.clients.insert(name.clone(), client);

        // Mark as static server (for background refresh and stats)
        self.static_servers.insert(name.clone(), ());

        info!("Registered static MCP server: {}", name);
    }

    // ========================================================================
    // Tool Operations (delegate to clients via inventory)
    // ========================================================================

    /// List all available tools from all servers
    pub fn list_tools(&self) -> Vec<ToolInfo> {
        self.inventory
            .list_tools()
            .into_iter()
            .map(|(_tool_name, _server_name, tool_info)| tool_info)
            .collect()
    }

    /// Call a tool by name with automatic type coercion
    ///
    /// Accepts either JSON string or parsed Map as arguments.
    /// Automatically converts string numbers to actual numbers based on tool schema.
    pub async fn call_tool(
        &self,
        tool_name: &str,
        args: impl Into<ToolArgs>,
    ) -> McpResult<CallToolResult> {
        // Get tool info for schema and server
        let (server_name, tool_info) = self
            .inventory
            .get_tool(tool_name)
            .ok_or_else(|| McpError::ToolNotFound(tool_name.to_string()))?;

        // Convert args with type coercion based on schema
        let tool_schema = tool_info.parameters.as_ref();
        let args_map = args
            .into()
            .into_map(tool_schema)
            .map_err(McpError::InvalidArguments)?;

        // Get client for that server
        let client = self
            .get_client(&server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

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

    /// Get a tool by name
    pub fn get_tool(&self, tool_name: &str) -> Option<ToolInfo> {
        self.inventory
            .get_tool(tool_name)
            .map(|(_server_name, tool_info)| tool_info)
    }

    // ========================================================================
    // Prompt Operations
    // ========================================================================

    /// Get a prompt by name
    pub async fn get_prompt(
        &self,
        prompt_name: &str,
        args: Option<Map<String, serde_json::Value>>,
    ) -> McpResult<GetPromptResult> {
        // Get server that owns this prompt
        let (server_name, _prompt_info) = self
            .inventory
            .get_prompt(prompt_name)
            .ok_or_else(|| McpError::PromptNotFound(prompt_name.to_string()))?;

        // Get client for that server
        let client = self
            .get_client(&server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

        // Get the prompt
        let request = GetPromptRequestParam {
            name: prompt_name.to_string(),
            arguments: args,
        };

        client
            .get_prompt(request)
            .await
            .map_err(|e| McpError::Transport(format!("Failed to get prompt: {}", e)))
    }

    /// List all available prompts
    pub fn list_prompts(&self) -> Vec<PromptInfo> {
        self.inventory
            .list_prompts()
            .into_iter()
            .map(|(_prompt_name, _server_name, prompt_info)| prompt_info)
            .collect()
    }

    // ========================================================================
    // Resource Operations
    // ========================================================================

    /// Read a resource by URI
    pub async fn read_resource(&self, uri: &str) -> McpResult<ReadResourceResult> {
        // Get server that owns this resource
        let (server_name, _resource_info) = self
            .inventory
            .get_resource(uri)
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))?;

        // Get client for that server
        let client = self
            .get_client(&server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

        // Read the resource
        let request = ReadResourceRequestParam {
            uri: uri.to_string(),
        };

        client
            .read_resource(request)
            .await
            .map_err(|e| McpError::Transport(format!("Failed to read resource: {}", e)))
    }

    /// List all available resources
    pub fn list_resources(&self) -> Vec<ResourceInfo> {
        self.inventory
            .list_resources()
            .into_iter()
            .map(|(_resource_uri, _server_name, resource_info)| resource_info)
            .collect()
    }

    // ========================================================================
    // Inventory Management
    // ========================================================================

    /// Refresh inventory for a specific server
    pub async fn refresh_server_inventory(&self, server_name: &str) -> McpResult<()> {
        let client = self
            .get_client(server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        info!("Refreshing inventory for server: {}", server_name);
        self.load_server_inventory_internal(server_name, &client)
            .await;
        Ok(())
    }

    /// Start background refresh for all static servers
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
                    debug!(
                        "Background refresh: Refreshing {} static server(s)",
                        server_names.len()
                    );

                    for server_name in server_names {
                        if let Err(e) = self.refresh_server_inventory(&server_name).await {
                            warn!("Background refresh failed for '{}': {}", server_name, e);
                        }
                    }

                    debug!("Background refresh: Completed refresh cycle");
                }
            }
        })
    }

    // ========================================================================
    // Additional Tool/Prompt/Resource Methods
    // ========================================================================

    /// Check if a tool exists
    pub fn has_tool(&self, name: &str) -> bool {
        self.inventory.has_tool(name)
    }

    /// Get prompt info by name
    pub fn get_prompt_info(&self, name: &str) -> Option<PromptInfo> {
        self.inventory.get_prompt(name).map(|(_server, info)| info)
    }

    /// Get resource info by URI
    pub fn get_resource_info(&self, uri: &str) -> Option<ResourceInfo> {
        self.inventory.get_resource(uri).map(|(_server, info)| info)
    }

    /// Subscribe to resource changes
    pub async fn subscribe_resource(&self, uri: &str) -> McpResult<()> {
        let (server_name, _resource_info) = self
            .inventory
            .get_resource(uri)
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))?;

        let client = self
            .get_client(&server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

        debug!("Subscribing to '{}' on '{}'", uri, server_name);

        client
            .peer()
            .subscribe(SubscribeRequestParam {
                uri: uri.to_string(),
            })
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to subscribe: {}", e)))
    }

    /// Unsubscribe from resource changes
    pub async fn unsubscribe_resource(&self, uri: &str) -> McpResult<()> {
        let (server_name, _resource_info) = self
            .inventory
            .get_resource(uri)
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))?;

        let client = self
            .get_client(&server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

        debug!("Unsubscribing from '{}' on '{}'", uri, server_name);

        client
            .peer()
            .unsubscribe(UnsubscribeRequestParam {
                uri: uri.to_string(),
            })
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to unsubscribe: {}", e)))
    }

    /// List all connected servers
    pub fn list_servers(&self) -> Vec<String> {
        self.clients.iter().map(|e| e.key().clone()).collect()
    }

    /// Disconnect from all servers (for cleanup)
    pub async fn shutdown(&self) {
        let keys: Vec<String> = self.clients.iter().map(|e| e.key().clone()).collect();

        for name in keys {
            if let Some((_, client)) = self.clients.remove(&name) {
                // Try to unwrap Arc to call cancel
                match Arc::try_unwrap(client) {
                    Ok(client) => {
                        if let Err(e) = client.cancel().await {
                            warn!("Error disconnecting from '{}': {}", name, e);
                        }
                    }
                    Err(_) => {
                        warn!("Could not shutdown '{}': client still in use", name);
                    }
                }
            }
        }
    }

    // ========================================================================
    // Statistics and Accessors
    // ========================================================================

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
    /// Discover and cache tools/prompts/resources for a connected server
    ///
    /// This method is public to allow workflow-based inventory loading.
    /// It discovers all tools, prompts, and resources from the client and caches them in the inventory.
    pub async fn load_server_inventory(
        inventory: &Arc<ToolInventory>,
        server_name: &str,
        client: &Arc<McpClient>,
    ) {
        // Tools
        match client.peer().list_all_tools().await {
            Ok(ts) => {
                info!("Discovered {} tools from '{}'", ts.len(), server_name);
                for t in ts {
                    let tool_info = ToolInfo {
                        name: t.name.to_string(),
                        description: t.description.as_deref().unwrap_or_default().to_string(),
                        server: server_name.to_string(),
                        parameters: Some(serde_json::Value::Object((*t.input_schema).clone())),
                    };
                    inventory.insert_tool(t.name.to_string(), server_name.to_string(), tool_info);
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_name, e),
        }

        // Prompts
        match client.peer().list_all_prompts().await {
            Ok(ps) => {
                info!("Discovered {} prompts from '{}'", ps.len(), server_name);
                for p in ps {
                    let prompt_info = PromptInfo {
                        name: p.name.clone(),
                        description: p.description.clone(),
                        server: server_name.to_string(),
                        arguments: p.arguments.clone().map(|args| {
                            args.into_iter().map(|arg| serde_json::json!(arg)).collect()
                        }),
                    };
                    inventory.insert_prompt(p.name.clone(), server_name.to_string(), prompt_info);
                }
            }
            Err(e) => debug!("No prompts or failed to list on '{}': {}", server_name, e),
        }

        // Resources
        match client.peer().list_all_resources().await {
            Ok(rs) => {
                info!("Discovered {} resources from '{}'", rs.len(), server_name);
                for r in rs {
                    let resource_info = ResourceInfo {
                        uri: r.uri.clone(),
                        name: r.name.clone(),
                        description: r.description.clone(),
                        mime_type: r.mime_type.clone(),
                        server: server_name.to_string(),
                    };
                    inventory.insert_resource(
                        r.uri.clone(),
                        server_name.to_string(),
                        resource_info,
                    );
                }
            }
            Err(e) => debug!("No resources or failed to list on '{}': {}", server_name, e),
        }

        // Mark server as refreshed
        inventory.mark_refreshed(server_name);
    }

    /// Discover and cache tools/prompts/resources for a connected server (internal wrapper)
    async fn load_server_inventory_internal(&self, server_name: &str, client: &McpClient) {
        // Tools
        match client.peer().list_all_tools().await {
            Ok(ts) => {
                info!("Discovered {} tools from '{}'", ts.len(), server_name);
                for t in ts {
                    let tool_info = ToolInfo {
                        name: t.name.to_string(),
                        description: t.description.as_deref().unwrap_or_default().to_string(),
                        server: server_name.to_string(),
                        parameters: Some(serde_json::Value::Object((*t.input_schema).clone())),
                    };
                    self.inventory.insert_tool(
                        t.name.to_string(),
                        server_name.to_string(),
                        tool_info,
                    );
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_name, e),
        }

        // Prompts
        match client.peer().list_all_prompts().await {
            Ok(ps) => {
                info!("Discovered {} prompts from '{}'", ps.len(), server_name);
                for p in ps {
                    let prompt_info = PromptInfo {
                        name: p.name.clone(),
                        description: p.description.clone(),
                        server: server_name.to_string(),
                        arguments: p.arguments.clone().map(|args| {
                            args.into_iter().map(|arg| serde_json::json!(arg)).collect()
                        }),
                    };
                    self.inventory.insert_prompt(
                        p.name.clone(),
                        server_name.to_string(),
                        prompt_info,
                    );
                }
            }
            Err(e) => debug!("No prompts or failed to list on '{}': {}", server_name, e),
        }

        // Resources
        match client.peer().list_all_resources().await {
            Ok(rs) => {
                info!("Discovered {} resources from '{}'", rs.len(), server_name);
                for r in rs {
                    let resource_info = ResourceInfo {
                        uri: r.uri.clone(),
                        name: r.name.clone(),
                        description: r.description.clone(),
                        mime_type: r.mime_type.clone(),
                        server: server_name.to_string(),
                    };
                    self.inventory.insert_resource(
                        r.uri.clone(),
                        server_name.to_string(),
                        resource_info,
                    );
                }
            }
            Err(e) => debug!("No resources or failed to list on '{}': {}", server_name, e),
        }

        // Mark server as refreshed
        self.inventory.mark_refreshed(server_name);
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
    fn server_key(config: &McpServerConfig) -> String {
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
    /// Number of cached prompts
    pub prompt_count: usize,
    /// Number of cached resources
    pub resource_count: usize,
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

        let manager = McpManager::new(
            config,
            Duration::from_secs(300),
            Duration::from_secs(300),
            100,
        )
        .await
        .unwrap();
        assert_eq!(manager.list_static_servers().len(), 0);
    }
}
