//! MCP client management and orchestration.
//!
//! Manages both static MCP servers (from config) and dynamic MCP servers (from requests).
//! Static clients are never evicted; dynamic clients use LRU eviction via connection pool.

use std::{borrow::Cow, collections::HashMap, sync::Arc, time::Duration};

use backoff::ExponentialBackoffBuilder;
use dashmap::DashMap;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION};
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
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use crate::mcp::{
    config::{McpConfig, McpProxyConfig, McpServerConfig, McpTransport, Prompt, RawResource, Tool},
    connection_pool::McpConnectionPool,
    error::{McpError, McpResult},
    inventory::ToolInventory,
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
            let server_key = Self::server_key(server_config);
            match Self::connect_server(server_config, global_proxy).await {
                Ok(client) => {
                    let client_arc = Arc::new(client);
                    // Load inventory for this server
                    Self::load_server_inventory(&inventory, &server_key, &client_arc).await;
                    static_clients.insert(server_key.clone(), client_arc);
                    info!(
                        "Connected to static server '{}' (key: {})",
                        server_config.name, server_key
                    );
                }
                Err(e) => {
                    error!(
                        "Failed to connect to static server '{}': {}",
                        server_config.name, e
                    );
                }
            }
        }

        if static_clients.is_empty() {
            info!("No static MCP servers connected");
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

    pub async fn get_client(&self, server_key: &str) -> Option<Arc<McpClient>> {
        if let Some(client) = self.static_clients.get(server_key) {
            return Some(Arc::clone(client.value()));
        }
        self.connection_pool.get(server_key)
    }

    pub async fn get_or_create_client(
        &self,
        server_config: McpServerConfig,
    ) -> McpResult<Arc<McpClient>> {
        let server_key = Self::server_key(&server_config);

        // Check if this matches an existing static server
        if let Some(client) = self.static_clients.get(&server_key) {
            return Ok(Arc::clone(client.value()));
        }
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

        self.inventory.clear_server_tools(&server_key);
        Self::load_server_inventory(&self.inventory, &server_key, &client).await;
        Ok(client)
    }

    pub fn list_static_servers(&self) -> Vec<String> {
        self.static_clients
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    pub fn is_static_server(&self, server_key: &str) -> bool {
        self.static_clients.contains_key(server_key)
    }

    pub fn register_static_server(&self, server_key: String, client: Arc<McpClient>) {
        self.static_clients.insert(server_key.clone(), client);
        info!("Registered static MCP server: {}", server_key);
    }

    /// List all available tools from all servers
    pub fn list_tools(&self) -> Vec<Tool> {
        self.inventory
            .list_tools()
            .into_iter()
            .map(|(_tool_name, _server_name, tool_info)| tool_info)
            .collect()
    }

    /// List tools only from specific servers plus all static servers
    ///
    /// This method filters tools to only include:
    /// 1. Tools from static servers (always visible)
    /// 2. Tools from the specified dynamic servers
    ///
    /// This provides request-scoped tool isolation while maintaining
    /// global visibility for static servers.
    pub fn list_tools_for_servers(&self, server_keys: &[String]) -> Vec<Tool> {
        self.inventory
            .list_tools()
            .into_iter()
            .filter(|(_tool_name, server_key, _tool_info)| {
                // Include if:
                // 1. It's a static server (check by name in static_clients)
                // 2. It's in the requested servers list
                self.is_static_server(server_key) || server_keys.contains(server_key)
            })
            .map(|(_tool_name, _server_key, tool_info)| tool_info)
            .collect()
    }

    /// Call a tool on a specific server with already-parsed arguments.
    pub async fn call_tool_on_server(
        &self,
        server_key: &str,
        tool_name: &str,
        arguments: Option<Map<String, serde_json::Value>>,
    ) -> McpResult<CallToolResult> {
        let client = self
            .get_client(server_key)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_key.to_string()))?;

        let request = CallToolRequestParam {
            name: Cow::Owned(tool_name.to_string()),
            arguments,
        };

        client
            .call_tool(request)
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to call tool: {}", e)))
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
        let tool_schema = Some(serde_json::Value::Object((*tool_info.input_schema).clone()));
        let args_map = args
            .into()
            .into_map(tool_schema.as_ref())
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
    pub fn get_tool(&self, tool_name: &str) -> Option<Tool> {
        self.inventory
            .get_tool(tool_name)
            .map(|(_server_name, tool_info)| tool_info)
    }

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
    pub fn list_prompts(&self) -> Vec<Prompt> {
        self.inventory
            .list_prompts()
            .into_iter()
            .map(|(_prompt_name, _server_name, prompt_info)| prompt_info)
            .collect()
    }

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
    pub fn list_resources(&self) -> Vec<RawResource> {
        self.inventory
            .list_resources()
            .into_iter()
            .map(|(_resource_uri, _server_name, resource_info)| resource_info)
            .collect()
    }

    /// Refresh inventory for a specific server
    pub async fn refresh_server_inventory(&self, server_key: &str) -> McpResult<()> {
        let client = self
            .get_client(server_key)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_key.to_string()))?;

        info!("Refreshing inventory for server: {}", server_key);
        Self::load_server_inventory(&self.inventory, server_key, &client).await;
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

    /// Check if a tool exists
    pub fn has_tool(&self, name: &str) -> bool {
        self.inventory.has_tool(name)
    }

    /// Get prompt info by name
    pub fn get_prompt_info(&self, name: &str) -> Option<Prompt> {
        self.inventory.get_prompt(name).map(|(_server, info)| info)
    }

    /// Get resource info by URI
    pub fn get_resource_info(&self, uri: &str) -> Option<RawResource> {
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
        let (tools, prompts, resources) = self.inventory.counts();
        McpManagerStats {
            static_server_count: self.static_clients.len(),
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
        server_key: &str,
        client: &Arc<McpClient>,
    ) {
        // Tools
        match client.peer().list_all_tools().await {
            Ok(ts) => {
                info!("Discovered {} tools from '{}'", ts.len(), server_key);
                for t in ts {
                    inventory.insert_tool(t.name.to_string(), server_key.to_string(), t);
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_key, e),
        }

        // Prompts
        match client.peer().list_all_prompts().await {
            Ok(ps) => {
                info!("Discovered {} prompts from '{}'", ps.len(), server_key);
                for p in ps {
                    inventory.insert_prompt(p.name.clone(), server_key.to_string(), p);
                }
            }
            Err(e) => debug!("No prompts or failed to list on '{}': {}", server_key, e),
        }

        // Resources
        match client.peer().list_all_resources().await {
            Ok(rs) => {
                info!("Discovered {} resources from '{}'", rs.len(), server_key);
                for r in rs {
                    inventory.insert_resource(r.uri.clone(), server_key.to_string(), r.raw);
                }
            }
            Err(e) => debug!("No resources or failed to list on '{}': {}", server_key, e),
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

    fn build_default_headers(
        headers: Option<&HashMap<String, String>>,
        token: Option<&String>,
        include_token: bool,
    ) -> McpResult<HeaderMap> {
        let mut header_map = HeaderMap::new();

        if let Some(headers) = headers {
            for (key, value) in headers {
                let name = HeaderName::from_bytes(key.as_bytes())
                    .map_err(|_| McpError::Config(format!("Invalid header name: {}", key)))?;
                let header_value = HeaderValue::from_str(value)
                    .map_err(|_| McpError::Config(format!("Invalid header value for {}", key)))?;
                header_map.insert(name, header_value);
            }
        }

        if include_token {
            if let Some(token) = token {
                if !header_map.contains_key(AUTHORIZATION) {
                    let value = HeaderValue::from_str(&format!("Bearer {}", token))
                        .map_err(|_| McpError::Auth("Invalid bearer token".to_string()))?;
                    header_map.insert(AUTHORIZATION, value);
                }
            }
        }

        Ok(header_map)
    }

    fn hash_server_identity(
        url: &str,
        token: Option<&String>,
        headers: Option<&HashMap<String, String>>,
    ) -> String {
        let mut fingerprint = String::new();
        fingerprint.push_str(url);
        fingerprint.push('\n');
        if let Some(token) = token {
            fingerprint.push_str(token);
        }
        fingerprint.push('\n');

        if let Some(headers) = headers {
            let mut items: Vec<(String, String)> = headers
                .iter()
                .map(|(k, v)| (k.to_lowercase(), v.clone()))
                .collect();
            items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            for (key, value) in items {
                fingerprint.push_str(&key);
                fingerprint.push(':');
                fingerprint.push_str(&value);
                fingerprint.push('\n');
            }
        }

        let digest = Sha256::digest(fingerprint.as_bytes());
        format!("{:x}", digest)
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

            McpTransport::Sse {
                url,
                token,
                headers,
            } => {
                // Resolve proxy configuration
                let proxy_config = crate::mcp::proxy::resolve_proxy_config(config, global_proxy);

                let default_headers =
                    Self::build_default_headers(headers.as_ref(), token.as_ref(), true)?;
                let client = if default_headers.is_empty() {
                    crate::mcp::proxy::create_http_client(proxy_config)?
                } else {
                    let mut builder =
                        reqwest::Client::builder().connect_timeout(Duration::from_secs(10));

                    // Apply proxy configuration using proxy.rs helper
                    if let Some(proxy_cfg) = proxy_config {
                        builder = crate::mcp::proxy::apply_proxy_to_builder(builder, proxy_cfg)?;
                    }

                    builder
                        .default_headers(default_headers)
                        .build()
                        .map_err(|e| McpError::Transport(format!("build HTTP client: {}", e)))?
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

            McpTransport::Streamable {
                url,
                token,
                headers,
            } => {
                let proxy_config = crate::mcp::proxy::resolve_proxy_config(config, global_proxy);
                // Note: Streamable transport doesn't support proxy yet
                if proxy_config.is_some() {
                    warn!(
                        "Proxy configuration detected but not supported for Streamable transport on server '{}'",
                        config.name
                    );
                }
                let default_headers = Self::build_default_headers(headers.as_ref(), None, false)?;
                let client = if default_headers.is_empty() {
                    crate::mcp::proxy::create_http_client(proxy_config)?
                } else {
                    let mut builder =
                        reqwest::Client::builder().connect_timeout(Duration::from_secs(10));

                    if let Some(proxy_cfg) = proxy_config {
                        builder = crate::mcp::proxy::apply_proxy_to_builder(builder, proxy_cfg)?;
                    }

                    builder
                        .default_headers(default_headers)
                        .build()
                        .map_err(|e| McpError::Transport(format!("build HTTP client: {}", e)))?
                };

                let mut cfg = StreamableHttpClientTransportConfig::with_uri(url.as_str());
                if let Some(tok) = token {
                    cfg.auth_header = Some(format!("Bearer {}", tok));
                }

                let transport = StreamableHttpClientTransport::with_client(client, cfg);

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

    /// Generate a unique key for a server config based on its transport
    pub fn server_key(config: &McpServerConfig) -> String {
        match &config.transport {
            McpTransport::Streamable {
                url,
                token,
                headers,
            } => {
                let digest = Self::hash_server_identity(url, token.as_ref(), headers.as_ref());
                format!("{}#{}", url, digest)
            }
            McpTransport::Sse {
                url,
                token,
                headers,
            } => {
                let digest = Self::hash_server_identity(url, token.as_ref(), headers.as_ref());
                format!("{}#{}", url, digest)
            }
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

        let manager = McpManager::new(config, 100).await.unwrap();
        assert_eq!(manager.list_static_servers().len(), 0);
    }
}
