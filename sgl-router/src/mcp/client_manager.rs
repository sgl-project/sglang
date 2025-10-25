use std::{borrow::Cow, collections::HashMap, sync::Arc, time::Duration};

use backoff::ExponentialBackoffBuilder;
use rmcp::{
    model::{
        CallToolRequestParam, GetPromptRequestParam, GetPromptResult, ReadResourceRequestParam,
        ReadResourceResult,
    },
    service::RunningService,
    transport::{
        sse_client::SseClientConfig, streamable_http_client::StreamableHttpClientTransportConfig,
        ConfigureCommandExt, SseClientTransport, StreamableHttpClientTransport, TokioChildProcess,
    },
    RoleClient, ServiceExt,
};
use serde::{Deserialize, Serialize};

use crate::mcp::{
    config::{McpConfig, McpServerConfig, McpTransport},
    error::{McpError, McpResult},
    inventory::ToolInventory,
};

/// Information about an available tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub server: String,
    pub parameters: Option<serde_json::Value>,
}

/// Information about an available prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptInfo {
    pub name: String,
    pub description: Option<String>,
    pub server: String,
    pub arguments: Option<Vec<serde_json::Value>>,
}

/// Information about an available resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    pub server: String,
}

/// Manages MCP client connections and tool execution
pub struct McpClientManager {
    /// Original config for server refresh operations
    config: McpConfig,
    /// Shared tool inventory with TTL and refresh tracking
    inventory: Arc<ToolInventory>,
    /// Map of server_name -> MCP client
    clients: HashMap<String, RunningService<RoleClient, ()>>,
}

impl McpClientManager {
    /// Create a new manager and connect to all configured servers
    pub async fn new(config: McpConfig, inventory: Arc<ToolInventory>) -> McpResult<Self> {
        let mut mgr = Self {
            config: config.clone(),
            inventory,
            clients: HashMap::new(),
        };

        // Get global proxy config for all servers
        let global_proxy = mgr.config.proxy.as_ref();

        for server_config in &mgr.config.servers {
            match Self::connect_server(server_config, global_proxy).await {
                Ok(client) => {
                    mgr.load_server_inventory(&server_config.name, &client)
                        .await;
                    mgr.clients.insert(server_config.name.clone(), client);
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to connect to server '{}': {}",
                        server_config.name,
                        e
                    );
                }
            }
        }

        if mgr.clients.is_empty() {
            return Err(McpError::ConnectionFailed(
                "Failed to connect to any MCP servers".to_string(),
            ));
        }
        Ok(mgr)
    }

    /// Discover and cache tools/prompts/resources for a connected server
    async fn load_server_inventory(
        &self,
        server_name: &str,
        client: &RunningService<RoleClient, ()>,
    ) {
        // Tools
        match client.peer().list_all_tools().await {
            Ok(ts) => {
                tracing::info!("Discovered {} tools from '{}'", ts.len(), server_name);
                for t in ts {
                    // Populate shared inventory
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
            Err(e) => tracing::warn!("Failed to list tools from '{}': {}", server_name, e),
        }

        // Prompts
        match client.peer().list_all_prompts().await {
            Ok(ps) => {
                tracing::info!("Discovered {} prompts from '{}'", ps.len(), server_name);
                for p in ps {
                    // Populate shared inventory
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
            Err(e) => tracing::debug!("No prompts or failed to list on '{}': {}", server_name, e),
        }

        // Resources
        match client.peer().list_all_resources().await {
            Ok(rs) => {
                tracing::info!("Discovered {} resources from '{}'", rs.len(), server_name);
                for r in rs {
                    // Populate shared inventory
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
            Err(e) => tracing::debug!("No resources or failed to list on '{}': {}", server_name, e),
        }

        // Mark server as refreshed in inventory
        self.inventory.mark_refreshed(server_name);
    }

    /// Refresh inventory for a specific server
    ///
    /// Re-discovers tools, prompts, and resources from the server and updates
    /// both internal caches and shared inventory with fresh data.
    pub async fn refresh_server_inventory(&self, server_name: &str) -> McpResult<()> {
        let client = self.client_for(server_name)?;

        tracing::info!("Refreshing inventory for server: {}", server_name);

        // Refresh tools
        match client.peer().list_all_tools().await {
            Ok(tools) => {
                tracing::debug!("Refreshed {} tools from '{}'", tools.len(), server_name);
                for tool in tools {
                    // Update shared inventory
                    let tool_info = ToolInfo {
                        name: tool.name.to_string(),
                        description: tool.description.as_deref().unwrap_or_default().to_string(),
                        server: server_name.to_string(),
                        parameters: Some(serde_json::Value::Object((*tool.input_schema).clone())),
                    };
                    self.inventory.insert_tool(
                        tool.name.to_string(),
                        server_name.to_string(),
                        tool_info,
                    );
                }
            }
            Err(e) => {
                tracing::warn!("Failed to refresh tools from '{}': {}", server_name, e);
                return Err(McpError::ToolExecution(format!(
                    "Failed to refresh tools: {}",
                    e
                )));
            }
        }

        // Refresh prompts
        match client.peer().list_all_prompts().await {
            Ok(prompts) => {
                tracing::debug!("Refreshed {} prompts from '{}'", prompts.len(), server_name);
                for prompt in prompts {
                    // Update shared inventory
                    let prompt_info = PromptInfo {
                        name: prompt.name.clone(),
                        description: prompt.description.clone(),
                        server: server_name.to_string(),
                        arguments: prompt.arguments.clone().map(|args| {
                            args.into_iter().map(|arg| serde_json::json!(arg)).collect()
                        }),
                    };
                    self.inventory.insert_prompt(
                        prompt.name.clone(),
                        server_name.to_string(),
                        prompt_info,
                    );
                }
            }
            Err(e) => {
                tracing::debug!(
                    "No prompts or failed to refresh on '{}': {}",
                    server_name,
                    e
                );
            }
        }

        // Refresh resources
        match client.peer().list_all_resources().await {
            Ok(resources) => {
                tracing::debug!(
                    "Refreshed {} resources from '{}'",
                    resources.len(),
                    server_name
                );
                for resource in resources {
                    // Update shared inventory
                    let resource_info = ResourceInfo {
                        uri: resource.uri.clone(),
                        name: resource.name.clone(),
                        description: resource.description.clone(),
                        mime_type: resource.mime_type.clone(),
                        server: server_name.to_string(),
                    };
                    self.inventory.insert_resource(
                        resource.uri.clone(),
                        server_name.to_string(),
                        resource_info,
                    );
                }
            }
            Err(e) => {
                tracing::debug!(
                    "No resources or failed to refresh on '{}': {}",
                    server_name,
                    e
                );
            }
        }

        // Mark server as refreshed in inventory
        self.inventory.mark_refreshed(server_name);

        tracing::info!("Successfully refreshed inventory for: {}", server_name);
        Ok(())
    }

    /// Connect to a single MCP server with retry logic for remote transports
    async fn connect_server(
        config: &McpServerConfig,
        global_proxy: Option<&crate::mcp::McpProxyConfig>,
    ) -> McpResult<RunningService<RoleClient, ()>> {
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
        global_proxy: Option<&crate::mcp::McpProxyConfig>,
    ) -> McpResult<RunningService<RoleClient, ()>> {
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
                        tracing::error!(
                            "Permanent error connecting to '{}': {} - not retrying",
                            config.name,
                            e
                        );
                        Err(backoff::Error::permanent(e))
                    } else {
                        tracing::warn!("Failed to connect to '{}', retrying: {}", config.name, e);
                        Err(backoff::Error::transient(e))
                    }
                }
            }
        })
        .await
    }

    /// Determine if an error is permanent (should not retry) or transient (should retry)
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
            // Tool-related errors shouldn't occur during connection
            _ => false,
        }
    }

    /// Internal implementation of server connection
    async fn connect_server_impl(
        config: &McpServerConfig,
        global_proxy: Option<&crate::mcp::McpProxyConfig>,
    ) -> McpResult<RunningService<RoleClient, ()>> {
        tracing::info!(
            "Connecting to MCP server '{}' via {:?}",
            config.name,
            config.transport
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

                tracing::info!("Connected to stdio server '{}'", config.name);
                Ok(client)
            }

            McpTransport::Sse { url, token } => {
                // Resolve proxy configuration (server-specific overrides global)
                let proxy_config = crate::mcp::proxy::resolve_proxy_config(config, global_proxy);

                // Create HTTP client with proxy support
                let client = if token.is_some() {
                    // Build client with both proxy and auth headers
                    let mut builder = reqwest::Client::builder()
                        .timeout(Duration::from_secs(30))
                        .connect_timeout(Duration::from_secs(10));

                    // Apply proxy configuration
                    if let Some(proxy_cfg) = proxy_config {
                        if let Some(ref http_proxy) = proxy_cfg.http {
                            let mut proxy = reqwest::Proxy::http(http_proxy).map_err(|e| {
                                McpError::Config(format!("Invalid HTTP proxy: {}", e))
                            })?;
                            if let Some(ref no_proxy) = proxy_cfg.no_proxy {
                                proxy = proxy.no_proxy(reqwest::NoProxy::from_string(no_proxy));
                            }
                            if let (Some(ref username), Some(ref password)) =
                                (&proxy_cfg.username, &proxy_cfg.password)
                            {
                                proxy = proxy.basic_auth(username, password);
                            }
                            builder = builder.proxy(proxy);
                        }
                        if let Some(ref https_proxy) = proxy_cfg.https {
                            let mut proxy = reqwest::Proxy::https(https_proxy).map_err(|e| {
                                McpError::Config(format!("Invalid HTTPS proxy: {}", e))
                            })?;
                            if let Some(ref no_proxy) = proxy_cfg.no_proxy {
                                proxy = proxy.no_proxy(reqwest::NoProxy::from_string(no_proxy));
                            }
                            if let (Some(ref username), Some(ref password)) =
                                (&proxy_cfg.username, &proxy_cfg.password)
                            {
                                proxy = proxy.basic_auth(username, password);
                            }
                            builder = builder.proxy(proxy);
                        }
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
                    // Use proxy.rs function for non-auth case
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

                tracing::info!("Connected to SSE server '{}' at {}", config.name, url);
                Ok(client)
            }

            McpTransport::Streamable { url, token } => {
                // TODO(Phase 7): Add proxy support for Streamable transport
                // StreamableHttpClientTransport doesn't expose HTTP client configuration,
                // so proxy support requires changes to rmcp library or custom transport impl
                let _proxy_config = crate::mcp::proxy::resolve_proxy_config(config, global_proxy);
                if _proxy_config.is_some() {
                    tracing::warn!(
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

                tracing::info!(
                    "Connected to streamable HTTP server '{}' at {}",
                    config.name,
                    url
                );
                Ok(client)
            }
        }
    }

    fn client_for(&self, server_name: &str) -> McpResult<&RunningService<RoleClient, ()>> {
        self.clients
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))
    }

    fn tool_entry(&self, name: &str) -> McpResult<String> {
        self.inventory
            .get_tool(name)
            .map(|(server_name, _tool_info)| server_name)
            .ok_or_else(|| McpError::ToolNotFound(name.to_string()))
    }

    fn prompt_entry(&self, name: &str) -> McpResult<String> {
        self.inventory
            .get_prompt(name)
            .map(|(server_name, _prompt_info)| server_name)
            .ok_or_else(|| McpError::PromptNotFound(name.to_string()))
    }

    fn resource_entry(&self, uri: &str) -> McpResult<String> {
        self.inventory
            .get_resource(uri)
            .map(|(server_name, _resource_info)| server_name)
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))
    }

    /// Call a tool by name
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> McpResult<rmcp::model::CallToolResult> {
        let server_name = self.tool_entry(tool_name)?;
        let client = self.client_for(&server_name)?;

        tracing::debug!("Calling tool '{}' on '{}'", tool_name, server_name);

        client
            .peer()
            .call_tool(CallToolRequestParam {
                name: Cow::Owned(tool_name.to_string()),
                arguments,
            })
            .await
            .map_err(|e| McpError::ToolExecution(format!("Tool call failed: {}", e)))
    }

    /// Get all available tools
    pub fn list_tools(&self) -> Vec<ToolInfo> {
        self.inventory
            .list_tools()
            .into_iter()
            .map(|(_tool_name, _server_name, tool_info)| tool_info)
            .collect()
    }

    /// Get a specific tool by name
    pub fn get_tool(&self, name: &str) -> Option<ToolInfo> {
        self.inventory
            .get_tool(name)
            .map(|(_server_name, tool_info)| tool_info)
    }

    /// Check if a tool exists
    pub fn has_tool(&self, name: &str) -> bool {
        self.inventory.has_tool(name)
    }

    /// Get list of connected servers
    pub fn list_servers(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }

    /// Get a prompt by name with arguments
    pub async fn get_prompt(
        &self,
        prompt_name: &str,
        arguments: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> McpResult<GetPromptResult> {
        let server_name = self.prompt_entry(prompt_name)?;
        let client = self.client_for(&server_name)?;

        tracing::debug!("Getting prompt '{}' from '{}'", prompt_name, server_name);

        client
            .peer()
            .get_prompt(GetPromptRequestParam {
                name: prompt_name.to_string(),
                arguments,
            })
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to get prompt: {}", e)))
    }

    /// List all available prompts
    pub fn list_prompts(&self) -> Vec<PromptInfo> {
        self.inventory
            .list_prompts()
            .into_iter()
            .map(|(_prompt_name, _server_name, prompt_info)| prompt_info)
            .collect()
    }

    /// Get a specific prompt info by name
    pub fn get_prompt_info(&self, name: &str) -> Option<PromptInfo> {
        self.inventory
            .get_prompt(name)
            .map(|(_server_name, prompt_info)| prompt_info)
    }

    /// Read a resource by URI
    pub async fn read_resource(&self, uri: &str) -> McpResult<ReadResourceResult> {
        let server_name = self.resource_entry(uri)?;
        let client = self.client_for(&server_name)?;

        tracing::debug!("Reading resource '{}' from '{}'", uri, server_name);

        client
            .peer()
            .read_resource(ReadResourceRequestParam {
                uri: uri.to_string(),
            })
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to read resource: {}", e)))
    }

    /// List all available resources
    pub fn list_resources(&self) -> Vec<ResourceInfo> {
        self.inventory
            .list_resources()
            .into_iter()
            .map(|(_resource_uri, _server_name, resource_info)| resource_info)
            .collect()
    }

    /// Get a specific resource info by URI
    pub fn get_resource_info(&self, uri: &str) -> Option<ResourceInfo> {
        self.inventory
            .get_resource(uri)
            .map(|(_server_name, resource_info)| resource_info)
    }

    /// Subscribe to resource changes
    pub async fn subscribe_resource(&self, uri: &str) -> McpResult<()> {
        let server_name = self.resource_entry(uri)?;
        let client = self.client_for(&server_name)?;

        tracing::debug!("Subscribing to '{}' on '{}'", uri, server_name);

        client
            .peer()
            .subscribe(rmcp::model::SubscribeRequestParam {
                uri: uri.to_string(),
            })
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to subscribe: {}", e)))
    }

    /// Unsubscribe from resource changes
    pub async fn unsubscribe_resource(&self, uri: &str) -> McpResult<()> {
        let server_name = self.resource_entry(uri)?;
        let client = self.client_for(&server_name)?;

        tracing::debug!("Unsubscribing from '{}' on '{}'", uri, server_name);

        client
            .peer()
            .unsubscribe(rmcp::model::UnsubscribeRequestParam {
                uri: uri.to_string(),
            })
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to unsubscribe: {}", e)))
    }

    /// Spawn a background task to periodically refresh server inventories
    ///
    /// This task will refresh all connected servers at the specified interval
    /// to keep the inventory fresh and detect new/removed tools.
    ///
    /// # Arguments
    /// * `refresh_interval` - How often to refresh (default: 5 minutes)
    ///
    /// # Returns
    /// A JoinHandle that can be used to cancel the background task
    pub fn spawn_background_refresh(
        self: Arc<Self>,
        refresh_interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(refresh_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                tracing::debug!("Background refresh: Refreshing all server inventories");

                // Get list of servers to refresh
                let server_names: Vec<String> = self.clients.keys().cloned().collect();

                for server_name in server_names {
                    if let Err(e) = self.refresh_server_inventory(&server_name).await {
                        tracing::warn!("Background refresh failed for '{}': {}", server_name, e);
                    }
                }

                tracing::debug!("Background refresh: Completed refresh cycle");
            }
        })
    }

    /// Disconnect from all servers (for cleanup)
    pub async fn shutdown(&mut self) {
        for (name, client) in self.clients.drain() {
            if let Err(e) = client.cancel().await {
                tracing::warn!("Error disconnecting from '{}': {}", name, e);
            }
        }
    }
}
