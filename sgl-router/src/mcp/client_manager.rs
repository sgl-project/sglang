use std::{borrow::Cow, collections::HashMap, time::Duration};

use backoff::ExponentialBackoffBuilder;
use dashmap::DashMap;
use rmcp::{
    model::{
        CallToolRequestParam, GetPromptRequestParam, GetPromptResult, Prompt,
        ReadResourceRequestParam, ReadResourceResult, Resource, Tool as McpTool,
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
    /// Map of server_name -> MCP client
    clients: HashMap<String, RunningService<RoleClient, ()>>,
    /// Map of tool_name -> (server_name, tool_definition)
    tools: DashMap<String, (String, McpTool)>,
    /// Map of prompt_name -> (server_name, prompt_definition)
    prompts: DashMap<String, (String, Prompt)>,
    /// Map of resource_uri -> (server_name, resource_definition)
    resources: DashMap<String, (String, Resource)>,
}

impl McpClientManager {
    /// Create a new manager and connect to all configured servers
    pub async fn new(config: McpConfig) -> McpResult<Self> {
        let mut mgr = Self {
            clients: HashMap::new(),
            tools: DashMap::new(),
            prompts: DashMap::new(),
            resources: DashMap::new(),
        };

        for server_config in config.servers {
            match Self::connect_server(&server_config).await {
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
                    if self.tools.contains_key(t.name.as_ref()) {
                        tracing::warn!(
                            "Tool '{}' from server '{}' is overwriting an existing tool.",
                            &t.name,
                            server_name
                        );
                    }
                    self.tools
                        .insert(t.name.to_string(), (server_name.to_string(), t));
                }
            }
            Err(e) => tracing::warn!("Failed to list tools from '{}': {}", server_name, e),
        }

        // Prompts
        match client.peer().list_all_prompts().await {
            Ok(ps) => {
                tracing::info!("Discovered {} prompts from '{}'", ps.len(), server_name);
                for p in ps {
                    if self.prompts.contains_key(&p.name) {
                        tracing::warn!(
                            "Prompt '{}' from server '{}' is overwriting an existing prompt.",
                            &p.name,
                            server_name
                        );
                    }
                    self.prompts
                        .insert(p.name.clone(), (server_name.to_string(), p));
                }
            }
            Err(e) => tracing::debug!("No prompts or failed to list on '{}': {}", server_name, e),
        }

        // Resources
        match client.peer().list_all_resources().await {
            Ok(rs) => {
                tracing::info!("Discovered {} resources from '{}'", rs.len(), server_name);
                for r in rs {
                    if self.resources.contains_key(&r.uri) {
                        tracing::warn!(
                            "Resource '{}' from server '{}' is overwriting an existing resource.",
                            &r.uri,
                            server_name
                        );
                    }
                    self.resources
                        .insert(r.uri.clone(), (server_name.to_string(), r));
                }
            }
            Err(e) => tracing::debug!("No resources or failed to list on '{}': {}", server_name, e),
        }
    }

    /// Connect to a single MCP server with retry logic for remote transports
    async fn connect_server(config: &McpServerConfig) -> McpResult<RunningService<RoleClient, ()>> {
        let needs_retry = matches!(
            &config.transport,
            McpTransport::Sse { .. } | McpTransport::Streamable { .. }
        );
        if needs_retry {
            Self::connect_server_with_retry(config).await
        } else {
            Self::connect_server_impl(config).await
        }
    }

    /// Connect with exponential backoff retry for remote servers
    async fn connect_server_with_retry(
        config: &McpServerConfig,
    ) -> McpResult<RunningService<RoleClient, ()>> {
        let backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_secs(1))
            .with_max_interval(Duration::from_secs(30))
            .with_max_elapsed_time(Some(Duration::from_secs(30)))
            .build();

        backoff::future::retry(backoff, || async {
            match Self::connect_server_impl(config).await {
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
                let transport = if let Some(tok) = token {
                    let client = reqwest::Client::builder()
                        .default_headers({
                            let mut headers = reqwest::header::HeaderMap::new();
                            headers.insert(
                                reqwest::header::AUTHORIZATION,
                                format!("Bearer {}", tok).parse().map_err(|e| {
                                    McpError::Transport(format!("auth token: {}", e))
                                })?,
                            );
                            headers
                        })
                        .build()
                        .map_err(|e| McpError::Transport(format!("build HTTP client: {}", e)))?;

                    let cfg = SseClientConfig {
                        sse_endpoint: url.clone().into(),
                        ..Default::default()
                    };

                    SseClientTransport::start_with_client(client, cfg)
                        .await
                        .map_err(|e| McpError::Transport(format!("create SSE transport: {}", e)))?
                } else {
                    SseClientTransport::start(url.as_str())
                        .await
                        .map_err(|e| McpError::Transport(format!("create SSE transport: {}", e)))?
                };

                let client = ().serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize SSE client: {}", e))
                })?;

                tracing::info!("Connected to SSE server '{}' at {}", config.name, url);
                Ok(client)
            }

            McpTransport::Streamable { url, token } => {
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

    fn tool_entry(&self, name: &str) -> McpResult<(String, McpTool)> {
        self.tools
            .get(name)
            .map(|e| e.value().clone())
            .ok_or_else(|| McpError::ToolNotFound(name.to_string()))
    }

    fn prompt_entry(&self, name: &str) -> McpResult<(String, Prompt)> {
        self.prompts
            .get(name)
            .map(|e| e.value().clone())
            .ok_or_else(|| McpError::PromptNotFound(name.to_string()))
    }

    fn resource_entry(&self, uri: &str) -> McpResult<(String, Resource)> {
        self.resources
            .get(uri)
            .map(|e| e.value().clone())
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))
    }

    /// Call a tool by name
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> McpResult<rmcp::model::CallToolResult> {
        let (server_name, _tool) = self.tool_entry(tool_name)?;
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
        self.tools
            .iter()
            .map(|entry| {
                let tool_name = entry.key().clone();
                let (server_name, tool) = entry.value();
                ToolInfo {
                    name: tool_name,
                    description: tool.description.as_deref().unwrap_or_default().to_string(),
                    server: server_name.clone(),
                    parameters: Some(serde_json::Value::Object((*tool.input_schema).clone())),
                }
            })
            .collect()
    }

    /// Get a specific tool by name
    pub fn get_tool(&self, name: &str) -> Option<ToolInfo> {
        self.tools.get(name).map(|entry| {
            let (server_name, tool) = entry.value();
            ToolInfo {
                name: name.to_string(),
                description: tool.description.as_deref().unwrap_or_default().to_string(),
                server: server_name.clone(),
                parameters: Some(serde_json::Value::Object((*tool.input_schema).clone())),
            }
        })
    }

    /// Check if a tool exists
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
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
        let (server_name, _prompt) = self.prompt_entry(prompt_name)?;
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
        self.prompts
            .iter()
            .map(|entry| {
                let name = entry.key().clone();
                let (server_name, prompt) = entry.value();
                PromptInfo {
                    name,
                    description: prompt.description.clone(),
                    server: server_name.clone(),
                    arguments: prompt
                        .arguments
                        .clone()
                        .map(|args| args.into_iter().map(|arg| serde_json::json!(arg)).collect()),
                }
            })
            .collect()
    }

    /// Get a specific prompt info by name
    pub fn get_prompt_info(&self, name: &str) -> Option<PromptInfo> {
        self.prompts.get(name).map(|entry| {
            let (server_name, prompt) = entry.value();
            PromptInfo {
                name: name.to_string(),
                description: prompt.description.clone(),
                server: server_name.clone(),
                arguments: prompt
                    .arguments
                    .clone()
                    .map(|args| args.into_iter().map(|arg| serde_json::json!(arg)).collect()),
            }
        })
    }

    /// Read a resource by URI
    pub async fn read_resource(&self, uri: &str) -> McpResult<ReadResourceResult> {
        let (server_name, _resource) = self.resource_entry(uri)?;
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
        self.resources
            .iter()
            .map(|entry| {
                let uri = entry.key().clone();
                let (server_name, resource) = entry.value();
                ResourceInfo {
                    uri,
                    name: resource.name.clone(),
                    description: resource.description.clone(),
                    mime_type: resource.mime_type.clone(),
                    server: server_name.clone(),
                }
            })
            .collect()
    }

    /// Get a specific resource info by URI
    pub fn get_resource_info(&self, uri: &str) -> Option<ResourceInfo> {
        self.resources.get(uri).map(|entry| {
            let (server_name, resource) = entry.value();
            ResourceInfo {
                uri: uri.to_string(),
                name: resource.name.clone(),
                description: resource.description.clone(),
                mime_type: resource.mime_type.clone(),
                server: server_name.clone(),
            }
        })
    }

    /// Subscribe to resource changes
    pub async fn subscribe_resource(&self, uri: &str) -> McpResult<()> {
        let (server_name, _resource) = self.resource_entry(uri)?;
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
        let (server_name, _resource) = self.resource_entry(uri)?;
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

    /// Disconnect from all servers (for cleanup)
    pub async fn shutdown(&mut self) {
        for (name, client) in self.clients.drain() {
            if let Err(e) = client.cancel().await {
                tracing::warn!("Error disconnecting from '{}': {}", name, e);
            }
        }
        self.tools.clear();
        self.prompts.clear();
        self.resources.clear();
    }
}
