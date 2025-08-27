// types.rs - All MCP data structures in one place (Python-aligned)
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ===== Errors (simplified from errors.rs) =====
#[derive(Error, Debug)]
pub enum MCPError {
    #[error("Connection failed: {0}")]
    ConnectionError(String),
    #[error("Invalid URL: {0}")]
    InvalidURL(String),
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    #[error("Tool execution failed: {0}")]
    ToolExecutionError(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

pub type MCPResult<T> = Result<T, MCPError>;

// Add From implementations for common error types
impl From<serde_json::Error> for MCPError {
    fn from(err: serde_json::Error) -> Self {
        MCPError::SerializationError(err.to_string())
    }
}

impl From<reqwest::Error> for MCPError {
    fn from(err: reqwest::Error) -> Self {
        MCPError::ConnectionError(err.to_string())
    }
}

// ===== Config (simplified from config.rs) =====
#[derive(Clone, Debug)]
pub struct MCPConfig {
    pub connection_timeout_ms: u64,
    pub dev_mode: bool,
}

impl MCPConfig {
    pub fn dev_mode() -> Self {
        Self {
            connection_timeout_ms: 30000,
            dev_mode: true,
        }
    }
}

// ===== MCP Protocol Types (matching Python's approach) =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    pub jsonrpc: String,
    pub id: String,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResponse {
    pub jsonrpc: String,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<MCPErrorResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPErrorResponse {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

// ===== MCP Server Response Types =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResponse {
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsResponse {
    pub tools: Vec<ToolInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<serde_json::Value>,
}

// ===== Simplified Types (matching Python's usage) =====
pub type ToolCall = serde_json::Value; // Python uses dict
pub type ToolResult = serde_json::Value; // Python uses dict

// ===== Connection Types (matching Python's approach) =====
#[derive(Debug, Clone)]
pub enum ConnectionType {
    Http(String),  // HTTP/SSE URL
    Stdio(String), // Command to run (e.g., "python server.py")
}

// ===== Tool Session (async context manager equivalent) =====
pub struct ToolSession {
    pub connection: ConnectionType,
    pub client: reqwest::Client,
    pub process: Option<tokio::process::Child>,
    pub session_initialized: bool,
}

impl ToolSession {
    pub async fn new(connection_str: String) -> MCPResult<Self> {
        let connection =
            if connection_str.starts_with("http://") || connection_str.starts_with("https://") {
                ConnectionType::Http(connection_str)
            } else {
                ConnectionType::Stdio(connection_str)
            };

        let mut session = Self {
            connection,
            client: reqwest::Client::new(),
            process: None,
            session_initialized: false,
        };

        // Initialize the session (like Python's await session.initialize())
        session.initialize().await?;
        Ok(session)
    }

    pub async fn new_http(url: String) -> MCPResult<Self> {
        let mut session = Self {
            connection: ConnectionType::Http(url),
            client: reqwest::Client::new(),
            process: None,
            session_initialized: false,
        };

        session.initialize().await?;
        Ok(session)
    }

    pub async fn new_stdio(command: String) -> MCPResult<Self> {
        Ok(Self {
            connection: ConnectionType::Stdio(command),
            client: reqwest::Client::new(),
            process: None,
            session_initialized: false,
        })
    }

    /// Initialize the session (like Python's await session.initialize())
    pub async fn initialize(&mut self) -> MCPResult<()> {
        if self.session_initialized {
            return Ok(());
        }

        let init_request = MCPRequest {
            jsonrpc: "2.0".to_string(),
            id: "init".to_string(),
            method: "initialize".to_string(),
            params: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {}
            })),
        };

        match &self.connection {
            ConnectionType::Http(url) => {
                let response = self
                    .client
                    .post(url)
                    .header("Content-Type", "application/json")
                    .json(&init_request)
                    .send()
                    .await
                    .map_err(|e| MCPError::ConnectionError(format!("Initialize failed: {}", e)))?;

                let mcp_response: MCPResponse = response.json().await.map_err(|e| {
                    MCPError::SerializationError(format!(
                        "Failed to parse initialize response: {}",
                        e
                    ))
                })?;

                if let Some(error) = mcp_response.error {
                    return Err(MCPError::ProtocolError(format!(
                        "Initialize error: {}",
                        error.message
                    )));
                }

                self.session_initialized = true;
                Ok(())
            }
            ConnectionType::Stdio(_) => Err(MCPError::ProtocolError(
                "Stdio initialization not yet implemented".to_string(),
            )),
        }
    }

    /// Call a tool (following MCP tools/call specification)
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> MCPResult<serde_json::Value> {
        if !self.session_initialized {
            return Err(MCPError::ProtocolError(
                "Session not initialized. Call initialize() first.".to_string(),
            ));
        }

        use serde_json::json;

        let request = MCPRequest {
            jsonrpc: "2.0".to_string(),
            id: format!(
                "call_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            ),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": name,
                "arguments": arguments
            })),
        };

        match &self.connection {
            ConnectionType::Http(url) => {
                let response = self
                    .client
                    .post(url)
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| MCPError::ConnectionError(format!("Tool call failed: {}", e)))?;

                let mcp_response: MCPResponse = response.json().await.map_err(|e| {
                    MCPError::SerializationError(format!("Failed to parse tool response: {}", e))
                })?;

                if let Some(error) = mcp_response.error {
                    return Err(MCPError::ToolExecutionError(format!(
                        "Tool '{}' failed: {}",
                        name, error.message
                    )));
                }

                mcp_response.result.ok_or_else(|| {
                    MCPError::ProtocolError("No result in tool response".to_string())
                })
            }

            ConnectionType::Stdio(_command) => Err(MCPError::ProtocolError(
                "Stdio tool calls not yet implemented".to_string(),
            )),
        }
    }

    /// Check if session is ready for tool calls
    pub fn is_ready(&self) -> bool {
        match &self.connection {
            ConnectionType::Http(_) => self.session_initialized,
            ConnectionType::Stdio(_) => false, // Stdio not implemented yet
        }
    }

    /// Get connection info for debugging
    pub fn connection_info(&self) -> String {
        match &self.connection {
            ConnectionType::Http(url) => format!("HTTP: {}", url),
            ConnectionType::Stdio(cmd) => format!("Stdio: {}", cmd),
        }
    }
}

// ===== Multi-Tool Session Manager (matching Python's tool_session_ctxs pattern) =====
pub struct MultiToolSessionManager {
    sessions: HashMap<String, ToolSession>,
    server_urls: HashMap<String, String>, // tool_name -> server_url mapping
}

impl Default for MultiToolSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiToolSessionManager {
    /// Create new multi-tool session manager
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            server_urls: HashMap::new(),
        }
    }

    /// Add tools from an MCP server (similar to Python's tool_session_ctxs creation)
    pub async fn add_tools_from_server(
        &mut self,
        server_url: String,
        tool_names: Vec<String>,
    ) -> MCPResult<()> {
        for tool_name in tool_names {
            // Store server URL mapping
            self.server_urls
                .insert(tool_name.clone(), server_url.clone());

            // Create session for this tool
            let session = ToolSession::new(server_url.clone()).await?;
            self.sessions.insert(tool_name, session);
        }
        Ok(())
    }

    /// Get session for a specific tool (matches Python's tool_sessions[tool_name])
    pub fn get_session(&self, tool_name: &str) -> Option<&ToolSession> {
        self.sessions.get(tool_name)
    }

    /// Execute tool with automatic session management
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> MCPResult<serde_json::Value> {
        let session = self
            .sessions
            .get(tool_name)
            .ok_or_else(|| MCPError::ToolNotFound(format!("No session for tool: {}", tool_name)))?;

        session.call_tool(tool_name, arguments).await
    }

    /// Execute multiple tools concurrently (enhanced feature)
    pub async fn call_tools_concurrent(
        &self,
        tool_calls: Vec<(String, serde_json::Value)>,
    ) -> Vec<MCPResult<serde_json::Value>> {
        let futures: Vec<_> = tool_calls
            .into_iter()
            .map(|(tool_name, args)| async move { self.call_tool(&tool_name, args).await })
            .collect();

        futures::future::join_all(futures).await
    }

    /// Get all available tool names
    pub fn list_tools(&self) -> Vec<String> {
        self.sessions.keys().cloned().collect()
    }

    /// Check if tool is available
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.sessions.contains_key(tool_name)
    }

    /// Get session statistics
    pub fn session_stats(&self) -> SessionStats {
        let total_sessions = self.sessions.len();
        let ready_sessions = self.sessions.values().filter(|s| s.is_ready()).count();
        let unique_servers = self
            .server_urls
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len();

        SessionStats {
            total_sessions,
            ready_sessions,
            unique_servers,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionStats {
    pub total_sessions: usize,
    pub ready_sessions: usize,
    pub unique_servers: usize,
}
