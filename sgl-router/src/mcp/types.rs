// types.rs - All MCP data structures
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid;

// ===== Errors =====
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

// ===== MCP Protocol Types =====
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

// ===== Types =====
pub type ToolCall = serde_json::Value; // Python uses dict
pub type ToolResult = serde_json::Value; // Python uses dict

// ===== Connection Types =====
#[derive(Debug, Clone)]
pub struct HttpConnection {
    pub url: String,
}

// ===== Tool Session =====
pub struct ToolSession {
    pub connection: HttpConnection,
    pub client: reqwest::Client,
    pub session_initialized: bool,
}

impl ToolSession {
    pub async fn new(connection_str: String) -> MCPResult<Self> {
        if !connection_str.starts_with("http://") && !connection_str.starts_with("https://") {
            return Err(MCPError::InvalidURL(format!(
                "Only HTTP/HTTPS URLs are supported: {}",
                connection_str
            )));
        }

        let mut session = Self {
            connection: HttpConnection {
                url: connection_str,
            },
            client: reqwest::Client::new(),
            session_initialized: false,
        };

        // Initialize the session
        session.initialize().await?;
        Ok(session)
    }

    pub async fn new_http(url: String) -> MCPResult<Self> {
        Self::new(url).await
    }

    /// Initialize the session
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

        let response = self
            .client
            .post(&self.connection.url)
            .header("Content-Type", "application/json")
            .json(&init_request)
            .send()
            .await
            .map_err(|e| MCPError::ConnectionError(format!("Initialize failed: {}", e)))?;

        let mcp_response: MCPResponse = response.json().await.map_err(|e| {
            MCPError::SerializationError(format!("Failed to parse initialize response: {}", e))
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

    /// Call a tool using MCP tools/call
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
            id: format!("call_{}", uuid::Uuid::new_v4()),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": name,
                "arguments": arguments
            })),
        };

        let response = self
            .client
            .post(&self.connection.url)
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

        mcp_response
            .result
            .ok_or_else(|| MCPError::ProtocolError("No result in tool response".to_string()))
    }

    /// Check if session is ready for tool calls
    pub fn is_ready(&self) -> bool {
        self.session_initialized
    }

    /// Get connection info
    pub fn connection_info(&self) -> String {
        format!("HTTP: {}", self.connection.url)
    }
}

// ===== Multi-Tool Session Manager =====
pub struct MultiToolSessionManager {
    sessions: HashMap<String, ToolSession>, // server_url -> session
    tool_to_server: HashMap<String, String>, // tool_name -> server_url mapping
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
            tool_to_server: HashMap::new(),
        }
    }

    /// Add tools from an MCP server (optimized to share sessions per server)
    pub async fn add_tools_from_server(
        &mut self,
        server_url: String,
        tool_names: Vec<String>,
    ) -> MCPResult<()> {
        // Create one session per server URL (if not already exists)
        if !self.sessions.contains_key(&server_url) {
            let session = ToolSession::new(server_url.clone()).await?;
            self.sessions.insert(server_url.clone(), session);
        }

        // Map all tools to this server URL
        for tool_name in tool_names {
            self.tool_to_server.insert(tool_name, server_url.clone());
        }
        Ok(())
    }

    /// Get session for a specific tool
    pub fn get_session(&self, tool_name: &str) -> Option<&ToolSession> {
        let server_url = self.tool_to_server.get(tool_name)?;
        self.sessions.get(server_url)
    }

    /// Execute tool with automatic session management
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> MCPResult<serde_json::Value> {
        let server_url = self
            .tool_to_server
            .get(tool_name)
            .ok_or_else(|| MCPError::ToolNotFound(format!("No mapping for tool: {}", tool_name)))?;

        let session = self.sessions.get(server_url).ok_or_else(|| {
            MCPError::ToolNotFound(format!("No session for server: {}", server_url))
        })?;

        session.call_tool(tool_name, arguments).await
    }

    /// Execute multiple tools concurrently
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
        self.tool_to_server.keys().cloned().collect()
    }

    /// Check if tool is available
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tool_to_server.contains_key(tool_name)
    }

    /// Get session statistics
    pub fn session_stats(&self) -> SessionStats {
        let total_sessions = self.sessions.len();
        let ready_sessions = self.sessions.values().filter(|s| s.is_ready()).count();
        let unique_servers = self.sessions.len(); // Now sessions = servers

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
