// types.rs - All MCP data structures in one place (Python-aligned)
use serde::{Deserialize, Serialize};
// Note: HashMap import removed as it's not currently used
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
pub type ToolCall = serde_json::Value;   // Python uses dict
pub type ToolResult = serde_json::Value; // Python uses dict

// ===== Tool Session (async context manager equivalent) =====
pub struct ToolSession {
    pub url: String,
    pub client: reqwest::Client,
}

impl ToolSession {
    pub async fn new(url: String) -> MCPResult<Self> {
        Ok(Self {
            url,
            client: reqwest::Client::new(),
        })
    }
    
    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> MCPResult<serde_json::Value> {
        use serde_json::json;
        
        let request = MCPRequest {
            jsonrpc: "2.0".to_string(),
            id: "tool_call_1".to_string(),
            method: "call_tool".to_string(),
            params: Some(json!({
                "name": name,
                "arguments": arguments
            })),
        };
        
        // Send request via HTTP POST (simplified for now)
        let response = self.client
            .post(&self.url)
            .json(&request)
            .send()
            .await
            .map_err(|e| MCPError::ConnectionError(e.to_string()))?;
        
        let mcp_response: MCPResponse = response
            .json()
            .await
            .map_err(|e| MCPError::ConnectionError(e.to_string()))?;
        
        if let Some(error) = mcp_response.error {
            return Err(MCPError::ProtocolError(error.message));
        }
        
        mcp_response.result.ok_or_else(|| MCPError::ProtocolError("No result".to_string()))
    }
}