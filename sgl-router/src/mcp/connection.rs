use crate::mcp::{MCPError, MCPResult};
use crate::mcp::types::{ConnectionInfo, ConnectionType, MCPRequest, MCPResponse, ToolRegistry, ToolMetadata};
use std::process::Stdio;
use tokio::process::{Child, Command};
use std::collections::HashMap;

pub struct LocalConnection {
    process: Option<Child>,
    connection_info: ConnectionInfo,
    tools: ToolRegistry,
}

impl LocalConnection {
    pub fn new(server_command: String, args: Vec<String>) -> Self {
        let connection_info = ConnectionInfo {
            connection_type: ConnectionType::Stdio,
            endpoint: format!("{} {}", server_command, args.join(" ")),
            timeout_ms: 5000,
        };

        Self {
            process: None,
            connection_info,
            tools: HashMap::new(),
        }
    }

    pub async fn connect(&mut self) -> MCPResult<()> {
        let parts: Vec<&str> = self.connection_info.endpoint.split_whitespace().collect();
        if parts.is_empty() {
            return Err(MCPError::ConfigurationError("Empty server command".to_string()));
        }

        let command = parts[0];
        let args = &parts[1..];

        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| MCPError::ConnectionError(format!("Failed to spawn process: {}", e)))?;

        self.process = Some(child);
        
        self.discover_tools().await?;
        
        Ok(())
    }

    pub async fn disconnect(&mut self) -> MCPResult<()> {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill().await;
            let _ = process.wait().await;
        }
        Ok(())
    }

    pub async fn send_request(&mut self, request: MCPRequest) -> MCPResult<MCPResponse> {
        if self.process.is_none() {
            return Err(MCPError::ConnectionError("Not connected to MCP server".to_string()));
        }

        let request_json = serde_json::to_string(&request)
            .map_err(|e| MCPError::ParseError(format!("Failed to serialize request: {}", e)))?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let response = MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(serde_json::json!({
                "status": "success",
                "data": "Mock response from local connection"
            })),
            error: None,
        };

        Ok(response)
    }

    async fn discover_tools(&mut self) -> MCPResult<()> {
        let mock_tools = vec![
            ToolMetadata {
                name: "file_read".to_string(),
                description: Some("Read contents of a file".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                })),
                deterministic: true,
            },
            ToolMetadata {
                name: "web_search".to_string(),
                description: Some("Search the web for information".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                })),
                deterministic: false,
            },
        ];

        for tool in mock_tools {
            self.tools.insert(tool.name.clone(), tool);
        }

        Ok(())
    }

    pub fn get_available_tools(&self) -> &ToolRegistry {
        &self.tools
    }

    pub fn is_connected(&self) -> bool {
        self.process.is_some()
    }
}