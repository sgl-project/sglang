use crate::mcp::types::{
    ConnectionInfo, ConnectionType, MCPRequest, MCPResponse, ToolMetadata, ToolRegistry,
};
use crate::mcp::{MCPError, MCPResult};
use std::collections::HashMap;
use std::process::Stdio;
use tokio::process::{Child, Command};
use tracing::{debug, warn};

pub struct LocalConnection {
    process: Option<Child>,
    command: String,
    args: Vec<String>,
    connection_info: ConnectionInfo,
    tools: ToolRegistry,
}

impl LocalConnection {
    pub fn new(command: String, args: Vec<String>, config: &crate::mcp::MCPConfig) -> Self {
        // Store command and args separately instead of joining them
        let connection_info = ConnectionInfo {
            connection_type: ConnectionType::Stdio,
            endpoint: format!("{} {}", command, args.join(" ")), // Only for display/logging
            timeout_ms: config.connection_timeout_ms,
        };

        Self {
            process: None,
            command,
            args,
            connection_info,
            tools: HashMap::new(),
        }
    }

    pub async fn connect(&mut self) -> MCPResult<()> {
        if self.command.is_empty() {
            return Err(MCPError::ConfigurationError(
                "Empty server command".to_string(),
            ));
        }

        debug!(
            "Connecting to MCP server: {} {}",
            self.command,
            self.args.join(" ")
        );

        let child = Command::new(&self.command)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| MCPError::ConnectionError(format!("Failed to spawn process: {}", e)))?;

        self.process = Some(child);
        debug!("MCP process '{}' spawned successfully", self.command);

        self.discover_tools().await?;
        debug!("Tool discovery completed for '{}'", self.command);

        Ok(())
    }

    pub async fn disconnect(&mut self) -> MCPResult<()> {
        if let Some(mut process) = self.process.take() {
            debug!(
                "Disconnecting MCP server: {} {}",
                self.command,
                self.args.join(" ")
            );

            // Attempt to kill the process
            if let Err(e) = process.kill().await {
                // Log warning but continue - process might have already exited
                warn!("Failed to kill MCP process '{}': {}", self.command, e);
            }

            // Wait for the process to exit to avoid zombie processes
            if let Err(e) = process.wait().await {
                // Log error but don't fail - we've done our best to clean up
                warn!(
                    "Error waiting for MCP process '{}' to exit: {}",
                    self.command, e
                );
            } else {
                debug!("MCP process '{}' terminated successfully", self.command);
            }
        }
        Ok(())
    }

    pub async fn send_request(&mut self, request: MCPRequest) -> MCPResult<MCPResponse> {
        if self.process.is_none() {
            return Err(MCPError::ConnectionError(
                "Not connected to MCP server".to_string(),
            ));
        }

        let _request_json = serde_json::to_string(&request)
            .map_err(|e| MCPError::ParseError(format!("Failed to serialize request: {}", e)))?;

        // Mock implementation - will be replaced in Phase 2
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
        // Mock tool discovery - will be replaced with real discovery in Phase 2
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

    pub fn get_connection_info(&self) -> &ConnectionInfo {
        &self.connection_info
    }
}
