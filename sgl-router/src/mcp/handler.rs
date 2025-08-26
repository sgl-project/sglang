use crate::mcp::connection::LocalConnection;
use crate::mcp::executor::SimpleExecutor;
use crate::mcp::types::{ToolCall, ToolRegistry, ToolResult};
use crate::mcp::{MCPConfig, MCPError, MCPResult};
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct MCPToolHandler {
    executor: SimpleExecutor,
    config: MCPConfig,
    connections: RwLock<HashMap<String, LocalConnection>>,
}

impl MCPToolHandler {
    pub async fn new_dev_mode() -> MCPResult<Self> {
        let config = MCPConfig::dev_mode();
        Ok(Self {
            executor: SimpleExecutor::new(config.execution_timeout()),
            config,
            connections: RwLock::new(HashMap::new()),
        })
    }

    pub async fn new_with_config(config: MCPConfig) -> MCPResult<Self> {
        Ok(Self {
            executor: SimpleExecutor::new(config.execution_timeout()),
            config,
            connections: RwLock::new(HashMap::new()),
        })
    }

    pub async fn add_local_server(
        &self,
        server_id: String,
        command: String,
        args: Vec<String>,
    ) -> MCPResult<()> {
        let mut connection = LocalConnection::new(command, args, &self.config);
        connection.connect().await?;

        let mut connections = self.connections.write().await;
        connections.insert(server_id, connection);

        Ok(())
    }

    pub async fn execute_tool(&self, tool_call: ToolCall) -> MCPResult<ToolResult> {
        self.validate_tool_call(&tool_call).await?;

        // Parse the tool name to extract server_id and tool_name
        let (server_id, _tool_name) = self.parse_tool_name(&tool_call.name)?;

        // Get the connection for this server
        let mut connections = self.connections.write().await;
        let connection = connections
            .get_mut(&server_id)
            .ok_or_else(|| {
                MCPError::ConnectionError(format!("Server '{}' not found", server_id))
            })?;

        // Execute the tool using the connection
        self.executor.execute_tool(tool_call, connection).await
    }

    pub async fn get_available_tools(&self) -> MCPResult<ToolRegistry> {
        let mut all_tools = ToolRegistry::new();

        let connections = self.connections.read().await;
        for (server_id, connection) in connections.iter() {
            let tools = connection.get_available_tools();
            for (tool_name, tool_meta) in tools.iter() {
                let qualified_name = format!("{}:{}", server_id, tool_name);
                all_tools.insert(qualified_name, tool_meta.clone());
            }
        }

        Ok(all_tools)
    }

    pub async fn remove_server(&self, server_id: &str) -> MCPResult<()> {
        let mut connections = self.connections.write().await;

        if let Some(mut connection) = connections.remove(server_id) {
            connection.disconnect().await?;
        }

        Ok(())
    }

    pub async fn health_check(&self) -> MCPResult<HashMap<String, bool>> {
        let mut health_status = HashMap::new();

        let connections = self.connections.read().await;
        for (server_id, connection) in connections.iter() {
            health_status.insert(server_id.clone(), connection.is_connected());
        }

        Ok(health_status)
    }

    pub fn get_config(&self) -> &MCPConfig {
        &self.config
    }

    async fn validate_tool_call(&self, tool_call: &ToolCall) -> MCPResult<()> {
        if tool_call.name.is_empty() {
            return Err(MCPError::ValidationError(
                "Tool name cannot be empty".to_string(),
            ));
        }

        let available_tools = self.get_available_tools().await?;
        if !available_tools.contains_key(&tool_call.name) {
            return Err(MCPError::ValidationError(format!(
                "Tool '{}' not found",
                tool_call.name
            )));
        }

        Ok(())
    }

    fn parse_tool_name(&self, qualified_name: &str) -> MCPResult<(String, String)> {
        let parts: Vec<&str> = qualified_name.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(MCPError::ValidationError(format!(
                "Invalid tool name format '{}'. Expected 'server:tool'",
                qualified_name
            )));
        }
        Ok((parts[0].to_string(), parts[1].to_string()))
    }
}
