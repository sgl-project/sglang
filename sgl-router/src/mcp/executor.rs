use crate::mcp::connection::LocalConnection;
use crate::mcp::types::{MCPRequest, ToolCall, ToolResult};
use crate::mcp::{MCPError, MCPResult};
use std::time::{Duration, Instant};
use tokio::time::timeout;

pub struct SimpleExecutor {
    timeout: Duration,
}

impl SimpleExecutor {
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }

    pub async fn execute_tool(
        &self,
        tool_call: ToolCall,
        tool_name: &str,
        connection: &mut LocalConnection,
    ) -> MCPResult<ToolResult> {
        let start_time = Instant::now();

        let result = timeout(
            self.timeout,
            self.execute_tool_internal(tool_call.clone(), tool_name, connection),
        )
        .await
        .map_err(|_| {
            MCPError::TimeoutError(format!(
                "Tool '{}' timed out after {:?}",
                tool_call.name, self.timeout
            ))
        })?;

        let execution_time = start_time.elapsed();

        match result {
            Ok(value) => Ok(ToolResult {
                success: true,
                result: Some(value),
                error: None,
                execution_time_ms: execution_time.as_millis() as u64,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                result: None,
                error: Some(e.to_string()),
                execution_time_ms: execution_time.as_millis() as u64,
            }),
        }
    }

    async fn execute_tool_internal(
        &self,
        tool_call: ToolCall,
        tool_name: &str,
        connection: &mut LocalConnection,
    ) -> MCPResult<serde_json::Value> {
        let request = MCPRequest {
            jsonrpc: "2.0".to_string(),
            id: uuid::Uuid::new_v4().to_string(),
            method: format!("tools/{}", tool_name),
            params: Some(tool_call.arguments),
        };

        // Use the connection to send the request
        let response = connection.send_request(request).await?;

        // Extract the result from the response
        response.result.ok_or_else(|| {
            MCPError::ExecutionError(
                response
                    .error
                    .map(|e| e.message)
                    .unwrap_or_else(|| "Empty response from tool".to_string()),
            )
        })
    }
}
