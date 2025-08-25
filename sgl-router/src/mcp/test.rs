use crate::mcp::types::ToolCall;
use crate::mcp::{MCPConfig, MCPError, MCPToolHandler};

#[tokio::test]
async fn test_mcp_handler_creation() {
    let handler = MCPToolHandler::new_dev_mode().await;
    assert!(handler.is_ok());

    let handler = handler.unwrap();
    assert_eq!(handler.get_config().connection_timeout_ms, 5000);
    assert_eq!(handler.get_config().execution_timeout_ms, 30000);
}

#[tokio::test]
async fn test_custom_config() {
    let config = MCPConfig::production();
    let handler = MCPToolHandler::new_with_config(config).await;
    assert!(handler.is_ok());

    let handler = handler.unwrap();
    assert_eq!(handler.get_config().connection_timeout_ms, 10000);
    assert_eq!(handler.get_config().execution_timeout_ms, 60000);
}

#[tokio::test]
async fn test_local_server_management() {
    let handler = MCPToolHandler::new_dev_mode().await.unwrap();

    let result = handler
        .add_local_server(
            "test-server".to_string(),
            "echo".to_string(),
            vec!["hello".to_string()],
        )
        .await;
    assert!(result.is_ok());

    let health = handler.health_check().await.unwrap();
    assert!(health.contains_key("test-server"));

    let result = handler.remove_server("test-server").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_tool_execution_mock() {
    let handler = MCPToolHandler::new_dev_mode().await.unwrap();

    let result = handler
        .add_local_server("test-server".to_string(), "echo".to_string(), vec![])
        .await;
    assert!(result.is_ok());

    let tool_call = ToolCall {
        name: "test-server:file_read".to_string(),
        arguments: serde_json::json!({"path": "/tmp/test.txt"}),
    };

    let result = handler.execute_tool(tool_call).await;
    assert!(result.is_ok());

    let tool_result = result.unwrap();
    assert!(tool_result.success);
    assert!(tool_result.result.is_some());
    assert!(tool_result.execution_time_ms > 0);
}

#[tokio::test]
async fn test_get_available_tools() {
    let handler = MCPToolHandler::new_dev_mode().await.unwrap();

    let result = handler
        .add_local_server("test-server".to_string(), "echo".to_string(), vec![])
        .await;
    assert!(result.is_ok());

    let tools = handler.get_available_tools().await.unwrap();
    assert!(tools.contains_key("test-server:file_read"));
    assert!(tools.contains_key("test-server:web_search"));

    assert_eq!(tools.len(), 2);
}

#[tokio::test]
async fn test_basic_usage_workflow() {
    // Create handler
    let handler = MCPToolHandler::new_dev_mode().await;
    assert!(handler.is_ok(), "Failed to create MCPToolHandler");
    let handler = handler.unwrap();

    // Verify initial state
    let initial_health = handler.health_check().await.unwrap();
    assert!(
        initial_health.is_empty(),
        "Handler should start with no servers"
    );

    // Add a local MCP server (using 'echo' as a mock command that exists)
    let add_result = handler
        .add_local_server(
            "my-server".to_string(),
            "echo".to_string(),
            vec!["mock_mcp_server".to_string()],
        )
        .await;
    assert!(
        add_result.is_ok(),
        "Failed to add local MCP server: {:?}",
        add_result.err()
    );

    // Verify server was added
    let health = handler.health_check().await.unwrap();
    assert!(
        health.contains_key("my-server"),
        "Server 'my-server' not found in health check"
    );
    assert_eq!(health.len(), 1, "Should have exactly one server");

    // Check available tools after adding server
    let tools = handler.get_available_tools().await.unwrap();
    assert!(
        tools.contains_key("my-server:file_read"),
        "file_read tool not found"
    );
    assert!(
        tools.contains_key("my-server:web_search"),
        "web_search tool not found"
    );
    assert_eq!(tools.len(), 2, "Should have exactly 2 tools");

    // Execute a tool
    let result = handler
        .execute_tool(ToolCall {
            name: "my-server:file_read".to_string(),
            arguments: serde_json::json!({"path": "/tmp/file.txt"}),
        })
        .await;

    assert!(result.is_ok(), "Tool execution failed: {:?}", result.err());
    let tool_result = result.unwrap();

    // Verify tool execution result
    assert!(tool_result.success, "Tool execution should succeed");
    assert!(
        tool_result.result.is_some(),
        "Tool result should contain data"
    );
    assert!(
        tool_result.error.is_none(),
        "Tool execution should not have errors"
    );
    assert!(
        tool_result.execution_time_ms > 0,
        "Execution time should be recorded"
    );
    assert!(
        tool_result.execution_time_ms < 1000,
        "Mock execution should be fast (<1s)"
    );

    // Verify the result content
    let result_data = tool_result.result.unwrap();
    assert!(result_data.is_object(), "Result should be a JSON object");

    // Check health again
    let final_health = handler.health_check().await.unwrap();
    assert!(
        final_health.contains_key("my-server"),
        "Server should still be healthy"
    );
    assert!(
        *final_health.get("my-server").unwrap(),
        "Server should be connected"
    );

    // Clean up - remove server
    let remove_result = handler.remove_server("my-server").await;
    assert!(remove_result.is_ok(), "Failed to remove server");

    // Verify server was removed
    let cleanup_health = handler.health_check().await.unwrap();
    assert!(
        !cleanup_health.contains_key("my-server"),
        "Server should be removed from health check"
    );
    assert!(
        cleanup_health.is_empty(),
        "No servers should remain after cleanup"
    );
}

#[tokio::test]
async fn test_invalid_tool_execution() {
    let handler = MCPToolHandler::new_dev_mode().await.unwrap();

    // Add server
    handler
        .add_local_server(
            "my-server".to_string(),
            "echo".to_string(),
            vec!["mock_mcp_server".to_string()],
        )
        .await
        .unwrap();

    // Try to execute non-existent tool
    let result = handler
        .execute_tool(ToolCall {
            name: "my-server:nonexistent_tool".to_string(),
            arguments: serde_json::json!({}),
        })
        .await;

    assert!(result.is_err(), "Should fail for non-existent tool");
    match result.unwrap_err() {
        MCPError::ValidationError(msg) => {
            assert!(
                msg.contains("not found"),
                "Error should mention tool not found"
            );
        }
        other => panic!("Expected ValidationError, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_empty_tool_name_validation() {
    let handler = MCPToolHandler::new_dev_mode().await.unwrap();

    // Try to execute tool with empty name
    let result = handler
        .execute_tool(ToolCall {
            name: "".to_string(),
            arguments: serde_json::json!({}),
        })
        .await;

    assert!(result.is_err(), "Should fail for empty tool name");
    match result.unwrap_err() {
        MCPError::ValidationError(msg) => {
            assert!(
                msg.contains("cannot be empty"),
                "Error should mention empty tool name"
            );
        }
        other => panic!("Expected ValidationError, got: {:?}", other),
    }
}
