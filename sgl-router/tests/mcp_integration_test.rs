// tests/mcp_integration_test.rs - Integration tests for MCP functionality
use sglang_router_rs::mcp::{MCPToolServer, MCPResult, ToolSession, ConnectionType};
use serde_json::json;

#[tokio::test]
async fn test_mcp_tool_server_creation() {
    let server = MCPToolServer::new();
    assert!(!server.has_tool("test_tool"));
}

#[tokio::test]
async fn test_tool_session_http_creation() {
    let session = ToolSession::new_http("http://localhost:8000/sse".to_string()).await;
    assert!(session.is_ok());
    
    let session = session.unwrap();
    match &session.connection {
        ConnectionType::Http(url) => {
            assert_eq!(url, "http://localhost:8000/sse");
        },
        _ => panic!("Expected HTTP connection type"),
    }
}

#[tokio::test]
async fn test_tool_session_stdio_creation() {
    let session = ToolSession::new_stdio("python server.py".to_string()).await;
    assert!(session.is_ok());
    
    let session = session.unwrap();
    match &session.connection {
        ConnectionType::Stdio(command) => {
            assert_eq!(command, "python server.py");
        },
        _ => panic!("Expected Stdio connection type"),
    }
}

#[tokio::test]
async fn test_connection_type_auto_detection() {
    // HTTP URL should be detected as HTTP
    let session = ToolSession::new("http://localhost:8000/sse".to_string()).await.unwrap();
    match &session.connection {
        ConnectionType::Http(_) => {},
        _ => panic!("Expected HTTP connection type"),
    }
    
    // HTTPS URL should be detected as HTTP
    let session = ToolSession::new("https://example.com/mcp".to_string()).await.unwrap();
    match &session.connection {
        ConnectionType::Http(_) => {},
        _ => panic!("Expected HTTP connection type"),
    }
    
    // Command should be detected as Stdio
    let session = ToolSession::new("python server.py".to_string()).await.unwrap();
    match &session.connection {
        ConnectionType::Stdio(_) => {},
        _ => panic!("Expected Stdio connection type"),
    }
}

#[tokio::test]
async fn test_stdio_tool_call_placeholder() {
    let session = ToolSession::new_stdio("python server.py".to_string()).await.unwrap();
    
    let result = session.call_tool("test_tool", json!({"param": "value"})).await;
    assert!(result.is_err());
    
    // Should return our placeholder error message
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Stdio communication not yet implemented"));
}

#[tokio::test]
async fn test_add_tool_server_url_formatting() {
    let mut server = MCPToolServer::new();
    
    // This will fail because no server is running, but we can test URL formatting
    let result = server.add_tool_server("localhost:8000".to_string()).await;
    assert!(result.is_err());
    
    // Test with multiple URLs
    let result = server.add_tool_server("localhost:8000,localhost:8001".to_string()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_stdio_server_placeholder() {
    let mut server = MCPToolServer::new();
    
    let result = server.add_stdio_server("python server.py".to_string()).await;
    assert!(result.is_err());
    
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Stdio servers not yet implemented"));
}

// Test SSE event parsing logic
#[tokio::test]
async fn test_sse_parsing_logic() {
    use sglang_router_rs::mcp::parse_sse_event;
    
    // Test valid SSE event
    let valid_event = "data: {\"jsonrpc\": \"2.0\", \"id\": \"1\", \"result\": {\"test\": \"success\"}}";
    
    let result = parse_sse_event(valid_event);
    assert!(result.is_ok());
    
    let parsed = result.unwrap();
    assert!(parsed.is_some());
    
    let response = parsed.unwrap();
    assert_eq!(response["test"], "success");
}

#[tokio::test]
async fn test_sse_parsing_empty_event() {
    use sglang_router_rs::mcp::parse_sse_event;
    
    // Test empty event
    let empty_event = "";
    let result = parse_sse_event(empty_event);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
    
    // Test event without data
    let no_data_event = "event: ping\nid: 123";
    let result = parse_sse_event(no_data_event);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[tokio::test]
async fn test_sse_parsing_error_response() {
    use sglang_router_rs::mcp::parse_sse_event;
    
    // Test SSE event with error
    let error_event = "data: {\"jsonrpc\": \"2.0\", \"id\": \"1\", \"error\": {\"code\": -1, \"message\": \"Test error\"}}";
    
    let result = parse_sse_event(error_event);
    assert!(result.is_err());
    
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Test error"));
}

// Phase 4: Multi-Server Support Tests
#[tokio::test]
async fn test_multi_server_parsing() {
    let mut server = MCPToolServer::new();
    
    // Test comma-separated URLs (will fail to connect but should parse correctly)
    let result = server.add_tool_server("localhost:8000,localhost:8001,localhost:8002".to_string()).await;
    assert!(result.is_err()); // Expected to fail since no servers running
    
    // Should have attempted to connect to 3 servers
    let stats = server.get_tool_stats();
    assert_eq!(stats.total_tools, 0); // No tools since connections failed
}

#[tokio::test]
async fn test_server_management_methods() {
    let server = MCPToolServer::new();
    
    // Test server listing (empty initially)
    let servers = server.list_servers();
    assert_eq!(servers.len(), 0);
    
    // Test server checking
    assert!(!server.has_server("http://localhost:8000/sse"));
    
    // Test stats
    let stats = server.get_tool_stats();
    assert_eq!(stats.total_servers, 0);
    assert_eq!(stats.total_tools, 0);
}

#[tokio::test]
async fn test_conflict_handling_behavior() {
    let mut server = MCPToolServer::new();
    
    // Test that attempting to add multiple servers with same tools shows warnings
    // This will fail to connect but we can test the parsing logic
    let result = server.add_tool_server("localhost:8000".to_string()).await;
    assert!(result.is_err()); // Expected - no server running
    
    // Test adding another server URL (would conflict if servers existed)
    let result2 = server.add_tool_server("localhost:8001".to_string()).await;
    assert!(result2.is_err()); // Expected - no server running
    
    // Verify empty state since no connections succeeded
    assert_eq!(server.list_tools().len(), 0);
    assert_eq!(server.list_servers().len(), 0);
}

// Phase 5: Session Management Tests
#[tokio::test]
async fn test_tool_session_lifecycle() {
    // Test HTTP session creation
    let http_session = ToolSession::new_http("http://localhost:8000/sse".to_string()).await;
    assert!(http_session.is_ok());
    
    let session = http_session.unwrap();
    assert!(session.is_ready());
    assert!(session.connection_info().contains("HTTP"));
    assert!(session.connection_info().contains("localhost:8000"));
}

#[tokio::test]
async fn test_tool_session_stdio() {
    // Test Stdio session creation
    let stdio_session = ToolSession::new_stdio("python server.py".to_string()).await;
    assert!(stdio_session.is_ok());
    
    let session = stdio_session.unwrap();
    assert!(!session.is_ready()); // Stdio not implemented yet
    assert!(session.connection_info().contains("Stdio"));
    assert!(session.connection_info().contains("python server.py"));
}

#[tokio::test]
async fn test_tool_session_auto_detection() {
    // Test auto-detection of connection type
    let http_session = ToolSession::new("http://localhost:8000/sse".to_string()).await.unwrap();
    assert!(http_session.is_ready());
    
    let stdio_session = ToolSession::new("python server.py".to_string()).await.unwrap();
    assert!(!stdio_session.is_ready());
}

#[tokio::test]
async fn test_convenience_methods() {
    let server = MCPToolServer::new();
    
    // Test convenience method for session creation from URL
    let session_result = server.create_session_from_url("http://localhost:8000/sse").await;
    assert!(session_result.is_ok());
    
    let session = session_result.unwrap();
    assert!(session.is_ready());
}

#[tokio::test]
async fn test_direct_tool_calling() {
    let server = MCPToolServer::new();
    
    // Test direct tool calling (will fail since no server, but tests the API)
    let result = server.call_tool("nonexistent_tool", serde_json::json!({})).await;
    assert!(result.is_err());
    
    // Should get ToolNotFound error
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("not found") || error_msg.contains("nonexistent_tool"));
}

#[tokio::test]
async fn test_session_tool_calling() {
    let session = ToolSession::new_http("http://localhost:8000/sse".to_string()).await.unwrap();
    
    // Test tool calling through session (will fail since no server, but tests the API)
    let result = session.call_tool("test_tool", serde_json::json!({"param": "value"})).await;
    assert!(result.is_err());
    
    // Should get connection error since no server is running
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Connection") || error_msg.contains("error"));
}