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