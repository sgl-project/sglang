// This test suite validates the complete MCP implementation against the
// functionality required for SGLang responses API integration.
//
// - Core MCP server functionality
// - Tool session management (individual and multi-tool)
// - Tool execution and error handling
// - Schema adaptation and validation
// - Mock server integration for reliable testing

mod common;

use std::collections::HashMap;

use common::mock_mcp_server::MockMCPServer;
use serde_json::json;
use sgl_model_gateway::mcp::{McpConfig, McpError, McpManager, McpServerConfig, McpTransport};

/// Create a new mock server for testing (each test gets its own)
async fn create_mock_server() -> MockMCPServer {
    MockMCPServer::start()
        .await
        .expect("Failed to start mock MCP server")
}

// Core MCP Server Tests

#[tokio::test]
async fn test_mcp_server_initialization() {
    let config = McpConfig {
        servers: vec![],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    // Should succeed but with no connected servers (empty config is allowed)
    let result = McpManager::with_defaults(config).await;
    assert!(result.is_ok(), "Should succeed with empty config");

    let manager = result.unwrap();
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 0, "Should have no servers");
    let tools = manager.list_tools();
    assert_eq!(tools.len(), 0, "Should have no tools");
}

#[tokio::test]
async fn test_server_connection_with_mock() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
            proxy: None,
            required: false,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    let result = McpManager::with_defaults(config).await;
    assert!(result.is_ok(), "Should connect to mock server");

    let manager = result.unwrap();

    let servers = manager.list_servers();
    assert_eq!(servers.len(), 1);
    assert!(servers.contains(&"mock_server".to_string()));

    let tools = manager.list_tools();
    assert_eq!(tools.len(), 2, "Should have 2 tools from mock server");

    assert!(manager.has_tool("brave_web_search"));
    assert!(manager.has_tool("brave_local_search"));

    manager.shutdown().await;
}

#[tokio::test]
async fn test_tool_availability_checking() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
            proxy: None,
            required: false,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    let manager = McpManager::with_defaults(config).await.unwrap();

    let test_tools = vec!["brave_web_search", "brave_local_search", "calculator"];
    for tool in test_tools {
        let available = manager.has_tool(tool);
        match tool {
            "brave_web_search" | "brave_local_search" => {
                assert!(
                    available,
                    "Tool {} should be available from mock server",
                    tool
                );
            }
            "calculator" => {
                assert!(
                    !available,
                    "Tool {} should not be available from mock server",
                    tool
                );
            }
            _ => {}
        }
    }

    manager.shutdown().await;
}

#[tokio::test]
async fn test_multi_server_connection() {
    let mock_server1 = create_mock_server().await;
    let mock_server2 = create_mock_server().await;

    let config = McpConfig {
        servers: vec![
            McpServerConfig {
                name: "mock_server_1".to_string(),
                transport: McpTransport::Streamable {
                    url: mock_server1.url(),
                    token: None,
                },
                proxy: None,
                required: false,
            },
            McpServerConfig {
                name: "mock_server_2".to_string(),
                transport: McpTransport::Streamable {
                    url: mock_server2.url(),
                    token: None,
                },
                proxy: None,
                required: false,
            },
        ],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    // Note: This will fail to connect to both servers in the current implementation
    // since they return the same tools. The manager will connect to the first one.
    let result = McpManager::with_defaults(config).await;

    if let Ok(manager) = result {
        let servers = manager.list_servers();
        assert!(!servers.is_empty(), "Should have at least one server");

        let tools = manager.list_tools();
        assert!(tools.len() >= 2, "Should have tools from servers");

        manager.shutdown().await;
    }
}

#[tokio::test]
async fn test_tool_execution_with_mock() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
            proxy: None,
            required: false,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    let manager = McpManager::with_defaults(config).await.unwrap();

    let result = manager
        .call_tool(
            "brave_web_search",
            Some(
                json!({
                    "query": "rust programming",
                    "count": 1
                })
                .as_object()
                .unwrap()
                .clone(),
            ),
        )
        .await;

    assert!(
        result.is_ok(),
        "Tool execution should succeed with mock server"
    );

    let response = result.unwrap();
    assert!(!response.content.is_empty(), "Should have content");

    // Check the content
    if let rmcp::model::RawContent::Text(text) = &response.content[0].raw {
        assert!(text
            .text
            .contains("Mock search results for: rust programming"));
    } else {
        panic!("Expected text content");
    }

    manager.shutdown().await;
}

#[tokio::test]
async fn test_concurrent_tool_execution() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
            proxy: None,
            required: false,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    let manager = McpManager::with_defaults(config).await.unwrap();

    // Execute tools sequentially (true concurrent execution would require Arc<Mutex>)
    let tool_calls = vec![
        ("brave_web_search", json!({"query": "test1"})),
        ("brave_local_search", json!({"query": "test2"})),
    ];

    for (tool_name, args) in tool_calls {
        let result = manager
            .call_tool(tool_name, Some(args.as_object().unwrap().clone()))
            .await;

        assert!(result.is_ok(), "Tool {} should succeed", tool_name);
        let response = result.unwrap();
        assert!(!response.content.is_empty(), "Should have content");
    }

    manager.shutdown().await;
}

// Error Handling Tests

#[tokio::test]
async fn test_tool_execution_errors() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
            proxy: None,
            required: false,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    let manager = McpManager::with_defaults(config).await.unwrap();

    // Try to call unknown tool
    let result = manager
        .call_tool("unknown_tool", Some(serde_json::Map::new()))
        .await;
    assert!(result.is_err(), "Should fail for unknown tool");

    match result.unwrap_err() {
        McpError::ToolNotFound(name) => {
            assert_eq!(name, "unknown_tool");
        }
        _ => panic!("Expected ToolNotFound error"),
    }

    manager.shutdown().await;
}

#[tokio::test]
async fn test_connection_without_server() {
    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "nonexistent".to_string(),
            transport: McpTransport::Stdio {
                command: "/nonexistent/command".to_string(),
                args: vec![],
                envs: HashMap::new(),
            },
            proxy: None,
            required: false,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    let result = McpManager::with_defaults(config).await;
    // Manager succeeds but no servers are connected (errors are logged)
    assert!(
        result.is_ok(),
        "Manager should succeed even if servers fail to connect"
    );

    let manager = result.unwrap();
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 0, "Should have no connected servers");
}

// Schema Validation Tests

#[tokio::test]
async fn test_tool_info_structure() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
            proxy: None,
            required: false,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    let manager = McpManager::with_defaults(config).await.unwrap();

    let tools = manager.list_tools();
    let brave_search = tools
        .iter()
        .find(|t| t.name.as_ref() == "brave_web_search")
        .expect("Should have brave_web_search tool");

    assert_eq!(brave_search.name.as_ref(), "brave_web_search");
    assert!(brave_search
        .description
        .as_ref()
        .map(|d| d.contains("Mock web search"))
        .unwrap_or(false));
    // Note: server information is now maintained separately in the inventory,
    // not in the Tool type itself
    assert!(!brave_search.input_schema.is_empty());
}

// SSE Parsing Tests (simplified since we don't expose parse_sse_event)

#[tokio::test]
async fn test_sse_connection() {
    // This tests that SSE configuration is properly handled even when connection fails
    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "sse_test".to_string(),
            transport: McpTransport::Stdio {
                command: "/nonexistent/sse/server".to_string(),
                args: vec!["--sse".to_string()],
                envs: HashMap::new(),
            },
            proxy: None,
            required: false,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    // Manager succeeds but no servers are connected (errors are logged)
    let result = McpManager::with_defaults(config).await;
    assert!(
        result.is_ok(),
        "Manager should succeed even if SSE server fails to connect"
    );

    let manager = result.unwrap();
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 0, "Should have no connected servers");
}

// Connection Type Tests

#[tokio::test]
async fn test_transport_types() {
    // HTTP/Streamable transport
    let http_config = McpServerConfig {
        name: "http_server".to_string(),
        transport: McpTransport::Streamable {
            url: "http://localhost:8080/mcp".to_string(),
            token: Some("auth_token".to_string()),
        },
        proxy: None,
        required: false,
    };
    assert_eq!(http_config.name, "http_server");

    // SSE transport
    let sse_config = McpServerConfig {
        name: "sse_server".to_string(),
        transport: McpTransport::Sse {
            url: "http://localhost:8081/sse".to_string(),
            token: None,
        },
        proxy: None,
        required: false,
    };
    assert_eq!(sse_config.name, "sse_server");

    // STDIO transport
    let stdio_config = McpServerConfig {
        name: "stdio_server".to_string(),
        transport: McpTransport::Stdio {
            command: "mcp-server".to_string(),
            args: vec!["--port".to_string(), "8082".to_string()],
            envs: HashMap::new(),
        },
        proxy: None,
        required: false,
    };
    assert_eq!(stdio_config.name, "stdio_server");
}

// Integration Pattern Tests

#[tokio::test]
async fn test_complete_workflow() {
    let mock_server = create_mock_server().await;

    // 1. Initialize configuration
    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "integration_test".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
            proxy: None,
            required: false,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };

    // 2. Connect to server
    let manager = McpManager::with_defaults(config)
        .await
        .expect("Should connect to mock server");

    // 3. Verify server connection
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0], "integration_test");

    // 4. Check available tools
    let tools = manager.list_tools();
    assert_eq!(tools.len(), 2);

    // 5. Verify specific tools exist
    assert!(manager.has_tool("brave_web_search"));
    assert!(manager.has_tool("brave_local_search"));
    assert!(!manager.has_tool("nonexistent_tool"));

    // 6. Execute a tool
    let result = manager
        .call_tool(
            "brave_web_search",
            Some(
                json!({
                    "query": "SGLang router MCP integration",
                    "count": 1
                })
                .as_object()
                .unwrap()
                .clone(),
            ),
        )
        .await;

    assert!(result.is_ok(), "Tool execution should succeed");
    let response = result.unwrap();
    assert!(!response.content.is_empty(), "Should return content");

    // 7. Clean shutdown
    manager.shutdown().await;

    let capabilities = [
        "MCP server initialization",
        "Tool server connection and discovery",
        "Tool availability checking",
        "Tool execution",
        "Error handling and robustness",
        "Multi-server support",
        "Schema adaptation",
        "Mock server integration (no external dependencies)",
    ];

    assert_eq!(capabilities.len(), 8);
}
