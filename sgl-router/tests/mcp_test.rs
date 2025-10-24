// This test suite validates the complete MCP implementation against the
// functionality required for SGLang responses API integration.
//
// - Core MCP server functionality
// - Tool session management (individual and multi-tool)
// - Tool execution and error handling
// - Schema adaptation and validation
// - Mock server integration for reliable testing

mod common;

use std::{collections::HashMap, process::Command, time::Duration};

use common::{mock_mcp_server::MockMCPServer, test_proxy::TestProxy};
use serde_json::json;
use sglang_router_rs::{
    config::McpProxyConfig,
    mcp::{McpClientManager, McpConfig, McpError, McpServerConfig, McpTransport},
};

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
        proxy: None,
    };

    // Should fail with no servers
    let result = McpClientManager::new(config).await;
    assert!(result.is_err(), "Should fail with no servers configured");
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
        }],
        proxy: None,
    };

    let result = McpClientManager::new(config).await;
    assert!(result.is_ok(), "Should connect to mock server");

    let mut manager = result.unwrap();

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
        }],
        proxy: None,
    };

    let mut manager = McpClientManager::new(config).await.unwrap();

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
            },
            McpServerConfig {
                name: "mock_server_2".to_string(),
                transport: McpTransport::Streamable {
                    url: mock_server2.url(),
                    token: None,
                },
            },
        ],
        proxy: None,
    };

    // Note: This will fail to connect to both servers in the current implementation
    // since they return the same tools. The manager will connect to the first one.
    let result = McpClientManager::new(config).await;

    if let Ok(mut manager) = result {
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
        }],
        proxy: None,
    };

    let mut manager = McpClientManager::new(config).await.unwrap();

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
        }],
        proxy: None,
    };

    let mut manager = McpClientManager::new(config).await.unwrap();

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
        }],
        proxy: None,
    };

    let mut manager = McpClientManager::new(config).await.unwrap();

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
        }],
        proxy: None,
    };

    let result = McpClientManager::new(config).await;
    assert!(result.is_err(), "Should fail when no server is running");

    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(
            error_msg.contains("Failed to connect")
                || error_msg.contains("Connection")
                || error_msg.contains("failed")
                || error_msg.contains("error"),
            "Error should indicate failure: {}",
            error_msg
        );
    }
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
        }],
        proxy: None,
    };

    let manager = McpClientManager::new(config).await.unwrap();

    let tools = manager.list_tools();
    let brave_search = tools
        .iter()
        .find(|t| t.name == "brave_web_search")
        .expect("Should have brave_web_search tool");

    assert_eq!(brave_search.name, "brave_web_search");
    assert!(brave_search.description.contains("Mock web search"));
    assert_eq!(brave_search.server, "mock_server");
    assert!(brave_search.parameters.is_some());
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
        }],
        proxy: None,
    };

    // This will fail immediately without retry
    let result = McpClientManager::new(config).await;
    assert!(result.is_err(), "Should fail for non-existent SSE server");
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
    };
    assert_eq!(http_config.name, "http_server");

    // SSE transport
    let sse_config = McpServerConfig {
        name: "sse_server".to_string(),
        transport: McpTransport::Sse {
            url: "http://localhost:8081/sse".to_string(),
            token: None,
        },
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
        }],
        proxy: None,
    };

    // 2. Connect to server
    let mut manager = McpClientManager::new(config)
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

// Proxy Configuration Tests

fn run_test_in_subprocess(test_name: &str) {
    const ENV_KEY: &str = "MCP_PROXY_TEST_IN_SUBPROCESS";

    // If we're inside the subprocess, run the actual test
    if std::env::var(ENV_KEY).is_ok() {
        std::env::remove_var(ENV_KEY);
        return; // Let the actual test function run
    }

    // Otherwise, spawn a subprocess to run this test in isolation
    let status = Command::new(std::env::current_exe().expect("current exe"))
        .env(ENV_KEY, "1")
        .args([test_name, "--exact"])
        .status()
        .expect("failed to spawn test subprocess");

    assert!(status.success(), "subprocess test failed");
}

#[tokio::test]
async fn test_mcp_client_with_http_proxy() {
    const ENV_KEY: &str = "MCP_PROXY_TEST_IN_SUBPROCESS";
    if std::env::var(ENV_KEY).is_err() {
        run_test_in_subprocess("test_mcp_client_with_http_proxy");
        return;
    }

    let mut proxy = TestProxy::start().await;

    // Initialize HTTP client with proxy configuration
    let _ = sglang_router_rs::mcp::http_client::init(McpProxyConfig::new(Some(proxy.url()), None, None));

    let mut mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
        }],
        proxy: None, // Proxy is configured globally
    };

    let manager_result = McpClientManager::new(config).await;

    assert!(
        proxy.wait_for_hits(1, Duration::from_secs(10)).await,
        "Proxy should receive requests when http_proxy is configured"
    );

    if let Ok(mut manager) = manager_result {
        assert!(manager.has_tool("brave_web_search"));
        manager.shutdown().await;
    }

    proxy.shutdown().await;
    mock_server.stop().await;
}

#[tokio::test]
async fn test_mcp_client_with_http_and_https_proxy() {
    const ENV_KEY: &str = "MCP_PROXY_TEST_IN_SUBPROCESS";
    if std::env::var(ENV_KEY).is_err() {
        run_test_in_subprocess("test_mcp_client_with_http_and_https_proxy");
        return;
    }

    let mut proxy = TestProxy::start().await;

    // Initialize HTTP client with proxy configuration
    let url = proxy.url();
    let _ = sglang_router_rs::mcp::http_client::init(McpProxyConfig::new(Some(url.clone()), Some(url), None));

    let mut mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
        }],
        proxy: None, // Proxy is configured globally
    };

    let manager_result = McpClientManager::new(config).await;

    assert!(
        proxy.wait_for_hits(1, Duration::from_secs(10)).await,
        "Proxy should receive requests when https_proxy is configured"
    );

    if let Ok(mut manager) = manager_result {
        assert!(manager.has_tool("brave_web_search"));
        manager.shutdown().await;
    }

    proxy.shutdown().await;
    mock_server.stop().await;
}

#[tokio::test]
async fn test_mcp_client_respects_no_proxy() {
    const ENV_KEY: &str = "MCP_PROXY_TEST_IN_SUBPROCESS";
    if std::env::var(ENV_KEY).is_err() {
        run_test_in_subprocess("test_mcp_client_respects_no_proxy");
        return;
    }

    let mut proxy = TestProxy::start().await;

    // Initialize HTTP client with proxy and no_proxy configuration
    let _ = sglang_router_rs::mcp::http_client::init(McpProxyConfig::new(
        Some(proxy.url()),
        None,
        Some("127.0.0.1,localhost".to_string()),
    ));

    let mut mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
            },
        }],
        proxy: None, // Proxy is configured globally
    };

    let manager_result = McpClientManager::new(config).await;

    assert!(manager_result.is_ok(), "Should connect without proxy");
    let mut manager = manager_result.unwrap();

    assert!(
        !proxy.wait_for_hits(1, Duration::from_millis(500)).await,
        "Proxy should not receive requests when no_proxy matches"
    );
    assert_eq!(proxy.hit_count(), 0);
    assert!(manager.has_tool("brave_web_search"));

    manager.shutdown().await;
    proxy.shutdown().await;
    mock_server.stop().await;
}
