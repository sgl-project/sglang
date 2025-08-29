// tests/common/mock_mcp_server.rs - Mock MCP server for testing

use axum::{
    extract::Json, http::StatusCode, response::Json as ResponseJson, routing::post, Router,
};
use serde_json::{json, Value};
use tokio::net::TcpListener;

/// Mock MCP server that returns hardcoded responses for testing
pub struct MockMCPServer {
    pub port: u16,
    pub server_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MockMCPServer {
    /// Start a mock MCP server on an available port
    pub async fn start() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Find an available port
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();

        let app = Router::new().route("/mcp", post(handle_mcp_request));

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("Mock MCP server failed to start");
        });

        // Give the server a moment to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(MockMCPServer {
            port,
            server_handle: Some(server_handle),
        })
    }

    /// Get the full URL for this mock server
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.port)
    }

    /// Stop the mock server
    pub async fn stop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
            // Wait a moment for cleanup
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    }
}

impl Drop for MockMCPServer {
    fn drop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }
    }
}

/// Handle MCP requests and return mock responses
async fn handle_mcp_request(Json(request): Json<Value>) -> Result<ResponseJson<Value>, StatusCode> {
    // Parse the JSON-RPC request
    let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");

    let id = request
        .get("id")
        .and_then(|i| i.as_str())
        .unwrap_or("unknown");

    let response = match method {
        "initialize" => {
            // Mock initialize response
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "serverInfo": {
                        "name": "Mock MCP Server",
                        "version": "1.0.0"
                    },
                    "instructions": "Mock server for testing"
                }
            })
        }
        "tools/list" => {
            // Mock tools list response
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [
                        {
                            "name": "brave_web_search",
                            "description": "Mock web search tool",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "count": {"type": "integer"}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "brave_local_search",
                            "description": "Mock local search tool",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    ]
                }
            })
        }
        "tools/call" => {
            // Mock tool call response
            let empty_json = json!({});
            let params = request.get("params").unwrap_or(&empty_json);
            let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let empty_args = json!({});
            let arguments = params.get("arguments").unwrap_or(&empty_args);

            match tool_name {
                "brave_web_search" => {
                    let query = arguments
                        .get("query")
                        .and_then(|q| q.as_str())
                        .unwrap_or("test");
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": format!("Mock search results for: {}", query)
                                }
                            ],
                            "isError": false
                        }
                    })
                }
                "brave_local_search" => {
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Mock local search results"
                                }
                            ],
                            "isError": false
                        }
                    })
                }
                _ => {
                    // Unknown tool
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -1,
                            "message": format!("Unknown tool: {}", tool_name)
                        }
                    })
                }
            }
        }
        _ => {
            // Unknown method
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("Method not found: {}", method)
                }
            })
        }
    };

    Ok(ResponseJson(response))
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::MockMCPServer;
    use serde_json::{json, Value};

    #[tokio::test]
    async fn test_mock_server_startup() {
        let mut server = MockMCPServer::start().await.unwrap();
        assert!(server.port > 0);
        assert!(server.url().contains(&server.port.to_string()));
        server.stop().await;
    }

    #[tokio::test]
    async fn test_mock_server_responses() {
        let mut server = MockMCPServer::start().await.unwrap();
        let client = reqwest::Client::new();

        // Test initialize
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": "1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {}
            }
        });

        let response = client
            .post(server.url())
            .json(&init_request)
            .send()
            .await
            .unwrap();

        assert!(response.status().is_success());
        let json: Value = response.json().await.unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["result"]["serverInfo"]["name"], "Mock MCP Server");

        server.stop().await;
    }
}
