//! Shared MCP utilities for routers.
//!
//! This module provides shared MCP-related functionality that can be
//! used across different router implementations (OpenAI, gRPC regular, gRPC harmony).

use std::{collections::HashSet, sync::Arc};

use rmcp::model::CallToolResult;
use smg_mcp::{
    error::{McpError, McpResult},
    tool_args::ToolArgs,
    McpConfig, McpManager, McpServerConfig, McpTransport, Tool,
};
use tracing::warn;

use crate::protocols::responses::{ResponseTool, ResponseToolType};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum tool loop iterations (safety limit).
///
/// Used as fallback when user doesn't specify `max_tool_calls`.
/// All routers use this same value.
pub const DEFAULT_MAX_ITERATIONS: usize = 10;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for MCP tool calling loops.
///
/// Provides a common structure for loop configuration across routers.
#[derive(Debug, Clone)]
pub struct McpLoopConfig {
    /// Maximum iterations as safety limit (default: DEFAULT_MAX_ITERATIONS).
    /// Prevents infinite loops when max_tool_calls is not set by user.
    pub max_iterations: usize,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

// ============================================================================
// Request-scoped MCP tools
// ============================================================================

/// The MCP tools one `/v1/responses` request may list and call.
///
/// Static servers (registered from the gateway's MCP config) are shared through
/// the process-wide [`McpManager`]. Servers named in the request body
/// (`server_url` + `authorization`) are connected through a manager owned by
/// this request, so their sessions, credentials and tools are never visible to
/// another request. A request tool shadows a static tool with the same name
/// within this request only.
pub struct RequestMcpTools {
    shared: Arc<McpManager>,
    request: Option<McpManager>,
}

impl RequestMcpTools {
    /// Tools of the static servers only (the request named no MCP servers).
    pub fn static_only(shared: Arc<McpManager>) -> Self {
        Self {
            shared,
            request: None,
        }
    }

    /// All tools visible to this request: its own servers' tools, then static tools.
    pub fn list_tools(&self) -> Vec<Tool> {
        let mut tools = self
            .request
            .as_ref()
            .map(McpManager::list_tools)
            .unwrap_or_default();
        let shadowed: HashSet<String> = tools.iter().map(|t| t.name.to_string()).collect();
        tools.extend(
            self.shared
                .list_tools_for_servers(&[])
                .into_iter()
                .filter(|t| !shadowed.contains(t.name.as_ref())),
        );
        tools
    }

    /// Call a tool from this request's servers or from a static server.
    pub async fn call_tool(
        &self,
        tool_name: &str,
        args: impl Into<ToolArgs>,
    ) -> McpResult<CallToolResult> {
        if let Some(request) = &self.request {
            if request.has_tool(tool_name) {
                return request.call_tool(tool_name, args).await;
            }
        }
        match self.shared.inventory().get_tool(tool_name) {
            Some((server_name, _)) if self.shared.is_static_server(&server_name) => {
                self.shared.call_tool(tool_name, args).await
            }
            _ => Err(McpError::ToolNotFound(tool_name.to_string())),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract MCP server label from request tools.
///
/// Searches for the first MCP tool in the tools array and returns its server_label.
/// Falls back to a default value if no MCP tool with server_label is found.
pub fn extract_server_label(tools: Option<&[ResponseTool]>, default_label: &str) -> String {
    tools
        .and_then(|tools| {
            tools.iter().find_map(|tool| {
                if matches!(tool.r#type, ResponseToolType::Mcp) {
                    tool.server_label.clone()
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| default_label.to_string())
}

// ============================================================================
// MCP Connection
// ============================================================================

/// Ensure MCP clients are connected for all request-level MCP tools.
///
/// This function extracts MCP server configurations from ALL request tools (server_url, authorization)
/// and connects them through an `McpManager` owned by this request. The shared manager's pool is
/// keyed by URL only, so a session (and the credential it was opened with) pooled there would be
/// reused by any later request naming the same URL, and its tools would land in the global
/// inventory. The request's connections close when the returned value is dropped.
///
/// Returns `Some(tools)` if MCP tools were found and at least one client connected,
/// `None` if no MCP tools with server_url were found or none could be connected.
pub async fn ensure_request_mcp_client(
    mcp_manager: &Arc<McpManager>,
    tools: &[ResponseTool],
) -> Option<Arc<RequestMcpTools>> {
    let request_config = McpConfig {
        servers: Vec::new(),
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
    };
    let request_manager = match McpManager::new(request_config, tools.len().max(1)).await {
        Ok(manager) => manager,
        Err(err) => {
            warn!("Failed to create request MCP manager: {}", err);
            return None;
        }
    };
    let mut connected = false;
    let mut has_mcp_tools = false;

    // Process all MCP tools
    for tool in tools {
        if matches!(tool.r#type, ResponseToolType::Mcp) && tool.server_url.is_some() {
            has_mcp_tools = true;
            let Some(server_url) = tool.server_url.as_ref().map(|s| s.trim().to_string()) else {
                continue;
            };

            // Validate URL scheme
            if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
                warn!(
                    "Ignoring MCP server_url with unsupported scheme: {}",
                    server_url
                );
                continue;
            }

            // Extract server label and auth token
            let name = tool
                .server_label
                .clone()
                .unwrap_or_else(|| "request-mcp".to_string());
            let token = tool.authorization.clone();

            // Determine transport type based on URL pattern
            let transport = if server_url.contains("/sse") {
                McpTransport::Sse {
                    url: server_url.clone(),
                    token,
                }
            } else {
                McpTransport::Streamable {
                    url: server_url.clone(),
                    token,
                }
            };

            // Create server config
            let server_config = McpServerConfig {
                name,
                transport,
                proxy: None,
                required: false,
            };

            // Get the server key for logging
            let server_key = McpManager::server_key(&server_config);

            // Connect through this request's manager, never the shared one
            match request_manager.get_or_create_client(server_config).await {
                Ok(_client) => connected = true,
                Err(err) => {
                    warn!(
                        "Failed to get/create MCP connection for {}: {}",
                        server_key, err
                    );
                    // Continue processing other tools
                }
            }
        }
    }

    if has_mcp_tools && connected {
        Some(Arc::new(RequestMcpTools {
            shared: Arc::clone(mcp_manager),
            request: Some(request_manager),
        }))
    } else {
        None
    }
}

#[cfg(test)]
mod request_mcp_credential_tests {
    use std::sync::Arc;

    use axum::http::request::Parts;
    use rmcp::{
        handler::server::{router::tool::ToolRouter, tool::Extension},
        model::{
            CallToolResult, Content, Implementation, ProtocolVersion, ServerCapabilities,
            ServerInfo,
        },
        tool, tool_handler, tool_router,
        transport::streamable_http_server::{
            session::local::LocalSessionManager, StreamableHttpService,
        },
        ErrorData, ServerHandler,
    };
    use smg_mcp::{McpConfig, McpManager, McpServerConfig, McpTransport};

    use super::{ensure_request_mcp_client, RequestMcpTools};
    use crate::protocols::responses::ResponseTool;

    // All tokens and notes are synthetic.
    const TOKEN_A: &str = "SYNTHETIC_TENANT_A_TOKEN";
    const TOKEN_B: &str = "SYNTHETIC_TENANT_B_TOKEN";
    const TOKEN_OPERATOR: &str = "SYNTHETIC_OPERATOR_TOKEN";
    const NOTES_A: &str = "owner=tenant-a notes=SYNTHETIC_TENANT_A_SECRET";
    const NOTES_B: &str = "owner=tenant-b notes=(empty)";
    const NOTES_OPERATOR: &str = "owner=operator";
    const NOTES_NONE: &str = "owner=none";

    /// What a per-user MCP server returns for the bearer token on the HTTP request.
    fn notes_for_caller(parts: &Parts) -> CallToolResult {
        let mut caller = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        // smg-mcp passes "Bearer <token>" to rmcp, which adds the scheme again.
        while let Some(rest) = caller.strip_prefix("Bearer ") {
            caller = rest.to_string();
        }
        let notes = match caller.as_str() {
            TOKEN_A => NOTES_A,
            TOKEN_B => NOTES_B,
            TOKEN_OPERATOR => NOTES_OPERATOR,
            _ => NOTES_NONE,
        };
        CallToolResult::success(vec![Content::text(notes)])
    }

    fn server_info() -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: None,
        }
    }

    /// A per-user notes service that a request names in its `tools`.
    #[derive(Clone)]
    struct NotesServer {
        tool_router: ToolRouter<NotesServer>,
    }

    #[tool_router]
    impl NotesServer {
        fn new() -> Self {
            Self {
                tool_router: Self::tool_router(),
            }
        }

        #[tool(description = "Return the private notes of the calling user")]
        fn read_private_notes(
            &self,
            Extension(parts): Extension<Parts>,
        ) -> Result<CallToolResult, ErrorData> {
            Ok(notes_for_caller(&parts))
        }
    }

    #[tool_handler]
    impl ServerHandler for NotesServer {
        fn get_info(&self) -> ServerInfo {
            server_info()
        }
    }

    /// A static server from the gateway's MCP config. It has a tool with the same
    /// name as `NotesServer` and one of its own.
    #[derive(Clone)]
    struct OperatorServer {
        tool_router: ToolRouter<OperatorServer>,
    }

    #[tool_router]
    impl OperatorServer {
        fn new() -> Self {
            Self {
                tool_router: Self::tool_router(),
            }
        }

        #[tool(description = "Return the private notes of the calling user")]
        fn read_private_notes(
            &self,
            Extension(parts): Extension<Parts>,
        ) -> Result<CallToolResult, ErrorData> {
            Ok(notes_for_caller(&parts))
        }

        #[tool(description = "Return the principal of the calling user")]
        fn operator_whoami(
            &self,
            Extension(parts): Extension<Parts>,
        ) -> Result<CallToolResult, ErrorData> {
            Ok(notes_for_caller(&parts))
        }
    }

    #[tool_handler]
    impl ServerHandler for OperatorServer {
        fn get_info(&self) -> ServerInfo {
            server_info()
        }
    }

    async fn serve<S: ServerHandler + Clone + Send + 'static>(new_server: fn() -> S) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let service = StreamableHttpService::new(
            move || Ok(new_server()),
            LocalSessionManager::default().into(),
            Default::default(),
        );
        tokio::spawn(async move {
            axum::serve(listener, axum::Router::new().nest_service("/mcp", service)).await
        });
        url
    }

    /// The process-wide manager, built as `AppContext::with_mcp_manager` does.
    async fn gateway_mcp() -> Arc<McpManager> {
        Arc::new(
            McpManager::with_defaults(McpConfig {
                servers: Vec::new(),
                pool: Default::default(),
                proxy: None,
                warmup: Vec::new(),
                inventory: Default::default(),
            })
            .await
            .unwrap(),
        )
    }

    /// Register a static server the way the `mcp_registration` workflow does.
    async fn register_static_server(gateway: &Arc<McpManager>, name: &str, url: &str) {
        let config = McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::Streamable {
                url: url.to_string(),
                token: Some(TOKEN_OPERATOR.to_string()),
            },
            proxy: None,
            required: false,
        };
        let client = Arc::new(McpManager::connect_server(&config, None).await.unwrap());
        McpManager::load_server_inventory(&gateway.inventory(), name, &client).await;
        gateway.register_static_server(name.to_string(), client);
    }

    /// The `tools` entry of a `/v1/responses` request body.
    fn request_tools(url: &str, authorization: Option<&str>) -> Vec<ResponseTool> {
        vec![serde_json::from_value(serde_json::json!({
            "type": "mcp",
            "server_label": "notes",
            "server_url": url,
            "authorization": authorization,
        }))
        .unwrap()]
    }

    async fn call(mcp: &RequestMcpTools, tool_name: &str) -> String {
        let result = mcp.call_tool(tool_name, "{}").await.unwrap();
        serde_json::to_value(&result.content).unwrap()[0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn tool_names(mcp: &RequestMcpTools) -> Vec<String> {
        let mut names: Vec<String> = mcp
            .list_tools()
            .iter()
            .map(|t| t.name.to_string())
            .collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn test_request_mcp_session_is_not_shared_across_credentials() {
        let url = serve(NotesServer::new).await;
        let gateway = gateway_mcp().await;

        // Three /v1/responses requests from different tenants name the same MCP server.
        for (authorization, expected) in [
            (Some(TOKEN_A), NOTES_A),
            (Some(TOKEN_B), NOTES_B),
            (None, NOTES_NONE),
        ] {
            let mcp = ensure_request_mcp_client(&gateway, &request_tools(&url, authorization))
                .await
                .unwrap();
            assert_eq!(call(&mcp, "read_private_notes").await, expected);
        }

        // Request-supplied servers never enter the process-wide pool or inventory.
        assert_eq!(gateway.connection_pool().len(), 0);
        assert!(gateway.list_tools().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_requests_to_same_server_keep_their_credentials() {
        let url = serve(NotesServer::new).await;
        let gateway = gateway_mcp().await;
        let cases = [
            (Some(TOKEN_A), NOTES_A),
            (Some(TOKEN_B), NOTES_B),
            (None, NOTES_NONE),
        ];

        // Twelve requests connect concurrently; all are alive before any calls the tool.
        let expected: Vec<&str> = (0..12).map(|i| cases[i % cases.len()].1).collect();
        let requests = futures::future::join_all((0..12).map(|i| {
            let tools = request_tools(&url, cases[i % cases.len()].0);
            let gateway = gateway.clone();
            async move { ensure_request_mcp_client(&gateway, &tools).await.unwrap() }
        }))
        .await;

        let handles: Vec<_> = requests
            .into_iter()
            .zip(expected)
            .map(|(mcp, expected)| {
                tokio::spawn(async move {
                    for _ in 0..3 {
                        assert_eq!(call(&mcp, "read_private_notes").await, expected);
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.await.unwrap();
        }
        assert_eq!(gateway.connection_pool().len(), 0);
    }

    #[tokio::test]
    async fn test_request_tools_are_private_to_their_request() {
        let url = serve(NotesServer::new).await;
        let gateway = gateway_mcp().await;

        let tenant_a = ensure_request_mcp_client(&gateway, &request_tools(&url, Some(TOKEN_A)))
            .await
            .unwrap();
        assert_eq!(tool_names(&tenant_a), ["read_private_notes"]);

        // While tenant A's request is still running, a request that names no MCP
        // server can neither list nor call tenant A's tool.
        let other = RequestMcpTools::static_only(gateway.clone());
        assert!(other.list_tools().is_empty());
        let err = other
            .call_tool("read_private_notes", "{}")
            .await
            .unwrap_err();
        assert!(matches!(err, smg_mcp::error::McpError::ToolNotFound(_)));

        assert_eq!(call(&tenant_a, "read_private_notes").await, NOTES_A);
    }

    #[tokio::test]
    async fn test_static_servers_stay_shared() {
        let request_url = serve(NotesServer::new).await;
        let static_url = serve(OperatorServer::new).await;
        let gateway = gateway_mcp().await;
        register_static_server(&gateway, "operator", &static_url).await;

        // A request without MCP servers sees the static tools, called with the
        // operator's credential.
        let static_only = RequestMcpTools::static_only(gateway.clone());
        assert_eq!(
            tool_names(&static_only),
            ["operator_whoami", "read_private_notes"]
        );
        assert_eq!(
            call(&static_only, "read_private_notes").await,
            NOTES_OPERATOR
        );

        // A request naming its own server sees its own tool in place of the static
        // tool with the same name, and still sees the other static tools.
        let tenant_a =
            ensure_request_mcp_client(&gateway, &request_tools(&request_url, Some(TOKEN_A)))
                .await
                .unwrap();
        assert_eq!(
            tool_names(&tenant_a),
            ["operator_whoami", "read_private_notes"]
        );
        assert_eq!(call(&tenant_a, "read_private_notes").await, NOTES_A);
        assert_eq!(call(&tenant_a, "operator_whoami").await, NOTES_OPERATOR);

        // That shadowing is invisible to every other request.
        assert_eq!(
            call(&static_only, "read_private_notes").await,
            NOTES_OPERATOR
        );
        let no_mcp = RequestMcpTools::static_only(gateway.clone());
        assert_eq!(call(&no_mcp, "read_private_notes").await, NOTES_OPERATOR);

        // The static server keeps its single shared session.
        assert_eq!(gateway.list_static_servers(), ["operator"]);
        assert_eq!(gateway.connection_pool().len(), 0);
    }
}
