// MCP Client for SGLang Router
//
// This module provides a complete MCP (Model Context Protocol) client implementation
// supporting multiple transport types (stdio, SSE, HTTP) and all MCP features:
// - Tools: Discovery and execution
// - Prompts: Reusable templates for LLM interactions
// - Resources: File/data access with subscription support
// - OAuth: Secure authentication for remote servers

pub mod client_manager;
pub mod config;
pub mod error;
pub mod oauth;
pub mod proxy;

// Re-export the main types for convenience
pub use client_manager::{McpClientManager, PromptInfo, ResourceInfo, ToolInfo};
pub use config::{
    InventoryConfig, McpConfig, McpPoolConfig, McpProxyConfig, McpServerConfig, McpTransport,
    WarmupServer,
};
pub use error::{McpError, McpResult};
pub use proxy::{create_http_client, resolve_proxy_config};
