// MCP Client for SGLang Router
//
// This module provides a complete MCP (Model Context Protocol) client implementation
// supporting multiple transport types (stdio, SSE, HTTP) and all MCP features:
// - Tools: Discovery and execution
// - Prompts: Reusable templates for LLM interactions
// - Resources: File/data access with subscription support
// - OAuth: Secure authentication for remote servers

pub mod config;
pub mod connection_pool;
pub mod error;
pub mod inventory;
pub mod manager;
pub mod oauth;
pub mod proxy;
pub mod tool_args;

// Re-export the main types for convenience
pub use config::{
    InventoryConfig, McpConfig, McpPoolConfig, McpProxyConfig, McpServerConfig, McpTransport,
    PromptInfo, ResourceInfo, ToolInfo, WarmupServer,
};
pub use connection_pool::{CachedConnection, McpConnectionPool, PoolStats};
pub use error::{McpError, McpResult};
pub use inventory::ToolInventory;
pub use manager::{McpManager, McpManagerStats};
pub use proxy::{create_http_client, resolve_proxy_config};
pub use tool_args::ToolArgs;
