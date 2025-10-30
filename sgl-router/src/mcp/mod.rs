//! Model Context Protocol (MCP) client implementation.
//!
//! Provides MCP client functionality including tools, prompts, resources, and OAuth.
//! Supports stdio, SSE, and HTTP transports with connection pooling and caching.

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
    Prompt, RawResource, Tool, WarmupServer,
};
pub use connection_pool::{CachedConnection, McpConnectionPool, PoolStats};
pub use error::{McpError, McpResult};
pub use inventory::ToolInventory;
pub use manager::{McpManager, McpManagerStats};
pub use proxy::{create_http_client, resolve_proxy_config};
pub use tool_args::ToolArgs;
