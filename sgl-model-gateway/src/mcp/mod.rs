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

// Re-export types used outside this module
pub use config::{McpConfig, McpServerConfig, McpTransport, Tool};
pub use manager::McpManager;
