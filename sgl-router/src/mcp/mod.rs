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

// Re-export the main types for convenience
pub use client_manager::{McpClientManager, PromptInfo, ResourceInfo, ToolInfo};
pub use config::{McpConfig, McpServerConfig, McpTransport};
pub use error::{McpError, McpResult};
