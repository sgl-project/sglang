//! Shared MCP (Model Context Protocol) utilities for routers.
//!
//! This module provides shared MCP-related functionality that can be
//! used across different router implementations (OpenAI, gRPC regular, gRPC harmony).
//!
//! ## Modules
//!
//! - [`mcp_connection`]: Connection validation for request-scoped MCP servers
//! - [`tool_loop`]: Trait-based abstraction for MCP tool calling loops

mod mcp_connection;
pub mod tool_loop;

pub use mcp_connection::ensure_request_mcp_client;
pub use tool_loop::{
    execute_mcp_loop, extract_server_label, IterationOutcome, LoopExitReason, McpLoopBackend,
    McpLoopConfig, ParsedToolCall, ToolExecutionResult,
};
