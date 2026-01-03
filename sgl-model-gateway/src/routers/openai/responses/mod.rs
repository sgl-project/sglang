//! OpenAI-compatible responses handling module
//!
//! This module provides comprehensive support for OpenAI Responses API with:
//! - Streaming and non-streaming response handling
//! - MCP (Model Context Protocol) tool interception and execution
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Response accumulation for persistence
//! - Tool call detection and output index remapping

mod accumulator;
mod common;
pub mod mcp;
mod non_streaming;
mod streaming;
mod tool_handler;
mod utils;

// Re-export the main entry point handlers
// Re-export MCP functions for non-streaming path
pub use mcp::{execute_tool_loop, prepare_mcp_payload_for_streaming};
pub use non_streaming::handle_non_streaming_response;
pub use streaming::handle_streaming_response;
// Re-export utility functions used by router
pub use utils::{mask_tools_as_mcp, patch_streaming_response_json};
