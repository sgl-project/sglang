//! Harmony Responses API implementation with multi-turn MCP tool support
//!
//! This module implements the Harmony Responses API orchestration logic,
//! coordinating full pipeline execution with MCP tool support for multi-turn conversations.
//!
//! ## Architecture
//!
//! Multi-turn pipeline orchestration (NOT just a tool loop):
//! - Serves Harmony Responses API requests end-to-end
//! - Each iteration executes FULL pipeline (worker selection + client acquisition + execution + parsing)
//! - Handles MCP tool execution and history building between iterations
//! - Clean separation: serving orchestration vs. pipeline stages (stages/)
//!
//! ## Module Structure
//!
//! - `context` - HarmonyResponsesContext
//! - `non_streaming` - Non-streaming entry point and tool loop
//! - `streaming` - Streaming entry point and tool loop
//! - `execution` - MCP tool execution logic
//! - `common` - Shared helpers and state tracking

mod common;
mod context;
mod execution;
mod non_streaming;
mod streaming;

// Public exports
pub use context::HarmonyResponsesContext;
pub use execution::{convert_mcp_tools_to_response_tools, ToolResult};
pub use non_streaming::serve_harmony_responses;
pub use streaming::serve_harmony_responses_stream;
