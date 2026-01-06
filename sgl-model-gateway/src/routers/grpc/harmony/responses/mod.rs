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

pub(crate) mod common;
pub(crate) mod context;
pub(crate) mod execution;
pub(crate) mod non_streaming;
pub(crate) mod streaming;

// Re-export types accessed via harmony::responses::TypeName
pub(crate) use context::HarmonyResponsesContext;
pub(crate) use execution::ToolResult;
pub(crate) use non_streaming::serve_harmony_responses;
pub(crate) use streaming::serve_harmony_responses_stream;
