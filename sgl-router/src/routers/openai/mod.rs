//! OpenAI-compatible router implementation
//!
//! This module provides OpenAI-compatible API routing with support for:
//! - Streaming and non-streaming responses
//! - MCP (Model Context Protocol) tool calling
//! - Response storage and conversation management
//! - Multi-turn tool execution loops
//! - SSE (Server-Sent Events) streaming
//! - Pipeline-based request processing (Phase 1: Foundation)

// Pipeline infrastructure (Phase 1)
pub mod context;
pub mod pipeline;
pub mod stages;

// Existing modules
pub mod conversations;
pub mod mcp;
mod responses;
mod router;
mod streaming;
mod utils;

// Tests (Phase 1)
#[cfg(test)]
mod tests;

// Re-export the main router type for external use
pub use router::OpenAIRouter;
