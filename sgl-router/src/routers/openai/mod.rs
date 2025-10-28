//! OpenAI-compatible router implementation
//!
//! This module provides OpenAI-compatible API routing with support for:
//! - Streaming and non-streaming responses
//! - MCP (Model Context Protocol) tool calling
//! - Response storage and conversation management
//! - Multi-turn tool execution loops
//! - SSE (Server-Sent Events) streaming

pub mod conversations;
pub mod mcp;
mod responses;
mod router;
mod streaming;
mod utils;

// Re-export the main router type for external use
pub use router::OpenAIRouter;
