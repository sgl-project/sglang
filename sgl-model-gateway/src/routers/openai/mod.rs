//! OpenAI-compatible router implementation
//!
//! This module provides OpenAI-compatible API routing with support for:
//! - Streaming and non-streaming responses
//! - MCP (Model Context Protocol) tool calling
//! - Response storage and conversation management
//! - Multi-turn tool execution loops
//! - SSE (Server-Sent Events) streaming

mod accumulator;
mod context;
pub mod conversations;
pub mod mcp;
pub mod provider;
mod responses;
mod router;
mod streaming;
mod tool_handler;

// Re-export the main types for external use
pub use provider::{Provider, ProviderError, ProviderRegistry};
pub use router::OpenAIRouter;
