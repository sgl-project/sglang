//! OpenAI-compatible router implementation
//!
//! This module provides OpenAI-compatible API routing with support for:
//! - Streaming and non-streaming responses
//! - MCP (Model Context Protocol) tool calling
//! - Response storage and conversation management
//! - Multi-turn tool execution loops
//! - SSE (Server-Sent Events) streaming

mod context;
mod provider;
pub mod responses;
mod router;

pub use router::OpenAIRouter;
