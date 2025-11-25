//! OpenAI-compatible router implementation
//!
//! This module provides two separate pipelines:
//! - Chat pipeline: Lightweight proxy for /v1/chat/completions (4 stages)
//! - Responses pipeline: Full-featured for /v1/responses with MCP (8 stages)

// Two pipeline modules
pub mod chat;
pub mod responses;

// Shared modules
pub mod conversations;
mod router;
mod utils;

// Re-export the main router type for external use
pub use router::OpenAIRouter;
