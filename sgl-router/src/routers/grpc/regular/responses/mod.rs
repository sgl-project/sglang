//! Regular gRPC Router `/v1/responses` endpoint implementation
//!
//! This module handles all responses-specific logic for the regular (non-Harmony) pipeline including:
//! - Request validation
//! - Conversation history and response chain loading
//! - Streaming support
//! - MCP tool loop wrapper
//! - Response persistence

// Module declarations
pub mod context;
mod conversions;
mod handlers;
pub mod tool_loop;
pub mod types;

// Public exports
pub use context::ResponsesContext;
pub use handlers::route_responses;
pub use types::BackgroundTaskInfo;
