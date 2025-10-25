//! gRPC Router `/v1/responses` endpoint implementation
//!
//! This module handles all responses-specific logic including:
//! - Request validation
//! - Conversation history and response chain loading
//! - Background mode execution
//! - Streaming support
//! - MCP tool loop wrapper
//! - Response persistence

// Module declarations
mod conversions;
mod handlers;
mod harmony;
mod harmony_utils;
pub mod streaming;
pub mod tool_loop;
pub mod types;

// Public exports
// Re-export for internal use
pub(crate) use handlers::load_conversation_history;
pub use handlers::{cancel_response_impl, get_response_impl, route_responses};
pub use harmony::route_harmony_responses;
pub use harmony_utils::is_harmony_model;
pub use types::BackgroundTaskInfo;
