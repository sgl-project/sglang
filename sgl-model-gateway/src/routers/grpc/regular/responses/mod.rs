//! Regular gRPC Router `/v1/responses` endpoint implementation
//!
//! This module handles all responses-specific logic for the regular (non-Harmony) pipeline.
//!
//! ## Architecture
//!
//! - `handlers` - Entry points: route_responses (thin dispatcher)
//! - `non_streaming` - Non-streaming execution with MCP tool loop
//! - `streaming` - Streaming execution with MCP tool loop
//! - `common` - Shared helpers: ToolLoopState, tool preparation, MCP metadata builders
//! - `conversions` - Request/response conversion between Responses and Chat formats
//! - `context` - ResponsesContext and BackgroundTaskInfo

mod common;
mod context;
mod conversions;
mod handlers;
mod non_streaming;
mod streaming;

// Public exports
pub(crate) use context::ResponsesContext;
pub(crate) use handlers::route_responses;
