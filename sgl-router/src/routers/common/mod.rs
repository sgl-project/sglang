//! Common utilities shared across routers
//!
//! This module contains utilities that are used by multiple routers:
//! - Event type constants
//! - Item ID generation
//! - SSE formatting helpers
//! - MCP event builders
//!
//! These utilities ensure consistency across OpenAI router (proxy mode)
//! and gRPC router (generation mode).

pub mod event_types;
pub mod item_ids;
pub mod mcp_events;
pub mod sse_format;
