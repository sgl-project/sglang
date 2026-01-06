//! Shared response functionality used by both regular and harmony implementations

pub(crate) mod handlers;
pub(crate) mod streaming;
pub(crate) mod utils;

// Re-export commonly used items
pub(crate) use streaming::build_sse_response;
pub(crate) use utils::{ensure_mcp_connection, persist_response_if_needed};
