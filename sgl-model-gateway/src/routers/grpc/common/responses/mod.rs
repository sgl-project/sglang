//! Shared response functionality used by both regular and harmony implementations

pub mod handlers;
pub mod streaming;
pub mod utils;

pub use handlers::{cancel_response_impl, get_response_impl};
pub use streaming::{build_sse_response, OutputItemType, ResponseStreamEventEmitter};
pub use utils::{
    build_mcp_call_item, build_mcp_list_tools_item, ensure_mcp_connection,
    persist_response_if_needed,
};
