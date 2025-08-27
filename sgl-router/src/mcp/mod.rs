// mod.rs - Minimal exports (Python-aligned)
pub mod types;
pub mod tool_server;

pub use types::{MCPError, MCPResult, MCPConfig, ToolCall, ToolResult, ToolSession, ConnectionType};
pub use tool_server::{MCPToolServer, parse_sse_event, ToolCache, ToolMetadata, ToolStats};