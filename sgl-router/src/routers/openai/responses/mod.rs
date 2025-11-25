//! Responses pipeline module
//!
//! Full-featured pipeline for /v1/responses endpoint.
//! Supports MCP tools, conversation history, and persistence (8 stages).

pub mod context;
pub mod pipeline;
pub mod stages;

// Responses-specific modules (not exported outside responses)
pub(super) mod mcp;
pub(super) mod streaming;
pub(super) mod utils;

// Re-export main types
pub use context::{
    ContextOutput, DiscoveryOutput, ExecutionResult, FinalResponse, McpOutput, PayloadOutput,
    ProcessedResponse, ResponsesDependencies, ResponsesProcessingState, ResponsesRequestContext,
    ResponsesRequestInput, ValidationOutput,
};
// Re-export MCP utilities for external use (e.g., gRPC router, tests)
pub use mcp::{ensure_request_mcp_client, ToolLoopState};
pub use pipeline::ResponsesPipeline;
