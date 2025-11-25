//! Responses pipeline module
//!
//! Full-featured pipeline for /v1/responses endpoint.
//! Supports MCP tools, conversation history, and persistence (8 stages).

pub mod context;
pub mod pipeline;
pub mod stages;
pub(super) mod utils;

// Re-export main types
pub use context::{
    ContextOutput, DiscoveryOutput, ExecutionResult, FinalResponse, McpOutput, PayloadOutput,
    ProcessedResponse, ResponsesDependencies, ResponsesProcessingState, ResponsesRequestContext,
    ResponsesRequestInput, ValidationOutput,
};
pub use pipeline::ResponsesPipeline;
