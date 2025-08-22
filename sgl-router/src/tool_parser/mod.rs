/// Tool parser module for handling function/tool calls in model outputs
///
/// This module provides infrastructure for parsing tool calls from various model formats.
/// Phase 1 focuses on core infrastructure: types, traits, registry, and partial JSON parsing.
pub mod errors;
pub mod partial_json;
pub mod registry;
pub mod state;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use errors::{ToolParserError, ToolParserResult};
pub use registry::ParserRegistry;
pub use state::{ParsePhase, ParseState};
pub use traits::{PartialJsonParser, ToolParser};
pub use types::{FunctionCall, PartialToolCall, StreamResult, TokenConfig, ToolCall};
