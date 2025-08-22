/// Tool parser module for handling function/tool calls in model outputs
///
/// This module provides infrastructure for parsing tool calls from various model formats.
pub mod errors;
pub mod json_parser;
pub mod partial_json;
pub mod registry;
pub mod state;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use errors::{ToolParserError, ToolParserResult};
pub use json_parser::JsonParser;
pub use registry::ParserRegistry;
pub use state::{ParsePhase, ParseState};
pub use traits::{PartialJsonParser, ToolParser};
pub use types::{FunctionCall, PartialToolCall, StreamResult, TokenConfig, ToolCall};
