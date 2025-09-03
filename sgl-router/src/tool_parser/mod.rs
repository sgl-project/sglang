/// Tool parser module for handling function/tool calls in model outputs
///
/// This module provides infrastructure for parsing tool calls from various model formats.
// Core modules
pub mod errors;
pub mod partial_json;
pub mod python_literal_parser;
pub mod registry;
pub mod state;
pub mod traits;
pub mod types;

// Parser implementations
pub mod parsers;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use errors::{ToolParserError, ToolParserResult};
pub use registry::ParserRegistry;
pub use state::{ParsePhase, ParseState};
pub use traits::{PartialJsonParser, ToolParser};
pub use types::{FunctionCall, PartialToolCall, StreamResult, TokenConfig, ToolCall};

// Re-export parsers for convenience
pub use parsers::{
    DeepSeekParser, Glm4MoeParser, GptOssParser, JsonParser, KimiK2Parser, LlamaParser,
    MistralParser, PythonicParser, QwenParser, Step3Parser,
};
