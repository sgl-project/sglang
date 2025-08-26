/// Tool parser module for handling function/tool calls in model outputs
///
/// This module provides infrastructure for parsing tool calls from various model formats.
pub mod errors;
pub mod json_parser;
pub mod llama_parser;
pub mod mistral_parser;
pub mod partial_json;
pub mod python_literal_parser;
pub mod pythonic_parser;
pub mod qwen_parser;
pub mod registry;
pub mod state;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use errors::{ToolParserError, ToolParserResult};
pub use json_parser::JsonParser;
pub use llama_parser::LlamaParser;
pub use mistral_parser::MistralParser;
pub use pythonic_parser::PythonicParser;
pub use qwen_parser::QwenParser;
pub use registry::ParserRegistry;
pub use state::{ParsePhase, ParseState};
pub use traits::{PartialJsonParser, ToolParser};
pub use types::{FunctionCall, PartialToolCall, StreamResult, TokenConfig, ToolCall};
