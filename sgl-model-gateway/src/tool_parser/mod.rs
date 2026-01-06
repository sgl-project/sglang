/// Tool parser module for handling function/tool calls in model outputs
///
/// This module provides infrastructure for parsing tool calls from various model formats.
// Core modules
pub mod errors;
pub mod factory;
pub mod partial_json;
pub mod traits;
pub mod types;

// Parser implementations
pub mod parsers;

#[cfg(test)]
mod tests;

// Re-export types used outside this module
pub use factory::{ParserFactory, PooledParser};
pub use parsers::{
    DeepSeekParser, Glm4MoeParser, JsonParser, KimiK2Parser, LlamaParser, MinimaxM2Parser,
    MistralParser, PythonicParser, QwenParser, Step3Parser,
};
pub use traits::ToolParser;
pub use types::{FunctionCall, PartialToolCall, StreamingParseResult, ToolCall};
