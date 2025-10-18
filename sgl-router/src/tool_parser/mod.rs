/// Tool parser module for handling function/tool calls in model outputs
///
/// This module provides infrastructure for parsing tool calls from various model formats.
// Core modules
pub mod errors;
pub mod factory;
pub mod partial_json;
pub mod state;
pub mod traits;
pub mod types;

// Parser implementations
pub mod parsers;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use errors::{ParserError, ParserResult};
pub use factory::{ParserFactory, ParserRegistry, PooledParser};
// Re-export parsers for convenience
pub use parsers::{
    DeepSeekParser, Glm4MoeParser, GptOssParser, JsonParser, KimiK2Parser, LlamaParser,
    MistralParser, PythonicParser, QwenParser, Step3Parser,
};
pub use traits::{PartialJsonParser, ToolParser};
pub use types::{FunctionCall, PartialToolCall, StreamingParseResult, ToolCall};
