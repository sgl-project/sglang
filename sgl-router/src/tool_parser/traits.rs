use crate::tool_parser::{
    errors::ToolParserResult,
    state::ParseState,
    types::{StreamResult, ToolCall},
};
use async_trait::async_trait;

/// Core trait for all tool parsers
#[async_trait]
pub trait ToolParser: Send + Sync {
    /// Parse complete tool calls from final output
    /// Returns (remaining_normal_text, tool_calls) tuple
    async fn parse_complete(&self, output: &str) -> ToolParserResult<(String, Vec<ToolCall>)>;

    /// Parse tool calls from model output (streaming)
    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult>;

    /// Check if text contains tool calls in this parser's format
    fn detect_format(&self, text: &str) -> bool;

    /// Optionally expose a token-aware parser implementation.
    /// Default returns `None`, meaning the parser only supports text input.
    fn as_token_parser(&self) -> Option<&dyn TokenToolParser> {
        None
    }
}

/// Trait for partial JSON parsing
pub trait PartialJsonParser: Send + Sync {
    /// Parse potentially incomplete JSON
    fn parse(&self, input: &str) -> ToolParserResult<(serde_json::Value, usize)>;

    /// Check if JSON is complete
    fn is_complete(&self, input: &str) -> bool;

    /// Get the maximum parsing depth
    fn max_depth(&self) -> usize;
}

#[async_trait]
pub trait TokenToolParser: ToolParser {
    /// Parse complete tool calls when provided with raw token IDs.
    async fn parse_complete_tokens(
        &self,
        tokens: &[u32],
    ) -> ToolParserResult<(String, Vec<ToolCall>)>;

    /// Streaming parser entrypoint for token chunks.
    async fn parse_incremental_tokens(
        &self,
        tokens: &[u32],
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult>;
}
