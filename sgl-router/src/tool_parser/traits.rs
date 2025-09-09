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
    async fn parse_complete(&self, output: &str) -> ToolParserResult<Vec<ToolCall>>;

    /// Parse tool calls from model output (streaming)
    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult>;

    /// Check if text contains tool calls in this parser's format
    fn detect_format(&self, text: &str) -> bool;
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
