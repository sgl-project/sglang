use async_trait::async_trait;
use serde_json::json;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::ParserResult,
        types::{StreamingParseResult, ToolCall},
    },
};

/// Core trait for all tool parsers
#[async_trait]
pub trait ToolParser: Send + Sync {
    /// Parse complete tool calls from final output
    /// Returns (remaining_normal_text, tool_calls) tuple
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)>;

    /// Parse tool calls from model output (streaming)
    /// Parsers now maintain internal state, so self is mutable
    ///
    /// # Arguments
    /// * `chunk` - New text chunk from model output
    /// * `tools` - List of available tools for validation
    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult>;

    /// Check if text contains tool calls in this parser's format
    fn has_tool_markers(&self, text: &str) -> bool;

    /// Optionally expose a token-aware parser implementation.
    /// Default returns `None`, meaning the parser only supports text input.
    fn as_token_parser(&self) -> Option<&dyn TokenToolParser> {
        None
    }

    /// Get unstreamed tool call arguments
    /// Returns tool call items for arguments that have been parsed but not yet streamed
    fn get_unstreamed_tool_args(&self) -> Option<Vec<crate::tool_parser::types::ToolCallItem>> {
        None
    }

    /// Reset the parser state for reuse across requests.
    /// This should clear all buffers and reset state to initial values.
    fn reset(&mut self) {
        // Default no-op implementation
    }

    /// Build xgrammar structural tag for this parser's format.
    ///
    /// # Arguments
    /// * `tools` - List of available tools
    /// * `at_least_one` - Require at least one tool call (for tool_choice="required")
    /// * `stop_after_first` - Stop after first tool call (for parallel_tool_calls=false)
    ///
    /// # Returns
    /// JSON string of structural tag in xgrammar format
    ///
    /// # Default Implementation
    /// Returns `triggered_tags` format using parser-specific begin/end/trigger patterns.
    fn build_structural_tag(
        &self,
        tools: &[Tool],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<String, String> {
        let mut tags = Vec::new();
        let mut triggers = std::collections::HashSet::new();

        for tool in tools {
            let name = &tool.function.name;

            let (begin, end, trigger) = self.get_format_info(name);

            let schema = tool.function.parameters.clone();

            tags.push(json!({
                "format": "tag",
                "begin": begin,
                "content": {
                    "format": "json_schema",
                    "schema": schema
                },
                "end": end
            }));

            triggers.insert(trigger);
        }

        let structural_tag = json!({
            "format": "triggered_tags",
            "triggers": triggers.into_iter().collect::<Vec<_>>(),
            "tags": tags,
            "at_least_one": at_least_one,
            "stop_after_first": stop_after_first
        });

        serde_json::to_string(&structural_tag)
            .map_err(|e| format!("Failed to serialize structural tag: {}", e))
    }

    /// Get format-specific begin/end/trigger patterns for a tool.
    ///
    /// # Returns
    /// Tuple of (begin, end, trigger)
    ///
    /// This method must be implemented by each parser to provide model-specific formats.
    fn get_format_info(&self, tool_name: &str) -> (String, String, String);
}

/// Trait for partial JSON parsing
pub trait PartialJsonParser: Send + Sync {
    /// Parse potentially incomplete JSON
    fn parse(&self, input: &str) -> ParserResult<(serde_json::Value, usize)>;

    /// Check if JSON is complete
    fn is_complete(&self, input: &str) -> bool;

    /// Get the maximum parsing depth
    fn max_depth(&self) -> usize;
}

#[async_trait]
pub trait TokenToolParser: ToolParser {
    /// Parse complete tool calls when provided with raw token IDs.
    async fn parse_complete_tokens(&self, tokens: &[u32]) -> ParserResult<(String, Vec<ToolCall>)>;

    /// Streaming parser entrypoint for token chunks.
    /// Parsers maintain internal state, so self is mutable
    async fn parse_incremental_tokens(
        &mut self,
        tokens: &[u32],
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult>;
}
