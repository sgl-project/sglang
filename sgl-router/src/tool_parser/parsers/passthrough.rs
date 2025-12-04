//! Passthrough parser that returns text unchanged
//!
//! This parser is used as a fallback for unknown models where no specific
//! tool call parsing should be performed. It simply returns the input text
//! with no tool calls detected.

use async_trait::async_trait;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::ParserResult,
        traits::ToolParser,
        types::{StreamingParseResult, ToolCall, ToolCallItem},
    },
};

/// Passthrough parser that returns text unchanged with no tool calls
#[derive(Default)]
pub struct PassthroughParser;

impl PassthroughParser {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ToolParser for PassthroughParser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Return text unchanged with no tool calls
        Ok((output.to_string(), vec![]))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        // Return chunk unchanged with no tool calls
        Ok(StreamingParseResult {
            normal_text: chunk.to_string(),
            calls: vec![],
        })
    }

    fn has_tool_markers(&self, _text: &str) -> bool {
        // Passthrough never detects tool calls
        false
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        None
    }
}
