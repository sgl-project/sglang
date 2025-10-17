use async_trait::async_trait;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::ParserResult,
        traits::{TokenToolParser, ToolParser},
        types::{StreamingParseResult, ToolCall},
    },
};

/// Placeholder for the Harmony-backed GPT-OSS parser.
///
/// regex implementation. This struct will be fleshed out in subsequent phases to
/// reuse Harmony's tokenizer and message reconstruction logic.
#[derive(Default)]
pub struct GptOssHarmonyParser;

impl GptOssHarmonyParser {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ToolParser for GptOssHarmonyParser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Temporary stub: fall back to returning the raw text with no tool calls.
        // Later phases will decode Harmony tokens into structured tool calls.
        Ok((output.to_string(), Vec::new()))
    }

    async fn parse_incremental(
        &mut self,
        _chunk: &str,
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        // Temporary stub until the Harmony streaming pipeline is implemented.
        Ok(StreamingParseResult::default())
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        // Reuse the legacy heuristics for now; this will be replaced with Harmony-specific
        // start-token detection when the parser is fully implemented.
        text.contains("<|channel|>commentary")
    }

    fn as_token_parser(&self) -> Option<&dyn TokenToolParser> {
        Some(self)
    }
}

#[async_trait]
impl TokenToolParser for GptOssHarmonyParser {
    async fn parse_complete_tokens(
        &self,
        _tokens: &[u32],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        // Placeholder until Harmony integration lands. Returning an empty tool list ensures
        // that enabling the parser without full implementation results in a no-op rather
        // than a runtime panic.
        Ok((String::new(), Vec::new()))
    }

    async fn parse_incremental_tokens(
        &mut self,
        _tokens: &[u32],
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        Ok(StreamingParseResult::default())
    }
}
