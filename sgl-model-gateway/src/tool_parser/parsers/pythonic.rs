//! Pythonic format parser for tool calls
///
/// Handles Python function call syntax within square brackets:
/// ```text
/// [tool1(arg1=val1, arg2=val2), tool2(arg1=val3, arg2=val3)]
/// ```
///
/// This format is used by Llama models and uses Python literals
/// rather than JSON for arguments.
/// Reference: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct?chat_template=default
use async_trait::async_trait;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::ParserResult,
        traits::ToolParser,
        types::{StreamingParseResult, ToolCall},
    },
};

/// Pythonic format parser for tool calls
#[derive(Default)]
pub struct PythonicParser;

impl PythonicParser {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ToolParser for PythonicParser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(output) {
            return Ok((output.to_string(), vec![]));
        }

        let idx = output.find("[tool_calls_section_begin]").unwrap();
        let normal_text = output[..idx].to_string();

        Ok((normal_text, vec![]))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        Ok(StreamingParseResult {
            normal_text: chunk.to_string(),
            calls: vec![],
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("[tool_calls_section_begin]")
    }

    fn get_format_info(&self, _tool_name: &str) -> (String, String, String) {
        (String::new(), String::new(), String::new())
    }
}
