use async_trait::async_trait;

use super::json_parser::JsonParser;
use crate::tool_parser::{
    errors::ToolParserResult,
    state::ParseState,
    traits::ToolParser,
    types::{StreamResult, TokenConfig, ToolCall},
};

/// Llama 3.2 format parser for tool calls
///
/// Handles the Llama 3.2 specific format:
/// `<|python_tag|>{"name": "func", "arguments": {...}}`
///
/// Also supports plain JSON without the python_tag prefix
pub struct LlamaParser {
    /// Underlying JSON parser with Llama-specific configuration
    json_parser: JsonParser,
}

impl LlamaParser {
    /// Create a new Llama parser
    pub fn new() -> Self {
        // Configure JSON parser with Llama's python_tag token
        // Note: No end token for python_tag format
        let json_parser = JsonParser::with_config(TokenConfig {
            start_tokens: vec!["<|python_tag|>".to_string()],
            end_tokens: vec!["".to_string()], // Empty end token
            separator: ";".to_string(), // Llama uses semicolon for multiple calls (though not well supported)
        });

        Self { json_parser }
    }
}

impl Default for LlamaParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for LlamaParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // First try with the configured python_tag parser
        let (_json_normal_text, tools) = self.json_parser.parse_complete(text).await?;

        if !tools.is_empty() {
            // Extract normal text before the python tag
            // JsonParser doesn't preserve normal text for single start tokens, so we do it manually
            let normal_text = if let Some(tag_pos) = text.find("<|python_tag|>") {
                text[..tag_pos].to_string()
            } else {
                String::new()
            };
            return Ok((normal_text, tools));
        }

        // If no results and text starts with '{', try plain JSON
        if text.trim_start().starts_with('{') {
            // Create a temporary plain JSON parser
            let plain_parser = JsonParser::new();
            let (_json_normal_text, tools) = plain_parser.parse_complete(text).await?;
            // For plain JSON, don't extract normal text (consistent with JsonParser behavior)
            return Ok((String::new(), tools));
        }

        // No tool calls found, return original text as normal text
        Ok((text.to_string(), vec![]))
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        // First, try with the configured json_parser (which handles python_tag)
        let result = self.json_parser.parse_incremental(chunk, state).await?;

        // If we get Incomplete and no python_tag in buffer, might be plain JSON
        if matches!(result, StreamResult::Incomplete) {
            let trimmed = state.buffer.trim_start();
            if trimmed.starts_with('{') && !state.buffer.contains("<|python_tag|>") {
                // Likely plain JSON, try with a plain parser
                // Note: We need to be careful not to double-add the chunk
                let plain_parser = JsonParser::new();
                // The chunk was already added to state.buffer by json_parser above
                // So we call with empty string to just process what's in the buffer
                return plain_parser.parse_incremental("", state).await;
            }
        }

        Ok(result)
    }

    fn detect_format(&self, text: &str) -> bool {
        // Llama format if contains python_tag or starts with JSON object
        text.contains("<|python_tag|>")
            || (text.trim_start().starts_with('{')
                && (text.contains(r#""name""#) || text.contains(r#""function""#)))
    }
}
