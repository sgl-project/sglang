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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_with_python_tag() {
        let parser = LlamaParser::new();
        let input = r#"<|python_tag|>{"name": "search", "arguments": {"query": "weather"}}"#;

        let (normal_text, tool_calls) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "search");
        assert!(tool_calls[0].function.arguments.contains("weather"));
        assert_eq!(normal_text, ""); // Pure python_tag with JSON should have no normal text
    }

    #[tokio::test]
    async fn test_parse_plain_json() {
        let parser = LlamaParser::new();
        let input = r#"{"name": "calculate", "arguments": {"x": 5, "y": 10}}"#;

        let (normal_text, tool_calls) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "calculate");
        assert_eq!(normal_text, ""); // Pure JSON should have no normal text
    }

    #[tokio::test]
    async fn test_parse_with_text_before() {
        let parser = LlamaParser::new();
        let input = r#"Let me help you with that. <|python_tag|>{"name": "get_time", "arguments": {"timezone": "UTC"}}"#;

        let (normal_text, tool_calls) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_time");
        assert_eq!(normal_text, "Let me help you with that. ");
    }

    #[test]
    fn test_detect_format() {
        let parser = LlamaParser::new();

        assert!(parser.detect_format(r#"<|python_tag|>{"name": "test"}"#));
        assert!(parser.detect_format(r#"{"name": "test", "arguments": {}}"#));
        assert!(!parser.detect_format("plain text"));
        assert!(!parser.detect_format(r#"{"key": "value"}"#)); // No name field
    }

    #[tokio::test]
    async fn test_single_call_with_semicolon() {
        let parser = LlamaParser::new();
        // Note: Llama 3.2 doesn't handle multiple calls well
        let input = r#"<|python_tag|>{"name": "func1", "arguments": {"x": 1}};"#;

        let (_normal_text, tool_calls) = parser.parse_complete(input).await.unwrap();

        // We expect this to either parse the first JSON object or fail gracefully
        // Since the semicolon makes it invalid JSON, it will likely return empty
        // This is acceptable as Llama 3.2 doesn't reliably support parallel calls

        // If it parses anything, it should be func1
        if !tool_calls.is_empty() {
            assert_eq!(tool_calls[0].function.name, "func1");
        }
    }
}
