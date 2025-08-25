use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
};

/// JSON format parser for tool calls
///
/// Handles various JSON formats for function calling:
/// - Single tool call: {"name": "fn", "arguments": {...}}
/// - Multiple tool calls: [{"name": "fn1", "arguments": {...}}, ...]
/// - With parameters instead of arguments: {"name": "fn", "parameters": {...}}
///
/// Supports configurable token markers for different models
pub struct JsonParser {
    /// Token(s) that mark the start of tool calls
    start_tokens: Vec<String>,
    /// Token(s) that mark the end of tool calls
    end_tokens: Vec<String>,
    /// Separator between multiple tool calls (reserved for future use)
    _separator: String,
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex patterns for extracting content between tokens
    extractors: Vec<Regex>,
}

impl JsonParser {
    /// Create a new JSON parser with default configuration
    pub fn new() -> Self {
        Self::with_config(
            vec![], // No wrapper tokens by default
            vec![],
            ", ".to_string(),
        )
    }

    /// Create a parser with custom token configuration
    pub fn with_config(
        start_tokens: Vec<String>,
        end_tokens: Vec<String>,
        separator: String,
    ) -> Self {
        // Build extraction patterns for each token pair
        let extractors = start_tokens
            .iter()
            .zip(end_tokens.iter())
            .filter_map(|(start, end)| {
                if !start.is_empty() && !end.is_empty() {
                    // Use (?s) flag to enable DOTALL mode so . matches newlines
                    let pattern =
                        format!(r"(?s){}(.*?){}", regex::escape(start), regex::escape(end));
                    Regex::new(&pattern).ok()
                } else {
                    None
                }
            })
            .collect();

        Self {
            start_tokens,
            end_tokens,
            _separator: separator,
            partial_json: PartialJson::default(),
            extractors,
        }
    }

    /// Extract JSON content from text, handling wrapper tokens if configured
    fn extract_json_content<'a>(&self, text: &'a str) -> &'a str {
        let mut content = text.trim();

        // Try each extractor pattern
        for extractor in &self.extractors {
            if let Some(captures) = extractor.captures(content) {
                if let Some(matched) = captures.get(1) {
                    content = matched.as_str().trim();
                    break;
                }
            }
        }

        // Handle special case where there's a start token but no end token
        for (start, end) in self.start_tokens.iter().zip(self.end_tokens.iter()) {
            if !start.is_empty() && end.is_empty() {
                content = content.strip_prefix(start).unwrap_or(content);
            }
        }

        content
    }

    /// Parse a single JSON object into a ToolCall
    fn parse_single_object(&self, obj: &Value) -> ToolParserResult<Option<ToolCall>> {
        // Check if this looks like a tool call
        let name = obj
            .get("name")
            .or_else(|| obj.get("function"))
            .and_then(|v| v.as_str());

        if let Some(name) = name {
            // Get arguments - support both "arguments" and "parameters" keys
            let empty_obj = Value::Object(serde_json::Map::new());
            let args = obj
                .get("arguments")
                .or_else(|| obj.get("parameters"))
                .unwrap_or(&empty_obj);

            // Convert arguments to JSON string
            let arguments = serde_json::to_string(args)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate a unique ID if not provided
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));

            Ok(Some(ToolCall {
                id,
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            }))
        } else {
            Ok(None)
        }
    }

    /// Parse JSON value(s) into tool calls
    fn parse_json_value(&self, value: &Value) -> ToolParserResult<Vec<ToolCall>> {
        let mut tools = Vec::new();

        match value {
            Value::Array(arr) => {
                // Parse each element in the array
                for item in arr {
                    if let Some(tool) = self.parse_single_object(item)? {
                        tools.push(tool);
                    }
                }
            }
            Value::Object(_) => {
                // Single tool call
                if let Some(tool) = self.parse_single_object(value)? {
                    tools.push(tool);
                }
            }
            _ => {
                // Not a valid tool call format
                return Ok(vec![]);
            }
        }

        Ok(tools)
    }

    /// Check if text contains potential tool call markers
    fn has_tool_markers(&self, text: &str) -> bool {
        // If no start tokens configured, check for JSON structure
        if self.start_tokens.is_empty() {
            // For JSON, we just need to see the start of an object or array
            return text.contains('{') || text.contains('[');
        }

        // Check for any start token
        self.start_tokens.iter().any(|token| text.contains(token))
    }
}

impl Default for JsonParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for JsonParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<Vec<ToolCall>> {
        // Extract JSON content from wrapper tokens if present
        let json_content = self.extract_json_content(text);

        // Try to parse as JSON
        match serde_json::from_str::<Value>(json_content) {
            Ok(value) => self.parse_json_value(&value),
            Err(_) => {
                // Not valid JSON, return empty
                Ok(vec![])
            }
        }
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        state.buffer.push_str(chunk);

        // Check if we have potential tool calls
        if !self.has_tool_markers(&state.buffer) {
            // No tool markers, return as incomplete
            return Ok(StreamResult::Incomplete);
        }

        // Extract JSON content
        let json_content = self.extract_json_content(&state.buffer);

        // Try to parse with partial JSON parser
        match self.partial_json.parse_value(json_content) {
            Ok((value, consumed)) => {
                // Check if we have a complete JSON structure
                if consumed == json_content.len() {
                    // Complete JSON, parse tool calls
                    let tools = self.parse_json_value(&value)?;
                    if !tools.is_empty() {
                        // Clear buffer since we consumed everything
                        state.buffer.clear();

                        // Return the first tool as complete (simplified for Phase 2)
                        if let Some(tool) = tools.into_iter().next() {
                            return Ok(StreamResult::ToolComplete(tool));
                        }
                    }
                } else {
                    // Partial JSON, try to extract tool name
                    if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
                        // Simple implementation for Phase 2
                        // Just return the tool name once we see it
                        if !state.in_string {
                            state.in_string = true; // Use as a flag for "name sent"
                            return Ok(StreamResult::ToolName {
                                index: 0,
                                name: name.to_string(),
                            });
                        }

                        // Check for complete arguments
                        if let Some(args) =
                            value.get("arguments").or_else(|| value.get("parameters"))
                        {
                            if let Ok(args_str) = serde_json::to_string(args) {
                                // Return arguments as a single update
                                return Ok(StreamResult::ToolArguments {
                                    index: 0,
                                    arguments: args_str,
                                });
                            }
                        }
                    }
                }
            }
            Err(_) => {
                // Failed to parse even as partial JSON
                // Keep buffering
            }
        }

        Ok(StreamResult::Incomplete)
    }

    fn detect_format(&self, text: &str) -> bool {
        // Check if text contains JSON-like structure
        if self.has_tool_markers(text) {
            // Try to extract and parse
            let json_content = self.extract_json_content(text);

            // Check if it looks like valid JSON for tool calls
            if let Ok(value) = serde_json::from_str::<Value>(json_content) {
                match value {
                    Value::Object(ref obj) => {
                        // Check for tool call structure
                        obj.contains_key("name") || obj.contains_key("function")
                    }
                    Value::Array(ref arr) => {
                        // Check if array contains tool-like objects
                        arr.iter().any(|v| {
                            v.as_object().is_some_and(|o| {
                                o.contains_key("name") || o.contains_key("function")
                            })
                        })
                    }
                    _ => false,
                }
            } else {
                false
            }
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_single_tool_call() {
        let parser = JsonParser::new();
        let input = r#"{"name": "get_weather", "arguments": {"location": "San Francisco"}}"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "get_weather");
    }

    #[tokio::test]
    async fn test_parse_multiple_tool_calls() {
        let parser = JsonParser::new();
        let input = r#"[
            {"name": "get_weather", "arguments": {"location": "SF"}},
            {"name": "search", "arguments": {"query": "news"}}
        ]"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].function.name, "get_weather");
        assert_eq!(result[1].function.name, "search");
    }

    #[tokio::test]
    async fn test_parse_with_parameters_key() {
        let parser = JsonParser::new();
        let input = r#"{"name": "calculate", "parameters": {"x": 10, "y": 20}}"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "calculate");
        assert!(result[0].function.arguments.contains("10"));
    }

    #[tokio::test]
    async fn test_parse_with_wrapper_tokens() {
        let parser = JsonParser::with_config(
            vec!["<tool>".to_string()],
            vec!["</tool>".to_string()],
            ", ".to_string(),
        );

        let input = r#"<tool>{"name": "test", "arguments": {}}</tool>"#;
        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "test");
    }

    #[test]
    fn test_detect_format() {
        let parser = JsonParser::new();

        assert!(parser.detect_format(r#"{"name": "test", "arguments": {}}"#));
        assert!(parser.detect_format(r#"[{"name": "test"}]"#));
        assert!(!parser.detect_format("plain text"));
        assert!(!parser.detect_format(r#"{"key": "value"}"#));
    }

    #[tokio::test]
    async fn test_streaming_parse() {
        // Phase 2 simplified streaming test
        // Just verify that streaming eventually produces a complete tool call
        let parser = JsonParser::new();
        let mut state = ParseState::new();

        // Send complete JSON in one go (simplified for Phase 2)
        let full_json = r#"{"name": "get_weather", "arguments": {"location": "SF"}}"#;

        let result = parser
            .parse_incremental(full_json, &mut state)
            .await
            .unwrap();

        // Should get a complete tool immediately with complete JSON
        match result {
            StreamResult::ToolComplete(tool) => {
                assert_eq!(tool.function.name, "get_weather");
                assert!(tool.function.arguments.contains("SF"));
            }
            _ => panic!("Expected ToolComplete for complete JSON input"),
        }
    }
}
