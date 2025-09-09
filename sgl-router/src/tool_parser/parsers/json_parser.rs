use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, TokenConfig, ToolCall},
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
    /// Token configuration for parsing
    token_config: TokenConfig,
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex patterns for extracting content between tokens
    extractors: Vec<Regex>,
}

impl JsonParser {
    /// Create a new JSON parser with default configuration
    pub fn new() -> Self {
        Self::with_config(TokenConfig {
            start_tokens: vec![],
            end_tokens: vec![],
            separator: ", ".to_string(),
        })
    }

    /// Create a parser with custom token configuration
    pub fn with_config(config: TokenConfig) -> Self {
        // Build extraction patterns for each token pair
        let extractors: Vec<Regex> = config
            .iter_pairs()
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
            token_config: config,
            partial_json: PartialJson::default(),
            extractors,
        }
    }

    /// Extract JSON content from text, handling wrapper tokens if configured
    fn extract_json_content<'a>(&self, text: &'a str) -> &'a str {
        let mut content = text;

        // Try each extractor pattern (for tokens with both start and end)
        for extractor in &self.extractors {
            if let Some(captures) = extractor.captures(content) {
                if let Some(matched) = captures.get(1) {
                    return matched.as_str().trim();
                }
            }
        }

        // Handle special case where there's a start token but no end token
        for (start, end) in self.token_config.iter_pairs() {
            if !start.is_empty() && end.is_empty() {
                // Find the start token and extract everything after it
                if let Some(pos) = content.find(start) {
                    content = &content[pos + start.len()..];
                    return content.trim();
                }
            }
        }

        content.trim()
    }

    /// Try to extract a JSON object or array from text that may contain other content
    fn extract_json_from_text(&self, text: &str) -> Option<String> {
        // Look for JSON object starting with {
        if let Some(start) = text.find('{') {
            let mut depth = 0;
            let mut in_string = false;
            let mut escape_next = false;

            for (i, ch) in text[start..].char_indices() {
                if escape_next {
                    escape_next = false;
                    continue;
                }

                match ch {
                    '\\' if in_string => escape_next = true,
                    '"' if !in_string => in_string = true,
                    '"' if in_string => in_string = false,
                    '{' if !in_string => depth += 1,
                    '}' if !in_string => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(text[start..start + i + 1].to_string());
                        }
                    }
                    _ => {}
                }
            }
        }

        // Look for JSON array starting with [
        if let Some(start) = text.find('[') {
            let mut depth = 0;
            let mut in_string = false;
            let mut escape_next = false;

            for (i, ch) in text[start..].char_indices() {
                if escape_next {
                    escape_next = false;
                    continue;
                }

                match ch {
                    '\\' if in_string => escape_next = true,
                    '"' if !in_string => in_string = true,
                    '"' if in_string => in_string = false,
                    '[' if !in_string => depth += 1,
                    ']' if !in_string => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(text[start..start + i + 1].to_string());
                        }
                    }
                    _ => {}
                }
            }
        }

        None
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
        if self.token_config.start_tokens.is_empty() {
            // For JSON, we just need to see the start of an object or array
            return text.contains('{') || text.contains('[');
        }

        // Check for any start token
        self.token_config
            .start_tokens
            .iter()
            .any(|token| text.contains(token))
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
        // Check if we have multiple start tokens (e.g., multiple <|python_tag|> markers)
        if !self.token_config.start_tokens.is_empty() {
            let start_token = &self.token_config.start_tokens[0];
            if !start_token.is_empty() && text.matches(start_token).count() > 1 {
                // We have multiple occurrences of the start token
                let mut all_tools = Vec::new();
                let mut remaining = text;

                while let Some(start_pos) = remaining.find(start_token.as_str()) {
                    // Extract content after this start token
                    let after_token = &remaining[start_pos + start_token.len()..];

                    // Find where this JSON ends (look for the next start token or end of string)
                    let end_pos = if let Some(next_start) = after_token.find(start_token.as_str()) {
                        next_start
                    } else {
                        after_token.len()
                    };

                    let json_content = &after_token[..end_pos];

                    // Try to extract and parse JSON from this segment
                    if let Some(extracted) = self.extract_json_from_text(json_content) {
                        if let Ok(value) = serde_json::from_str::<Value>(&extracted) {
                            if let Ok(tools) = self.parse_json_value(&value) {
                                all_tools.extend(tools);
                            }
                        }
                    }

                    // Move to the next segment
                    remaining = &remaining[start_pos + start_token.len() + end_pos..];
                    if remaining.is_empty() {
                        break;
                    }
                }

                if !all_tools.is_empty() {
                    return Ok(all_tools);
                }
            }
        }

        // Extract JSON content from wrapper tokens if present
        let json_content = self.extract_json_content(text);

        // Try to parse as JSON first
        match serde_json::from_str::<Value>(json_content) {
            Ok(value) => self.parse_json_value(&value),
            Err(_) => {
                // If parse failed, check if we have multiple JSON objects separated by the configured separator
                // This handles cases like: {"name": "func1", ...};{"name": "func2", ...}
                if !self.token_config.separator.is_empty()
                    && json_content.contains(&self.token_config.separator)
                {
                    let mut all_tools = Vec::new();

                    // Split by separator and try to parse each part
                    let parts: Vec<&str> =
                        json_content.split(&self.token_config.separator).collect();
                    for part in parts {
                        let trimmed = part.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        // Try to parse this part as JSON
                        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
                            if let Ok(tools) = self.parse_json_value(&value) {
                                all_tools.extend(tools);
                            }
                        } else if let Some(extracted) = self.extract_json_from_text(trimmed) {
                            // Try extracting JSON from this part
                            if let Ok(value) = serde_json::from_str::<Value>(&extracted) {
                                if let Ok(tools) = self.parse_json_value(&value) {
                                    all_tools.extend(tools);
                                }
                            }
                        }
                    }

                    if !all_tools.is_empty() {
                        return Ok(all_tools);
                    }
                }

                // If no wrapper tokens configured and parse failed,
                // try to extract JSON from mixed text
                if self.token_config.start_tokens.is_empty() {
                    if let Some(extracted) = self.extract_json_from_text(text) {
                        if let Ok(value) = serde_json::from_str::<Value>(&extracted) {
                            return self.parse_json_value(&value);
                        }
                    }
                }
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

        // Extract JSON content first to check for separators
        let extracted_json = self.extract_json_content(&state.buffer);

        // Handle multiple JSON objects with separators
        // Check if we have a separator and potentially multiple JSON objects
        let separator = &self.token_config.separator;
        if !separator.is_empty() && extracted_json.contains(separator.as_str()) {
            // Try to find a complete JSON object before the separator
            if let Some(separator_pos) = extracted_json.find(separator.as_str()) {
                // Get JSON before separator
                let before_separator = &extracted_json[..separator_pos];

                // Try to parse the JSON before the separator
                match serde_json::from_str::<Value>(before_separator) {
                    Ok(value) => {
                        // Parse tool calls from this JSON
                        let tools = self.parse_json_value(&value)?;
                        if !tools.is_empty() {
                            // We need to figure out how much to remove from the original buffer
                            // Find where the separator is in the original buffer and remove up to and including it
                            if let Some(sep_in_original) = state.buffer.find(separator.as_str()) {
                                let remaining =
                                    state.buffer[sep_in_original + separator.len()..].to_string();
                                state.buffer = remaining;
                            }

                            // Return the first tool as complete
                            if let Some(tool) = tools.into_iter().next() {
                                return Ok(StreamResult::ToolComplete(tool));
                            }
                        }
                    }
                    Err(_) => {
                        // Failed to parse, continue to try other methods
                    }
                }
            }
        }

        // Handle multiple start tokens (e.g., multiple <|python_tag|> markers)
        if !self.token_config.start_tokens.is_empty() {
            let start_token = &self.token_config.start_tokens[0];
            if !start_token.is_empty() {
                // Find all occurrences of start token
                let occurrences: Vec<_> =
                    state.buffer.match_indices(start_token.as_str()).collect();
                if occurrences.len() > 1 {
                    // We have multiple start tokens, try to process the first complete one
                    let first_pos = occurrences[0].0;
                    let second_pos = occurrences[1].0;

                    // Extract content between first and second start token
                    let first_json_section = &state.buffer[first_pos..second_pos];
                    let json_content = self.extract_json_content(first_json_section);

                    // Try to parse this as complete JSON
                    if let Ok(value) = serde_json::from_str::<Value>(json_content) {
                        // Parse tool calls from this JSON
                        let tools = self.parse_json_value(&value)?;
                        if !tools.is_empty() {
                            // Remove the processed section from buffer
                            let remaining = state.buffer[second_pos..].to_string();
                            state.buffer = remaining;

                            // Return the first tool as complete
                            if let Some(tool) = tools.into_iter().next() {
                                return Ok(StreamResult::ToolComplete(tool));
                            }
                        }
                    }
                }
            }
        }

        // Regular single JSON parsing
        // Extract JSON content
        let json_content = self.extract_json_content(&state.buffer);

        // Try to parse with partial JSON parser
        match self.partial_json.parse_value(json_content) {
            Ok((value, consumed)) => {
                // Check if we have a complete JSON structure
                if consumed == json_content.len() {
                    // Check if this is truly complete or just has null from incomplete parsing
                    // We need to ensure the JSON actually ends properly (not cut off mid-key)
                    let trimmed = json_content.trim();
                    let looks_complete = trimmed.ends_with('}') || trimmed.ends_with(']');

                    if looks_complete {
                        // Complete JSON, parse tool calls
                        let tools = self.parse_json_value(&value)?;
                        if !tools.is_empty() {
                            // Clear buffer since we consumed everything
                            state.buffer.clear();

                            // Return the first tool as complete
                            // TODO simplified version, address more complex version
                            if let Some(tool) = tools.into_iter().next() {
                                return Ok(StreamResult::ToolComplete(tool));
                            }
                        }
                    }
                } else {
                    // Partial JSON, try to extract tool name
                    if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
                        // TODO simplified version, address more complex version
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
                            if let Some(obj) = v.as_object() {
                                obj.contains_key("name") || obj.contains_key("function")
                            } else {
                                false
                            }
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
        let parser = JsonParser::with_config(TokenConfig {
            start_tokens: vec!["<tool>".to_string()],
            end_tokens: vec!["</tool>".to_string()],
            separator: ", ".to_string(),
        });

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
        // Just verify that streaming eventually produces a complete tool call
        let parser = JsonParser::new();
        let mut state = ParseState::new();

        // Send complete JSON in one go
        // TODO simplified version, address more complex version
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
