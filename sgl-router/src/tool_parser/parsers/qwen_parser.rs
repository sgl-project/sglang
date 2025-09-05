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

/// Qwen format parser for tool calls
///
/// Handles the Qwen 2.5/3 specific format:
/// `<tool_call>\n{"name": "func", "arguments": {...}}\n</tool_call>`
///
/// Features:
/// - XML-style tags with JSON content
/// - Support for multiple sequential tool calls
/// - Newline-aware parsing
pub struct QwenParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex for extracting tool calls
    extractor: Regex,
}

impl QwenParser {
    /// Create a new Qwen parser
    pub fn new() -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let pattern = r"(?s)<tool_call>\n(.*?)\n</tool_call>";
        let extractor = Regex::new(pattern).expect("Valid regex pattern");

        Self {
            partial_json: PartialJson::default(),
            extractor,
        }
    }

    /// Extract all tool call blocks from text
    fn extract_tool_calls<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.extractor
            .captures_iter(text)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str()))
            .collect()
    }

    /// Parse a single JSON object into a ToolCall
    fn parse_single_object(&self, obj: &Value, index: usize) -> ToolParserResult<Option<ToolCall>> {
        let name = obj.get("name").and_then(|v| v.as_str());

        if let Some(name) = name {
            // Get arguments - Qwen uses "arguments" key
            let empty_obj = Value::Object(serde_json::Map::new());
            let args = obj.get("arguments").unwrap_or(&empty_obj);

            // Convert arguments to JSON string
            let arguments = serde_json::to_string(args)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate ID with index for multiple tools
            let id = format!("qwen_call_{}", index);

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

    /// Check if text contains Qwen tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    /// Find the start position of a tool call
    fn find_tool_start(&self, text: &str) -> Option<usize> {
        text.find("<tool_call>\n")
    }

    /// Find the end position of a tool call
    fn find_tool_end(&self, text: &str, start_pos: usize) -> Option<usize> {
        let search_from = start_pos + "<tool_call>\n".len();
        text[search_from..]
            .find("\n</tool_call>")
            .map(|pos| search_from + pos + "\n</tool_call>".len())
    }

    /// Check if buffer ends with a partial token
    fn ends_with_partial_token(&self, buffer: &str) -> Option<usize> {
        // Check for partial start token
        let start_token = "<tool_call>\n";
        // Use inclusive range to check if entire buffer could be a prefix
        for i in 1..=start_token.len().min(buffer.len()) {
            if start_token.starts_with(&buffer[buffer.len() - i..]) {
                return Some(i);
            }
        }

        // Check for partial end token
        let end_token = "\n</tool_call>";
        // Only check if buffer ends with a partial match (not the complete token without newline)
        // If buffer ends with "</tool_call>", that's not a partial token - it's missing the newline
        if buffer.ends_with("</tool_call>") {
            // This is a complete end tag, just missing the leading newline
            // Not a partial token situation
            return None;
        }
        // Use inclusive range to check if entire buffer could be a prefix
        (1..=end_token.len().min(buffer.len()))
            .find(|&i| end_token.starts_with(&buffer[buffer.len() - i..]))
    }
}

impl Default for QwenParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for QwenParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<Vec<ToolCall>> {
        // Check if text contains Qwen format
        if !self.has_tool_markers(text) {
            return Ok(vec![]);
        }

        // Extract all tool call blocks
        let tool_blocks = self.extract_tool_calls(text);
        let mut tools = Vec::new();

        for (index, json_str) in tool_blocks.iter().enumerate() {
            // Parse each JSON block
            match serde_json::from_str::<Value>(json_str.trim()) {
                Ok(value) => {
                    if let Some(tool) = self.parse_single_object(&value, index)? {
                        tools.push(tool);
                    }
                }
                Err(_) => {
                    // Skip malformed JSON blocks
                    continue;
                }
            }
        }

        Ok(tools)
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        state.buffer.push_str(chunk);

        // Check for partial token at end of buffer
        if let Some(_partial_len) = self.ends_with_partial_token(&state.buffer) {
            // Hold back the partial token
            return Ok(StreamResult::Incomplete);
        }

        // Check if we have the start marker
        if !self.has_tool_markers(&state.buffer) {
            return Ok(StreamResult::Incomplete);
        }

        // Find start and end positions
        if let Some(start_pos) = self.find_tool_start(&state.buffer) {
            // Check if we have the complete tool call
            if let Some(end_pos) = self.find_tool_end(&state.buffer, start_pos) {
                // Extract the JSON content
                let json_start = start_pos + "<tool_call>\n".len();
                let json_end = end_pos - "\n</tool_call>".len();
                let json_str = &state.buffer[json_start..json_end];

                // Parse the complete JSON
                match serde_json::from_str::<Value>(json_str.trim()) {
                    Ok(value) => {
                        if let Some(tool) = self.parse_single_object(&value, 0)? {
                            // Clear the consumed part from buffer using drain for efficiency
                            state.buffer.drain(..end_pos);
                            return Ok(StreamResult::ToolComplete(tool));
                        }
                    }
                    Err(_) => {
                        // JSON parsing failed, might be incomplete
                    }
                }
            } else {
                // We have start but no end yet - try partial parsing
                let json_start = start_pos + "<tool_call>\n".len();
                let partial_json = &state.buffer[json_start..];

                // Remove trailing newline if present (might be start of end token)
                let partial_json = partial_json.trim_end();

                // Try to parse with partial JSON parser
                match self.partial_json.parse_value(partial_json) {
                    Ok((value, _consumed)) => {
                        // Extract tool name if available
                        if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
                            // Check if we've already sent the name
                            if !state.in_string {
                                state.in_string = true; // Use as flag for "name sent"
                                return Ok(StreamResult::ToolName {
                                    index: 0,
                                    name: name.to_string(),
                                });
                            }

                            // Check for arguments
                            if let Some(args) = value.get("arguments") {
                                if let Ok(args_str) = serde_json::to_string(args) {
                                    return Ok(StreamResult::ToolArguments {
                                        index: 0,
                                        arguments: args_str,
                                    });
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // Failed to parse even as partial JSON
                        // Keep buffering
                    }
                }
            }
        }

        Ok(StreamResult::Incomplete)
    }

    fn detect_format(&self, text: &str) -> bool {
        // Check if text contains Qwen-specific markers. If not, it's not this format.
        if !self.has_tool_markers(text) {
            return false;
        }

        // Try to extract tool calls to see if we have a complete, valid one.
        let tool_blocks = self.extract_tool_calls(text);
        for json_str in &tool_blocks {
            if let Ok(value) = serde_json::from_str::<Value>(json_str.trim()) {
                if let Some(obj) = value.as_object() {
                    if obj.contains_key("name") && obj.contains_key("arguments") {
                        // Found a valid, complete tool call.
                        return true;
                    }
                }
            }
        }

        // If we have the marker but no valid complete tool call,
        // it could be a partial stream. We should detect this as the format.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_qwen_format() {
        let parser = QwenParser::new();
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "Beijing", "units": "celsius"}}
</tool_call>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "get_weather");
        assert!(result[0].function.arguments.contains("Beijing"));
    }

    #[tokio::test]
    async fn test_parse_multiple_tools() {
        let parser = QwenParser::new();
        let input = r#"<tool_call>
{"name": "search", "arguments": {"query": "rust programming"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"expression": "2 + 2"}}
</tool_call>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].function.name, "search");
        assert_eq!(result[1].function.name, "calculate");
    }

    #[tokio::test]
    async fn test_with_normal_text() {
        let parser = QwenParser::new();
        let input = r#"Let me help you with that.
<tool_call>
{"name": "get_info", "arguments": {"topic": "Rust"}}
</tool_call>
Here are the results."#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "get_info");
    }

    #[tokio::test]
    async fn test_nested_json_structures() {
        let parser = QwenParser::new();
        let input = r#"<tool_call>
{
    "name": "process_data",
    "arguments": {
        "data": {
            "nested": {
                "array": [1, 2, 3],
                "object": {"key": "value"}
            }
        }
    }
}
</tool_call>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "process_data");
        assert!(result[0].function.arguments.contains("nested"));
    }

    #[test]
    fn test_detect_format() {
        let parser = QwenParser::new();

        assert!(parser.detect_format(
            r#"<tool_call>
{"name": "test", "arguments": {}}
</tool_call>"#
        ));

        assert!(parser.detect_format(
            r#"Text before <tool_call>
{"name": "test", "arguments": {}}
</tool_call> text after"#
        ));

        assert!(!parser.detect_format(r#"{"name": "test", "arguments": {}}"#));
        assert!(!parser.detect_format("plain text"));

        // Partial format should still be detected
        assert!(parser.detect_format("<tool_call>"));
    }

    #[tokio::test]
    async fn test_streaming_partial() {
        let parser = QwenParser::new();
        let mut state = ParseState::new();

        // Simulate streaming chunks
        let chunks = vec![
            "<tool_call>\n",
            r#"{"name": "search","#,
            r#" "arguments": {"query":"#,
            r#" "rust"}}"#,
            "\n</tool_call>",
        ];

        let mut found_name = false;
        let mut found_complete = false;

        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &mut state).await.unwrap();

            match result {
                StreamResult::ToolName { name, .. } => {
                    assert_eq!(name, "search");
                    found_name = true;
                }
                StreamResult::ToolComplete(tool) => {
                    assert_eq!(tool.function.name, "search");
                    found_complete = true;
                }
                _ => {}
            }
        }

        assert!(found_name || found_complete); // At least one should be found
    }
}
