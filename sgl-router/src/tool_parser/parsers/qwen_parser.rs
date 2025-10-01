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
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Qwen format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where the first tool call begins
        let idx = text.find("<tool_call>").unwrap(); // Safe because has_tool_markers checked
        let normal_text = text[..idx].to_string();

        // Extract tool calls
        let mut tools = Vec::new();
        for (index, captures) in self.extractor.captures_iter(text).enumerate() {
            if let Some(json_str) = captures.get(1) {
                let parsed = serde_json::from_str::<Value>(json_str.as_str().trim())
                    .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))
                    .and_then(|v| self.parse_single_object(&v, index));

                match parsed {
                    Ok(Some(tool)) => tools.push(tool),
                    Ok(None) => continue,
                    Err(e) => {
                        tracing::warn!("Failed to parse tool call {}: {:?}", index, e);
                        continue;
                    }
                }
            }
        }

        // If no tools were successfully parsed despite having markers, return entire text as fallback
        if tools.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, tools))
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
            // No tool markers detected - return all buffered content as normal text
            let normal_text = std::mem::take(&mut state.buffer);
            return Ok(StreamResult::NormalText(normal_text));
        }

        // Check for text before tool markers and extract it as normal text
        if let Some(marker_pos) = state.buffer.find("<tool_call>") {
            if marker_pos > 0 {
                // We have text before the tool marker - extract it as normal text
                let normal_text: String = state.buffer.drain(..marker_pos).collect();
                return Ok(StreamResult::NormalText(normal_text));
            }
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
                        // JSON parsing failed, might be incomplete or malformed
                        // If we have what looks like a complete tool call block, treat as normal text
                        if state.buffer[start_pos..end_pos].contains("\n</tool_call>") {
                            let malformed_text: String = state.buffer.drain(..end_pos).collect();
                            return Ok(StreamResult::NormalText(malformed_text));
                        }
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
        self.has_tool_markers(text)
    }
}
