use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
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
    ) -> ToolParserResult<StreamingParseResult> {
        state.buffer.push_str(chunk);
        let mut result = StreamingParseResult::new();

        // Check for partial end token
        if self.ends_with_partial_token(&state.buffer).is_some() {
            return Ok(result);
        }

        // Phase 1: Check for normal text before tool markers
        if !state.in_tool_section {
            if let Some(marker_pos) = state.buffer.find("<tool_call>") {
                if marker_pos > 0 {
                    result.normal_text = state.buffer.drain(..marker_pos).collect();
                    state.in_tool_section = true;
                    return Ok(result);
                }
                state.in_tool_section = true;
            } else if !self.has_partial_marker(&state.buffer) {
                result.normal_text = std::mem::take(&mut state.buffer);
                return Ok(result);
            }
        }

        // Phase 2: Process tool calls
        if state.in_tool_section {
            self.process_tool_calls(state, &mut result)?;
        }

        Ok(result)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}

impl QwenParser {
    fn has_partial_marker(&self, buffer: &str) -> bool {
        // Check if buffer ends with partial tool marker
        buffer.ends_with("<tool") ||
        buffer.ends_with("<tool_") ||
        buffer.ends_with("<tool_c") ||
        buffer.ends_with("<tool_ca") ||
        buffer.ends_with("<tool_cal")
    }

    fn process_tool_calls(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        let start_marker = "<tool_call>\n";
        let end_marker = "\n</tool_call>";

        if let Some(start_pos) = state.buffer.find(start_marker) {
            let json_start = start_pos + start_marker.len();

            if let Some(end_pos) = state.buffer[json_start..].find(end_marker) {
                // Complete tool call found
                let json_str = state.buffer[json_start..json_start + end_pos].to_string();

                match serde_json::from_str::<Value>(&json_str) {
                    Ok(value) => {
                        let index = state.partial_tools.len();
                        self.process_tool_json(&value, index, state, result)?;
                        state.buffer.drain(..json_start + end_pos + end_marker.len());
                    }
                    Err(_) => {
                        // Malformed JSON, treat as text
                        let text: String = state.buffer.drain(
                            ..json_start + end_pos + end_marker.len()
                        ).collect();
                        result.normal_text = text;
                    }
                }
            } else {
                // Incomplete tool call, try partial parsing
                let partial_json = state.buffer[json_start..].to_string();
                self.process_partial_json(&partial_json, state, result)?;
            }
        }

        Ok(())
    }

    fn process_tool_json(
        &self,
        value: &Value,
        index: usize,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Ensure we have partial tool entry
        let partial = state.ensure_partial_tool(index);

        // Extract and send tool name if not sent
        if !partial.name_sent {
            if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
                partial.name = Some(name.to_string());
                partial.id = Some(format!("qwen_call_{}", uuid::Uuid::new_v4()));
                partial.name_sent = true;

                result.tool_calls.push(ToolCallItem {
                    tool_index: index,
                    id: partial.id.clone(),
                    name: partial.name.clone(),
                    arguments_delta: String::new(),
                });
                return Ok(());
            }
        }

        // Stream arguments
        if partial.name_sent {
            if let Some(args) = value.get("arguments") {
                let args_str = serde_json::to_string(args)?;

                if args_str.len() > partial.streamed_arguments.len() {
                    let delta = &args_str[partial.streamed_arguments.len()..];

                    result.tool_calls.push(ToolCallItem {
                        tool_index: index,
                        id: None,
                        name: None,
                        arguments_delta: delta.to_string(),
                    });

                    partial.streamed_arguments = args_str;
                }
            }
        }

        Ok(())
    }

    fn process_partial_json(
        &self,
        partial_json: &str,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Remove trailing newline if present (might be start of end token)
        let partial_json = partial_json.trim_end();

        // Try to parse with partial JSON parser
        match self.partial_json.parse_value(partial_json) {
            Ok((value, _consumed)) => {
                let index = state.partial_tools.len();

                // Ensure we have partial tool entry
                let partial = state.ensure_partial_tool(index);

                // Extract tool name if available
                if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
                    if !partial.name_sent {
                        partial.name = Some(name.to_string());
                        partial.id = Some(format!("qwen_call_{}", uuid::Uuid::new_v4()));
                        partial.name_sent = true;

                        result.tool_calls.push(ToolCallItem {
                            tool_index: index,
                            id: partial.id.clone(),
                            name: partial.name.clone(),
                            arguments_delta: String::new(),
                        });
                    }

                    // Check for arguments
                    if let Some(args) = value.get("arguments") {
                        if let Ok(args_str) = serde_json::to_string(args) {
                            if args_str.len() > partial.streamed_arguments.len() {
                                let delta = &args_str[partial.streamed_arguments.len()..];

                                result.tool_calls.push(ToolCallItem {
                                    tool_index: index,
                                    id: None,
                                    name: None,
                                    arguments_delta: delta.to_string(),
                                });

                                partial.streamed_arguments = args_str;
                            }
                        }
                    }
                }
            }
            Err(_) => {
                // Failed to parse even as partial JSON - keep buffering
            }
        }

        Ok(())
    }

}
