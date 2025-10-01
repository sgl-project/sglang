use async_trait::async_trait;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Mistral format parser for tool calls
///
/// Handles the Mistral-specific format:
/// `[TOOL_CALLS] [{"name": "func", "arguments": {...}}, ...]`
///
/// Features:
/// - Bracket counting for proper JSON array extraction
/// - Support for multiple tool calls in a single array
/// - String-aware parsing to handle nested brackets in JSON
pub struct MistralParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
}

impl MistralParser {
    /// Create a new Mistral parser
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
        }
    }

    /// Extract JSON array using bracket counting
    ///
    /// Handles nested brackets in JSON content by tracking:
    /// - String boundaries (quotes)
    /// - Escape sequences
    /// - Bracket depth
    fn extract_json_array<'a>(&self, text: &'a str) -> Option<&'a str> {
        self.extract_json_array_with_pos(text).map(|(_, json)| json)
    }

    fn extract_json_array_with_pos<'a>(&self, text: &'a str) -> Option<(usize, &'a str)> {
        const BOT_TOKEN: &str = "[TOOL_CALLS] [";

        // Find the start of the token
        let start_idx = text.find(BOT_TOKEN)?;

        // Start from the opening bracket after [TOOL_CALLS]
        // The -1 is to include the opening bracket that's part of the token
        let json_start = start_idx + BOT_TOKEN.len() - 1;

        let mut bracket_count = 0;
        let mut in_string = false;
        let mut escape_next = false;

        let bytes = text.as_bytes();

        for i in json_start..text.len() {
            let char = bytes[i];

            if escape_next {
                escape_next = false;
                continue;
            }

            if char == b'\\' {
                escape_next = true;
                continue;
            }

            if char == b'"' && !escape_next {
                in_string = !in_string;
                continue;
            }

            if !in_string {
                if char == b'[' {
                    bracket_count += 1;
                } else if char == b']' {
                    bracket_count -= 1;
                    if bracket_count == 0 {
                        // Found the matching closing bracket
                        return Some((start_idx, &text[json_start..=i]));
                    }
                }
            }
        }

        // Incomplete array (no matching closing bracket found)
        None
    }

    /// Parse tool calls from a JSON array
    fn parse_json_array(&self, json_str: &str) -> ToolParserResult<Vec<ToolCall>> {
        let value: Value = serde_json::from_str(json_str)
            .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

        let mut tools = Vec::new();

        if let Value::Array(arr) = value {
            for (index, item) in arr.iter().enumerate() {
                if let Some(tool) = self.parse_single_object(item, index)? {
                    tools.push(tool);
                }
            }
        } else {
            // Single object case (shouldn't happen with Mistral format, but handle it)
            if let Some(tool) = self.parse_single_object(&value, 0)? {
                tools.push(tool);
            }
        }

        Ok(tools)
    }

    /// Parse a single JSON object into a ToolCall
    fn parse_single_object(&self, obj: &Value, index: usize) -> ToolParserResult<Option<ToolCall>> {
        let name = obj.get("name").and_then(|v| v.as_str());

        if let Some(name) = name {
            // Get arguments - Mistral uses "arguments" key
            let empty_obj = Value::Object(serde_json::Map::new());
            let args = obj.get("arguments").unwrap_or(&empty_obj);

            // Convert arguments to JSON string
            let arguments = serde_json::to_string(args)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate ID with index for multiple tools
            let id = format!("mistral_call_{}", index);

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

    /// Check if text contains Mistral tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("[TOOL_CALLS]")
    }
}

impl Default for MistralParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for MistralParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Mistral format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Extract JSON array from Mistral format with position
        if let Some((start_idx, json_array)) = self.extract_json_array_with_pos(text) {
            // Extract normal text before BOT_TOKEN
            let normal_text_before = if start_idx > 0 {
                text[..start_idx].to_string()
            } else {
                String::new()
            };

            match self.parse_json_array(json_array) {
                Ok(tools) => Ok((normal_text_before, tools)),
                Err(e) => {
                    // If JSON parsing fails, return the original text as normal text
                    tracing::warn!("Failed to parse tool call: {}", e);
                    Ok((text.to_string(), vec![]))
                }
            }
        } else {
            // Markers present but no complete array found
            Ok((text.to_string(), vec![]))
        }
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamingParseResult> {
        state.buffer.push_str(chunk);
        let mut result = StreamingParseResult::new();

        const BOT_TOKEN: &str = "[TOOL_CALLS] ";

        // Phase 1: Check for normal text before tool markers
        if !state.in_tool_section {
            if let Some(token_pos) = state.buffer.find(BOT_TOKEN) {
                if token_pos > 0 {
                    result.normal_text = state.buffer.drain(..token_pos).collect();
                    state.in_tool_section = true;
                    return Ok(result);
                }
                state.in_tool_section = true;
                // Remove the [TOOL_CALLS] token from buffer
                state.buffer.drain(..BOT_TOKEN.len());
            } else {
                // Check if we might have a partial token at the end
                if self.has_partial_token(&state.buffer) {
                    return Ok(result);
                }

                // No tool marker, return as normal text
                result.normal_text = std::mem::take(&mut state.buffer);
                return Ok(result);
            }
        }

        // Phase 2: Process tool calls array
        if state.in_tool_section {
            self.process_tool_array(state, &mut result)?;
        }

        Ok(result)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}

impl MistralParser {
    fn has_partial_token(&self, buffer: &str) -> bool {
        const BOT_TOKEN: &str = "[TOOL_CALLS] ";
        for i in 1..BOT_TOKEN.len().min(buffer.len()) {
            if BOT_TOKEN.starts_with(&buffer[buffer.len() - i..]) {
                return true;
            }
        }
        false
    }

    fn process_tool_array(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try to parse partial JSON array
        match self.partial_json.parse_value(&state.buffer) {
            Ok((value, consumed)) => {
                if let Some(array) = value.as_array() {
                    // Process each tool in the array
                    for (idx, tool_value) in array.iter().enumerate() {
                        self.process_single_tool(idx, tool_value, state, result)?;
                    }
                }

                // Check if we've consumed everything
                if consumed == state.buffer.len() && state.buffer.ends_with(']') {
                    // Tools complete, clear buffer
                    state.buffer.clear();
                    state.mode = crate::tool_parser::state::ParseMode::Complete;
                }
            }
            Err(_) => {
                // Incomplete JSON, try to extract partial tool info
                self.try_extract_partial_tools(state, result)?;
            }
        }

        Ok(())
    }

    fn process_single_tool(
        &self,
        index: usize,
        tool_value: &Value,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Ensure we have a partial tool entry
        let partial = state.ensure_partial_tool(index);

        // Extract and send tool name if not sent
        if !partial.name_sent {
            if let Some(name) = tool_value.get("name").and_then(|v| v.as_str()) {
                partial.name = Some(name.to_string());
                partial.id = Some(format!("mistral_call_{}", uuid::Uuid::new_v4()));
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
            if let Some(args_value) = tool_value.get("arguments") {
                let args_str = serde_json::to_string(args_value)?;

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

    fn try_extract_partial_tools(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try to extract partial tool info using bracket counting
        let mut in_string = false;
        let mut escape = false;
        let mut depth = 0;
        let mut object_start = None;
        let mut tool_index = 0;

        // Clone buffer to avoid borrow issues
        let buffer = state.buffer.clone();
        for (i, ch) in buffer.char_indices() {
            if escape {
                escape = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape = true,
                '"' => in_string = !in_string,
                '{' if !in_string => {
                    if depth == 1 && object_start.is_none() {
                        object_start = Some(i);
                    }
                    depth += 1;
                }
                '}' if !in_string => {
                    depth -= 1;
                    if depth == 1 && object_start.is_some() {
                        // We have a complete object
                        let obj_str = &buffer[object_start.unwrap()..=i];
                        if let Ok(value) = serde_json::from_str::<Value>(obj_str) {
                            self.process_single_tool(tool_index, &value, state, result)?;
                            tool_index += 1;
                        }
                        object_start = None;
                    }
                }
                '[' if !in_string => depth += 1,
                ']' if !in_string => depth -= 1,
                _ => {}
            }
        }

        // Try to process partial object if we have one
        if let Some(start) = object_start {
            let partial_obj = &buffer[start..];
            if let Ok((value, _)) = self.partial_json.parse_value(partial_obj) {
                self.process_single_tool(tool_index, &value, state, result)?;
            }
        }

        Ok(())
    }

}
