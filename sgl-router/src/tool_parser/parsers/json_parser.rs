use async_trait::async_trait;
use serde_json::Value;
use tracing;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::{ParseState, ParseMode},
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// JSON format parser for tool calls
///
/// Handles pure JSON formats for function calling:
/// - Single tool call: {"name": "fn", "arguments": {...}}
/// - Multiple tool calls: [{"name": "fn1", "arguments": {...}}, ...]
/// - With parameters instead of arguments: {"name": "fn", "parameters": {...}}
pub struct JsonParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
}

impl JsonParser {
    /// Create a new JSON parser
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
        }
    }

    /// Try to extract a first valid JSON object or array from text that may contain other content
    /// Returns (json_string, normal_text) where normal_text is text before and after the JSON
    fn extract_json_from_text(&self, text: &str) -> Option<(String, String)> {
        let mut in_string = false;
        let mut escape = false;
        let mut stack: Vec<char> = Vec::with_capacity(8);
        let mut start: Option<usize> = None;

        for (i, ch) in text.char_indices() {
            if escape {
                escape = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape = true,
                '"' => in_string = !in_string,
                _ if in_string => {}
                '{' | '[' => {
                    if start.is_none() {
                        start = Some(i);
                    }
                    stack.push(ch);
                }
                '}' | ']' => {
                    let Some(open) = stack.pop() else {
                        // Stray closer - reset and continue looking for next valid JSON
                        start = None;
                        continue;
                    };

                    let valid = (open == '{' && ch == '}') || (open == '[' && ch == ']');
                    if !valid {
                        // Mismatch - reset and continue looking
                        start = None;
                        stack.clear();
                        continue;
                    }

                    if stack.is_empty() {
                        let s = start.unwrap();
                        let e = i + ch.len_utf8();
                        let potential_json = &text[s..e];

                        // Validate that this is actually valid JSON before returning
                        if serde_json::from_str::<Value>(potential_json).is_ok() {
                            let json = potential_json.to_string();
                            let normal = format!("{}{}", &text[..s], &text[e..]);
                            return Some((json, normal));
                        } else {
                            // Not valid JSON, reset and continue looking
                            start = None;
                            continue;
                        }
                    }
                }
                _ => {}
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

    /// Check if text contains JSON tool call markers (complete markers)
    fn has_tool_markers(&self, text: &str) -> bool {
        (text.contains('{') || text.contains('[')) && text.contains("name")
    }

    /// Check if buffer could be building toward a tool call pattern
    fn has_partial_start_token(&self, buffer: &str) -> bool {
        // Check if buffer ends with a partial match of tool call patterns
        let patterns = [r#"{"name""#, r#"[{"name""#];

        for pattern in &patterns {
            // Check if buffer ends with any partial of this pattern
            for i in 1..=buffer.len().min(pattern.len()) {
                if pattern.starts_with(&buffer[buffer.len() - i..]) {
                    return true;
                }
            }
        }
        false
    }

    /// Find common prefix between two JSON strings (helper for incremental args)

    /// Check if the current text represents a complete JSON structure
    fn is_complete_json(&self, text: &str) -> bool {
        serde_json::from_str::<Value>(text).is_ok()
    }
}

impl Default for JsonParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for JsonParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Always use extract_json_from_text to handle both pure JSON and mixed content
        if let Some((extracted_json, normal_text)) = self.extract_json_from_text(text) {
            let parsed = serde_json::from_str::<Value>(&extracted_json)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))
                .and_then(|v| self.parse_json_value(&v));

            match parsed {
                Ok(tools) => return Ok((normal_text, tools)),
                Err(e) => tracing::warn!("parse_complete failed: {:?}", e),
            }
        }

        // No valid JSON found, return original text as normal text
        Ok((text.to_string(), vec![]))
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamingParseResult> {
        state.buffer.push_str(chunk);
        let mut result = StreamingParseResult::new();

        // Phase 1: Check for normal text before JSON
        if !state.in_tool_section {
            // Try to find JSON start
            if let Some(json_start) = self.find_json_start(&state.buffer) {
                if json_start > 0 {
                    result.normal_text = state.buffer.drain(..json_start).collect();
                    state.in_tool_section = true;
                    return Ok(result);
                }
                state.in_tool_section = true;
            } else if !self.might_be_json_start(&state.buffer) {
                // No JSON structure found, return as normal text
                result.normal_text = std::mem::take(&mut state.buffer);
                return Ok(result);
            }
        }

        // Phase 2: Process JSON tool calls
        if state.in_tool_section {
            self.process_json_tools(state, &mut result)?;
        }

        Ok(result)
    }


    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}

impl JsonParser {
    fn find_json_start(&self, buffer: &str) -> Option<usize> {
        // Look for start of JSON object or array
        for (i, ch) in buffer.char_indices() {
            if ch == '{' || ch == '[' {
                return Some(i);
            }
        }
        None
    }

    fn might_be_json_start(&self, buffer: &str) -> bool {
        // Check if buffer might be leading up to JSON
        buffer.trim().is_empty() || buffer.ends_with('{') || buffer.ends_with('[')
    }

    fn process_json_tools(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try to parse the buffer as partial JSON
        match self.partial_json.parse_value(&state.buffer) {
            Ok((value, consumed)) => {
                let is_array = value.is_array();

                // Process as array or single object
                if is_array {
                    // When processing an array, each item gets its index from the array position
                    let array = value.as_array().unwrap();
                    for (index, tool_value) in array.iter().enumerate() {
                        self.process_single_tool(index, tool_value, state, result, consumed)?;
                    }

                    // Check if we've consumed everything and JSON is complete
                    if consumed == state.buffer.len() && self.is_complete_json(&state.buffer) {
                        state.buffer.clear();
                        state.mode = ParseMode::Complete;
                        // For arrays, set current_tool_id to the number of tools processed
                        state.current_tool_id = array.len();
                    }
                } else {
                    // For single tool calls, use current_tool_id
                    self.process_single_tool(state.current_tool_id, &value, state, result, consumed)?;

                    // Check if we've consumed everything and JSON is complete
                    if consumed == state.buffer.len() && self.is_complete_json(&state.buffer) {
                        state.buffer.clear();
                        state.mode = ParseMode::Complete;
                        // Increment tool ID for the next single tool
                        state.current_tool_id += 1;
                    }
                }
            }
            Err(e) => {
                // Error means malformed JSON, not just incomplete
                tracing::warn!("Failed to parse JSON tool calls: {}", e);
                // Clear buffer to avoid getting stuck on bad JSON
                state.buffer.clear();
                state.in_tool_section = false;
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
        consumed: usize,
    ) -> ToolParserResult<()> {
        // Check if complete before borrowing state mutably
        let buffer_len = state.buffer.len();
        let is_complete = consumed == buffer_len && self.is_complete_json(&state.buffer);

        // Ensure we have a partial tool entry
        let partial = state.ensure_partial_tool(index);

        // Extract and send tool name if not sent
        if !partial.name_sent {
            if let Some(name) = tool_value.get("name")
                .or_else(|| tool_value.get("function"))
                .and_then(|v| v.as_str()) {

                // Save the name but don't send yet
                partial.name = Some(name.to_string());

                // Only send the name if:
                // 1. The JSON is complete, OR
                // 2. We have arguments/parameters (which means name is definitely complete)
                let has_args = tool_value.get("arguments").is_some()
                    || tool_value.get("parameters").is_some();

                if is_complete || has_args {
                    partial.id = Some(format!("call_{}", uuid::Uuid::new_v4()));
                    partial.name_sent = true;

                    result.tool_calls.push(ToolCallItem {
                        tool_index: index,
                        id: partial.id.clone(),
                        name: partial.name.clone(),
                        arguments_delta: String::new(),
                    });
                }
            }
        }

        // Get partial tool again
        let partial = state.ensure_partial_tool(index);

        // Handle arguments
        if let Some(args_value) = tool_value.get("arguments")
            .or_else(|| tool_value.get("parameters")) {

            // Skip if arguments are null (partial parser returns null for incomplete)
            if args_value.is_null() {
                return Ok(());
            }


            // Serialize current arguments
            let cur_args_json = serde_json::to_string(args_value)?;

            // How much have we already sent?
            let sent = partial.streamed_arguments.len();

            let argument_diff = if is_complete {
                // If complete, send everything from position 'sent' onward
                if cur_args_json.len() > sent {
                    cur_args_json[sent..].to_string()
                } else {
                    String::new()
                }
            } else if !partial.arguments_buffer.is_empty() {
                // If incomplete and we have previous arguments, find common prefix
                if cur_args_json != partial.arguments_buffer {
                    let prefix = self.find_common_prefix(&partial.arguments_buffer, &cur_args_json);
                    if prefix.len() > sent {
                        prefix[sent..].to_string()
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else {
                // First time with incomplete JSON - DON'T send anything yet, just save state
                String::new()
            };

            if !argument_diff.is_empty() {
                // Update what we've sent
                partial.streamed_arguments.push_str(&argument_diff);

                result.tool_calls.push(ToolCallItem {
                    tool_index: index,
                    id: None,
                    name: None,
                    arguments_delta: argument_diff,
                });
            }

            // Save current args for next comparison
            partial.arguments_buffer = cur_args_json;
        }

        Ok(())
    }

    fn find_common_prefix(&self, prev: &str, current: &str) -> String {
        let min_len = prev.len().min(current.len());
        for i in 0..min_len {
            if prev.as_bytes()[i] != current.as_bytes()[i] {
                return current[..i].to_string();
            }
        }
        current[..min_len].to_string()
    }
}
