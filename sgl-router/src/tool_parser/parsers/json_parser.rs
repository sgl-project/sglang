use async_trait::async_trait;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, StreamingParseResult, ToolCall, ToolCallItem},
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
    fn find_common_prefix(&self, prev: &str, current: &str) -> String {
        let mut common = String::new();
        let prev_chars: Vec<char> = prev.chars().collect();
        let current_chars: Vec<char> = current.chars().collect();

        for (i, &prev_char) in prev_chars.iter().enumerate() {
            if let Some(&current_char) = current_chars.get(i) {
                if prev_char == current_char {
                    common.push(prev_char);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        common
    }

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
        // Append new text to buffer
        state.buffer.push_str(chunk);
        let current_text = &state.buffer;

        // Check if current text has tool_call markers or we're continuing from previous tool
        if !self.has_tool_markers(current_text)
            && !(state.current_tool_id >= 0 && current_text.starts_with(", "))
        {
            // Only clear buffer if we're sure no tool call is starting
            if !self.has_partial_start_token(&state.buffer) {
                let normal_text = std::mem::take(&mut state.buffer);
                return Ok(StreamingParseResult::with_normal_text(normal_text));
            } else {
                // Might be partial token, keep buffering
                return Ok(StreamingParseResult::new());
            }
        }

        // Try to parse as partial JSON
        match self.partial_json.parse_value(current_text) {
            Ok((value, _consumed)) => {
                // Handle parameters/arguments consistency (match Python behavior)
                let mut current_tool_call = value;
                if let Some(obj) = current_tool_call.as_object_mut() {
                    if obj.contains_key("parameters") && !obj.contains_key("arguments") {
                        if let Some(params) = obj.remove("parameters") {
                            obj.insert("arguments".to_string(), params);
                        }
                    }
                }

                // Check if we have a function name
                if let Some(function_name) = current_tool_call.get("name").and_then(|v| v.as_str())
                {
                    // Case 1: Handle tool name streaming
                    if !state.current_tool_name_sent {
                        // Initialize tool tracking if this is the first tool
                        if state.current_tool_id == -1 {
                            state.current_tool_id = 0;
                            state.streamed_args_for_tool.push(String::new());
                        }
                        // Ensure streamed_args_for_tool is large enough
                        while state.streamed_args_for_tool.len() <= state.current_tool_id as usize {
                            state.streamed_args_for_tool.push(String::new());
                        }

                        // Send the tool name with empty parameters
                        let tool_call = ToolCallItem {
                            tool_index: state.current_tool_id as usize,
                            name: Some(function_name.to_string()),
                            parameters: String::new(),
                        };
                        state.current_tool_name_sent = true;
                        return Ok(StreamingParseResult::with_tool_calls(vec![tool_call]));
                    }

                    // Case 2: Handle streaming arguments
                    if let Some(cur_arguments) = current_tool_call.get("arguments") {
                        let cur_args_json = serde_json::to_string(cur_arguments)
                            .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

                        // Check if this is a complete JSON structure
                        let is_current_complete = self.is_complete_json(current_text);

                        let mut argument_diff = None;

                        if is_current_complete {
                            // Send all remaining arguments for complete tool
                            let sent = state.streamed_args_for_tool[state.current_tool_id as usize].len();
                            argument_diff = Some(cur_args_json[sent..].to_string());

                            // Tool is complete - reset state for next tool
                            state.buffer.clear();
                            if (state.current_tool_id as usize) < state.prev_tool_call_arr.len() {
                                state.prev_tool_call_arr[state.current_tool_id as usize].clear();
                            }
                            state.current_tool_name_sent = false;
                            state.streamed_args_for_tool[state.current_tool_id as usize].clear();
                            state.current_tool_id += 1;
                        } else if !state.prev_tool_call_arr.is_empty()
                            && (state.current_tool_id as usize) < state.prev_tool_call_arr.len()
                        {
                            // For incomplete tool, calculate incremental changes
                            let prev_args = state.prev_tool_call_arr[state.current_tool_id as usize]
                                .get("arguments");
                            if let Some(prev_args) = prev_args {
                                let prev_args_json = serde_json::to_string(prev_args)
                                    .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;
                                if cur_args_json != prev_args_json {
                                    let prefix = self.find_common_prefix(&prev_args_json, &cur_args_json);
                                    let sent = state.streamed_args_for_tool[state.current_tool_id as usize].len();
                                    if prefix.len() > sent {
                                        argument_diff = Some(prefix[sent..].to_string());
                                    }
                                }
                            }
                        }

                        // Send the argument diff if there's something new
                        if let Some(diff) = argument_diff {
                            if !diff.is_empty() {
                                let tool_call = ToolCallItem {
                                    tool_index: state.current_tool_id as usize,
                                    name: None, // No name for argument deltas
                                    parameters: diff.clone(),
                                };
                                if !is_current_complete {
                                    state.streamed_args_for_tool[state.current_tool_id as usize] += &diff;
                                }
                                return Ok(StreamingParseResult::with_tool_calls(vec![tool_call]));
                            }
                        }

                        // Update prev_tool_call_arr with current state
                        if state.current_tool_id >= 0 {
                            // Ensure prev_tool_call_arr is large enough
                            while state.prev_tool_call_arr.len() <= state.current_tool_id as usize {
                                state.prev_tool_call_arr.push(serde_json::Map::new());
                            }
                            if let Some(obj) = current_tool_call.as_object() {
                                state.prev_tool_call_arr[state.current_tool_id as usize] = obj.clone();
                            }
                        }
                    }
                }
            }
            Err(_) => {
                // Failed to parse even as partial JSON - continue waiting for more data
            }
        }

        Ok(StreamingParseResult::new())
    }


    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}
