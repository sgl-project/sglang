use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::protocols::spec::Tool;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
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

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Flag for whether current tool's name has been sent to client
    current_tool_name_sent: bool,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Token configuration
    bot_token: &'static str,
    eot_token: &'static str,
    tool_call_separator: &'static str,
}

impl JsonParser {
    /// Create a new JSON parser
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            bot_token: "[",
            eot_token: "]",
            tool_call_separator: ",",
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

    /// Get a mapping of tool names to their indices
    fn get_tool_indices(&self, tools: &[Tool]) -> HashMap<String, usize> {
        tools
            .iter()
            .enumerate()
            .map(|(i, tool)| (tool.function.name.clone(), i))
            .collect()
    }

    /// Check if buffer ends with a partial bot_token
    fn ends_with_partial_token(&self, buffer: &str, bot_token: &str) -> bool {
        if bot_token.is_empty() {
            return false;
        }

        for i in 1..=buffer.len().min(bot_token.len()) {
            if let Some(buffer_end) = buffer.get(buffer.len() - i..) {
                if bot_token.starts_with(buffer_end) {
                    return true;
                }
            }
        }
        false
    }

    /// Ensure arrays have enough capacity for current_tool_id
    fn ensure_capacity(&mut self) {
        if self.current_tool_id < 0 {
            return;
        }

        let needed_len = (self.current_tool_id + 1) as usize;

        if self.prev_tool_call_arr.len() < needed_len {
            self.prev_tool_call_arr
                .resize_with(needed_len, || Value::Null);
        }

        if self.streamed_args_for_tool.len() < needed_len {
            self.streamed_args_for_tool
                .resize_with(needed_len, String::new);
        }
    }

    /// Check if text contains tool calls
    fn has_tool_call(&self, text: &str) -> bool {
        text.contains('[') || text.contains('{')
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
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ToolParserResult<StreamingParseResult> {
        // Append new text to buffer
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if current_text has tool_call
        let has_tool_start = self.has_tool_call(current_text)
            || (self.current_tool_id >= 0 && current_text.starts_with(self.tool_call_separator));

        if !has_tool_start {
            // Only clear buffer if we're sure no tool call is starting
            if !self.ends_with_partial_token(&self.buffer, self.bot_token) {
                let normal_text = self.buffer.clone();
                self.buffer.clear();

                // Remove eot_token if present
                let normal_text = if !self.eot_token.is_empty() {
                    normal_text.replace(self.eot_token, "")
                } else {
                    normal_text
                };

                return Ok(StreamingParseResult {
                    normal_text,
                    calls: vec![],
                });
            } else {
                // Might be partial bot_token, keep buffering
                return Ok(StreamingParseResult::default());
            }
        }

        // Build tool indices
        let tool_indices = self.get_tool_indices(tools);

        // Determine start index for JSON parsing
        let start_idx = if let Some(pos) = current_text.find(self.bot_token) {
            pos + self.bot_token.len()
        } else if self.current_tool_id >= 0 && current_text.starts_with(self.tool_call_separator) {
            self.tool_call_separator.len()
        } else {
            0
        };

        if start_idx >= current_text.len() {
            return Ok(StreamingParseResult::default());
        }

        // Parse partial JSON
        let json_str = &current_text[start_idx..];

        let (obj, end_idx) = match self.partial_json.parse_value(json_str) {
            Ok(result) => result,
            Err(_) => {
                return Ok(StreamingParseResult::default());
            }
        };

        // Check if JSON is complete
        let is_complete =
            end_idx == json_str.len() && serde_json::from_str::<Value>(json_str).is_ok();

        // Validate tool name if present
        if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
            if !tool_indices.contains_key(name) {
                // Invalid tool name - reset state
                self.buffer.clear();
                self.current_tool_id = -1;
                self.current_tool_name_sent = false;
                if !self.streamed_args_for_tool.is_empty() {
                    self.streamed_args_for_tool.pop();
                }
                return Ok(StreamingParseResult::default());
            }
        }

        // Handle parameters/arguments aliasing
        let current_tool_call = if obj.get("arguments").is_none() {
            if let Some(params) = obj.get("parameters") {
                let mut cloned = obj.clone();
                if let Value::Object(ref mut map) = cloned {
                    map.insert("arguments".to_string(), params.clone());
                }
                cloned
            } else {
                obj.clone()
            }
        } else {
            obj.clone()
        };

        let mut result = StreamingParseResult::default();

        // Case 1: Handle tool name streaming
        if !self.current_tool_name_sent {
            if let Some(function_name) = current_tool_call.get("name").and_then(|v| v.as_str()) {
                if tool_indices.contains_key(function_name) {
                    // Initialize if first tool
                    if self.current_tool_id == -1 {
                        self.current_tool_id = 0;
                        self.streamed_args_for_tool.push(String::new());
                    } else if self.current_tool_id as usize >= self.streamed_args_for_tool.len() {
                        // Ensure capacity for subsequent tools
                        self.ensure_capacity();
                    }

                    // Send tool name with empty parameters
                    self.current_tool_name_sent = true;
                    result.calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: Some(function_name.to_string()),
                        parameters: String::new(),
                    });
                }
            }
        }
        // Case 2: Handle streaming arguments
        else {
            if let Some(cur_arguments) = current_tool_call.get("arguments") {
                let tool_id = self.current_tool_id as usize;
                let sent = self
                    .streamed_args_for_tool
                    .get(tool_id)
                    .map(|s| s.len())
                    .unwrap_or(0);
                let cur_args_json = serde_json::to_string(cur_arguments)
                    .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

                let prev_arguments = self
                    .prev_tool_call_arr
                    .get(tool_id)
                    .and_then(|v| v.get("arguments"));

                let mut argument_diff: Option<String> = None;

                // If JSON is complete, send all remaining arguments
                if is_complete {
                    argument_diff = Some(cur_args_json[sent..].to_string());

                    // Remove processed portion, keep unprocessed content
                    self.buffer = current_text[start_idx + end_idx..].to_string();

                    // Clear completed tool data
                    if tool_id < self.prev_tool_call_arr.len() {
                        self.prev_tool_call_arr[tool_id] = Value::Null;
                    }
                    self.current_tool_name_sent = false;
                    if tool_id < self.streamed_args_for_tool.len() {
                        self.streamed_args_for_tool[tool_id].clear();
                    }
                    self.current_tool_id += 1;
                }
                // If still parsing, send incremental changes
                else if let Some(prev_args) = prev_arguments {
                    let prev_args_json = serde_json::to_string(prev_args)
                        .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

                    if cur_args_json != prev_args_json {
                        // Find common prefix
                        let prefix: String = prev_args_json
                            .chars()
                            .zip(cur_args_json.chars())
                            .take_while(|(c1, c2)| c1 == c2)
                            .map(|(c, _)| c)
                            .collect();
                        argument_diff = Some(prefix[sent..].to_string());
                    }
                }

                // Send the argument diff if there's something new
                if let Some(diff) = argument_diff {
                    if !diff.is_empty() {
                        if !is_complete && tool_id < self.streamed_args_for_tool.len() {
                            self.streamed_args_for_tool[tool_id].push_str(&diff);
                        }

                        result.calls.push(ToolCallItem {
                            tool_index: tool_id,
                            name: None,
                            parameters: diff,
                        });
                    }
                }
            }
        }

        // Update prev_tool_call_arr with current state
        if self.current_tool_id >= 0 {
            self.ensure_capacity();
            let tool_id = self.current_tool_id as usize;

            if tool_id < self.prev_tool_call_arr.len() {
                self.prev_tool_call_arr[tool_id] = current_tool_call;
            }
        }

        Ok(result)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.current_tool_name_sent = false;
        self.streamed_args_for_tool.clear();
    }

    fn detect_format(&self, text: &str) -> bool {
        let trimmed = text.trim();
        (trimmed.starts_with('[') || trimmed.starts_with('{')) && trimmed.contains(r#""name""#)
    }
}
