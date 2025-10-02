use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;

use crate::protocols::spec::Tool;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
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
/// - Buffering for partial end tokens
pub struct QwenParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,

    /// Regex for extracting tool calls in parse_complete
    extractor: Regex,

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

    /// Buffer for normal text that might precede partial end tokens
    normal_text_buffer: String,

    /// Token configuration
    bot_token: &'static str,
    eot_token: &'static str,
    tool_call_separator: &'static str,
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
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            normal_text_buffer: String::new(),
            bot_token: "<tool_call>\n",
            eot_token: "\n</tool_call>",
            tool_call_separator: "\n",
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

    /// Check if buffer ends with a partial eot_token
    fn ends_with_partial_eot(&self, buffer: &str) -> bool {
        if self.eot_token.is_empty() {
            return false;
        }

        for i in 1..=buffer.len().min(self.eot_token.len()) {
            if let Some(buffer_end) = buffer.get(buffer.len() - i..) {
                if self.eot_token.starts_with(buffer_end) {
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

    /// Check if text has tool call
    fn has_tool_call(&self, text: &str) -> bool {
        text.contains("<tool_call>")
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
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ToolParserResult<StreamingParseResult> {
        // Append new text to buffer
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if current_text has tool_call
        let has_tool_start = self.has_tool_call(current_text)
            || (self.current_tool_id >= 0
                && current_text.starts_with(self.tool_call_separator));

        if !has_tool_start {
            // Only clear buffer if we're sure no tool call is starting
            if !self.ends_with_partial_token(&self.buffer, self.bot_token) {
                let normal_text = self.buffer.clone();
                self.buffer.clear();

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
        } else if self.current_tool_id >= 0
            && current_text.starts_with(self.tool_call_separator)
        {
            self.tool_call_separator.len()
        } else {
            0
        };

        if start_idx >= current_text.len() {
            return Ok(StreamingParseResult::default());
        }

        // Handle partial end token buffering (Qwen-specific)
        // If we're in the middle of streaming and buffer ends with partial eot, hold it back
        let json_str = if self.current_tool_id >= 0 && self.ends_with_partial_eot(&self.buffer) {
            // Find how much of the eot_token is partial
            let mut partial_len = 0;
            for i in 1..=self.eot_token.len().min(self.buffer.len()) {
                if let Some(buffer_end) = self.buffer.get(self.buffer.len() - i..) {
                    if self.eot_token.starts_with(buffer_end) {
                        partial_len = i;
                    }
                }
            }

            // Extract JSON without the partial eot
            let safe_end = current_text.len() - partial_len;
            if start_idx >= safe_end {
                return Ok(StreamingParseResult::default());
            }
            &current_text[start_idx..safe_end]
        } else {
            &current_text[start_idx..]
        };

        // Parse partial JSON
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
        self.normal_text_buffer.clear();
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}
