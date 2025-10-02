use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use uuid;

use crate::protocols::spec::Tool;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Llama 3.2 format parser for tool calls
///
/// Handles the Llama 3.2 specific format:
/// `<|python_tag|>{"name": "func", "parameters": {...}}`
///
/// Also supports plain JSON without the python_tag prefix
pub struct LlamaParser {
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

impl LlamaParser {
    /// Create a new Llama parser
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            bot_token: "<|python_tag|>",
            eot_token: "",
            tool_call_separator: ";",
        }
    }

    /// Extract content after python_tag token
    fn extract_content_after_python_tag(&self, text: &str) -> Option<(String, String)> {
        const PYTHON_TAG: &str = "<|python_tag|>";

        if let Some(tag_pos) = text.find(PYTHON_TAG) {
            let normal_text = text[..tag_pos].to_string();
            let json_content = text[tag_pos + PYTHON_TAG.len()..].to_string();
            Some((normal_text, json_content))
        } else {
            None
        }
    }

    /// Parse a single JSON object into a ToolCall (Llama format: name + parameters)
    fn parse_single_object(&self, obj: &Value) -> ToolParserResult<Option<ToolCall>> {
        // Llama format only: {"name": "function_name", "parameters": {...}}
        let name = obj.get("name").and_then(|v| v.as_str());

        if let Some(name) = name {
            // Llama uses "parameters" key
            let empty_obj = Value::Object(serde_json::Map::new());
            let parameters = obj.get("parameters").unwrap_or(&empty_obj);

            // Convert parameters to JSON string
            let arguments = serde_json::to_string(parameters)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate a unique ID for Llama calls
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("llama_call_{}", uuid::Uuid::new_v4()));

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
    fn has_python_tag(&self, text: &str) -> bool {
        text.contains("<|python_tag|>")
    }

    /// Parse semicolon-separated JSON objects
    fn parse_semicolon_separated(&self, content: &str) -> ToolParserResult<Vec<ToolCall>> {
        let mut all_tools = Vec::new();

        // Split by semicolon and parse each JSON object
        for part in content.split(';') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Try to parse this part as a single JSON object
            match serde_json::from_str::<Value>(trimmed) {
                Ok(value) => {
                    if let Some(tool) = self.parse_single_object(&value)? {
                        all_tools.push(tool);
                    }
                }
                Err(e) => {
                    // Skip invalid JSON parts in semicolon-separated list
                    tracing::warn!("Failed to parse tool call: {}", e);
                }
            }
        }

        Ok(all_tools)
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

    /// Check if text has tool call
    fn has_tool_call(&self, text: &str) -> bool {
        text.contains("<|python_tag|>") || text.contains('{')
    }
}

impl Default for LlamaParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for LlamaParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Extract normal text and JSON content
        let (normal_text, json_content) =
            if let Some((normal, json)) = self.extract_content_after_python_tag(text) {
                (normal, json)
            } else if text.trim_start().starts_with('{') {
                (String::new(), text.to_string())
            } else {
                // No JSON structure found
                return Ok((text.to_string(), vec![]));
            };

        // Parse the JSON content (may contain semicolon-separated objects)
        let tools = if json_content.contains(';') {
            self.parse_semicolon_separated(&json_content)?
        } else {
            // Try single JSON object
            let parsed = serde_json::from_str::<Value>(json_content.trim())
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))
                .and_then(|v| {
                    self.parse_single_object(&v)
                        .map(|opt| opt.map_or_else(Vec::new, |tool| vec![tool]))
                });

            parsed.unwrap_or_else(|e| {
                tracing::warn!("Failed to parse tool call: {:?}", e);
                vec![]
            })
        };

        // If we couldn't parse any tools, return the original text
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
        // Llama format if contains python_tag or starts with JSON object
        text.contains("<|python_tag|>")
            || (text.trim_start().starts_with('{') && text.contains(r#""name""#))
    }
}
