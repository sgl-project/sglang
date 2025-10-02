use async_trait::async_trait;
use serde_json::Value;

use crate::protocols::spec::Tool;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    parsers::helpers,
    partial_json::PartialJson,
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

impl MistralParser {
    /// Create a new Mistral parser
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            bot_token: "[TOOL_CALLS] [",
            eot_token: "]",
            tool_call_separator: ", ",
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

            // Generate unique ID
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("mistral_call_{}", uuid::Uuid::new_v4()));

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
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ToolParserResult<StreamingParseResult> {
        // Append new text to buffer
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if current_text has tool_call
        let has_tool_start = self.has_tool_markers(current_text)
            || (self.current_tool_id >= 0 && current_text.starts_with(self.tool_call_separator));

        if !has_tool_start {
            // Only clear buffer if we're sure no tool call is starting
            if !helpers::ends_with_partial_token(&self.buffer, self.bot_token) {
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
        let tool_indices = helpers::get_tool_indices(tools);

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
                // Invalid tool name - skip this tool, preserve indexing for next tool
                tracing::warn!("Invalid tool name '{}' - skipping", name);
                helpers::reset_current_tool_state(
                    &mut self.buffer,
                    &mut self.current_tool_name_sent,
                    &mut self.streamed_args_for_tool,
                    &self.prev_tool_call_arr,
                );
                return Ok(StreamingParseResult::default());
            }
        }

        // Handle parameters/arguments aliasing
        let current_tool_call = helpers::normalize_arguments_field(obj.clone());

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
                        helpers::ensure_capacity(
                            self.current_tool_id,
                            &mut self.prev_tool_call_arr,
                            &mut self.streamed_args_for_tool,
                        );
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

                // Compute diff: everything after what we've already sent
                let diff = cur_args_json[sent..].to_string();

                // Send diff if there's new content
                if !diff.is_empty() {
                    // Only accumulate if not complete
                    if !is_complete && tool_id < self.streamed_args_for_tool.len() {
                        self.streamed_args_for_tool[tool_id].push_str(&diff);
                    }

                    result.calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: diff,
                    });
                }

                // If JSON is complete, advance to next tool
                if is_complete {
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
            }
        }

        // Update prev_tool_call_arr with current state
        if self.current_tool_id >= 0 {
            helpers::ensure_capacity(
                self.current_tool_id,
                &mut self.prev_tool_call_arr,
                &mut self.streamed_args_for_tool,
            );
            let tool_id = self.current_tool_id as usize;

            if tool_id < self.prev_tool_call_arr.len() {
                self.prev_tool_call_arr[tool_id] = current_tool_call;
            }
        }

        Ok(result)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}
