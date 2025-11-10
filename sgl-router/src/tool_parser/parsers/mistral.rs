use async_trait::async_trait;
use serde_json::Value;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::{ParserError, ParserResult},
        parsers::helpers,
        partial_json::PartialJson,
        traits::ToolParser,
        types::{FunctionCall, StreamingParseResult, ToolCall},
    },
};

/// Mistral format parser for tool calls
///
/// Handles the Mistral-specific format:
/// `[TOOL_CALLS] [{"name": "func", "arguments": {...}}, ...]`
///
/// Reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3?chat_template=default
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

    /// Track whether we've already stripped the closing ] bracket
    array_closed: bool,
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
            array_closed: false,
        }
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
    fn parse_json_array(&self, json_str: &str) -> ParserResult<Vec<ToolCall>> {
        let value: Value = serde_json::from_str(json_str)
            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        let mut tools = Vec::new();

        if let Value::Array(arr) = value {
            for item in arr.iter() {
                if let Some(tool) = self.parse_single_object(item)? {
                    tools.push(tool);
                }
            }
        } else {
            // Single object case (shouldn't happen with Mistral format, but handle it)
            if let Some(tool) = self.parse_single_object(&value)? {
                tools.push(tool);
            }
        }

        Ok(tools)
    }

    /// Parse a single JSON object into a ToolCall
    fn parse_single_object(&self, obj: &Value) -> ParserResult<Option<ToolCall>> {
        let name = obj.get("name").and_then(|v| v.as_str());

        if let Some(name) = name {
            // Get arguments - Mistral uses "arguments" key
            let empty_obj = Value::Object(serde_json::Map::new());
            let args = obj.get("arguments").unwrap_or(&empty_obj);

            // Convert arguments to JSON string
            let arguments = serde_json::to_string(args)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

            Ok(Some(ToolCall {
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            }))
        } else {
            Ok(None)
        }
    }
}

impl Default for MistralParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for MistralParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
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
    ) -> ParserResult<StreamingParseResult> {
        // Append new text to buffer
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if current_text has tool_call
        let has_tool_start = self.has_tool_markers(current_text)
            || (self.current_tool_id > 0 && current_text.starts_with(self.tool_call_separator));

        if !has_tool_start {
            // Only clear buffer if we're sure no tool call is starting
            if helpers::ends_with_partial_token(&self.buffer, self.bot_token).is_none() {
                let mut normal_text = self.buffer.clone();
                self.buffer.clear();

                // Strip ] only once (the closing bracket of [TOOL_CALLS] array)
                // current_tool_id > 0 means we've parsed at least one tool
                if !self.array_closed
                    && self.current_tool_id > 0
                    && normal_text.starts_with(self.eot_token)
                {
                    normal_text = normal_text
                        .strip_prefix(self.eot_token)
                        .unwrap()
                        .to_string();
                    self.array_closed = true;
                }

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
        } else if self.current_tool_id > 0 && current_text.starts_with(self.tool_call_separator) {
            self.tool_call_separator.len()
        } else {
            0
        };

        helpers::handle_json_tool_streaming(
            current_text,
            start_idx,
            &mut self.partial_json,
            &tool_indices,
            &mut self.buffer,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
            &mut self.prev_tool_call_arr,
        )
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("[TOOL_CALLS]")
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<crate::tool_parser::types::ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        helpers::reset_parser_state(
            &mut self.buffer,
            &mut self.prev_tool_call_arr,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
        );
        self.array_closed = false;
    }
}
