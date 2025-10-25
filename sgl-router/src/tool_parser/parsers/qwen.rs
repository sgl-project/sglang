use async_trait::async_trait;
use regex::Regex;
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
    fn parse_single_object(&self, obj: &Value) -> ParserResult<Option<ToolCall>> {
        let name = obj.get("name").and_then(|v| v.as_str());

        if let Some(name) = name {
            // Get arguments - Qwen uses "arguments" key
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

impl Default for QwenParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for QwenParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Qwen format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where the first tool call begins
        let idx = text.find("<tool_call>").unwrap(); // Safe because has_tool_markers checked
        let normal_text = text[..idx].to_string();

        // Extract tool calls
        let mut tools = Vec::new();
        for captures in self.extractor.captures_iter(text) {
            if let Some(json_str) = captures.get(1) {
                let parsed = serde_json::from_str::<Value>(json_str.as_str().trim())
                    .map_err(|e| ParserError::ParsingFailed(e.to_string()))
                    .and_then(|v| self.parse_single_object(&v));

                match parsed {
                    Ok(Some(tool)) => tools.push(tool),
                    Ok(None) => continue,
                    Err(e) => {
                        tracing::warn!("Failed to parse tool call: {:?}", e);
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
    ) -> ParserResult<StreamingParseResult> {
        // Append new text to buffer
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if current_text has tool_call
        let has_tool_start = self.has_tool_markers(current_text)
            || (self.current_tool_id >= 0 && current_text.starts_with(self.tool_call_separator));

        if !has_tool_start {
            // Only clear buffer if we're sure no tool call is starting
            if helpers::ends_with_partial_token(&self.buffer, self.bot_token).is_none() {
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
        let tool_indices = helpers::get_tool_indices(tools);

        // Determine start index for JSON parsing
        let start_idx = if let Some(pos) = current_text.find(self.bot_token) {
            pos + self.bot_token.len()
        } else if self.current_tool_id >= 0 && current_text.starts_with(self.tool_call_separator) {
            self.tool_call_separator.len()
        } else {
            0
        };

        let mut result = helpers::handle_json_tool_streaming(
            current_text,
            start_idx,
            &mut self.partial_json,
            &tool_indices,
            &mut self.buffer,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
            &mut self.prev_tool_call_arr,
        )?;

        // Qwen-specific: Handle partial end tokens in normal text
        // After tool calls complete, normal text might contain partial "</tool_call>" tags
        if !result.normal_text.is_empty() {
            self.normal_text_buffer.push_str(&result.normal_text);

            // Check if buffer contains complete end token (without leading newline)
            let end_token_without_newline = &self.eot_token[1..]; // "</tool_call>"
            if self.normal_text_buffer.contains(end_token_without_newline) {
                // Complete end token found - clean it and return
                let cleaned_text = self
                    .normal_text_buffer
                    .replace(end_token_without_newline, "");
                self.normal_text_buffer.clear();
                result.normal_text = cleaned_text;
            } else {
                // Check if buffer might contain partial end token at the end
                if let Some(partial_match_len) = helpers::ends_with_partial_token(
                    &self.normal_text_buffer,
                    end_token_without_newline,
                ) {
                    // Keep potential partial match in buffer, return the rest
                    let split_point = self.normal_text_buffer.len() - partial_match_len;
                    result.normal_text = self.normal_text_buffer[..split_point].to_string();
                    self.normal_text_buffer = self.normal_text_buffer[split_point..].to_string();
                } else {
                    // No partial match, return all buffered text
                    result.normal_text = self.normal_text_buffer.clone();
                    self.normal_text_buffer.clear();
                }
            }
        }

        Ok(result)
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<tool_call>")
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
    }
}
