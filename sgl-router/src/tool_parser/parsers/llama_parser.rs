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
    fn parse_single_object(&self, obj: &Value) -> ParserResult<Option<ToolCall>> {
        // Llama format only: {"name": "function_name", "parameters": {...}}
        let name = obj.get("name").and_then(|v| v.as_str());

        if let Some(name) = name {
            // Llama uses "parameters" key
            let empty_obj = Value::Object(serde_json::Map::new());
            let parameters = obj.get("parameters").unwrap_or(&empty_obj);

            // Convert parameters to JSON string
            let arguments = serde_json::to_string(parameters)
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

    /// Parse semicolon-separated JSON objects
    fn parse_semicolon_separated(&self, content: &str) -> ParserResult<Vec<ToolCall>> {
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
}

impl Default for LlamaParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for LlamaParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
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
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))
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
        // Llama format if contains python_tag or starts with JSON object
        text.contains("<|python_tag|>") || text.trim_start().starts_with('{')
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
