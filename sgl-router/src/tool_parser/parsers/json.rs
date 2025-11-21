use async_trait::async_trait;
use serde_json::Value;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::{ParserError, ParserResult},
        parsers::helpers,
        partial_json::PartialJson,
        traits::ToolParser,
        types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
    },
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

    /// Separator between multiple tool calls
    tool_call_separator: &'static str,

    /// Track whether we're parsing array format `[...]` vs single object `{...}`
    is_array_format: bool,

    /// Track whether we've already stripped the closing ] bracket (for array format)
    array_closed: bool,
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
            tool_call_separator: ",",
            is_array_format: false,
            array_closed: false,
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
    fn parse_single_object(&self, obj: &Value) -> ParserResult<Option<ToolCall>> {
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

    /// Parse JSON value(s) into tool calls
    fn parse_json_value(&self, value: &Value) -> ParserResult<Vec<ToolCall>> {
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
}

impl Default for JsonParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for JsonParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Always use extract_json_from_text to handle both pure JSON and mixed content
        if let Some((extracted_json, normal_text)) = self.extract_json_from_text(text) {
            let parsed = serde_json::from_str::<Value>(&extracted_json)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))
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
    ) -> ParserResult<StreamingParseResult> {
        // Append new text to buffer
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Determine format on first parse (array vs single object)
        if self.current_tool_id == -1 && self.has_tool_markers(current_text) {
            self.is_array_format = current_text.trim().starts_with('[');
        }

        // Check if current_text has tool_call
        // Once array is closed, don't treat [ or { as tool markers
        let has_tool_start = (!self.array_closed && self.has_tool_markers(current_text))
            || (self.current_tool_id > 0 && current_text.starts_with(self.tool_call_separator));

        if !has_tool_start {
            let mut normal_text = self.buffer.clone();
            self.buffer.clear();

            // Strip ] only once (the closing bracket of JSON array format)
            // Only for array format and only if we haven't already closed it
            if self.is_array_format
                && !self.array_closed
                && self.current_tool_id > 0
                && normal_text.starts_with("]")
            {
                normal_text = normal_text.strip_prefix("]").unwrap().to_string();
                self.array_closed = true;
            }

            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // Build tool indices
        let tool_indices = helpers::get_tool_indices(tools);

        // Determine start index for JSON parsing
        // JSON can start with [ (array) or { (single object)
        let start_idx = if let Some(bracket_pos) = current_text.find('[') {
            let brace_pos = current_text.find('{');
            match brace_pos {
                Some(bp) => bp,
                _ => bracket_pos,
            }
        } else if let Some(brace_pos) = current_text.find('{') {
            brace_pos
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
        let trimmed = text.trim();
        trimmed.starts_with('[') || trimmed.starts_with('{')
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
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
        self.is_array_format = false;
        self.array_closed = false;
    }
}
