use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::{ParserError, ParserResult},
        parsers::helpers,
        traits::ToolParser,
        types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
    },
};

/// DeepSeek V3 format parser for tool calls
///
/// Handles the DeepSeek V3 specific format that uses Unicode tokens:
/// `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{args}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
///
/// Features:
/// - Unicode token delimiters
/// - JSON arguments in code blocks
/// - Support for multiple sequential tool calls
pub struct DeepSeekParser {
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting function details
    func_detail_extractor: Regex,
    /// Regex for matching partial tool calls during streaming
    partial_tool_call_regex: Regex,
    /// Regex pattern for removing completed tool calls from buffer
    tool_call_end_pattern: Regex,

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
}

impl DeepSeekParser {
    /// Create a new DeepSeek parser
    pub fn new() -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let func_detail_pattern = r"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)\n```json\n(.*?)\n```<｜tool▁call▁end｜>";
        let func_detail_extractor = Regex::new(func_detail_pattern).expect("Valid regex pattern");

        // Partial pattern for streaming - uses .* (greedy) not .*? to match all partial content
        let partial_pattern = r"(?s)<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)";
        let partial_tool_call_regex = Regex::new(partial_pattern).expect("Valid regex pattern");

        // Pattern for removing completed tool calls
        let end_pattern = r"(?s)<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>";
        let tool_call_end_pattern = Regex::new(end_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            func_detail_extractor,
            partial_tool_call_regex,
            tool_call_end_pattern,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
        }
    }

    /// Parse a single tool call block - throws error if parsing fails
    fn parse_tool_call(&self, block: &str) -> ParserResult<ToolCall> {
        let captures = self.func_detail_extractor.captures(block).ok_or_else(|| {
            ParserError::ParsingFailed("Failed to match tool call pattern".to_string())
        })?;

        // Get function type (should be "function")
        let func_type = captures.get(1).map_or("", |m| m.as_str());
        if func_type != "function" {
            return Err(ParserError::ParsingFailed(format!(
                "Invalid function type: {}",
                func_type
            )));
        }

        // Get function name
        let func_name = captures.get(2).map_or("", |m| m.as_str()).trim();
        if func_name.is_empty() {
            return Err(ParserError::ParsingFailed(
                "Empty function name".to_string(),
            ));
        }

        // Get JSON arguments
        let json_args = captures.get(3).map_or("{}", |m| m.as_str()).trim();

        // Parse JSON arguments
        let value = serde_json::from_str::<Value>(json_args)
            .map_err(|e| ParserError::ParsingFailed(format!("Invalid JSON: {}", e)))?;

        // Create arguments object
        let args = if value.is_object() {
            value
        } else {
            // If not an object, wrap it
            serde_json::json!({ "value": value })
        };

        let arguments =
            serde_json::to_string(&args).map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(ToolCall {
            function: FunctionCall {
                name: func_name.to_string(),
                arguments,
            },
        })
    }
}

impl Default for DeepSeekParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for DeepSeekParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        let idx = text.find("<｜tool▁calls▁begin｜>").unwrap();
        let normal_text = text[..idx].to_string();

        // Try to extract tool calls, log warnings for failures
        let mut tools = Vec::new();
        for mat in self.tool_call_extractor.find_iter(text) {
            match self.parse_tool_call(mat.as_str()) {
                Ok(tool) => tools.push(tool),
                Err(e) => {
                    tracing::warn!("Failed to parse tool call: {}", e);
                    continue;
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
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if we have a tool call (either the start token or individual tool call)
        let has_tool_call =
            self.has_tool_markers(current_text) || current_text.contains("<｜tool▁call▁begin｜>");

        if !has_tool_call {
            // No tool markers detected - return all buffered content as normal text
            // Strip out end tokens if present
            let mut normal_text = std::mem::take(&mut self.buffer);
            for e_token in ["<｜tool▁calls▁end｜>", "```", "<｜tool▁call▁end｜>"] {
                normal_text = normal_text.replace(e_token, "");
            }
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // Build tool indices for validation
        let tool_indices = helpers::get_tool_indices(tools);

        let mut calls: Vec<ToolCallItem> = Vec::new();

        // Try to match the partial tool call pattern
        if let Some(captures) = self.partial_tool_call_regex.captures(current_text) {
            let func_name = captures.get(2).map_or("", |m| m.as_str()).trim();
            let func_args_raw = captures.get(3).map_or("", |m| m.as_str()).trim();

            // Validate tool name
            if !tool_indices.contains_key(func_name) {
                // Invalid tool name - skip this tool, preserve indexing for next tool
                tracing::warn!("Invalid tool name '{}' - skipping", func_name);
                helpers::reset_current_tool_state(
                    &mut self.buffer,
                    &mut self.current_tool_name_sent,
                    &mut self.streamed_args_for_tool,
                    &self.prev_tool_call_arr,
                );
                return Ok(StreamingParseResult::default());
            }

            // Initialize state if this is the first tool call
            if self.current_tool_id == -1 {
                self.current_tool_id = 0;
                self.prev_tool_call_arr = Vec::new();
                self.streamed_args_for_tool = vec![String::new()];
            }

            // Ensure we have enough entries in our tracking arrays
            helpers::ensure_capacity(
                self.current_tool_id,
                &mut self.prev_tool_call_arr,
                &mut self.streamed_args_for_tool,
            );

            // Send tool name if not sent yet
            if !self.current_tool_name_sent {
                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: Some(func_name.to_string()),
                    parameters: String::new(),
                });
                self.current_tool_name_sent = true;

                // Store the tool call info for serving layer completions endpoint
                let tool_id = self.current_tool_id as usize;
                if self.prev_tool_call_arr.len() <= tool_id {
                    self.prev_tool_call_arr
                        .resize_with(tool_id + 1, || Value::Null);
                }
                self.prev_tool_call_arr[tool_id] = serde_json::json!({
                    "name": func_name,
                    "arguments": {},
                });
            } else {
                // Compute incremental diff
                let tool_id = self.current_tool_id as usize;
                let last_sent = self
                    .streamed_args_for_tool
                    .get(tool_id)
                    .map(|s| s.as_str())
                    .unwrap_or("");

                let argument_diff = func_args_raw
                    .strip_prefix(last_sent)
                    .unwrap_or(func_args_raw);

                if !argument_diff.is_empty() {
                    calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: argument_diff.to_string(),
                    });
                    if tool_id < self.streamed_args_for_tool.len() {
                        self.streamed_args_for_tool[tool_id].push_str(argument_diff);
                    }
                }

                // Check if JSON is complete
                if helpers::is_complete_json(func_args_raw) {
                    // Update the stored arguments
                    if let Ok(parsed_args) = serde_json::from_str::<Value>(func_args_raw) {
                        let tool_id = self.current_tool_id as usize;
                        if tool_id < self.prev_tool_call_arr.len() {
                            if let Some(obj) = self.prev_tool_call_arr[tool_id].as_object_mut() {
                                obj.insert("arguments".to_string(), parsed_args);
                            }
                        }
                    }

                    // Find the end of the current tool call and remove only that part from buffer
                    if let Some(mat) = self.tool_call_end_pattern.find(current_text) {
                        // Remove the completed tool call from buffer, keep any remaining content
                        self.buffer = current_text[mat.end()..].to_string();
                    } else {
                        self.buffer.clear();
                    }

                    let result = StreamingParseResult {
                        normal_text: String::new(),
                        calls,
                    };

                    self.current_tool_id += 1;
                    self.current_tool_name_sent = false;
                    return Ok(result);
                }
            }
        }

        Ok(StreamingParseResult {
            normal_text: String::new(),
            calls,
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<｜tool▁calls▁begin｜>")
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.current_tool_name_sent = false;
        self.streamed_args_for_tool.clear();
    }
}
