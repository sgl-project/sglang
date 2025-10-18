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

/// GLM-4 MoE format parser for tool calls
///
/// Handles the GLM-4 MoE specific format:
/// `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
///
/// Features:
/// - XML-style tags for tool calls
/// - Key-value pairs for arguments
/// - Support for multiple sequential tool calls
pub struct Glm4MoeParser {
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting function details
    func_detail_extractor: Regex,
    /// Regex for extracting argument key-value pairs
    arg_extractor: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Token configuration
    bot_token: &'static str,
    eot_token: &'static str,
}

impl Glm4MoeParser {
    /// Create a new GLM-4 MoE parser
    pub fn new() -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<tool_call>.*?</tool_call>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let func_detail_pattern = r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>";
        let func_detail_extractor = Regex::new(func_detail_pattern).expect("Valid regex pattern");

        let arg_pattern = r"(?s)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>";
        let arg_extractor = Regex::new(arg_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            func_detail_extractor,
            arg_extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            bot_token: "<tool_call>",
            eot_token: "</tool_call>",
        }
    }

    /// Parse arguments from key-value pairs
    fn parse_arguments(&self, args_text: &str) -> ParserResult<serde_json::Map<String, Value>> {
        let mut arguments = serde_json::Map::new();

        for capture in self.arg_extractor.captures_iter(args_text) {
            let key = capture.get(1).map_or("", |m| m.as_str()).trim();
            let value_str = capture.get(2).map_or("", |m| m.as_str()).trim();

            // Try to parse the value as JSON first, fallback to string
            let value = if let Ok(json_val) = serde_json::from_str::<Value>(value_str) {
                json_val
            } else {
                // Try parsing as Python literal (similar to Python's ast.literal_eval)
                if value_str == "true" || value_str == "True" {
                    Value::Bool(true)
                } else if value_str == "false" || value_str == "False" {
                    Value::Bool(false)
                } else if value_str == "null" || value_str == "None" {
                    Value::Null
                } else if let Ok(num) = value_str.parse::<i64>() {
                    Value::Number(num.into())
                } else if let Ok(num) = value_str.parse::<f64>() {
                    if let Some(n) = serde_json::Number::from_f64(num) {
                        Value::Number(n)
                    } else {
                        Value::String(value_str.to_string())
                    }
                } else {
                    Value::String(value_str.to_string())
                }
            };

            arguments.insert(key.to_string(), value);
        }

        Ok(arguments)
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str) -> ParserResult<Option<ToolCall>> {
        if let Some(captures) = self.func_detail_extractor.captures(block) {
            // Get function name
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

            // Get arguments text
            let args_text = captures.get(2).map_or("", |m| m.as_str());

            // Parse arguments
            let arguments = self.parse_arguments(args_text)?;

            let arguments_str = serde_json::to_string(&arguments)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

            Ok(Some(ToolCall {
                function: FunctionCall {
                    name: func_name.to_string(),
                    arguments: arguments_str,
                },
            }))
        } else {
            Ok(None)
        }
    }

    /// Parse and return StreamingParseResult (mirrors Python's detect_and_parse)
    /// Parse all tool calls from text (shared logic for complete and incremental parsing)
    fn parse_tool_calls_from_text(&self, text: &str) -> ParserResult<Vec<ToolCall>> {
        let mut tools = Vec::new();

        for mat in self.tool_call_extractor.find_iter(text) {
            match self.parse_tool_call(mat.as_str()) {
                Ok(Some(tool)) => tools.push(tool),
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!("Failed to parse tool call: {}", e);
                    continue;
                }
            }
        }

        Ok(tools)
    }
}

impl Default for Glm4MoeParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for Glm4MoeParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains GLM-4 MoE format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        let idx = text.find("<tool_call>").unwrap();
        let normal_text = text[..idx].to_string();

        // Parse all tool calls using shared helper
        let tools = self.parse_tool_calls_from_text(text)?;

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
        // Python logic: Wait for complete tool call, then parse it all at once
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if we have bot_token
        let start = current_text.find(self.bot_token);
        if start.is_none() {
            self.buffer.clear();
            // If we're in the middle of streaming (current_tool_id > 0), don't return text
            let normal_text = if self.current_tool_id > 0 {
                String::new()
            } else {
                current_text.clone()
            };
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // Check if we have eot_token (end of tool call)
        let end = current_text.find(self.eot_token);
        if let Some(end_pos) = end {
            // We have a complete tool call!

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

            // Parse the complete block using shared helper
            let block_end = end_pos + self.eot_token.len();
            let parsed_tools = self.parse_tool_calls_from_text(&current_text[..block_end])?;

            // Extract normal text before tool calls
            let idx = current_text.find(self.bot_token);
            let normal_text = if let Some(pos) = idx {
                current_text[..pos].trim().to_string()
            } else {
                String::new()
            };

            // Build tool indices for validation
            let tool_indices = helpers::get_tool_indices(tools);

            let mut calls = Vec::new();

            if !parsed_tools.is_empty() {
                // Take the first tool and convert to ToolCallItem
                let tool_call = &parsed_tools[0];
                let tool_id = self.current_tool_id as usize;

                // Validate tool name
                if !tool_indices.contains_key(&tool_call.function.name) {
                    // Invalid tool name - skip this tool, preserve indexing for next tool
                    tracing::warn!("Invalid tool name '{}' - skipping", tool_call.function.name);
                    helpers::reset_current_tool_state(
                        &mut self.buffer,
                        &mut false, // glm4_moe doesn't track name_sent per tool
                        &mut self.streamed_args_for_tool,
                        &self.prev_tool_call_arr,
                    );
                    return Ok(StreamingParseResult::default());
                }

                calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: Some(tool_call.function.name.clone()),
                    parameters: tool_call.function.arguments.clone(),
                });

                // Store in tracking arrays
                if self.prev_tool_call_arr.len() <= tool_id {
                    self.prev_tool_call_arr
                        .resize_with(tool_id + 1, || Value::Null);
                }

                // Parse parameters as JSON and store
                if let Ok(args) = serde_json::from_str::<Value>(&tool_call.function.arguments) {
                    self.prev_tool_call_arr[tool_id] = serde_json::json!({
                        "name": tool_call.function.name,
                        "arguments": args,
                    });
                }

                if self.streamed_args_for_tool.len() <= tool_id {
                    self.streamed_args_for_tool
                        .resize_with(tool_id + 1, String::new);
                }
                self.streamed_args_for_tool[tool_id] = tool_call.function.arguments.clone();

                self.current_tool_id += 1;
            }

            // Remove processed portion from buffer
            self.buffer = current_text[block_end..].to_string();
            return Ok(StreamingParseResult { normal_text, calls });
        }

        // No complete tool call yet - return normal text before start token
        let start_pos = start.unwrap();
        let normal_text = current_text[..start_pos].to_string();
        self.buffer = current_text[start_pos..].to_string();

        Ok(StreamingParseResult {
            normal_text,
            calls: vec![],
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.bot_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
    }
}
