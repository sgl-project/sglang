use std::collections::HashMap;

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

/// Step3 format parser for tool calls
///
/// Handles the Step3 specific format with steptml XML:
/// `<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="{name}"><steptml:parameter name="{k}">{v}</steptml:parameter></steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>`
///
/// Features:
/// - Unicode token delimiters
/// - StepTML XML format for invocations
/// - Support for multiple sequential tool calls
pub struct Step3Parser {
    /// Regex for extracting tool call blocks
    tool_call_extractor: Regex,
    /// Regex for extracting steptml invocations
    invoke_extractor: Regex,
    /// Regex for extracting parameters
    param_extractor: Regex,

    /// Buffer for accumulating chunks
    buffer: String,

    /// Token configuration
    bot_token: &'static str,
    eot_token: &'static str,
    tool_call_begin: &'static str,
    tool_call_end: &'static str,
    tool_sep: &'static str,

    /// Streaming state variables (mirrors Python's Step3Detector)
    in_tool_block: bool,
    tool_block_finished: bool,
    current_function_name: String,
    current_parameters: serde_json::Map<String, Value>,
    in_tool_call: bool,
    function_name_sent: bool,

    /// Standard state machine fields
    prev_tool_call_arr: Vec<Value>,
    current_tool_id: i32,
    streamed_args_for_tool: Vec<String>,
}

impl Step3Parser {
    /// Create a new Step3 parser
    pub fn new() -> Self {
        // Pattern for individual tool calls
        let tool_call_pattern = r"(?s)<｜tool_call_begin｜>.*?<｜tool_call_end｜>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        // Pattern for steptml invocations
        let invoke_pattern = r#"(?s)<steptml:invoke name="([^"]+)">(.+?)</steptml:invoke>"#;
        let invoke_extractor = Regex::new(invoke_pattern).expect("Valid regex pattern");

        // Pattern for steptml parameters - using non-greedy match for values to handle < characters
        let param_pattern = r#"(?s)<steptml:parameter name="([^"]+)">(.+?)</steptml:parameter>"#;
        let param_extractor = Regex::new(param_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            invoke_extractor,
            param_extractor,

            buffer: String::new(),

            bot_token: "<｜tool_calls_begin｜>",
            eot_token: "<｜tool_calls_end｜>",
            tool_call_begin: "<｜tool_call_begin｜>",
            tool_call_end: "<｜tool_call_end｜>",
            tool_sep: "<｜tool_sep｜>",

            // Streaming state variables
            in_tool_block: false,
            tool_block_finished: false,
            current_function_name: String::new(),
            current_parameters: serde_json::Map::new(),
            in_tool_call: false,
            function_name_sent: false,

            // Standard state machine fields
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
        }
    }

    /// Reset streaming state for the next tool call
    fn reset_streaming_state(&mut self) {
        self.in_tool_call = false;
        self.function_name_sent = false;
        self.current_function_name.clear();
        self.current_parameters.clear();
    }

    /// Parse partial tool call for streaming scenarios (mirrors Python's _parse_partial_tool_call)
    fn parse_partial_tool_call(
        &mut self,
        tool_indices: &HashMap<String, usize>,
    ) -> ParserResult<StreamingParseResult> {
        let mut calls = Vec::new();

        // Check if we have tool_sep (means we're past the type declaration)
        if !self.buffer.contains(self.tool_sep) {
            return Ok(StreamingParseResult {
                normal_text: String::new(),
                calls,
            });
        }

        // Clone the buffer to avoid borrow conflicts
        let buffer_clone = self.buffer.clone();
        let parts: Vec<&str> = buffer_clone.splitn(2, self.tool_sep).collect();
        if parts.len() != 2 {
            return Ok(StreamingParseResult {
                normal_text: String::new(),
                calls,
            });
        }

        let type_part = parts[0].trim();
        let invoke_part = parts[1];

        // Check if it's a function type
        if type_part != "function" {
            // Invalid tool type, skip this tool call
            self.reset_streaming_state();
            return Ok(StreamingParseResult {
                normal_text: String::new(),
                calls,
            });
        }

        // Try to extract function name if not sent yet
        if !self.function_name_sent {
            if let Some(captures) = self.invoke_extractor.captures(invoke_part) {
                let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

                // Validate function name
                if tool_indices.contains_key(func_name) {
                    self.current_function_name = func_name.to_string();
                    self.function_name_sent = true;

                    // Initialize tool tracking
                    if self.current_tool_id == -1 {
                        self.current_tool_id = 0;
                    }

                    // Ensure tracking arrays are large enough
                    helpers::ensure_capacity(
                        self.current_tool_id,
                        &mut self.prev_tool_call_arr,
                        &mut self.streamed_args_for_tool,
                    );

                    // Store tool call info
                    let tool_id = self.current_tool_id as usize;
                    self.prev_tool_call_arr[tool_id] = serde_json::json!({
                        "name": func_name,
                        "arguments": {},
                    });

                    // Send tool name with empty parameters
                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: Some(func_name.to_string()),
                        parameters: String::new(),
                    });
                } else {
                    // Invalid function name
                    tracing::warn!("Invalid function name: {}", func_name);
                    self.reset_streaming_state();
                    return Ok(StreamingParseResult {
                        normal_text: String::new(),
                        calls,
                    });
                }
            } else {
                // Function name not complete yet
                return Ok(StreamingParseResult {
                    normal_text: String::new(),
                    calls,
                });
            }
        }

        // Parse parameters incrementally
        if self.function_name_sent {
            // Extract all complete parameters
            let mut new_params = serde_json::Map::new();
            for capture in self.param_extractor.captures_iter(invoke_part) {
                let param_name = capture.get(1).map_or("", |m| m.as_str()).trim();
                let param_value_str = capture.get(2).map_or("", |m| m.as_str()).trim();

                // Try to parse the value as JSON first, fallback to string
                let param_value =
                    if let Ok(json_val) = serde_json::from_str::<Value>(param_value_str) {
                        json_val
                    } else {
                        // Try parsing as Python literal
                        if param_value_str == "true" || param_value_str == "True" {
                            Value::Bool(true)
                        } else if param_value_str == "false" || param_value_str == "False" {
                            Value::Bool(false)
                        } else if param_value_str == "null" || param_value_str == "None" {
                            Value::Null
                        } else if let Ok(num) = param_value_str.parse::<i64>() {
                            Value::Number(num.into())
                        } else if let Ok(num) = param_value_str.parse::<f64>() {
                            if let Some(n) = serde_json::Number::from_f64(num) {
                                Value::Number(n)
                            } else {
                                Value::String(param_value_str.to_string())
                            }
                        } else {
                            Value::String(param_value_str.to_string())
                        }
                    };

                new_params.insert(param_name.to_string(), param_value);
            }

            // Check if we have new parameters to stream
            if new_params != self.current_parameters {
                // Build the JSON content without the closing brace for streaming
                let diff = if self.current_parameters.is_empty() {
                    // First parameters - send opening brace and content
                    let params_content =
                        serde_json::to_string(&new_params).unwrap_or_else(|_| "{}".to_string());
                    if params_content.len() > 2 {
                        // Send everything except the closing brace
                        params_content[..params_content.len() - 1].to_string()
                    } else {
                        "{".to_string()
                    }
                } else {
                    // Subsequent parameters - calculate the incremental diff
                    let old_json = serde_json::to_string(&self.current_parameters)
                        .unwrap_or_else(|_| "{}".to_string());
                    let new_json =
                        serde_json::to_string(&new_params).unwrap_or_else(|_| "{}".to_string());

                    // Remove closing braces for comparison
                    let old_without_brace = &old_json[..old_json.len() - 1];
                    let new_without_brace = &new_json[..new_json.len() - 1];

                    // The new content should extend the old content
                    new_without_brace
                        .strip_prefix(old_without_brace)
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                };

                if !diff.is_empty() {
                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: diff.clone(),
                    });
                    let tool_id = self.current_tool_id as usize;
                    if tool_id < self.streamed_args_for_tool.len() {
                        self.streamed_args_for_tool[tool_id].push_str(&diff);
                    }
                }

                // Update current state
                self.current_parameters = new_params.clone();
                let tool_id = self.current_tool_id as usize;
                if tool_id < self.prev_tool_call_arr.len() {
                    if let Some(obj) = self.prev_tool_call_arr[tool_id].as_object_mut() {
                        obj.insert("arguments".to_string(), Value::Object(new_params));
                    }
                }
            }

            // Check if tool call is complete
            if self.buffer.contains(self.tool_call_end) {
                // Send closing brace if we've sent any parameters
                let tool_id = self.current_tool_id as usize;
                if tool_id < self.streamed_args_for_tool.len()
                    && !self.streamed_args_for_tool[tool_id].is_empty()
                {
                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: "}".to_string(),
                    });
                    self.streamed_args_for_tool[tool_id].push('}');
                }

                // Find the end position
                if let Some(end_idx) = self.buffer.find(self.tool_call_end) {
                    // Remove the processed tool call from buffer
                    self.buffer = self.buffer[end_idx + self.tool_call_end.len()..].to_string();
                }

                // Reset state for next tool call
                self.reset_streaming_state();
                self.current_tool_id += 1;
            }
        }

        Ok(StreamingParseResult {
            normal_text: String::new(),
            calls,
        })
    }

    /// Parse parameters from steptml format
    fn parse_steptml_parameters(
        &self,
        params_text: &str,
    ) -> ParserResult<serde_json::Map<String, Value>> {
        let mut parameters = serde_json::Map::new();

        for capture in self.param_extractor.captures_iter(params_text) {
            let param_name = capture.get(1).map_or("", |m| m.as_str()).trim();
            let param_value_str = capture.get(2).map_or("", |m| m.as_str()).trim();

            // Try to parse the value as JSON first, fallback to string
            let param_value = if let Ok(json_val) = serde_json::from_str::<Value>(param_value_str) {
                json_val
            } else {
                // Try parsing as Python literal
                if param_value_str == "true" || param_value_str == "True" {
                    Value::Bool(true)
                } else if param_value_str == "false" || param_value_str == "False" {
                    Value::Bool(false)
                } else if param_value_str == "null" || param_value_str == "None" {
                    Value::Null
                } else if let Ok(num) = param_value_str.parse::<i64>() {
                    Value::Number(num.into())
                } else if let Ok(num) = param_value_str.parse::<f64>() {
                    if let Some(n) = serde_json::Number::from_f64(num) {
                        Value::Number(n)
                    } else {
                        Value::String(param_value_str.to_string())
                    }
                } else {
                    Value::String(param_value_str.to_string())
                }
            };

            parameters.insert(param_name.to_string(), param_value);
        }

        Ok(parameters)
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str) -> ParserResult<Option<ToolCall>> {
        // Check if it contains function marker and tool separator
        if !block.contains("function") || !block.contains("<｜tool_sep｜>") {
            return Ok(None);
        }

        // Split by tool separator
        let parts: Vec<&str> = block.split("<｜tool_sep｜>").collect();
        if parts.len() != 2 {
            return Ok(None);
        }

        // Check if it's a function type
        if !parts[0].contains("function") {
            return Ok(None);
        }

        let invoke_part = parts[1];

        // Extract steptml invoke
        if let Some(captures) = self.invoke_extractor.captures(invoke_part) {
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

            // Validate function name is not empty
            if func_name.is_empty() {
                return Ok(None);
            }

            let params_text = captures.get(2).map_or("", |m| m.as_str());

            // Parse parameters
            let parameters = self.parse_steptml_parameters(params_text)?;

            let arguments_str = serde_json::to_string(&parameters)
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
}

impl Default for Step3Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for Step3Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        let idx = text.find("<｜tool_calls_begin｜>").unwrap();
        let normal_text = text[..idx].to_string();

        // Extract tool calls
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

        // Build tool indices for validation
        let tool_indices = helpers::get_tool_indices(tools);

        // Stage 1: If we've finished the tool block, everything is normal text
        if self.tool_block_finished {
            let normal_text = std::mem::take(&mut self.buffer);
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // Stage 2: Check if tool block hasn't started yet
        if !self.in_tool_block {
            if self.buffer.contains(self.bot_token) {
                let idx = self.buffer.find(self.bot_token).unwrap();
                let normal_text = self.buffer[..idx].to_string();
                self.buffer = self.buffer[idx + self.bot_token.len()..].to_string();
                self.in_tool_block = true;
                return Ok(StreamingParseResult {
                    normal_text,
                    calls: vec![],
                });
            } else {
                // Check if we might have a partial bot_token
                if helpers::ends_with_partial_token(&self.buffer, self.bot_token).is_some() {
                    return Ok(StreamingParseResult::default()); // Wait for more text
                } else {
                    let normal_text = std::mem::take(&mut self.buffer);
                    return Ok(StreamingParseResult {
                        normal_text,
                        calls: vec![],
                    });
                }
            }
        }

        // We're inside the tool block
        let mut calls = Vec::new();

        // Stage 3: Check if tool block is ending
        if self.buffer.contains(self.eot_token) {
            let idx = self.buffer.find(self.eot_token).unwrap();

            // If we're in the middle of a tool call, we need to handle it
            if self.in_tool_call {
                // The buffer before eot_token might contain the end of the current tool call
                let before_eot = &self.buffer[..idx];
                if before_eot.contains(self.tool_call_end) {
                    // Parse this final tool call
                    let result = self.parse_partial_tool_call(&tool_indices)?;
                    calls.extend(result.calls);
                } else {
                    // Incomplete tool call - log warning
                    tracing::warn!("Tool block ended with incomplete tool call");
                }
            }

            let remaining = self.buffer[idx + self.eot_token.len()..].to_string();
            self.buffer.clear();
            self.tool_block_finished = true;

            // Reset any partial tool call state
            self.reset_streaming_state();

            return Ok(StreamingParseResult {
                normal_text: remaining,
                calls,
            });
        }

        // Stage 4: Check if we're in a tool call or need to start one
        if !self.in_tool_call {
            if self.buffer.contains(self.tool_call_begin) {
                let idx = self.buffer.find(self.tool_call_begin).unwrap();
                // Remove any content before tool call begin (shouldn't happen but be safe)
                self.buffer = self.buffer[idx + self.tool_call_begin.len()..].to_string();
                self.in_tool_call = true;
                self.function_name_sent = false;
                self.current_function_name.clear();
                self.current_parameters.clear();
                // Fall through to parse the partial tool call
            } else {
                // Wait for tool call to begin
                return Ok(StreamingParseResult::default());
            }
        }

        // Stage 5: Parse partial tool call
        if self.in_tool_call {
            return self.parse_partial_tool_call(&tool_indices);
        }

        Ok(StreamingParseResult::default())
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.bot_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        // Reset standard state
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();

        // Reset Step3-specific fields
        self.in_tool_block = false;
        self.tool_block_finished = false;
        self.current_function_name.clear();
        self.current_parameters.clear();
        self.in_tool_call = false;
        self.function_name_sent = false;
    }
}
