use std::{collections::HashMap, fmt::Write as FmtWrite};

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

/// MiniMax M2 format parser for tool calls
///
/// Handles the MiniMax M2 specific format:
/// `<minimax:tool_call><invoke name="func"><parameter name="key">value</parameter></invoke></minimax:tool_call>`
///
/// Features:
/// - Namespaced XML tags (`minimax:tool_call`)
/// - Function wrapped in `<invoke name="...">` tags
/// - Parameters as `<parameter name="key">value</parameter>`
/// - Incremental JSON streaming for parameters
pub struct MinimaxM2Parser {
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting function details from invoke tag
    invoke_extractor: Regex,
    /// Regex for extracting parameter key-value pairs
    param_extractor: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Current function name being parsed
    current_function_name: String,

    /// Current parameters being accumulated
    current_parameters: HashMap<String, Value>,

    /// Whether we're inside a tool call block
    in_tool_call: bool,

    /// Whether the function name has been sent for current tool
    function_name_sent: bool,

    /// Whether we're waiting for </minimax:tool_call> after </invoke>
    waiting_for_tool_call_end: bool,

    /// Token configuration
    tool_call_start_token: &'static str,
    tool_call_end_token: &'static str,
    invoke_start_prefix: &'static str,
    invoke_end_token: &'static str,
}

impl MinimaxM2Parser {
    /// Parse a value from string with consistent logic
    #[inline]
    fn parse_value(text: &str) -> Value {
        // Try parsing as common literals first
        match text {
            "true" | "True" => return Value::Bool(true),
            "false" | "False" => return Value::Bool(false),
            "null" | "None" => return Value::Null,
            _ => {}
        }

        // Try parsing as number
        if let Ok(num) = text.parse::<i64>() {
            return Value::Number(num.into());
        }

        if let Ok(num) = text.parse::<f64>() {
            if let Some(n) = serde_json::Number::from_f64(num) {
                return Value::Number(n);
            }
        }

        // Default to string
        Value::String(text.to_string())
    }

    /// Create a new MiniMax M2 parser
    pub fn new() -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<minimax:tool_call>.*?</minimax:tool_call>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let invoke_pattern = r#"(?s)<invoke\s+name="([^"]+)">(.*?)</invoke>"#;
        let invoke_extractor = Regex::new(invoke_pattern).expect("Valid regex pattern");

        let param_pattern = r#"(?s)<parameter\s+name="([^"]+)">(.*?)</parameter>"#;
        let param_extractor = Regex::new(param_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            invoke_extractor,
            param_extractor,
            buffer: String::with_capacity(1024), // Pre-allocate reasonable capacity
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            current_function_name: String::new(),
            current_parameters: HashMap::new(),
            in_tool_call: false,
            function_name_sent: false,
            waiting_for_tool_call_end: false,
            tool_call_start_token: "<minimax:tool_call>",
            tool_call_end_token: "</minimax:tool_call>",
            invoke_start_prefix: r#"<invoke name=""#,
            invoke_end_token: "</invoke>",
        }
    }

    /// Parse parameters from parameter tags
    fn parse_parameters(&self, params_text: &str) -> ParserResult<serde_json::Map<String, Value>> {
        let mut parameters = serde_json::Map::new();

        for capture in self.param_extractor.captures_iter(params_text) {
            let key = capture.get(1).map_or("", |m| m.as_str()).trim();
            let value_str = capture.get(2).map_or("", |m| m.as_str());

            // Decode XML entities and parse value
            let decoded_value = self.decode_xml_entities(value_str);

            // Note: We keep JSON-like strings as strings (not parsed JSON)
            // This matches the behavior of other parsers like GLM4 MOE
            let value = Self::parse_value(&decoded_value);

            parameters.insert(key.to_string(), value);
        }

        Ok(parameters)
    }

    /// Decode common XML entities
    fn decode_xml_entities(&self, text: &str) -> String {
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str) -> ParserResult<Option<ToolCall>> {
        if let Some(captures) = self.invoke_extractor.captures(block) {
            // Get function name from invoke tag attribute
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

            // Get parameters text
            let params_text = captures.get(2).map_or("", |m| m.as_str());

            // Parse parameters
            let parameters = self.parse_parameters(params_text)?;

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

    /// Parse and stream parameters incrementally
    fn parse_and_stream_parameters(&mut self, text: &str, _tools: &[Tool]) -> Vec<ToolCallItem> {
        let mut calls = Vec::new();

        // Find all complete parameter patterns in the buffer
        let param_matches: Vec<_> = self
            .param_extractor
            .captures_iter(text)
            .map(|cap| {
                let name = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
                let value_str = cap.get(2).map_or("", |m| m.as_str());
                let decoded = self.decode_xml_entities(value_str);

                // Try parsing as JSON first (for nested objects/arrays)
                let value = if decoded.starts_with('{') || decoded.starts_with('[') {
                    if let Ok(json_val) = serde_json::from_str::<Value>(&decoded) {
                        json_val
                    } else {
                        Self::parse_value(&decoded)
                    }
                } else {
                    Self::parse_value(&decoded)
                };

                (name, value)
            })
            .collect();

        // Build new parameters map
        let mut new_params = HashMap::new();
        for (name, value) in param_matches {
            new_params.insert(name, value);
        }

        // If we have new parameters that weren't in current_parameters, stream them
        if !new_params.is_empty() && new_params != self.current_parameters {
            let tool_id = self.current_tool_id as usize;

            // Ensure we have enough capacity
            while self.streamed_args_for_tool.len() <= tool_id {
                self.streamed_args_for_tool.push(String::new());
            }

            // Build incremental JSON with single allocation
            if self.current_parameters.is_empty() {
                // First parameters - start JSON object but don't close it
                let mut json_fragment = String::with_capacity(256);
                json_fragment.push('{');

                let mut first = true;
                for (key, value) in &new_params {
                    if !first {
                        json_fragment.push_str(", ");
                    }
                    write!(
                        &mut json_fragment,
                        "{}: {}",
                        serde_json::to_string(key).unwrap(),
                        serde_json::to_string(value).unwrap()
                    )
                    .unwrap();
                    first = false;
                }

                calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: None,
                    parameters: json_fragment.clone(),
                });

                self.streamed_args_for_tool[tool_id] = json_fragment;
            } else {
                // Additional parameters - add them incrementally
                let new_keys: Vec<_> = new_params
                    .keys()
                    .filter(|k| !self.current_parameters.contains_key(*k))
                    .collect();

                if !new_keys.is_empty() {
                    let mut json_fragment = String::with_capacity(128);

                    for key in new_keys {
                        let value = &new_params[key];
                        write!(
                            &mut json_fragment,
                            ", {}: {}",
                            serde_json::to_string(key).unwrap(),
                            serde_json::to_string(value).unwrap()
                        )
                        .unwrap();
                    }

                    calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: json_fragment.clone(),
                    });

                    self.streamed_args_for_tool[tool_id].push_str(&json_fragment);
                }
            }

            // Update current parameters
            self.current_parameters = new_params;

            // Update prev_tool_call_arr
            while self.prev_tool_call_arr.len() <= tool_id {
                self.prev_tool_call_arr.push(Value::Null);
            }
            self.prev_tool_call_arr[tool_id] = serde_json::json!({
                "name": self.current_function_name,
                "arguments": self.current_parameters,
            });
        }

        calls
    }
}

impl Default for MinimaxM2Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for MinimaxM2Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains MiniMax M2 format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Parse all tool calls using shared helper (filters out ones in thinking tags)
        let tools = self.parse_tool_calls_from_text(text)?;

        // If no tools were successfully parsed, return entire text as fallback
        if tools.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        // Find the position of the first successfully extracted tool call
        // We need to match the extracted tools with their positions in text
        let mut first_valid_tool_pos = None;
        for mat in self.tool_call_extractor.find_iter(text) {
            // Check if this tool call was successfully extracted
            // by trying to parse it
            if let Ok(Some(_)) = self.parse_tool_call(mat.as_str()) {
                first_valid_tool_pos = Some(mat.start());
                break;
            }
        }

        // Determine what text to return as normal_text
        let normal_text = if let Some(pos) = first_valid_tool_pos {
            // Return text up to the first valid tool call
            text[..pos].to_string()
        } else {
            // No valid tool calls found, return entire text
            text.to_string()
        };

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let mut normal_text = String::new();
        let mut calls = Vec::new();

        // Build tool indices for validation
        let tool_indices = helpers::get_tool_indices(tools);

        loop {
            // If we're waiting for the tool call end tag, check for it first
            if self.waiting_for_tool_call_end {
                if let Some(end_pos) = self.buffer.find(self.tool_call_end_token) {
                    // Complete tool call found
                    self.buffer =
                        self.buffer[end_pos + self.tool_call_end_token.len()..].to_string();
                    self.in_tool_call = false;
                    self.waiting_for_tool_call_end = false;
                    self.function_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();
                    self.current_tool_id += 1;
                    continue;
                } else {
                    // End tag not complete yet, wait for more text
                    break;
                }
            }

            // If we're not in a tool call and don't see a start token, return normal text
            if !self.in_tool_call && !self.buffer.contains(self.tool_call_start_token) {
                // Check if buffer might contain a partial start token at the end
                // We need to keep potential partial tokens in the buffer
                let mut could_be_partial = false;
                for i in 1..self.tool_call_start_token.len() {
                    if self.buffer.ends_with(&self.tool_call_start_token[..i]) {
                        could_be_partial = true;
                        // Return everything except the potential partial token
                        let end = self.buffer.len() - i;
                        normal_text = self.buffer[..end].to_string();
                        self.buffer = self.buffer[end..].to_string();
                        break;
                    }
                }

                if !could_be_partial {
                    // No partial token, return all as normal text
                    normal_text = self.buffer.clone();
                    self.buffer.clear();
                }
                break;
            }

            // Look for tool call start
            if !self.in_tool_call {
                if let Some(start) = self.buffer.find(self.tool_call_start_token) {
                    normal_text = self.buffer[..start].to_string();
                    self.buffer =
                        self.buffer[start + self.tool_call_start_token.len()..].to_string();

                    self.in_tool_call = true;
                    self.function_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();

                    continue;
                } else {
                    // No start token found
                    break;
                }
            }

            // We're in a tool call, try to parse function name if not sent yet
            if !self.function_name_sent {
                // Look for function name pattern: <invoke name="...">
                if let Some(pos) = self.buffer.find(self.invoke_start_prefix) {
                    // Find the closing quote after name=
                    let name_start = pos + self.invoke_start_prefix.len();
                    if let Some(quote_end) = self.buffer[name_start..].find('"') {
                        let function_name = self.buffer[name_start..name_start + quote_end]
                            .trim()
                            .to_string();

                        // Validate function name
                        if tool_indices.contains_key(&function_name) {
                            self.current_function_name = function_name.clone();
                            self.function_name_sent = true;

                            // Initialize tool call tracking
                            if self.current_tool_id == -1 {
                                self.current_tool_id = 0;
                            }

                            // Ensure tracking arrays are large enough
                            while self.prev_tool_call_arr.len() <= self.current_tool_id as usize {
                                self.prev_tool_call_arr.push(Value::Null);
                            }
                            while self.streamed_args_for_tool.len() <= self.current_tool_id as usize
                            {
                                self.streamed_args_for_tool.push(String::new());
                            }

                            // Send tool name with empty parameters
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: Some(function_name),
                                parameters: String::new(),
                            });

                            // Remove processed part up to and including the closing >
                            if let Some(close_bracket) =
                                self.buffer[name_start + quote_end..].find('>')
                            {
                                self.buffer = self.buffer
                                    [name_start + quote_end + close_bracket + 1..]
                                    .to_string();
                            }
                            continue;
                        } else {
                            // Invalid function name, reset state
                            tracing::warn!("Invalid function name: {}", function_name);
                            self.in_tool_call = false;
                            normal_text.push_str(&self.buffer);
                            self.buffer.clear();
                            break;
                        }
                    }
                }
                // Function name not complete yet, wait for more text
                break;
            }

            // Parse parameters incrementally
            if self.function_name_sent {
                // Process parameters and get any calls to emit
                // Note: We need to be careful here - parse_and_stream_parameters needs
                // to work with the buffer but we can't pass &self.buffer directly
                // due to borrow checker. Instead, we'll refactor slightly.
                // For now, keep the clone but mark it as a TODO for future optimization
                let buffer_copy = self.buffer.clone(); // TODO: Optimize this
                let parameter_calls = self.parse_and_stream_parameters(&buffer_copy, tools);
                calls.extend(parameter_calls);

                // Check if tool call is complete (</invoke> found)
                if let Some(invoke_end) = self.buffer.find(self.invoke_end_token) {
                    // Add closing brace to complete the JSON object
                    let tool_id = self.current_tool_id as usize;
                    if tool_id < self.streamed_args_for_tool.len() {
                        let current_streamed = &self.streamed_args_for_tool[tool_id];
                        if !current_streamed.is_empty() && !current_streamed.ends_with('}') {
                            // Count opening and closing braces to check if JSON is complete
                            let open_braces = current_streamed.matches('{').count();
                            let close_braces = current_streamed.matches('}').count();
                            if open_braces > close_braces {
                                calls.push(ToolCallItem {
                                    tool_index: tool_id,
                                    name: None,
                                    parameters: "}".to_string(),
                                });
                                self.streamed_args_for_tool[tool_id].push('}');
                            }
                        }
                    }

                    // Move buffer past the </invoke>
                    self.buffer =
                        self.buffer[invoke_end + self.invoke_end_token.len()..].to_string();

                    // Check if we have the closing </minimax:tool_call>
                    if let Some(end_pos) = self.buffer.find(self.tool_call_end_token) {
                        // Complete tool call found
                        self.buffer =
                            self.buffer[end_pos + self.tool_call_end_token.len()..].to_string();
                        self.in_tool_call = false;
                        self.function_name_sent = false;
                        self.current_function_name.clear();
                        self.current_parameters.clear();
                        self.current_tool_id += 1;
                        continue;
                    } else {
                        // End tag not complete yet, mark that we're waiting for it
                        self.waiting_for_tool_call_end = true;
                        break;
                    }
                }
                // Tool call not complete yet, wait for more text
                break;
            }
        }

        Ok(StreamingParseResult { normal_text, calls })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.tool_call_start_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
        self.current_function_name.clear();
        self.current_parameters.clear();
        self.in_tool_call = false;
        self.function_name_sent = false;
        self.waiting_for_tool_call_end = false;
    }
}
