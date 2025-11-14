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

/// Qwen Coder format parser for tool calls
///
/// Handles the Qwen Coder specific XML format:
/// `<tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>`
///
/// Features:
/// - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
/// - XML-style function declaration: `<function=name>`
/// - XML-style parameters: `<parameter=key>value</parameter>`
///
/// Reference: Python implementation in qwen3_coder_detector.py
pub struct QwenCoderParser {
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

    /// Token configuration
    individual_tool_start_token: &'static str,
    tool_call_separator: &'static str,

    /// XML format streaming state
    /// Whether we're currently parsing an XML format tool call
    in_xml_tool_call: bool,
    /// Current function name for XML format
    xml_current_function_name: String,
    /// Current parameters for XML format (as JSON map)
    xml_current_parameters: serde_json::Map<String, Value>,
    /// Streamed parameters for XML format (for diff calculation)
    xml_streamed_parameters: serde_json::Map<String, Value>,
    /// Whether we're currently inside a parameter tag
    in_parameter: bool,
    /// Current parameter key being parsed
    current_parameter_key: String,
    /// Buffer for current parameter value (accumulated across chunks)
    current_parameter_value: String,

    /// Precompiled regex patterns for XML format parsing
    /// Pattern for matching function name: <function=name>
    xml_function_pattern: Regex,
    /// Pattern for matching complete parameter: <parameter=key>value</parameter>
    xml_param_pattern: Regex,
    /// Pattern for matching parameter start tag: <parameter=key>
    xml_param_start_pattern: Regex,
}

impl QwenCoderParser {
    /// Create a new Qwen Coder parser
    pub fn new() -> Self {
        // Support XML format: <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>
        let pattern = r"(?s)<tool_call>\s*(.*?)\s*</tool_call>";
        let extractor = Regex::new(pattern).expect("Valid regex pattern");

        // Precompile XML format regex patterns for performance
        // Use (?s) flag for DOTALL mode to handle newlines in parameter values
        let xml_function_pattern =
            Regex::new(r"<function=([^>]+)>").expect("Valid XML function pattern");
        let xml_param_pattern = Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>")
            .expect("Valid XML parameter pattern");
        let xml_param_start_pattern =
            Regex::new(r"<parameter=([^>]+)>").expect("Valid XML parameter start pattern");

        Self {
            extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            individual_tool_start_token: "<tool_call>\n",
            tool_call_separator: "\n",
            in_xml_tool_call: false,
            xml_current_function_name: String::new(),
            xml_current_parameters: serde_json::Map::new(),
            xml_streamed_parameters: serde_json::Map::new(),
            in_parameter: false,
            current_parameter_key: String::new(),
            current_parameter_value: String::new(),
            xml_function_pattern,
            xml_param_pattern,
            xml_param_start_pattern,
        }
    }

    /// Parse XML format tool call: <function=name><parameter=key>value</parameter></function>
    fn parse_xml_format(&self, content: &str) -> ParserResult<Option<ToolCall>> {
        // Use precompiled regex patterns
        let function_captures = self
            .xml_function_pattern
            .captures(content)
            .ok_or_else(|| ParserError::ParsingFailed("No function name found".to_string()))?;

        let function_name = function_captures
            .get(1)
            .ok_or_else(|| ParserError::ParsingFailed("Function name capture failed".to_string()))?
            .as_str()
            .trim()
            .to_string();

        if function_name.is_empty() {
            return Ok(None);
        }

        let mut parameters = serde_json::Map::new();

        for cap in self.xml_param_pattern.captures_iter(content) {
            if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                let key = key_match.as_str().trim().to_string();
                let value = value_match.as_str().trim();

                // Try to parse value as JSON (similar to Python's _safe_val which tries json.loads)
                // This will parse numbers, booleans, null, objects, arrays, and strings
                let json_value = match serde_json::from_str::<Value>(value) {
                    Ok(v) => v,
                    Err(_) => {
                        // If JSON parsing fails, keep as string
                        Value::String(value.to_string())
                    }
                };
                parameters.insert(key, json_value);
            }
        }

        let arguments = serde_json::to_string(&parameters)
            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(Some(ToolCall {
            function: FunctionCall {
                name: function_name,
                arguments,
            },
        }))
    }

    /// Parse XML format tool calls incrementally (similar to Python Qwen3CoderDetector)
    fn parse_xml_incremental(
        &mut self,
        current_text: &str,
        tool_indices: &std::collections::HashMap<String, usize>,
    ) -> ParserResult<StreamingParseResult> {
        let mut normal_text = String::new();
        let mut calls: Vec<ToolCallItem> = vec![];

        // If we're not in a tool call and don't see a start token, return normal text
        if !self.in_xml_tool_call && !current_text.contains("<tool_call>") {
            normal_text = self.buffer.clone();
            self.buffer.clear();
            return Ok(StreamingParseResult { normal_text, calls });
        }

        // Look for tool call start
        if !self.in_xml_tool_call {
            if let Some(s) = current_text.find("<tool_call>") {
                normal_text.push_str(&current_text[..s]);
                self.buffer = current_text[s + "<tool_call>".len()..].to_string();
                self.in_xml_tool_call = true;
                self.xml_current_function_name.clear();
                self.xml_current_parameters.clear();
                self.xml_streamed_parameters.clear();
                self.current_tool_name_sent = false;
                self.in_parameter = false;
                self.current_parameter_key.clear();
                self.current_parameter_value.clear();
            } else {
                // Partial start token, keep buffering
                return Ok(StreamingParseResult::default());
            }
        }

        // We're in a tool call, try to parse function name if not sent yet
        if !self.current_tool_name_sent {
            // Use precompiled regex pattern
            if let Some(captures) = self.xml_function_pattern.captures(&self.buffer) {
                if let Some(name_match) = captures.get(1) {
                    let function_name = name_match.as_str().trim().to_string();

                    // Validate function name
                    if tool_indices.contains_key(&function_name) {
                        self.xml_current_function_name = function_name.clone();
                        self.current_tool_name_sent = true;

                        // Initialize tool call tracking
                        if self.current_tool_id == -1 {
                            self.current_tool_id = 0;
                        }

                        // Ensure tracking arrays are large enough
                        while self.prev_tool_call_arr.len() <= self.current_tool_id as usize {
                            self.prev_tool_call_arr
                                .push(Value::Object(serde_json::Map::new()));
                        }
                        while self.streamed_args_for_tool.len() <= self.current_tool_id as usize {
                            self.streamed_args_for_tool.push(String::new());
                        }

                        // Store tool call info
                        let mut tool_obj = serde_json::Map::new();
                        tool_obj.insert("name".to_string(), Value::String(function_name.clone()));
                        tool_obj.insert(
                            "arguments".to_string(),
                            Value::Object(serde_json::Map::new()),
                        );
                        self.prev_tool_call_arr[self.current_tool_id as usize] =
                            Value::Object(tool_obj);

                        // Send tool name with empty parameters
                        calls.push(ToolCallItem {
                            tool_index: self.current_tool_id as usize,
                            name: Some(function_name),
                            parameters: String::new(),
                        });

                        // Remove the processed function declaration
                        self.buffer = self.buffer[captures.get(0).unwrap().end()..].to_string();
                    } else {
                        // Invalid function name, reset state
                        self.in_xml_tool_call = false;
                        normal_text.push_str(&self.buffer);
                        self.buffer.clear();
                        return Ok(StreamingParseResult { normal_text, calls });
                    }
                }
            }
        }

        // Parse parameters incrementally
        if self.current_tool_name_sent {
            // First, try to find and parse any complete parameter blocks in the buffer
            // This handles cases where parameters arrive in chunks but are complete
            for cap in self.xml_param_pattern.captures_iter(&self.buffer) {
                if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                    let key = key_match.as_str().trim().to_string();
                    let value = value_match.as_str().trim();

                    // Only process if we haven't already streamed this parameter
                    if !self.xml_streamed_parameters.contains_key(&key) {
                        // Try to parse value as JSON (similar to Python's _safe_val which tries json.loads)
                        // This will parse numbers, booleans, null, objects, arrays, and strings
                        let json_value = match serde_json::from_str::<Value>(value) {
                            Ok(v) => v,
                            Err(_) => {
                                // If JSON parsing fails, keep as string
                                Value::String(value.to_string())
                            }
                        };

                        // Add to current parameters
                        self.xml_current_parameters
                            .insert(key.clone(), json_value.clone());

                        // Stream the parameter update
                        let value_json = serde_json::to_string(&json_value)
                            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

                        let json_fragment = if self.xml_streamed_parameters.is_empty() {
                            format!("{{\"{}\": {}}}", key, value_json)
                        } else {
                            format!(", \"{}\": {}", key, value_json)
                        };

                        calls.push(ToolCallItem {
                            tool_index: self.current_tool_id as usize,
                            name: None,
                            parameters: json_fragment.clone(),
                        });

                        // Update streamed args
                        let current_args =
                            &mut self.streamed_args_for_tool[self.current_tool_id as usize];
                        if current_args.is_empty() {
                            *current_args = format!("{{\"{}\": {}}}", key, value_json);
                        } else {
                            // Trim trailing whitespace before checking for closing brace
                            let trimmed = current_args.trim_end();
                            if let Some(stripped) = trimmed.strip_suffix('}') {
                                *current_args = format!("{}{}}}", stripped, json_fragment);
                            } else {
                                *current_args = format!("{}{}", trimmed, json_fragment);
                            }
                        }

                        // Update streamed parameters
                        self.xml_streamed_parameters.insert(key, json_value);
                    }
                }
            }

            // Use precompiled regex pattern
            // Check if we're entering a new parameter
            if !self.in_parameter {
                if let Some(cap) = self.xml_param_start_pattern.captures(&self.buffer) {
                    if let Some(key_match) = cap.get(1) {
                        self.current_parameter_key = key_match.as_str().trim().to_string();
                        self.current_parameter_value.clear();
                        self.in_parameter = true;

                        // Remove the opening tag from buffer
                        if let Some(m) = cap.get(0) {
                            self.buffer = self.buffer[m.end()..].to_string();
                        }
                    }
                }
            }

            // If we're in a parameter, accumulate value until we see </parameter>
            if self.in_parameter {
                if let Some(end_pos) = self.buffer.find("</parameter>") {
                    // Found complete parameter
                    let value = self.buffer[..end_pos].trim().to_string();
                    self.current_parameter_value.push_str(&value);

                    // Remove the closing tag and processed content from buffer
                    self.buffer = self.buffer[end_pos + "</parameter>".len()..].to_string();

                    // Parse and add the parameter
                    let key = self.current_parameter_key.clone();
                    let value_str = self.current_parameter_value.trim().to_string();

                    // Try to parse value as JSON (similar to Python's _safe_val which tries json.loads)
                    // This will parse numbers, booleans, null, objects, arrays, and strings
                    let json_value = match serde_json::from_str::<Value>(&value_str) {
                        Ok(v) => v,
                        Err(_) => {
                            // If JSON parsing fails, keep as string
                            Value::String(value_str)
                        }
                    };

                    // Add to current parameters
                    self.xml_current_parameters
                        .insert(key.clone(), json_value.clone());

                    // Stream the parameter update
                    let value_json = serde_json::to_string(&json_value)
                        .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

                    let json_fragment = if self.xml_streamed_parameters.is_empty() {
                        format!("{{\"{}\": {}}}", key, value_json)
                    } else {
                        format!(", \"{}\": {}", key, value_json)
                    };

                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: json_fragment.clone(),
                    });

                    // Update streamed args
                    let current_args =
                        &mut self.streamed_args_for_tool[self.current_tool_id as usize];
                    if current_args.is_empty() {
                        *current_args = format!("{{\"{}\": {}}}", key, value_json);
                    } else {
                        // Trim trailing whitespace before checking for closing brace
                        // This ensures robust handling even if there's trailing whitespace
                        let trimmed = current_args.trim_end();
                        if let Some(stripped) = trimmed.strip_suffix('}') {
                            // Remove the closing brace, add new parameter, add closing brace
                            // Use stripped string to avoid issues with trailing whitespace
                            *current_args = format!("{}{}}}", stripped, json_fragment);
                        } else {
                            // No closing brace found, append the fragment directly
                            // Trim any trailing whitespace first to ensure clean JSON
                            *current_args = format!("{}{}", trimmed, json_fragment);
                        }
                    }

                    // Update streamed parameters
                    self.xml_streamed_parameters.insert(key, json_value);

                    // Reset parameter state
                    self.in_parameter = false;
                    self.current_parameter_key.clear();
                    self.current_parameter_value.clear();

                    // Update prev_tool_call_arr
                    if let Some(tool_obj) =
                        self.prev_tool_call_arr[self.current_tool_id as usize].as_object_mut()
                    {
                        tool_obj.insert(
                            "arguments".to_string(),
                            Value::Object(self.xml_current_parameters.clone()),
                        );
                    }
                } else {
                    // Parameter value is incomplete, accumulate it
                    // Check if there's any content before a potential partial closing tag
                    if let Some(partial_end) = self.buffer.find("</") {
                        // There might be a partial closing tag, only take content before it
                        self.current_parameter_value
                            .push_str(&self.buffer[..partial_end]);
                        self.buffer = self.buffer[partial_end..].to_string();
                    } else {
                        // No closing tag yet, accumulate all content
                        self.current_parameter_value.push_str(&self.buffer);
                        self.buffer.clear();
                    }
                }
            }

            // Check if tool call is complete
            if self.buffer.contains("</tool_call>") {
                // Before completing, check if we need to send final parameters
                // Only send if we have parameters that haven't been fully streamed
                if !self.xml_current_function_name.is_empty()
                    && !self.xml_current_parameters.is_empty()
                {
                    // Check if all parameters have been streamed
                    let all_streamed = self
                        .xml_current_parameters
                        .iter()
                        .all(|(k, _)| self.xml_streamed_parameters.contains_key(k));

                    if !all_streamed {
                        // Some parameters haven't been streamed, send final complete arguments
                        let final_args_json =
                            serde_json::to_string(&self.xml_current_parameters)
                                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

                        calls.push(ToolCallItem {
                            tool_index: self.current_tool_id as usize,
                            name: None, // Final update, no name change
                            parameters: final_args_json,
                        });
                    } else if !self.xml_streamed_parameters.is_empty() {
                        // All parameters streamed, but ensure JSON is complete
                        // Check if streamed args JSON is complete (has closing brace)
                        // Trim trailing whitespace before checking to handle edge cases
                        let streamed_args =
                            &self.streamed_args_for_tool[self.current_tool_id as usize];
                        if !streamed_args.trim_end().ends_with('}') && !streamed_args.is_empty() {
                            // JSON incomplete, send closing brace
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: None,
                                parameters: "}".to_string(),
                            });
                        }
                    }
                }

                // Complete the tool call
                if let Some(end_pos) = self.buffer.find("</tool_call>") {
                    self.buffer = self.buffer[end_pos + "</tool_call>".len()..].to_string();
                }
                self.in_xml_tool_call = false;
                self.current_tool_id += 1;
                self.xml_current_function_name.clear();
                self.xml_current_parameters.clear();
                self.xml_streamed_parameters.clear();
                self.current_tool_name_sent = false;
                self.in_parameter = false;
                self.current_parameter_key.clear();
                self.current_parameter_value.clear();
            }
        }

        Ok(StreamingParseResult { normal_text, calls })
    }
}

impl Default for QwenCoderParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for QwenCoderParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Qwen Coder format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where the first tool call begins
        let idx = text.find("<tool_call>").unwrap(); // Safe because has_tool_markers checked
        let normal_text = text[..idx].to_string();

        // Extract tool calls
        let mut tools = Vec::new();
        for captures in self.extractor.captures_iter(text) {
            if let Some(content_str) = captures.get(1) {
                let content = content_str.as_str().trim();

                // Parse XML format
                match self.parse_xml_format(content) {
                    Ok(Some(tool)) => tools.push(tool),
                    Ok(None) => continue,
                    Err(e) => {
                        tracing::warn!("Failed to parse XML tool call: {:?}", e);
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
            || (self.current_tool_id > 0 && current_text.starts_with(self.tool_call_separator))
            || self.in_xml_tool_call;

        if !has_tool_start {
            // Only clear buffer if we're sure no tool call is starting
            if helpers::ends_with_partial_token(&self.buffer, self.individual_tool_start_token)
                .is_none()
            {
                let normal_text = self.buffer.clone();
                self.buffer.clear();

                return Ok(StreamingParseResult {
                    normal_text,
                    calls: vec![],
                });
            } else {
                // Might be partial individual_tool_start_token, keep buffering
                return Ok(StreamingParseResult::default());
            }
        }

        // Build tool indices
        let tool_indices = helpers::get_tool_indices(tools);

        // Use XML format streaming parsing
        self.parse_xml_incremental(current_text, &tool_indices)
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<tool_call>")
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
        // Reset XML format state
        self.in_xml_tool_call = false;
        self.xml_current_function_name.clear();
        self.xml_current_parameters.clear();
        self.xml_streamed_parameters.clear();
        self.in_parameter = false;
        self.current_parameter_key.clear();
        self.current_parameter_value.clear();
    }
}
