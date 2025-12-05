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
/// Reference: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8?chat_template=default
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
    /// Currently streaming parameter key (if any)
    xml_streaming_parameter_key: Option<String>,
    /// Streamed parameter value string (for incremental updates)
    xml_streamed_parameter_value: String,
    /// Whether we're currently inside a parameter tag
    in_parameter: bool,
    /// Current parameter key being streamed
    current_parameter_key: String,
    /// Current parameter value being accumulated
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
            xml_streaming_parameter_key: None,
            xml_streamed_parameter_value: String::new(),
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
                        helpers::ensure_capacity(
                            self.current_tool_id,
                            &mut self.prev_tool_call_arr,
                            &mut self.streamed_args_for_tool,
                        );

                        // Store tool call info
                        self.prev_tool_call_arr[self.current_tool_id as usize] = serde_json::json!({
                            "name": function_name,
                            "arguments": {}
                        });

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
            let mut processed_ranges: Vec<(usize, usize)> = Vec::new();

            for cap in self.xml_param_pattern.captures_iter(&self.buffer) {
                if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                    let key = key_match.as_str().trim().to_string();
                    let value = value_match.as_str().trim();

                    // Only process if we haven't already streamed this parameter
                    if !self.xml_streamed_parameters.contains_key(&key) {
                        // Get the full match range to remove from buffer later
                        let full_match = cap.get(0).unwrap();
                        let start = full_match.start();
                        let end = full_match.end();

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

                        // Note: {{ and }} are Rust string formatting escape sequences
                        // {{ becomes a single {, }} becomes a single }
                        // So format!("{{\"{}\": {}}}", key, value_json) produces: {"key": value_json}
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
                        self.xml_streamed_parameters.insert(key.clone(), json_value);

                        // Clear incremental streaming state for this parameter since it's now complete
                        if self.xml_streaming_parameter_key.as_ref() == Some(&key) {
                            self.xml_streaming_parameter_key = None;
                            self.xml_streamed_parameter_value.clear();
                        }

                        // Record the range to remove from buffer
                        processed_ranges.push((start, end));
                    }
                }
            }

            // Use precompiled regex pattern
            // Check if we're entering a new parameter (for streaming/incremental parsing)
            // Note: We only enter this path if the parameter is NOT complete (no </parameter> yet)
            // Complete parameters are handled by the fast path above (xml_param_pattern)
            if !self.in_parameter {
                if let Some(cap) = self.xml_param_start_pattern.captures(&self.buffer) {
                    if let Some(key_match) = cap.get(1) {
                        let key = key_match.as_str().trim().to_string();
                        // Skip if this parameter was already processed as complete
                        if !self.xml_streamed_parameters.contains_key(&key) {
                            self.current_parameter_key = key;
                        self.current_parameter_value.clear();
                        self.in_parameter = true;

                            // Remove the opening tag from buffer to avoid re-matching
                            // This is necessary for streaming: we need to accumulate only the value
                            // without the tag, so that when </parameter> arrives, we can extract
                            // the clean value. If we don't remove the tag, it would interfere
                            // with value accumulation and cause duplicate matches.
                        if let Some(m) = cap.get(0) {
                            self.buffer = self.buffer[m.end()..].to_string();
                                // Trim leading whitespace after opening tag (model may include newlines)
                                self.buffer = self.buffer.trim_start().to_string();
                            }
                        }
                    }
                }
            }

            // If we're in a parameter, accumulate value until we see </parameter>
            if self.in_parameter {
                if let Some(end_pos) = self.buffer.find("</parameter>") {
                    // Found complete parameter
                    // If we've been streaming, current_parameter_value already contains the full value
                    // Otherwise, extract from buffer
                    if self.current_parameter_value.trim().is_empty() {
                        // Not streamed yet, extract from buffer
                    let value = self.buffer[..end_pos].trim().to_string();
                        self.current_parameter_value = value;
                    } else {
                        // Already streaming, current_parameter_value has the value
                        // Just extract any remaining part from buffer (should be minimal)
                        let remaining = self.buffer[..end_pos].trim();
                        if !remaining.is_empty() && !self.current_parameter_value.ends_with(remaining) {
                            // Only append if it's truly new content
                            self.current_parameter_value.push_str(remaining);
                        }
                    }

                    // Remove the closing tag and processed content from buffer
                    self.buffer = self.buffer[end_pos + "</parameter>".len()..].to_string();

                    // Parse and add the parameter
                    let key = self.current_parameter_key.clone();
                    // Trim the accumulated value to remove leading/trailing whitespace from XML format
                    // This removes any trailing newlines or whitespace that might have been included
                    self.current_parameter_value = self.current_parameter_value.trim().to_string();
                    let value_str = self.current_parameter_value.clone();

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

                    // Check if this parameter was already streamed incrementally
                    // Check if current_args already contains this key
                    let current_args =
                        &mut self.streamed_args_for_tool[self.current_tool_id as usize];
                    let key_pattern = format!("\"{}\"", key);
                    let was_streamed = current_args.contains(&key_pattern);

                    if !was_streamed {
                        // Parameter wasn't streamed yet, send complete JSON fragment
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
                    if current_args.is_empty() {
                        *current_args = format!("{{\"{}\": {}}}", key, value_json);
                    } else {
                        let trimmed = current_args.trim_end();
                        if let Some(stripped) = trimmed.strip_suffix('}') {
                            *current_args = format!("{}{}}}", stripped, json_fragment);
                        } else {
                            *current_args = format!("{}{}", trimmed, json_fragment);
                            }
                        }
                    } else {
                        // Parameter was already streamed incrementally
                        // value_str is already trimmed, but the streamed value may have trailing whitespace
                        // We need to find and replace the parameter value in current_args with the trimmed version
                        let key_pattern = format!("\"{}\":", key);
                        if let Some(key_pos) = current_args.find(&key_pattern) {
                            // Find where the value starts (after ": " or ":")
                            let after_key = &current_args[key_pos + key_pattern.len()..];
                            let value_start_marker = if after_key.starts_with(" \"") {
                                ": \""
                            } else if after_key.starts_with(':') {
                                ":"
                            } else {
                                ""
                            };

                            if !value_start_marker.is_empty() {
                                if let Some(marker_pos) = current_args[key_pos..].find(value_start_marker) {
                                    let value_start = key_pos + marker_pos + value_start_marker.len();
                                    // Find where the value ends (before closing quote, comma, or })
                                    // For string values, look for the closing quote
                                    if let Some(quote_end) = current_args[value_start..].find('"') {
                                        let value_in_args = &current_args[value_start..value_start + quote_end];
                                        // Remove trailing whitespace, including escaped newlines (\\n)
                                        let mut trimmed_value = value_in_args.trim_end().to_string();
                                        // Also remove trailing \\n (escaped newline in JSON)
                                        while trimmed_value.ends_with("\\n") {
                                            trimmed_value = trimmed_value[..trimmed_value.len() - 2].to_string();
                                        }
                                        // Trim again in case there were spaces before \\n
                                        trimmed_value = trimmed_value.trim_end().to_string();

                                        // If there's trailing whitespace or \\n, replace it
                                        if value_in_args != trimmed_value {
                                            let before_value = &current_args[..value_start];
                                            let after_value = &current_args[value_start + quote_end..];
                                            *current_args = format!("{}{}\"{}", before_value, trimmed_value, after_value);
                                        }
                                    }
                                }
                            }
                        }

                        // Ensure string is closed
                        let final_trimmed = current_args.trim_end();
                        if !final_trimmed.ends_with('"') && !final_trimmed.ends_with('}') {
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: None,
                                parameters: "\"".to_string(),
                            });
                            *current_args = format!("{}\"", final_trimmed);
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
                    // Parameter value still streaming - accumulate and send incremental updates
                    // Extract value up to next XML tag (to avoid including </parameter> in value)
                    let next_tag_pos = self.buffer.find('<').unwrap_or(self.buffer.len());
                    let mut new_content = self.buffer[..next_tag_pos].to_string();

                    // Always trim trailing whitespace from new_content to avoid accumulating newlines
                    // This prevents newlines from being accumulated during streaming
                    let trimmed_new = new_content.trim_end();
                    if trimmed_new != new_content {
                        // Trim trailing whitespace to prevent newline accumulation
                        new_content = trimmed_new.to_string();
                        // Also trim current value to keep them in sync
                        self.current_parameter_value = self.current_parameter_value.trim_end().to_string();
                    }

                    if new_content != self.current_parameter_value {
                        // Calculate incremental value (only new characters)
                        // current_parameter_value is already trimmed, so use it directly
                        let incremental = if new_content.starts_with(&self.current_parameter_value) {
                            &new_content[self.current_parameter_value.len()..]
                        } else {
                            // Value changed unexpectedly, use full new content
                            &new_content
                        };

                        if !incremental.is_empty() {
                            let key = &self.current_parameter_key;

                            // Escape incremental value for JSON string using serde_json
                            // This is more reliable than manual replace chains
                            let escaped = serde_json::to_string(incremental)
                                .unwrap_or_else(|_| incremental.replace('"', "\\\""))
                                .trim_matches('"')
                                .to_string();

                            // Stream only the incremental value (not full JSON fragment)
                            let current_args =
                                &mut self.streamed_args_for_tool[self.current_tool_id as usize];

                            if current_args.is_empty() {
                                // First parameter - start JSON object
                                let fragment = format!("{{\"{}\": \"{}", key, escaped);
                                calls.push(ToolCallItem {
                                    tool_index: self.current_tool_id as usize,
                                    name: None,
                                    parameters: fragment.clone(),
                                });
                                *current_args = fragment;
                            } else {
                                // Check if we're starting a new parameter or continuing current one
                                let trimmed = current_args.trim_end();

                                // Check if current_args already contains this key (parameter already started)
                                let key_pattern = format!("\"{}\"", key);
                                let key_already_in_args = trimmed.contains(&key_pattern);

                                if key_already_in_args {
                                    // This parameter was already started, just append incremental value
                                    // Check if we're in the middle of the value (string not closed)
                                    if trimmed.ends_with('"') {
                                        // String already closed, this shouldn't happen for streaming
                                        // But if it does, we need to reopen it or it's a new parameter
                                        // Actually, if string is closed, we shouldn't be here
                                        // This means parameter was completed, so skip
                                    } else {
                                        // Continuing current parameter value - just append escaped chars
                                        calls.push(ToolCallItem {
                                            tool_index: self.current_tool_id as usize,
                                            name: None,
                                            parameters: escaped.clone(),
                                        });
                                        // Remove trailing quote if present (shouldn't be, but be safe)
                                        if let Some(stripped) = trimmed.strip_suffix('"') {
                                            *current_args = format!("{}{}\"", stripped, escaped);
                                        } else {
                                            *current_args = format!("{}{}", trimmed, escaped);
                                        }
                                    }
                                } else {
                                    // New parameter - previous one must be complete
                                    if trimmed.ends_with('}') {
                                        // Previous parameter complete, start new one
                                        let fragment = format!(", \"{}\": \"{}", key, escaped);
                                        calls.push(ToolCallItem {
                                            tool_index: self.current_tool_id as usize,
                                            name: None,
                                            parameters: fragment.clone(),
                                        });
                                        if let Some(stripped) = trimmed.strip_suffix('}') {
                                            *current_args = format!("{}{}}}", stripped, fragment);
                                        }
                    } else {
                                        // Previous parameter not properly closed, close it first
                                        if !trimmed.ends_with('"') {
                                            // Close the previous parameter's string
                                            calls.push(ToolCallItem {
                                                tool_index: self.current_tool_id as usize,
                                                name: None,
                                                parameters: "\"".to_string(),
                                            });
                                            *current_args = format!("{}\"", trimmed);
                                        }
                                        // Now start new parameter
                                        let fragment = format!(", \"{}\": \"{}", key, escaped);
                                        calls.push(ToolCallItem {
                                            tool_index: self.current_tool_id as usize,
                                            name: None,
                                            parameters: fragment.clone(),
                                        });
                                        let trimmed = current_args.trim_end();
                                        if let Some(stripped) = trimmed.strip_suffix('"') {
                                            *current_args = format!("{}{}}}", stripped, fragment);
                                        }
                                    }
                                }
                            }

                            // Update accumulated value
                            self.current_parameter_value = new_content;

                            // Remove processed content from buffer to avoid re-processing
                            // new_content is from buffer[..next_tag_pos], so remove that part
                            self.buffer = self.buffer[next_tag_pos..].to_string();
                        }
                    }
                }
            } else {
                // No parameter tag found, reset state if we were in a parameter
                if self.in_parameter {
                    self.in_parameter = false;
                    self.current_parameter_key.clear();
                    self.current_parameter_value.clear();
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
                self.xml_streaming_parameter_key = None;
                self.xml_streamed_parameter_value.clear();
                self.current_tool_name_sent = false;
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
        self.xml_streaming_parameter_key = None;
        self.xml_streamed_parameter_value.clear();
    }
}
