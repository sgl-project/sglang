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
    tool_call_start_token: &'static str,
    tool_call_end_token: &'static str,

    /// XML format streaming state
    in_tool_call: bool,
    current_function_name: String,
    current_parameters: serde_json::Map<String, Value>,

    /// Precompiled regex patterns for XML format parsing
    xml_function_pattern: Regex,
    xml_param_pattern: Regex,
}

/// Decode HTML entities in a string (equivalent to Python's html.unescape)
///
/// Handles common HTML entities like &amp; &lt; &gt; &quot; &#39; and numeric entities
fn html_unescape(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '&' {
            let mut entity = String::new();
            let mut consumed_semicolon = false;
            while let Some(&next) = chars.peek() {
                if next == ';' {
                    chars.next();
                    consumed_semicolon = true;
                    break;
                }
                if next.is_alphanumeric() || next == '#' {
                    entity.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            let decoded = match entity.as_str() {
                "amp" => "&",
                "lt" => "<",
                "gt" => ">",
                "quot" => "\"",
                "apos" => "'",
                "nbsp" => "\u{00A0}",
                s if s.starts_with('#') => {
                    let num_str = &s[1..];
                    let code_point = if num_str.starts_with('x') || num_str.starts_with('X') {
                        u32::from_str_radix(&num_str[1..], 16).ok()
                    } else {
                        num_str.parse::<u32>().ok()
                    };
                    if let Some(cp) = code_point {
                        if let Some(ch) = char::from_u32(cp) {
                            result.push(ch);
                            continue;
                        }
                    }
                    // Invalid numeric entity, reconstruct original
                    result.push('&');
                    result.push_str(&entity);
                    if consumed_semicolon {
                        result.push(';');
                    }
                    continue;
                }
                _ => {
                    // Unknown entity, reconstruct original
                    result.push('&');
                    result.push_str(&entity);
                    if consumed_semicolon {
                        result.push(';');
                    }
                    continue;
                }
            };
            result.push_str(decoded);
        } else {
            result.push(c);
        }
    }

    result
}

/// Parse a raw parameter value, similar to Python's _safe_val
///
/// 1. Decode HTML entities
/// 2. Try to parse as JSON (numbers, booleans, null, objects, arrays)
/// 3. Fall back to string if JSON parsing fails
fn safe_val(raw: &str) -> Value {
    let unescaped = html_unescape(raw.trim());

    // Try JSON parsing first
    if let Ok(v) = serde_json::from_str::<Value>(&unescaped) {
        return v;
    }

    // Handle Python-style literals (True, False, None)
    match unescaped.as_str() {
        "True" => return Value::Bool(true),
        "False" => return Value::Bool(false),
        "None" => return Value::Null,
        _ => {}
    }

    // Fall back to string
    Value::String(unescaped)
}

impl QwenCoderParser {
    /// Create a new Qwen Coder parser
    pub fn new() -> Self {
        // Support XML format: <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>
        let pattern = r"(?s)<tool_call>\s*(.*?)\s*</tool_call>";
        let extractor = Regex::new(pattern).expect("Valid regex pattern");

        // Precompile XML format regex patterns for performance
        let xml_function_pattern =
            Regex::new(r"<function=([^>]+)>").expect("Valid XML function pattern");
        let xml_param_pattern = Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>")
            .expect("Valid XML parameter pattern");

        Self {
            extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            tool_call_start_token: "<tool_call>",
            tool_call_end_token: "</tool_call>",
            in_tool_call: false,
            current_function_name: String::new(),
            current_parameters: serde_json::Map::new(),
            xml_function_pattern,
            xml_param_pattern,
        }
    }

    /// Parse XML format tool call: <function=name><parameter=key>value</parameter></function>
    fn parse_xml_format(&self, content: &str) -> ParserResult<Option<ToolCall>> {
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
                let value = value_match.as_str();
                let json_value = safe_val(value);
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

    /// Parse and stream complete parameters from buffer
    /// Returns tool call items to emit (similar to Python's _parse_and_stream_parameters)
    fn parse_and_stream_parameters(&mut self) -> ParserResult<Vec<ToolCallItem>> {
        let mut calls: Vec<ToolCallItem> = vec![];

        // Find all complete parameter patterns in buffer
        let mut new_params = serde_json::Map::new();
        for cap in self.xml_param_pattern.captures_iter(&self.buffer) {
            if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                let key = key_match.as_str().trim().to_string();
                let value = value_match.as_str();
                let json_value = safe_val(value);
                new_params.insert(key, json_value);
            }
        }

        // Calculate parameter diff and stream updates
        if new_params != self.current_parameters {
            let current_args = &mut self.streamed_args_for_tool[self.current_tool_id as usize];

            if self.current_parameters.is_empty() {
                // First parameter(s) - build JSON fragment (without closing brace)
                let mut items = Vec::new();
                for (key, value) in &new_params {
                    let key_json =
                        serde_json::to_string(key).unwrap_or_else(|_| format!("\"{}\"", key));
                    let value_json = serde_json::to_string(value).unwrap_or_default();
                    items.push(format!("{}: {}", key_json, value_json));
                }
                let json_fragment = format!("{{{}", items.join(", "));

                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: None,
                    parameters: json_fragment.clone(),
                });
                *current_args = json_fragment;
            } else {
                // Additional parameters - add them incrementally
                let new_keys: Vec<_> = new_params
                    .keys()
                    .filter(|k| !self.current_parameters.contains_key(*k))
                    .collect();

                if !new_keys.is_empty() {
                    let mut continuation_parts = Vec::new();
                    for key in new_keys {
                        if let Some(value) = new_params.get(key) {
                            let key_json = serde_json::to_string(key)
                                .unwrap_or_else(|_| format!("\"{}\"", key));
                            let value_json = serde_json::to_string(value).unwrap_or_default();
                            continuation_parts.push(format!("{}: {}", key_json, value_json));
                        }
                    }

                    let json_fragment = format!(", {}", continuation_parts.join(", "));

                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: json_fragment.clone(),
                    });
                    current_args.push_str(&json_fragment);
                }
            }

            // Update current state
            self.current_parameters = new_params.clone();
            if let Some(tool_obj) =
                self.prev_tool_call_arr[self.current_tool_id as usize].as_object_mut()
            {
                tool_obj.insert("arguments".to_string(), Value::Object(new_params));
            }
        }

        Ok(calls)
    }

    /// Reset streaming state for next tool call
    fn reset_streaming_state(&mut self) {
        self.in_tool_call = false;
        self.current_tool_name_sent = false;
        self.current_function_name.clear();
        self.current_parameters.clear();
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
        let idx = text.find(self.tool_call_start_token).unwrap();
        let normal_text = text[..idx].to_string();

        // Extract tool calls
        let mut tools = Vec::new();
        for captures in self.extractor.captures_iter(text) {
            if let Some(content_str) = captures.get(1) {
                let content = content_str.as_str().trim();

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

        // If no tools were successfully parsed despite having markers, return entire text
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

        let mut normal_text = String::new();
        let mut calls: Vec<ToolCallItem> = vec![];

        // Build tool indices for validation
        let tool_indices = helpers::get_tool_indices(tools);

        loop {
            // If we're not in a tool call and don't see a start token, return normal text
            if !self.in_tool_call && !self.buffer.contains(self.tool_call_start_token) {
                // Check for partial start token
                if helpers::ends_with_partial_token(&self.buffer, self.tool_call_start_token)
                    .is_none()
                {
                    normal_text.push_str(&self.buffer);
                    self.buffer.clear();
                }
                break;
            }

            // Look for tool call start
            if !self.in_tool_call {
                if let Some(s) = self.buffer.find(self.tool_call_start_token) {
                    normal_text.push_str(&self.buffer[..s]);
                    self.buffer = self.buffer[s + self.tool_call_start_token.len()..].to_string();
                    self.in_tool_call = true;
                    self.current_tool_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();
                    continue;
                } else {
                    break;
                }
            }

            // We're in a tool call, try to parse function name if not sent yet
            if !self.current_tool_name_sent {
                if let Some(captures) = self.xml_function_pattern.captures(&self.buffer) {
                    if let Some(name_match) = captures.get(1) {
                        let function_name = name_match.as_str().trim().to_string();

                        // Validate function name
                        if tool_indices.contains_key(&function_name) {
                            self.current_function_name = function_name.clone();
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

                            // Send tool name
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: Some(function_name),
                                parameters: String::new(),
                            });

                            // Remove processed function declaration from buffer
                            self.buffer = self.buffer[captures.get(0).unwrap().end()..].to_string();
                            continue;
                        } else {
                            // Invalid function name, reset state
                            tracing::warn!("Invalid function name: {}", function_name);
                            self.reset_streaming_state();
                            normal_text.push_str(&self.buffer);
                            self.buffer.clear();
                            break;
                        }
                    }
                } else {
                    // Function name not complete yet, wait for more text
                    break;
                }
            }

            // Parse parameters (only complete ones)
            if self.current_tool_name_sent {
                let param_calls = self.parse_and_stream_parameters()?;
                calls.extend(param_calls);

                // Check if tool call is complete
                if let Some(end_pos) = self.buffer.find(self.tool_call_end_token) {
                    // Close JSON object if we have parameters
                    let current_args = &self.streamed_args_for_tool[self.current_tool_id as usize];
                    if !current_args.is_empty() {
                        // Count braces to check if JSON is complete
                        let open_braces = current_args.matches('{').count();
                        let close_braces = current_args.matches('}').count();
                        if open_braces > close_braces {
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: None,
                                parameters: "}".to_string(),
                            });
                            self.streamed_args_for_tool[self.current_tool_id as usize].push('}');
                        }
                    }

                    // Complete the tool call
                    self.buffer =
                        self.buffer[end_pos + self.tool_call_end_token.len()..].to_string();
                    self.reset_streaming_state();
                    self.current_tool_id += 1;
                    continue;
                } else {
                    // Tool call not complete yet, wait for more text
                    break;
                }
            }

            break;
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
        helpers::reset_parser_state(
            &mut self.buffer,
            &mut self.prev_tool_call_arr,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
        );
        self.reset_streaming_state();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_unescape_basic() {
        assert_eq!(html_unescape("&amp;"), "&");
        assert_eq!(html_unescape("&lt;"), "<");
        assert_eq!(html_unescape("&gt;"), ">");
        assert_eq!(html_unescape("&quot;"), "\"");
        assert_eq!(html_unescape("&apos;"), "'");
    }

    #[test]
    fn test_html_unescape_numeric() {
        assert_eq!(html_unescape("&#60;"), "<");
        assert_eq!(html_unescape("&#x3C;"), "<");
        assert_eq!(html_unescape("&#x3c;"), "<");
    }

    #[test]
    fn test_html_unescape_mixed() {
        assert_eq!(
            html_unescape("Hello &amp; World &lt;tag&gt;"),
            "Hello & World <tag>"
        );
    }

    #[test]
    fn test_html_unescape_unknown() {
        // Unknown entities with semicolon should be preserved as-is
        assert_eq!(html_unescape("&unknown;"), "&unknown;");
        // Unterminated entities should NOT have semicolon added
        assert_eq!(html_unescape("&foo bar"), "&foo bar");
        assert_eq!(html_unescape("&"), "&");
        assert_eq!(html_unescape("& "), "& ");
    }

    #[test]
    fn test_safe_val_json() {
        assert_eq!(safe_val("42"), Value::Number(42.into()));
        assert_eq!(safe_val("1.5"), serde_json::json!(1.5));
        assert_eq!(safe_val("true"), Value::Bool(true));
        assert_eq!(safe_val("false"), Value::Bool(false));
        assert_eq!(safe_val("null"), Value::Null);
        assert_eq!(
            safe_val(r#"{"key": "value"}"#),
            serde_json::json!({"key": "value"})
        );
        assert_eq!(safe_val(r#"[1, 2, 3]"#), serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_safe_val_python_literals() {
        assert_eq!(safe_val("True"), Value::Bool(true));
        assert_eq!(safe_val("False"), Value::Bool(false));
        assert_eq!(safe_val("None"), Value::Null);
    }

    #[test]
    fn test_safe_val_string_fallback() {
        assert_eq!(
            safe_val("hello world"),
            Value::String("hello world".to_string())
        );
        assert_eq!(safe_val("  spaces  "), Value::String("spaces".to_string()));
    }

    #[test]
    fn test_safe_val_html_entities() {
        assert_eq!(safe_val("&lt;div&gt;"), Value::String("<div>".to_string()));
        assert_eq!(
            safe_val("Tom &amp; Jerry"),
            Value::String("Tom & Jerry".to_string())
        );
    }
}
