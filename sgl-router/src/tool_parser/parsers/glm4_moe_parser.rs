use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
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
        }
    }

    /// Check if text contains GLM-4 MoE tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    /// Parse arguments from key-value pairs
    fn parse_arguments(&self, args_text: &str) -> ToolParserResult<serde_json::Map<String, Value>> {
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
    fn parse_tool_call(&self, block: &str) -> ToolParserResult<Option<ToolCall>> {
        if let Some(captures) = self.func_detail_extractor.captures(block) {
            // Get function name
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

            // Get arguments text
            let args_text = captures.get(2).map_or("", |m| m.as_str());

            // Parse arguments
            let arguments = self.parse_arguments(args_text)?;

            let arguments_str = serde_json::to_string(&arguments)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate ID
            let id = format!("glm4_call_{}", uuid::Uuid::new_v4());

            Ok(Some(ToolCall {
                id,
                r#type: "function".to_string(),
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

impl Default for Glm4MoeParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for Glm4MoeParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains GLM-4 MoE format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        let idx = text.find("<tool_call>").unwrap();
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
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        state.buffer.push_str(chunk);

        // Check for tool markers
        if !self.has_tool_markers(&state.buffer) {
            // No tool markers detected - return all buffered content as normal text
            let normal_text = std::mem::take(&mut state.buffer);
            return Ok(StreamResult::NormalText(normal_text));
        }

        // Check for text before tool markers and extract it as normal text
        if let Some(marker_pos) = state.buffer.find("<tool_call>") {
            if marker_pos > 0 {
                // We have text before the tool marker - extract it as normal text
                let normal_text: String = state.buffer.drain(..marker_pos).collect();
                return Ok(StreamResult::NormalText(normal_text));
            }
        }

        // Look for start of tool call
        if let Some(start_pos) = state.buffer.find("<tool_call>") {
            // Look for the end of this tool call
            let search_from = start_pos + "<tool_call>".len();
            if let Some(end_pos) = state.buffer[search_from..].find("</tool_call>") {
                let end_abs = search_from + end_pos + "</tool_call>".len();

                // Extract and parse the complete tool call
                let tool_call_text = &state.buffer[start_pos..end_abs];

                if let Some(tool) = self.parse_tool_call(tool_call_text)? {
                    // Remove the processed part from buffer
                    state.buffer.drain(..end_abs);

                    return Ok(StreamResult::ToolComplete(tool));
                }
            } else {
                // Tool call not complete yet, try to extract partial info
                let partial = &state.buffer[search_from..];

                // Try to extract function name (first line after <tool_call>)
                if let Some(name_end) = partial.find('\n') {
                    let func_name = partial[..name_end].trim();

                    if !func_name.is_empty() && !state.in_string {
                        state.in_string = true; // Mark name as sent
                        return Ok(StreamResult::ToolName {
                            index: 0,
                            name: func_name.to_string(),
                        });
                    }

                    // Try to extract partial arguments
                    let args_text = &partial[name_end + 1..];
                    let partial_args = self.parse_arguments(args_text)?;

                    if !partial_args.is_empty() {
                        let args_str = serde_json::to_string(&partial_args)
                            .unwrap_or_else(|_| "{}".to_string());

                        return Ok(StreamResult::ToolArguments {
                            index: 0,
                            arguments: args_str,
                        });
                    }
                }
            }
        }

        Ok(StreamResult::Incomplete)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}
