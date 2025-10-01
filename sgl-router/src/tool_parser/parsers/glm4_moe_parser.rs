use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
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
    ) -> ToolParserResult<StreamingParseResult> {
        state.buffer.push_str(chunk);
        let mut result = StreamingParseResult::new();

        // Phase 1: Check for normal text before tool markers
        if !state.in_tool_section {
            if let Some(marker_pos) = state.buffer.find("<tool_call>") {
                if marker_pos > 0 {
                    result.normal_text = state.buffer.drain(..marker_pos).collect();
                    state.in_tool_section = true;
                    return Ok(result);
                }
                state.in_tool_section = true;
            } else if !self.has_partial_marker(&state.buffer) {
                result.normal_text = std::mem::take(&mut state.buffer);
                return Ok(result);
            }
        }

        // Phase 2: Process tool calls
        if state.in_tool_section {
            self.process_tool_calls(state, &mut result)?;
        }

        Ok(result)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}

impl Glm4MoeParser {
    fn has_partial_marker(&self, buffer: &str) -> bool {
        // Check if buffer ends with partial tool marker
        buffer.ends_with("<tool") ||
        buffer.ends_with("<tool_") ||
        buffer.ends_with("<tool_c") ||
        buffer.ends_with("<tool_ca") ||
        buffer.ends_with("<tool_cal")
    }

    fn process_tool_calls(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        let start_marker = "<tool_call>";
        let end_marker = "</tool_call>";

        while let Some(start_pos) = state.buffer.find(start_marker) {
            let content_start = start_pos + start_marker.len();

            if let Some(end_pos) = state.buffer[content_start..].find(end_marker) {
                let tool_content = state.buffer[content_start..content_start + end_pos].to_string();

                // Process this tool call
                self.process_single_tool(&tool_content, state, result)?;

                // Remove processed portion
                state.buffer.drain(..content_start + end_pos + end_marker.len());
            } else {
                // Incomplete tool call, try to extract partial
                let partial_content = state.buffer[content_start..].to_string();
                self.process_partial_tool(&partial_content, state, result)?;
                break;
            }
        }

        Ok(())
    }

    fn process_single_tool(
        &self,
        tool_content: &str,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Parse GLM-4 format: function_name\n<arg_key>...</arg_key><arg_value>...</arg_value>
        if let Some(newline_pos) = tool_content.find('\n') {
            let func_name = tool_content[..newline_pos].trim();
            let args_content = &tool_content[newline_pos + 1..];

            // Parse arguments
            let mut args = serde_json::Map::new();
            for capture in self.arg_extractor.captures_iter(args_content) {
                if let (Some(key_match), Some(value_match)) = (capture.get(1), capture.get(2)) {
                    let key = key_match.as_str();
                    let value_str = value_match.as_str();

                    // Try to parse value as JSON, fallback to string
                    let value = serde_json::from_str::<Value>(value_str)
                        .unwrap_or_else(|_| Value::String(value_str.to_string()));

                    args.insert(key.to_string(), value);
                }
            }

            let args_json = serde_json::to_string(&args)?;

            // Emit complete tool call
            let index = state.partial_tools.len();
            self.emit_tool_call(func_name, &args_json, index, state, result)?;
        }

        Ok(())
    }

    fn process_partial_tool(
        &self,
        partial_content: &str,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try to extract function name (before newline)
        if let Some(newline_pos) = partial_content.find('\n') {
            let func_name = partial_content[..newline_pos].trim();

            if !func_name.is_empty() {
                let index = state.partial_tools.len();

                // Ensure we have partial tool entry
                let partial = state.ensure_partial_tool(index);

                // Send name if not sent
                if !partial.name_sent {
                    partial.name = Some(func_name.to_string());
                    partial.id = Some(format!("glm4_call_{}", uuid::Uuid::new_v4()));
                    partial.name_sent = true;

                    result.tool_calls.push(ToolCallItem {
                        tool_index: index,
                        id: partial.id.clone(),
                        name: partial.name.clone(),
                        arguments_delta: String::new(),
                    });
                }

                // Try to extract partial arguments
                let args_content = &partial_content[newline_pos + 1..];
                let mut partial_args = serde_json::Map::new();

                for capture in self.arg_extractor.captures_iter(args_content) {
                    if let (Some(key_match), Some(value_match)) = (capture.get(1), capture.get(2)) {
                        let key = key_match.as_str();
                        let value_str = value_match.as_str();

                        let value = serde_json::from_str::<Value>(value_str)
                            .unwrap_or_else(|_| Value::String(value_str.to_string()));

                        partial_args.insert(key.to_string(), value);
                    }
                }

                if !partial_args.is_empty() {
                    let args_str = serde_json::to_string(&partial_args)?;

                    if args_str.len() > partial.streamed_arguments.len() {
                        let delta = &args_str[partial.streamed_arguments.len()..];

                        result.tool_calls.push(ToolCallItem {
                            tool_index: index,
                            id: None,
                            name: None,
                            arguments_delta: delta.to_string(),
                        });

                        partial.streamed_arguments = args_str;
                    }
                }
            }
        }

        Ok(())
    }

    fn emit_tool_call(
        &self,
        func_name: &str,
        args_json: &str,
        index: usize,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Ensure we have partial tool entry
        let partial = state.ensure_partial_tool(index);
        partial.name = Some(func_name.to_string());
        partial.id = Some(format!("glm4_call_{}", uuid::Uuid::new_v4()));

        result.tool_calls.push(ToolCallItem {
            tool_index: index,
            id: partial.id.clone(),
            name: partial.name.clone(),
            arguments_delta: args_json.to_string(),
        });

        Ok(())
    }

}
