use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
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
        }
    }

    /// Check if text contains Step3 tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<｜tool_calls_begin｜>")
    }

    /// Parse parameters from steptml format
    fn parse_steptml_parameters(
        &self,
        params_text: &str,
    ) -> ToolParserResult<serde_json::Map<String, Value>> {
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
    fn parse_tool_call(&self, block: &str) -> ToolParserResult<Option<ToolCall>> {
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
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate ID
            let id = format!("step3_call_{}", uuid::Uuid::new_v4());

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

impl Default for Step3Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for Step3Parser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
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
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamingParseResult> {
        state.buffer.push_str(chunk);
        let mut result = StreamingParseResult::new();

        // Phase 1: Check for normal text before tool markers
        if !state.in_tool_section {
            if let Some(marker_pos) = state.buffer.find("<｜tool_calls_begin｜>") {
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

impl Step3Parser {
    fn has_partial_marker(&self, buffer: &str) -> bool {
        // Check if buffer ends with partial tool marker
        let markers = ["<｜", "<｜tool", "<｜tool_", "<｜tool_calls", "<｜tool_calls_"];
        for marker in &markers {
            if buffer.ends_with(marker) {
                return true;
            }
        }
        false
    }

    fn process_tool_calls(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        let call_start = "<｜tool_call_begin｜>";
        let call_end = "<｜tool_call_end｜>";

        while let Some(start_pos) = state.buffer.find(call_start) {
            let content_start = start_pos + call_start.len();

            if let Some(end_pos) = state.buffer[content_start..].find(call_end) {
                // Extract tool content first (owned string to avoid borrow issues)
                let tool_content = state.buffer[content_start..content_start + end_pos].to_string();

                // Process this tool call
                if let Some(sep_pos) = tool_content.find("<｜tool_sep｜>") {
                    if tool_content[..sep_pos].contains("function") {
                        let invoke_content = tool_content[sep_pos + "<｜tool_sep｜>".len()..].to_string();
                        self.process_steptml_invoke(&invoke_content, state, result)?;
                    }
                }

                // Remove processed portion
                state.buffer.drain(..content_start + end_pos + call_end.len());
            } else {
                // Incomplete tool call
                let partial_content = &state.buffer[content_start..].to_string();
                self.process_partial_steptml(&partial_content, state, result)?;
                break;
            }
        }

        Ok(())
    }

    fn process_steptml_invoke(
        &self,
        invoke_content: &str,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Parse steptml:invoke format
        if let Some(captures) = self.invoke_extractor.captures(invoke_content) {
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();
            let params_text = captures.get(2).map_or("", |m| m.as_str());

            let index = state.partial_tools.len();

            // Create partial tool if needed
            let partial = state.ensure_partial_tool(index);

            // Send name if not sent
            if !partial.name_sent && !func_name.is_empty() {
                partial.name = Some(func_name.to_string());
                partial.id = Some(format!("step3_call_{}", uuid::Uuid::new_v4()));
                partial.name_sent = true;

                result.tool_calls.push(ToolCallItem {
                    tool_index: index,
                    id: partial.id.clone(),
                    name: partial.name.clone(),
                    arguments_delta: String::new(),
                });
            }

            // Parse and stream parameters
            if partial.name_sent {
                let parameters = self.parse_steptml_parameters(params_text)?;
                let args_str = serde_json::to_string(&parameters)?;

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

        Ok(())
    }

    fn process_partial_steptml(
        &self,
        partial_content: &str,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try to extract partial tool information
        if let Some(sep_pos) = partial_content.find("<｜tool_sep｜>") {
            if partial_content[..sep_pos].contains("function") {
                let after_sep = &partial_content[sep_pos + "<｜tool_sep｜>".len()..];

                // Try to extract function name from steptml:invoke
                if let Some(name_match) = self.invoke_extractor.captures(after_sep) {
                    let func_name = name_match.get(1).map_or("", |m| m.as_str()).trim();

                    if !func_name.is_empty() {
                        let index = state.partial_tools.len();

                        // Ensure we have partial tool entry
                        let partial = state.ensure_partial_tool(index);

                        if !partial.name_sent {
                            partial.name = Some(func_name.to_string());
                            partial.id = Some(format!("step3_call_{}", uuid::Uuid::new_v4()));
                            partial.name_sent = true;

                            result.tool_calls.push(ToolCallItem {
                                tool_index: index,
                                id: partial.id.clone(),
                                name: partial.name.clone(),
                                arguments_delta: String::new(),
                            });
                        }

                        // Try to extract partial parameters
                        if let Some(params_text) = name_match.get(2) {
                            let parameters = self.parse_steptml_parameters(params_text.as_str())?;

                            if !parameters.is_empty() {
                                let args_str = serde_json::to_string(&parameters)?;

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
                }
            }
        }

        Ok(())
    }

}
