use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
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
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting function details
    func_detail_extractor: Regex,
}

impl DeepSeekParser {
    /// Create a new DeepSeek parser
    pub fn new() -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let func_detail_pattern = r"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)\n```json\n(.*?)\n```<｜tool▁call▁end｜>";
        let func_detail_extractor = Regex::new(func_detail_pattern).expect("Valid regex pattern");

        Self {
            partial_json: PartialJson::default(),
            tool_call_extractor,
            func_detail_extractor,
        }
    }

    /// Check if text contains DeepSeek tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<｜tool▁calls▁begin｜>")
    }

    /// Parse a single tool call block - throws error if parsing fails
    fn parse_tool_call(&self, block: &str) -> ToolParserResult<ToolCall> {
        let captures = self.func_detail_extractor.captures(block).ok_or_else(|| {
            ToolParserError::ParsingFailed("Failed to match tool call pattern".to_string())
        })?;

        // Get function type (should be "function")
        let func_type = captures.get(1).map_or("", |m| m.as_str());
        if func_type != "function" {
            return Err(ToolParserError::ParsingFailed(format!(
                "Invalid function type: {}",
                func_type
            )));
        }

        // Get function name
        let func_name = captures.get(2).map_or("", |m| m.as_str()).trim();
        if func_name.is_empty() {
            return Err(ToolParserError::ParsingFailed(
                "Empty function name".to_string(),
            ));
        }

        // Get JSON arguments
        let json_args = captures.get(3).map_or("{}", |m| m.as_str()).trim();

        // Parse JSON arguments
        let value = serde_json::from_str::<Value>(json_args)
            .map_err(|e| ToolParserError::ParsingFailed(format!("Invalid JSON: {}", e)))?;

        // Create arguments object
        let args = if value.is_object() {
            value
        } else {
            // If not an object, wrap it
            serde_json::json!({ "value": value })
        };

        let arguments = serde_json::to_string(&args)
            .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

        // Generate ID
        let id = format!("deepseek_call_{}", uuid::Uuid::new_v4());

        Ok(ToolCall {
            id,
            r#type: "function".to_string(),
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
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
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
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamingParseResult> {
        state.buffer.push_str(chunk);
        let mut result = StreamingParseResult::new();

        // Phase 1: Check for normal text before tool markers
        if !state.in_tool_section {
            if let Some(marker_pos) = state.buffer.find("<｜tool▁calls▁begin｜>") {
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

impl DeepSeekParser {
    fn has_partial_marker(&self, buffer: &str) -> bool {
        // Check if buffer ends with partial tool marker
        let markers = ["<｜", "<｜tool", "<｜tool▁", "<｜tool▁calls", "<｜tool▁calls▁"];
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
        let call_start = "<｜tool▁call▁begin｜>";
        let call_end = "<｜tool▁call▁end｜>";

        while let Some(start_pos) = state.buffer.find(call_start) {
            let content_start = start_pos + call_start.len();

            if let Some(end_pos) = state.buffer[content_start..].find(call_end) {
                let tool_content = state.buffer[content_start..content_start + end_pos].to_string();

                // Process this tool call
                self.process_single_tool(&tool_content, state, result)?;

                // Remove processed portion
                state.buffer.drain(..content_start + end_pos + call_end.len());
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
        // Parse format: function<｜tool▁sep｜>{name}\n```json\n{args}\n```
        if let Some(sep_pos) = tool_content.find("<｜tool▁sep｜>") {
            if tool_content[..sep_pos].contains("function") {
                let after_sep = &tool_content[sep_pos + "<｜tool▁sep｜>".len()..];

                // Extract function name (before newline)
                if let Some(name_end) = after_sep.find('\n') {
                    let func_name = after_sep[..name_end].trim();

                    // Extract JSON arguments
                    if let Some(json_start) = after_sep.find("```json\n") {
                        let json_content_start = json_start + "```json\n".len();
                        if let Some(json_end) = after_sep[json_content_start..].find("\n```") {
                            let json_str = &after_sep[json_content_start..json_content_start + json_end];

                            // Create tool call
                            let index = state.partial_tools.len();
                            self.emit_tool_call(func_name, json_str, index, state, result)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn process_partial_tool(
        &self,
        partial_content: &str,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try to extract partial tool information
        if let Some(sep_pos) = partial_content.find("<｜tool▁sep｜>") {
            if partial_content[..sep_pos].contains("function") {
                let after_sep = &partial_content[sep_pos + "<｜tool▁sep｜>".len()..];

                // Try to extract function name
                if let Some(name_end) = after_sep.find('\n') {
                    let func_name = after_sep[..name_end].trim();

                    if !func_name.is_empty() {
                        let index = state.partial_tools.len();

                        // Ensure we have partial tool entry
                        let partial = state.ensure_partial_tool(index);

                        // Send name if not sent
                        if !partial.name_sent {
                            partial.name = Some(func_name.to_string());
                            partial.id = Some(format!("deepseek_call_{}", uuid::Uuid::new_v4()));
                            partial.name_sent = true;

                            result.tool_calls.push(ToolCallItem {
                                tool_index: index,
                                id: partial.id.clone(),
                                name: partial.name.clone(),
                                arguments_delta: String::new(),
                            });
                        }

                        // Try to extract partial JSON arguments
                        if let Some(json_start) = after_sep.find("```json\n") {
                            let json_content_start = json_start + "```json\n".len();
                            let partial_json = &after_sep[json_content_start..];

                            // Remove trailing ``` if present
                            let partial_json = if partial_json.ends_with("\n```") {
                                &partial_json[..partial_json.len() - 4]
                            } else {
                                partial_json
                            };

                            // Try to parse partial JSON
                            if let Ok((value, _)) = self.partial_json.parse_value(partial_json) {
                                let args_str = serde_json::to_string(&value)?;

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

    fn emit_tool_call(
        &self,
        func_name: &str,
        json_str: &str,
        index: usize,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Ensure we have partial tool entry
        let partial = state.ensure_partial_tool(index);

        // Send complete tool call
        partial.name = Some(func_name.to_string());
        partial.id = Some(format!("deepseek_call_{}", uuid::Uuid::new_v4()));

        result.tool_calls.push(ToolCallItem {
            tool_index: index,
            id: partial.id.clone(),
            name: partial.name.clone(),
            arguments_delta: json_str.to_string(),
        });

        Ok(())
    }

}
