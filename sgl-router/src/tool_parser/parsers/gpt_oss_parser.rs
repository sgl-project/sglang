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

/// GPT-OSS format parser for tool calls
///
/// Handles the GPT-OSS specific channel format:
/// `<|channel|>commentary to={namespace.function}<|constrain|>json<|message|>{json_args}<|call|>`
///
/// Features:
/// - Channel-based format with commentary
/// - Namespaced function calls
/// - JSON arguments
pub struct GptOssParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex for extracting complete function calls
    function_call_extractor: Regex,
    /// Regex for extracting streaming function calls
    streaming_extractor: Regex,
}

impl GptOssParser {
    /// Create a new GPT-OSS parser
    pub fn new() -> Self {
        // Pattern for complete function calls with to= parameter
        // Handles optional <|start|>assistant prefix and whitespace after function name
        let function_call_pattern = r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary to=([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*<\|constrain\|>json<\|message\|>(.*?)<\|call\|>(?:commentary)?";
        let function_call_extractor =
            Regex::new(function_call_pattern).expect("Valid regex pattern");

        // Pattern for streaming function calls (incomplete)
        let streaming_pattern = r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary to=([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*<\|constrain\|>json<\|message\|>(.*)";
        let streaming_extractor = Regex::new(streaming_pattern).expect("Valid regex pattern");

        Self {
            partial_json: PartialJson::default(),
            function_call_extractor,
            streaming_extractor,
        }
    }

    /// Check if text contains GPT-OSS tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<|channel|>commentary to=")
    }

    /// Extract function name from full namespace (e.g., "functions.get_weather" -> "get_weather")
    fn extract_function_name(&self, full_name: &str) -> String {
        if let Some(dot_pos) = full_name.rfind('.') {
            full_name[dot_pos + 1..].to_string()
        } else {
            full_name.to_string()
        }
    }
}

impl Default for GptOssParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for GptOssParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains GPT-OSS format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        let mut tools = Vec::new();
        let mut _tool_index = 0;

        // Extract all function calls
        for captures in self.function_call_extractor.captures_iter(text) {
            if let (Some(name_match), Some(args_match)) = (captures.get(1), captures.get(2)) {
                let full_function_name = name_match.as_str();
                let args_content = args_match.as_str().trim();

                // Extract actual function name
                let function_name = self.extract_function_name(full_function_name);

                // Parse JSON arguments
                let arguments = if args_content.is_empty() {
                    "{}".to_string()
                } else {
                    match serde_json::from_str::<Value>(args_content) {
                        Ok(value) => serde_json::to_string(&value)
                            .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?,
                        Err(_) => {
                            // Skip malformed JSON
                            continue;
                        }
                    }
                };

                // Generate unique ID
                let id = format!("gpt_oss_call_{}", uuid::Uuid::new_v4());

                tools.push(ToolCall {
                    id,
                    r#type: "function".to_string(),
                    function: FunctionCall {
                        name: function_name,
                        arguments,
                    },
                });

                _tool_index += 1;
            }
        }

        Ok((String::new(), tools)) // GPT-OSS parser returns empty normal text
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
            if let Some(marker_pos) = state.buffer.find("<|channel|>commentary to=") {
                if marker_pos > 0 {
                    // Check for optional <|start|>assistant prefix
                    let actual_start = if marker_pos >= "<|start|>assistant".len() &&
                        &state.buffer[marker_pos - "<|start|>assistant".len()..marker_pos] == "<|start|>assistant" {
                        marker_pos - "<|start|>assistant".len()
                    } else {
                        marker_pos
                    };

                    if actual_start > 0 {
                        result.normal_text = state.buffer.drain(..actual_start).collect();
                        state.in_tool_section = true;
                        return Ok(result);
                    }
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

impl GptOssParser {
    fn has_partial_marker(&self, buffer: &str) -> bool {
        // Check if buffer ends with partial channel marker
        let markers = ["<|", "<|chan", "<|channel", "<|channel|", "<|channel|>comm", "<|start", "<|start|"];
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
        // Look for complete tool calls
        let mut found_complete = false;
        let buffer_clone = state.buffer.clone();

        for capture in self.function_call_extractor.captures_iter(&buffer_clone) {
            if let (Some(func_match), Some(args_match)) = (capture.get(1), capture.get(2)) {
                let func_name = func_match.as_str();
                let args_json = args_match.as_str();

                // Parse and emit complete tool
                let index = state.partial_tools.len();
                self.emit_tool_call(func_name, args_json, index, state, result)?;
                found_complete = true;

                // Remove processed content
                let match_end = capture.get(0).unwrap().end();
                state.buffer.drain(..match_end);
            }
        }

        if !found_complete {
            // Try to extract partial tool info
            self.try_extract_partial_tools(state, result)?;
        }

        Ok(())
    }

    fn try_extract_partial_tools(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Look for partial tool calls
        let buffer = state.buffer.clone();
        for capture in self.streaming_extractor.captures_iter(&buffer) {
            if let Some(func_match) = capture.get(1) {
                let func_name = func_match.as_str();
                let index = state.partial_tools.len();

                // Ensure we have partial tool entry
                let partial = state.ensure_partial_tool(index);

                // Send name if not sent
                if !partial.name_sent {
                    partial.name = Some(func_name.to_string());
                    partial.id = Some(format!("gpt_oss_call_{}", uuid::Uuid::new_v4()));
                    partial.name_sent = true;

                    result.tool_calls.push(ToolCallItem {
                        tool_index: index,
                        id: partial.id.clone(),
                        name: partial.name.clone(),
                        arguments_delta: String::new(),
                    });
                }

                // Try to extract partial JSON arguments
                if let Some(args_match) = capture.get(2) {
                    let partial_json = args_match.as_str();

                    // Remove trailing <|call|> if present
                    let partial_json = if partial_json.ends_with("<|call|>") {
                        &partial_json[..partial_json.len() - "<|call|>".len()]
                    } else {
                        partial_json
                    };

                    // Try to parse with partial JSON parser
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
        partial.id = Some(format!("gpt_oss_call_{}", uuid::Uuid::new_v4()));

        result.tool_calls.push(ToolCallItem {
            tool_index: index,
            id: partial.id.clone(),
            name: partial.name.clone(),
            arguments_delta: args_json.to_string(),
        });

        Ok(())
    }

}
