use async_trait::async_trait;
use regex::Regex;

use crate::tool_parser::{
    errors::ToolParserResult,
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Kimi K2 format parser for tool calls
///
/// Handles the Kimi K2 specific format:
/// `<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|><|tool_calls_section_end|>`
///
/// Features:
/// - Token-based delimiters
/// - Function calls with explicit indexing
/// - JSON arguments
pub struct KimiK2Parser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting partial tool calls (streaming)
    stream_tool_call_extractor: Regex,
}

impl KimiK2Parser {
    /// Create a new Kimi K2 parser
    pub fn new() -> Self {
        // Pattern for complete tool calls
        let tool_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        // Pattern for streaming (partial) tool calls
        let stream_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)";
        let stream_tool_call_extractor = Regex::new(stream_pattern).expect("Valid regex pattern");

        Self {
            partial_json: PartialJson::default(),
            tool_call_extractor,
            stream_tool_call_extractor,
        }
    }

    /// Check if text contains Kimi K2 tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<|tool_calls_section_begin|>")
    }

    /// Parse function ID to extract name and index
    fn parse_function_id(&self, id: &str) -> Option<(String, usize)> {
        // Format: functions.{name}:{index} or namespace.functions.{name}:{index}
        // Extract everything after the last dot before the colon as the function name
        if let Some(colon_pos) = id.rfind(':') {
            let before_colon = &id[..colon_pos];
            let index_str = &id[colon_pos + 1..];

            // Find the last dot to extract the function name
            if let Some(dot_pos) = before_colon.rfind('.') {
                let func_name = &before_colon[dot_pos + 1..];

                if let Ok(index) = index_str.parse::<usize>() {
                    return Some((func_name.to_string(), index));
                }
            }
        }
        None
    }
}

impl Default for KimiK2Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for KimiK2Parser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        let idx = text.find("<|tool_calls_section_begin|>").unwrap();
        let normal_text = text[..idx].to_string();

        // Try to extract tool calls
        let mut tools = Vec::new();
        for captures in self.tool_call_extractor.captures_iter(text) {
            if let (Some(id_match), Some(args_match)) = (
                captures.name("tool_call_id"),
                captures.name("function_arguments"),
            ) {
                let function_id = id_match.as_str();
                let function_args = args_match.as_str();

                // Parse function ID
                if let Some((func_name, _index)) = self.parse_function_id(function_id) {
                    // Try to parse JSON arguments
                    match serde_json::from_str::<serde_json::Value>(function_args) {
                        Ok(_) => {
                            // Generate unique ID
                            let id = format!("kimi_call_{}", uuid::Uuid::new_v4());

                            tools.push(ToolCall {
                                id,
                                r#type: "function".to_string(),
                                function: FunctionCall {
                                    name: func_name,
                                    arguments: function_args.to_string(),
                                },
                            });
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to parse JSON arguments for {}: {}",
                                func_name,
                                e
                            );
                            continue;
                        }
                    }
                } else {
                    tracing::warn!("Failed to parse function ID: {}", function_id);
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
            if let Some(marker_pos) = state.buffer.find("<|tool_calls_section_begin|>") {
                if marker_pos > 0 {
                    result.normal_text = state.buffer.drain(..marker_pos).collect();
                    state.in_tool_section = true;
                    return Ok(result);
                }
                state.in_tool_section = true;
                // Remove the marker from buffer
                let marker = "<|tool_calls_section_begin|>";
                state.buffer.drain(..marker.len());
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

impl KimiK2Parser {
    fn has_partial_marker(&self, buffer: &str) -> bool {
        // Check if buffer ends with partial tool marker
        let markers = ["<|", "<|tool", "<|tool_", "<|tool_calls", "<|tool_calls_"];
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
        for capture in self.tool_call_extractor.captures_iter(&state.buffer.clone()) {
            if let (Some(id_match), Some(args_match)) =
                (capture.name("tool_call_id"), capture.name("function_arguments")) {

                let tool_id = id_match.as_str();
                let args_json = args_match.as_str();

                // Parse tool ID: functions.{name}:{index}
                if let Some((name_part, _index_part)) = tool_id.split_once(':') {
                    let func_name = name_part.strip_prefix("functions.").unwrap_or(name_part);

                    let index = state.partial_tools.len();
                    self.emit_tool_call(func_name, args_json, index, state, result)?;
                    found_complete = true;
                }
            }
        }

        if found_complete {
            // Remove processed content up to end marker if present
            if let Some(end_pos) = state.buffer.find("<|tool_calls_section_end|>") {
                state.buffer.drain(..end_pos + "<|tool_calls_section_end|>".len());
                state.mode = crate::tool_parser::state::ParseMode::Complete;
            }
        } else {
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
        for capture in self.stream_tool_call_extractor.captures_iter(&buffer) {
            if let Some(id_match) = capture.name("tool_call_id") {
                let tool_id = id_match.as_str();

                // Parse tool ID: functions.{name}:{index}
                if let Some((name_part, index_str)) = tool_id.split_once(':') {
                    let func_name = name_part.strip_prefix("functions.").unwrap_or(name_part);
                    let tool_index = index_str.parse::<usize>().unwrap_or(0);

                    // Ensure we have partial tool entry
                    let partial = state.ensure_partial_tool(tool_index);

                    // Send name if not sent
                    if !partial.name_sent {
                        partial.name = Some(func_name.to_string());
                        partial.id = Some(format!("kimi_call_{}", uuid::Uuid::new_v4()));
                        partial.name_sent = true;

                        result.tool_calls.push(ToolCallItem {
                            tool_index,
                            id: partial.id.clone(),
                            name: partial.name.clone(),
                            arguments_delta: String::new(),
                        });
                    }

                    // Try to extract partial arguments
                    if let Some(args_match) = capture.name("function_arguments") {
                        let partial_json = args_match.as_str();

                        // Try to parse with partial JSON parser
                        if let Ok((value, _)) = self.partial_json.parse_value(partial_json) {
                            let args_str = serde_json::to_string(&value)?;

                            if args_str.len() > partial.streamed_arguments.len() {
                                let delta = &args_str[partial.streamed_arguments.len()..];

                                result.tool_calls.push(ToolCallItem {
                                    tool_index,
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
        partial.id = Some(format!("kimi_call_{}", uuid::Uuid::new_v4()));

        result.tool_calls.push(ToolCallItem {
            tool_index: index,
            id: partial.id.clone(),
            name: partial.name.clone(),
            arguments_delta: args_json.to_string(),
        });

        Ok(())
    }

}
